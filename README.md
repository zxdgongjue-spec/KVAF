# KVAF

KVAF 是面向 KV Cache 的 Runtime。它负责管理 KV block 在远端 DDR 中的地址、
生命周期和访问一致性，并通过可替换的 AccessOperator 执行 BatchPut/BatchGet
数据搬运。当前硬件实现使用 NPU AIV 算子通过 UB/RMA 访问 CPU DDR；本地 Mock
实现用于不具备硬件环境时验证 Runtime 语义。

KVAF 不实现 Prefix Cache 策略、调度器或 vLLM 业务逻辑。上层 Adapter 负责将模型
请求映射为 Key、block 和 Tensor；KVAF 负责这些 Key 对应的远端存储和批量访问。

```text
调用方 / vLLM Adapter（仓外）
             │ Key、KV block、NPU Tensor
             ▼
        KVAF Runtime
        RuntimeCore
     ┌───────┴────────┐
     ▼                ▼
MetaManager      AccessOperator
Key → DDR 地址    BatchPut / BatchGet 数据面
     │                │
     └───────┬────────┘
             ▼
         远端 DDR region
```

## Runtime 核心能力

`RuntimeCore` 组合 `MetaManager`、`AccessOperator` 和内部 DDR allocator，提供
同步的 KV 生命周期：

| 接口 | 语义 |
| --- | --- |
| `Initialize` | 申请并注册一块可管理的远端 DDR，初始化数据面并向 MetaManager 注册 region。任一步失败都会释放已获得资源。 |
| `BatchPut` | 先由 MetaManager 为 Key 预留 slot，再由 Operator 搬运数据；搬运成功后提交映射，失败则回滚预留 slot。 |
| `PrepareAccessTable` | 对一批 Key 建立稳定读快照，生成扁平的 address/length/valid 访问表。 |
| `BatchGet` | 使用 AccessTable 和调用方输入提交批量读取；NPU 路径默认只向指定 stream 入队。 |
| `ReleaseAccessTable` | 释放稳定读快照，解除 Key 的 pin。 |
| `BatchFree` | 回收 Key 的 DDR slot；仍被 AccessTable 持有的 Key 不会被提前释放。 |
| `Shutdown` | 关闭数据面、释放远端 DDR 和 Runtime 内部状态。 |

MetaManager 是内存态的 Key → 远端地址映射。它将写入分为“预留、搬运、提交”三个
阶段，避免写入失败的 Key 被读路径看到；读路径通过稳定快照避免 BatchGet 与 BatchFree
并发时访问失效地址。

调用方持有输入/输出 buffer、NPU Tensor 和 stream；KVAF 持有远端 DDR、元数据、
AccessTable 锁和内部传输资源。调用方必须在同一 BatchGet stream 完成前保持相关 Tensor
和 `DeviceAccessTable` 有效。

## 模块

| 模块 | 职责 |
| --- | --- |
| `common/` | 跨模块共享的类型、请求和 `Status`。 |
| `meta_manager/` | Key 到远端 DDR 地址的内存态元数据、slot 分配、访问表锁定和释放。 |
| `access_operator/` | AccessOperator 接口、AIV BatchGet/BatchPut host 侧适配，以及 Direct-HCCP/UMDK Jetty 传输资源。 |
| `runtime/` | `Runtime` 接口、`RuntimeCore`、`NpuRuntime`、`CpuRuntime`、TCP `RuntimeService` 与 peer 控制协议。 |
| `bindings/` | 可选的 pybind 模块，导出 Python `kvaf.NpuRuntime` 和 Mock Runtime。 |
| `tests/` | Python Mock 测试、CPU 服务/NPU 进程端到端测试和 Runtime 性能测试。 |

## CPU/NPU 分离部署

CPU/NPU 分离是当前 AIV 数据面的部署方式：CPU 进程提供远端 DDR 和 UMDK 资源，
NPU 进程运行 KVAF RuntimeCore、MetaManager 和 AIV AccessOperator。

```text
CPU 主机                                              NPU 主机
────────                                              ────────
kvaf_cpu_service                                      kvaf.NpuRuntime
  CpuRuntime                                            RuntimeCore
  - CPU DDR                                             - MetaManager
  - UMDK segment / token / Jetty                        - AIV BatchPut / BatchGet
  - session 资源回收                                   - NPU AccessTable
        │                                                        │
        └──────── TCP：DDR 申请、EID、CTP、QP key ──────────────┘
                                                                 │
                                  UB / RMA：AIV 直接读写 CPU DDR
```

TCP 只承载控制信息；KV 数据不会经过 TCP。NPU AIV kernel 完成实际的 UB/RMA
读写。

### CPU Runtime

`CpuRuntime` 由独立的 `kvaf_cpu_service` 进程创建。它只依赖 UB 驱动、UMDK
公开头文件和 `liburma.so`，不依赖 CANN、HCCP、ACL 或 NPU 驱动。

- 自动枚举可用 UMDK device/EID；
- 为 NPU session 申请 CPU DDR，并注册 UMDK segment 和 token；
- 接收 NPU 侧 CTP、PSN、QP key 和 EID，创建并导入 CPU 侧 Jetty；
- 支持多个独立 session；NPU 正常 shutdown 后回收其 DDR、segment、token 和 Jetty；
- `RuntimeService` 只负责 TCP 监听和连接分发，资源生命周期由 `CpuRuntime` 管理。

CPU Runtime 当前不执行 BatchGet 或 BatchPut 算子。

### NPU Runtime

`NpuRuntime` 在连接 CPU 服务后初始化远端 DDR，并组合 MetaManager 与
AccessOperator。

- `link_to(cpu_ip:port)` 后申请 CPU DDR 并建立 CPU–NPU 链路；
- 自动发现 NPU 本端 EID：优先使用 HCCL rootinfo 中与当前物理卡匹配的 CLOS PG EID，
  不可用时回退到 `/proc/asdrv_ub/pair_info`；
- 自动枚举 CPU 服务端的 UMDK EID 候选；对候选组合执行真实的 64B AIV WRITE 和 CQ
  完成验证，只使用确实可访问 CPU DDR 的链路；
- `BatchPut`：将 NPU Tensor 中的 KV block 写入 CPU DDR，并提交元数据；
- `PrepareAccessTable` / `access_table_view`：为 BatchGet 生成稳定的 CPU 或 NPU
  AccessTable；
- `BatchGet`：使用 NPU Tensor 发起 AIV 读取；默认仅入队到调用方 stream；
- `BatchFree`：释放 Key 对应的 slot。仍被 AccessTable 持有的 Key 会返回 busy。

一个 `NpuRuntime` 当前连接一个 CPU 服务并使用一个远端 DDR region。异常退出后的
session 恢复、多个 peer 的选择和 CPU 侧数据面不属于当前范围。

## 环境要求

| 角色 | 必需条件 |
| --- | --- |
| CPU 服务机 | 已加载的 UB 驱动、`/dev/uburma`、可用的 `liburma.so`。 |
| NPU 服务机 | Ascend 驱动、CANN、`torch_npu`、Python 3 和 C++17 编译器。 |
| NPU kernel 构建 | CANN 路径中需要 `include/acl/acl.h`、`tools/ccec_compiler/bin` 和 SIMT 开发头 `simt_api/device_atomic_functions.h`。 |
| 两端网络 | NPU 能通过 TCP 访问 CPU 服务地址；UB 网络和平台拓扑已配置。 |

`build.sh` 不安装 CANN、驱动或 UB 驱动。这些由平台环境提供。CPU 构建若没有可用的
UMDK 源码，会拉取固定版本的 UMDK；若系统没有 `liburma.so`，只构建所需 `urma`
target，不执行全局安装。

## 构建

以下命令均在仓库根目录执行。

### 构建 CPU 服务

```bash
./build.sh cpu
```

可按环境显式提供 UMDK 路径或动态库：

```bash
KVAF_UMDK_ROOT=/path/to/umdk \
KVAF_URMA_LIBRARY=/path/to/liburma.so \
./build.sh cpu
```

构建结果为：

```text
build-cpu/runtime/kvaf_cpu_service
```

### 构建 NPU Runtime 与 Python binding

先加载与驱动匹配的 CANN 环境，并设置工具包根目录：

```bash
source /path/to/cann/set_env.sh
export ASCEND_CANN_PACKAGE_PATH=/path/to/cann

# 若 SIMT 开发头安装在独立目录，设置其上级 include 目录。
# 该目录下必须存在 simt_api/device_atomic_functions.h。
export ASCEND_SIMT_API_DIR=/path/to/simt/include

./build.sh npu
```

如果 SIMT 开发头位于 `${ASCEND_CANN_PACKAGE_PATH}/include`，无需设置
`ASCEND_SIMT_API_DIR`。构建结果包括：

```text
build-npu/python/kvaf/
build-npu/bin/kvaf_execute_batch_get_kernels.o
build-npu/bin/kvaf_execute_batch_put_kernels.o
```

`build.sh npu` 会为所选 `PYTHON` 安装缺失的 `pybind11`。如遇到
`simt_api/device_atomic_functions.h` 缺失，请安装与当前 CANN 匹配的 SIMT/asc-devkit
开发包，或将 `ASCEND_SIMT_API_DIR` 指向实际头文件所在的 include 目录。

## 启动与端到端验证

### 1. 启动 CPU 服务

在 CPU 主机上执行：

```bash
./build-cpu/runtime/kvaf_cpu_service \
  --listen 0.0.0.0 --port 28999
```

服务会自动发现 UB device/EID，并打印 `CANDIDATE_DISCOVERED` 与 `LISTEN`。只有诊断
固定候选时才需要传入 `--ub-device NAME --eid-index N`。

### 2. 运行 NPU 端到端测试

在 NPU 主机上执行：

```bash
export KVAF_CPU_SERVICE_ADDR=<cpu-ip>:28999
export DEVICE_ID=0

tests/run_remote_cpu_e2e.sh "$PWD/build-npu"
```

测试会依次验证 Runtime 初始化、CPU DDR 分配、EID/Jetty 建链、AIV BatchPut、
BatchGet 结果和 teardown。正常完成时：

- NPU 侧输出 Runtime 初始化和正确性通过信息；
- CPU 侧出现 `REGION_ALLOCATED`、`CHANNEL_CREATED` 与 `REGION_RELEASE`。

### 3. Runtime 性能测试

CPU 服务保持运行，在 NPU 主机执行：

```bash
KVAF_CPU_SERVICE_ADDR=<cpu-ip>:28999 \
DEVICE_ID=0 \
tests/run_remote_cpu_runtime_perf.sh "$PWD/build-npu"
```

脚本先执行完整数据路径正确性 preflight；通过后输出各 Runtime API 的
`PERF_RESULT`，包括初始化、BatchPut、AccessTable、BatchGet、BatchFree 和 shutdown。
默认总超时为 600 秒，可通过 `KVAF_REMOTE_CPU_PERF_TIMEOUT` 调整。

性能场景可通过以下变量调整：

```text
KVAF_RUNTIME_PERF_BATCH
KVAF_RUNTIME_PERF_TABLE_CAPACITY
KVAF_RUNTIME_PERF_REQUESTS
KVAF_RUNTIME_PERF_ACTIVE_PERCENT
KVAF_RUNTIME_PERF_BLOCK_BYTES
KVAF_RUNTIME_PERF_SOURCE_STRIDE
KVAF_RUNTIME_PERF_AIV_CORES
KVAF_RUNTIME_PERF_JETTYS_PER_CORE
KVAF_RUNTIME_PERF_WARMUP
KVAF_RUNTIME_PERF_ITERATIONS
```

## Python 调用流程

`NpuRuntime` 的 CPU 服务地址是唯一需要由应用指定的链路参数；EID 与 UB device 由
Runtime 选择。

```python
import kvaf

runtime = kvaf.NpuRuntime(
    batch_get_kernel_object_path="build-npu/bin/kvaf_execute_batch_get_kernels.o",
    device_id=0,
    npu_core_count=1,
    batch_put_kernel_object_path="build-npu/bin/kvaf_execute_batch_put_kernels.o",
    jettys_per_core=2,
)

runtime.link_to("<cpu-ip>:28999")
runtime.initialize(ddr_bytes=16 * 1024**3, slot_size=576)

# 使用 NPU Tensor 调用 batch_put / batch_get；具体 Tensor 形状见 tests/python/。
# table = runtime.prepare_access_table(keys)
# device_table = runtime.access_table_view(table, device="npu")
# runtime.batch_get(device_table, indices, mask, output_indices, output)

runtime.shutdown()
```

`link_to()` 必须先于 `initialize()` 调用。调用方需要在 BatchGet 所在 stream 完成前保持
输入/输出 Tensor 和 `DeviceAccessTable` 有效；Runtime 只管理内部 DDR、元数据和内部
传输资源。

## 本地无硬件检查

本地 Mock 不验证 UMDK、CANN 或 AIV 数据面，但可验证 Runtime、MetaManager 和 pybind
基础接口：

```bash
cmake -S . -B build-local \
  -DBUILD_TESTING=ON \
  -DKVAF_BUILD_PYTHON=ON
cmake --build build-local --parallel
ctest --test-dir build-local --output-on-failure
```

更多 Runtime 生命周期、AccessTable 所有权、AIV Tensor 契约和性能指标说明见
[`runtime/README.md`](runtime/README.md)。
