# KVAF

KVAF 是面向 KV Cache 的 Runtime。它负责管理 KV block 在存储设备（DDR）中的地址、
生命周期和访问一致性，并提供高性能的 BatchPut/BatchGet
数据搬运能力。

当前目录包含 KVAF 同步 Runtime POC、MetaManager 地址管理实现，以及可选的
A5 AIV BatchGet/BatchPut demo/adapter。


## KVAF-Runtime 核心能力

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
```text
kvaf/
├── common/include/        # 模块间共享的类型、请求与 Status
├── runtime/               # KVAF Runtime
├── meta_manager/          # Meta Manager
├── access_operator/       # KV Access Operator
├── bindings/              # 可选的 Python/Pybind 注册入口
└── python/kvaf/           # Python 包入口
```

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
`CacheOS-vLLM Adaptor` 不属于当前代码仓；它后续在 vLLM 侧实现
| 模块 | 职责 |
| --- | --- |
| `common/` | 跨模块共享的类型、请求和 `Status`。 |
| `meta_manager/` | Key 到远端 DDR 地址的内存态元数据、slot 分配、访问表锁定和释放。 |
| `access_operator/` | AccessOperator 接口、AIV BatchGet/BatchPut host 侧适配，以及 Direct-HCCP/UMDK Jetty 传输资源。 |
| `runtime/` | `Runtime` 接口、`RuntimeCore`、`NpuRuntime`、`CpuRuntime`、TCP `RuntimeService` 与 peer 控制协议。 |
| `bindings/` | 可选的 pybind 模块，导出 Python `kvaf.NpuRuntime` 和 Mock Runtime。 |
| `tests/` | Python Mock 测试、CPU 服务/NPU 进程端到端测试和 Runtime 性能测试。 |

## CPU/NPU 分离部署

CPU/NPU 分离是当前 AIV 数据面的部署方式：CPU 进程提供远端 DDR 资源，
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

TCP 用于交换控制信息，初始化UB建链；KV 数据 通过 NPU AIV kernel 完成实际的读写。

### CPU Runtime

`CpuRuntime` 由独立的 `kvaf_cpu_service` 进程创建。默认 URMA 后端依赖于 UB 驱动、UMDK
公开头文件和 `liburma.so`（无须 CANN、HCCP、ACL 或 NPU 驱动）。

- 自动枚举可用 UMDK device/EID；
- 为 NPU session 申请 CPU DDR，并注册 UMDK segment 和 token；
- 接收 NPU 侧 CTP、PSN、QP key 和 EID，创建并导入 CPU 侧 Jetty；
- 支持多个独立 session；NPU 正常 shutdown 后回收其 DDR、segment、token 和 Jetty；
- `RuntimeService` 只负责 TCP 监听和连接分发，资源生命周期由 `CpuRuntime` 管理。

### NPU Runtime

`NpuRuntime` 在连接 CPU 服务后初始化远端 DDR。

- `link_to(cpu_ip:port)` 后申请 CPU DDR 并建立 CPU–NPU 链路；
- 自动发现 NPU 本端 EID：优先使用 HCCL rootinfo 中与当前物理卡匹配的 CLOS PG EID，
  不可用时回退到 `/proc/asdrv_ub/pair_info`；
- 可通过 `npu_eids=[...]` 严格指定多个 NPU EID，BatchGet 为每个 EID
  创建独立 Direct-HCCP context/MR/channel，并把 AIV core/Jetty worker
  轮转分配到各路径；BatchPut 保持单 EID，使用列表第一项；
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
| CPU 服务机（URMA） | 已加载的 UB 驱动、`/dev/uburma`、可用的 `liburma.so`。 |
| CPU 服务机（UBmem） | 支持 Host P2P DDR、Fabric/VMM 接口的 Ascend 驱动和 HAL 库。 |
| NPU 服务机 | Ascend 驱动、CANN、`torch_npu`、Python 3 和 C++17 编译器。 |
| NPU kernel 构建 | CANN 路径中需要 `include/acl/acl.h`、`tools/ccec_compiler/bin` 和 SIMT 开发头 `simt_api/device_atomic_functions.h`。 |
| 两端网络 | NPU 能通过 TCP 访问 CPU 服务地址；UB 网络和平台拓扑已配置。 |


## 构建

以下命令均在仓库根目录执行。

### 构建 CPU 服务

```bash
./build.sh cpu
```

默认 UMDK 路径为仓库同级的 `../third_party/umdk`，`liburma.so` 会自动从该目录或
系统库路径发现；只有非默认环境才需要设置 `KVAF_UMDK_ROOT` 或 `KVAF_URMA_LIBRARY`。

构建结果为：

```text
build-cpu/runtime/kvaf_cpu_service
```

### 构建 NPU Runtime 与 Python binding

默认使用 `/usr/local/Ascend/ascend-toolkit/latest`；使用其他 CANN 版本时，将两处路径
替换为实际安装目录。

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/ascend-toolkit/latest
./build.sh npu
```

构建结果包括：

```text
build-npu/python/kvaf/
build-npu/bin/kvaf_execute_batch_get_kernels.o
build-npu/bin/kvaf_execute_batch_put_kernels.o
```


## 启动与端到端验证

### 1. 启动 CPU 服务

在 CPU 主机上执行：

```bash
./build-cpu/runtime/kvaf_cpu_service \
  --listen 0.0.0.0 --port 28999
```

服务启动后打印 `LISTEN`；首次 URMA 请求触发 device/EID 枚举并打印 `CANDIDATE_DISCOVERED`。
也可按需要传入 `--ub-device NAME --eid-index N` 来选定 URMA 端口。
N 端性能脚本开启 `--adaptive-ub-device` 后，K 端未指定设备时会从脚本配置的
`KVAF_ADAPTIVE_UB_DEVICES` 中测速选优；K 端指定单个 `--ub-device` 时走原生路径。
详见 [Runtime 启动选优](runtime/README.md#可选启动-ub-device-选优)；N/K 两端需同时升级至控制协议 v6。

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
KVAF_NPU_EIDS
KVAF_RUNTIME_PERF_WARMUP
KVAF_RUNTIME_PERF_ITERATIONS
```

## Python 调用流程

`NpuRuntime` 通过构造参数 `transport="urma"`（默认）或 `transport="ubmem"` 选择协议。
以下 EID 与 Jetty 配置适用于 URMA；默认自动选择 EID 与 UB device，需要多 EID 并行时
可以显式提供严格的 NPU EID 列表。

```python
import kvaf

runtime = kvaf.NpuRuntime(
    batch_get_kernel_object_path="build-npu/bin/kvaf_execute_batch_get_kernels.o",
    batch_get_token_bytes=656,
    batch_get_tokens_per_block=128,
    device_id=0,
    npu_core_count=2,
    batch_put_kernel_object_path="build-npu/bin/kvaf_execute_batch_put_kernels.o",
    transport="urma",  # 使用 UBmem 时改为 "ubmem"，其余 Runtime 接口保持相同。
    jettys_per_core=2,
    # 可选：每项是32位连续十六进制；空列表保持单 EID 自动发现。
    npu_eids=[
        "00112233445566778899aabbccddeeff",
        "102132435465768798a9bacbdcedfe0f",
    ],
)
runtime.link_to("<cpu-ip>:28999")
runtime.initialize(ddr_bytes=4096 * 656 * 128, slot_size=656 * 128)

# 使用 NPU Tensor 调用 batch_put / batch_get；具体 Tensor 形状见 tests/python/。
# table = runtime.prepare_access_table(keys)
# device_table = runtime.access_table_view(table, device="npu")
# runtime.batch_get(
#     device_table, indices, offsets, miss_count, output_indices, output
# )
runtime.shutdown()
```

显式列表是原子配置：所有 EID 都必须属于当前物理 NPU，并成功创建
BatchGet context/MR/channel；任一路径失败都会使该 CPU EID 候选失败，不会回退到
rootinfo 或 `pair_info`。启动 AIV WRITE/CQ 探测由 BatchPut 在列表第一个 EID
上完成。EID 数不能超过 `npu_core_count * jettys_per_core`。当前多 EID 指多个
NPU 本端 EID，它们连接到同一个已选 CPU EID/DDR region；CPU 侧多 EID region striping
不在这一实现中。

性能脚本也接受逗号分隔的 `KVAF_NPU_EIDS`：

```bash
KVAF_NPU_EIDS=00112233445566778899aabbccddeeff,102132435465768798a9bacbdcedfe0f \
tests/run_remote_cpu_batch_get_perf.sh /absolute/path/to/build-npu \
  --aiv-cores 2 --jettys-per-core 1
```

Runtime 先按 Jetty-major 顺序把每个 core 的主 Jetty 分散到不同 EID，再分配次 Jetty。
大批量请求会随各 core/Jetty 的 128-WQE post 自然分摊；每个 core 支持 1 到 64 个
Jetty。很小的请求如果没有启用所有
Jetty，则不会产生足够 post 使用所有配置路径。

性能测试可以使用 `--sort-input-indices` 对 `miss_count` 有效前缀排序。测试按实际
远端地址和 slot 内 offset 排序，并同步重排 `indices`、`offsets` 和
`output_indices`；请求构造和排序不计入 BatchGet 计时。

`link_to()` 必须先于 `initialize()` 调用。调用方需要在 BatchGet 所在 stream 完成前保持
输入/输出 Tensor 和 `DeviceAccessTable` 有效；Runtime 只管理内部 DDR、元数据和内部
传输资源。

### UBmem 传输

UBmem 复用同一个 CPU service。CPU 端通过 Ascend HAL 分配 Host P2P DDR 并导出
Fabric handle，NPU 通过 ACL 导入、映射后使用 MTE2/UB/MTE3 搬运数据。CPU 服务根据
分配请求选择后端，无需指定协议；只使用 UBmem 时可用 `./build.sh cpu --without-urma`
构建。Get/Put 的每核 UB 预算为 32 KiB，Get 最多使用 8 级流水，Put 使用双缓冲。

CPU 服务启动后，可用下面的命令测试 `4096 × 656 B` 的 BatchGet。示例使用本机服务，
CPU DDR 地址空间为 328 MiB；远程部署时替换服务地址。

```bash
KVAF_CPU_SERVICE_ADDR=127.0.0.1:28999 \
tests/run_remote_cpu_batch_get_perf.sh "$PWD/build-npu" \
  --transport ubmem --request-length 4096 --table-capacity 4096 \
  --aiv-cores 16 --output-access-pattern contiguous
```

### URMA BatchGet 总在途请求预算

`NpuRuntime(..., batch_get_max_inflight_total=2048)` 默认限制单次 BatchGet 所有
AIV 核合计的未确认 READ 数。小请求保留整批路径；大请求每核使用两个交叠组，
完成最老组后补发。设为 0 可对照原实现。性能脚本接受 `--max-inflight-total`，
或环境变量 `KVAF_BATCH_GET_MAX_INFLIGHT_TOTAL`。修改预算需要重新 capture。
Host、Python binding 和 Kernel 必须一起重编。详见
[设计、统计和验证步骤](docs/batch_get_inflight_control.md)。
