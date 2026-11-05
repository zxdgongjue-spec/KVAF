# KVAF

KVAF 是面向 KV Cache 的 Runtime。它负责管理 KV block 在存储设备（DDR）中的地址、
生命周期和访问一致性，并提供高性能的 BatchPut/BatchGet
数据搬运能力。

当前支持 AIV URMA、AIV UBmem、AICPU Put/Gather 和 AICPU Put/Scatter。
CPU 端使用同一个服务程序，NPU 端默认构建一份包含全部后端的 Python 包，
通过 `NpuRuntime` 构造参数选择本次 Runtime 使用的实现。


## KVAF-Runtime 核心能力

`RuntimeCore` 组合 `MetaManager`、`AccessOperator` 和内部 DDR allocator，提供
同步的 KV 生命周期：

| 接口 | 语义 |
| --- | --- |
| `Initialize` | 申请并注册远端 DDR，初始化数据面并注册 region。失败时清理已获得资源；清理失败则保留状态供重试。 |
| `BatchPut` | 先由 MetaManager 为 Key 预留 slot，再由 Operator 搬运数据；搬运成功后提交映射，失败则回滚预留 slot。 |
| `PrepareAccessTable` | 对一批 Key 建立稳定读快照，生成扁平的 address/length/valid 访问表。 |
| `BatchGet` | 使用 AccessTable 和调用方输入提交批量读取；NPU 路径默认只向指定 stream 入队。 |
| `ReleaseAccessTable` | 释放稳定读快照，解除 Key 的 pin。 |
| `BatchFree` | 回收 Key 的 DDR slot；仍被 AccessTable 持有的 Key 不会被提前释放。 |
| `Shutdown` | 先停止并排空传输，再释放远端 DDR 和内部资源；清理失败可重试。 |

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
├── python/kvaf/           # Python 包入口
├── scripts/               # 构建辅助脚本
└── tests/                 # 主机回归、硬件功能及性能测试
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
`CacheOS-vLLM Adaptor` 在 vLLM 侧实现，不属于当前代码仓。

| 模块 | 职责 |
| --- | --- |
| `common/` | 跨模块共享的类型、请求和 `Status`。 |
| `meta_manager/` | Key 到远端 DDR 地址的内存态元数据、slot 分配、访问表锁定和释放。 |
| `access_operator/` | AIV URMA、UBmem、AICPU Gather/Scatter/Put 算子，以及 HCCP/UMDK 和内存映射资源。 |
| `runtime/` | `Runtime` 接口、`RuntimeCore`、`NpuRuntime`、`CpuRuntime`、TCP `RuntimeService` 与 peer 控制协议。 |
| `bindings/` | 可选的 pybind 模块，导出 Python `kvaf.NpuRuntime` 和 Mock Runtime。 |
| `tests/` | 配置、构建和清理回归，Python Mock 测试，CPU/NPU 端到端及性能测试。 |

## CPU/NPU 分离部署

CPU 进程提供 DDR 资源，并按需启动 Gather 或 CPU Push 工作线程。
NPU 进程运行 RuntimeCore、MetaManager 和选定的 AccessOperator。

```text
CPU 主机                                              NPU 主机
────────                                              ────────
kvaf_cpu_service                                      kvaf.NpuRuntime
  CpuRuntime                                            RuntimeCore
  - CPU DDR                                             - MetaManager
  - URMA 注册 / UBmem Fabric handle                     - AIV / AICPU AccessOperator
  - Gather / CPU Push 工作线程                           - NPU AccessTable
  - session 资源回收                                     - 调用方 Tensor / stream
        │                                                        │
        └──────── TCP：DDR 申请、连接与会话控制 ──────────────────┘
        └──────── 数据：URMA 读写 / UBmem 映射搬运 ───────────────┘
```

TCP 交换控制信息。KV 数据由选定的数据面搬运：AIV/AICPU URMA 读写、
CPU Push + AICPU Scatter，或 UBmem 映射后的 AIV MTE 搬运。

### CPU Runtime

`CpuRuntime` 由独立的 `kvaf_cpu_service` 进程创建。同一个服务可接收不同客户端的
URMA、UBmem 请求，后端资源按需初始化。URMA 依赖 UB 驱动、UMDK 头文件和
`liburma.so`；UBmem 依赖 Ascend HAL 和支持 Host P2P DDR 的驱动。

- URMA 自动枚举可用 device/EID，为 session 注册 DDR segment、token 和 Jetty；
- Gather 请求按需创建 CPU gather worker；Scatter 请求按需创建 CPU push worker；
- UBmem 申请 Host P2P DDR 并导出 Fabric handle，交给 NPU 导入和映射；
- 支持多个独立 session，关闭时先停止传输，再回收对应 DDR 和通信资源；
- `RuntimeService` 只负责 TCP 监听和连接分发，资源生命周期由 `CpuRuntime` 管理。

### NPU Runtime

`NpuRuntime` 在连接 CPU 服务后初始化远端 DDR。

- 先 `link_to(cpu_ip:port)`，再由 `initialize()` 申请 DDR 并初始化所选数据面；
- URMA 自动发现 NPU 本端 EID：优先使用 HCCL rootinfo 中与当前物理卡匹配的 CLOS PG EID，
  不可用时回退到 `/proc/asdrv_ub/pair_info`；
- AIV URMA 将显式 `npu_eids` 列表作为多路径条带；AICPU Gather/Scatter 将列表
  作为逐一尝试的候选，最终选用一个 EID；
- URMA 候选初始化使用所选后端的 BatchPut 验证链路，失败并完成清理后尝试下一候选；
- `BatchPut`：将 NPU Tensor 中的 KV block 写入 CPU DDR，并提交元数据；
- `PrepareAccessTable` / `access_table_view`：为 BatchGet 生成稳定的 CPU 或 NPU
  AccessTable；
- `BatchGet`：通过所选后端搬运数据；默认仅入队到调用方 stream；
- `BatchFree`：释放 Key 对应的 slot。仍被 AccessTable 持有的 Key 会返回 busy。

一个 `NpuRuntime` 连接一个 CPU 服务并使用一个远端 DDR region。后端在构造时固定，
Get/Put 成对使用；本期不支持同一 Runtime 混用 UBmem 和 URMA，也不支持按次切换算法。
切换配置需关闭原 Runtime 后创建新 Runtime，缓存数据不会自动迁移。

### 后端选择

| `transport` | `aicpu_batch_get_mode` | BatchPut | BatchGet |
| --- | --- | --- | --- |
| `"urma"`（默认） | `""`（默认） | AIV URMA | AIV URMA |
| `"urma"` | `"gather_read"` | AICPU UMDK | CPU Gather + AICPU READ |
| `"urma"` | `"cpu_push_scatter"` | AICPU UMDK | CPU Push + AICPU Scatter |
| `"ubmem"` | `""` | AIV UBmem | AIV UBmem |

`ubmem` 搭配非空 mode、未知 mode 以及 `gather_read_direct` 都会报错。
AICPU-only 构建也需要显式填写 mode；空 mode 始终表示请求 AIV，不会自动改选。

## 环境要求

| 角色 | 必需条件 |
| --- | --- |
| CPU 服务机（URMA） | 已加载的 UB 驱动、`/dev/uburma`、可用的 `liburma.so`。 |
| CPU 服务机（UBmem） | 支持 Host P2P DDR、Fabric/VMM 接口的 Ascend 驱动和 HAL 库。 |
| NPU 服务机 | Ascend 驱动、CANN、`torch_npu`、Python 3 和 C++17 编译器。 |
| NPU kernel 构建 | CANN 路径中需要 `include/acl/acl.h`、`tools/ccec_compiler/bin` 和 SIMT 开发头 `simt_api/device_atomic_functions.h`。 |
| AICPU kernel 构建 | 默认全量构建还需要 AICPU AArch64 交叉编译器、UMDK 头文件及设备侧 securec；安装需要对所选 CANN 包目录和加载配置有写权限。 |
| 两端网络 | NPU 能通过 TCP 访问 CPU 服务地址；UB 网络和平台拓扑已配置。 |


## 构建

以下命令均在仓库根目录执行。

### 构建 CPU 服务

```bash
./build.sh cpu
```

默认 UMDK 路径为仓库同级的 `../third_party/umdk`，`liburma.so` 会自动从该目录或
系统库路径发现；只有非默认环境才需要设置 `KVAF_UMDK_ROOT` 或 `KVAF_URMA_LIBRARY`。
只需要 UBmem 时可用 `./build.sh cpu --without-urma`，该产物不能服务 URMA 请求。

构建结果为：

```text
build-cpu/runtime/kvaf_cpu_service
```

### 构建 NPU Runtime 与 Python binding

先加载实际安装的 CANN 环境。下面以 `/usr/local/Ascend/ascend-toolkit/latest` 为例；
其他环境请替换为对应路径。

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/ascend-toolkit/latest
./build.sh npu
```

默认同时构建 AIV URMA、UBmem、AICPU Put/Gather/Scatter，输出一份 Python 包。
构建成功后自动将 AICPU 设备包安装到所选 CANN 根目录；无需 `--install-kernel`。
`ASCEND_HOME_PATH` 和 `ASCEND_CANN_PACKAGE_PATH` 同时设置时必须指向同一真实目录。
安装完成后重启 NPU 客户端进程。

按需裁剪或检查安装：

```bash
./build.sh npu --without-aicpu      # 原来的 AIV/UBmem 构建，不安装 AICPU 包
./build.sh npu --without-python     # 全部后端及设备包，不构建 Python binding
./build.sh npu --check-kernel       # 检查已安装的 Gather/Scatter 包，不重新构建
./build.sh aicpu-urma               # 仅 AICPU Put/Gather，构建后自动安装
./build.sh aicpu-urma --cpu-push     # 仅 AICPU Put/Gather/Scatter，构建后自动安装
```

这些是按需选择的不同命令，不需要依次执行。`aicpu-urma` 默认输出到
`build-aicpu-urma`；统一构建使用 `build-npu`，运行时通过 `transport` 和
`aicpu_batch_get_mode` 选择后端，无需切换 Python 包。
`--without-aicpu` 不检查 AICPU 依赖，也不修改已经安装的 AICPU 包；需要卸载时单独执行
`./build.sh npu --uninstall-kernel`。原 `--install-kernel` 参数保留兼容。

构建结果包括：

```text
build-npu/python/kvaf/
build-npu/bin/kvaf_execute_batch_get_kernels.o
build-npu/bin/kvaf_execute_batch_put_kernels.o
build-npu/access_operator/src/aicpu_urma/kernel.json
build-npu/access_operator/src/aicpu_urma/kvaf-aicpu-urma.tar.gz
```

构建与运行选择相同的 CANN 根目录。AICPU 的固定安装位置为：

```text
<CANN_ROOT>/opp/vendors/cust/op_impl/aicpu/config/kvaf_aicpu_urma.json
<CANN_ROOT>/opp/vendors/cust/op_impl/aicpu/kernel/kvaf-aicpu-urma.tar.gz
```

Runtime 自动定位已安装的 AICPU 包，不再接受 `aicpu_urma_kernel_json_path`。
AIV/UBmem 的 `.o` 仍需显式传入，参数要求见下方 Python 调用说明。

## 启动与端到端验证

### 1. 启动 CPU 服务

在 CPU 主机上执行：

```bash
./build-cpu/runtime/kvaf_cpu_service \
  --listen 0.0.0.0 --port 28999 \
  --gather-threads 8 --gather-mode active --gather-affinity off
```

此启动命令可供全部四种后端共用。三个 `--gather-*` 参数仅控制 CPU Gather 的
线程预算、等待方式和绑核行为，不影响 AIV、UBmem 或 CPU Push Scatter 的执行。
Gather 的 CPU 线程预算须不小于客户端 AICPU worker 数；上例支持后续的 1/2/4/5 worker 测试。

服务启动后打印 `LISTEN`；首次 URMA 请求触发 device/EID 枚举并打印 `CANDIDATE_DISCOVERED`。
也可按需要传入 `--ub-device NAME --eid-index N` 来选定 URMA 端口。
AIV N 端性能脚本开启 `--adaptive-ub-device` 后，K 端未指定设备时会从脚本配置的
`KVAF_ADAPTIVE_UB_DEVICES` 中测速选优；K 端指定单个 `--ub-device` 时走原生路径。
详见 [Runtime 启动选优](runtime/README.md#可选启动-ub-device-选优)；N/K 两端需同时升级至控制协议 v6。

### 2. 验证 AIV URMA

在 NPU 主机上加载 CANN 环境后执行。以下用本机 CPU 服务举例；跨主机部署时将
`127.0.0.1` 替换为 CPU 服务的可达地址，`DEVICE_ID` 改为实际设备号。

```bash
export PYTHONPATH="$PWD/build-npu/python${PYTHONPATH:+:$PYTHONPATH}"
export KVAF_CPU_SERVICE_ADDR="127.0.0.1:28999"
export DEVICE_ID=0

tests/run_remote_cpu_e2e.sh "$PWD/build-npu"
```

测试会依次验证 Runtime 初始化、CPU DDR 分配、EID/Jetty 建链、AIV BatchPut、
BatchGet 结果和 teardown。正常完成时：

- NPU 侧输出 Runtime 初始化和正确性通过信息；
- CPU 侧出现 `REGION_ALLOCATED`、`CHANNEL_CREATED` 与 `REGION_RELEASE`。

该入口调用 [test_npu_batch_get.py](tests/python/test_npu_batch_get.py)，mode 省略时验证
默认 AIV 路径。硬件测试出现 `SKIP` 表示未实际执行，不应视为通过。

### 3. 验证 AICPU Gather / Scatter

保持同一个 CPU 服务和上面的 NPU 环境，使用统一构建的 Python 包：

```bash
python3 access_operator/demo/aicpu_batch_get/python/test_batch_get.py \
  --device-id "$DEVICE_ID" --peer "$KVAF_CPU_SERVICE_ADDR" \
  --modes gather_read cpu_push_scatter --workers 1 2 4 5 \
  --blocks 4096 --token-bytes 656 --tokens-per-block 2 \
  --output-row-step 1 --warmup 3 --iterations 20
```

实测 CPU Push Scatter 在 8 个 AICPU worker 时存在问题，当前测试示例统一使用
1、2、4、5 个 worker，暂不推荐 Scatter 使用 8 个 worker。

[test_batch_get.py](access_operator/demo/aicpu_batch_get/python/test_batch_get.py)
每轮先执行 AICPU BatchPut，再执行 Gather/Scatter 并比较整个输出。它还检查两种
Table 重载、空/部分请求前缀、随机输入和输出行、异步调用后同 stream 的 NPU 消费。
两种模式乘以四种 worker 数，应输出 8 条 `AICPU_BATCH_GET_PASS`。
将 `--output-row-step` 改为 `2` 可进一步检查输出间隔未被误写。

可追加 `--npu-eids "实际EID1,实际EID2"`，或设置 `KVAF_NPU_EIDS`。Gather 和 Scatter
都会逐一尝试候选，最终选用一个 EID。要验证失败回退，应将格式合法但对当前连接
不可用的候选放在前面，已确认可用的候选放在后面，并检查 `UMDK_EID_SELECTED` 日志。
脚本不再接受 `--kernel-json` 或 `--source-mode direct`。

### 4. 验证 UBmem

使用支持 Host P2P DDR/Fabric 映射的硬件和驱动，保持上述 CPU 服务运行：

```bash
export KVAF_RUN_NPU_TESTS=1
export KVAF_BATCH_GET_KERNEL="$PWD/build-npu/bin/kvaf_execute_batch_get_kernels.o"
export KVAF_BATCH_PUT_KERNEL="$PWD/build-npu/bin/kvaf_execute_batch_put_kernels.o"

python3 tests/python/test_npu_batch_get.py --transport ubmem
```

同一个测试文件覆盖 UBmem Put/Get 和数据校验；不设置 AICPU mode。

### 5. Runtime 性能测试

CPU 服务保持运行，在 NPU 主机执行：

```bash
tests/run_remote_cpu_runtime_perf.sh "$PWD/build-npu"
```

此脚本默认测 AIV URMA；追加 `--transport ubmem` 可测 UBmem。
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

AICPU 性能测试分别使用 [Gather sweep](tests/run_aicpu_gather_read_sweep.sh) 和
[Scatter sweep](tests/run_aicpu_cpu_push_scatter_sweep.sh)，构建目录参数均可填写
`"$PWD/build-npu"`。默认扫描较大的源池；小规模验证可用 `--table-capacity` 缩小源池。

### 6. 无 NPU 的主机回归

使用新的主机构建目录，默认不启用任何设备后端：

```bash
cmake -S . -B build-host-tests -DBUILD_TESTING=ON -DKVAF_BUILD_PYTHON=OFF
cmake --build build-host-tests --parallel 4
ctest --test-dir build-host-tests --output-on-failure
```

其中 [npu_runtime_config_test.cc](tests/npu_runtime_config_test.cc) 验证后端和安装路径，
[aicpu_kernel_loader_test.cc](tests/aicpu_kernel_loader_test.cc) 用 ACL 桩检查内核入口，
[test_aicpu_configuration.py](tests/python/test_aicpu_configuration.py) 验证 CLI 与包检查，
[test_build_script.py](tests/python/test_build_script.py) 模拟编译步骤并实际执行临时目录安装。
这些回归不能替代 CANN 设备编译和真机 Put/Get 验证。

## Python 调用流程

通过 `transport` 和 `aicpu_batch_get_mode` 选择前述四种组合。前三个基础参数
`batch_get_kernel_object_path`、`batch_get_token_bytes`、`batch_get_tokens_per_block`
必须传入。原有参数直到 `transport` 仍支持位置传参；新增的
`aicpu_batch_get_mode` 仅接受关键字传参，其他配置也建议使用关键字。

| 内核路径参数 | AIV URMA | UBmem | AICPU Gather/Scatter |
| --- | --- | --- | --- |
| `batch_get_kernel_object_path` | 必须提供有效 `.o` | 必须提供有效 `.o` | 必须传参，填 `""` 即可 |
| `batch_put_kernel_object_path` | 必须提供，初始化时用于 Put 探测 | 使用 Put 时必须提供；只用 Get 可省略 | 可省略，不使用 |

AIV/UBmem 的 `.o` 不会自动定位。AICPU 从 CANN 安装目录定位自己的设备包，
Get/Put 不使用上述 `.o`。`npu_core_count` 表示 AIV core 数或 AICPU worker 数，
AICPU 的参数校验范围为 1 到 8；推荐测试配置及 Scatter 的已知限制见前文验证示例。

下面使用默认 AIV URMA；改成 `transport="ubmem"` 可使用 UBmem，mode 保持为空。

```python
import os
from pathlib import Path

import torch
import torch_npu
import kvaf

device_id = int(os.environ.get("DEVICE_ID", "0"))
torch.npu.set_device(device_id)
build_dir = Path("build-npu").resolve()
runtime = kvaf.NpuRuntime(
    batch_get_kernel_object_path=str(build_dir / "bin/kvaf_execute_batch_get_kernels.o"),
    batch_get_token_bytes=656,
    batch_get_tokens_per_block=128,
    device_id=device_id,
    npu_core_count=2,
    batch_put_kernel_object_path=str(build_dir / "bin/kvaf_execute_batch_put_kernels.o"),
    transport="urma",
    aicpu_batch_get_mode="",
    jettys_per_core=2,
    npu_eids=[],  # URMA 默认自动发现；UBmem 不使用 EID。
)
runtime.link_to(os.environ.get("KVAF_CPU_SERVICE_ADDR", "127.0.0.1:28999"))
runtime.initialize(ddr_bytes=4096 * 656 * 128, slot_size=656 * 128)

# 使用 NPU Tensor 调用 batch_put / batch_get；具体 Tensor 形状见 tests/python/。
# table = runtime.prepare_access_table(keys)
# device_table = runtime.access_table_view(table, device="npu")
# runtime.batch_get(
#     device_table, indices, offsets, miss_count, output_indices, output
# )
runtime.shutdown()
```

`link_to()` 必须先于 `initialize()`。调用方需要在 BatchGet 所在 stream 完成前保持
输入/输出 Tensor 和 `DeviceAccessTable` 有效；Runtime 管理内部 DDR、元数据和传输资源。

### EID 与调优参数

| 后端 | `npu_eids=[]` | 显式 `npu_eids=[eid1, eid2, ...]` |
| --- | --- | --- |
| AIV URMA | 自动发现，选用一个可用 EID | BatchGet 多 EID 条带，BatchPut 使用第一项 |
| AICPU Gather/Scatter | 自动发现候选 | 按顺序尝试，最终选用一个；Get/Put 共用选定 EID |
| UBmem | 不使用 EID | 不用于该后端 |

显式 EID 使用 32 位连续十六进制字符串。AICPU 候选失败后先清理资源，再尝试下一项；
清理失败则停止并保留资源供重试。`urma_device` 和 `urma_eid_index` 已从
`NpuRuntime` 参数中删除，设备侧 UMDK 根据 Host 选定的 EID 匹配 device/index。

对 AIV URMA，显式列表是原子配置：所有 EID 都必须属于当前物理 NPU，并成功创建
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

AIV Runtime 先按 Jetty-major 顺序把每个 core 的主 Jetty 分散到不同 EID，再分配次 Jetty。
大批量请求会随各 core/Jetty 的 128-WQE post 自然分摊；每个 core 支持 1 到 64 个
Jetty。很小的请求如果没有启用所有
Jetty，则不会产生足够 post 使用所有配置路径。

性能测试可以使用 `--sort-input-indices` 对 `miss_count` 有效前缀排序。测试按实际
远端地址和 slot 内 offset 排序，并同步重排 `indices`、`offsets` 和
`output_indices`；请求构造和排序不计入 BatchGet 计时。

Jetty 数、BatchGet 在途预算和启动带宽调优属于 AIV URMA 配置；AICPU 不启用
`adaptive_ub_device`。测试脚本读取 `KVAF_NPU_EIDS` 后传给 Runtime，Runtime 本身
不读取该环境变量。

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

### AICPU Gather / Scatter

使用与 AIV/UBmem 相同的 `build-npu/python` 包，显式指定 mode 即可：

```python
import os
import torch
import torch_npu
import kvaf

device_id = int(os.environ.get("DEVICE_ID", "0"))
torch.npu.set_device(device_id)
runtime = kvaf.NpuRuntime(
    batch_get_kernel_object_path="",
    batch_get_token_bytes=656,
    batch_get_tokens_per_block=2,
    device_id=device_id,
    npu_core_count=4,
    transport="urma",
    aicpu_batch_get_mode="gather_read",  # Scatter 改为 "cpu_push_scatter"。
    npu_eids=[],
)
runtime.link_to(os.environ.get("KVAF_CPU_SERVICE_ADDR", "127.0.0.1:28999"))
runtime.initialize(ddr_bytes=4096 * 656 * 2, slot_size=656 * 2)
# batch_put、prepare_access_table、batch_get 的调用接口保持一致。
runtime.shutdown()
```

两个 mode 都使用 AICPU BatchPut；`gather_read` 使用 CPU gather + AICPU UMDK READ，
`cpu_push_scatter` 使用 CPU SGL WRITE 到 HBM staging，再由 AICPU scatter 到输出。
两种公开 Table 重载都按 `addresses[indices[i]] + offsets[i]` 解释源地址，
`gather_read_direct` 不对外开放。

更详细的数据流和性能说明见 [GatherRead](docs/aicpu_gather_read.md) 与
[CPU Push Scatter](docs/aicpu_cpu_push_scatter.md)。升级时使用同次源码构建 Host 库和设备包，
安装后重启 NPU 客户端，避免 Host/设备 ABI 不一致。
