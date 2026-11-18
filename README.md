# KVAF

KVAF 是面向 KV Cache 的 Runtime。它负责管理 KV block 在存储设备（DDR）中的地址、
生命周期和访问一致性，并提供高性能的 BatchPut/BatchGet
数据搬运能力。

当前支持 AIV URMA、AIV UBmem、AICPU Put/Gather 和 AICPU Put/Scatter。
CPU 端使用同一个服务程序，NPU 端默认构建一份包含全部后端的 Python 包，
通过 `NpuRuntime` 配置默认及启用的实现；per-call 模式允许在调用时选择算法，
每种 Get 的资源和执行状态首次准备后驻留复用。四种 Get 均提供单节点图租约。

实现边界与部署步骤见 [Get 图模式演进](docs/get_graph_evolution.md)，
功能和性能验证见 [验证指南](docs/cache_transport_execution_validation.md)。


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
├── common/include/          # 跨模块的类型、请求、Status 和协议布局
├── meta_manager/            # Key → DDR 元数据、slot 与访问表锁
├── runtime/
│   ├── include/             # 公共 Runtime 接口
│   ├── src/                 # RuntimeCore、NpuRuntime 入口和配置
│   ├── execution/           # 调度、执行资源管理；上下文和绑定接口在 include/
│   ├── memory/              # 缓存池、DDR 注册与分配
│   ├── peer/                # 链路与控制协议
│   └── cpu_service/         # 服务入口、backends、URMA 基础、Gather/Push worker
├── access_operator/
│   ├── algorithms/get/      # Direct、Gather、Scatter，按执行实现细分
│   ├── algorithms/put/      # URMA AIV、URMA AICPU、UBMEM AIV
│   ├── providers/           # Runtime adapter 和算法资源 owner
│   ├── transports/          # Direct-HCCP、AICPU URMA、UBMEM
│   └── include/             # 公共接口与资源视图
├── bindings/                # Python/Pybind 注册入口
├── python/kvaf/             # Python 包入口
├── scripts/                 # 构建辅助脚本
└── tests/                   # 主机回归、硬件功能及性能测试
```

具体算法的 Host、ABI、kernel 分布见 [AccessOperator 目录](access_operator/README.md#目录)，
Runtime 的内部划分见 [Runtime 源码目录](runtime/README.md#源码目录)。公共接口、CMake
目标和算法行为保持不变；尚未合入的 AIV Gather/Scatter 与 CPU Pull 不创建占位目录。

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

一个 `NpuRuntime` 连接一个 CPU 服务并使用一个远端 DDR region。legacy 模式使用构造时
固定的后端组合；per-call 模式按已启用的 Put/Get 配置选择算法，混合模式可在同一缓存池上
使用 URMA 和 UBMEM。Get 的执行上下文按配置驻留，用户输出的长期注册绑定是可选能力。

### Legacy 后端选择

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
build-npu/access_operator/transports/aicpu_urma/kernel.json
build-npu/access_operator/transports/aicpu_urma/kvaf-aicpu-urma.tar.gz
```

构建与运行选择相同的 CANN 根目录。AICPU 的固定安装位置为：

```text
<CANN_ROOT>/opp/vendors/cust/op_impl/aicpu/config/kvaf_aicpu_urma.json
<CANN_ROOT>/opp/vendors/cust/op_impl/aicpu/kernel/kvaf-aicpu-urma.tar.gz
```

Runtime 自动定位已安装的 AICPU 包，不再接受 `aicpu_urma_kernel_json_path`。
AIV/UBmem 的 `.o` 仍需显式传入，参数要求见下方 Python 调用说明。

## 启动与性能测试

在 CPU 主机上启动与 NPU 构建来自同一提交的服务：

```bash
./build-cpu/runtime/kvaf_cpu_service \
  --listen 0.0.0.0 --port 21111 \
  --gather-threads 8 --gather-mode active
```

四种 Get 共用这个服务。Gather 的 CPU 线程预算须不小于客户端 AICPU worker 数；
其他后端不使用这两个 `--gather-*` 参数。服务启动后打印 `LISTEN`。
按实际路由可增加 `--ub-device NAME --eid-index N`。
CPU 亲和性由外部 `numactl` 或 `taskset` 设置。

在 NPU 主机加载 CANN 环境、构建并安装设备包后，从仓库根目录运行：

```bash
KVAF_CPU_SERVICE_ADDR=127.0.0.1:21111 DEVICE_ID=2 \
bash tests/run_remote_cpu_batch_get_perf.sh build-npu \
  --backend aiv --aiv-cores 2 --execution both \
  --request-length 32768 --table-capacity 4096 --miss-count 4096 \
  --token-bytes 656 --tokens-per-block 128 --jettys-per-core 2 \
  --seed 20260903 --no-sort-input-indices \
  --source-access-pattern random --output-access-pattern contiguous \
  --warmup 10 --iterations 100 --kernel-profile
```

跨主机部署时将 `127.0.0.1` 替换为 CPU 服务的可达地址。
`--backend` 支持 `aiv`、`ubmem`、`gather_read`、`cpu_push_scatter`；
各后端的核数和完整计时范围见[统一 Get 性能测试](docs/batch_get_performance.md)。
脚本在正式计时之外执行数据预检和输出校验。

Put 使用 `tests/run_remote_cpu_batch_put_perf.sh`，支持 URMA AIV、UBMEM AIV、URMA AICPU。
三种 Put 的命令、输入复用及验证范围见[性能验证步骤](docs/cache_transport_execution_validation.md)。

需要分析整个 Runtime 的 API 成本时运行：

```bash
KVAF_CPU_SERVICE_ADDR=127.0.0.1:21111 DEVICE_ID=2 \
bash tests/run_remote_cpu_runtime_perf.sh build-npu
```

该性能脚本测量初始化、BatchPut、AccessTable 准备和上传、BatchGet、BatchFree、shutdown，
输出各 API 的 `PERF_RESULT`。追加 `--transport ubmem` 可测 UBMEM。
默认总超时 600 秒，可通过 `KVAF_REMOTE_CPU_PERF_TIMEOUT` 调整。

`tests/python/` 保留性能入口和 trace 汇总工具，公共模块位于 `tests/python/utils/`。
独立功能、配置和单元测试及其 CMake 注册已移除。

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
