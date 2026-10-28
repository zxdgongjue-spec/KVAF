# KVAF Runtime

## 模块定位

KVAF Runtime 是对外的 KV 控制面入口。它不处理 Prefix Key 或
Forward 语义；它只拥有内部 DDR 的生命周期，编排 MetaManager 与
AccessOperator，并把调用方已经准备好的输入/输出 buffer 交给数据面。

## 已实现能力

- `Initialize`：经 `DdrMemoryAllocator` 分配内部 DDR，初始化 Operator，
  再向 MetaManager 注册远端 region；任一前置步骤失败会释放已获得资源。
- `BatchPut`：`PreparePut -> ExecuteBatchPut -> CommitPut`；数据面失败时
  自动 `AbortPut`，因此 Key 和 slot 可重新使用。配置 BatchPut kernel 的 NPU
  路径使用同步 AIV Direct-HCCP WRITE；未配置时保留同步
  `aclrtMemcpy(DeviceToHost)` mock。
- `PrepareAccessTable` / `GetAccessTableView` / `ReleaseAccessTable`：持有
  MetaManager 稳定读锁，生成一维 address/length/valid 表，并在 Release 时
  解锁。
- `BatchGet`：校验调用方的 index、miss-count、output 和（NPU 路径）Device
  AccessTable buffer 后，向调用方 stream 提交 `AccessOperator::ExecuteBatchGet`。
  Python 默认不在 KVAF 内同步；同一 stream 的后续操作自然按序执行。
- `BatchFree`：将删除请求交给 MetaManager；仍被 AccessTable 持有的 Key
  会返回 `kBusy`。

异步接口、BatchPut 图捕获管理及图对象生命周期不在本次 POC 范围。

## CPU Runtime 服务与 NPU Link

`Runtime` 是角色接口；`RuntimeCore` 保留已有的 DDR、MetaManager 与 Operator
编排。当前 `NpuRuntime` 持有 NPU 数据面，`CpuRuntime` 持有 CPU DDR、UMDK
segment/token/Jetty 与会话。TCP 监听是无状态的 `RuntimeService`：它只接收连接并
分发给 Runtime，不拥有硬件资源。

CPU 服务只需要 UB 驱动、`/dev/uburma`、UMDK 公开头文件和 `liburma.so`；不链接
或加载 CANN、HCCP、ACL。仓库根目录的统一构建入口会自动拉取固定 UMDK 源码，并在系统
缺少 `liburma.so` 时只构建所需的 UMDK `urma` target：

```bash
cd /path/to/kvaf
./build.sh cpu

./build-cpu/runtime/kvaf_cpu_service \
  --listen 0.0.0.0 --port 28999
```

CPU DDR region 支持两种页分配模式；未指定时默认使用 `thp`：

```bash
# 默认模式：2 MiB 对齐，MADV_HUGEPAGE，并在注册前主动触页；
# 是否全部合并仍由 Linux THP 决定。
./build-cpu/runtime/kvaf_cpu_service \
  --listen 0.0.0.0 --port 28999

# 保留原始普通内存行为：4 KiB 对齐，操作系统仍可能自行使用 THP。
./build-cpu/runtime/kvaf_cpu_service \
  --listen 0.0.0.0 --port 28999 --memory-page-mode default
```

需要固定 NUMA 位置时，应在 UB device 所在 node 启动服务，例如
`numactl --cpunodebind=N --membind=N ...`。`REGION_ALLOCATED` 日志会记录
`page_mode`。默认 THP 模式还应通过 `/proc/<pid>/smaps` 的 `AnonHugePages` 验证实际
合并量；Linux 使用大页并不保证 UDMA/IOMMU 一定保留相同的页表粒度。

默认自动枚举所有可用 UMDK UB device/EID；`--ub-device NAME` 和
`--eid-index N` 仅用于诊断。CPU 端可同时保持多个 NPU session，NPU 正常
`shutdown` 后会释放对应 DDR、segment、token 与 Jetty。

NPU 应用默认只配置 CPU 服务地址，不配置 EID 或 UDMA device：

```python
runtime = kvaf.NpuRuntime(
    batch_get_kernel,
    batch_get_token_bytes=656,
    batch_get_tokens_per_block=128,
    device_id=0,
    batch_put_kernel_object_path=batch_put_kernel,
)
runtime.link_to("141.61.92.71:28999")
runtime.initialize(ddr_bytes, kv_block_bytes)
```

已知设备名时也可设置 `peer_ub_device="udmac0d1e2"`，只在该设备的 EID 中按
连通性选择，不执行带宽标定。`peer_ub_device` 与 `adaptive_ub_device=True` 互斥。

`link_to` 必须在 `initialize` 前调用。初始化会从 HCCL root-info 读取当前物理卡的
CLOS PG EID、追加 `/proc/asdrv_ub/pair_info` fallback，并组合 CPU UMDK 候选。只有
64B AIV WRITE 与 CQ 完成成功的 EID 对才会被选中；这同时验证 DDR 注册、segment
import、CTP、双侧 Jetty 和真实 RMA 写路径。`link_to` 是 NPU Runtime 初始化的前置条件；
未建立 CPU 服务 Link 时，`initialize` 会返回 `InvalidState`。
`batch_get_token_bytes` 和 `batch_get_tokens_per_block` 无默认值，
构造 `NpuRuntime` 时必须显式传入。

### 可选启动 UB device 选优

`--adaptive-ub-device` 仅在 K 端未指定 `--ub-device` 时启用启动测速：N 端脚本从
`KVAF_ADAPTIVE_UB_DEVICES` 配置的候选中选择，默认值为
`udmac0d1e2,udmac0d1e3,udmac1d1e2,udmac1d1e3`，探测请求数由
`KVAF_ADAPTIVE_UB_MISS_COUNT` 控制（默认 2048）。脚本先用轻量初始化识别 K 侧固定
设备；未固定时依次通过 `peer_ub_device` 运行完整建表和随机 BatchGet，以固定的
656B token、128 tokens-per-block、2 个 AIV core、每 core 2 个 Jetty、1 次预热和
5 次采样所得 `useful_GiBps` 最高者为最佳设备。K 端指定单个 `--ub-device` 时走原生
路径；后续正式 shape 均复用选择结果。完整校准日志写入 `ub_device_selection.log`，CSV
的 `selected_ub_device` 列记录设备名。N/K 两端必须同时使用控制协议 v7。

```bash
KVAF_CPU_SERVICE_ADDR=141.61.92.71:28999 \
KVAF_ADAPTIVE_UB_DEVICES=udmac0d1e2,udmac0d1e3,udmac1d1e2,udmac1d1e3 \
KVAF_ADAPTIVE_UB_MISS_COUNT=2048 \
  ./tests/run_remote_cpu_batch_get_miss_count_perf.sh /absolute/path/to/build-npu \
  --adaptive-ub-device
```

需要同时使用多个 NPU EID 时，设置 `NpuRuntimeOptions::npu_eids`（Python
构造参数同名）。列表中每个 EID 必须是 32 位连续十六进制字符；Runtime
为每个 EID 建立独立 BatchGet RaCtx、MR import 和 channel，并将
core/Jetty worker 按 Jetty-major 顺序轮转分配。BatchPut 保持原有单 EID
实现，使用列表第一项。显式列表不使用 rootinfo/`pair_info` 回退，
任一 EID 失败即使整组初始化失败。这种多 NPU EID 配置共享一个已选
CPU EID 和 DDR region。测试脚本可用逗号分隔的 `KVAF_NPU_EIDS` 传入。

### 按 AIV core 指定 CPU EID

`peer_eids_by_core` 为每个逻辑 AIV core（`GetBlockIdx()`）指定一个 CPU EID，
该 core 的所有 Jetty 均使用这一 EID。支持 2、4、8 核等合法 Runtime 核数，
非空列表长度必须等于 `npu_core_count`，每项为非零的 32 位十六进制 EID。
允许重复项，例如 `[A,A,A,A]` 与 `[A,B,C,D]` 可以作为同核数对照。

```python
runtime = kvaf.NpuRuntime(
    batch_get_kernel,
    batch_get_token_bytes=656,
    batch_get_tokens_per_block=128,
    batch_put_kernel_object_path=batch_put_kernel,
    npu_core_count=8,
    jettys_per_core=2,
    npu_eids=[npu_eid],
    peer_eids_by_core=cpu_eids[:8],
)
runtime.link_to(cpu_service_address)
runtime.initialize(ddr_bytes, 656 * 128)
```

CPU 只分配一块 DDR，按不同 EID 分别注册 segment/token ID；endpoint 0 为列表
第一项，并供 BatchPut 使用。重复 EID 复用注册。同一个 NPU RaCtx 导入多个远端
segment，HBM output 仍只在该本地 context 注册一次。每个 worker 的 EID、远端
token 和 TPN 来自其对应路径；kernel ABI、地址表和执行调度保持不变。

本模式目前要求至多一个显式 `npu_eids`，且 `adaptive_ub_device=False`。
`npu_eids=[]` 时会寻找能够访问整组 CPU EID 的一个本地 EID。CPU EID 不会被自动
替换；`peer_ub_device` 若设置，则所有目标必须属于该设备。CPU 服务的
`--ub-device/--eid-index` 过滤同样有效。不同 CPU device 上的 EID 可以同时使用，
前提是同一 DDR 的多 context 注册、NPU 导入和连通性均在部署环境可用。

初始化除原有 WRITE 探测外，会通过默认 endpoint 写入两轮不同数据，再通过每核
每个 Jetty READ 并校验。探测使用一个 slot，要求 `slot_size` 至少为
`batch_get_token_bytes * batch_get_tokens_per_block`，结束后释放探测 key。
探测 buffer 因 transport 注册缓存而保留到 shutdown。任一绑定、注册、建链或
READ 失败会回滚本次初始化。先等待调用方的异步任务结束再 shutdown。
日志 `PEER_CORE_CANDIDATE`、`CORE_PEER_MAPPING`、`REGION_ENDPOINT_READY` 和
`CHANNEL_CREATED` 可关联 core、Jetty、两端 EID、CPU device、endpoint、token、TPN。
`PEER_ENDPOINT_READ_VALIDATED round=1` 表示所有 worker 通过了两轮数据验证。

单组性能测试支持 `--peer-eids-by-core` 或环境变量 `KVAF_PEER_EIDS_BY_CORE`。
扫描脚本默认运行 2/4/8 核 × 每核 1/2/4 Jetty × 共享/不同 EID，共 18 组；
固定 NPU EID、请求量、随机种子及总 inflight 预算，日志分别保存且拒绝覆盖旧结果：

```bash
# CPU_EIDS 为至少 8 个不同 CPU EID，逗号分隔；NPU_EID 为一个本地 EID。
KVAF_CPU_SERVICE_ADDR=CPU_IP:28999 \
  python3 tests/python/run_peer_eid_sweep.py /absolute/path/to/build-npu \
  --peer-eids "$CPU_EIDS" --npu-eid "$NPU_EID" \
  --cores 2 4 8 --jettys 1 2 4 --max-inflight-total 2048 \
  --output-dir /absolute/path/to/peer_eid_results
```

加 `--dry-run` 可预览全部命令。`--requests` 默认 8192，须足以覆盖所有 worker；
核数、Jetty 数、预算改变时请保持共享与不同 EID 两组的其他参数一致。

每组测试保存独立日志，并向日志目录下的 `peer_eid_sweep_summary.csv` 写入一行结果。
`--output-dir` 与 `--log-dir` 等价；也可通过 `KVAF_PEER_EID_SWEEP_LOG_DIR` 设置目录，
未指定时使用当前目录下的 `peer_eid_sweep_logs`。`--csv` 或
`KVAF_PEER_EID_SWEEP_CSV` 可以单独指定 CSV 路径；命令行选项优先于对应环境变量。
例如：

```bash
KVAF_CPU_SERVICE_ADDR=CPU_IP:28999 \
  python3 tests/python/run_peer_eid_sweep.py /absolute/path/to/build-npu \
  --peer-eids "$CPU_EIDS" --npu-eid "$NPU_EID" \
  --cores 4 8 --jettys 1 2 4 --iterations 20 \
  --log-dir /absolute/path/to/peer_eid_results \
  --csv /absolute/path/to/peer_eid_results/summary.csv
```

CSV 包含 `shared`/`distinct` 模式、核数、Jetty 数、请求及 block 配置、inflight 预算、
两端 EID、样本数、时钟频率、状态和对应日志路径。性能字段与 miss-count 扫描采用相同口径：

| 字段 | 含义 |
| --- | --- |
| `mean_us`、`p50_us`、`p90_us`、`p99_us` | AIV kernel total 耗时，每个样本取各 core 最大耗时，再统计分位数。 |
| `p50_GBps` | 请求有效字节数 / kernel p50 耗时，十进制 GB/s。 |
| `post_*_us`、`completion_wait_*_us` | AIV 发请求、等待完成阶段的耗时。 |
| `batch_get_*_ms`、`batch_get_useful_GiBps` | 完整 BatchGet 的主机侧耗时，以及按平均耗时计算的有效 GiB/s。 |

AIV 微秒值直接由日志中的 cycles 换算，避免使用已舍入的毫秒值。
`--kernel-profile-clock-mhz`（或 `KVAF_BATCH_GET_KERNEL_CLOCK_MHZ`）应设置为平台的
GetSystemCycle 频率，默认 1000 MHz；该值同时传给测试进程并写入 CSV。
每行立即刷新到文件。子进程失败、正确性标记缺失或性能字段不完整时，写入 `FAIL` 和错误原因，
保留之前的结果并以非零状态退出；失败行的性能字段留空。所有组完成后输出
`PEER_EID_SWEEP_CSV_WRITTEN path=... rows=...`。已有日志或 CSV 不会被覆盖；
重复测试请更换输出目录（若单独指定 CSV，也需更换 CSV 路径）。`--dry-run` 不创建文件。

控制协议已从 v6 升至 v7，即使未配置此功能，也需要同时重编译和更新 CPU/NPU 两端。

现有 `tests/run_remote_cpu_e2e.sh` 与
`tests/run_remote_cpu_runtime_perf.sh` 接受 `KVAF_CPU_SERVICE_ADDR`，但该变量
只由测试脚本转换为上述 `link_to` 调用，不再是 Runtime 内部配置来源。

## 资源所有权

Runtime 持有且负责释放：内部 DDR、AccessTable 状态和 Key 稳定性锁。

调用方持有且必须在其 BatchGet 所在 stream 完成前保持有效：BatchPut 的连续 `input` buffer、
与 Key 一一对应的 Host `input_offsets` / `lengths` 元数据、BatchGet 的
index/offset/miss-count/output-index/output buffer，以及 NPU 上的 address/valid 表。Runtime
不会为这些调用方 buffer 分配、释放、隐式复制或注册链路。

`BatchPutRequest` 的输入语义为：`keys[i]` 的源数据位于
`input + input_offsets[i]`，长度为 `lengths[i]`。Runtime 校验每个范围位于
`input` 内，以 `lengths` 调用 `PreparePut`，再将相同请求和目标地址描述交给
Operator。因此同一 batch 可以包含不同长度的 KV block；input 可以包含对齐
padding，offset 无需固定步长。

配置 AIV BatchPut 时，`DefaultAccessOperator` 将 Host metadata 复用 staging
buffer 上传到 NPU，并将 `PreparePut` 返回的 destination address 上传为
`uint64[row_count]`；NPU input HBM 则按 `(地址, bytes)` 注册/缓存。BatchPut 使用
调用方 stream 并在该 stream 同步完成后才允许 Runtime Commit；core status 是
Operator 内部诊断状态，不是 Python 参数或返回值。

NPU BatchGet 的 `status` 和 `queue_state` 是 AIV kernel ABI 必需的内部 device
buffer，由 `DefaultAccessOperator.Initialize` 申请并在 `Shutdown` 释放；Python
和 `BatchGetRequest` 都不传它们。显式 `synchronize=True` 时，Operator 可读取
其 completion status 并以 `[KVAF_AIV] BATCH_GET_DEVICE_ERROR` 诊断日志记录
设备侧错误；它不会成为 Python 的额外返回值。

## 当前 POC 的 NPU AccessTable 边界

`PrepareAccessTable` 生成 Runtime-owned 的扁平 Host `address/length/valid` 表并
持有对应 Key 锁。C++ `GetAccessTableView` 返回指向三段连续 Host 内存的 non-owning
view，不是 Tensor，也不会复制数据。

Python `NpuRuntime.access_table_view(handle, device="npu", capacity=N)` 则创建
`DeviceAccessTable`：它持有 `addresses: torch.int64[N]` 和 `valid: torch.uint8[N]`
两个 NPU Tensor，并在图外完成一次 Host-to-NPU materialize。这个对象可直接传给
`NpuRuntime.batch_get`，调用者无需再传 `access_table` handle、裸指针、字节数或
手写 scatter。也可以通过重载直接传入 `addresses` 和 `valid` Tensor；此时 Runtime
无法验证对应 Host AccessTable 的生命周期和表项长度，调用方必须自行保证。
`capacity` 可大于有效 key 数；有效数据位于前缀，其余 `valid=0`。

`DeviceAccessTable` 及其两个 Tensor 必须存活到关联 BatchGet stream 完成，且其
source handle 必须在此后才可 `ReleaseAccessTable`。Runtime 不额外注册 caller 的
address/valid Tensor。

`valid` 是每个 AccessTable entry 的安全位：`1` 表示可访问，`0` 表示 padding
或无效 entry。调用方必须只把有效访问放进 `[0, miss_count)` 请求前缀；该前缀
之外的容量行不会被 AIV kernel 读取或搬运。
当前 InMemory MetaManager 对不存在的 Key 在 `PrepareAccessTable` 的
`LockKeys` 阶段直接返回 `kNotFound`，因此不会产生一张含该 Key 的
`valid=0` 表。

当前 AIV BatchGet ABI 通过 `batch_get_token_bytes` 配置每个 WQE 读取的单 token
KV 字节数，通过 `batch_get_tokens_per_block` 配置每个 block 的 token 数。
每个 NPU AccessTable valid entry 长度必须等于两者乘积。`indices[i]` 选择
entry，`offsets[i]` 给出 entry 内
字节偏移，最终地址为 `address_table[indices[i]] + offsets[i]`。Runtime 在
`PrepareAccessTable` 时一次性
计算这一不变量，之后每次 BatchGet 只做 O(1) 比较；不会在数据面热路径重新扫描
整张表。

设备侧 `core_status.result`（例如 `kInvalidLookup`）是 Operator 内部诊断状态，
不是 Python BatchGet 的输入/输出。显式 `synchronize=True` 时会记录错误 marker；
默认 enqueue-only 调用由调用方在同一 stream 上的后续工作或自身同步决定何时观察
完成。参数校验、ACL 分配/提交和显式 stream 同步失败仍以 Runtime 错误返回。

## 测试

默认测试均由 Python 脚本定义场景和断言，并直接调用 pybind 的
`kvaf.Runtime`；没有 C++ Mock driver 或子进程协议：

```bash
cmake -S . -B build -DBUILD_TESTING=ON -DKVAF_BUILD_PYTHON=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

`tests/python/test_runtime_mock.py` 可以直接编辑；其中覆盖 Runtime 初始化、
Put/Get 生命周期、AccessTable 锁定、slot 回收、参数校验和 Operator 失败回滚。
脚本会以 `try/except` 探测 `torch_npu`、MPS、CPU 和无 PyTorch 环境：MPS 可用时
用于在本机生成 caller-side 测试数据，然后直接传给 CPU Mock。MPS 不是 CANN
AIV 或 Direct-HCCP 的替身，因此不能作为 NPU 数据面验证结论。

## A5 / CANN BatchGet 集成

`KVAF_ENABLE_AIV_BATCH_GET=ON` 配合 `KVAF_BUILD_PYTHON=ON` 会构建 merged
demo 的 kernel/host 库、`DefaultAccessOperator` 和 `kvaf.NpuRuntime`。
Python 脚本创建 NPU Tensor；通过 Runtime materialize `DeviceAccessTable`，然后
直接传入 `batch_get`：

```bash
./build.sh npu

KVAF_CPU_SERVICE_ADDR=141.61.92.71:28999 \
DEVICE_ID=0 \
KVAF_NPU_TOKEN_BYTES=656 \
KVAF_NPU_TOKENS_PER_BLOCK=128 \
KVAF_NPU_ACTIVE_PERCENT=5 \
tests/run_remote_cpu_e2e.sh "$PWD/build-npu"
```

该脚本用小规模 case 验证 Runtime 初始化、调用方 Device Tensor、Runtime
materialize 的 `DeviceAccessTable`、BatchGet 输出字节、`output A -> B -> A` 的 local-buffer 缓存复用及
teardown。它以 `KVAF_NPU_ACTIVE_PERCENT` 决定 `miss_count`（例如 `5` 表示约 5%
请求有效），只为 `[0, miss_count)` 前缀写入 KV；随后可用 Runtime 的 ACL D2H mock
`batch_put` 将 caller-owned NPU source Tensor 写入 Runtime-owned DDR。该小型
BatchGet 用例不传 BatchPut kernel，因而不启动 Direct-HCCP BatchPut transport。
默认使用 2 个 AIV core，每个 core 使用 2 个 Jetty；可通过
`KVAF_NPU_CORE_COUNT` 和 `KVAF_NPU_JETTYS_PER_CORE` 覆盖。BatchGet 每个 core
支持 1 到 64 个 Jetty，BatchPut 仍最多使用 2 个。

设置 `KVAF_NPU_GRAPH_REPLAYS` 可在同一个 Python 用例中额外运行 Python-owned 的
raw `kvaf.AclGraph` 图模式：

```bash
KVAF_NPU_GRAPH_REPLAYS=2 \
KVAF_NPU_WARMUP=10 \
KVAF_NPU_ITERATIONS=100 \
# 其余 KVAF_RUN_NPU_TESTS / kernel / shape 环境变量与上例相同
timeout 240s python3 tests/python/test_npu_batch_get.py
```

图对象、capture、replay 和销毁都由 Python 测试持有的 `kvaf.AclGraph` 管理；KVAF
Runtime 与 AccessOperator 不创建或保存图。`AclGraph` 是 pybind 暴露的薄封装，直接
调用 ACL Runtime Graph 的 `aclmdlRICaptureBegin`、`aclmdlRICaptureEnd`、
`aclmdlRIExecuteAsync` 和 `aclmdlRIDestroy`。脚本先在图外 eager 执行
`output A -> B -> A`，使 A 的 local HBM 注册进入 transport cache；再在 raw ACL
capture 中只 capture `runtime.batch_get(..., stream=capture_stream)`。

每次 replay 前，Python 在同一 ACL stream 上以
`AclGraph.copy_host_to_device_async` 更新固定地址的 `address_table`、`indices`、
`miss_count`、`output_indices`；capture 中的 BatchGet 参数和所有 NPU tensor 地址不变。正常
性能口径是 `graph_replay_to_stream_completion`：五项 H2D 更新会在 replay 前完成并同步，
不进入计时窗口；计时仅包含 graph replay 到同一 stream 完成，与 eager 的
`runtime_batch_get_call_to_stream_completion` 口径对齐。默认 graph 的 warmup/iterations
与 eager 一律使用相同的 `KVAF_NPU_WARMUP` / `KVAF_NPU_ITERATIONS` 采样计划。

AIV demo 使用原生
`aclrtMemcpyAsync(fixed_device_inputs, stream) -> aclmdlRIExecuteAsync(graph, stream)`。
Python 测试使用调用方持有的 `torch.npu.Stream`；`AclGraph.replay(stream)` 显式传入
capture 使用的同一 stream。图对象会在 Runtime shutdown 前 reset，避免图仍持有调用方
buffer 时释放 HCCP transport。

Python `batch_get` 的 `stream=None` 默认解析为 `torch.npu.current_stream()`，
`synchronize=False` 默认只提交工作。同一 stream 的后续 Torch/ACL 运算天然等待
BatchGet；如果调用方显式切换 stream，则由调用方建立 event/wait 依赖，并在
teardown 前完成每个使用过的 stream。只有排障或单独等待时才需要
`synchronize=True`。

正确性验证完成后，脚本会输出
`PERF_RESULT metric=runtime_batch_get_call_to_stream_completion`。这个计时窗口
从 `Runtime.BatchGet` 调用开始，到 AIV adapter 中的 stream 同步完成结束，排除
DDR 注册、Device buffer 初始化和 Host-to-Device 表更新。输出包括 mean、p50、
p90、p99 和 useful GiB/s。`ctest` 和上面的命令均设置 240 秒上限；未设置
`KVAF_RUN_NPU_TESTS=1` 时会明确跳过。

`DIRECT_LOCAL_BUFFER_ACQUIRE_NEW` 会始终记录新的 output HBM 注册；高频的
cache-hit / release 日志默认关闭，避免压测污染终端。如需排查 LocalBuffer 引用
计数，可额外设置 `KVAF_AIV_VERBOSE_LOCAL_BUFFER=1`。

## A5 / CANN BatchPut 集成

`tests/python/test_npu_batch_put.py` 通过 `kvaf.NpuRuntime` 验证真实的 AIV
Direct-HCCP BatchPut 路径，不使用 demo test binary 或 ACL D2H mock。默认每轮写入
4096 个 1 KiB block。写入完成后通过 Runtime 的 Host AccessTable 检查 Meta 已提交
的长度、有效位和远端 DDR payload。测试流程与 BatchGet 一致：先进行一次正确性
验证，再执行 warmup 和正式采样，最后重新校验采样结果。测试会输出 `PERF_RESULT`，统计
`runtime.batch_put()` 到 stream 完成的 mean、p50、p90、p99 和有效 GB/s。

BatchPut 的 Runtime binding 依赖 BatchGet 的公共 NPU Runtime 构建，因此
`build.sh npu` 一次构建两个 AIV kernel、host transport 与 Python binding。端到端
正确性入口与上节相同；它会在 BatchGet 前验证 AIV Direct-HCCP BatchPut。

测试规模复用 BatchGet 的 `KVAF_NPU_REQUESTS`、`KVAF_NPU_TOKEN_BYTES`、
`KVAF_NPU_TOKENS_PER_BLOCK`、
`KVAF_NPU_CORE_COUNT`、`KVAF_NPU_JETTYS_PER_CORE`、`KVAF_NPU_WARMUP` 和
`KVAF_NPU_ITERATIONS`。启用 `BUILD_TESTING` 后，该脚本也会注册为
`kvaf_npu_batch_put_python`；没有设置 `KVAF_RUN_NPU_TESTS=1` 时会明确跳过。

## NPU Runtime 性能脚本

`tests/python/test_npu_runtime_perf.py` 是独立的 steady-state 性能脚本，不纳入
`ctest`。它分别测量：固定 128-key batch 的 `BatchPut`、一次全量
`PrepareAccessTable`、一次全量 `access_table_view(device="npu")`、token 级随机请求
的 `BatchGet`、全量 `ReleaseAccessTable`，以及 128-key batch 的 `BatchFree`。
BatchGet/Put 的每个样本均在 Runtime 调用后由调用方 `torch.npu.synchronize()`
结束；随机源计划与数据填充不进入 BatchGet 提交计时。

每个 BatchGet 样本生成 `source_token_ids`、随机输出位置和一个单元素
NPU `int32 miss_count`。调用前派生 `indices`、`offsets` 和
`output_indices`；只有这三个数组的 `[0, miss_count)` 前缀会被搬运，
不再生成 mask。脚本输出 submit、synchronize 和 BatchGet 的
`PERF_RESULT`。

```bash
KVAF_CPU_SERVICE_ADDR=141.61.92.71:28999 \
DEVICE_ID=0 \
KVAF_RUNTIME_PERF_TOKEN_BYTES=656 \
KVAF_RUNTIME_PERF_TOKENS_PER_BLOCK=128 \
KVAF_REMOTE_CPU_PERF_TIMEOUT=600 \
tests/run_remote_cpu_runtime_perf.sh "$PWD/build-npu"
```

测试默认 `batch=16`、`token_bytes=656`、`tokens_per_block=128`，
全表为 2,097,152 个 83,968B entry，每次 BatchGet 32,768 requests、32 个 AIV core、
每 core 2 个 Jetty、3 次 warmup 与
100 个样本。可用
`KVAF_RUNTIME_PERF_BATCH`、
`KVAF_RUNTIME_PERF_AIV_CORES`、`KVAF_RUNTIME_PERF_WARMUP`、
`KVAF_RUNTIME_PERF_ITERATIONS`、`KVAF_RUNTIME_PERF_ACTIVE_PERCENT`、
`KVAF_RUNTIME_PERF_TOKEN_BYTES` 和 `KVAF_RUNTIME_PERF_TOKENS_PER_BLOCK` 覆盖；
Jetty 数量可通过
`KVAF_RUNTIME_PERF_JETTYS_PER_CORE` 覆盖。全量 data population 会写入 2,097,152 个
逻辑 KV entry；脚本显式配置 AIV Direct-HCCP BatchPut kernel，BatchPut 性能样本
测量 Runtime metadata upload、AIV WRITE 及其 caller-stream 完成，不包含全量数据
填充。
脚本以 CPU `torch.int64` Tensor 传入全量 Prepare key，避免 Python `list[int]` 转换污染
这项控制面性能数据。
