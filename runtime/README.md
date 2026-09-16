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

CPU DDR region 支持四种页分配模式，通过 CPU 服务启动参数选择：

```bash
# 保留原行为：4 KiB 对齐的普通匿名内存；操作系统仍可能自行使用 THP。
./build-cpu/runtime/kvaf_cpu_service \
  --listen 0.0.0.0 --port 28999 --memory-page-mode default

# 显式使用基础页：独立匿名 VMA + MADV_NOHUGEPAGE，不受节点 THP=always 策略影响。
./build-cpu/runtime/kvaf_cpu_service \
  --listen 0.0.0.0 --port 28999 --memory-page-mode base-4k

# 2 MiB 对齐，MADV_HUGEPAGE，并在注册前主动触页；是否全部合并仍由 Linux THP 决定。
./build-cpu/runtime/kvaf_cpu_service \
  --listen 0.0.0.0 --port 28999 --memory-page-mode thp

# 显式从预留的 2 MiB hugetlb 池分配；region 字节数必须是 2 MiB 的整数倍。
./build-cpu/runtime/kvaf_cpu_service \
  --listen 0.0.0.0 --port 28999 --memory-page-mode hugetlb-2m
```

`base-4k` 只对当前 CPU service 分配的 DDR region 禁用 THP，不修改
`/sys/kernel/mm/transparent_hugepage/*` 的节点全局策略；分配时若
`MADV_NOHUGEPAGE` 失败则直接拒绝请求，不会静默回退。它适合在节点设置为
`THP=always` 时建立稳定的基础页对照组。测试期间应确认服务日志包含
`page_mode=base-4k`，并通过 `/proc/<pid>/smaps` 验证对应 region 的
`AnonHugePages` 为 0。

`hugetlb-2m` 不会在失败时静默回退。启动服务前必须在目标 NUMA node
预留足够的 2 MiB hugepages；例如默认 2 GiB region 至少需要 1024 页：

```bash
grep -E 'HugePages|Hugepagesize' /proc/meminfo
sudo sysctl -w vm.nr_hugepages=1100
```

需要固定 NUMA 位置时，应在 UB device 所在 node 启动服务，例如
`numactl --cpunodebind=N --membind=N ...`。`REGION_ALLOCATED` 日志会记录
`page_mode`。THP 模式还应通过 `/proc/<pid>/smaps` 的 `AnonHugePages` 验证实际
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

每张 N 卡可在 K 端显式限定的 UB device 中独立测速选优。
K 端重复传入 `--ub-device`，只发布这些设备的全部可用 EID，例如：

```bash
./build-cpu/runtime/kvaf_cpu_service --listen 0.0.0.0 --port 28999 \
  --ub-device udmac0d1e2 \
  --ub-device udmac0d1e3 \
  --ub-device udmac1d1e2 \
  --ub-device udmac1d1e3
```

未指定白名单时仍枚举所有设备；指定名称没有可用 EID 时服务启动报错。
N 端在原有构造参数上增加 `adaptive_ub_device=True`：

```python
runtime = kvaf.NpuRuntime(
    batch_get_kernel,
    batch_get_token_bytes=656,
    batch_get_tokens_per_block=128,
    device_id=0,
    npu_core_count=24,  # 与业务配置保持一致
    batch_put_kernel_object_path=batch_put_kernel,
    adaptive_ub_device=True,
    adaptive_ub_miss_count=2048,
)
runtime.link_to("141.61.92.71:28999")
runtime.initialize(ddr_bytes, kv_block_bytes)
```

- 默认 `False` 保持首个连通候选即选中的行为。
- 开启后，仅当 K 端显式传入至少一个 `--ub-device` 时执行测速，并在所有显式
  设备中选优，不限制设备数或 EID 数。K 端未指定 `--ub-device` 时，跳过测速并
  保持原有“首个可用候选”逻辑。
- 寻优单位是 K 端 `ub-device`，不是 EID 组合。每个设备内部按 K EID、N EID 的
  顺序尝试建链，首个通过建链和 WRITE/CQ 验证的组合代表该设备参加一次测速，
  该设备剩余 EID 不再测试。显式 `npu_eids` 仍作为一个严格的完整多 EID 组，
  不会拆开或回退。K 端指定 `--eid-index 0` 时，每个设备只有一个 K EID，自动
  发现的 N EID 只作为建链回退，因此 4 个设备最多得到 4 个测速结果。
- 每个设备的可用组合先完成原有 WRITE/CQ 验证，再在独立临时 region 中执行真实
  BatchGet。标定沿用 Runtime 的 `batch_get_token_bytes` 和
  `batch_get_tokens_per_block`，请求数由 `adaptive_ub_miss_count` 指定。单次性能测试
  会传入当前 shape 的 `miss_count`；miss-count sweep 固定用 `656 × 2048` 校准一次。
  core/Jetty 配置沿用业务配置，所有候选使用同一组数据与请求。
- 先写入并校验数据，预热 1 次、采样 5 次，按每次最慢 core 的 kernel 总周期数
  取中位数，选择耗时最短的组合。相同 N 卡、相同搬运字节数下，这等价于选择最高
  标定带宽。建链、分配和 Host 同步开销不计入排名；日志输出周期数和 bytes/cycle，
  不假定硬件时钟频率。
- 探测 region 根据当前 block 大小和 miss count 计算，并向上对齐到 block 大小与
  2 MiB 的最小公倍数，支持现有 `hugetlb-2m` 页模式。各设备串行创建和释放；每个
  组合完成后先销毁通道和 region，再释放 NPU
  buffer。全部探测结束后，按排名建立用户请求大小的正式 region，校验连通性，
  最优组合重建失败时尝试下一名；没有可用组合则初始化失败。
- `UB_DEVICE_PROBE_PLAN` 记录各设备可能尝试的建链组合数，
  `UB_DEVICE_PROBE_RESULT` 每个连通设备最多记录一次标定结果，`UB_DEVICE_SELECTED` 记录正式
  选择；用 `peer_candidate` 对照 K 端 `CANDIDATE_DISCOVERED` 的设备名称/EID。
  结果属于当前 NpuRuntime session，后续 BatchGet/BatchPut 复用该选择；不做运行中
  切换。不同 N 卡可以选不同 K 端设备。

现有性能脚本可直接验证：

```bash
KVAF_CPU_SERVICE_ADDR=141.61.92.71:28999 DEVICE_ID=0 \
  ./tests/run_remote_cpu_batch_get_perf.sh /absolute/path/to/build-npu \
  --adaptive-ub-device --token-bytes 656 --miss-count 2048
```

miss-count sweep 传入 `--adaptive-ub-device` 时，会先自动启动一个 `656 × 2048`
校准进程，保存 `UB_DEVICE_SELECTED` 的设备名，再以 `peer_ub_device` 固定该设备运行
全部正式 shape；用户只需启动一次脚本。后续进程仍独立建链，同一设备内保留 EID
回退，但不再执行带宽标定。CSV 为每个 shape 写入 `selected_ub_device`；未执行自适应
选优或固定设备时该列为空。

EID 组合数只影响找到每个设备首个可用链路前的回退开销；沿用现有
传输/kernel 超时，当前不新增整个选优过程的硬时间上限。多卡同时启动会相互
争抢带宽，标定效果需要在目标部署并发下验证。sweep 的固定选择代表 `656 × 2048`
校准 shape。

**升级要求：**候选查询包含独立设备数和显式过滤标记，并支持在分配前查询候选
所属设备，控制协议从 v4 升到 v6。
N 端库和 K 端服务必须一起重新构建/部署，即使未开启自适应也不能混用旧协议。

需要同时使用多个 NPU EID 时，设置 `NpuRuntimeOptions::npu_eids`（Python
构造参数同名）。列表中每个 EID 必须是 32 位连续十六进制字符；Runtime
为每个 EID 建立独立 BatchGet RaCtx、MR import 和 channel，并将
core/Jetty worker 按 Jetty-major 顺序轮转分配。BatchPut 保持原有单 EID
实现，使用列表第一项。显式列表不使用 rootinfo/`pair_info` 回退，
任一 EID 失败即使整组初始化失败。当前这些 NPU 路径仍共享一个已选
CPU EID 和 DDR region。测试脚本可用逗号分隔的 `KVAF_NPU_EIDS` 传入。

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
KVAF_NPU_JETTYS_PER_CORE=2 \
tests/run_remote_cpu_e2e.sh "$PWD/build-npu"
```

该脚本用小规模 case 验证 Runtime 初始化、调用方 Device Tensor、Runtime
materialize 的 `DeviceAccessTable`、BatchGet 输出字节、`output A -> B -> A` 的 local-buffer 缓存复用及
teardown。它以 `KVAF_NPU_ACTIVE_PERCENT` 决定 `miss_count`（例如 `5` 表示约 5%
请求有效），只为 `[0, miss_count)` 前缀写入 KV；随后可用 Runtime 的 ACL D2H mock
`batch_put` 将 caller-owned NPU source Tensor 写入 Runtime-owned DDR。该小型
BatchGet 用例不传 BatchPut kernel，因而不启动 Direct-HCCP BatchPut transport。
每个 AIV core 默认使用 2 个 Jetty；可通过 `KVAF_NPU_JETTYS_PER_CORE=1` 切换为
单 Jetty 验证。

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
