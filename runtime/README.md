# KVAF Runtime

## 源码目录

公共 C++ 头文件继续放在 `include/`，包含方式与构建 target 不变。实现按职责组织：

- `src/`：`RuntimeCore`、`NpuRuntime` 入口和 NPU 配置。
- `execution/`：执行配置解析、Dispatcher 与执行资源管理；驻留上下文及绑定的公共定义仍在 `include/`。
- `memory/`：缓存池服务、池分配与 session 工厂、CPU DDR 注册复用、UBMEM allocator。
- `peer/`：NPU/CPU 链路和 execution 控制协议。
- `cpu_service/`：CPU Runtime 与服务入口、TCP 分发；`backends/` 放 URMA/UBMEM backend，`urma/` 放发现和通道基础，`gather/`、`push/` 放对应 worker。

私有头文件以 Runtime 根目录为包含根，例如 `memory/cpu_shared_registration.h`；
公开头文件不依赖这些私有路径。CPU 服务可执行文件仍输出到 `build-*/runtime/kvaf_cpu_service`。


## 模块定位

KVAF Runtime 是对外的 KV 控制面入口。它不处理 Prefix Key 或
Forward 语义；它只拥有内部 DDR 的生命周期，编排 MetaManager 与
AccessOperator，并把调用方已经准备好的输入/输出 buffer 交给数据面。

## 已实现能力

- `Initialize`：经 `DdrMemoryAllocator` 分配 CacheRegion，初始化 Dispatcher 和
  MetaManager；具体 provider 的数据面资源在首次使用或 warmup 时准备。
- `BatchPut`：准备资源和事务后提交数据面，确认完成才 `CommitPut`；确定失败可
  `AbortPut`，未知结果保留事务和引用，需 recover 确认。NPU 路径按执行配置选择
  AIV URMA、AICPU 或 UBmem；AIV 缺少 kernel 时明确报错，不回退到 D2H mock。
- `PrepareAccessTable` / `GetAccessTableView` / `ReleaseAccessTable`：持有
  MetaManager 稳定读锁，生成一维 address/length/valid 表；Release 使公开句柄失效，
  最后一个在途请求或图节点引用释放后才解锁。
- `BatchGet`：校验调用方的 index、miss-count、output 和（NPU 路径）Device
  AccessTable buffer 后，向调用方 stream 提交 `AccessOperator::ExecuteBatchGet`。
  Python 默认不在 KVAF 内同步；同一 stream 的后续操作自然按序执行。
- `BatchFree`：将删除请求交给 MetaManager；仍被 AccessTable 持有的 Key
  会返回 `kBusy`。

Get 支持异步提交和显式 Drain；每个图节点使用独立租约，同一 Runtime 可在同一
stream 上串行回放多张图，每张图可包含多个 Get。图对象生命周期由上层管理，
BatchPut 图模式未接入；不支持跨 stream 并发回放。

## CPU Runtime 服务与 NPU Link

`Runtime` 是角色接口；`RuntimeCore` 保留已有的 DDR、MetaManager 与 Operator
编排。当前 `NpuRuntime` 持有 NPU 数据面，`CpuRuntime` 持有 CPU DDR、UMDK
segment/token/Jetty 与会话。TCP 监听是无状态的 `RuntimeService`：它只接收连接并
分发给 Runtime，不拥有硬件资源。

CPU 服务需要 UB 驱动、`/dev/uburma`、UMDK 公开头文件、`liburma.so` 和 OpenMP；不链接
或加载 CANN、HCCP、ACL。仓库根目录的统一构建入口会自动拉取固定 UMDK 源码，并在系统
缺少 `liburma.so` 时只构建所需的 UMDK `urma` target：

```bash
cd /path/to/kvaf
./build.sh cpu

./build-cpu/runtime/kvaf_cpu_service \
  --listen 0.0.0.0 --port 28999
```

CPU Gather 默认线程总预算为 10，等待模式为 `active`；可用 `--gather-threads N` 和
`--gather-mode adaptive|active|passive` 显式覆盖。

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
`numactl --cpunodebind=N --membind=N ...`。Gather 线程继承外部 CPU 亲和性，在允许的
CPU 范围内由系统调度；也可用 `taskset -c CPU_LIST` 限定调度范围。`REGION_ALLOCATED` 日志会记录
`page_mode`。默认 THP 模式还应通过 `/proc/<pid>/smaps` 的 `AnonHugePages` 验证实际
合并量；Linux 使用大页并不保证 UDMA/IOMMU 一定保留相同的页表粒度。

默认自动枚举所有可用 UMDK UB device/EID；`--ub-device NAME` 和
`--eid-index N` 仅用于诊断。CPU 端可同时保持多个 NPU session，NPU 正常
`shutdown` 后会释放对应 DDR、segment、token 与 Jetty。

NPU 应用默认只配置 CPU 服务地址，不配置 EID 或 UDMA device：

```python
put = kvaf.PutExecutionConfig("urma", "aiv")
get = kvaf.GetExecutionConfig("urma", "aiv", "direct")
runtime = kvaf.NpuRuntime(
    batch_get_token_bytes=656,
    batch_get_tokens_per_block=128,
    device_id=0,
    transport_mode="urma", default_put=put, default_get=get,
)
runtime.link_to("141.61.92.71:28999")
runtime.initialize(ddr_bytes, kv_block_bytes)
```

已知设备名时可设置 `peer_ub_device="udmac0d1e2"`，固定 CPU 侧候选设备。
Runtime 使用该设备的首个候选，不进行启动带宽测速或探测写入。

`link_to` 必须在 `initialize` 前调用。URMA 初始化从 HCCL root-info 读取当前物理卡的
CLOS PG EID，并尝试追加 `/proc/asdrv_ub/pair_info` 的候选；未指定 EID 时采用首项。
初始化建立 CacheRegion 和元数据；具体 transport、MR、队列、Gather 工作区及 AICPU
包在首次使用或显式 warmup 时准备，初始化成功不代表数据面已通过 RMA 验证。
未建立 CPU 服务 Link 时，`initialize` 返回 `InvalidState`。
`batch_get_token_bytes` 和 `batch_get_tokens_per_block` 无默认值，构造时必须提供；
其他构造参数仅支持关键字。算法通过默认执行配置或本次 `execution` 指定。默认核数/worker 数
为 URMA AIV Put/Get 2、UBMEM AIV Put/Get 16、AICPU Put 1、AICPU Gather/Scatter Get 5；
URMA AIV Jetty 数默认 2、URMA AIV Get 在途上限默认 2048。显式资源项可覆盖各自默认值。
AIV/UBMEM 默认使用本次构建目录 `bin/` 下相应 Put/Get `.o` 的绝对路径，自定义
`./build.sh npu --build-dir DIR` 自动生效。`put_resources/get_resources` 可按配置覆盖预算
和路径；AICPU 仍从 CANN 安装目录定位设备包。不再提供旧全局后端参数或 adaptive 启动选优参数。

### 固定 UB device 后测量

启动带宽选优已移除。需要固定 CPU 侧 UB 设备时，直接在统一 Get 性能入口传入
`--peer-ub-device`，并在各组对照中保持相同设置：

```bash
KVAF_CPU_SERVICE_ADDR=127.0.0.1:21111 DEVICE_ID=2 \
bash tests/run_remote_cpu_batch_get_perf.sh build-npu \
  --backend aiv --peer-ub-device udmac0d1e2 \
  --aiv-cores 2 --request-length 32768 --table-capacity 4096 \
  --miss-count 4096 --token-bytes 656 --tokens-per-block 128 \
  --warmup 10 --iterations 100
```

需要同时使用多个 NPU EID 时，设置 `NpuRuntimeOptions::npu_eids`（Python
构造参数同名）。列表中每个 EID 必须是 32 位连续十六进制字符；Runtime
为每个 EID 建立独立 BatchGet RaCtx、MR import 和 channel，并将
core/Jetty worker 按 Jetty-major 顺序轮转分配。BatchPut 保持原有单 EID
实现，使用列表第一项。显式列表不使用 rootinfo/`pair_info` 回退，
AIV Get 在首次准备时按自己的 `cores * jettys` 预算校验整个列表；
任一 EID 的路径准备失败会使该 provider 准备失败，不会截短列表。当前这些 NPU 路径仍共享一个已选
CPU EID 和 DDR region。测试脚本可用逗号分隔的 `KVAF_NPU_EIDS` 传入。

现有 `tests/run_remote_cpu_batch_get_perf.sh` 与
`tests/run_remote_cpu_runtime_perf.sh` 接受 `KVAF_CPU_SERVICE_ADDR`，但该变量
只由测试脚本转换为上述 `link_to` 调用，不再是 Runtime 内部配置来源。

## 资源所有权

Runtime 持有且负责释放：内部 DDR、AccessTable 状态和 Key 稳定性锁。

调用方持有且必须在其 BatchGet 所在 stream 完成前保持有效：BatchPut 的连续 `input` buffer、
与 Key 一一对应的 Host `input_offsets` / `lengths` 元数据、BatchGet 的
index/offset/miss-count/output-index/output buffer，以及 NPU 上的 address/valid 表。Runtime
不创建或释放这些调用方分配；provider 负责必要的注册、映射和请求元数据 staging。
Python 保留在途调用所需引用，C++ 调用方自行保持分配存活；外部 stream 同步不能
代替 `drain_get()` 的执行租约及引用回收。

`BatchPutRequest` 的输入语义为：`keys[i]` 的源数据位于
`input + input_offsets[i]`，长度为 `lengths[i]`。Runtime 校验每个范围位于
`input` 内，以 `lengths` 调用 `PreparePut`，再将相同请求和目标地址描述交给
Operator。因此同一 batch 可以包含不同长度的 KV block；input 可以包含对齐
padding，offset 无需固定步长。

配置 AIV BatchPut 时，`DefaultAccessOperator` 将 Host metadata 复用 staging
buffer 上传到 NPU，并将 `PreparePut` 返回的 destination address 上传为
`uint64[row_count]`；普通 Put 临时注册 NPU input HBM，显式输入绑定可跨调用保留注册。BatchPut 使用
调用方 stream 并在该 stream 同步完成后才允许 Runtime Commit；core status 是
Operator 内部诊断状态，不是 Python 参数或返回值。

NPU BatchGet 的 `status` 和 `queue_state` 是 AIV kernel ABI 必需的内部 device
buffer，在相应 provider 首次准备时申请并在安全关闭时释放；Python
和 `BatchGetRequest` 都不传它们。AIV/UBMEM eager 完成路径只等待并释放请求资源，
不回读或检查 Get 状态报告；显式 profile 快照仍读取并验证设备报告。

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

`DeviceAccessTable` 及其两个 Tensor 必须在关联 BatchGet 完成前保持有效。已提交的
Get 或图节点保留 Host AccessTable 引用；`ReleaseAccessTable` 使公开句柄失效，
最后一个引用退出后才解除槽位 pin。直接传入裸 address/valid Tensor 时，调用方
须自行保证其中地址指向的 DDR 仍有效。Runtime 不额外注册 caller 的 address/valid Tensor。

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
不是 Python BatchGet 的输入/输出。默认 enqueue-only 调用由调用方在同一 stream
上的后续工作或自身同步决定何时观察完成。参数校验、ACL 分配/提交和显式 stream 同步失败仍以 Runtime 错误返回。

## A5 / CANN BatchGet 性能测试

`KVAF_ENABLE_AIV_BATCH_GET=ON` 配合 `KVAF_BUILD_PYTHON=ON` 构建 AIV kernel/host 库、
`DefaultAccessOperator` 和 `kvaf.NpuRuntime`。统一构建命令包含四种 Get 后端：

```bash
./build.sh npu
KVAF_CPU_SERVICE_ADDR=127.0.0.1:21111 DEVICE_ID=2 \
bash tests/run_remote_cpu_batch_get_perf.sh build-npu \
  --backend aiv --aiv-cores 2 --execution both \
  --request-length 32768 --table-capacity 4096 --miss-count 4096 \
  --token-bytes 656 --tokens-per-block 128 \
  --warmup 10 --iterations 100 --kernel-profile
```

CPU service 须预先启动。替换 `--backend` 及对应的 `--aiv-cores` / `--aicpu-workers` 可测 UBMEM、Gather、Scatter，
具体参数及计时范围见[统一 Get 性能测试](../docs/batch_get_performance.md)。
独立功能测试及其 CMake 注册已移除，数据预检和输出校验继续在性能计时外执行。
默认关闭 kernel profiling；上例显式使用 `--kernel-profile` 开启设备打点与报告读取，
测无 profiling 的端到端性能时去掉该参数。`--multi-graph-check` 可在计时前验证
同 stream 上两张图（其中一张含两个 Get）及其中一张释放后的继续回放。

图对象由 Python 持有；Runtime 不创建或保存图。`kvaf.AclGraph` 直接封装
`aclmdlRICaptureBegin`、`aclmdlRICaptureEnd`、`aclmdlRIExecuteAsync` 和 `aclmdlRIDestroy`；
统一性能入口也支持 `--graph-api torch`。捕获前通过 `begin_get_graph` 准备配置、stream、
输出和容量，捕获只提交 Get。回放前完成输入更新，计时只包含 replay 与 stream 等待。
同一 Runtime 可在同一 stream 上串行回放多张图，每个 Get 节点使用独立租约。
销毁一张图后结束它的全部租约；所有租约释放后才能关闭 Runtime。图租约存活时，
Get 方向专用于图。同算法有多个图节点时，状态和 profile 查询通过 `graph_lease`
选择节点；profiling 在准备图前开启，在销毁图并结束租约后关闭。
用法见 [Get 图模式](../docs/cache_transport_execution_implementation.md#get-图模式)。

`batch_get(stream=None, synchronize=False)` 默认在 `torch.npu.current_stream()` 上提交。
同一 stream 的后续操作等待 Get；换 stream 时由调用方建立 event/wait 依赖。
`drain_get()` 完成在途请求并清理请求引用。性能测试默认使用 `async-drain`，
一次同步调用使用 `sync-call`。
AIV/UBMEM eager Get 不回读设备状态；ACL 提交与同步失败仍返回错误。

`DIRECT_LOCAL_BUFFER_ACQUIRE_NEW` 记录新 output HBM 注册；排查引用计数时可设置
`KVAF_AIV_VERBOSE_LOCAL_BUFFER=1`，正式性能测量应关闭高频日志。

## A5 / CANN BatchPut 集成

`tests/python/test_npu_batch_put.py` 通过 `kvaf.NpuRuntime` 验证真实的 AIV
Direct-HCCP BatchPut 路径，不使用独立测试程序或 ACL D2H mock。默认每轮写入
4096 个 83,968 B block（656 B/token × 128 token/block）。写入完成后通过 Runtime 的 Host AccessTable 检查 Meta 已提交
的长度、有效位和槽位地址；当前预检属于 metadata 校验。测试流程与 BatchGet 一致：先进行一次正确性
验证，再执行 warmup 和正式采样，最后重新校验采样结果。测试会输出 `PERF_RESULT`，统计
`runtime.batch_put()` 到 stream 完成的 mean、p50、p90、p99 和有效 GB/s。

BatchPut 的 Runtime binding 依赖 BatchGet 的公共 NPU Runtime 构建，因此
`build.sh npu` 一次构建两个 AIV kernel、host transport 与 Python binding。性能入口为
`tests/run_remote_cpu_batch_put_perf.sh`，支持 `--transport urma|ubmem --executor aiv|aicpu`。

测试规模使用 `KVAF_NPU_REQUESTS`、`KVAF_NPU_TOKEN_BYTES`、
`KVAF_NPU_TOKENS_PER_BLOCK`、
`KVAF_NPU_CORE_COUNT`、`KVAF_NPU_JETTYS_PER_CORE`、`KVAF_NPU_WARMUP` 和
`KVAF_NPU_ITERATIONS`，由上述 shell 入口设置设备测试环境。

## NPU Runtime 性能脚本

`tests/python/test_npu_runtime_perf.py` 是独立的 steady-state 性能脚本，不纳入
`ctest`。它分别测量：固定 128-key batch 的 `BatchPut`、一次全量
`PrepareAccessTable`、一次全量 `access_table_view(device="npu")`、token 级随机请求
的 `BatchGet`、全量 `ReleaseAccessTable`，以及 128-key batch 的 `BatchFree`。
BatchPut 样本直接测同步调用到返回；BatchGet 异步样本直接测提交到 `drain_get()`
返回，不先重复外部同步。标签分别为 `synchronous_batch_put` 和 `async_submit_to_drain`；
历史分段名 `batch_get_synchronize` 现在表示 Drain 耗时。
随机源计划与数据填充不进入 BatchGet 提交计时。

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
