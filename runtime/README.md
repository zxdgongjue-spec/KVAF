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
- `BatchGet`：校验调用方的 index、mask、output 和（NPU 路径）Device
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

默认自动枚举所有可用 UMDK UB device/EID；`--ub-device NAME` 和
`--eid-index N` 仅用于诊断。CPU 端可同时保持多个 NPU session，NPU 正常
`shutdown` 后会释放对应 DDR、segment、token 与 Jetty。

NPU 应用只配置 CPU 服务地址，不配置 EID 或 UDMA device：

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

`link_to` 必须在 `initialize` 前调用。初始化会从 HCCL root-info 读取当前物理卡的
CLOS PG EID、追加 `/proc/asdrv_ub/pair_info` fallback，并组合 CPU UMDK 候选。只有
64B AIV WRITE 与 CQ 完成成功的 EID 对才会被选中；这同时验证 DDR 注册、segment
import、CTP、双侧 Jetty 和真实 RMA 写路径。`link_to` 是 NPU Runtime 初始化的前置条件；
未建立 CPU 服务 Link 时，`initialize` 会返回 `InvalidState`。
`batch_get_token_bytes` 和 `batch_get_tokens_per_block` 无默认值，
构造 `NpuRuntime` 时必须显式传入。

## CPU Runtime 访问 NPU HBM

当 CPU 需要直接读写 NPU HBM 时，NPU 侧运行 `kvaf_npu_service`，CPU 侧使用
`CpuRuntime::InitializeClient`。NPU 侧的命令行服务由 `NpuRuntime` 的 HBM provider
模式承载（`NpuRuntime(NpuServiceOptions)` + `InitializeHbmService()`）；两者复用同一套 `PeerLink` 控制协议：NPU service
负责 `aclrtMalloc` HBM、Direct-HCCP segment/token/Jetty 注册和 EID 选择；CPU Runtime
负责 UMDK 本地 buffer、CTP/Jetty 导入，并通过 URMA 发起 READ/WRITE。HBM 的
`urma_seg_t` descriptor 在 TCP 上原样传输，不依赖 `urma_admin`。

HBM 使用 `aclrtMalloc(ACL_MEM_MALLOC_HUGE_FIRST)` 优先分配大页；大页不足时
可能回退到普通页。Direct-HCCP 注册仍使用 non-pin device-memory 路径。

构建：

```bash
./build.sh npu
./build.sh cpu
```

启动 NPU HBM service（默认监听 `127.0.0.1:28999`）。不指定 `--device-id` 时，服务会在 CPU
第一次申请时按请求选择 NPU device，并按 device 和 EID 候选缓存 HBM context：

```bash
build-npu/runtime/kvaf_npu_service --listen 0.0.0.0 --port 28999
```

CPU 侧可通过 `CpuRuntime::InitializeClient` 建链并使用 `RuntimeCore::BatchPut` /
`BatchGet`。仓库中的 `runtime/kvaf_cpu_runtime_npu_hbm_e2e` 是最小验证程序：

```bash
build-cpu/runtime/kvaf_cpu_runtime_npu_hbm_e2e \
  --peer <npu-host>:28999 --bytes 4096 --slot-size 4096 --device-id 0 \
  --doorbell-wqes 32
```

CPU 初始化按 UB 设备名、EID index 顺序遍历本地候选，逐个尝试 NPU 的 EID
候选。NPU 候选与 AIV 共用发现逻辑：先取指定物理设备的 HCCL CLOS EID，再追加
去重后的 `pair_info` EID。每组建链后在四个 CPU Jetty 上各做一次最多 64B 的
WRITE 并等待 CQ；第一组成功即选用，失败则释放本次资源后继续尝试。日志
`LINK_ATTEMPT`、`LINK_FAILED`、`LINK_SELECTED` 记录所选路径，不进行带宽排名。
客户端的 `--ub-device`、`--eid-index` 只筛选 CPU 本地候选。
候选查询使用控制协议 v5，CPU 客户端与 NPU 服务端需要一起重新构建、重启。

逐条比较 EID 链路带宽，可在 CPU 侧运行独立测试（`./build.sh cpu` 会构建该程序）：

```bash
./build-cpu/tests/kvaf_cpu_urma_eid_perf \
  --peer 127.0.0.1:29888 --device-id 3 \
  --output results/cpu_urma_eid_bandwidth.csv
```

`--peer` 指向已启动的 NPU HBM service；按部署修改地址、端口和 NPU device ID。
测试遍历所有本地 UB 设备的非零 EID × 指定 NPU device 的全部候选 EID。
每次固定一个组合，建立四条独立 Jetty 通道，探测成功后执行完整 WRITE/READ
回读校验，再 warmup 和采样，最后再次回读校验。失败组合记录原因后继续下一组，
不会自动切换到另一条链路。每组结束先销毁 transport 再释放远端 HBM；若释放失败，
停止遍历并保留已收集的结果。

默认 `--count 8192 --token-bytes 4096 --remote-sgl-size 1 --warmup 5 --iterations 100`，
即每次 WRITE 32 MiB，8192 条 WR；本地和远端行地址均连续。所有组合使用相同负载，
固定 4 个 Jetty、4 个 OpenMP worker，`wqes_per_doorbell=0`。这些负载选项可按需修改。
测量范围为现有 `transport.total_ns`（包含分组、参数检查、segment 查找、WR 构造、
提交和完成等待），不包含建链、预注册内存、回读校验和 warmup，也不经过 Runtime/Python。
`mean_GBps = payload_bytes / mean_ns`，使用十进制 GB/s。

CSV 每个 EID 组合一行，写完立即 flush；包括两端设备/EID index/EID、导出的 segment EID、
状态及错误、负载参数、实际 WR 数、平均/p50/p99 延迟、平均/p50 带宽和 prepare/post/
completion wait 平均耗时。`remote_sgl_size` 列是请求值，实际分组仍按设备能力限制，
可结合 `wr_count_min/max` 和 transport 日志确认。失败行的性能字段留空。
终端最后输出成功组合按平均带宽降序排列的 `EID_PERF_RANK`，以及包含 CSV 路径的
`EID_PERF_SUMMARY`。无成功组合或资源清理失败时返回非零退出码。

绑核对比可使用下面的测试入口，负载和 CSV 格式与上述程序一致：

```bash
tests/run_cpu_urma_eid_perf_pinned.sh "$PWD/build-cpu" \
  --peer 127.0.0.1:29888 --device-id 3 \
  --count 8192 --token-bytes 4096 --remote-sgl-size 1 \
  --output results/cpu_urma_eid_bandwidth_pinned.csv
```

脚本在 Linux 上从调用进程允许使用的 CPU 中选四个不同物理核，优先选同一
NUMA 节点，使用 OpenMP affinity 将四个 worker 分别固定到一个逻辑 CPU。
每个 worker 仍独占一条 Jetty，负责其 WR 构造和 post；完成轮询仍由调用线程执行。
`EID_PERF_AFFINITY` 输出目标 CPU/NUMA 节点，OpenMP 的 affinity 日志显示实际线程绑定。
此方式没有在每批传输中增加设置/恢复 affinity 的系统调用。
脚本不自动判断 UB 设备所在的 NUMA 节点；需要指定核时，在命令前加
`taskset -c 8,10,12,14`（按实际拓扑修改），脚本会在该集合内选核。
绑核仅对本次启动的测试进程生效，未绑核基线可直接运行原测试程序并使用另一份 CSV。

由于 `951ce25` 在设备上出现 SIGBUS 回归，当前保留回退后的原始 WR 构造路径，
单独扩展为 4 Jetty/worker，以便验证并发提交。`./build.sh cpu` 同时清空 `CMAKE_BUILD_TYPE`，避免旧构建目录
继续保留该提交设置的 Release 模式。EID 测试保留阶段日志，用于进一步定位崩溃。

CPU URMA 固定创建 4 个 Jetty，并使用 4 个 OpenMP worker，每条 WR 的本地
SGE 数固定为 1。常量位于 `cpu_urma_transport.cc`：`kJettyCount`、
`kOmpThreadCount`、`kLocalSgeCount`。不再提供 `jetty_count`、`omp_thread_count`、
`sgl_size` 初始化参数或对应命令行选项。

每个 CPU Jetty 的 `CreateChannel` 请求都会在 NPU 服务端创建独立的目标
Jetty 和 CQ；四条通道共享 HBM segment、token 和设备 context。服务端
`CHANNEL_READY` 日志中的 `channel_index`、`target_jetty_id` 和
`target_jfc_id` 可用于确认通道对应关系。释放 HBM session 时先释放各通道，
再注销并释放 HBM。此行为需要重新构建并重启 `kvaf_npu_service`。

每个 worker 独占一个 Jetty 的 WR 链表、SQ 和提交操作，构造好一批就直接提交，
不需要等待另一个 Jetty 的构造完成。`wqes_per_doorbell > 0` 控制每批 WR 数，
为 0（默认）时每个 worker 构造完整 Jetty 链表后提交；两种方式都由 worker
自行下发。每个 Jetty 仅最后一个 WR 请求 CQE，worker 汇合后由调用线程轮询完成。
WRITE 的 `place_order`、`comp_order` 均为 0，用于与独立测试对照；READ 保留
末尾 strong-order 和完成顺序设置。CPU JFS/JFR 深度为 8192，JFC 深度为 16384；
NPU 每条目标通道的 SQ/RQ 深度为 1024，CQ 深度为 16384。

`CpuRuntime.initialize(..., remote_sgl_size=N)` 控制 WRITE 的远端 SGL 分组。
默认 1 是单条 WR 对应一条数据的基线。大于 1 时，一个连续本地 DDR SGE 可以
写入多个指定 HBM 地址，不增加 host memcpy；本地地址不连续、达到有效远端 SGE
数量或累计长度超过 uint32 上限时拆分 WR。READ 保持每条 WR 一个本地和一个远端
SGE。已注册的完整 CPU 输入复用原 segment；否则查找或注册覆盖整个源区间的 segment。

不设固定的 31 上限。请求值为正 uint32，实际每条 WR 的远端 SGE 数上限取请求值、
设备 `max_jfs_rsge` 和 Jetty ABI 字段容量中的较小值。例如请求 128、设备报告 31，
实际按最多 31 个分组；日志同时打印请求值和生效值。6000 条连续本地行在生效值为 1
时生成 6000 个 WQE，为 31 时生成 194 个（累计长度未超限）。

基线命令：

```bash
tests/run_cpu_batch_put_perf.sh /absolute/path/to/build-cpu \
  --request-length 8192 --miss-count 6000 --token-bytes 576 \
  --remote-sgl-size 1 --warmup 50 --iterations 500
```

对比测试仅修改 `--remote-sgl-size`。脚本先做 BatchPut/BatchGet 数据校验，再记录
实际 `wr_count` 和端到端耗时。`prepare_ns`、`post_ns` 取构造和提交合计耗时最长
worker 的两项时间，不把并行 worker 的时间相加；`poll_ns` 为汇合后完成等待时间，
`total_ns` 为包含注册、校验、线程调度和所有容量批次的 transport 总墙钟时间。

Python HBM 集成测试与性能脚本支持 `KVAF_CPU_URMA_REMOTE_SGL_SIZE` 和
`KVAF_CPU_URMA_DOORBELL_WQES`。设备上的实际 WRITE 提交及回读校验用于确认
provider 的多远端 SGE 行为。

CPU URMA transport/operator 的源码归属 `access_operator/include/cpu_urma` 和
`access_operator/src/cpu_urma`；Runtime 仅负责 HBM region、MetaManager 和 operator
注入。CPU BatchPut 的 `CPU DRAM -> HBM` 数据流对应 AIV BatchGet，CPU BatchGet 的
`HBM -> CPU DRAM` 数据流对应 AIV BatchPut。可用下面的脚本按与 AIV BatchGet
相同的 token-bytes/miss-count 组合测试 CPU BatchPut。测试始终传入
`request_length` 容量的 keys、offsets、lengths 和 packed input，并由显式
`miss_count` 只执行紧凑前缀 `[0, miss_count)`；未激活的尾部不会分配 slot 或生成
URMA WQE：

```bash
mkdir -p "$PWD/results/cpu_batch_put_sgl8"

KVAF_CPU_BATCH_PUT_SWEEP_LOG_DIR="$PWD/results/cpu_batch_put_sgl8" \
tests/run_cpu_batch_put_miss_count_perf.sh "$PWD/build-cpu" \
  --peer 127.0.0.1:29888 --device-id 3 \
  --request-length 8192 --doorbell-wqes 32 \
  --remote-sgl-size 8 --iterations 1000 --warmup 5
```

脚本默认扫描 `token_bytes={128,256,512,656,1024,2048,4096}` 和
`miss_count={32,64,128,256,512,1024,2048,4096,8192}`，共 63 种组合；因此
`request_length` 至少要设为 8192。生成两份 CSV，每份均包含 63 行数据和表头
`token_bytes,miss_count,mean_us,p50_us,p90_us,p99_us,p50_GBps`：

- `cpu_batch_put_sweep_summary.csv`：完整 `runtime.batch_put()` 调用的延迟和带宽，
  可用 `KVAF_CPU_BATCH_PUT_SWEEP_CSV` 覆盖路径。
- `cpu_batch_put_transport_total.csv`：CPU URMA transport 的延迟和带宽，
  可用 `KVAF_CPU_BATCH_PUT_COMPARISON_CSV` 覆盖路径。

两份文件各自使用对应操作的原始纳秒分位数计算
`p50_GBps = token_bytes × miss_count / p50_ns`（十进制 GB/s），避免从 CSV 的三位
小数微秒值反算造成误差。两个路径必须不同。每个组合的原始日志保存在上述目录，
便于排查失败。

`transport total` 从取得 transport 提交锁后开始，直到整个请求的最后一个完成通知
被确认；包含 transport 内的参数校验、工作区准备、segment 查找/回退注册、WR/SGE
准备、提交、轮询和超过队列容量时各批次之间的间隙。不包含 Python binding、Runtime
元数据处理、operator 组装/预注册和取得提交锁之前的等待。它是软件观测的 transport
有效带宽，不能等同于硬件纯链路带宽。`prepare/post/poll` 保留为分段诊断，三者之和
不再冒充完整 `total_ns`。

原始日志中 `batch_put` 和 `batch_put_transport_total` 各自的 `p50_GBps` 分别采用
各自的 p50 延迟；`useful_GBps` 使用各自的平均延迟。均值和分位数均先对每次完整调用
取样，再排除预热进行统计，分位数采用排序后的线性插值。

更新此次计时实现后，需要重新编译 CPU Python binding（例如
`cmake --build build-cpu -j`），测试会拒绝缺少 `total_scope=transport_call` 的旧 binding。
请使用新输出目录重新测量；旧版 `cpu_batch_put_sweep_summary.csv` 的 `p50_GBps` 也是
transport 口径，与旧 `cpu_batch_put_transport_total.csv` 相同，并非端到端带宽。
历史文件不会被自动删除，旧的分段计时也无法补算成新的完整 transport 耗时。

`--eid-index` 仅在需要固定候选时指定；默认由 CPU UMDK 和 NPU `pair_info`/RA EID
列表自动选择。NPU service 的 region/session 表支持同时存在多块 HBM；每块 region
独立注册 segment、Jetty 和 token，并按 session 释放。

现有 `tests/run_remote_cpu_e2e.sh` 与
`tests/run_remote_cpu_runtime_perf.sh` 接受 `KVAF_CPU_SERVICE_ADDR`，但该变量
只由测试脚本转换为上述 `link_to` 调用，不再是 Runtime 内部配置来源。

## 资源所有权

Runtime 持有且负责释放：内部 DDR、AccessTable 状态和 Key 稳定性锁。

调用方持有且必须在其 BatchGet 所在 stream 完成前保持有效：BatchPut 的连续 `input` buffer、
与 Key 一一对应的 Host `input_offsets` / `lengths` 元数据、BatchGet 的
index/offset/mask/output-index/output buffer，以及 NPU 上的 address/valid 表。Runtime
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
或无效 entry。AIV kernel 对 `mask=1 && valid=0` 返回设备侧 `kInvalidLookup`，
且不会对该地址发起 UDMA 读；padding 应由 Adapter 同时以 `mask=0` 跳过。
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
teardown。它以 `KVAF_NPU_ACTIVE_PERCENT` 构造稀疏 mask（例如 `5` 表示 5% 请求
有效），只为 `mask=1` 的请求写入 KV；随后可用 Runtime 的 ACL D2H mock
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
`offsets`、`mask`、`output_indices`；capture 中的 BatchGet 参数和所有 NPU tensor 地址不变。正常
性能口径是 `graph_replay_to_stream_completion`：四项 H2D 更新会在 replay 前完成并同步，
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

每个 BatchGet 样本先生成 `source_token_ids`、随机输出位置和每 2048-request
分组的 `miss_count`。调用前再派生
`indices = source_token_ids // tokens_per_block`、
`offsets = (source_token_ids % tokens_per_block) * token_bytes`、`output_indices` 和
mask。输入计划与
BatchGet Tensor 构造都在计时窗口外完成并同步，不输出输入构造阶段耗时。
脚本仅输出 submit、synchronize 和 BatchGet 的 `PERF_RESULT`。

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

### AIV BatchGet HBM Allocation Comparison

For a focused benchmark with three terminal summary lines, use the independent
test below. It populates only the requested CPU region once and measures only
BatchGet through device completion. Every request is active (no miss-count
grouping). Preparation, preflight, warmup, verification and teardown are outside
the measured samples. The complete native output and failure traceback are
saved to the temporary log named in the PASS/FAIL line.

```bash
# CPU host: keep this service running.
./build-cpu/runtime/kvaf_cpu_service --listen 0.0.0.0 --port 28999

# NPU host: use the same build for all output allocators.
for allocator in torch acl-normal acl-huge; do
  python3 tests/python/aiv_batch_get_bw.py \
    --peer CPU_HOST:28999 --build-dir build-npu \
    --allocator "$allocator" --blocks 2048 --block-bytes 4096 \
    --cores 32 --jettys-per-core 2 --warmup 20 --iterations 200
done
```

`--blocks 1024 --block-bytes 4096` transfers exactly 4 MiB;
`--blocks 2048 --block-bytes 4096` transfers exactly 8 MiB. For the same 8 MiB
with fewer WRs, use `--blocks 128 --block-bytes 65536`. The default address
pattern is sequential; `--pattern random` uses fixed-seed permutations of
both source and destination rows. Each iteration reuses that request plan.

`KVAF_RUNTIME_PERF_HBM_ALLOCATOR=torch` (default) uses the caller's
`torch.empty` output. `acl-normal` uses an owned `NormalHbmBuffer` allocated
with `aclrtMalloc(ACL_MEM_MALLOC_NORMAL_ONLY)`, matching the previous NpuService policy and
bypassing the Torch allocator. The raw buffer is accepted only as BatchGet
output; it is not a Torch Tensor. Allocation and CPU readback validation are
outside the timed loop. The benchmark releases the buffer after Runtime
shutdown, because Runtime caches its Direct-HCCP registration.

The focused `aiv_batch_get_bw.py` test also accepts `--allocator acl-huge`.
It uses the same raw buffer and `aclrtMalloc` call as `acl-normal`, changing
only the policy to `ACL_MEM_MALLOC_HUGE_FIRST`. Allocation size, registration
and timing are unchanged; it does not use the Torch memory pool. HUGE_FIRST
prefers huge pages but may fall back to normal pages, so the mode name does
not prove the actual page size. Rebuild the NPU binding before using it.

After rebuilding `build-npu`, set the same settings for both runs:

```bash
export KVAF_CPU_SERVICE_ADDR=CPU_HOST:28999
export DEVICE_ID=0
export KVAF_RUNTIME_PERF_REQUESTS=2048
export KVAF_RUNTIME_PERF_TOKEN_BYTES=4096
export KVAF_RUNTIME_PERF_TOKENS_PER_BLOCK=1
export KVAF_RUNTIME_PERF_TABLE_CAPACITY=32768
export KVAF_RUNTIME_PERF_ACTIVE_PERCENT=100
export KVAF_RUNTIME_PERF_WARMUP=20
export KVAF_RUNTIME_PERF_ITERATIONS=200

KVAF_RUNTIME_PERF_HBM_ALLOCATOR=torch \
  tests/run_remote_cpu_runtime_perf.sh "$PWD/build-npu"
KVAF_RUNTIME_PERF_HBM_ALLOCATOR=acl-normal \
  tests/run_remote_cpu_runtime_perf.sh "$PWD/build-npu"
```

Both modes allocate an 8 MiB output and populate 128 MiB of remote CPU memory.
The existing miss-count contract allows at most 2047 active requests per
2048-request group, so the actual payload is 8 MiB minus 4 KiB. Compare the
`operation=batch_get` result, whose timing includes stream completion, and
check `RUNTIME_PERF_CORRECTNESS_PASS phase=batch_get`.
