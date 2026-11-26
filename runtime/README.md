# KVAF Runtime

## 源码目录

公共 C++ 头文件继续放在 `include/`，包含方式与构建 target 不变。实现按职责组织：

- `src/`：`RuntimeCore`、`NpuRuntime` 入口和 NPU 配置，以及 CPU Pull executor、源注册、READ 控制服务与 NPU raw READ 实现。
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
独立 NPU Pull 管理 HBM 注册及连接，不要求创建 DDR 缓存池。

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
- `BatchCpuPull` / 显式 `PreparePut`：独立 CPU Pull executor 借用同一个 DDR 池，
  与普通 Put 共用写事务和恢复入口的互斥规则；不会占用 Get 队列。
- `NpuRuntime::Initialize()` 只初始化基础状态；`InitializeDdr()` 初始化缓存池。
  `ListenHbm` / `BatchGetRawAddr` 可在无 CPU、无 DDR 的 Runtime 上使用，
  正常 raw 异步输入和 stream 由应用保活并同步；关闭 source/runtime 前由应用销毁全部捕获图。

CPU Pull 的目的 MR 通过 `GetOffloadRegion()` 首次准备并独立授权；源注册和建链
显式进行。具体合同见 [CPU Pull](../docs/cpu_pull_offload_usage.md)、
[NPU raw READ](../docs/hbm_raw_addr_read.md) 与 [建链流程](../docs/connection_handshake_flows.md)。

普通缓存 Get 支持异步提交和显式 Drain；每个图节点使用独立租约，同一 Runtime 可在同一
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

统一构建默认使用 `RelWithDebInfo`，可通过 `--build-type` 覆盖。

CPU Gather 默认线程总预算为 12，等待模式为 `active`，内部绑核默认为 `on`。
`--gather-threads N` 和 `--gather-mode adaptive|active|passive` 适用于 Gather Read 和 AIV Gather→Push。
`--gather-affinity on|off` 控制这两条 AIV 路径的内部绑核。AICPU Gather Read 的 CPU 线程
遵循进程亲和性。纯 SGL Push 的执行路径不读取这三个 Gather 参数。

需要固定 NUMA 位置时，应在 UB device 所在 node 启动服务，例如
`numactl --cpunodebind=N --membind=N ...`。也可用 `taskset -c CPU_LIST` 限定允许的 CPU。
AIV Gather 和 AIV Gather→Push 开启内部绑核时，在这个允许范围内为 worker 选核。

服务启动时记录 CPU 亲和性。普通 URMA 缓存操作使用匹配的 `--cpunodebind N --membind N` 时，通过 Linux CPU topology 确定节点所属的 CPU Chip，只枚举该 Chip 上所有 FE 的非零 EID。
未绑定时枚举所有 UB 设备。候选按设备名和 EID index 排序。服务端 `--ub-device NAME` 选该设备首项，
加 `--eid-index N` 可精确固定。设备超出绑定范围或候选为空时返回错误。
例如 Node 0、1 同属于 Chip 0，绑定两者中任一节点得到相同的 Chip 0 候选表。部署设备 `udmac<C>d<D>e<E>` 的 `C` 对应 Linux `physical_package_id`。
其他设备可通过 Linux `numa_node` 确认所属 Chip。只提供固件邻近域时，通过 ACPI SRAT 和 CPU firmware UID 转换为 Linux 节点。
无法确定归属或属于其他 Chip 的设备记为 `NUMA_DEVICE_SKIPPED`。
CPU 端可同时保持多个 NPU session，NPU 正常
`shutdown` 后会释放对应 DDR、segment、token 与 Jetty。

### URMA DDR 大页配置

`--memory-page-mode` 是 `kvaf_cpu_service` 的启动参数，控制 URMA 分配的 CPU DDR。
`urma+ubmem` 模式下的共享缓存 DDR 由 UBmem 分配器提供。

| CPU 服务参数 | 分配方式 |
| --- | --- |
| 省略参数或 `--memory-page-mode thp` | 默认模式。按 2 MiB 对齐，用 `MADV_HUGEPAGE` 请求透明大页，并在注册前主动触页。实际合并量可查看 `/proc/<pid>/smaps` 中的 `AnonHugePages`。 |
| `--memory-page-mode default` | 按 4 KiB 对齐申请普通内存。内核仍可按系统策略使用 THP。 |
| `--memory-page-mode hugetlb-<size>` | 从所选大小的 Linux HugeTLB 页池分配。页大小须受宿主机内核支持。 |

以下以 **NUMA 1 上的 1 GiB HugeTLB** 为例。所有 shell 命令都在 CPU 服务所在的 Linux 宿主机执行。

#### 1. 确认 NUMA 节点和页大小

根据 CPU 侧 UB device 的拓扑选择节点。已知设备的 PCI BDF 时，可查看
`/sys/bus/pci/devices/<BDF>/numa_node`。用 `numactl --hardware` 查看节点和内存，
将示例中的 `node1` 改成实际节点对应的目录，再查看该节点支持的页池。

```bash
numactl --hardware
ls /sys/devices/system/node/node1/hugepages
```

下表列出当前回归测试覆盖的全部页大小。按目标节点实际存在的页池选择参数。
`<size>` 使用小写二进制单位 `k/m/g/t/p/e`，换算成字节后须是 2 的幂。

| 页大小 | 页池目录 | `--memory-page-mode` 的值 |
| --- | --- | --- |
| 64 KiB | `hugepages-64kB` | `hugetlb-64k` |
| 2 MiB | `hugepages-2048kB` | `hugetlb-2m` |
| 16 MiB | `hugepages-16384kB` | `hugetlb-16m` |
| 32 MiB | `hugepages-32768kB` | `hugetlb-32m` |
| 512 MiB | `hugepages-524288kB` | `hugetlb-512m` |
| 1 GiB | `hugepages-1048576kB` | `hugetlb-1g` |
| 16 GiB | `hugepages-16777216kB` | `hugetlb-16g` |

#### 2. 计算并预留大页

每块 DDR 所需的页数向上取整，再把页数加起来。例如，328 MiB 的 DDR 需要 1 个 1 GiB 大页。
使用 AICPU Gather 时，还要为它的 DDR 工作区预留大页。[工作区配置](../docs/aicpu_get_windows.md#配置)

以下命令以 root 身份执行。向 `nr_hugepages` 写入 `2`，表示将 NUMA 1 的 1 GiB 页池目标总数
设为 2 页，共 2 GiB。写入值低于现有页数时会缩减页池。

```bash
echo 2 > /sys/devices/system/node/node1/hugepages/hugepages-1048576kB/nr_hugepages
cat /sys/devices/system/node/node1/hugepages/hugepages-1048576kB/nr_hugepages
cat /sys/devices/system/node/node1/hugepages/hugepages-1048576kB/free_hugepages
```

回读 `nr_hugepages` 应为 `2`。如果写入成功但回读小于 `2`，说明未能预留到目标页数，通常是
节点上的连续物理内存不足。可在 Linux 启动时预留后再次检查。`free_hugepages` 是当前空闲页数，
应至少达到本次 DDR 申请所需的页数。
[Linux HugeTLB 页池说明](https://docs.kernel.org/admin-guide/mm/hugetlbpage.html)

#### 3. 启动 CPU 服务并核对分配

在仓库根目录启动服务，将 CPU 和内存绑定到预留大页的同一节点，并传入对应的页模式。

```bash
numactl --cpunodebind=1 --membind=1 \
  ./build-cpu/runtime/kvaf_cpu_service \
  --listen 0.0.0.0 --port 28999 --memory-page-mode hugetlb-1g
```

使用其他节点时，同时修改页池目录和 `numactl` 的节点编号。使用其他页大小时，同时修改页池目录、
所需页数和启动参数中的页模式。
NPU 客户端照常通过 `initialize(ddr_bytes, slot_size)` 或 `initialize_ddr(ddr_bytes, slot_size)`
指定缓存容量。客户端申请 DDR 时，CPU 服务才会从页池取页。检查 CPU 服务的 `DDR_ALLOCATED`
日志确认 `page_mode=hugetlb-1g`，上述示例的映射长度为 `backing_bytes=1073741824`（1 GiB）。
客户端可用容量和传给 URMA 的注册长度仍以申请值为准。UDMA/IOMMU 的页表粒度取决于驱动和硬件。
页池不足或内核不支持所选页大小时，显式 HugeTLB 分配会失败。

### NPU 连接方式

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

`link_to` 必须在 `initialize_ddr(ddr_bytes, slot_size)` 或带 DDR 参数的 `initialize` 前调用；
无参数 `initialize()` 只初始化基础设备，可用于无 DDR 的 NPU Pull。
显式调用 `initialize_ddr()` 前须先完成无参数 `initialize()`；带 DDR 参数的兼容入口会自动完成基础初始化。
普通缓存 URMA 通过候选数量查询的 TCP 对端地址判断场景。直连服务使用回环地址或本机网卡地址时判为本机，
其他地址判为远端。端口转发会改变可见对端。
本机自动选择 NPU 的 `pair_info.remote_eid`，缺失或通道不唯一时使用该物理卡首个有效 CLOS；
远端只使用 CLOS。显式 `npu_eids` 始终优先。本机且两侧均未固定 CPU 设备时，在已过滤的候选中匹配同一活动通道的
`pair_info.local_eid`，未命中取首项。显式 CPU 设备优先，远端 Put 和 AICPU 默认使用候选首项。
`PEER_LINK_EID_SELECTED` 日志记录两端 EID、场景、物理卡号、CPU 候选及选择原因。
普通 URMA 缓存与 CPU Pull 共用 CPU Service 按亲和性过滤的候选表，通过扩展候选查询取得完整 EID。部署时先更新 CPU Service。
初始化建立 CacheRegion 和元数据；具体 transport、MR、队列、Gather 工作区及 AICPU
包在首次使用或显式 warmup 时准备，初始化成功不代表数据面已通过 RMA 验证。
未建立 CPU 服务 Link 时，DDR 初始化返回 `InvalidState`。
`batch_get_token_bytes` 和 `batch_get_tokens_per_block` 可同时省略，用于独立 Pull；
普通缓存 Get 在 provider 首次准备时要求有效布局。显式布局必须同时为正数，
其他构造参数仅支持关键字。算法通过默认执行配置或本次 `execution` 指定。默认核数/worker 数
为 URMA AIV Put 2、Get 本机 2 / 远端 A+K 4、UBMEM AIV Put/Get 16、AICPU Put 1、AICPU Gather/Scatter Get 6。
URMA AIV Get 每核 Jetty 数默认为本机 2、远端 A+K 1，Put 每核默认 2 个。Get 在途上限默认 2048。显式资源项可覆盖各自默认值。
AIV/UBMEM 默认使用本次构建目录 `bin/` 下相应 Put/Get `.o` 的绝对路径，自定义
`./build.sh npu --build-dir DIR` 自动生效。`put_resources/get_resources` 可按配置覆盖预算
和路径；AICPU 仍从 CANN 安装目录定位设备包。不再提供旧全局后端参数或 adaptive 启动选优参数。

CPU Pull 每个 DDR region 绑定一个 CPU 端点。本机首次 `offload_region(source_physical_device_id=...)` 用指定源物理卡的 `pair_info.local_eid` 匹配候选，省略时采用 Runtime 所在卡。未命中或远端自动选择时取表中首项。单个 `peer_ub_device` 或服务端设备配置优先。逗号分隔的设备列表用于远端 AIV Direct 和 AICPU Gather BatchGet。

### 远端 A+K 的多 FE BatchGet

AIV Direct 和 AICPU Gather 共用 CPU FE 选择。远端默认选两个不同 UB device 的 `eid_index=0`；
`KVAF_DEFAULT_PEER_FE_COUNT` 可设置自动选择数量，实际数量受可用设备和通信 worker 数约束。
服务端固定设备或 index 时尊重其选择。

通过 `peer_ub_device="devC,devA,devB"` 或 `--peer-ub-device devC,devA,devB` 指定有序 FE 列表，
覆盖自动数量。多设备列表要求各设备有 index 0；单设备取其首个候选。
显式 FE 数不得超过 AIV 的总 Jetty worker 数或 Gather 的 worker 数。
本机链路和多个 NPU EID 的配置使用一个 CPU FE，同时指定多个 CPU FE 会报错。
Put 和 AICPU Scatter 使用列表首项。

建链时固定分配后端的通信资源。AIV 按 Jetty 分配 FE；Gather 按 `worker_id % FE数` 分配，
同一 worker 的控制链和数据链绑定同一个 FE，运行中不换 FE。
各 FE 注册同一块 DDR；Gather 额外 FE 共用一份 staging 和一组 CPU gather 线程。
首次使用或 warmup 时通过 `GET_FE_SELECTION` 核对数量，通过 `GATHER_ROUTE` 核对 Gather 的 worker 绑定。
AIV 还可通过 `DIRECT_AIV_ENTITY_TEMPLATE_READY` 的 `eid_path` 核对各 Jetty。部署时同步更新 NPU host 库
和 Direct Get kernel。多 FE 复用当前 CPU service 的 execution session 协议。

### 固定 UB device 后测量

启动带宽选优已移除。需要固定 CPU 侧 UB 设备时，直接在统一 Get 性能入口传入
`--peer-ub-device`，并在各组对照中保持相同设置：

```bash
KVAF_CPU_SERVICE_ADDR=127.0.0.1:21111 DEVICE_ID=2 \
bash tests/run_remote_cpu_batch_get_perf.sh build-npu \
  --backend aiv --peer-ub-device udmac0d1e2 \
  --request-length 32768 --table-capacity 4096 \
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
`DefaultAccessOperator` 和 `kvaf.NpuRuntime`。统一构建包含 AIV Gather 和 Scatter。
以下使用原有 Get 的统一性能入口测 URMA AIV Direct。

```bash
./build.sh npu
KVAF_CPU_SERVICE_ADDR=127.0.0.1:21111 DEVICE_ID=2 \
bash tests/run_remote_cpu_batch_get_perf.sh build-npu \
  --backend aiv --execution both \
  --request-length 32768 --table-capacity 4096 --miss-count 4096 \
  --token-bytes 656 --tokens-per-block 128 \
  --warmup 10 --iterations 100 --kernel-profile
```

CPU service 须预先启动。替换 `--backend` 及对应的 `--aiv-cores` / `--aicpu-workers` 可测 UBMEM、Gather、Scatter，
具体参数及计时范围见[统一 Get 性能测试](../docs/batch_get_performance.md)。
AIV Gather/Scatter 的独立测试入口和能力边界见 [AIV 路径说明](../docs/aiv_batch_get_stage1.md)。
普通 Get 的数据预检和输出校验随性能入口执行，位于性能计时外；Pull 功能和
Host 故障注入验证见 [诊断入口](../tests/diagnostics/README.md)。
默认关闭 kernel profiling；上例显式使用 `--kernel-profile` 开启设备打点与报告读取，
测无 profiling 的端到端性能时去掉该参数。`--multi-graph-check` 可在计时前验证
多节点报告隔离、自动收集/捕获、两条 stream 上由调用方串行执行的图与 eager 共用输出。

图对象由 Python 持有；Runtime 不创建或保存图。`kvaf.AclGraph` 直接封装
`aclmdlRICaptureBegin`、`aclmdlRICaptureEnd`、`aclmdlRIExecuteAsync` 和 `aclmdlRIDestroy`；
统一性能入口也支持 `--graph-api torch`。`runtime.graph_resources()` 通过一次真实预热的
`prepare(stream)` 自动收集各 Get，再用 `capture(stream)` 包住外部图捕获；模型继续
调用普通 `batch_get`。底层显式 begin/end_get_graph 接口仍保留。
单张图固定 stream，不同图可使用不同 stream；调用方保证图与普通 Get 不重叠使用共享资源，
KVAF 不代理原生 replay 或自动安排跨 stream event。回放前完成输入更新，计时只包含
replay 与 stream 等待。图销毁后 close 对应资源；所有图租约释放后才能关闭 Runtime。
图状态和 profile 查询始终通过 `graph_lease` 选择节点（可从 `resources.leases` 取得）；
省略租约查询 eager。profiling 在准备该实例图前开启，在全部租约结束后关闭。
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
全表按 `batch × 1024 × 78` 个 83,968B entry 分配，每次 BatchGet 32,768 requests。URMA AIV Get 在本机默认每卡 2 核、每核 2 个 Jetty，在远端 A+K 默认每卡 4 核、每核 1 个 Jetty，
Put 默认每卡 2 核、每核 2 个 Jetty。测试执行 3 次 warmup 与
100 个样本。可用
`KVAF_RUNTIME_PERF_BATCH`、
`KVAF_RUNTIME_PERF_AIV_CORES`、`KVAF_RUNTIME_PERF_WARMUP`、
`KVAF_RUNTIME_PERF_ITERATIONS`、`KVAF_RUNTIME_PERF_ACTIVE_PERCENT`、
`KVAF_RUNTIME_PERF_TOKEN_BYTES` 和 `KVAF_RUNTIME_PERF_TOKENS_PER_BLOCK` 覆盖；
Jetty 数量可通过
`KVAF_RUNTIME_PERF_JETTYS_PER_CORE` 覆盖。脚本通过 AIV Direct-HCCP BatchPut 填充全表，BatchPut 性能样本
测量 Runtime metadata upload、AIV WRITE 及其 caller-stream 完成，不包含全量数据
填充。
脚本以 CPU `torch.int64` Tensor 传入全量 Prepare key，避免 Python `list[int]` 转换污染
这项控制面性能数据。
