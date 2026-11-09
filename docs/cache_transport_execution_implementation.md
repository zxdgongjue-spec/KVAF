# 缓存访问执行框架：实现与使用

基线：`origin/main` 的 `1c13103`；分支：`codex/cache-transport-execution`。
设计依据：[缓存访问执行框架设计](cache_transport_execution_design.md)。
验收入口与命令：[验证步骤](cache_transport_execution_validation.md)。

## 实现范围

| 项目 | 实现 |
|---|---|
| 请求级执行配置 | C++ / Python 的完整配置替换、默认配置、enabled 集合、空集合关闭方向 |
| 普通 Put | URMA AIV、URMA AICPU、UBMEM AIV |
| 普通 Get | URMA AIV Direct、URMA AICPU Gather / Scatter、UBMEM AIV Direct |
| 混合池 | `urma+ubmem` 从初始化起分配同一份 Fabric DDR，URMA 会话借用并注册该 allocation |
| CPU 资源隔离 | 每个执行配置独立 context、token、MR、队列及 Gather / Push worker；Gather 与 Scatter 可同时启用 |
| 规范地址 | MetaManager / AccessTable 使用固定 allocation 基址；AIV WQE、AICPU transfer、CPU Push 提交时转换为注册基址 |
| UBMEM 映射 | Runtime 内按设备、Fabric handle 和映射长度共享；最后一个引用关闭，失败保留并阻止复用 |
| CPU HAL/SVM | 首次使用时加载，按库路径在进程内共享，跨池及 CPU Runtime 重建保留；池内存与映射仍各自释放 |
| 生命周期 | Put / Get 分别串行、跨方向独立；AccessTable 失效后仍由在途请求或 graph 保留槽位引用 |
| 资源管理 | 独立 `ExecutionResourceManager`，Put/Get 类型化租约与视图；provider 转交物理资源所有权，算子只借用 |
| 单次执行 | 独立 Put/Get `ExecutionContext`，持有请求副本、租约及 provider 执行状态，完成后归还 |
| 恢复协议 | 能力协商、服务端 epoch、幂等创建和控制 ID、查询、可重试关闭、显式 `recover` |
| Put 在线恢复 | 查询原次 WRITE 的完成结果；成功提交、确定失败回滚、未知继续保留，不重放数据搬运 |
| 资源预算 | 各配置独立 cores / Jetty / in-flight / 请求容量 / CPU Gather 线程数 / kernel 路径 |
| 预热 | 按容量建链、准备内部 buffer、Gather worker、Put metadata；可指定实际调用 stream |
| Graph | URMA / UBMEM AIV Direct Get 的显式租约，固定输出、提前准备、独占 Get 域、保留 Tensor 和表引用 |
| AIV Gather / Scatter | 仍按约定只预留配置及 provider 接口，启用时明确报未合入 |

当前已接入的普通访问组合为 3 种 Put × 4 种 Get。上述为实现状态，**不代表目标
硬件上的交叉组合、并发性能和跨 transport 可见性已验收**。AIV Gather / Scatter
合入后才能补齐设计中的 18 对组合。

CPU Pull / HBM Pull 的独立模块尚未合入本分支，按当前约定暂不接入，未额外引入参考分支。
启动带宽自动调优保留在旧入口；请求级入口按已选路由延迟建链，不执行初始化写入探测，
不自动切换后端或路由。

## Region、协议和回收

`CacheRegionService` 将 allocation 与 session 分开持有。PoolReady 不创建普通访问的
URMA context 或队列；URMA 路由发现只枚举候选。Gather 工作区在初始化时预留，始终
位于 MetaManager 的逻辑缓存范围之外。新注册不会清零或替换既有缓存数据。

新控制面使用命令 20、版本 1 的显式小端外层记录；其中封装的旧控制命令仍沿用
主线 v6 的 ABI。CPU 服务使用有界连接线程池，慢操作仅锁定对应记录。
同一 client / create-request ID 只能携带相同配置；session 控制命令也按 ID 去重。
回复丢失时客户端保留原 ID，恢复或清理先重发原操作，确认旧资源关闭后才允许新建。
服务端 epoch 改变时拒绝继续创建，避免重启后把旧请求误作新分配。

关闭顺序为本地执行停止、远端 session 停止和注销、最后释放 Region。CPU Push 的
STOP、队列删除、MR 注销、context 删除或 Fabric unmap 失败都会保留所有权。
Region 有 session 时拒绝释放。控制记录保留关闭墓碑；单 session 的控制历史上限
为 4096 条，到达上限后需要关闭并重建，不淘汰尚可重放的 ID。

CPU 的请求级与旧 UBMEM 分配入口统一通过 `AcquireProcessHal` 获取 HAL。
首次加载由互斥锁保护，成功后由进程级注册表持有强引用；加载失败不缓存，纯 URMA
分配不触发加载。注册表保留到进程退出，不参与 C++ 静态析构，避免其他资源尚在清理
时卸载 HAL。每个配置库路径只保留一个 HAL 对象，注册表不持有池或缓存数据。
`Mapping::Close` 仍逐池执行 unmap、释放虚拟地址和物理 allocation。

此生命周期用于避免已观察到的重建失败：卸载 HAL 后，用户态初始化标志被重置，
驱动侧 Host SVM 进程上下文仍存在；再次初始化的 OPEN ioctl 返回 `EEXIST`，
最终表现为 `halMemCreate ret=38`。已经进入此状态的旧服务需要退出并以修复版本
重新启动；仅更新二进制或重启 NPU 客户端无法恢复原 CPU 进程。

Put 不排空 Get。异步 Get 的 `drain_get()` 等待完整数据面并收集设备结果。未确认完成
的 Put 保留事务和输入引用，可通过 `recover(原 Put 配置)` 查询原次执行并完成事务；
若仍无法证明 DMA 完成，恢复继续失败并保留所有权。资源准备失败也可显式
`recover(execution)`，它不改变配置。
跨 transport 的可见性依赖已完成的 WRITE / MTE 和目标平台一致性契约，控制面 CPU
fence 不冒充数据面完成。

## 资源租约与执行上下文

`ExecutionResourceManager` 独立于 Dispatcher 保存两个方向的资源记录。每条记录拥有
`ExecutionResourceOwner`。管理器直接调用 owner 的 `Initialize`、`Warmup` 和 `Close`，
不再通过算子初始化或构造临时算子来关闭资源。兼容的旧调用者仍可通过
`AccessOperator` 持有独立 owner；请求级入口在初始化前通过 `TakeResources()`
将所有权转移给管理器。

已接入后端的资源对象与执行算法分为以下结构：

| 资源所有者 | 准备好的借用视图 | 无状态执行算法 |
|---|---|---|
| `AivExecutionResources`：Get/Put session、注册、队列和执行工作区 | `UrmaAivPutView` / `UrmaAivGetView` | `UrmaAivPutOperator` / `UrmaAivGetOperator` |
| `AicpuExecutionResources`：UMDK session、Gather/Push 服务及工作区 | `UrmaAicpuPutView` / `UrmaAicpuGatherView` / `UrmaAicpuScatterView` | 对应的 `UrmaAicpu*Operator` |
| `ubmem::UbmemResources`：共享映射引用、kernel 模块、每个 stream 的执行槽位 | `UbmemPutView` / `UbmemGetView` | `UbmemPutOperator` / `UbmemGetOperator` |
| `PeerExecutionResources`：CPU 执行 session、本地资源 owner 及适配门面 | 外层 `SessionAccessOperator` 转发到具体后端 | 不承担 kernel 执行 |

`DefaultAccessOperator`、`UmdkAccessOperator` 和 `ubmem::AccessOperator` 保留为公开
兼容适配层，内部不再有持有通信资源的 `Impl`。具体算法通过
`Launch(request, concrete_view, invocation)` 接口执行；映射、设备内存分配和本地
buffer 注册由资源对象的 Prepare 阶段完成。Launch 组织参数、提交 kernel 和解析结果，
不创建通信会话或注册应用 buffer。AICPU 的 transport helper 也区分 `Prepare` 与
`ExecutePrepared`；原 `Execute` 仅保留为兼容组合入口。

`AivInvocation`、`AicpuInvocation`、`UbmemInvocation` 保存本次请求状态，由
`ExecutionContext` 持有。资源 owner 借用当前绑定的 invocation，用于在异常关闭时
先等待在途工作；它不接管该 invocation 的所有权。旧入口另有显式 legacy 状态。

管理器按本 Runtime 的固定 Region、执行配置和不可变创建参数复用资源；设备、EID、
对端、ABI、资源预算及容量固定在创建器中，不跨 Runtime / Region 代次复用。
Dispatcher 先通过注册的适配层调用 `DescribeRequirements`，再把容量需求交给
Acquire；请求大小转换前检查溢出，容量超预算、溢出或未对齐的请求不会初始化通信资源。当前不可变
provider/ABI/设备配置在注册工厂中固定，Describe 声明本次调用需要的容量。
Put/Get 分别使用 `PutResourceLease` / `GetResourceLease`；外层方向视图负责兼容接口
转发，内部算法接收后端具体视图。具体 provider 校验资源 owner、已绑定调用状态及
准备阶段的输入，拒绝把其他调用的状态或未准备资源用于 Launch。

仅描述请求不会打开 CPU session。`PeerExecutionResources` 在 Initialize 时才创建并
发送执行 session 请求；未初始化 owner 的 Close 不会补发远端 CREATE。

首次 Acquire 只初始化该记录，其他方向可独立初始化。初始化失败会缓存失败状态，
必须显式恢复；清理失败不会替换旧 owner。已有租约时直接 Acquire 返回 Busy，正常
请求由 Dispatcher 的方向锁和 Drain 协调。预热归还租约后保留资源，未完成操作和
graph 则持续占有租约。

| 状态 | 持有位置 |
|---|---|
| 通信会话、队列、注册、工作区、物理 status/event 槽位、持久序号 | 资源 owner，跨调用复用 |
| 本次请求和 Put 地址描述副本、配置、stream、完成结果、pending / graph 状态 | ExecutionContext 及其 provider ExecutionState |
| Put 元数据事务、AccessTable 及槽位 pin | RuntimeCore，随执行完成或事务恢复释放 |
| Python Tensor / stream 强引用 | Python Runtime，安全完成或恢复前保留 |

普通 Get 完成后归还租约；Put 还需等待 Commit/Abort 和完成确认。关闭在状态仍绑定时
先调用 owner 清理，再解绑并销毁执行状态，最后归还租约和销毁资源。关闭失败保留
完整依赖，允许重试；AIV transport 的关闭错误也会向上传递，不再被 void 析构接口吞掉。

## Put 恢复语义

`runtime.recover(put_config)` 不重新提交 BatchPut。它只解析原次 AIV status、AICPU
WRITE 序号对应的结果或 UBMEM stream / status：

| 确认结果 | 元数据处理 | 调用结果 |
|---|---|---|
| Succeeded | CommitPut，确认完成后释放执行状态和输入引用 | 成功 |
| Failed，且确认不会再访问原 buffer | AbortPut，确认完成后释放执行状态和输入引用 | 返回原次失败 |
| Unknown | 保留原事务、buffer、注册、会话和租约 | CompletionUnknown 或底层完成检查错误，之后仍可恢复 |

必须使用原次 Put 的实际配置。元数据提交/回滚失败可以重试；元数据已经完成而确认
失败时，重试只做确认，不重复 Commit/Abort，也不再次提交 WRITE。Python 即使在
安全回滚后收到原次异常，也会释放已不再使用的输入引用。

AIV/AICPU 的远端 WR 错误或 CQ 超时不直接等同于安全失败，无法证明远端访问结束时
仍保留 Unknown；不能通过忽略错误、重建队列或重放 Put 来消除未决事务。没有未决
事务时，`recover` 才执行该配置资源的关闭和重建。`shutdown()` 仍可在确认数据面已
停止后回滚尚未提交的事务并清理整个 Runtime。

## Python 配置和预热

未使用新增配置时保留旧入口。启用请求级入口后，省略 enabled 表示设计的完整目标
集合；因此当前应显式裁剪掉未合入的 AIV Gather / Scatter。

```python
put = kvaf.PutExecutionConfig("urma", "aiv")
put_cpu = kvaf.PutExecutionConfig("urma", "aicpu")
put_mapped = kvaf.PutExecutionConfig("ubmem", "aiv")
direct = kvaf.GetExecutionConfig("urma", "aiv", "direct")
gather = kvaf.GetExecutionConfig("urma", "aicpu", "gather")
scatter = kvaf.GetExecutionConfig("urma", "aicpu", "scatter")
mapped = kvaf.GetExecutionConfig("ubmem", "aiv", "direct")

runtime = kvaf.NpuRuntime(
    batch_get_kernel_object_path="get_urma.o",
    batch_get_token_bytes=656,
    batch_get_tokens_per_block=128,
    batch_put_kernel_object_path="put_urma.o",
    transport="urma+ubmem",
    default_put=put, default_get=direct,
    enabled_put_configs=[put, put_cpu, put_mapped],
    enabled_get_configs=[direct, gather, scatter, mapped],
    put_resources=[
        (put, kvaf.ExecutionResources(cores=2, jettys=2)),
        (put_mapped, kvaf.ExecutionResources(cores=2, kernel_path="put_ubmem.o")),
    ],
    get_resources=[
        (direct, kvaf.ExecutionResources(cores=4, jettys=2, max_inflight_total=2048)),
        (gather, kvaf.ExecutionResources(cores=2, cpu_threads=8, request_capacity=4096)),
        (mapped, kvaf.ExecutionResources(cores=4, kernel_path="get_ubmem.o")),
    ],
)
runtime.link_to("cpu-host:port")
runtime.initialize(ddr_bytes, slot_size)
runtime.warmup(gather, request_capacity=4096, stream=read_stream)
runtime.batch_get(table, indices, offsets, miss_count, output_indices, output,
                  stream=read_stream, execution=gather)
runtime.batch_put(other_keys, source, input_offsets, lengths,
                  stream=write_stream, execution=put)
runtime.drain_get()
```

`cores=0` / `jettys=0` 继承旧构造参数；`max_inflight_total=None` 继承旧预算，显式 0
关闭限流。`request_capacity` 是该 provider 接受的请求上限，范围 1..65536；现有
Gather / Scatter kernel 的内部工作区仍按其固定最大 ABI 容量预留。
`cpu_threads=0` 使用 CPU 服务默认值。Jetty 数可调项用于 URMA AIV，in-flight 限流用于
URMA AIV Get，CPU 线程预算用于 AICPU Gather；其他后端不接受无效的调优字段。
标准构建的 Get/Put `.o` 各自包含 URMA 与 UBMEM 入口，同一方向可共用该文件。
上例中的独立文件名仅演示自定义构建时的 provider 路径覆盖。
预热不提交 key、不搬运应用数据，也不注册尚未提供的应用 buffer；普通 eager 调用
仍在完成后注销临时注册。C++ 非拥有 buffer 由调用方保持至完成。

## Direct Get Graph 租约

```python
lease = runtime.begin_get_graph(direct, output, request_capacity=capacity,
                                stream=read_stream)
# 在应用自己的 graph capture 作用域内，记录一次 KVAF Get：
runtime.batch_get(table, indices, offsets, miss_count, output_indices, output,
                  stream=read_stream, execution=direct, graph_lease=lease)
# 结束 capture，按应用协议 replay；确定以后不再 replay 后：
runtime.end_get_graph(lease, stream=last_replay_stream)
```

一个租约记录一次 KVAF Get，可随 graph 多次 replay。准备与捕获使用同一 stream，
输出地址及大小固定，请求容量不得超过预留值。捕获/重放期间不创建链路、注册输出、
分配设备状态或执行 Host Drain。租约期间 Get eager、Get 预热/恢复及 Runtime 关闭
返回 Busy；Put 仍可访问其他 key。

释放前调用方必须停止后续 replay，并将此前所有 replay 排序到所提供 stream 的
完成点。释放等待完成、检查最后一次结果，再注销输出并释放表/Tensor 引用。
失败保留租约。未 replay 的 graph 或已检查过的设备结果错误，可以显式
`discard_result=True` 放弃结果检查后回收；它仍等待 stream 完成，不跳过 DMA 安全检查。
Gather / Scatter 不声明 capture 能力；旧入口保留已有 graph 使用方式。

## 部署与验证

CPU 服务、Host 库及设备 kernel 必须一起重建：

- AIV Get 使用 `KvafExecuteBatchGetAivKernelV3`，Put 使用 `KvafExecuteBatchPutAivKernelV2`。
- AICPU 包要求 `KvafAicpuUmdkControlSessionsV3`，包含多会话协议及新的请求布局。
- 旧 CPU 服务不能用于新模式；能力协商会在分配缓存前拒绝不兼容服务。

便携测试覆盖配置、Python 构造、独立方向、显式恢复、graph 排他与表 pin、未决事务、
Region/session 状态机、epoch/重复 ID、截断协议、真实 TCP 回复丢失、AIV Get/Put kernel
模型在不同规范/注册基址下的 payload、AICPU transfer 地址转换、输入边界及共享映射关闭失败。
可通过 `KVAF_TEST_URMA_INCLUDE_DIR` 和 `KVAF_TEST_ACL_INCLUDE_DIR` 启用无需设备库的
设备注册表及 ACL 映射 mock 测试。CPU 服务以 `KVAF_CPU_ENABLE_URMA=OFF` 完成便携构建。
新增便携回归覆盖独立 owner、类型化租约、同资源并发首次获取、执行状态保留、
原次 Put 的完成恢复、元数据及确认失败重试和不重复提交。TCP 测试需要允许本机
回环监听；测试名称及命令见验证文档。

本机没有 Ascend 设备或正式 CANN 编译环境；NPU Host、AICPU 与 CPU URMA 源码使用
本地 SDK stub 进行语法检查。目标环境仍须完成正式 SDK 编译、新 kernel 部署、真实
payload 的交叉写读、Get/Put 重叠、混合可见性、graph replay 和故障退出验收。
