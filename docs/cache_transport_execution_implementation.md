# 缓存访问执行框架：实现与使用

本文描述 `codex/cache-transport-execution` 的 `5cce05e7`：算法首次使用准备、
统一执行路径、同 stream 多图及多个 Get 节点均已接入；实现状态不代表真机或性能验收通过。
初始开发基线为
`1c13103`，本分支已合入 main 的 `de4fd37e02fa86357f71fe4fcbbbecc3e75d1d58`。
设计与后续边界见[框架设计](cache_transport_execution_design.md)；构建、性能入口及
验证覆盖边界集中在[验证步骤](cache_transport_execution_validation.md)。

## 当前范围

| 能力 | 当前实现 |
|---|---|
| Put | URMA AIV、URMA AICPU、UBMEM AIV；同步调用及元数据事务 |
| Get | URMA AIV Direct、UBMEM AIV Direct、URMA AICPU Gather、URMA AICPU Scatter |
| 执行配置 | 默认配置、逐次完整配置替换、按构建支持自动可用、首次使用校验资源预算 |
| 资源复用 | 按 Runtime / Region / 执行配置延迟建立 provider；Put/Get 执行状态按配置驻留 |
| 输出绑定 | URMA AIV Direct、AICPU Gather 显式保留输出注册，每个 Get 实例各一个 |
| 输入绑定 | URMA AIV、AICPU Put 显式保留输入注册；整个 Put 方向同时只能有一个绑定 |
| 混合池 | `urma+ubmem` 初始化时分配同一份 Fabric DDR，URMA 注册与 UBMEM 映射使用同一 allocation |
| 并发 | Put/Get 两方向独立；同方向串行，多个实例驻留不增加同方向并发请求数 |
| Graph | 四种 Get 支持预热自动收集节点及底层显式租约；不同图可用不同 stream，与普通 Get 由调用方串行调度 |
| 恢复 | 资源重建、幂等控制协议、原次 Put 完成查询及事务恢复、失败保留资源以供重试 |
| 诊断 | Get/Put Host 分段计时、四种 Get kernel profile、AICPU 显式结果检查 |

普通访问组合为 3 种 Put × 4 种 Get。AIV Gather/Scatter 仍只预留配置和 provider 接口，
首次请求时明确报 provider 不可用；独立 CPU Pull/HBM Pull 模块按约定暂不接入。
图内混合算法不在本轮验收范围；Put 不进入图，也不支持跨 stream 并发回放。

所有请求统一通过 Dispatcher 按本次配置或默认配置选择驻留实例，不会每次重新创建
通信资源。旧构造参数与执行模式开关已移除；初始化只建立缓存与元数据，首次使用
或 warmup 按已选路由准备数据面，不执行启动探测写入、自动测速或切换路由。

历史便携测试结果不等于当前版本的硬件验收。真实 DMA、混合可见性、图回放和性能结果以目标
Ascend 环境的验证为准。

## 目录与调用结构

生产实现按算法、provider 与 transport 分开，原 `demo` 目录不再承担生产代码入口：

```text
bindings/                         Python Tensor、stream、引用保活与 ACL Graph 适配
common/include/                   请求、配置、地址描述、ABI 和计时结构
meta_manager/                     key/slot 映射、稳定读、Put 元数据事务
runtime/
  src/                            NpuRuntime 组装、RuntimeCore 业务语义
  execution/                      Dispatcher、ResourceManager、配置解析
  memory/                         Region/allocation、CPU DDR 注册及共享池
  peer/                           控制协议与 PeerLink
  cpu_service/                    CPU Runtime、URMA 后端、Gather/Push worker
access_operator/
  include/                        公共资源/调用/view 契约
  providers/{urma_aiv,urma_aicpu,ubmem_aiv,common}/
                                  资源 owner、兼容适配器、具体调用状态
  transports/{direct_hccp,aicpu_urma,ubmem}/
                                  传输会话、动态库、注册/导入、映射
  algorithms/
    get/direct/{urma_aiv,ubmem_aiv,common}/
    get/gather/urma_aicpu/
    get/scatter/urma_aicpu/
    get/common/aicpu/
    put/{urma_aiv,urma_aicpu,ubmem_aiv,common}/
                                  算法 Host 提交、kernel、ABI 与构建入口
```

`NpuRuntime` 组装 allocator、MetaManager 和 RuntimeCore，统一使用
`ExecutionDispatcher`。Dispatcher 通过
`ExecutionResourceManager` 取得所选配置的 owner，再由兼容适配器准备具体 view，调用
`Launch(request, concrete_view, invocation)`。

RuntimeCore 管理 DDR 生命周期、AccessTable pin 和 Put 事务；Dispatcher 管理执行选择、
方向占用、绑定与完成；owner 管理物理资源；算法消费准备好的资源。`PeerLink` 负责
创建/查询/关闭远端执行 session 等控制 RPC，实际 payload 由 URMA 或 UBMEM 数据面搬运。

## 所有权、驻留与执行占用

| 对象 | 责任与生命周期 |
|---|---|
| `RuntimeCore` | Put 元数据事务、Host AccessTable 和槽位 pin；完成/恢复前保持依赖 |
| `ExecutionResourceManager` | 两方向的配置记录、provider 和资源 owner；管理初始化、失败、恢复和关闭 |
| `PutResourceReference` / `GetResourceReference` | 长期固定对应 owner 和绑定状态，不占用执行；随驻留实例保留 |
| `PutResourceLease` / `GetResourceLease` | 普通调用的独占执行占用；Put 保留至事务及确认成功，普通 Get 保留至安全完成 |
| `GetExecutionContext` | 每 Get 配置一个驻留实例；含 provider 状态、预热容量/stream、输出绑定和当前请求字段 |
| `GetGraphNode` | 每图租约独立保存节点状态和完成报告，固定驻留实例资源地址；与普通 Get 串行共存 |
| `PutExecutionContext` | 每 Put 配置一个驻留实例；含长期资源引用、provider 状态和准备缓存；短期租约及完成记录按次保留到事务确认 |
| Python Runtime | 持有在途、输入/输出绑定和 graph 所需 Tensor/stream 强引用 |

Get 首次调用或显式预热时创建实例，固定资源引用并创建/绑定 provider `ExecutionState`。
同一 stream、请求容量未增长的调用复用准备结果，跳过重复 Describe/Warmup 和状态构造；
切换 stream 或增容会重新检查并准备。每次仍验证请求及绑定边界，取得轻量执行租约，
更新当前 request/table/options。同步调用成功或 `drain_get()` 成功后清除这些借用字段，
释放执行占用，保留实例及可复用资源。恢复/关闭失败保留原实例，不能用新实例覆盖它。

普通 Put 同样按配置驻留。首次使用或 Warmup 通过 `PinPut` 固定资源，创建并绑定
provider 状态；同 stream、请求容量未增长时复用准备缓存，增容或换 stream 重新准备。
每次仍校验请求/输入绑定并取得短期租约，不能把缓存命中解释为跳过边界检查。

`PreparePut(request, options)` 保存准备好的实例、请求身份和 stream；Runtime 预留元数据后，
`ExecuteBatchPut` 消费同一实例及租约，避免再次 Describe/Warmup、创建或绑定状态。
直接调用 Dispatcher Execute 时仍支持内部准备。`AcknowledgePut` 成功后清本次完成信息，
归还租约并保留实例。准备取消入口 `CancelPreparedPut` 只释放尚未提交的准备占用，
不清除已提交的 pending Put；它供 Runtime 的准备保护对象在元数据拒绝或提交前异常时调用，
同步 Mock 或仅实现默认接口的 AccessOperator 可保持无操作；这不代表另一条 NPU 执行路径，
也不新增 Python 调用要求。

请求向量和 `PutAccessDesc` 仅在同步 Launch 期间借用，不复制进 Context，也不供恢复流程
访问；provider 自己暂存异步使用的 metadata。显式输入绑定挂在同一实例上，解绑后实例
仍驻留。普通 Put 驻留不自动长期保留输入 MR，也不实现跨算法的 context、HBM 注册或
远端导入共享。

输出注册不会因 Get 实例驻留而自动保留。需要长期复用同一输出时显式调用
`begin_get_output`；未绑定的普通请求仍按相应 provider 的临时注册规则清理。
“首次看到输出便自动长期缓存”的策略尚未实现。HBM 输入/输出 allocation 始终属于
调用方；C++ 使用非拥有 buffer view，Python 绑定负责延长必要引用，但调用方仍不能
在 DMA 或绑定存续期间 resize、替换或复用同一存储。

### 执行完成时具体释放什么

资源租约、Host 表引用和 Python 对象引用分别保护不同的依赖，不能相互替代：

| 持有项 | 保护范围 | 完成后的变化 |
|---|---|---|
| 执行租约 | 当前调用使用的队列、工作区和调用状态不被提前关闭或复用 | 归还独占占用；驻留实例的长期 `ResourceReference`、`ExecutionState` 和准备缓存继续保留 |
| Host `AccessTableState` 的共享引用 | 地址/长度/有效位，以及 `LockContext` 固定的缓存记录和 DDR slot | 去掉本次请求引用；仅在最后一个公开句柄、在途请求或 graph 引用退出时才 `UnlockKeys` 并销毁 Host 表 |
| Python Tensor/stream 强引用 | 设备地址表、索引、offset、miss_count、输出行号、输出和 stream 等对象 | 去掉 KVAF 的额外引用；不复制 Tensor 数据，也不强制释放调用方仍持有的对象 |

`release_access_table()` 只使公开句柄失效。已经提交的 Get 或 graph 仍可持有其表；
`UnlockKeys` 解除槽位 pin，不自动删除 key 或释放整个 DDR 池，后续 `batch_free()`
才可回收相应记录。直接传入裸 `addresses/valid` Tensor 时没有 Host 表引用，Python
对象保活只保护这些 Tensor，不能保护其中地址所指向的 CPU DDR 数据。

成功同步 Get 返回时清除本次执行引用；异步 Get 在成功 `drain_get()` 后清除。
下一次普通 Get 也会先 Drain 前一次调用。单独同步 stream 只等待设备，不能替代框架
清理。完成未确认或清理失败时仍保留必要引用，不能在异常返回时无条件释放。

跨调用绑定另有结束边界：`end_get_output()` 成功后解除相应输出注册和绑定的 Python
引用，`end_put_input()` 成功后解除输入绑定；两者都保留算法实例。当前 `end_get_output()`
不统一清空 Get 方向的表和在途 Python 引用，因为它可能只结束另一个空闲算法的绑定；
这些引用由下一次成功的方向 Drain 或安全 Shutdown 清理。
图持有的表和 Tensor 引用则一直保留到 `end_get_graph()` 成功，不随单次 replay 完成释放。

C++ 的非拥有 buffer view 不提供 Python 层的对象保活，调用方仍须维持底层存储。
Python 强引用也不阻止调用方主动 resize、替换 storage 或破坏跨 stream 顺序。
若省略上述保护，需把这些责任明确转交调用方；若完成后永不归还引用，则会延长 Tensor
占用或槽位 pin，影响内存回收和后续调用。相关实现见
[`GetExecutionContext::ClearRequest`](../runtime/include/execution_context.h)、
[`RuntimeCore`](../runtime/src/runtime.cc) 和
[`NpuPythonRuntime`](../bindings/src/npu_python_runtime.cc)。

### 资源 owner 与算法 view

| 资源所有者 | 借用视图 | 无状态执行算法 |
|---|---|---|
| `AivExecutionResources` | `UrmaAivPutView` / `UrmaAivGetView` | `UrmaAivPutOperator` / `UrmaAivGetOperator` |
| `AicpuExecutionResources` | `UrmaAicpuPutView` / `UrmaAicpuGatherView` / `UrmaAicpuScatterView` | 对应 `UrmaAicpu*Operator` |
| `ubmem::UbmemResources` | `UbmemPutView` / `UbmemGetView` | `UbmemPutOperator` / `UbmemGetOperator` |
| `PeerExecutionResources` | 外层 `SessionAccessOperator` 转发至具体后端 | 持有远端执行 session 与本地 owner，不执行 kernel |

`DefaultAccessOperator`、`UmdkAccessOperator`、`ubmem::AccessOperator` 保留为兼容适配层。
请求级入口通过 `TakeResources()` 转移物理资源所有权；管理器直接调用 owner 的
Initialize/Warmup/Close。外层 `PutResourceView/GetResourceView` 是适配接口，内部算法
使用后端类型化 view。`PreparePutResources/PrepareGetResources` 准备映射、工作区和
注册；Launch 组织参数、提交和处理结果，不创建通信会话或注册应用 buffer。
`MetaManager::PreparePut` 则始终表示元数据事务准备。

`AivInvocation/AicpuInvocation/UbmemInvocation` 是 provider 调用状态。owner 借用当前
绑定状态，以便关闭时先等待在途工作。旧执行模式分支已删除，但仍保留初始化前或解绑后
使用的默认状态。持久 SQ/CQ 位置、通信序号、队列和共享工作区不能随着清除一次请求而重置；
graph 节点另有独立调用状态及完成报告，避免同算法多个节点相互覆盖报告。

`ExecutionViewCommon` 包含 Region、方向和资源/绑定/准备代次，无分配、注册或关闭接口。
owner 重建、重新绑定状态、相关工作区重新准备会使旧 view 失效；独立 Put/Get 工作区
分别更新准备代次。view 的校验不能代替
生命周期管理，借用 view 必须由仍存活的资源引用和执行租约保护。

## 内存绑定与共享边界

MetaManager 的固定 slot 映射和公共 AccessTable 使用 canonical 地址。通信注册范围可含
对齐和辅助工作区，不能拿注册长度当逻辑缓存容量。提交地址仍为：

```text
registered_address = registered_base + (canonical_address - canonical_base)
```

`UrmaRemoteMemoryDesc` 只保存注册地址、长度、raw token、EID 和注册 key，不含驱动句柄。
raw token 只在构造 AIV WQE entity 时转换为 WQE token。`CpuDdrRegistration` 持有 CPU
token/MR；`HccpRemoteMemoryBinding` 和设备侧 `UmdkRemoteMemoryBinding` 持有具体导入。
AICPU 的 `urma_target_seg_t*` 和函数指针只在设备所属进程内使用，Host view 使用
`UmdkContext {session, owner_pid}` 等描述，不传设备侧 C++ 指针。

| 共享对象 | 身份与回收 | 仍独立的部分 |
|---|---|---|
| HCCP transport API 表 | 进程首次完整加载成功的 provider；一份库引用保留至退出 | 每 transport 的 RaCtx、注册、Jetty/CQ、entity 和状态 |
| AICPU `ApiEnvironment` | 实际加载的 liburma 对象；共享初始化和函数表，最后使用者安全退出后回收 | session 的 context、HBM 注册、远端 import 和队列 |
| CPU `CpuUrmaMemoryPool` | 一次 allocation 代次；兼容 candidate/EID/context 下共享 DDR 注册 | 执行 session 的通道、Gather/Push worker、协议状态 |
| UBMEM HAL | 按配置库路径在进程内首次加载，成功后保留至退出 | 各池物理 allocation、虚拟地址和映射 |
| UBMEM `DeviceMappingCache` | Runtime 内按设备、完整 Fabric handle 和映射长度复用 | Put/Get metadata、完成状态和执行槽位 |

CPU 共享注册的键包含 allocation/context 代次、地址范围、candidate/EID、权限和凭据。
同址新 allocation 或不同 context 不命中旧注册。session 关闭先停工作与队列、归还内存
引用；Region 关闭再注销 MR/token、关闭 context、最后释放 DDR。任何一步失败保留尚需
依赖，允许重试。CPU execution session 的 wire 身份仍独立，本次共享没有改变 wire 协议。

CPU 共享 MR 不代表 NPU/AICPU 远端 import 自动共享。AIV Get/Put 使用相同
`UrmaBindingCache` 实现但各自持有缓存；键含方向、context/代次、远端
epoch/region/session、范围、token、EID、权限和完整 key。AICPU 同一 session 的多个
worker 复用其远端导入，不同本地 context 继续独立导入。更广泛的 HBM 注册、context 和
远端绑定共享仍需明确 storage 代次和兼容身份。

AICPU 准备时将函数表复制到 session，执行入口解析一次 session，WR/CQ 热路径不增加
环境查询。AICPU 环境最终清理失败时保留句柄并阻止新引用进入。HCCP 成功加载后锁定
本进程 provider，不为每个 transport 追加永久 dlopen 引用；切换 SDK/库需使用新进程。

AICPU 队列布局由 `UmdkQueueProfile` 表达：Put 为 1 control / 0 data，Gather 为
N control / N data，CPU 通道数相匹配。Scatter 使用自己的 CPU Push 工作区与队列。
队列布局优化不改变 Put 同步事务语义。

## Region、协议与恢复

`CacheRegionService` 分开持有 allocation 和执行 session。PoolReady 不建立普通访问的
URMA context/队列；路由发现只枚举候选。主池不再预留 Gather 辅助空间。首次 Gather
准备通过 `kPrepareGatherWorkspace`（命令 14）独立分配、注册 CPU 工作区，同一 session
复用且禁止原地扩容；关闭时先停止 worker、释放通道，再注销/释放工作区。CPU Gather
使用主池读取 payload，独立工作区承载描述符、控制字和 staging，二者无需地址相邻。
按需注册不会清零或替换已有缓存数据。

Runtime 构造和请求级 PoolReady 不解析或加载 AICPU 包。首次对应 provider 准备时检查
CANN 安装并解析所需 entry，成功后缓存；Scatter 不要求 Gather/Put 的入口存在。显式
`warmup`、绑定或 `begin_get_graph` 也属于首次准备，图捕获开始前必须完成。

控制面使用命令 20、版本 1 的显式小端外层记录，内部旧控制命令沿用主线 v6 ABI。
CPU 服务使用有界连接线程池，慢操作锁定对应记录。同一 client/create-request ID 必须
携带相同配置，控制操作也按 ID 去重。回复丢失时保留原 ID，先重发或查询原操作，确认
旧资源关闭后才新建。epoch 改变时拒绝把旧请求解释为新资源。单 session 控制历史上限
4096 条，达到后需关闭重建，不淘汰尚可重放的 ID。

资源管理器限定在一个 Runtime/Region 代次内，工厂固定设备、路由、ABI、资源预算等
创建参数。仅 Describe 不打开远端 session；首次 Acquire/Pin 才初始化对应配置。初始化
失败缓存失败状态，需显式 recover；关闭失败不替换 owner。正常关闭先停止本地执行，
关闭远端 session/归还共享引用，再释放 Region；有 session 的 Region 拒绝释放。

CPU HAL/SVM 通过 `AcquireProcessHal` 跨池和 CPU Runtime 重建保留。成功加载由进程
注册表强引用，失败加载不缓存，纯 URMA allocation 不触发加载。注册表不持有池数据；
池仍逐个 unmap、释放地址和物理内存。此边界避免卸载 HAL 后用户态初始化标志重置、
驱动 SVM 进程上下文仍存在，重开时 OPEN ioctl 返回 `EEXIST` 并导致 `halMemCreate ret=38`。
已经处于该错误状态的旧 CPU 服务须退出，以新进程恢复。

Put 不排空 Get，但访问的数据必须符合稳定读/写入规则。跨 transport 可见性依赖已完成
的 WRITE/MTE 和平台一致性，CPU fence 不能代替数据面完成。AccessTable handle 释放后，
在途请求或 graph 仍保留其槽位 pin，直到安全完成；空闲输出绑定不额外保留表/槽位。

### Put 事务恢复

`batch_put` 依次准备驻留实例并取得执行占用、预留元数据、在同一实例执行 WRITE、
Commit/Abort、确认完成。元数据准备返回错误或抛异常时，`CancelPreparedPut` 归还尚未
提交的执行占用，后续可继续使用原驻留实例；已经提交的工作仍由完成恢复流程处理。
`runtime.recover(原 Put 配置)` 查询原次执行，不重放 payload，也不使用已返回调用的
请求向量或目标描述：

| 原次结果 | 元数据及资源 |
|---|---|
| Succeeded | CommitPut；确认成功后清本次结果、归还短期租约，保留驻留状态和有效输入绑定 |
| 确定 Failed，且不会继续访问 buffer | AbortPut；确认成功后清本次结果、归还短期租约并保留实例，返回原次失败 |
| Unknown | 保留事务、输入引用、注册、会话和租约，后续继续查询 |

远端 WR 错误/CQ 超时不自动等于安全失败。元数据已完成而 ACK 失败时，重试只确认，
不重复 Commit/Abort。不能用重建队列或重放 Put 消除 Unknown。无未决事务时，recover
才关闭并重建该配置资源；Put 输入绑定须先结束。Get recover 要求该实例无输出绑定、
该实例无 graph，且普通在途 Get 已安全处理；无关实例的图不阻止恢复。失败保留原依赖。

```python
# 原次 batch_put 返回未决完成错误后，保留 source/stream，查询同一配置。
# 如再次失败，按错误继续保留；不能直接再次 batch_put 重放原事务。
runtime.recover(put)
# 返回确定结果后，已有 input_lease 可继续使用或显式结束。
```

`shutdown()` 在确认数据面停止后清理 Runtime，可回滚尚未提交的 Put 事务。
图租约必须先结束；关闭、同步、STOP、注销或 unmap 失败均保留必要所有权以供重试。

## 配置、预热与普通调用

当前 Python 构造只接受两个布局位置参数，其余配置均为关键字参数：

```python
runtime = kvaf.NpuRuntime(
    batch_get_token_bytes, batch_get_tokens_per_block,
    device_id=0, transport_mode="urma", default_put=None, default_get=None,
    put_resources=None, get_resources=None, npu_eids=[], peer_ub_device="",
)
```

旧构造调用不会静默兼容，须按下表迁移；C++ 同样改用
`NpuRuntimeOptions(token_bytes, tokens_per_block)` 及各算法资源项：

| 已移除的配置 | 当前配置方式 |
|---|---|
| `aicpu_batch_get_mode` | 用 `GetExecutionConfig` 明确 executor/algorithm，设为 `default_get` 或在调用时传 `execution`；Put 的选择独立由 `default_put` / `PutExecutionConfig` 决定 |
| `per_call_execution`、脚本 `--execution-mode` | 无替代开关，所有请求统一使用 Dispatcher；eager/graph 与同步/异步仍独立选择 |
| `transport` | `transport_mode` 规定池的传输范围，各次配置选择具体 transport |
| 全局核数、Jetty、in-flight 和 Get/Put kernel 路径 | 放入对应配置的 `ExecutionResources`，分别列入 `put_resources/get_resources` |
| `enabled_put_configs/enabled_get_configs` | 无需填写，按构建能力和 transport 模式选择；首次准备时检查具体资源 |
| `adaptive_ub_*` | 不再启动测速选路；需要固定路由时使用 `npu_eids/peer_ub_device` |

以下片段使用已创建的 NPU Tensor、stream 和缓存内容。当前构建支持的算法自动可用，
无需 enabled 列表；默认值只决定未传 execution 时选哪个算法，资源配置在首次使用时检查：

```python
put = kvaf.PutExecutionConfig("urma", "aiv")
put_cpu = kvaf.PutExecutionConfig("urma", "aicpu")
put_mapped = kvaf.PutExecutionConfig("ubmem", "aiv")
direct = kvaf.GetExecutionConfig("urma", "aiv", "direct")
gather = kvaf.GetExecutionConfig("urma", "aicpu", "gather")
scatter = kvaf.GetExecutionConfig("urma", "aicpu", "scatter")
mapped = kvaf.GetExecutionConfig("ubmem", "aiv", "direct")

runtime = kvaf.NpuRuntime(
    batch_get_token_bytes=656, batch_get_tokens_per_block=128,
    transport_mode="urma+ubmem", default_put=put, default_get=direct,
    put_resources=[(put, kvaf.ExecutionResources(cores=2, jettys=2, kernel_path="put.o"))],
    get_resources=[
        (direct, kvaf.ExecutionResources(cores=4, jettys=2, max_inflight_total=2048, kernel_path="get.o")),
        (gather, kvaf.ExecutionResources(cores=2, cpu_threads=8, request_capacity=4096)),
        (mapped, kvaf.ExecutionResources(cores=4, kernel_path="get.o")),
    ],
)
runtime.link_to("cpu-host:port")
runtime.initialize(ddr_bytes, slot_size)
runtime.warmup(gather, request_capacity=4096, stream=read_stream)
# table 是 access_table_view(handle, device="npu") 返回的 DeviceAccessTable。
runtime.batch_get(table, indices, offsets, miss_count, output_indices, output,
                  stream=read_stream, execution=gather)
runtime.batch_put(other_keys, source, input_offsets, lengths,
                  stream=write_stream, execution=put)
runtime.drain_get()
```

Put 输入为调用方拥有的连续 buffer，keys/input_offsets/lengths 可为 Python 整数序列或
连续 CPU int64 Tensor。Get 的 indices/offsets/output_indices 为 NPU int32 数组，
miss_count 为 NPU int32[1]；offsets 使用字节，只有 `[0, miss_count)` 请求有效。
输出每行大小须符合 token_bytes。使用 Runtime 生成的 DeviceAccessTable 时保留 Host
表生命周期；直接传 addresses/valid Tensor 时，调用方承担该地址表的有效性。

`cores=0` 使用下表对应算法的默认核数/worker 数；URMA AIV 的 `jettys=0` 使用默认 2。
`max_inflight_total=None` 使用默认 2048，显式 0 关闭限流。AIV/UBmem 的
`kernel_path=""` 使用构建时记录的默认绝对路径：Get 为
`<build-dir>/bin/kvaf_execute_batch_get_kernels.o`，Put 为
`<build-dir>/bin/kvaf_execute_batch_put_kernels.o`。`./build.sh npu` 默认构建到
`build-npu`，自定义 `--build-dir` 自动生效；应用可从其他目录启动。
非空 `kernel_path` 覆盖默认路径；移动构建产物后需重新构建或显式指定新路径。
AICPU 不设置 `kernel_path`，首次准备时从 CANN 安装目录定位包。
配置的 request_capacity 范围为 1..65536，是 provider 请求上限；Gather/Scatter 内部
工作区仍按固定 ABI 最大容量准备。`cpu_threads=0` 使用服务默认值。Jetty 参数用于
URMA AIV，in-flight 用于 URMA AIV Get，CPU 线程预算用于 AICPU Gather；其他后端拒绝
无效调优字段。标准构建的同一方向 `.o` 含 URMA/UBMEM 入口，也可按配置覆盖 kernel_path。

| 算法 | 默认核数/worker | 默认每核 Jetty | 默认在途上限 | 默认 kernel 来源 |
|---|---:|---:|---:|---|
| URMA AIV Put | 2 | 2 | 不适用 | 构建目录 Put `.o` |
| UBMEM AIV Put | 16 | 不适用 | 不适用 | 构建目录 Put `.o` |
| URMA AICPU Put | 1 | 内部管理 | 不适用 | CANN 安装包 |
| URMA AIV Direct Get | 2 | 2 | 2048 | 构建目录 Get `.o` |
| UBMEM AIV Direct Get | 16 | 不适用 | 不适用 | 构建目录 Get `.o` |
| AICPU Gather Get | 5 | 内部管理 | 不适用 | CANN 安装包 |
| AICPU Scatter Get | 5 | 内部管理 | 不适用 | CANN 安装包 |

本分支新增的 `request_capacity` 默认 65536。Gather 的 `cpu_threads=0` 沿用 CPU 服务
配置，与 main 一样服务默认 10 个线程；服务启动参数 `--gather-threads` 可覆盖。

预热不提交 key、不搬运应用 payload，也不注册未提供的应用 buffer。Put/Get 预热同时准备
对应配置的驻留执行状态，完成后归还短期执行租约。`batch_get` 默认异步；调用方可用同
stream 顺序消费，但外部 stream 同步不会替 Dispatcher 归还执行占用。
需要完成管理与释放在途引用时调用 `drain_get()`；
它自己等待完成，不必先做一次外部同步。`synchronize=True` 则直接完成本次调用。

### 显式输出绑定

两个支持注册的 Get 可分别绑定并交替调用，仍由同一 Get 调度方向顺序执行：

```python
ld = runtime.begin_get_output(direct, direct_output, capacity, read_stream)
lg = runtime.begin_get_output(gather, gather_output, capacity, read_stream)
runtime.batch_get(table, indices, offsets, miss_count, output_indices, direct_output,
                  stream=read_stream, execution=direct, output_lease=ld, synchronize=True)
runtime.batch_get(table, indices, offsets, miss_count, output_indices, gather_output,
                  stream=read_stream, execution=gather, output_lease=lg)
runtime.drain_get()
runtime.end_get_output(ld)
runtime.end_get_output(lg)
```

绑定后该实例的请求必须传匹配 lease、配置、输出地址/大小/device，容量不超上限。
普通 eager Get 可切换 stream，切换时重新准备所需执行槽位。结束空闲实例的绑定不影响
其他在途实例；结束正在执行的实例则先 Drain。失败保留绑定。UBMEM/Scatter 不需要
应用输出注册，不支持此输出绑定接口。输出绑定可与图共存；同上下文内图和 eager
各自持有注册引用，结束其中一个不会注销另一个仍引用的 MR。显式输出绑定的 eager
调用仍需携带其 output_lease；图节点使用自己的 graph_lease。

### 显式输入绑定

```python
li = runtime.begin_put_input(put_cpu, source, request_capacity=put_rows, stream=write_stream)
runtime.batch_put(keys, source, input_offsets, lengths,
                  stream=write_stream, execution=put_cpu, input_lease=li)
# 后续可更改 keys/offsets/lengths 和 source 内容，遵循原有写入及 stream 顺序契约。
runtime.end_put_input(li)
```

URMA AIV/AICPU Put 都支持。绑定固定输入地址、大小、device、stream、配置和容量上限；
整个 Put 域只有一个活动输入绑定，切换其他 Put 配置前需结束原绑定。输入绑定存续期间
必须传匹配 lease；未决事务须先 recover/确认。绑定复用该配置已有的驻留实例，End 成功
只结束输入绑定，保留实例及准备状态；End 失败保留绑定，不能释放源存储。
UBMEM 不需要输入 MR，使用普通 Put。URMA Put 的普通调用临时注册与显式绑定的
长期注册需区分，不再保留旧 AICPU Put 的隐式 MR 缓存路径。

## Get 图模式

四种 Get 推荐通过 `runtime.graph_resources()` 自动管理一张模型图的 Get 节点。用户不需要
逐层写准备清单或向模型 forward 传入 lease；普通 `batch_get` 在准备/捕获 scope 内自动路由。

```python
resources = runtime.graph_resources()
# 只包住一次与目标捕获路径相同的预热；内部的 batch_get 真实执行。
with torch.npu.stream(read_stream):
    with resources.prepare(read_stream):
        model_forward(dummy_inputs)

# 此 scope 不启动 torch capture，只关联 KVAF 的准备结果。
graph = torch.npu.NPUGraph()
with resources.capture(read_stream):
    with torch.npu.graph(graph, stream=read_stream):
        model_forward(dummy_inputs)

# 更新固定输入内容；调用方保证共享通信资源上无其他在途执行。
with torch.npu.stream(read_stream):
    graph.replay()
read_stream.synchronize()
# 可选报告查询（先在 prepare 前启用 profiling）：
# profiles = runtime.batch_get_kernel_profile(graph_lease=resources.leases[0])

# 不再 replay，先销毁图，再释放它持有的资源。
graph.reset()
resources.close()
```

每个资源对象只 prepare/capture 一次。prepare 结束时完成普通 Get 并在图外创建节点，
capture 检查调用配置、输出 storage/地址/大小、请求容量和节点顺序/数量；不匹配时拒绝，
不会在 capture 内补资源。相同输出的两次调用仍有两个节点。固定输出须在预热时已存在；
捕获期间才新分配的输出不满足此接口契约。capture 内 Get 必须 `synchronize=False`、
`profile=False`，不混用显式 `graph_lease/output_lease`。

模型预热可更新工作数据，但不能把预热结果当作已回放的图报告。每个图节点拥有独立报告；
索引、offset、miss_count、output_indices 的内容可变，存储地址必须保持有效。
Python 对输出及 storage 的保活不允许调用方 resize/set_ 或回收其底层 allocation。

一个图固定一个 stream，不同图可以使用不同 stream。外部原生 `graph.replay()` 保留，
KVAF 不代理 replay、不自动记录 event 或维护跨 stream 依赖。应用负责在共享队列/工作区上
串行执行图与普通 Get，例如先 `stream.synchronize()` 再切换到另一条 stream。图存活期间
普通 Get/Drain 可以工作，使用自己的状态；预热只能使用已固定容量，不能扩容图依赖。
profiling 切换和 recover 的阻塞按算法实例判断；关闭 Runtime 前仍须结束全部图。

显式 `begin_get_graph(cfg, output, request_capacity, stream)` →
`batch_get(..., execution=cfg, graph_lease=lease)` → `end_get_graph(lease, stream)`
仍是可用的底层接口，单个 lease 只捕获一次 Get；不使用自动 scope 时由调用者自己管理。
图与 eager 的输出绑定可以共存；AIV Direct/Gather 匹配同上下文已有注册时保留引用，
没有匹配项才新增注册，最后引用结束后注销。UBMEM/Scatter 不需要应用输出 MR。

无论正常结束还是 capture 异常，调用者都必须先退出 capture、停止 replay、确认在途完成、
销毁外部 graph，再调用 `resources.close()`。失败路径或未回放的图可使用
`resources.close(discard_result=True)`；失败保留未释放节点供重试，成功 close 可重复调用。
释放公共 AccessTable handle 不会回收仍由图节点固定的 DDR 槽位。

AICPU Gather/Scatter 用设备侧 ticket 推进每次 replay 的通信序号。正常结束检查该节点
最后一次 replay 状态；未 replay 的图可能报未完成，可明确 `discard_result=True` 取消。
同步或注销失败保留租约。共享 AICPU 会话失败后，先停止回放并销毁引用该算法实例的所有图，
再逐个调用 `end_get_graph(lease, stream, discard_result=True)`。同实例尚有租约未进入
丢弃流程时返回 Busy 并保留资源；最后一个租约关闭会话后，重试先前收到 Busy 的租约。
结束全部图租约后调用 `recover(cfg)`，不能直接复用可能损坏的队列。discard 不跳过 DMA
安全完成要求。
URMA AIV 和 UBMEM 的结果处理差异见下表。

## 状态、profiling 与计时口径

下表描述当前唯一执行路径。历史版本的同步、状态读取和注册生命周期可能不同；
跨 commit 性能对比需记录相应完成边界，不能假设与当前版本完全一致。

| Get 路径 | 每次 status 初始化 | 同步调用 / 异步 Drain | Graph 结束 |
|---|---|---|---|
| URMA AIV Direct | 不逐次清零 | 等待完成并清理请求，不回读或检查 Get 状态报告；ACL 提交/同步失败仍返回错误 | 等待 stream 并退役注册，不回读校验 status |
| UBMEM Direct | 同步/异步 eager 和 capture 均不逐次清零 | 等待完成并清理请求，不回读或检查 Get 状态报告；ACL 提交/同步失败仍返回错误 | 同步并回读检查；discard 跳结果检查但不跳同步 |
| AICPU Gather | 依赖设备 ticket/序号及报告；准备阶段初始化工作区 | 同步排入状态 D2H 并完成检查；异步 submit 不回拷，Drain 才执行 | 检查最新 replay 的完成和序号 |
| AICPU Scatter | 依赖设备 ticket/序号及报告；准备阶段初始化工作区 | 同步执行 Drain；异步 submit 不回拷，Drain 才执行 | 检查最新 replay 的 worker 状态和序号 |

UBMEM Put 继续保留状态初始化和覆盖行数检查；Get 不提供逐次初始化哨兵检测，
而依赖 kernel 覆盖报告。AIV/UBMEM 的显式 profile 快照在执行完成后读取并校验设备报告；
eager 完成路径的状态处理不适用于 Put 事务，Put 仍必须确认完成结果，未知结果保留事务。

`start_batch_get_kernel_profile`、`batch_get_kernel_profile`、`stop_batch_get_kernel_profile`
按 `execution=cfg` 路由至相应实例，省略时取默认 Get。图前启用、每轮安全完成后读取、
结束该实例全部租约后停用。图 profile 和 AICPU 状态查询始终传
`graph_lease=lease`（也可由 `resources.leases` 取得），报告对应该节点最近一次完成的结果。
不传租约查询普通 eager 实例，避免同一实例的图/eager 共存时混淆。
正常结束租约时保存最后成功的 profile 供结束后按配置读取；不能在 capture 内复制或
切换 profiling。AICPU 的 `check_batch_get_status(stream, execution=cfg, graph_lease=lease)`
检查具体节点，保留首个失败结果；它不代替 Dispatcher 的图租约回收。

设备计算保留 canonical 地址转换：AIV 在向量执行前计算地址差，lane 内加法；Gather
每 worker 预计算 registered staging 基址。AICPU Gather 每 worker 的入口解析一次
session registry，持有 session 读锁及 TLS scope，内部描述符发送、通知、READ 与 ACK
复用该 scope；worker 数据/控制队列锁和关闭的 session 写锁仍保留。共享环境不增加
逐 WR 的查找。

设备输入验证保留原数据路径：不为 AIV Get 重新打开原来关闭的全量预扫描；Gather/
Scatter 不增加逐 token 资源代次检查。调用方须保证索引、offset、地址及输出行有效，
形状、资源代次、范围、CQE/超时和完成检查按各层契约保留。

性能入口使用唯一执行路径，保留 `--execution eager|graph|both` 及同步/异步完成选项；
不再接受 `--execution-mode`。旧版本基线从独立 commit 构建并使用其匹配脚本与服务。
性能入口默认关闭 kernel profiling；显式加入 `--kernel-profile` 才启用设备打点、报告回读及
设备时间输出。数据预检、输出校验和 profile 快照在性能计时之外；多图检查通过
`--multi-graph-check` 在计时前执行，正式 graph 性能样本仍使用单节点图。
端到端对照须固定同步、profile、工作规模和线程配置：同步 call-to-return 与异步
submit-only 不能混比。当前异步 call-to-completion 直接使用 `drain_get()`，不先重复
外部同步；旧版本若使用外部 stream 同步，须单独注明其完成与清理边界。绑定 setup/teardown、首次建链及预热单独计量，不计入热调用
带宽；测冷启动时另行报告。具体命令与 trace 字段见[验证步骤](cache_transport_execution_validation.md)。

## 部署与兼容

CPU 服务、NPU Host 库和设备 kernel/包一起重新配置、构建和部署，启动新的客户端及
CPU 服务进程。库名和 `build-npu/bin` 产物位置保留；AICPU 构建包迁移到
`build-npu/access_operator/transports/aicpu_urma`，安装脚本使用该新目录。

| 组件 | 当前符号或约束 |
|---|---|
| AIV Get / Put | `KvafExecuteBatchGetAivKernelV3` / `KvafExecuteBatchPutAivKernelV2` |
| AICPU Control | 新 Host 要求 `KvafAicpuUmdkControlSessionsV5`，支持 QueueProfile |
| AICPU Gather / Scatter | `KvafAicpuGatherReadExecutionV3` / `KvafAicpuCpuPushScatterExecutionV3` |
| Control 兼容入口 | 旧 Control/V3/V4 保留原 Gather 队列布局；新 Host 不静默降级 |
| CPU 服务 | 请求级能力协商拒绝不兼容服务；`--gather-affinity` 已移除，使用当前线程调度参数 |

QueueProfile 使用原结构的 4 字节 padding，不改变其余字段偏移或 Put ABI；不能据此
混装旧设备包，因为新 Host 依赖 V5 能力。控制协议的兼容性也不替代全部组件一起更新。

原 `tests/` 顶层 C++ 功能测试及其 ACL/URMA stub、kernel CPU 模型已移除，
不再配置独立 SDK 头文件目录来启用这些测试。
独立 Python 功能、配置与单元测试及其 CMake 注册也已移除。
性能入口保留数据预检和输出校验，公共模块集中在 `tests/python/utils/`；
构建、性能测量和计时范围见验证文档。

本机没有 Ascend 设备或正式 CANN 编译环境。目标环境须完成正式 SDK 编译、新 kernel 部署、真实
payload 的交叉写读、Get/Put 重叠、混合可见性、graph replay 和故障退出验收。
