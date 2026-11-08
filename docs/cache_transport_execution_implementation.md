# 缓存访问执行框架：实现与使用

基线：`origin/main` 的 `1c13103`；分支：`codex/cache-transport-execution`。
设计依据：[缓存访问执行框架设计](cache_transport_execution_design.md)。

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
| 生命周期 | Put / Get 分别串行、跨方向独立；AccessTable 失效后仍由在途请求或 graph 保留槽位引用 |
| 恢复协议 | 能力协商、服务端 epoch、幂等创建和控制 ID、查询、可重试关闭、显式 `recover` |
| 资源预算 | 各配置独立 cores / Jetty / in-flight / 请求容量 / CPU Gather 线程数 / kernel 路径 |
| 预热 | 按容量建链、准备内部 buffer、Gather worker、Put metadata；可指定实际调用 stream |
| Graph | URMA / UBMEM AIV Direct Get 的显式租约，固定输出、提前准备、独占 Get 域、保留 Tensor 和表引用 |
| AIV Gather / Scatter | 仍按约定只预留配置及 provider 接口，启用时明确报未合入 |

当前已接入的普通访问组合为 3 种 Put × 4 种 Get。上述为实现状态，**不代表目标
硬件上的交叉组合、并发性能和跨 transport 可见性已验收**。AIV Gather / Scatter
合入后才能补齐设计中的 18 对组合。

CPU Pull / HBM Pull 属于设计的可选扩展，本次 main 没有相应集成，未额外引入参考分支。
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

Put 不排空 Get。异步 Get 的 `drain_get()` 等待完整数据面并收集设备结果。未确认完成
的 Put 保留事务和输入引用，需要成功 `shutdown()` 后才能清理；不能用 `recover`
绕过未决写入事务。资源准备失败可以显式 `recover(execution)`，它不改变配置。
跨 transport 的可见性依赖已完成的 WRITE / MTE 和目标平台一致性契约，控制面 CPU
fence 不冒充数据面完成。

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
UBMEM 与 URMA 必须使用各自的 kernel 文件。
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
本轮启用上述两个 mock 测试选项及 Python 绑定，24 项 CTest 分组运行通过；TCP 测试
需要允许本机回环监听。`git diff --check` 通过。

本机没有 Ascend 设备或正式 CANN 编译环境；NPU Host、AICPU 与 CPU URMA 源码使用
本地 SDK stub 进行语法检查。目标环境仍须完成正式 SDK 编译、新 kernel 部署、真实
payload 的交叉写读、Get/Put 重叠、混合可见性、graph replay 和故障退出验收。
