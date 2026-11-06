# 缓存按次执行：第一阶段实现

基线：2026-11-06 获取的 `origin/main`，提交 `1c13103`。
分支：`codex/cache-transport-execution`。
设计依据：[缓存访问与执行配置设计](cache_transport_execution_design.md)。

本分支实现配置、按方向调度、已有 provider 的适配和安全引用管理。
**尚未实现设计全文：混合模式的 CPU Region/绑定协议仍待实现，因此
`urma+ubmem` 明确拒绝初始化；不应把本分支标为 18 对交叉组合验收完成。**

## 本次落地

- C++ `TransportMode`、`Transport`、`Executor`、`GetAlgorithm`，以及
  `PutExecutionConfig` / `GetExecutionConfig`。保留 `Transport::kUrma/kUbmem` 源码兼容。
- 统一目标组合校验、完整默认值替换、enabled 集合裁剪、空集合关闭方向。
  编译能力及 kernel 缺失在 NpuRuntime 构造阶段报错。
- `ExecutionDispatcher` 保留 `AccessOperator` 适配接口，以独立 provider 实例作为
  Put/Get 资源所有者；每个配置在本方向内按需初始化一次，失败保留清理状态，
  不自动重试、不切换后端。初始化失败的会话需 Shutdown 成功后重新初始化。
- 新模式初始化不执行旧的写入探测；选定首个路由后，实际建链失败由首调返回。
  保留旧模式的候选探测与 graph 路径。新模式尚无自适应选路或显式重试接口。
- AIV Direct 按方向创建 transport；持久队列位置保留在实例中。
  新模式 Get 完成时收集设备状态并注销本次 HBM 注册，Put 同步写完后注销输入注册。
- AICPU Put/Get 分离 Host UmdkSession，设备侧改为 session ID 注册表及会话锁，
  URMA 库初始化按引用计数管理；删除 Put 前的 DrainGet。
  新模式通过 `kUnregisterBuffers` 在完成后回收临时 HBM 注册。
  **必须重新构建并部署本分支 AICPU kernel 包，不能配合旧的单会话设备库使用。**
- UBMEM 新模式每个方向使用独立 adapter 实例、状态、metadata、event 和映射租约。
  目前各实例分别 import 同一 Fabric 导出；统一共享映射缓存还未提取。
- Runtime 使用生命周期读写锁和两个方向锁，元数据表索引只短暂加锁。
  `ReleaseAccessTable` 使 handle 立即失效，最后一个在途引用退出才 UnlockKeys。
  `DrainGet()` 只等待 Get；Put 失败若无法确定 DMA 已停止，保留事务直至 Shutdown
  成功后 Abort，期间拒绝复用 Put 槽位。
- Python 增加 `execution=`、`drain_get()`、`warmup()`；Put 请求改为调用局部对象，
  等待时释放 GIL；新模式持有每次异步 Get 的 Tensor 引用，失败时保留依赖。
- `CacheRegionDesc::Translate` 提供带逻辑范围/溢出检查的固定地址转换接口。
  当前接入的 URMA 路径沿用主线相同基址，UBMEM 沿用其已有 kernel 偏移转换。

## 当前能力和边界

| 能力 | 本分支状态 |
|---|---|
| URMA AIV Put / Direct Get | 接入按次执行；需硬件验收 |
| URMA AICPU Put / Gather Get / Scatter Get | 接入按次执行；需硬件验收 |
| UBMEM AIV Put / Direct Get | 接入独立模式；需硬件验收 |
| AIV Gather / Scatter | 只预留配置和 provider 注册接口，构造时明确报未实现 |
| URMA + UBMEM 混合池 | 未实现；在分配 DDR 前报错 |
| 同一个 Runtime 同时启用 AICPU Gather 与 Scatter | CPU Region 仍存在单 worker 模式限制；构造时拒绝 |
| Direct 与一个 AICPU Get 后端切换 | 可配置，Gather workspace 在分配池时预留，不进入 MetaManager |
| CPU 侧 URMA context/token/MR 按方向独立 | 仍沿用主线 Region 级绑定，尚未拆分 |
| 远端创建 ID 幂等查询/恢复、能力协商 | 尚未实现，不自动重试未知结果 |
| 新模式 graph capture | 不支持；AIV/AICPU/UBMEM 显式拒绝 capture，旧入口保留原能力 |
| 各 provider 独立核数/worker/容量预算 | 仍使用现有 Runtime 参数，后续扩展 |
| 容量/buffer 级完整预热 | `warmup` 仅准备不依赖应用 buffer 的资源；不保证下一次调用零初始化 |
| CPU Pull / HBM Pull | 此 main 基线无对应集成，本分支未引入参考分支实现 |

CPU 侧上下文隔离、混合分配、幂等控制协议、带 buffer 注册的 graph 租约是后续阶段，
不能依据本地 Mock 测试宣称硬件并发或跨机制可见性通过。

## Python 示例

旧构造参数不变；没有新增 execution 配置时继续采用原后端入口。
通过 enabled/default 参数或 `per_call_execution=True` 显式启用新入口。
省略 enabled 时使用设计中的完整目标集合，因此当前未合入的 AIV Gather/Scatter
会触发能力校验；本版本请显式裁剪 enabled。

```python
put_aiv = kvaf.PutExecutionConfig("urma", "aiv")
put_aicpu = kvaf.PutExecutionConfig("urma", "aicpu")
get_direct = kvaf.GetExecutionConfig("urma", "aiv", "direct")
get_gather = kvaf.GetExecutionConfig("urma", "aicpu", "gather")

runtime = kvaf.NpuRuntime(
    batch_get_kernel_object_path="get.o",
    batch_get_token_bytes=656,
    batch_get_tokens_per_block=128,
    batch_put_kernel_object_path="put.o",
    npu_core_count=2,
    transport="urma",
    default_put=put_aiv,
    default_get=get_direct,
    enabled_put_configs=[put_aiv, put_aicpu],
    enabled_get_configs=[get_direct, get_gather],
)
runtime.link_to("cpu-host:port")
runtime.initialize(ddr_bytes, slot_size)  # 保持 main 的生命周期入口
runtime.warmup(get_gather)                # 不注册应用尚未提供的 buffer
runtime.batch_get(table, indices, offsets, miss_count, output_indices, output,
                  stream=read_stream, execution=get_gather)
runtime.batch_put(other_keys, source, input_offsets, lengths,
                  stream=write_stream, execution=put_aiv)
runtime.drain_get()
runtime.shutdown()
```

UBMEM 使用 `transport="ubmem", per_call_execution=True`，配套 UBMEM kernel
及 UBMEM 的默认 Put/Get 即可。裸地址接口和 C++ 非拥有 buffer 仍由调用方保证
完成前的生命周期与数据依赖；`drain_get()` 是收集 KVAF 错误的完成入口。

## 验证

便携构建包含 Python binding、原有回归及新增 `kvaf_execution_dispatcher_test` /
`kvaf_execution_config_python`。可选 `KVAF_TEST_URMA_INCLUDE_DIR` 启用实际设备端
session 注册表的 Host 测试（不加载 liburma），覆盖 ID 隔离、线程局部视图、独立锁和失败创建。新增测试覆盖组合校验、禁用方向、创建只执行一次、
Get 建链时 Put 独立推进、Get 在途时 Put 不排空、句柄失效与槽位 pin、同表切换、
Put 未决事务和关闭失败重试。

本机没有 CANN/Ascend 设备。硬件相关 Host 与 AICPU session 源码使用本地 hcomm
SDK stub 头文件执行语法检查。UBMEM 检查使用了两个仅为适配旧 stub 的宏别名；
这不代表正式 CANN SDK 构建或运行验证。合入/部署前需要在目标 CANN 环境编译，
部署新版设备库，并执行实际 payload、并发、故障及退出测试。
