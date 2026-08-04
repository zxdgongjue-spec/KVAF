# A5 HCCP 控制面 + AIV/SIMT 直写数据面

## 决策

KVAF 不再把“在 AICPU 内自行拼装 UMDK 私有资源”作为主路径。新的候选架构是：

- Host 长生命周期 owner 通过驱动随附的 `libra.so` 创建并持有 HCCP 资源；
- 优先使用 `RaCtx*` Context/MR/CQ/QP/TP 接口；若安装版本只提供 SHMEM
  已采用的旧接口，则评估 `RaAiQpCreate`/`RaRegisterMr` 路径；
- HCCP 返回已经映射到 Device 可见地址空间的 SQ、CQ、doorbell 和共享状态地址；
- AIV+SIMT 在 decode 热路径直接生成 WQE、更新 PI/doorbell、轮询 CQ；
- HCCP 不参与每个 1 KiB 请求，也不参与每个 batch 的控制交互。

这仍然是“不链接 HCOMM”的方案。这里复用的是 HCCP 的资源所有权、内核/驱动适配和
HAL 地址映射，不是 HCOMM 的 collective 或传输调度层。

## 为什么比 Direct-UMDK 更合适

固定 UMDK 源码在 Host 用户态可以创建普通 URMA 对象，但 KVAF 已经证实，仅复制这部分
代码不足以复现 A5 的 Device 可见资源：JFC/Jetty/doorbell 还需要产品相关的资源 ID、
UASID/进程身份、HAL map/unmap 以及 HCCP 服务端状态机。继续猜测这些私有参数会让
`CREATE_JFR` 失败，而且错误返回不能区分“参数错”“调用域不支持”和“官方 wrapper
尚未实现”。

HCCP 的 A5 产品实现已经包含这些步骤：

1. 创建并激活 JFC/Jetty；
2. 用产品/Die/Function 信息生成 map resource ID；
3. 通过 HAL 将 CQ、SQ 和 doorbell 映射给目标 Device 进程；
4. 把映射后的地址返回给客户端；
5. 销毁时按 unmap、deactivate、free 的逆序回收。

SHMEM 的现有实现也说明旧版 `libra.so` 的 `RaAiQpCreate` 能返回 `HccpAiQpInfo`，其中
包含 SQ/RQ/SCQ/RCQ 的 buffer、depth、WQEBB/CQE size、head/tail 和 doorbell 地址。
这些字段正是 AIV 数据面所需的信息。

## Decode 生命周期

一次 decode session/epoch 的资源生命周期如下：

1. **初始化一次**：加载绝对路径 `libra.so`，核对 build/version/symbol surface；
2. **注册一次或按内存池扩容**：Host DDR MR 注册，交换 MR key 与 endpoint metadata；
3. **建链一次**：创建 Context/CQ/QP，导入并绑定对端，等待 TP ready；
4. **重复传输**：每个可配置 batch（例如 4096 个 1 KiB descriptor）只更新描述符、
   PI/doorbell 并轮询 CQ；不重复注册 MR、创建 QP 或建立 TP；
5. **关闭一次**：停止提交，排空 CQ，然后严格逆序执行 unbind、unimport、QP destroy、
   CQ destroy、MR unregister、Context deinit，最后卸载库。

因此使用 HCCP 控制面不会给每次 H2D 增加 HCCP 调用开销。批量数、单项大小、batch
迭代次数和 descriptor capacity 仍应是独立运行时配置，不能固化为 4096×1 KiB。

## 官方接口也可能不完整

“官方接口”并不自动等于目标功能完整。KVAF 必须分别验证：

- 当前 CANN 的 `libra.so` 是否导出完整 `RaCtx*`；
- 若只有旧接口，`RaAiQpCreate` 返回结构是否与当前 SHMEM header ABI 匹配；
- `USER_CTL_NORMAL` 模式是否在当前 A5 产品和 caller domain 可用；
- 返回地址是否确实属于目标 NPU 的可访问地址空间；
- TP/import/bind 是否覆盖纯 UB/EID 建链，而不强制依赖 IP socket；
- 每个 destroy/unmap 的返回码和可重试语义是否完备。

固定 HCOMM 快照还存在一个明确的产品分歧：上层 `DEV_USED` 路径会请求 custom SQ，
但当前构建选中的 910_95 后端拒绝“非 CCU + custom SQ”；同仓的 910_96 实现才接受
这一组合。因此，看到 `RaCtxQpCreate` 或 `RaAiQpCreate` 符号不能推导当前安装库支持
AIV user-controlled SQ。必须识别服务器实际产品后端并做不提交 WQE 的模式探针。

HCCP 源码的一些异常清理分支会记录错误后继续回收，或忽略次级清理返回值。KVAF
wrapper 不能照搬这种“best effort”语义：它必须逐对象记录 ownership，严格验证逆序
清理，并在未证明完成时禁止复用同一资源。HCCP 解决的是产品适配和资源映射问题，
不是自动提供 KVAF 所需的端到端生命周期证明。

## 当前门禁

`scripts/preflight_a5_hccp_hal_capabilities.sh` 只读取 ELF 动态符号表；如果运行容器没有
SDK 的 `ascend_hal.h`，会将 raw-HAL 声明面记为 `unavailable`，但仍继续审计
`libra.so` 的两代接口。它不会 `dlopen`、不会调用 HCCP/HAL，也不会操作 NPU。

静态审计结果只能选择下一步适配器，不能授权真机执行。只有以下证据齐全后才会增加
一个资源级、无 WQE 的 HCCP backend probe：

1. 唯一 `libra.so` 路径与 ELF identity；
2. 完整的 modern 或 legacy symbol surface；
3. 与该二进制版本配套的结构体 ABI；
4. Host owner 的逐阶段 rollback 测试；
5. 返回队列地址的只读验证方案。
