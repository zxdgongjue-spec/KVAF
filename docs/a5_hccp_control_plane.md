# A5 HCCP 控制面 + AIV/SIMT 直写数据面

## 决策

KVAF 不再把“在 AICPU 内自行拼装 UMDK 私有资源”作为主路径。新的候选架构是：

- Host 长生命周期 owner 通过驱动随附的 `libra.so` 创建并持有 HCCP 资源；
- 优先使用 `RaCtx*` Context/MR/CQ/QP/TP 接口；若安装版本只提供 SHMEM
  已采用的旧接口，则评估 `RaAiQpCreate`/`RaRegisterMr` 路径；
- HCCP 返回已经映射到 Device 可见地址空间的 SQ、CQ、doorbell 和共享状态地址；
- AIV+SIMT 在 decode 热路径直接生成 WQE、更新 PI/doorbell、轮询 CQ；
- HCCP 不参与每个 1 KiB 请求，也不参与每个 batch 的控制交互。

直接 `libra.so` 仍是最终“不链接 HCOMM”的控制面。为尽快打通首个端到端 H2D，KVAF
同时允许一个降级候选：若安装环境没有完整 `libra.so`，但有版本匹配且资源 API 完整的
`libhcomm.so`，可以只在 epoch 初始化/销毁阶段复用 HCOMM 的 Endpoint/MR/Channel
资源管理。建链完成后仍由 AIV+SIMT 直写 SQ/CQ，HCOMM 不进入每次传输热路径。这个
降级候选不是 collective 路径，也不能仅凭符号齐全就授权真机运行。

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
`libra.so` 的两代接口，并审计现有 `HcommUrmaResources` 实际调用的
`libhcomm.so` Endpoint/MR/Channel 符号。库发现按显式路径、绝对且非空的
`LD_LIBRARY_PATH`、与 Host 架构相符的动态链接器缓存、CANN/Driver 固定布局依次进行，
并报告全部候选。这是针对 name-only `dlopen` 的保守发现顺序，不等价于某个未来可执行
文件的完整 ELF 解析顺序：`DT_RPATH/DT_RUNPATH` 尚未纳入；空/相对 loader path 因会
引入当前目录而被故意忽略。`ldconfig` 项会按 Host 架构过滤，其他显式/固定路径的 ELF
架构仍只标记为未证明。它不会 `dlopen`、不会调用 HCCP/HCOMM/HAL，也不会操作 NPU。

静态审计结果只能选择下一步适配器，不能授权真机执行。只有以下证据齐全后才会增加
一个资源级、无 WQE 的 HCCP backend probe：

1. 唯一 `libra.so` 路径与 ELF identity；
2. 完整的 modern 或 legacy symbol surface；
3. 与该二进制版本配套的结构体 ABI；
4. Host owner 的逐阶段 rollback 测试；
5. 返回队列地址的只读验证方案。

### 当前执行的第一阶段：公开 HCOMM resource ABI → HCCP

当前服务器的 `libra.so` 已被发现，且 `RaCtx*` 符号完整；但 CANN 安装包没有提供
`CtxInitCfg`、`CtxInitAttr`、`QpCreateInfo` 等 `RaCtx*` 参数/返回结构的公开头文件。
这些定义只存在于 HCCP 源码树的内部 include 路径。符号存在不能证明结构体 layout、枚举值、
padding 和 Device service package 与源码一致，因此 KVAF **不会**把源码私有头文件直接编进
一个调用已安装 `libra.so` 的真机程序。

第一条可执行阶段改为
`scripts/run_a5_hccp_resource_lifecycle_probe.sh`。它使用当前安装 CANN 已提供、可用
`hcomm_res.h` 编译校验的 resource C API 作为 HCCP bootstrap，按下面的有界流程执行：

```text
静态 library/header/symbol 审计
  -> npu-smi 明确证明选定 NPU idle
  -> 串行解析一个 Host/Device EID pair
  -> host-ddr: HAL 分配+映射 DDR -> Endpoint -> MR register -> MR export
  -> HCOMM/HCCP 资源 Close（重试） -> HAL unregister -> 释放 DDR
```

`device-hbm` 是第二个、独立的资源阶段：分配 Device HBM、Endpoint、MR register、
resource Close、释放 HBM。两阶段都不会创建 channel/Jetty/SQ/CQ，不会提交 WQE，也不会运行
AIV kernel。每个阶段必须输出 `KVAF_A5_HCCP_RESOURCE_LIFECYCLE_OK` 才允许进入下一个阶段。

清理顺序是资源先于内存。若 `HcommMemUnreg` / endpoint destroy 无法在有限重试内证明成功，
程序刻意保留可能仍被 DMA 引用的 mapping/backing，并输出
`KVAF_A5_HCCP_RESOURCE_TEARDOWN_UNPROVEN`；它不会把 Host DDR/HBM 提前释放后继续测试。
这不是 normal retry，后续 channel/WQE 阶段必须停止并保留结果目录用于诊断。

只有资源和 backing 均已确认回收后，probe 才调用 `aclrtResetDevice` 释放**本进程**的
ACL logical-device context，再调用 `aclFinalize`。这不是 `npu-smi set -t reset` 的物理卡
复位；若 cleanup 未证明，二者均不会调用，避免 runtime teardown 抢先释放仍可能被 DMA
引用的内存。

因此当前 “使用 HCOMM” 的边界非常严格：它只替代没有公开 ABI 的 `RaCtx*` 调用，实际
调用的仍是安装 CANN 配套 HCCP service；它绝不进入 AIV/SIMT 每次 decode H2D 的热路径。
资源阶段通过后，下一步会增加两方 AIV channel 的无 WQE create/ready/snapshot/close，最后
才打开可配置 descriptor batch 的 AIV+SIMT READ 数据面。

## 安装布局与控制面选择

HCCP 源码中的 Host client 目标名为 `ra`，但其安装位置不在
Driver 固定目录，而是 CANN 的 `${arch}-linux/lib64/libra.so`。因此，
仅检查 `/usr/local/Ascend/driver/lib64*` 会把已安装的 Host client
误报为缺失。静态预检现在按确定性的发现顺序审计显式路径、
安全的 `LD_LIBRARY_PATH` 项、同架构 `ldconfig` cache、CANN 布局和
Driver 布局，并报告所有额外同名库。该步骤仍只读 ELF，不会 `dlopen`，
也不会把该顺序宣称为已证明的运行时解析结果。

控制面候选顺序是：

1. 与当前 CANN/Driver/Device package 严格配套的 `libra.so` `RaCtx*`；
2. 同一 `libra.so` 的完整 legacy AI-QP 接口；
3. 已安装 `libhcomm.so` 的 resource C API，仅作 Endpoint/MR/Channel
   初始化 bootstrap，建链后仍由 AIV+SIMT 直写 SQ/CQ；
4. 前三者均不完整时停止，不再回退到猜测私有 HAL/UMDK
   参数。

第 3 项不是把 HCOMM 放回 decode 热路径。它只在 session/epoch
初始化和销毁阶段执行；重复的 4096×1 KiB 或其他可配置 batch
不调用 HCOMM/HCCP。这是当前最快打通端到端 H2D、同时保留
后续替换为直接 HCCP client 的分层方案。

## `halMemBindSibling` 的适用边界

`halMemBindSibling` 不是 Host `libra.so` client 或 Host raw-map 的通用
前置条件。它只在 HCCP Device/offline `CUSTOM_INTERFACE` 共享池
分支中用于 Host PID 和 AICPU PID 的 sibling bind。Host 侧将
JFC/Jetty/SQ/CQ/doorbell 映射到 `PROCESS_CP1` 主要依赖
`halResAddrMapV2/halResAddrUnmapV2`。

HCOMM 9.1 对 `halMemBindSibling` 只做精确名称的 `dlsym`，没有
alias 或版本回退。如果 Device/offline 分支进入后仍无法解析/调用，
源码会每 10 ms 重试，最多 12000 次，即约 120 s。预检必须
将这项标记为 Device/offline capability 和长时延风险，不能再用它
否定 Host client 候选；但对实际进入 UDMA Device service share-pool
初始化的 backend，它仍是阻断性依赖。Host `libascend_hal.so` 本体未导出
该符号也不能证明 Device service 环境必然缺失它，因为依赖 DSO 的解析闭包
和设备侧运行环境仍需单独核对。

## 为什么不单独自行编译 `libra.so`

HCCP 的 A5 直写队列路径是 `NETWORK_OFFLINE/HDC`：Host RA 通过
HDC 请求 Device `hccp_service.bin`，由设备侧与 Driver 配套的
UMDK/HAL 创建并映射 JFC/Jetty，再返回 Device-visible 队列地址。
HCOMM 9.1 的 `ra` target 还依赖 `ra_hdc`、`ra_peer`、`librs`、
TSD/network service 以及设备侧 compat package，不是可独立部署的最小
client SO。

所以，不应在共享服务器上用当前源码单独编译一个
`libra.so` 去搭配不透明的现有 Device package。首选是发现并审计
已安装的同版本 Host 库；若确实缺失，要么使用已安装
`libhcomm.so` 的资源 C API，要么取得官方同源 Host package。只有
能同时替换 Host 和 Device HCCP package 时，才考虑 exact-commit 的完整
HCOMM build。

若 direct `libra.so` 不完整而静态审计把 `libhcomm-resource-api` 列为静态符号候选，
仍需额外核对
HCOMM header/runtime 版本、AIV ChannelEntity ABI 和可重试逆序销毁；
`KVAF_HCCP_BACKEND_READY` 在这些验证完成前保持 `no`。
