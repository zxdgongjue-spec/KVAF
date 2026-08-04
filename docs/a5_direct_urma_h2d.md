# A5 AIV + SIMT + direct URMA + H2D

## 1. 结论

当前代码中保留了一条不调用、不链接 HCOMM 的候选 H2D
路径，但它只是“已编译的架构候选”，不是“已验证的可执行数据面”。
当前门禁状态为：

```text
HARDWARE_EXECUTION_DISABLED
ZERO_WQE_NOT_PROVEN
DYNAMIC_RUNTIME_PROBE_DISABLED
```

这三项状态的直接含义是：不再执行 zero-WQE、不再执行真实 H2D，
也不再启动会进入 AICPU 官方 `liburma.so.0` 的动态 runtime probe。
direct H2D runner 及其 candidate wrapper 会在拓扑、
EID、ACL、DDR/HBM 注册和 URMA 资源创建之前固定退出 2；
runtime probe 的三个动态模式同样固定退出 2。不提供环境变量绕过。

候选流程仅用于说明代码意图：

```text
Host DDR(HVA)
  -> halHostRegister(HOST_MEM_MAP_DEV)
  -> Host DVA
  -> Host liburma 注册 MR、创建 Jetty/TP
  -> URMA RM 管理 Jetty 交换 EID、MR、Jetty、TP
  -> AICPU liburma 注册 HBM MR、创建 Jetty/SQ/CQ
  -> 返回 AICPU shared device-VA 的 SQ/CQ/doorbell/TP/MR
  -> 32-lane SIMT 规划 descriptor
  -> 单 AIV 直写 64B UDMA READ SQE
  -> SQ doorbell
  -> DCCI 后轮询 CQE，并更新 CQ doorbell
  -> Host DDR 数据进入 Device HBM
```

官方 URMA 能力必须分层判定，不能由一个符号或一次 SUCCESS 返回推导
整条链路成立：

1. **符号层**：目标函数能否从实际 DSO 解析；
2. **返回码层**：入口是否接受当前 mode、token 和 endpoint；
3. **语义层**：SUCCESS 是否真正完成创建、advise、notify 或销毁，而不是
   no-op 或只记录内部错误；
4. **调用域层**：Host Linux userspace 入口是否有明文支持 AICPU/Device OS；
5. **生命周期层**：部分创建、并发 init、逆序 cleanup 和错误传播是否完整；
6. **AIV 私有队列 ABI 层**：SQ/CQ/doorbell 地址域、WQE/CQE 格式、owner bit、
   barrier 和 cache 语义是否与 A5 一致。

当前六层尚未形成可机器复核的完整证据链，因此 zero-WQE 也不能当作
“无数据的安全预检”。更完整的缺口和判定标准见
[A5 官方 URMA 接口能力边界与 KVAF 门禁](a5_official_urma_capability_audit.md)。

资源控制面固定来自 openEuler UMDK 26.06 源码：普通
`urma_create_jetty` 在 AICPU KFC 进程内执行，AIV 消费同一 unified
device-VA。代码不会把这个执行模型误标成 `CREATE_JETTY_EX/TGID-VA`；
最终可达性由真实的一条 SQE、一个 CQE、payload 和 canary 联合验证。

直接依赖中已经移除：

- memcache；
- memfabric；
- HCOMM headers、`libhcomm.so` 和 HCOMM resource API；
- 独立的旧 perftest/benchmark 源码树及其预编译参考 SO；
- 系统 `liburma.so`、`liburma_common.so`、`liburma-udma.so` 和
  `libummu.so`。

仍然需要：

- CANN/ACL/runtime、bisheng、Ascend HAL 和驱动；
- AICPU KFC 动态加载；
- clean Git 状态的 openEuler UMDK
  `12917421beb93c1351eb2f124bb2f34612d8bf2d`；
- 该 UMDK 在 `src/urma/bazel/urma_deps.bzl` 中锁定的 openEuler libummu
  `70de020f776d8def0d8a8941cc7cbc5928d66c10`；
- 当前运行内核对应的完整 UDMA 与 UMMU UAPI header。
- Host 与 `vllm-ascend-cs-kvaf` 内同路径可读写的 UMMU TID 字符设备
  `/dev/ummu/tid`；Host DDR 的 HAL 注册不能替代它。

Host 与 Device provider 都把固定版本 UMDK 的 common、完整 public URMA
core、UDMA 和官方 perftest UB/TCP 管理代码，以及配对 libummu 编进自身；private
core 在编译期禁用动态 provider 目录扫描。两端 UDMA 源码不再通过 `ST64B` 或
`TGID_VA` 编译宏伪造不同布局。private accessor 区分 Host 和 AICPU 执行域；
此外，Device 角色只为管理面 JFR 的小 RQ 选择固定 UMDK 已支持的
`UDMA_MMAP_KERNEL_BUF`，不改变任何 private struct 布局。构建与运行脚本同时
检查 `DT_NEEDED` 和 `ldd`，发现外部
`liburma`、`liburma_common`、`liburma-udma` 或 `libummu` 会直接失败。

这里的“绕过 HCOMM”是指 KVAF 自己完成资源控制。EID、MR、Jetty、TP
仍必须创建和交换，只是不再由 HCOMM/memfabric/memcache 代办。

## 2. 代码来源与职责

### 2.1 官方 UMDK 26.06

provider 固定从 UMDK commit
`12917421beb93c1351eb2f124bb2f34612d8bf2d` 源码编译：

- `src/urma/common`；
- `src/urma/lib/urma/core` 的完整默认 core，包括 `urma_perf.c` 和
  `ubagg_stats.c`；
- `src/urma/hw/udma`；
- `src/urma/tools/urma_perftest` 的资源管理、参数校验、日志，以及官方
  `perftest_mgmt_ub.c` 和 `perftest_mgmt_tcp.c` 后端；构建清单同时固定
  `perftest_mgmt.c/.h` 调度器来源，但 direct provider 不编入其 executable-level
  dispatcher。默认使用 URMA/UB 管理通道；TCP 仅保留为显式回退。

构建脚本要求两棵源码树的 `origin` 都是官方 GitCode 地址、工作树无修改，校验
精确 commit/tree，并在构建前后重新计算
完整 source manifest。它不链接预编译 perftest SO，也不使用系统 URMA 动态库
作为“参考 ABI”。UMDK common 和 core 直接编进 provider，动态 provider 扫描在
编译期关闭。build manifest 还记录构建时 `uname -r`；打包和运行前必须与当前
内核完全一致，不能把旧内核生成的 provider 搬到新内核继续使用。

### 2.2 UMDK 锁定的 libummu

UMDK 的依赖文件锁定 libummu commit
`70de020f776d8def0d8a8941cc7cbc5928d66c10`、version `1.0.5`、ABI
version `1`。构建脚本从依赖文件唯一解析这些值，要求 libummu 工作树 clean、
HEAD 精确匹配，并把其源码和 UAPI 直接编进两个 provider。手工指定另一份
libummu、只替换头文件或回退系统 `libummu.so` 都会 fail closed。

userspace 源码配对还不等于运行内核配对。构建必须找到当前 `uname -r` 对应的
UDMA `udma_abi.h` 和 UMMU `ummu_core.h`，并分别与 UMDK/libummu 携带的
header 执行逐字节比较。任一 header 缺失、不相同或构建期间发生变化，构建立即
失败。

### 2.3 AIV/SIMT 数据面

设备数据面遵循当前 UMDK/UDMA UAPI 和 A5 硬件格式：

- 64B READ SQE；
- remote EID 的硬件字节序；
- 24-bit TP ID；
- public `token_id >> 8` 得到 20-bit UDMA TID；
- READ local SGE 的 token word 保持 0，使用已注册 local VA；
- SQ doorbell offset；
- CQ owner/phase、DCCI 和 CQ consumer doorbell。

KVAF 增加了 descriptor 越界检查、单 owner、stream 同步、结果/canary
验证和失败隔离。

### 2.4 Host DDR

Host 端严格区分两个地址：

- HVA：CPU 填充和校验 DDR；
- DVA：`halHostRegister(..., HOST_MEM_MAP_DEV, ...)` 返回，交给 URMA 注册。

生命周期必须满足：

```text
分配 DDR
  -> Host 注册得到 DVA
  -> URMA register/import
  -> 数据传输和 CQ 完成
  -> 两端 disconnect/unimport
  -> 两端确认
  -> unregister MR/token
  -> 删除 Jetty/JFC/context
  -> Host unregister DVA
  -> 释放 DDR
```

在不能证明 DMA 已停止时，代码不会主动释放 backing memory。明确的
protocol/ownership ambiguity 才写 quarantine；底层明确返回 RETRYABLE 时，
原 worker 会保留同一 session、原句柄和 backing，并持续从 teardown 断点重试。
UMMU 的每个成功 `mmap` 都在返回前挂入预分配 ownership ledger，因而
`munmap` 失败不需要再分配内存就能保留精确重试地址。JFR/Jetty table unlock、
UMMU 命令完成状态或 ledger 所有权一旦无法证明，会触发进程生命周期的单向
`cleanup_unknown` 锁存；后续普通可重试错误不能清除它，Close 必须返回
`CLOSE_UNSAFE`，避免把真实未知状态误报成可重试。

UB 管理通道的三个 4 KiB Segment 分别显式持有 Token/TID。Close 严格按
`unregister Segment -> free Token` 的配对顺序逆序释放；若 Segment 注销失败，
Token handle 保留给同一 owner 重试，不使用 public `urma_register_seg()` 会在
失败清理中提前释放隐式 Token 的便利路径。

## 3. 为什么要分成 Host/Device 两个 provider

标准 `urma_query_*` API 不返回 userspace SQ/CQ 的 `qbuf`、硬件
doorbell 或 JFC `sw_db`。`urma_user_ctl(CREATE_JETTY_EX)` 只能解决新建
SQ 的部分信息，也不能查询普通 JFC 的完整 CQ，因此当前 AIV direct
路径仍需要一个受控的 UDMA 私有访问层。

两端使用固定 UMDK commit 中同一套 public/private UDMA 定义，但生成执行域
不同的 provider：

| 端 | provider private ABI | KVAF 产物 |
|---|---|---|
| Host | `official-umdk-udma-v1-host` | `libkvaf_urma_provider_host.so` |
| AICPU/Device | `official-umdk-udma-v1-device` | `libkvaf_urma_provider_device.so` |

不能用其他 UMDK tag、驱动 SDK 内的另一代 private header 或单独复制的 struct
替代它：public JFC/Jetty base 与 queue tail 发生变化时，会让 SQ、CQ、`sw_db`
全部错位。
`CREATE_JETTY_EX(tg_id)` 适用于其他进程代建并把 doorbell 映射到目标
TGID 的路径；本实现的资源本来就在 AICPU 内创建，强行改用它反而偏离
当前执行模型。

KVAF 把 private header 限制在 private accessor 和 source-matched UDMA
适配层，并为 Host/Device 角色分别编译。角色宏选择 accessor 执行域，并只在
Device 管理 JFR 的一次 RQ 分配中选择普通 kernel-buffer；它不定义
`ST64B`/`TGID_VA`，也不修改共享 context 的 `hugepage_enable`。
它从真实 queue 对象导出实际 `qbuf_size/baseblk_cnt`，并在返回任何地址前
校验：

- AArch64 public JFC/Jetty、private queue/context 的固定大小和关键偏移；
- SQ/CQ ring 非空、2 的幂、64B stride；
- qbuf 边界和 baseblock 数一致；
- PI/CI 初始值为 0；
- SQ 非 reserved 模式；
- CQ 软件 doorbell 非空；
- provider config 所声明的 Host/Device ABI 与 SO 一致。

构建阶段还会实际 `dlopen` 两个 SO，调用
`kvaf_urma_provider_private_abi_v2()` 证明角色，不能只靠 manifest 文本
声称 Host/Device 身份。

运行脚本仍拒绝 `sq_reserved=1`，因为该模式的设备地址语义还没有在 A5
上验证。

## 4. 资源创建与字段来源

两端 provider 执行：

```text
parse/check config
  -> establish URMA RM management Jetty
  -> 1-byte client handshake（server 从 CQE 学到 client EID/Jetty ID）
  -> create context
  -> create JFC/Jetty
  -> register local segment
  -> exchange EID/MR/Jetty/TP
  -> import peer segment/Jetty
  -> connect
  -> validate exact private queue ABI
  -> provider-ready 双端确认
```

默认 `KVAF_DIRECT_MANAGEMENT_MODE=ub`。Host 用自身 EID 创建固定的 RM+CTP 管理
Jetty 并预投 RECV；Device/AICPU 只需要 Host EID 和约定的 management Jetty ID
即可建链并发送握手。固定 UMDK 的旧 `urma_import_jetty()` 兼容路径会向 UDMA
传入全零 active-TP 配置，因此 Direct provider 不再调用它：Client 先以
`local EID + peer EID + RM + CTP` 调用 `urma_get_tp_list()`，严格取得一个
`tp_handle` 后再调用 `urma_import_jetty_ex()`；Server 收到握手 CQE、从硬件填充的
`remote_id` 得到 Client EID/Jetty ID 后执行相同流程。CTP 不需要交换 RTP 的
peer TP handle/PSN。随后双方才用 URMA SEND/RECV 交换正式 data MR、Jetty 和 TP
元数据。

这里借用了 HCCP 已验证的“先解析 TP、再扩展导入”生命周期，但没有引入 HCCP、
HCOMM、`libra.so` 或额外 socket。管理与 data JFS 都设置
`error_suspend=1`：第一个传输/内存错误会停止继续取 SQE，避免软件队列游标在错误
后继续前进并把可回收失败扩大成 reset-only 状态。

data Jetty 同样不再调用固定 UMDK 中的通用 `connect_jetty()`：Direct MVP 在 Segment
和 Jetty ID 两次固定交换完成后单独执行 `urma_get_tp_list()`，用 phase frame 同步其
结果，再以 `urma_import_jetty_ex()` 导入。成功返回的 Target Jetty 立即保存在 checked
context 中；后续任何一端 import/phase 失败都不在 create 路径做 best-effort unimport，
而是交给可重试 teardown。这样即使 unimport 暂时失败，也不会丢失句柄或只能复位。
该路径不创建 IP socket，不读取 `/proc/net/tcp{,6}`，也不依赖 Linux IPv4、IPv6、
路由或 `ipourma*` 地址配置。EID 即使显示成 IPv6 风格，也只是 128-bit URMA 标识。

`KVAF_DIRECT_MANAGEMENT_JETTY_ID` 必须是双方一致的 well-known ID。runner 默认按
`500 + physical_device_id` 推导，因此 NPU0..7 分别使用 500..507，并限制在保留范围
1..1023；0、历史 TCP 端口 21115 和范围外值会在创建资源前被拒绝。runner 启动 Host
后等待日志中的精确证明
`KVAF_URMA_MGMT_READY role=server jetty_id=... local_eid=...`，再启动 Device，不能用
固定 sleep 猜测。

调试旧环境时可显式设置 `KVAF_DIRECT_MANAGEMENT_MODE=tcp`。只有该模式才解析
`KVAF_DIRECT_CONTROL_SERVER_ADDRESS`、检查 TCP LISTEN socket，并向 worker 传入
`--control-*` 参数；它不是默认路径，也不是 UB 模式的隐式依赖。

AIV 使用的字段来源如下：

| 字段 | 来源 |
|---|---|
| SQ base/depth/doorbell | 固定 UMDK Device private accessor |
| CQ base/depth/sw_db | 固定 UMDK Device private accessor |
| remote MR | `remote_seg[0].ubva/len` |
| remote TID | remote public segment `token_id >> 8` |
| local object ID | local public segment `token_id >> 8`，仅用于资源校验，不写入 compact READ WQE |
| remote EID | imported target Jetty public EID，AIV 只反转一次 |
| TP ID | imported target Jetty 的 `tp.tpn` |
| token policy | `URMA_TOKEN_NONE`，SQE `tokenEn=0` |

不能使用旧 sidecar 的 `ai_jettys[].sq.tp_id`。当前 UDMA provider 没有实现
`urma_ops.get_tpn`，该值可能是负错误码转换后的无符号数。

## 5. 失败与 teardown

创建过程按阶段互相确认。管理握手和 CQ polling 都受同一个有界 provider deadline
约束，一端失败会唤醒另一端并进入 checked cleanup。正常关闭顺序是：

```text
AIV stream 完成
  -> SQ PI == CQ CI
  -> 两端 disconnect/unimport Jetty
  -> barrier
  -> 两端 unimport peer segment
  -> barrier
  -> unregister local segment/token
  -> 删除 data Jetty/JFC/context（按 owner context 校验，管理 context 仍存活）
  -> 删除 management target Jetty/local Jetty/JFR/segment/JFC/context
  -> 确认全局 live-resource 计数为 0
  -> urma_uninit
```

管理 Open 失败时，AICPU 还会在固定 256-byte result ABI 的保留字中返回首个叶子
错误，包括 get-TP、extended import、SEND 或 CQ 阶段、raw/effective errno 以及
URMA/CQE 状态。该记录在 checked Close 之前快照，后续外层 handshake 错误和 teardown
不会覆盖它；因此 Zero-WQE 再失败时可以直接区分路由/TP、Jetty import 和队列完成
问题，而不再只有通用的 `provider_status=-5`。

管理 context 必须留到 data context 完成双端 teardown 协议之后；否则 barrier 本身
会被提前拆掉。反过来，删除 data context 时也不能要求进程内所有 URMA 资源都为零，
因为此时管理 context 合法存活。因此 provider 的 context gate 只检查“该 context
拥有的资源”，最后关闭管理 context 后才执行全局零资源检查与 runtime uninit。

每次运行生成独立 `run_id`。`device.close_ready`、`device.done` 和两端最终
marker 都必须携带同一 nonce，旧目录中的 marker 不能触发本次释放。两端
分别原子写入 `exchange/host.final` 和 `exchange/device.final`；runner 用
严格解析器拒绝重复字段、软链接、错误角色、错误 nonce 和缺失的安全证明。

端点终态分为三类，runner 另外保留一个“证据暂缺”的可恢复状态：

退出码 2 专用于启动前的 fail-closed 预检失败，例如目标 NPU 忙、`npu-smi`
不可用/输出无法严格解析、构建来源不一致或端口已占用。此时 Host/Device 尚未
创建 DDR、HBM、URMA、SQ 或 CQ 资源，不需要 reset，也不会生成 quarantine。

| 终态 | 退出码 | 是否写 quarantine | 含义 |
|---|---:|---|---|
| `PASS` | 0 | 否 | 传输、校验和 teardown 全部通过 |
| `FAIL_SAFE` | 3 | 否 | 功能失败，但 stream/queue/provider 已关闭，Host 映射已注销；普通本地内存或句柄即使清理失败也不再可被 DMA，可直接修复后重试 |
| `FAIL_UNSAFE` | 4 | 是 | 同步、Close 或 backing 状态无法证明，必须保留进程和内存 |
| `PENDING_UNPROVEN` | 5 | 否，保留 active | worker 尚未退出、marker 尚未到齐或分类器暂时无法完成；不等同于不安全，不复位、不杀进程，稍后重跑脚本会自动复核 |

分类器内部用退出码 6 表示 `UNPROVEN`（缺失、损坏、错误 nonce 或证明不完整）；
runner 将它收敛为上表的退出码 5。它不会再把“证据不可解析”冒充成端点已经明确
发布的 `FAIL_UNSAFE`。反过来，只要任一端存在合法、nonce 一致的
`FAIL_UNSAFE final`，即使另一端 marker 或短 `fatal.*` 写失败，也会可靠进入
quarantine。

runner 收到 `Ctrl+C`（`INT`）时，若还没启动资源则安全退出；若 Host 已经
启动且仍在运行，则先保证 Device 也进入受记录的 pair 状态，再以 130 脱离监督
并保留 active。这样中断不会再次制造“只知道一个 PID”的半启动会话。

因此，payload/canary 校验失败但双方 provider Close、Host unregister 和 HBM
释放都成功时，不再误写 quarantine。只有两个进程的最终证明和正向证据都
精确通过时才打印：

```text
KVAF_A5_AIV_SIMT_DIRECT_URMA_H2D_ACCEPTANCE=PASS
```

Device 在发起 AIV DMA 前失败、或在同步完成后安全回收但无法继续协议时，会先
原子写入本次 `run_id` 对应的 `device.final=FAIL_SAFE`，再发布
`device.safe_abort`。Host 以完整 `device.final` 的 stream/queue/provider 安全字段
为权威证明，短 marker 只用于加速唤醒、写失败也不影响最终恢复。Host 的等待周期
到期只打印 pending 日志并继续持有资源；晚到的 nonce-bound 证明仍能触发正常
Close/unregister。Device 写 `close_ready` 失败时同样保留所有权并持续原子重试，
不会一端进入 Close、另一端永久等待文件而形成 teardown 死锁。

只有端点发布可严格解析的 `FAIL_UNSAFE final`（`fatal.host/fatal.device` 只是
nonce 校验后的早醒信号），或使用无法验证 teardown 的系统 UDMA 且本次已经
armed，才会生成：

```text
/data/.kvaf-state/KVAF/direct-urma.quarantine
```

普通超时、marker 暂缺、分类器损坏或 runner 中断只保留
`PENDING_UNPROVEN` active，不会提前声称必须复位。provider 明确返回
`CLOSE_RETRYABLE` 时也不会因为经过若干秒就升级成 unsafe：worker 先高频、
随后降频无限重试同一个 provider instance/session，runner 的有界等待到期后只
返回 5 并保留 live worker 和 active-session 锁。只有 stream 无法证明停止、
queue 不一致、跨 session/worker ownership 冲突、PLBI/PSYNC 状态未知，或 provider
明确返回 `CLOSE_UNSAFE` 时，端点才发布 fatal/`FAIL_UNSAFE` 并进入 quarantine。

worker 在可能取得 DDR/HBM 前忽略异步 HUP/INT/TERM/QUIT。部署兼容锁与
active-session 锁由 worker 继承；runner 在启动 Host 前写入
`/data/.kvaf-state/KVAF/direct-urma.active`，只有严格分类器证明
清理安全后才删除。active/quarantine 使用
`fsync(临时文件) -> rename -> fsync(状态目录)`，quarantine 持久化成功后才
允许删除 active。结果 manifest、固化分类器和构建来源也在第一个 launch intent
前落盘。

资源持有期的 C++ 异常由组合 owner 统一接管：provider 未证明关闭之前，独立的
Host mapping/HBM allocation 析构不会先释放 backing。若异常处理本身再次 OOM，
预装的 terminate handler 只用固定缓冲区写诊断并保持进程存活。AICPU provider
stream 的同步错误会在不触碰 result/backing 的情况下对同一 stream 重试一次；
第二次仍失败才进入 unsafe。

状态和结果使用与 checkout 无关、宿主机与 Docker 共享的机器级根
`/data/.kvaf-state`。因此 runner 被 `SIGKILL`、容器被删除或重建、
`git clean` 或仓库重拉之后，持久化证据仍会阻止第二个 singleton provider
会话并支持自动恢复。不要手工删除 `/data/.kvaf-state` 中的锁、active、
quarantine 或 active 引用的结果目录；`flock` 绑定 inode，删除锁文件会制造第二把
看似同名但实际不同的锁。

生产 runner 的 active/quarantine 是整机 KVAF singleton，不按 NPU 分片。一个未解决的
session 会阻止换另一张卡继续测试，这是有意的 fail-closed 约束；生产路径不提供
quarantine bypass，也不接受通过复制仓库、删除锁或改状态目录绕过它。

### 5.1 历史方案：受限的 NPU7 -> NPU2 风险接受实验（当前禁用）

> 本节只保留事故处置设计和旧证据的可审计性。下文出现的
> `zero-wqe`、`one-kib`、cross-NPU 和 detached launcher 命令名都不是当前
> 操作指南。全局 `HARDWARE_EXECUTION_DISABLED` 门禁不允许它们创建
> URMA/UMMU 资源，也没有风险接受开关可以解锁。

历史版本曾有一个范围刻意收窄的真机诊断模式
`risk-accepted-cross-npu2-v1`。它不是生产恢复机制，也不是对“不同 NPU 一定彼此
隔离”的声明。它曾用于以下具体情况：NPU7 已因一次旧版 zero-WQE 测试进入
quarantine，现场暂时不能重启，但需要验证修复后的代码能否在 NPU2 完成最小资源
生命周期和一条真实 H2D。

这个模式不会把 NPU7 判定为安全。启动前，read-only 验证器必须把操作员提供的
quarantine SHA-256 和 `run_id` 精确绑定到已知的 NPU7 事件，并复核源 commit、
zero-WQE 工作负载、最终失败原因、Host/Device PID、进程起始时刻、可执行文件和
命令行。相同验证会在 Host fork 前和 Device fork 前再次执行，防止验证后源状态或
进程身份发生变化。任何字段缺失、重复、变更或无法读取都会在目标资源创建前
fail closed。

历史实现唯一接受的源事件是 commit
`ed13d76f4a5a9c89b91c0524b2a4f53da2e3a874`、`h2d_transfer_count=0`，且
Device final 为 `FAIL_UNSAFE/device_provider_open_cleanup_failed` 的那一次运行。
模式不会接受“看起来相似”的另一份 quarantine，也不会把后续新故障加入白名单。

源与目标范围固定如下，不能通过环境变量扩展：

| 项目 | 源事件（只读保留） | 本次实验目标 |
|---|---|---|
| NPU | logical/physical NPU7 | logical/physical NPU2 |
| Host URMA | `udmac0d1e5/eid0` | `udmac1d1e4/eid0` |
| Device URMA | `udmac0d0e2/eid0` | `udmac0d0e2/eid0` |
| Host EID | `00000000003f050000100000df080b00` | `00000000007f040000100000df0a0b00` |
| Device EID | `000000000000020000100000df00e100` | `000000000000020000100000df004100` |
| management Jetty | 507 | 502 |

两侧 Device inventory 都可能显示 `udmac0d0e2`；该字符串来自各自所选 NPU 的
`hccn_tool -i <physical-device-id>`/AICPU URMA 可见域，不是宿主机范围内唯一的
物理卡编号。绑定身份必须同时包含 logical/physical device、精确 EID 和 EID index。
因此 NPU2 不能沿用先前按 `/dev/uburma` 名字猜出的 `udmac0d1e4`，runner 会以
NPU2 的现场 EID 表复核 `udmac0d0e2/eid0` 后才允许创建 context。这仍不构成 UMMU
跨 NPU 隔离证明。

风险入口不会向 runner 预填四个 URMA name/index selector；它固定 NPU2 和两条
EID 后清除调用者的 selector override，由 exact-EID resolver 现场选出端点。上表的
`udmac0d0e2/eid0` 是本次现场结果以及风险契约的 fail-closed 复核值，不是从 Host
`/dev/uburma` 推导出的默认值。

NPU7 的全局 quarantine、结果目录、Host/Device 进程和 backing 必须原样保留；实验
不能删除、移动、覆盖或向它们发送信号。NPU2 使用独立的实验 active、quarantine、
result 和锁命名空间，但仍持有跨 NPU 实验互斥锁，避免第二个风险接受会话并发启动。
NPU2 成功只说明该次 NPU2 证据闭环，不能清理或降级原 NPU7 quarantine。之后的
生产 runner 也仍会被全局 quarantine 拒绝。

cross-NPU2 tiny-stage 的默认 worker supervision deadline 为 180 秒。它保持
Device/provider control deadline 120 秒、Host ready deadline 30 秒以及派生的
Host control deadline 155 秒，并留出 25 秒终态收敛余量。该 deadline 只让 runner
更早返回 `PENDING_UNPROVEN`；不会取消 AICPU owner、发送信号或把未证明资源当作
已经销毁。

这里的剩余风险来自整机共享层：不同 UDMA device/EID/Jetty 并不能证明 UMMU 的
PASID/TID、驱动内部 ownership 或 AICPU worker/SO 状态按 NPU 完全隔离。因此这个
模式可能仍让 NPU2 或整机进入需要重启才能恢复的状态。它的价值是用最小增量实验
验证隔离假设，不是把这个假设当成既有保证。

实验只允许按以下两阶段推进：

1. `zero-WQE`：创建并交换资源，但提交 0 条数据 WQE；只有双端精确 `PASS`、全部
   teardown 字段为 1、进程退出且实验 active 已清理后，才能进入下一阶段。
2. `one-kib`：只提交 1 条 1024-byte H2D READ，并校验 payload、canary、CQ 和完整
   teardown。该受限入口不允许直接运行 4096 x 1 KiB 或自定义更大负载。

历史实验会先以 `cross-npu2-risk-offline-v1` 模式构建一次性 direct binary，并放在
`/data/.kvaf-state/KVAF/experimental/artifacts/` 下生成一次性的离线 direct binary；
它不会重建 provider，也不会改写 CANN/AICPU package。两个硬件阶段曾由
cross-NPU2 risk wrapper 统一进入；当前不再给出 wrapper 的调用语法。
`one-kib` 还必须提供前一轮 zero-WQE 的结果目录，入口会复核双端 final、清理位、
日志 marker、源 quarantine 摘要以及 artifact/commit 绑定后才提交一条 WQE。
更上层的 recovery coordinator 曾从固定全局
quarantine 派生精确摘要，生成一次性 artifact ID，并在 zero-WQE PASS 后写入只读
handoff，避免人工复制 EID、路径或结果目录。

出现以下任一情况必须立即停止，不得进入下一阶段：

- 源 quarantine 缺失、SHA-256 改变、`run_id` 不一致，或 NPU7 任一固定进程身份
  不再与验证证据一致；
- NPU2 忙、固定 EID/URMA device/management Jetty 不匹配，或预检不能证明目标
  端点可用；
- ACL/AICPU 返回 `507018`，出现 PLBI/PSYNC 超时，或 UMMU/PASID/TID ownership
  进入 unknown；
- 任一 final 缺失、损坏或 nonce 不匹配，worker 卡住，或
  `stream_quiesced`、`queue_drained`、`provider_closed`、`backing_released`、
  `allocations_released` 中任一项不是 1；Host 端还必须证明
  `host_mapping_unregistered=1`。

硬停止后不要 `kill -9` worker，也不要删除源或实验 quarantine/active。保留进程、
backing 和结果证据，按现场隔离或整机重启流程恢复。即使两个阶段都成功，仍应把
“跨 NPU 共享状态是否隔离”保留为未证明项；只有驱动/UMMU/AICPU 契约或额外故障
注入证据补齐后，才可以讨论将状态从整机 singleton 改成正式的 per-NPU 分片。

#### 5.1.1 第一次 NPU2 也未闭环时的单次 reuse 实验

第一次 `risk-accepted-cross-npu2-v1` 已经在 NPU2 留下
`PENDING_UNPROVEN`，不能把它当作成功经验直接重跑。现场不能重启时，仅保留一个
更窄的第二代诊断入口。它只允许 zero-WQE，并曾要求一个现已撤销、本文不再列出的
风险接受短语，同时逐字节固定原 NPU7
quarantine 和第一次 NPU2 active/result/PID-start-ticks 证据；任何一项变化都会在
新资源创建前停止。

该入口只允许 NPU2 zero-WQE，不允许 `one-kib` 或 batch。每次使用一个 16 位小写
十六进制 token，构建独立 provider、短于 AICPU ABI 上限的唯一 Device SO/SONAME、
增量 AICPU 包和 direct binary；不会覆盖 canonical provider/package。资源创建前还会
在 NPU2 严格 idle 的条件下，用唯一 SO 运行一次 bounded、resource-free owner
probe。probe 只验证该包可被 AICPU loader 加载以及 AIV/AICPU mailbox 可见性，不会
创建 URMA/UMMU context、MR、Jetty、SQ 或 CQ。

为防止硬件探针失败后不断换 token 重试，离线构建、文件校验和 NPU2 idle 校验通过后，
在第一个 AICPU 硬件探针前会原子发布只读 one-shot claim：
`/data/.kvaf-state/KVAF/experimental/npu-2-reuse/530f5998a0b646d48db858170a9996ff/30ef7336d212459a9317fd6647f6fa15/attempt.selected`。
claim 精确绑定 token 和 KVAF commit；一旦存在，新 token 与相同 token 都会被拒绝，
不能删除 claim 来制造第二次机会。新的运行使用 Jetty 602 和独立 state/result/lock
目录，但这仍不是 UMMU、AICPU 或驱动跨 NPU 隔离证明。

owner probe 或 zero-WQE 任一步失败，都只打印 hard-stop 证据并返回：入口不会自动
重试、给既有 worker 发信号、删除 active/quarantine、降级到 canonical SO，或继续
提交数据 WQE。即使 zero-WQE PASS，也只说明这一次资源创建和销毁闭环；在整机恢复
并补足隔离契约前，不能继续真实 H2D。

历史实验曾为 SSH/Xshell 粘贴问题提供 detached launcher。该入口能够固定并执行旧
提交，因此不能依赖“当前 checkout 已有硬门禁”来保证安全；启动、status、watch 和
recovery 子命令已全部从操作指南撤下。launcher 自身曾来自后续提交，但实验代码固定在
已经审阅和预检过的完整提交
`cc3cc0e8e08bf4882c290dc3ed0a5b40a45312ce`：该提交在 coordinator 内新增外部
expected-commit 强校验。入口先创建或复核该提交的 detached
worktree，再重新执行只读事故证据与 NPU2 idle 预检。预检通过后，它原子固定全局
launch reservation、token、完整 commit、日志和返回码身份，再通过 `docker exec -d`
启动该固定 worktree 内的 container-only coordinator。历史 status/watch 从全局只读
reservation 找到原运行，不依赖当前 checkout HEAD，也不会向 worker
发送信号。reservation、日志、dispatch-unproven 或返回码一旦存在，`start` 会拒绝
覆盖和重跑；Docker dispatch 失败只记录为“是否已启动未知”，不能伪装成 coordinator
已经返回。

早期 launcher `5615967b639f49e787221c317a95a0a1b4739015` 把
`/usr/local/Ascend/ascend-toolkit/set_env.sh` 本身是符号链接误判成 CANN 缺失。
若旧日志逐行精确停在 `stage=container-worker-started`、worker PID、UTC 时间和
`ERROR: CANN set_env.sh is missing`，且 worker rc 是 2，那么这只是 launcher
bootstrap 失败：它执行过只读的 `npu-smi` idle 查询，但没有进入 coordinator，因而
没有构建 token provider、安装 AICPU 包、发布 attempt claim、创建 URMA/UMMU 资源或
执行 zero-WQE。

不要删除或覆盖这次失败的 reservation、日志、worker selection 和 rc。旧版本曾提供
`recover-cann-bootstrap` 和 `recovery-status` 子命令；它们可能重新 dispatch 固定旧提交，
所以这里只记录名称，不给出可执行语法，当前也不得调用。

恢复入口只接受上面这个固定 launcher commit 的精确四行失败正文。它把旧四个文件的
SHA-256、device/inode/size/mode/link-count/owner 固定到新的只读 recovery
reservation，并要求原 dispatch/bootstrap 均无歧义、旧 nonce 不在任何 live
process 中、claim 与临时 claim 均不存在、token artifact/state/AICPU archive/注册项
均未产生。随后它重新验证两个历史事故、固定 worktree 和 NPU2 idle，再原子选择唯一
recovery worker。旧文件始终保持逐字节不变；recovery 路径一旦出现，不允许第三次
dispatch。

CANN 检查不再粗暴拒绝符号链接。入口只接受固定别名
`/usr/local/Ascend/ascend-toolkit/set_env.sh`，并要求它与
`/usr/local/Ascend/cann-9.1.0/set_env.sh` 解析到同一个、位于固定 CANN 9.1.0 根下的
root-owned 非可写文件。source 前后还会复核路径、元数据、哈希、
`ASCEND_HOME_PATH` 和 `bisheng`。这样既兼容安装器的正常版本链接，又不会静默切换到
另一套 CANN。`recovery-status` 会把旧返回明确标成 `KVAF_ORIGINAL_BOOTSTRAP_RC=2`
和 `KVAF_ORIGINAL_COORDINATOR_STARTED=no`；只有新日志同时包含 zero-WQE rc 0、PASS
以及精确 attempt claim 时才返回恢复成功。

active 状态在每次 fork 前先写 `host_launch_intent` 或
`device_launch_intent`，fork 成功后再补 PID。恢复逻辑按 phase 决定必须检查
单端还是双端终态，绝不会因为 SIGKILL 恰好发生在 fork 与 PID 落盘之间，就把
`pid=0` 当成“进程从未启动”。若终态已经证明安全但 worker 尚未退出，状态会
保留为 `safe_terminal_process_stuck`；worker 继续持有 active-session 锁。下次
执行会先打印记录的 PID，待 worker 自然退出后再按同一 `run_id` 自动复核并清理
active 状态，全程不会再次访问 topology、EID selector 或 NPU。

进程存活判断同时读取 `/proc/<pid>/stat`。`kill -0` 对僵尸进程仍会成功，因此不能
单独用它判定 worker 仍能执行：`Z/X/x` 状态会被视为已经退出并由父 runner 执行
`wait` 回收；只有终态已严格证明 `PASS/FAIL_SAFE` 时才随即删除 active。若 proc
状态不可读或无法解析，则仍按“可能存活”保留 active，避免观测故障放宽安全门。
`FAIL_SAFE` 刚发布时，runner 还会固定等待最多 2 秒，让正在执行
`aclrtResetDevice`/`aclFinalize` 的安全 worker 自然退出；不发送信号，超时后仍按
`safe_terminal_process_stuck` 保留状态。

端点在 `aclrtResetDevice`/`aclFinalize` 之前先写一份
`transport_safe_runtime_cleanup_pending` 的 `FAIL_SAFE` 证明。这里的
`aclrtResetDevice` 是进程运行时 context 清理，不是 `npu-smi` 的整卡复位。若
runtime detach 卡住，已证明关闭的 stream/queue/provider/Host 映射不会因此被
重新判成 DMA 不安全；detach 完成后 PASS 会原子覆盖这份临时证明。

`FAIL_SAFE` 不需要 reset，也不应手工创建或删除 quarantine。`FAIL_UNSAFE`
仍需先按站点流程 reset/硬件隔离所选 NPU，再终止记录的 PID 和删除
quarantine；不要在隔离前 `kill -9`。

`PENDING_UNPROVEN` 也不需要 reset。先让 worker 继续完成；active-session 锁释放
后再次运行同一脚本，它会在访问 topology/EID/NPU 之前用原 `run_id` 和结果目录中
固化的分类器自动清理安全状态。只有明确的 quarantine 才进入站点隔离流程。

历史 recovery-only 入口只获取持久锁并运行上述固化证据
分类：只清理 `PASS/FAIL_SAFE`，对 `FAIL_UNSAFE` 生成/保留 quarantine，对证据不完整
继续保留 active；成功后在 provider 校验、EID、NPU、ACL、DDR/HBM 和 URMA 之前退出。
该入口不要求加载 CANN 环境，也不读取或修改 provider 构建产物。因此无需手工删除状态
文件，也无需通过“故意使用失配 provider”来触发恢复。

Device provider 的 UMDK/UMMU 句柄必须由创建它们的同一 AICPU 进程和 SO instance
销毁。公开 KFC ABI 没有跨 invocation 的 worker affinity token，所以对多个短
invocation 采样到相同 PID/TID 不能作为所有权证明。

候选实现因此把 `Open -> 多轮 AIV BatchGet -> checked Close/retry` 放进同一个
`aicpu_kvaf_urma_epoch_owner_v1` invocation。provider 内还有进程级 epoch claim；
owner 存活时，其他 public/legacy Open/Close 都会被拒绝，避免第二个调用破坏它的
process-local 状态。Host 侧冻结新提交后，关闭命令与此前的 BatchGet kernel 排在
同一 control/data stream；stream FIFO 与同步结果才是关闭前 drain 的顺序证明。
GM 中的四个软件计数是额外的一致性证据，不是硬件 PI/CI 的替代品。

目标 CANN 9.1/driver 现场已经证明 `rtAicpuArgsEx_t.timeout=0` **没有建立无限
执行契约**：一次停留在 `OPENING` 的 owner 在约 1091 秒后仍被 runtime 以
`KERNEL_AICPU task timeout` 终止。1091 秒只是一次观测，不是稳定 SLA，也不应推断
该字段与平台 watchdog 的确切关系。Device Open 等待现在每秒输出
`KVAF_A5_AICPU_OWNER_PROGRESS`，记录 Host observer elapsed、AIV snapshot、owner
state/result、provider status/flags、身份有效位和 open result，便于区分未进入、
停在 `OPENING`、清理中和安全终态；这个可观测性修复本身不授予强杀或重新接管权限。

UB 管理连接失败的 Close 使用持久 phase cursor：先将管理 Jetty 与 shared JFR
切换到 `ERROR`，排空 JFC，然后才 unimport/delete Jetty、delete JFR、注销三个
Segment/Token、删除 JFC/context。管理 SEND 逐条计数；已知未完成 SEND 必须收到其
终态 CQE，提交结果不明时必须收到 `WR_FLUSH_ERR_DONE` barrier，不能把一次空 poll
误当成 SQ 已静默。每个失败都记录
`KVAF_URMA_MGMT_CLOSE_ERROR stage/status/errno/query_status/query_errno`。控制状态转换和
poll 只重试明确的 `EAGAIN/EBUSY/EINTR`；破坏性调用不因 `EINTR` 重放，Segment/Token
只有本次 UMMU 诊断明确给出 `RETRY_SAFE` 才额外允许重试。其他错误进入 UNSAFE，
不再被统一改写为 EAGAIN 无限循环。

接收侧也不能因一次空 CQ poll 就释放：即使 Server 尚未发送任何管理 SEND，两个
已投递 RECV 仍可能由硬件持有。Close 必须先观察到 Jetty/JFR 进入 ERROR 后产生的
`WR_FLUSH_ERR_DONE`，再观察一次空 CQ，才允许注销接收 Segment/Token。这个 flush
barrier 同时覆盖 recv-only 握手超时和正常收发后的关闭路径。只有成功返回了管理
Jetty 才启用这个要求；若 Open 在创建 Jetty 前失败，尚无任何管理 WQE 能被提交，
也不可能产生 send-side barrier，此时空 CQ 可以推进 partial-create 回滚。

Jetty/JFR/JFC 的内核 delete ioctl 每个对象只发一次；返回的 async/comp event
计数保存在 sidecar 中，随后以非阻塞 poll/ack 逐次收敛。只有 reported 与 acked
完全一致后才销毁 event condition/mutex 和队列映射。这样 Close 重入只继续清理事件
债务，不会再次发出已经可能生效的破坏性 ioctl。NPU2 风险验证的默认控制/总监督
超时分别收紧为 60/120 秒；总监督超时只控制 runner 何时返回诊断，不能撤销已经
提交的 AICPU owner。

只有具有明确内核/事件进度源的 `EAGAIN/EBUSY` 才能留待下一轮 Close。pthread
同步原语销毁失败、本地对象表仍有 live entry、链表或映射不变量破坏都没有后台进度
来源，必须锁存为 `cleanup_unknown/UNSAFE`，不能靠无限重试拖到 watchdog。UMMU 的
TABLE 存储与 queue mapping 分别记录释放 phase；因此 table 已释放而后续 queue
`munmap` 暂时失败时，重入只继续 queue 阶段，不会重放 `FREE_TID` 或把已完成的 table
释放误判成永久 `EBUSY`。

TID 的最终销毁顺序与配套官方 `libummu` 保持一致：`PLBI_ALL` 后先释放
MAPT/core/queue userspace mapping，再提交内核 `FREE_TID`。如果先发 `FREE_TID`，
当前驱动会在本进程仍持有 mapping 时持续返回 `EBUSY`；而旧实现又只在
`FREE_TID` 成功后释放 mapping，形成没有外部进度源的永久循环。每次失败同时输出
UMMU failure stage、pending operation、TID、live/pending/retained 数量和地址范围，
UDMA Token 层保留并向上传递该次 UMMU 的真实 errno。

UB server 只有在内核返回的实际 management Jetty ID 与请求值完全一致时才发布
READY。client 将 EID 解析、peer Jetty import、握手 SEND 和 CQE poll 分别记录为
独立失败阶段，避免再次把毫秒级本地 import/post 错误折叠成笼统的握手超时。

历史 candidate wrapper 曾能执行候选真机验证；当前它也受
`HARDWARE_EXECUTION_DISABLED` 约束。候选 owner
解决了 split Open/Close 的所有权缺口，但普通 KFC worker 是否允许跨整个模型
生命周期驻留，仍需验证当前 CANN 的 worker 配额、watchdog、取消和阻塞契约。
resource-free bounded probe 也仍只证明 AICPU/AIV 并发控制记录可见性。完整 epoch
边界和 decode 热路径见 [A5 direct URMA Decode H2D epoch 设计](a5_decode_epoch_h2d.md)。

### 5.2 历史方案：干净整机状态下的固定 NPU2 验证（当前禁用）

> 本节记录原三阶段方案，不表示重启后可以重新执行。OS reboot 能清除
> 旧设备状态，但不能补齐官方 caller-domain、cleanup 和 AIV 队列 ABI
> 契约。因此 `ZERO_WQE_NOT_PROVEN` 在重启后仍然成立。

历史 clean-NPU2 validator 是普通全局 singleton 的便捷入口，不是 5.1 的
cross-NPU bypass。它固定 logical/physical NPU2、两条精确 EID、UB
management 和 Jetty 502，并清除调用者提供的 UDMA device/index 猜测，由现场
exact-EID resolver 重新解析。它强制 `KVAF_DIRECT_ALLOW_BUSY_NPU=0`，拒绝所有
cross-NPU 风险接受变量，继续使用生产 active/quarantine 和全部历史 legacy lock。

因此，只要旧 NPU7 owner 仍持有 `/run/lock/kvaf-a5-direct-urma.lock`，或机器级
active/quarantine 尚未闭环，该入口会在 topology、ACL、DDR/HBM、URMA 之前失败；
它不会删除状态、发送信号或在线覆盖 AICPU package。旧方案要求复位/操作系统重启
后重新构建；当前即使完成重启也不能运行，因为能力证据缺口仍然存在。

OS reboot 会清除旧进程和设备状态，但不会清除 `/data` 中的恢复 barrier。历史流程
曾通过 reboot reconcile helper 的 `prepare`/`retire` 两阶段固化 boot proof。这里不再
提供调用顺序，因为 reboot proof 只能用于恢复旧状态，不能解锁新硬件实验。该 helper
会校验 boot ID、脚本/helper/HEAD 和原 marker 的
identity/SHA-256，持有 canonical、legacy、cross/reuse 全部协调锁，并确认没有 KVAF
direct-URMA 进程后，才把旧 marker 连同旧/新 boot ID 归档到
`retired-after-reboot/`，然后移除 canonical barrier。结果、lock 文件和所有 cross-NPU
实验证据不会被移动或删除；proof 被消费后不能用于跳过另一次真实重启。

一次调用严格按以下顺序建立三个独立验证 epoch，前一阶段非 PASS 就停止：

1. zero-WQE：只验证建链、注册和完整销毁；
2. one-kib：提交一条 1024-byte H2D READ；
3. batch：默认在一次 Open/register/connect 生命周期内提交 `4096 x 1024 B`，不会
   重复注册或建链 4096 次。

最终 batch 可通过 `KVAF_NPU2_TRANSFER_COUNT`、
`KVAF_NPU2_BYTES_PER_TRANSFER` 和 `KVAF_NPU2_BATCH_ITERATIONS` 修改；默认分别为
4096、1024、1。控制 timeout 默认 60 秒，单阶段 runner supervision deadline 默认
180 秒，失败后保留全部证据且不会自动进入下一阶段。历史 one-paste coordinator 会依次
完成 provider 构建、AICPU
打包、direct binary 构建、无 URMA 资源的 AICPU owner probe 和上述三阶段验证，避免
在终端逐段粘贴命令；每个子步骤仍使用原有 machine-wide guard。入口接受安装器生成
的 `ascend-toolkit/set_env.sh` 版本软链接，但要求它解析到固定 CANN 9.1.0；同时拒绝
全部 cross-NPU/reuse 环境变量并固定普通 provider、AICPU 包和构建目录，避免从事故
诊断 shell 继承版本化实验产物。

## 6. 操作命令

**当前仅允许 6.7 的两项静态检查。** 6.1～6.6 保留的源码准备、构建和
打包命令用于复现产物与审计 manifest，不会解锁硬件，也不是当前
要求操作员立即执行的下一步。在静态 package audit 和 HCCP/HAL 声明
preflight 完成前，不要重新打包、部署或进入任何 AICPU/URMA 动态路径。

### 6.1 服务器 B 宿主机：更新代码并准备固定源码

执行环境：`SERVER_B_HOST`

```bash
echo "EXECUTION_ENV=SERVER_B_HOST"
cd /data/c00982601/KVAF

git switch A5_AIV_URMA_cs
git pull --ff-only origin A5_AIV_URMA_cs
git log -1 --oneline

test -d /data/c00982601/umdk/.git || \
  git clone https://gitcode.com/openeuler/umdk.git \
    /data/c00982601/umdk
git -C /data/c00982601/umdk fetch origin
git -C /data/c00982601/umdk switch --detach \
  12917421beb93c1351eb2f124bb2f34612d8bf2d

test -d /data/c00982601/libummu/.git || \
  git clone https://gitcode.com/openeuler/libummu.git \
    /data/c00982601/libummu
git -C /data/c00982601/libummu fetch origin
git -C /data/c00982601/libummu switch --detach \
  70de020f776d8def0d8a8941cc7cbc5928d66c10

test "$(git -C /data/c00982601/umdk rev-parse HEAD)" = \
  12917421beb93c1351eb2f124bb2f34612d8bf2d
test "$(git -C /data/c00982601/libummu rev-parse HEAD)" = \
  70de020f776d8def0d8a8941cc7cbc5928d66c10
test -z "$(git -C /data/c00982601/umdk status --porcelain)"
test -z "$(git -C /data/c00982601/libummu status --porcelain)"

grep -E \
  'LIBUMMU_(REMOTE|COMMIT|VERSION|ABI_VERSION)' \
  /data/c00982601/umdk/src/urma/bazel/urma_deps.bzl
```

两棵上游源码树放在宿主机 `/data`，由已有同路径挂载提供给 Docker。不要把
它们复制进容器 overlay，也不要修改 root 公共账号的全局 Git 配置。

provider 还要验证运行内核的两份 UAPI。若 Docker 没有挂载宿主机的
kernel-devel 目录，可在宿主机把精确 header 复制到共享 `/data`：

```bash
echo "EXECUTION_ENV=SERVER_B_HOST"

(
  set -Ee
  kvaf_kernel_release=$(uname -r)
  kvaf_udma_uapi=
  kvaf_ummu_uapi=

  for kvaf_candidate in \
    "/usr/src/kernels/${kvaf_kernel_release}/include/uapi/ub/urma/udma/udma_abi.h" \
    "/lib/modules/${kvaf_kernel_release}/build/include/uapi/ub/urma/udma/udma_abi.h"
  do
    if test -r "$kvaf_candidate"; then
      kvaf_udma_uapi=$kvaf_candidate
      break
    fi
  done

  for kvaf_candidate in \
    "/usr/src/kernels/${kvaf_kernel_release}/include/uapi/linux/ummu_core.h" \
    "/lib/modules/${kvaf_kernel_release}/build/include/uapi/linux/ummu_core.h"
  do
    if test -r "$kvaf_candidate"; then
      kvaf_ummu_uapi=$kvaf_candidate
      break
    fi
  done

  test -n "$kvaf_udma_uapi" || {
    echo "ERROR: running-kernel UDMA UAPI not found" >&2
    exit 1
  }
  test -n "$kvaf_ummu_uapi" || {
    echo "ERROR: running-kernel UMMU UAPI not found" >&2
    exit 1
  }

  kvaf_shared_uapi=/data/c00982601/kernel-uapi/${kvaf_kernel_release}
  install -D -m 0444 \
    "$kvaf_udma_uapi" "$kvaf_shared_uapi/udma_abi.h"
  install -D -m 0444 \
    "$kvaf_ummu_uapi" "$kvaf_shared_uapi/ummu_core.h"
  sha256sum \
    "$kvaf_shared_uapi/udma_abi.h" \
    "$kvaf_shared_uapi/ummu_core.h"
)

docker ps --filter name=vllm-ascend-cs-kvaf
docker exec -it -u 0 vllm-ascend-cs-kvaf bash
```

如果任一 kernel UAPI 找不到，先安装与 `uname -r` 完全一致的 kernel-devel
或提供该运行内核的精确源码；不能从旧驱动、另一台机器或其他 kernel release
复制。现有 `vllm-ascend-cs-kvaf` 已挂载 `/data` 时不需要新建容器。

### 6.2 Docker：加载环境

执行环境：`VLLM_ASCEND_CS_KVAF_DOCKER`

运行前 `/dev/ummu/tid` 必须是可读写字符设备。若宿主机存在而 Docker 不存在，
需要重建同名容器并增加 `--device /dev/ummu/tid:/dev/ummu/tid:rwm`；
`--privileged` 只放开权限，不能保证容器创建后出现的嵌套设备节点自动可见。
runner 会在 EID、`npu-smi`、ACL、DDR/HBM 和 URMA 操作之前 fail closed。
除字符设备类型和权限位外，runner 还会执行一次不发 ioctl 的 `O_RDWR`
打开/关闭探测，以提前识别 device-cgroup 或 LSM 拒绝。

```bash
echo "EXECUTION_ENV=VLLM_ASCEND_CS_KVAF_DOCKER"
cd /data/c00982601/KVAF
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64:/usr/lib64:${LD_LIBRARY_PATH:-}
export KVAF_DIRECT_PROVIDER_BUILD_DIR=/data/c00982601/KVAF/build-a5-direct-urma-provider
export KVAF_UMDK_ROOT=/data/c00982601/umdk
export KVAF_LIBUMMU_ROOT=/data/c00982601/libummu
export KVAF_RUNTIME_UDMA_UAPI_HEADER=/data/c00982601/kernel-uapi/$(uname -r)/udma_abi.h
export KVAF_RUNTIME_UMMU_UAPI_HEADER=/data/c00982601/kernel-uapi/$(uname -r)/ummu_core.h

test "$(git -C "$KVAF_UMDK_ROOT" rev-parse HEAD)" = \
  12917421beb93c1351eb2f124bb2f34612d8bf2d
test "$(git -C "$KVAF_LIBUMMU_ROOT" rev-parse HEAD)" = \
  70de020f776d8def0d8a8941cc7cbc5928d66c10
test -r "$KVAF_RUNTIME_UDMA_UAPI_HEADER"
test -r "$KVAF_RUNTIME_UMMU_UAPI_HEADER"

cmp -s \
  "$KVAF_UMDK_ROOT/src/urma/hw/udma/kernel_headers/udma_abi.h" \
  "$KVAF_RUNTIME_UDMA_UAPI_HEADER" && \
  echo "KVAF_RUNTIME_UDMA_UAPI_MATCH=verified-exact"
cmp -s \
  "$KVAF_LIBUMMU_ROOT/kernel_headers/ummu_core.h" \
  "$KVAF_RUNTIME_UMMU_UAPI_HEADER" && \
  echo "KVAF_RUNTIME_UMMU_UAPI_MATCH=verified-exact"
```

若 kernel-devel 目录本来就在 Docker 内，两个 runtime 变量也可直接指向该目录；
不设置时脚本会检查上述两个标准 kernel-devel 路径。无论路径来自哪里，构建都会
要求内容与固定 UMDK/libummu header 完全相同。这里不再设置旧的 perftest root、
外部基准 SO、单独的 UMMU source root、system `liburma` 或 system
`liburma_common` 变量。

### 6.3 Docker：构建两端 provider

执行环境：`VLLM_ASCEND_CS_KVAF_DOCKER`

```bash
bash scripts/build_a5_direct_urma_provider.sh

ls -l \
  "$KVAF_DIRECT_PROVIDER_BUILD_DIR/libkvaf_urma_provider_host.so" \
  "$KVAF_DIRECT_PROVIDER_BUILD_DIR/libkvaf_urma_provider_device.so" \
  "$KVAF_DIRECT_PROVIDER_BUILD_DIR/kvaf_udma_uapi_probe" \
  "$KVAF_DIRECT_PROVIDER_BUILD_DIR/kvaf_udma_private_abi_probe" \
  "$KVAF_DIRECT_PROVIDER_BUILD_DIR/kvaf_ummu_uapi_probe" \
  "$KVAF_DIRECT_PROVIDER_BUILD_DIR/umdk-source.sha256" \
  "$KVAF_DIRECT_PROVIDER_BUILD_DIR/libummu-source.sha256" \
  "$KVAF_DIRECT_PROVIDER_BUILD_DIR/build.manifest" \
  "$KVAF_DIRECT_PROVIDER_BUILD_DIR/build.success"
```

脚本执行 fresh build，校验导出符号、AArch64 ELF、`DT_NEEDED`、RPATH、
两仓 clean commit/tree、源码前后哈希、UMDK→libummu 依赖锁、三个 ABI probe，
以及两份 runtime kernel UAPI。下面两条都必须无输出：

```bash
readelf -d "$KVAF_DIRECT_PROVIDER_BUILD_DIR/libkvaf_urma_provider_host.so" \
  | grep -E 'NEEDED.*lib(urma|urma_common|urma-udma|ummu|hcomm)\.so' || true
readelf -d "$KVAF_DIRECT_PROVIDER_BUILD_DIR/libkvaf_urma_provider_device.so" \
  | grep -E 'NEEDED.*lib(urma|urma_common|urma-udma|ummu|hcomm)\.so' || true

grep -E \
  '^(schema|source_layout|umdk_git_commit|libummu_git_commit|umdk_libummu_pairing|urma_core_mode|urma_common_mode|urma_common_dt_needed|perftest_source|perftest_management|runtime_udma_uapi_match|runtime_ummu_uapi_match|udma_uapi_probe_filename|udma_private_abi_probe_filename|ummu_uapi_probe_filename|udma_teardown_contract)=' \
  "$KVAF_DIRECT_PROVIDER_BUILD_DIR/build.manifest"

"$KVAF_DIRECT_PROVIDER_BUILD_DIR/kvaf_udma_uapi_probe"
"$KVAF_DIRECT_PROVIDER_BUILD_DIR/kvaf_udma_private_abi_probe"
"$KVAF_DIRECT_PROVIDER_BUILD_DIR/kvaf_ummu_uapi_probe"

grep -E \
  '^(schema|runtime_udma_uapi_header_sha256|runtime_ummu_uapi_header_sha256|udma_uapi_probe_sha256|udma_private_abi_probe_sha256|ummu_uapi_probe_sha256)=' \
  "$KVAF_DIRECT_PROVIDER_BUILD_DIR/build.success"
```

期望 manifest 为 `schema=kvaf-direct-urma-provider-v3`、
`source_layout=official-umdk-libummu-paired-v1`，两个 commit 与本节固定值一致，
`umdk_libummu_pairing=verified`，两个 runtime UAPI match 都为
`verified-exact`，且 `urma_common_dt_needed=absent`。success stamp 必须为
`schema=kvaf-direct-urma-provider-success-v3`。runner 在任何硬件资源创建前会
重新执行并校验三个 probe；源码、header、probe、manifest 或 success stamp
任一个变化都会以退出码 2 fail closed。

### 6.4 历史 Docker 步骤：安装 AICPU provider（当前禁用）

历史实现使用独立包名
`cann-kvaf-direct-urma-compat.tar.gz`，不覆盖 UB_MicroBenchmark 的
`compat/liburma_perftest.tar.gz`。当前
`scripts/package_a5_direct_urma_aicpu.sh` 在读取 CANN、状态目录、归档或
registry 前无条件返回 2，没有环境变量或风险接受开关可以解锁。
不要把该脚本当作当前部署命令，也不要手工复制 tar 或改写
`ascend_package_load.ini`。

历史注册块仅保留为已部署包静态审计时的识别记录：

```text
name:cann-kvaf-direct-urma-compat.tar.gz
install_path:2
optional:true
package_path:opp/vendors/cust/op_impl/aicpu/kernel
load_as_per_soc:false
```

只读脚本 `audit_a5_registered_aicpu_packages.sh` 会用上述记录检查当前
registry 已引用的归档，但不会添加、替换或修复任何包。历史归档的
精确白名单是：

```text
aicpu_kernels_device/libkvaf_urma_provider_device.so
aicpu_kernels_device/kvaf-direct-urma-provider.manifest
aicpu_kernels_device/kvaf-direct-urma-source.sha256
aicpu_kernels_device/kvaf-direct-urma-umdk-source.sha256
aicpu_kernels_device/kvaf-direct-urma-libummu-source.sha256
aicpu_kernels_device/kvaf_udma_uapi_probe
aicpu_kernels_device/kvaf_udma_private_abi_probe
aicpu_kernels_device/kvaf_ummu_uapi_probe
```

registry 的文件名后缀不能当作压缩格式契约。部分 CANN 内置条目虽然以
`.tar.gz` 结尾，实际可能是普通 tar、当前 `tar` 支持的其他压缩格式，或者
厂商私有容器；部分 `optional:true` 的跨产品 legacy 条目在当前安装中也可能
没有实体文件。因此审计器按内容使用 tar 自动检测，并采用以下分级：

- 可读 tar：逐个检查目标 ELF、成员名、SONAME、hash 和 alias；
- 缺失的非 KVAF `optional:true` 条目：输出 `PACKAGE_UNAVAILABLE`，不把一个
  实际不存在的包误判为冲突；
- 存在但不可用标准 tar 读取的非 KVAF 官方包：输出
  `PACKAGE_UNAUDITED`，将整体 `coverage` 标为 `partial`，既不猜测其内容，也
  不伪造冲突；
- 名称包含 `kvaf` 的目标包、`--replace-archive` 指定的目标，以及所有
  `--candidate-soname` 发布判定继续 fail closed。目标包不可读或缺失是错误；
  candidate 检查遇到 opaque 注册包会输出 `CANDIDATE_UNAUDITED_SCOPE` 并
  非零退出，因为“没有发现冲突”不等于“已经证明没有冲突”。

### 6.5 Docker：构建 AIV/SIMT smoke

执行环境：`VLLM_ASCEND_CS_KVAF_DOCKER`

```bash
if [[ -r \
  /data/c00982601/asc-devkit-9.1.0-20260722/include/simt_api/asc_simt.h ]]; then
  export KVAF_ASC_DEVKIT_PATH=/data/c00982601/asc-devkit-9.1.0-20260722
else
  export KVAF_ASC_DEVKIT_PATH=/data/c00982601/asc-devkit
fi

export KVAF_A5_DIRECT_BUILD_DIR=/data/c00982601/KVAF/build-a5-direct-urma
bash scripts/build_a5_direct_urma.sh
```

该 target 显式设置：

```text
KVAF_A5_BUILD_HCOMM_BACKEND=OFF
KVAF_A5_BUILD_DIRECT_URMA_BACKEND=ON
```

所以 direct binary 和 provider 的 `DT_NEEDED` 都不能含 `libhcomm`。

### 6.6 历史 Docker 步骤：精确确认 UDMA device/EID index（当前禁用）

执行环境：`VLLM_ASCEND_CS_KVAF_DOCKER`

历史 runner 以 EID 为 key 分别读取 Host `/sys/class/ubcore` 与 Device
`hccn_tool -g -dev_info`，不再从 NPU 号猜设备名。该 resolver 会访问实时设备
inventory，不属于当前静态检查。NPU7 的旧映射只保留为记录：

```text
HISTORICAL_ENDPOINT_RECORD (NOT SHELL)
  resolver: scripts/resolve_a5_urma_endpoints.py
  physical_device_id: 7
  host: 00000000003f050000100000df080b00 -> udmac0d1e5/eid0
  device: 000000000000020000100000df00e100 -> udmac0d0e2/eid0
  execution: HARDWARE_EXECUTION_DISABLED
```

未来恢复 runner 时仍必须重新做 exact-match；旧映射不能当作当前设备证据。

Host/Device peer EID 默认由
`scripts/resolve_a5_urma_eids.py` 从 `/proc/asdrv_ub/pair_info` 解析，不依赖
memcache。`dev_id` 是宿主机全局 selector；新代码统一通过宿主机和 Docker 都能
看到的 `/data/.kvaf-state/asdrv-ub-pair.lock` 串行化完整的
`write dev_id -> read pair_info -> validate/retry`。迁移期还会取得同一 namespace
中的旧 `/run/lock/kvaf_asdrv_ub_pair.lock`。不同容器的 `/run` 不能互相保护，
因此升级期间不要并发运行旧 resolver；所有调用方升级后以 `/data` 锁为准。

### 6.7 Docker：当前唯一允许的下一步（静态检查）

执行环境：`VLLM_ASCEND_CS_KVAF_DOCKER`

这一步不加载官方 `liburma.so.0`，不打开 NPU/URMA/UMMU 设备，不创建
context、JFC、JFR、Jetty、TP 或 MR，也不写 SQ/CQ。第一个脚本只把 CANN
registry 中已注册的 AICPU archive 复制到私有临时目录后做 ELF/符号
冲突审计；第二个脚本读取 `libra.so` 与 `libascend_hal.so` 动态符号表。
安装镜像如果不含 SDK 的 HAL header，会把 raw-HAL 声明面标记为 unavailable，
但仍继续识别 modern `RaCtx*` 与 SHMEM 使用的 legacy `RaAiQpCreate` 两套 HCCP
控制面。它不会加载任何库。

```bash
(
  set -Ee
  set -o pipefail

  echo "EXECUTION_ENV=VLLM_ASCEND_CS_KVAF_DOCKER"
  cd /data/c00982601/KVAF
  source /usr/local/Ascend/ascend-toolkit/set_env.sh

  bash scripts/audit_a5_registered_aicpu_packages.sh \
    --ascend-home "$ASCEND_HOME_PATH" \
    2>&1 | tee /tmp/kvaf_registered_aicpu_static_audit.log
  kvaf_audit_rc=${PIPESTATUS[0]}
  test "$kvaf_audit_rc" -eq 0

  bash scripts/preflight_a5_hccp_hal_capabilities.sh \
    2>&1 | tee /tmp/kvaf_hccp_hal_declaration_preflight.log
  kvaf_hal_rc=${PIPESTATUS[0]}
  test "$kvaf_hal_rc" -eq 0

  echo "KVAF_STATIC_CAPABILITY_COLLECTION_OK"
)
```

只做现状盘点时，package audit 的可接受结果必须为 `conflicts=0`、
`candidate_conflicts=0`、`candidate_unknowns=0`、`errors=0`。如果同时出现
`coverage=partial`，必须查看每一条 `PACKAGE_UNAVAILABLE` 和
`PACKAGE_UNAUDITED`；它表示官方内置包存在明确的未审计范围，不能写成
“全部注册包无冲突”。带 `--candidate-soname` 的发布判定还要求
`candidate_unknowns=0`；仅由实际缺失的 optional 包造成的 partial coverage
不会伪造冲突，但 opaque 范围会严格阻止发布。HCCP 检查完成时会输出
`KVAF_A5_HCCP_CONTROL_PLANE_STATIC_AUDIT_OK` 和实际候选
`libra-ra-ctx`、`libra-legacy-ai-qp` 或 `none-*`；同时始终输出
`KVAF_HCCP_BACKEND_READY=no`。符号完整只用于选择适配器，不能证明结构 ABI、
`USER_CTL_NORMAL`、地址可见性或生命周期语义。
两个命令都通过也不能执行 zero-WQE 或 H2D；它们只是为下一轮设计
选择 Host HCCP/HAL 控制面还是补齐官方 runtime 契约提供证据。

### 6.8 历史运行方案（当前硬禁用，不要执行）

> 从本节开始到第 7 节之前的命令、环境变量和判定矩阵只是历史设计
> 记录，不是当前运行指南。包括 epoch-owner、direct H2D 和动态 runtime probe
> 在内的
> resource-bearing/dynamic 入口均已被无条件门禁。下文中的“当前”均应读作
> “原设计曾计划”。

正式 runner 仍会由 AICPU gate 以退出码 2 停止。真机验证必须使用显式 candidate
wrapper；它在同一个 AICPU invocation 内持有 Device provider，并在同一个
control/data stream 上顺序执行全部 BatchGet 与 shutdown。默认负载是 4096 个
1 KiB 请求（每批 4 MiB），不是旧的“总共 4096 bytes”。

```text
HISTORICAL_EPOCH_OWNER_RECORD (NOT SHELL)
  logical_device_id: 7
  physical_device_id: 7
  host endpoint: udmac0d1e5/eid0
  device endpoint: udmac0d0e2/eid0
  management: UB, jetty_id=507
  former workload: 4096 descriptors x 1024 bytes x 10 batches
  former timeouts: provider=120s, run=600s, management-ready=30s
  former runner: scripts/run_a5_direct_urma_epoch_owner_validation.sh
  execution: HARDWARE_EXECUTION_DISABLED
```

runner 不再用固定 sleep 猜测 Host server 是否启动。它固定本次 Host PID 的
`start_ticks` 和可执行文件，并只接受本次 Host provider 写入日志的完整
`KVAF_URMA_MGMT_READY role=server jetty_id=<id> local_eid=<eid>` 行；只有确认该 PID
已创建管理 Jetty 并预投 RECV，才立即启动 Device。旧
`KVAF_DIRECT_SERVER_START_DELAY_SECONDS` 会直接报参数错误，避免遗留的 5 秒延迟
继续消耗 Host 管理握手 deadline。

若 Host 在 READY 前退出，先查看同一 `host.log` 中的
`KVAF_URMA_MGMT_ERROR stage=<stage> raw_errno=<raw>`，以及同一行的
`fallback_errno=<fallback> effective_errno=<effective>`。`stage`
精确区分 EID 解析、context/JFC/segment/JFR/Jetty 创建以及 server/client handshake；
每个独立 URMA 操作前都会清空 `errno`。固定源码会主动屏蔽动态 provider 目录扫描，
该扫描可能产生 `ENOENT`，因此不能再只凭最终 `provider_status=-2` 推断设备节点或
EID 缺失，必须以 `stage` 和该阶段的底层 UMDK 日志为准。该诊断不改变 provider
ABI、资源所有权或 checked teardown 行为。

Token/TID 分配失败还会先输出 `KVAF_URMA_TOKEN_ERROR`；Segment 注册失败会先输出
`KVAF_URMA_SEGMENT_ERROR`，其中包含同线程、同次 UMMU 操作的 failure stage、
retryability、pending/retained 计数和 kernel-register rollback 结果。每个顶层 UMMU
操作都会清空线程快照，避免把前一次失败或另一线程的失败误配到本次注册。

JFR 创建失败会先输出 `KVAF_URMA_JFR_ERROR`。它保留 public
`urma_create_jfr()` 指针接口原本会丢失的首个 checked-UDMA 叶子，区分参数/context
校验、index/RQ 分配、record/sleep doorbell、内核 `CREATE_JFR` 命令和 table 插入，
并分别记录创建错误和 rollback 错误。顶层 management diagnostic 的
`operation_status` 是底层返回值。旧日志里的 `completion_status` 并不是统一的
硬件完成码；它是 provider-specific 辅助槽。当前日志改名为
`management_auxiliary_status`，并用 `management_auxiliary_kind` 明确其语义；
CREATE_JFR 失败时该值是本地 `provider_stage_id`。即使
AICPU stderr 未被完整转发，也能从 Device owner progress 恢复失败阶段。该诊断不会
切换到 non-shared/embedded JFR，也不会设置 `non_pin`：A5 UB 要求 shared JFR，后两种
绕行均缺少当前 checked teardown 所需的公开所有权证明。若阶段是
`create_command`，应再结合同时段 UDMA driver 日志判断 RQ/index/DB pin 或 mailbox
失败；不得仅凭原来的通用 `EIO` 修改地址或销毁策略。

此前的 Host tracefs 结论已撤销。fallback trace 中执行 `CREATE_JFR` 的 comm 是
`hccn_tool-*`，而不是 KVAF AICPU worker；Host tracefs 也不能证明 Device OS/AICPU 的
ioctl 路径。因此不得再把那棵成功调用树解释成“KVAF Device 已进入 mailbox”，也不要
继续通过修改 ftrace 过滤器重复 zero-WQE。`trace_a5_direct_urma_npu2_jfr_failure.sh`
已在任何 Docker、tracefs 或硬件访问之前 fail-closed；它只保留历史实现供审计，不能再
启动一次硬件实验。

历史设计曾计划执行三阶段 Device-runtime A/B，逐级扩大资源面。
这三种模式现在全部由 `DYNAMIC_RUNTIME_PROBE_DISABLED` 门禁拒绝：

1. 历史动态 runtime probe 的 `discover` mode 在 AICPU 中按 handle
   `dlopen/dlsym/dladdr` 系统 `liburma.so.0`，不显式调用 `urma_init/create_*`。但
   `dlopen` 会执行官方 DSO constructor，constructor 是否初始化全局状态、打开设备或
   启动后台对象没有公开的 AICPU 契约，因此该阶段不再称为 resource-free，也不能用
   shell `timeout`/信号强杀。该阶段只把
   `init/uninit/get_device` 作为必需基础符号，仍会探测 context/JFC/JFR 可选符号，并以
   `required_symbol_mask/resolved_symbol_mask` 报告差集。v2 wire ABI 使用 Host 随机
   nonce、mode/EID/logical-device/queue-depth 回显、provider ABI token、固定 offset 与
   严格 enum/flag/cleanup/string/reserved 校验；旧 DSO、旧 ABI 或畸形结果不会被当作成功。
   结果行中的 symbol-surface 字段只表示对应入口可解析；官方 Device runtime 若裁剪了
   队列生命周期 API，会被单独归类，而不会伪装成 JFR 参数错误。由于单独的 `dlclose`
   不能证明 constructor 副作用已撤销，discover 即使解析成功也固定输出
   `cleanup_proven=0` 并保留 owner/quarantine 到 OS 重启或等价厂商恢复；它不会发布
   safe-terminal 成功标记；
2. symbol surface 明确成功后，先执行 `--mode init-device`，在同一 KFC 调用中只做
   `init -> get_device_by_name -> uninit`，用于把 provider/device discovery 与队列资源
   创建分开；
3. init-device 明确成功后，才选择 `--mode create-delete`，并在同一个 KFC 调用中完成
   `init -> context -> JFC -> JFR -> reverse delete -> uninit`。三种模式都先严格证明
   logical/physical NPU 和 NPU idle，并要求同一 caller-domain 风险确认；只有
   init-device/create-delete 额外要求 Device EID/UDMA endpoint 亲和性，discover 不要求
   EID 或 UDMA device。runner 不再用外层 `timeout` 强杀任何一种 probe，而是统一以独立
   session 启动长驻 supervisor。supervisor
   首先核对继承的 active/deploy lock FD 8/9 确实指向固定锁文件，然后按
   `quarantine -> active -> ready -> worker` 顺序进行 durable publish；父进程必须收到并
   核对 ready 中的 mode/run/boot、PID/start_ticks、worker path/SHA 和结果路径，才承认
   worker 已启动。Ctrl-C、SSH 断开或观察超时只结束观察者，不杀资源 owner。KFC、provider
   或 Host ACL teardown 任一环节不确定时，owner 留存且 active/quarantine 保留；只有
   provider 逆序清理、stream/GM、device context 与 ACL finalize 全部检查成功，日志已
   `fsync`，且 terminal 绑定日志 SHA256 后才移除。若观察 shell/SSH 在 owner 已安全完成
   之后中断，下一次 runner 会在验证当前二进制、CANN 或 provider 之前，仅依据旧会话
   自身封存的 dedicated schema、mode/boot/run ID、worker 路径与 SHA、PID/start_ticks、
   log SHA 和安全终态恢复 stale active/quarantine；这保证重编译或删除旧 worker 不会让
   安全终态永久不可恢复。borrowed-runtime 使用独立 blocked schema，在同一 boot 内
   永不自动恢复；只有严格校验旧 terminal/log/hash 后又证明 kernel boot ID 已变化，
   才允许执行一次“仅清理 barrier、不启动 probe”的 reboot recovery transaction。
   `HoldUnsafeOwner` 没有 terminal 时，recovery 只验证旧 boot 的
   quarantine/active/ready 全量身份后清障，并丢弃其 capability 结论；不会伪造安全终态。
   其他缺字段、摘要不符、owner 仍存活或终态不明确的情况一律保留。

该探针也不把官方 public API 视为天然完备：Device OS 没有公开的 UAPI/provider ABI
协商；`urma_init()` 成功不等价于目标 provider/device 可用；`create_*()` 返回 `NULL`
后只给 `errno`，没有半创建对象句柄或 cleanup certainty。新版 Host UMDK 已声明并实现
`alloc/set/get/active/deactive/free` 队列接口，其中 `get_opt` 可返回 SQ/RQ/CQ、DB、
PI/CI 地址；但同为 `URMA_API_VERSION=0.9` 的旧快照没有这些 callable API，相关结构字段
顺序也不同。因此 runtime discover 会分别报告“候选 modern wrapper”和从当前 checked
Direct-UB 路径审计出的 34 个公开符号精确并集
`current_direct_ub_symbol_surface`；后者即使为 `complete-semantics-unproven` 也不能
用来推断官方函数实现完整。即使符号存在，地址是否属于
AIV 可访问的 GM 域，以及 WQEBB/CQE layout、owner bit、doorbell、barrier/cache 语义
仍是未公开/未协商的私有契约。故 `create_* == NULL` 一律设置
`KVAF_URMA_RUNTIME_PROBE_CREATE_FAILURE_CLEANUP_UNPROVEN` 并保留 quarantine，不能
宣称安全清理。`operation_errno_valid=0` 时也不会用 `ENOENT/ENODEV/EIO` 猜错误码；
loader 失败只保留净化后的 `dlerror`。即使 create-delete 的公开 API 都返回成功，输出
仍会同时记录 `official_public_cleanup=returned-complete` 与
`official_side_effects_cleanup=unproven`：前者只证明安装 runtime 接受最小配置并完成
公开逆序调用，后者明确说明 constructor、线程和内核内部副作用没有公开的 AICPU
cleanup 契约。因此当前不会发出 accepted safe-terminal，也不自动解锁完整 H2D；
owner/quarantine 保留到重启或厂商等价恢复。

每条结果显式输出两个独立维度：`capability_class` 描述符号表面、runtime init、
endpoint/provider discovery、public create 或 cleanup contract 所处阶段；
`cleanup_proven` 只描述本次所有权是否安全结束。固定字段
`official_abi_handshake=unavailable`、
`documented_official_aicpu_call_contract=not-found-in-audited-sources`、
`local_runtime_compatibility=unproven`、
`aiv_address_visibility=unproven` 与 `private_queue_abi=unproven` 防止把一次最小成功
误当成官方 ABI 完备。KVAF probe 每次构建生成新的 build-instance ID 并用作独立
SONAME，另存 source/config 摘要；Host 验证期望 package 内 ELF SHA256，Device 仅回显
内嵌 build-instance ID，因此不声称已证明实际加载 ELF 的逐字节身份。新 SONAME 防止
同一 AICPU 进程把前一次构建的 loader mapping 当成当前包。探针同时要求每个计入 mask
的官方符号都由 `dladdr` 证明
来自与解析到的 `urma_init` 相同的 DSO，且 v2 nonce/token 回显吻合。但随后加载的官方
`liburma.so.0` 尚无内容哈希/源码版本握手，其 dlopen 依赖闭包和实际服务目标 device 的
底层 provider 身份仍固定为 `unproven`。

这里的 caller-domain 限制尤其重要。当前 UMDK 文档与实现把 public `urma_*` 定位为
Linux Host userspace API：实现会扫描 provider/sysfs、打开 `/dev/uburma`、执行 ioctl，
并依赖 pthread/dlopen。仓内没有 AICPU/Device OS 可调用契约。Ascend910D 特性表中的
“支持”只能证明硬件/provider 能力，不能证明同一 `liburma.so` ABI 可在 AICPU 中直接
调用。当前 public 实现本身也有待完善项，例如部分 `alloc_jfs/alloc_jfr` 路径在 provider
失败后缺少 NULL 检查，provider 实现与能力表对 RC/token policy 也存在不一致。因此
包括 discover 在内的 runtime probe 都不能把官方实现当作无 bug oracle。
旧实现曾记录一个 caller-domain 风险接受字段。该变量和完整接受短语已从操作文档
移除；它们**不能解锁任何模式，也不应在 shell 中设置**：

```text
HISTORICAL_RISK_FIELD (NOT SHELL)
  name: KVAF_ACCEPT_UNDOCUMENTED_AICPU_URMA
  value: retired; intentionally omitted
  execution effect: none
```

该决定会被写入 quarantine、active、ready、terminal 四份证据。discover 虽然不要求
EID/device endpoint，仍要求该变量并走相同的 durable quarantine，因为 DSO constructor
已经越过纯符号文件检查的边界。这个确认只授权一次受保护的能力实验，不把 public URMA 解释成
官方支持的 AICPU ABI，也不证明官方接口完整：缺少 caller-domain 契约、provider ABI
handshake、半创建对象的 cleanup handle、AIV 队列布局与 cache/doorbell 语义仍分别属于
待官方补齐或必须由匹配版本实测证明的缺口。

仓内参考的官方 `urma_npu` 接口同样不能被当作现成替代：其公开类型当前只有
`URMA_NPU_SDMA`，并明确标注 `URMA_NPU_ROCE // not support now`；实现依赖 Host/Device
HDC daemon、`halQueryDevpid` 和预留 SDMA channel。它可作为官方 H2D 生命周期参考，
但不是本目标的 AIV+SIMT 直写 UBC/CTP SQ/CQ 数据面。

判定矩阵是：

- symbol surface 成功而 init-device 失败：问题已收敛到 runtime 初始化、provider
  discovery 或 Device endpoint，不应再修改 JFR/Jetty/SQ/CQ；
- init-device 成功而 create-delete 失败：runtime/provider/device 基础面存在，问题才进入
  public context/JFC/JFR create 或 cleanup contract；
- Device runtime 最小 create-delete 的公开返回路径与 source-linked 路径结果不同：只能
  证明两条路径在 public JFR 生命周期前后发生分歧。根因仍包含头/库 ABI、provider、
  UAPI、AICPU caller domain、配置与内存 backing；停止盲改 JFR flag，但不能直接归因为
  某一版本错配；
- 两条 public JFR 路径均返回真实且有效的 `EINVAL`：只能证明当前两套 public 路径
  共同拒绝当前最小配置，仍不能区分 ABI/UAPI 错配、provider/KVAF/driver bug、权限或
  内存模型限制以及目标操作确实不支持；据此停止继续排列 JFR 参数并转查 Host
  HCCP-style owner，但不能宣布硬件故障或官方明确不支持；
- runtime library/provider 不存在：AICPU 镜像未提供完整官方 runtime，必须精确打包
  匹配版本或转 Host bootstrap。

Host HCCP-style 分支也必须先运行 `preflight_a5_hccp_hal_capabilities.sh`。主路径
优先检查驱动 `libra.so` 的 `RaCtx*`；若安装版只导出 SHMEM 已使用的 legacy
`RaAiQpCreate`/MR/QP 接口，则单独报告该候选。raw-HAL 只保留为回退研究：脚本检查
五个 HAL symbol，并在安装 SDK header 存在时编译精确函数签名、CP1 常量、结构成员和
HCCP URMA 资源枚举。HCCP 使用的 `UdmaVaInfo` 并不属于安装版公开 HAL：它位于固定 HCOMM
源码的内部 `rs_ctx_inner.h`。KVAF 的候选兼容头记录 HCOMM commit 与源文件哈希，并以
静态断言固定 24 字节布局和 0/4/8/16 字段偏移；它不复制 enum 数值，也不把私有结构
冒充公开接口。若 V2 map 公共类型/enum 缺失，门禁失败；私有布局编译通过也只算
`source-audited-...-runtime-unproven`。脚本固定输出 `KVAF_HCCP_BACKEND_READY=no`，因为
resource-ID/UASID 语义、安装 HCCP/HAL 与 HCOMM 快照配套及运行时生命周期仍未证明。
未来 backend 应优先绝对路径加载已检查的 `libra.so`，由 HCCP 完成产品相关 HAL
映射；只有 `libra.so` 功能面明确不足时才评估私有 HAL adapter。

Device Open 结果的 `reserved[9..10]` 同时带回只读 JFR 命令摘要：完整 64 位 JFC
handle、JFC ID、context page shift、`F_GETFD` errno，以及 UDATA 地址/大小、RQ/index
范围、4 KiB UDMA queue 对齐、JFC 创建时 sidecar 身份和
`non_pin/DTU/hugepage` 的检查位。完整地址、长度和创建时身份字段仅写入
AICPU leaf 日志，不声称它们全部被压缩到 `reserved[9..10]`。该诊断不会阻止、
增加或重试 ioctl；原始路径仍只发送一次 `CREATE_JFR`，避免诊断改变硬件语义。
若内核日志是 `failed to find jfc`，结合 creation-time handle/id/fd
identity 修 JFC 所有权；若是 `in_len/in_addr`，修加载模块与固定 UAPI 配对；若是 pin
失败，再评估 Host-owned resource owner。当前失败仍早于 Jetty/TP/import 阶段，因此
不能把 HCCP 的 Jetty/TP flag 当作此 `CREATE_JFR` 的直接修复。

`aff4de9` 的证据只证明 hugepage 失败；`f465ef1` 把 Device 管理 RQ 改成
`UDMA_MMAP_KERNEL_BUF` 后仍失败，证明不能把 index、record DB 或更上层
TLV/uobject/EID 检查排除。所提供的两个旧工作 UDMA userspace 实现都让 RQ、index、
record DB 使用 `MAP_PRIVATE|MAP_ANONYMOUS`，sleep DB 也使用匿名页；而 26.06 UMDK
在引入 separate-page-table 支持时，把前三者改成 kernel buffer、只保留 sleep DB 匿名。

因此 Device 使用统一的 context/SVA-aware queue 策略：若 context 的
`sva_sep_mode_en=0`，管理 JFR、decode 数据 JFR 的 RQ/index/record DB，以及管理/数据
Jetty SQ 都采用旧工作实现已验证的 anonymous pinned family；若为 1，则全部保持官方
kernel-buffer family。sleep DB 继续走原 payload 匿名路径，Host 保持 AUTO。策略只在
`kvaf_udma_queue_memory_policy_for_context()` 解析，避免 zero-WQE 通过后数据面又悄悄
切回另一地址域。错误串显式输出 `jfr_context_sva_separate` 和
`jfr_user_anonymous_memory`，避免再次因 compact flags 没有携带 bit 20 而误判。两种
分支都保留当前 UAPI、64 位 JFC handle、`non_pin=0`、`dtu=0`，且仍只发送一次
CREATE_JFR。

三类映射仍在 kernel JFR 删除后由原 checked `munmap` 状态机释放；anonymous record
DB 使用独占页并拒绝复用已有不同 backing 的同类型页，防止管理策略污染随后数据面。
管理 shared JFR 也按 HCCP 的 context 初始化顺序提前到 JFC 之后、任何 management
Segment 注册之前创建，depth 与 shared JFC 一致为 64。Jetty 继续复用这个 JFR；销毁
顺序仍是 Jetty -> shared JFR -> shared JFC -> context。该修改既减少失败时的回滚面，
也排除同 context 中先注册 UMMU Segment 对 JFR pin 域的干扰。

HCCP 能成功而 direct-UMDK 失败并不矛盾。HCCP 的 rdma_service 是长驻 owner：它用平台
安装且与驱动配套的 `liburma.so.0` 在 context 初始化时创建普通 shared JFC/JFR；AIV 的
DEV_USED Jetty 则另外走 `JETTY_MODE_USER_CTL_NORMAL`，由 HCCP/HAL 把 SQ/CQ/DB/UASID
映射到 CP1。KVAF 当前失败的是短驻 AICPU 中 public `/dev/uburma` standalone
`CREATE_JFR + pin` 路径，尚未到 Jetty。故 HCCP 成功证明硬件/EID 基础链路可用，但不
证明这条 AICPU public-create 路径可用。

若本轮统一内存策略、context-early shared JFR 后仍在同一个 `CREATE_JFR` 返回
`EINVAL`，停止继续排列 depth/handle/hugepage/Jetty flag。下一实现应转为 checked
user-control backend（custom RQ/index/SWDB、`CREATE_JFR_EX/CREATE_JETTY_EX`、完整 owner
sidecar），或让 HCCP-style 长驻 bootstrap owner 只创建和映射控制面资源。两者都只把
SQ/CQ/DB/UASID 描述符交给 AIV；decode 的 AIV+SIMT 直写数据面不经过 HCOMM/HCCP。

数据面参数显式传入 `--share_jfr true`，不再依赖固定 perftest 在 capability check
中把默认 `false` 隐式改写为 `true`。这保证后续 data Jetty 与 management Jetty
使用一致的 A5 UB receive-queue 契约，但它不是 management `create_jfr` 失败的修复。

Host binary 会先初始化 ACL/CANN，再加载 direct provider。CANN 依赖中的官方
`libummu` 构造函数可能已经打开 `/dev/ummu/tid`；A5 UMMU driver 又按
`tgid + task start_time` 为一个 owner 只建立一个 `proc_manager`，因此 provider
再次打开会返回 `EEXIST`。这不是设备被其他业务占用，也不是需要复位 NPU。
source-built UMMU 只在 fresh open 精确返回 `EEXIST` 时扫描当前进程 fd，并且先
`F_DUPFD_CLOEXEC`、后验证 duplicate。只有唯一匹配、`O_RDWR`、同一字符设备的
process descriptor 才会被采用；日志会输出
`KVAF_UMMU_CONTEXT_READY fd_mode=process_fd_dup`。KVAF 只关闭自己的 duplicate，
不会关闭 CANN 的原 fd；duplicate 共享 driver `private_data`，最后一个引用关闭前
`proc_manager` 不会释放。无候选、只读候选或多个候选均 fail closed，不会尝试用
`ALLOC_TID` 做有副作用的探测。

这个兼容路径要求一个 worker 在 provider 资源生命周期内不 fork，也不通过
`SCM_RIGHTS` 接收外部 UMMU fd。当前 runner 是 `fork + exec` 后才在独立 worker 中
加载 provider，所有相关 fd 都是 `CLOEXEC`，并满足该约束；未来集成到常驻 decode
进程时，应把 UMMU owner 收敛成进程内唯一 broker，通过显式 acquire/release 返回
duplicate，而不是让多个组件分别初始化 context。

Device provider 的控制超时仍由 `KVAF_H2D_TIMEOUT_SECONDS` 指定。Host 的管理窗口
独立使用 `KVAF_HOST_CONTROL_TIMEOUT_SECONDS`；默认值是 Device timeout 加 management
ready timeout 再加 5 秒，并在 manifest 中分别记录。AIV 对 AICPU owner 的 READY
观察使用单次一致性快照和 10 ms Host 节拍，不再设置一个会与 provider Open/cleanup
竞争的第二截止时间。普通管理握手超时必须等 owner 发布 `OPEN_FAILED_SAFE` 后按退出码
3 收敛；若 owner 本身长期无终态，runner 以退出码 5 保留 live worker 和 active
state，而不会仅因观察超时制造 quarantine。

failed-Open 的 checked teardown 会先经过 `CLOSING`、`CLOSE_RETRYING`，最后才发布
`OPEN_FAILED_SAFE`。前两者仍持有 provider 和同一 session 的重试 cursor，不是终态；
AIV 与 Host wrapper 都必须继续观察，且不得再次提交 Close。AIV 只有在验证 owner
identity、provider result session 和 active/teardown flags 后才允许等待这两个过渡态。
`CLOSED`、`OPEN_FAILED_SAFE` 才是已停止修改传输资源的完成态；明确的
`UNSAFE_WAITING` 仍按 ownership failure 处理。runner 在首次真正的 unsafe 证据后会
持久化 quarantine，并自动打印最多 320 行 Host/Device 日志，保留
`aiv_status/owner_state/provider_status` 供定位。

`KVAF_H2D_RUN_TIMEOUT_SECONDS` 从 Device 启动后计时；Host management-ready 阶段由
`KVAF_DIRECT_SERVER_READY_TIMEOUT_SECONDS` 单独计时。因此最坏墙钟监督时长是两者
之和，不应把 run timeout 理解为从 Host fork 起算的全局 deadline。

需要回归旧 TCP 管理面时，显式设置
`KVAF_DIRECT_MANAGEMENT_MODE=tcp`、`KVAF_DIRECT_CONTROL_PORT` 和
`KVAF_DIRECT_CONTROL_SERVER_ADDRESS`。此时 runner 才恢复 PID-bound LISTEN 证明和
IPv4/IPv6 地址校验；不要把它误当成 UB 默认路径的前置条件。

当前 `vllm-ascend-cs-kvaf` 挂载全部 0–7 号设备，runner 要求逻辑卡号与物理卡号
相等。二者不同时，无法在“不先创建 ACL device context”的前提下证明映射关系，
因此会在注册 DDR/HBM/URMA 前退出。

该 smoke 的 payload 上限是 64 MiB，避免错误参数在资源 armed 后触发数 GiB 的
Host 临时分配。必须显式填写 `KVAF_PHYSICAL_DEVICE_ID`；runner 不再为推导它而
提前调用会执行 `aclrtSetDevice` 的 ACL topology 查询，因为那会在空闲证明之前
选择/创建 Device context。可先在服务器 B 宿主机用 `npu-smi info -m` 核对映射，
再在 Docker 中填写实际值。runner 在启动 Host 之前执行 `npu-smi info`；只有
inventory 中恰好存在该卡，且进程表中明确出现
`No running processes found in NPU <physical-id>`、没有目标卡进程记录时才继续；
缺少进程行本身不算空闲证明。目标卡存在 vLLM、SHMEM 或其他 ACL 进程、命令失败、
输出截断或格式冲突时均以退出码 2 停止，且尚未 armed 任何传输资源。`npu-smi`
默认由 `timeout` 限制为 15 秒，只有诊断慢机器时才调整
`KVAF_NPU_SMI_TIMEOUT_SECONDS`（范围 1～120 秒）。

原始快照、解析结果和解析器会分别固化为结果目录中的 `npu-smi-info.txt`、
`npu-idle-check.txt`、`check_a5_npu_idle.py`，manifest 记录三者哈希。这个检查是
启动前的瞬时证明，不能阻止检查之后第三方抢占 NPU，因此硬件 smoke 仍应使用
调度系统隔离的专用卡。

旧设计包含过 busy-NPU 风险开关。该开关不能绕过当前 capability ledger 门禁，
也不再作为可复制的 shell 设置公开：

```text
HISTORICAL_BUSY_OVERRIDE (NOT SHELL)
  field: KVAF_DIRECT_ALLOW_BUSY_NPU
  former value: enabled
  current effect: none; hardware execution remains disabled
```

未来恢复测试时也必须默认保持该变量未设置。旧的
`KVAF_DIRECT_ALLOW_UNVERIFIED_UDMA_TEARDOWN` 已被删除；新 runner 只接受源码内置
UDMA+UMMU 和 `checked-retryable-v1`，没有回退到系统 provider 的风险开关。

历史 runner 的终态分类仍有事故分析价值，但不再附带执行命令：

| 历史退出码 | 历史含义 | 当前操作 |
|---|---|---|
| 0 | PASS | 仅作为旧证据，不解锁硬件 |
| 2 | PRECHECK_FAILED | 当前硬门禁的预期返回 |
| 3 | FAIL_SAFE | 保留日志；不得自动重跑硬件 |
| 4 | FAIL_UNSAFE | 保留 owner、backing 和 quarantine |
| 5 | PENDING_UNPROVEN | 不杀进程，不启动另一轮 |
| 130 | RUNNER_DETACHED | 仅按持久状态做恢复审计 |

多个 active pair 的 channel 选择同样属于历史运行参数；当前不得通过设置
`KVAF_PAIR_CHANNEL` 后重跑 direct smoke。

成功必须同时看到 Host DDR 注册、两端 provider ready、direct channel、
数据/canary、两端 teardown 和最终 acceptance marker。证据保存在
`/data/.kvaf-state/KVAF/results/a5_direct_urma_h2d_<run-tag>/`，
其中还包含 provider build/source
manifest，以及 direct smoke 的 build/source manifest。runner 会校验实际
链接的 `libascendcl.so`、官方 ACL/runtime header、二进制和构建时源码哈希；
失败的原地重编译会先使旧 manifest 失效，不能误用旧二进制继续上硬件。
Host DDR 注册固定加载真实 driver 路径
`/usr/local/Ascend/driver/lib64/driver/libascend_hal.so`，AICPU launch 只从已经由
build manifest 验证并加载的 `libascendcl.so` 依赖闭包解析，不再搜索任意
`libruntime.so`；结果 manifest 同时记录 HAL 路径与哈希。

direct build、provider build、AICPU package 和 runner 必须共享一个固定恢复根。
代码拒绝旧终端残留的 `KVAF_DIRECT_STATE_DIR`、`KVAF_DIRECT_RESULT_ROOT`、
`KVAF_DIRECT_*LOCK`、`KVAF_DIRECT_*FILE` 或 `KVAF_EID_LOCK_FILE` 分叉覆盖；若
看到 `cannot override the fixed direct URMA recovery root` 时，不要为了重跑硬件而
清空环境或状态。历史流程曾要求移除 `KVAF_GLOBAL_STATE_DIR`、
`KVAF_DIRECT_STATE_DIR`、结果根、EID lock 和 active/quarantine 路径覆盖；当前应停止，
保留现场并只运行静态审计或专用 recovery-only 入口。

首次从旧 `/run`/工作树状态版本升级时要安排一次维护窗口：确认旧 runner/worker
已经安全退出，或旧 quarantine 已按站点流程处理，再启用新版本。不要先删旧容器、
旧工作树或执行 `git clean -fdx`，因为旧版状态还没有迁入 `/data` 持久根。

## 7. 是否必须升级 CANN

绕过 HCOMM 本身不要求升级 CANN，也不再受
`COMM_PROTOCOL_UBC_CTP/ChannelHandle` 这类 HCOMM header 版本限制。

当前版本能否使用，以四项实测为准：

1. bisheng 能编译 `dav-3510` AIV + SIMT；
2. ACL/runtime 能执行 custom AICPU KFC；
3. compat loader 能加载上面的 Device provider；
4. 固定 UMDK/libummu 源码与当前运行内核的 UDMA/UMMU UAPI 完全匹配。

四项均通过就不必仅为 direct URMA 升级 CANN；任一项缺失才需要补匹配组件
或升级完整套件，不能只替换某个 include 目录。

## 8. 仍未完成的生产化工作

当前实现只是 A5 架构候选与静态能力审计对象，尚未达到可执行 MVP，
更不能宣称生产安全。在本节缺口和官方能力分层证据补齐前，
`HARDWARE_EXECUTION_DISABLED` 和 `ZERO_WQE_NOT_PROVEN` 必须保持：

- private accessor 仍依赖固定 UMDK 26.06 的 private UDMA 布局。最终应在
  UDMA provider 中新增 versioned
  `QUERY_DIRECT_QUEUE_INFO`，由 provider 自己返回 SQ/CQ 地址和 feature，
  彻底删除外部 private cast；
- source-built UDMA/UMMU 已替换上游会吞掉
  `ummu_ungrant`/`ummu_free_tid`、PLBI/PSYNC 与 userspace `munmap` 错误的销毁
  路径，并按 phase 保留 context/segment/token/Jetty/JFC/TID/queue/doorbell
  句柄。仍需在服务器 B 用当前驱动匹配的 full UAPI 完成 AArch64 构建和真机
  fault injection；本机全量单元测试不能替代这一步；
- provider 是进程级 singleton；当前候选 epoch claim 只支持一个连接/owner，尚未
  实现多租户 queue pool；
- 候选 owner 已把 Open/多轮 transfer/Close 合并到同一 AICPU invocation，并阻止
  外部 Open/Close 并发破坏状态；它仍需在目标 CANN 上验证长驻 KFC worker 的配额、
  watchdog、合作式取消和 runtime 退出契约，正式 runner 因此继续 fail closed；
- Host DVA 的 `non_pin` 注册和 AIV CQ cache 语义仍需当前 A5 固件实测；
- AIV 已用 32 个 SIMT lane 规划 descriptor，并按 SQ/CQ 深度分 wave 提交 4096 个
  WQE；当前仍是单 queue owner，尚未实现多 SQ 分片、图内异步完成和性能调优；
- CQ poll budget 到期会保留 owner/MR/queue 而不是冒险销毁；要做到无需 reset 的
  在线恢复，还需实现可续跑的 pending-CQ drain 状态机。

当前 smoke 已把 Host DDR/HBM pool、MR 和 Jetty/SQ/CQ 维持在整个 epoch 内，能够
验证多轮 BatchGet 复用；但每轮仍做 Host canary、D2H 校验和 Snapshot，所以不能
把 smoke 吞吐当作 decode 热路径性能。后续性能工作是图内 AccessTable 生产、异步
完成通知、doorbell 合并、尾部 fence/CQE、queue pool 与多 SQ 分片。
绕过 HCOMM 主要减少控制面依赖与建链开销；稳态带宽仍取决于长驻资源和批量
提交。
