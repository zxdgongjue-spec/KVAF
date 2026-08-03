# A5 AIV + SIMT + direct URMA + H2D

## 1. 结论

KVAF 现在有一条不调用、不链接 HCOMM 的实验性 H2D 路径：

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

Host 与 Device provider 都把固定版本 UMDK 的 common、完整 public URMA
core、UDMA 和官方 perftest UB/TCP 管理代码，以及配对 libummu 编进自身；private
core 在编译期禁用动态 provider 目录扫描。两端 UDMA 源码不再通过 `ST64B` 或
`TGID_VA` 编译宏伪造不同布局，只在 private accessor 上区分 Host 和 AICPU
执行域。构建与运行脚本同时检查 `DT_NEEDED` 和 `ldd`，发现外部
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

KVAF 把 private header 限制在
`provider/private_udma_accessor.c` 一个翻译单元中，并为 Host/Device 角色
分别编译。角色宏只选择 accessor 的 Host 或 AICPU 执行域；UMDK UDMA 源码本身
不定义 `ST64B`/`TGID_VA` 角色宏。
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
即可 `urma_import_jetty()` 并发送握手。Host 从接收 CQE 的 `remote_id` 得到 Device
EID/Jetty ID，随后双方用 URMA SEND/RECV 交换正式 data MR、Jetty 和 TP 元数据。
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

当前 active/quarantine 是整机 KVAF singleton，不按 NPU 分片。一个未解决的
session 会阻止换另一张卡继续测试，这是有意的 fail-closed 约束；在证明各 NPU
的 UDMA/driver ownership 完全隔离前，不要通过复制仓库或删除锁绕过它。

active 状态在每次 fork 前先写 `host_launch_intent` 或
`device_launch_intent`，fork 成功后再补 PID。恢复逻辑按 phase 决定必须检查
单端还是双端终态，绝不会因为 SIGKILL 恰好发生在 fork 与 PID 落盘之间，就把
`pid=0` 当成“进程从未启动”。若终态已经证明安全但 worker 尚未退出，状态会
保留为 `safe_terminal_process_stuck`；worker 继续持有 active-session 锁。下次
执行会先打印记录的 PID，待 worker 自然退出后再按同一 `run_id` 自动复核并清理
active 状态，全程不会再次访问 topology、EID selector 或 NPU。

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

Device provider 的 UMDK/UMMU 句柄必须由创建它们的同一 AICPU 进程和 SO instance
销毁。公开 KFC ABI 没有跨 invocation 的 worker affinity token，所以对多个短
invocation 采样到相同 PID/TID 不能作为所有权证明。

候选实现因此把 `Open -> 多轮 AIV BatchGet -> checked Close/retry` 放进同一个
`aicpu_kvaf_urma_epoch_owner_v1` invocation。provider 内还有进程级 epoch claim；
owner 存活时，其他 public/legacy Open/Close 都会被拒绝，避免第二个调用破坏它的
process-local 状态。Host 侧冻结新提交后，关闭命令与此前的 BatchGet kernel 排在
同一 control/data stream；stream FIFO 与同步结果才是关闭前 drain 的顺序证明。
GM 中的四个软件计数是额外的一致性证据，不是硬件 PI/CI 的替代品。

生产 runner 继续默认 fail closed；只有显式的
`run_a5_direct_urma_epoch_owner_validation.sh` 才能执行候选真机验证。候选 owner
解决了 split Open/Close 的所有权缺口，但普通 KFC worker 是否允许跨整个模型
生命周期驻留，仍需验证当前 CANN 的 worker 配额、watchdog、取消和阻塞契约。
resource-free bounded probe 也仍只证明 AICPU/AIV 并发控制记录可见性。完整 epoch
边界和 decode 热路径见 [A5 direct URMA Decode H2D epoch 设计](a5_decode_epoch_h2d.md)。

## 6. 操作命令

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

### 6.4 Docker：安装 AICPU provider

执行环境：`VLLM_ASCEND_CS_KVAF_DOCKER`

```bash
bash scripts/package_a5_direct_urma_aicpu.sh

tar -tzf \
  "$ASCEND_HOME_PATH/opp/vendors/cust/op_impl/aicpu/kernel/cann-kvaf-direct-urma-compat.tar.gz"
```

KVAF 使用独立包名 `cann-kvaf-direct-urma-compat.tar.gz`，不会覆盖
UB_MicroBenchmark 的 `compat/liburma_perftest.tar.gz`。安装脚本会把归档原子写入
`opp/vendors/cust/op_impl/aicpu/kernel/`，并在
`${ASCEND_HOME_PATH}/conf/ascend_package_load.ini` 中保留且仅保留以下注册块：

```text
name:cann-kvaf-direct-urma-compat.tar.gz
install_path:2
optional:true
package_path:opp/vendors/cust/op_impl/aicpu/kernel
load_as_per_soc:false
```

仅复制 tar 不会使 CANN/TDT 下发它；package 成功和 runtime preflight 都会严格
校验归档路径、内容、哈希及上述注册块。`optional:true` 只避免 KVAF 未使用时影响
其他 ACL 应用，KVAF 自己缺包或未注册时仍会在任何 URMA/NPU 操作前 fail closed。
该包不是外部 perftest 依赖，内容也不包含旧 `liburma_perftest.so`。

安装器先确认 registry 中的 KVAF 块“完全不存在或已经精确正确”，再原子发布经过
校验的 tar，最后原子追加 registration。首次安装若 registration 写入失败，新 tar
仍未注册、不会被 TDT 下发；重复安装时已有 registration 不改写，tar 始终以完整文件
原子替换；若替换后的复验或注册步骤失败，安装器会原子恢复旧的已注册 tar。重复或
畸形 registration 不会被静默修复，而会要求先人工核对。旧
`liburma_perftest.tar.gz` 文件可以保持未注册状态，但 registry 中不能再有同名旧注册，
避免两个包向 Device 解压同一个 provider SO。

这些文件位于 `vllm-ascend-cs-kvaf` 的容器 writable layer。停止/启动同一个容器会
保留它们，但删除并重建容器会丢失；重建后必须重新运行 package 脚本。正式部署应把
归档与 registration 固化进镜像，或放入受控的容器启动部署步骤。
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

### 6.6 Docker：精确确认 UDMA device/EID index

执行环境：`VLLM_ASCEND_CS_KVAF_DOCKER`

runner 现在以 EID 为 key 分别读取 Host `/sys/class/ubcore` 与 Device
`hccn_tool -g -dev_info`，不再从 NPU 号猜设备名。NPU7 的已确认映射可独立复核：

```bash
python3 scripts/resolve_a5_urma_endpoints.py \
  --physical-device-id 7 \
  --host-eid 00000000003f050000100000df080b00 \
  --device-eid 000000000000020000100000df00e100 \
  --shell
```

期望输出为 `udmac0d1e5/eid0` 与 `udmac0d0e2/eid0`。手工设置这些变量时，
runner 仍会重新做 exact-match；任一不一致都会在创建资源前退出 2。

Host/Device peer EID 默认由
`scripts/resolve_a5_urma_eids.py` 从 `/proc/asdrv_ub/pair_info` 解析，不依赖
memcache。`dev_id` 是宿主机全局 selector；新代码统一通过宿主机和 Docker 都能
看到的 `/data/.kvaf-state/asdrv-ub-pair.lock` 串行化完整的
`write dev_id -> read pair_info -> validate/retry`。迁移期还会取得同一 namespace
中的旧 `/run/lock/kvaf_asdrv_ub_pair.lock`。不同容器的 `/run` 不能互相保护，
因此升级期间不要并发运行旧 resolver；所有调用方升级后以 `/data` 锁为准。

### 6.7 Docker：运行 H2D

执行环境：`VLLM_ASCEND_CS_KVAF_DOCKER`

正式 runner 仍会由 AICPU gate 以退出码 2 停止。真机验证必须使用显式 candidate
wrapper；它在同一个 AICPU invocation 内持有 Device provider，并在同一个
control/data stream 上顺序执行全部 BatchGet 与 shutdown。默认负载是 4096 个
1 KiB 请求（每批 4 MiB），不是旧的“总共 4096 bytes”。

```bash
source /tmp/kvaf_npu7_pair.env
export KVAF_HOST_EID KVAF_DEVICE_EID KVAF_PAIR_CHANNEL
export KVAF_LOGICAL_DEVICE_ID=7
export KVAF_PHYSICAL_DEVICE_ID=7
export KVAF_HOST_URMA_DEVICE=udmac0d1e5
export KVAF_HOST_EID_INDEX=0
export KVAF_DEVICE_URMA_DEVICE=udmac0d0e2
export KVAF_DEVICE_EID_INDEX=0
export KVAF_PAIR_CHANNEL=0
unset KVAF_DIRECT_ALLOW_BUSY_NPU
unset KVAF_H2D_BYTES
export KVAF_H2D_TRANSFER_COUNT=4096
export KVAF_H2D_BYTES_PER_TRANSFER=1024
export KVAF_H2D_DESCRIPTOR_CAPACITY=4096
export KVAF_H2D_BATCH_ITERATIONS=10
export KVAF_H2D_TIMEOUT_SECONDS=120
export KVAF_H2D_RUN_TIMEOUT_SECONDS=600
unset KVAF_DIRECT_SERVER_START_DELAY_SECONDS
export KVAF_DIRECT_SERVER_READY_TIMEOUT_SECONDS=30
# 默认按 Device timeout + management-ready timeout + 5 秒计算；通常无需覆盖。
unset KVAF_HOST_CONTROL_TIMEOUT_SECONDS
export KVAF_DIRECT_MANAGEMENT_MODE=ub
# 可省略；NPU7 默认推导为 507。显式设置时必须位于 1..1023。
export KVAF_DIRECT_MANAGEMENT_JETTY_ID=507
# UB 模式不读取或传递任何 TCP/IP 控制参数。
unset KVAF_DIRECT_CONTROL_PORT
unset KVAF_DIRECT_CONTROL_SERVER_ADDRESS
unset KVAF_DIRECT_CONTROL_BIND_ADDRESS
export KVAF_A5_RUN_TAG="npu7_checked_$(date +%Y%m%d_%H%M%S)_$$"

bash scripts/run_a5_direct_urma_epoch_owner_validation.sh
```

runner 不再用固定 sleep 猜测 Host server 是否启动。它固定本次 Host PID 的
`start_ticks` 和可执行文件，并只接受本次 Host provider 写入日志的完整
`KVAF_URMA_MGMT_READY role=server jetty_id=<id> local_eid=<eid>` 行；只有确认该 PID
已创建管理 Jetty 并预投 RECV，才立即启动 Device。旧
`KVAF_DIRECT_SERVER_START_DELAY_SECONDS` 会直接报参数错误，避免遗留的 5 秒延迟
继续消耗 Host 管理握手 deadline。

Device provider 的控制超时仍由 `KVAF_H2D_TIMEOUT_SECONDS` 指定。Host 的管理窗口
独立使用 `KVAF_HOST_CONTROL_TIMEOUT_SECONDS`；默认值是 Device timeout 加 management
ready timeout 再加 5 秒，并在 manifest 中分别记录。AIV 对 AICPU owner 的 READY
观察使用单次一致性快照和 10 ms Host 节拍，不再设置一个会与 provider Open/cleanup
竞争的第二截止时间。普通管理握手超时必须等 owner 发布 `OPEN_FAILED_SAFE` 后按退出码
3 收敛；若 owner 本身长期无终态，runner 以退出码 5 保留 live worker 和 active
state，而不会仅因观察超时制造 quarantine。

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

只有明确接受业务干扰和后续 reset 可能影响同卡业务时，才可在同一 Docker 终端
显式绕过 busy 或 unproven 状态：

```bash
export KVAF_DIRECT_ALLOW_BUSY_NPU=1
```

正常测试必须保持该变量未设置。旧的
`KVAF_DIRECT_ALLOW_UNVERIFIED_UDMA_TEARDOWN` 已被删除；新 runner 只接受源码内置
UDMA+UMMU 和 `checked-retryable-v1`，没有回退到系统 provider 的风险开关。

运行后的处理只看退出码和最终终态：

```bash
set +e
bash scripts/run_a5_direct_urma_h2d_smoke.sh
kvaf_rc=$?
set -e
case "$kvaf_rc" in
  0) echo "PASS" ;;
  2) echo "PRECHECK_FAILED：未创建资源，修正参数/构建后重试" ;;
  3) echo "FAIL_SAFE：无需复位，可直接重试" ;;
  4) echo "FAIL_UNSAFE：保留进程和 backing，按 quarantine 指引处理" ;;
  5) echo "PENDING_UNPROVEN：不要复位或杀进程，稍后重跑脚本自动复核" ;;
  130) echo "RUNNER_DETACHED：pair 已记录，稍后重跑脚本自动复核" ;;
  *) echo "未分类 runner 错误：rc=$kvaf_rc；保留状态并检查输出" ;;
esac
```

若存在多个 active pair，按错误信息指定：

```bash
export KVAF_PAIR_CHANNEL=<实际channel>
bash scripts/run_a5_direct_urma_h2d_smoke.sh
```

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
看到 `cannot override the fixed direct URMA recovery root`，只需在当前 Docker
终端执行：

```bash
unset KVAF_GLOBAL_STATE_DIR KVAF_DIRECT_STATE_DIR KVAF_DIRECT_RESULT_ROOT
unset KVAF_EID_LOCK_FILE KVAF_EID_LEGACY_LOCK_FILE
unset KVAF_DIRECT_DEPLOY_RUN_LOCK KVAF_DIRECT_ACTIVE_SESSION_LOCK
unset KVAF_DIRECT_ACTIVE_SESSION_FILE KVAF_DIRECT_QUARANTINE_FILE
```

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

当前实现用于 A5 正向 MVP 和性能验证，还不能宣称生产安全：

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
