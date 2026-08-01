# A5 AIV + SIMT + direct URMA + H2D

## 1. 结论

KVAF 现在有一条不调用、不链接 HCOMM 的实验性 H2D 路径：

```text
Host DDR(HVA)
  -> halHostRegister(HOST_MEM_MAP_DEV)
  -> Host DVA
  -> Host liburma 注册 MR、创建 Jetty/TP
  -> TCP 交换 EID、MR、Jetty、TP
  -> AICPU liburma 注册 HBM MR、创建 Jetty/SQ/CQ
  -> 返回 AICPU shared device-VA 的 SQ/CQ/doorbell/TP/MR
  -> 32-lane SIMT 规划 descriptor
  -> 单 AIV 直写 64B UDMA READ SQE
  -> SQ doorbell
  -> DCCI 后轮询 CQE，并更新 CQ doorbell
  -> Host DDR 数据进入 Device HBM
```

这里复刻的是 `UB_MicroBenchmark_new` 已跑通的资源模型：普通
`urma_create_jetty` 在 AICPU KFC 进程内执行，AIV 消费同一 unified
device-VA。代码不会把这个事实误标成 `CREATE_JETTY_EX/TGID-VA`；
最终可达性由真实的一条 SQE、一个 CQE、payload 和 canary 联合验证。

直接依赖中已经移除：

- memcache；
- memfabric；
- HCOMM headers、`libhcomm.so` 和 HCOMM resource API。

仍然需要：

- CANN/ACL/runtime、bisheng、Ascend HAL 和驱动；
- AICPU KFC 动态加载；
- source-linked public `liburma` core、系统 `liburma_common.so` 和内核
  UDMA/UMMU；
- 与运行驱动同代的 `urma_perftest/ubengine` 源码快照；
- 与服务器 B 当前驱动同一构建产物的完整 UMMU UAPI header。

KVAF 不再链接系统 `liburma.so`、`liburma-udma.so` 或 `libummu.so`。Host 与
Device provider 分别把 private liburma core、source-matched UDMA
（Host=`ST64B`、AICPU=`TGID_VA`）和 checked UMMU 编进自身；private core 在
编译期禁用动态 provider 目录扫描。构建与运行脚本同时检查 `DT_NEEDED` 和
`ldd`，发现外部 `liburma`/`liburma-udma`/`libummu` 会直接失败。系统
`liburma.so` 只作为已验证参考产物参与 public ABI 对比，不进入运行依赖。

这里的“绕过 HCOMM”是指 KVAF 自己完成资源控制。EID、MR、Jetty、TP
仍必须创建和交换，只是不再由 HCOMM/memfabric/memcache 代办。

## 2. 代码来源与职责

### 2.1 借鉴 `urma_perftest`

复用并在 KVAF provider 中源码编译的部分包括：

- `perftest_parse_args` 和配置校验；
- TCP server/client 与两端配置交换；
- `urma_create_context`；
- JFC、Jetty、token、segment 的创建参数和顺序；
- EID、segment、Jetty、TP 信息交换；
- remote segment/Jetty import 和 connect；
- 对称 teardown 的基本顺序。

KVAF 不再链接预编译 `liburma_perftest.so`。构建脚本把选中的源码、
Host/Device provider、系统 URMA 动态库哈希和 build-id 绑定到 manifest，
避免旧 SO 被误打包。

上游 fork 的 `fill_ai_jetty()` 会按一份复制的 private struct 读取 queue。
KVAF 的 checked 路径不调用它，而是仅用 public URMA API 创建资源，完成后
由一个独立 accessor 按实测 ABI 只读提取 SQ/CQ，避免创建阶段混用两套
private layout。

### 2.2 借鉴 `UB_MicroBenchmark_new`

设备数据面复刻其 AIV 直写 UDMA 的关键规则：

- 64B READ SQE；
- remote EID 的硬件字节序；
- 24-bit TP ID；
- public `token_id >> 8` 得到 20-bit UDMA TID；
- READ local SGE 的 token word 保持 0，使用已注册 local VA；
- SQ doorbell offset；
- CQ owner/phase、DCCI 和 CQ consumer doorbell。

KVAF 增加了 descriptor 越界检查、单 owner、stream 同步、结果/canary
验证和失败隔离。

### 2.3 Host DDR

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

## 3. 为什么要分成 Host/Device 两个 provider

标准 `urma_query_*` API 不返回 userspace SQ/CQ 的 `qbuf`、硬件
doorbell 或 JFC `sw_db`。`urma_user_ctl(CREATE_JETTY_EX)` 只能解决新建
SQ 的部分信息，也不能查询普通 JFC 的完整 CQ，因此当前 AIV direct
路径仍需要一个受控的 UDMA 私有访问层。

两端使用同一个经过 `UB_MicroBenchmark_new` 随附 SO 验证的
`urma/code/include/urma_types.h` AIV private layout，但编译成两个角色
不同的 provider：

| 端 | provider private ABI | KVAF 产物 |
|---|---|---|
| Host | `PERFTEST_AIV_V1 / Host` | `libkvaf_urma_provider_host.so` |
| AICPU/Device | `PERFTEST_AIV_V1 / AICPU shared VA` | `libkvaf_urma_provider_device.so` |

不能用 `ubengine/ssapi` 里另一代 private header 替代它：两代 public
JFC/Jetty base 和 queue tail 不同，会让 SQ、CQ、`sw_db` 全部错位。
`CREATE_JETTY_EX(tg_id)` 适用于其他进程代建并把 doorbell 映射到目标
TGID 的路径；本实现的资源本来就在 AICPU 内创建，强行改用它反而偏离
已验证基线。

KVAF 把 private header 限制在
`provider/private_udma_accessor.c` 一个翻译单元中，并为 Host/Device 角色
分别编译。
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
  -> establish TCP connection
  -> create context
  -> create JFC/Jetty
  -> register local segment
  -> exchange EID/MR/Jetty/TP
  -> import peer segment/Jetty
  -> connect
  -> validate exact private queue ABI
  -> provider-ready 双端确认
```

AIV 使用的字段来源如下：

| 字段 | 来源 |
|---|---|
| SQ base/depth/doorbell | `PERFTEST_AIV_V1` Device private accessor |
| CQ base/depth/sw_db | `PERFTEST_AIV_V1` Device private accessor |
| remote MR | `remote_seg[0].ubva/len` |
| remote TID | remote public segment `token_id >> 8` |
| local object ID | local public segment `token_id >> 8`，仅用于资源校验，不写入 compact READ WQE |
| remote EID | imported target Jetty public EID，AIV 只反转一次 |
| TP ID | imported target Jetty 的 `tp.tpn` |
| token policy | `URMA_TOKEN_NONE`，SQE `tokenEn=0` |

不能使用旧 sidecar 的 `ai_jettys[].sq.tp_id`。当前 UDMA provider 没有实现
`urma_ops.get_tpn`，该值可能是负错误码转换后的无符号数。

## 5. 失败与 teardown

创建过程按阶段互相确认。一端失败会 shutdown TCP 控制连接，另一端不会
无限阻塞。正常关闭顺序是：

```text
AIV stream 完成
  -> SQ PI == CQ CI
  -> 两端 disconnect/unimport Jetty
  -> barrier
  -> 两端 unimport peer segment
  -> barrier
  -> unregister local segment/token
  -> 删除 Jetty/JFC/context
```

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

需要注意：本版本的 Device provider 仍要求同一 AICPU 进程和 SO instance 执行
Open/Close。`UB_MicroBenchmark_new` 在同一 stream 上分多次调用，已有日志也曾观察
到相同 PID/TID；但本地公开 KFC ABI 只提供 associated stream，没有 worker、进程或
SO instance affinity token。对多次调用采样 PID、进程 cookie、SO base 和 generation
只能描述过去的调度，不能绑定随后真正持有资源的 Open/Close，因此不能作为安全
前置证明。

新硬件运行现在由 `scripts/a5_aicpu_affinity_guard.sh` 默认 fail closed。旧 active
session/quarantine 仍会先恢复或报告；新运行则会在 EID selector、`npu-smi`、ACL
context 和 URMA/UMMU 资源之前退出，且没有环境变量绕过。只有以下任一条件经代码和
目标 CANN 版本验证后，才能修改该源码门并重新开放硬件 runner：

- 厂商提供同一 stream 的 KFC 调用固定在同一进程/SO instance 的正式 runtime
  契约，KVAF 同时校验该契约适用于当前 CANN/固件；
- 将 Open、等待 AIV 数据面和 Close 合并到一个 AICPU invocation，或实现一个长驻
  ownership service，使 liburma/UDMA/UMMU 指针和句柄始终由同一进程持有。

## 6. 操作命令

### 6.1 服务器 B 宿主机：更新代码

执行环境：`SERVER_B_HOST`

```bash
echo "EXECUTION_ENV=SERVER_B_HOST"
cd /data/c00982601/KVAF

git switch A5_AIV_URMA_cs
git pull --ff-only origin A5_AIV_URMA_cs
git log -1 --oneline

test -r \
  /data/c00982601/urma_perftest/ubengine/ssapi/userspace/udma/udma_u_common.h

docker ps --filter name=vllm-ascend-cs-kvaf
docker exec -it -u 0 vllm-ascend-cs-kvaf bash
```

不需要新建容器。`/data` 已由宿主机挂载到同路径，后续命令都在
`vllm-ascend-cs-kvaf` 内执行。

### 6.2 Docker：加载环境

执行环境：`VLLM_ASCEND_CS_KVAF_DOCKER`

```bash
echo "EXECUTION_ENV=VLLM_ASCEND_CS_KVAF_DOCKER"
cd /data/c00982601/KVAF
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64:/usr/lib64:${LD_LIBRARY_PATH:-}
export KVAF_URMA_PERFTEST_ROOT=/data/c00982601/urma_perftest
export KVAF_DIRECT_PROVIDER_BUILD_DIR=/data/c00982601/KVAF/build-a5-direct-urma-provider
export KVAF_REFERENCE_PERFTEST_LIBRARY=/data/c00982601/UB_MicroBenchmark_new/3rd/urma_perftest/liburma_perftest.so
```

先定位与服务器 B 当前驱动同一构建产物的完整 UMMU UAPI。不要使用
`urma_perftest/drivers/ubus/kernel` 下缺少 `hw_cap` 和 PLBI ioctl 的旧 header：

```bash
find /usr/src "/lib/modules/$(uname -r)/build" /usr/local/Ascend /data/c00982601 \
  -type f -path '*/include/uapi/linux/ummu_core.h' -print 2>/dev/null \
  | sort -u

# 对每个候选执行；六类定义必须全部存在，并确认它来自当前驱动包。
grep -nE \
  'hw_cap|HW_CAP_(EXPAN|DOUBLE_PLBI|KCMD_PLBI)|UMMU_IOCPLBI_(VA|ALL)' \
  /exact/driver-source/include/uapi/linux/ummu_core.h

export KVAF_UMMU_UAPI_ROOT=/exact/driver-source
test -r "$KVAF_UMMU_UAPI_ROOT/include/uapi/linux/ummu_core.h"
```

如果找不到该 header，停止在这里并补充当前驱动对应的开发/源码包；不能从旧
header 猜结构体布局或 ioctl 数值。构建会记录精确路径和 SHA-256，但“与运行
驱动同源”仍必须由服务器软件包来源保证。

若 `ldconfig` 中同一个 SONAME 有多份候选，显式指定 public ABI 参考库与实际
`liburma_common` runtime：

```bash
export KVAF_URMA_LIBRARY=/usr/lib64/liburma.so.0
export KVAF_URMA_COMMON_LIBRARY=/usr/lib64/liburma_common.so.0
```

`KVAF_REFERENCE_PERFTEST_LIBRARY` 只用于把
`UB_MicroBenchmark_new` 已验证过的 AIV private layout 产物记录进
build manifest，KVAF provider 不链接它。若该路径不存在，构建脚本才回退到
`urma_perftest/urma/code/output` 中的同名产物。

### 6.3 Docker：构建两端 provider

执行环境：`VLLM_ASCEND_CS_KVAF_DOCKER`

```bash
bash scripts/build_a5_direct_urma_provider.sh

ls -l \
  "$KVAF_DIRECT_PROVIDER_BUILD_DIR/libkvaf_urma_provider_host.so" \
  "$KVAF_DIRECT_PROVIDER_BUILD_DIR/libkvaf_urma_provider_device.so" \
  "$KVAF_DIRECT_PROVIDER_BUILD_DIR/kvaf_ummu_uapi_probe" \
  "$KVAF_DIRECT_PROVIDER_BUILD_DIR/build.manifest" \
  "$KVAF_DIRECT_PROVIDER_BUILD_DIR/build.success"
```

脚本执行 fresh build，校验导出符号、AArch64 ELF、`DT_NEEDED`、RPATH、
源码前后哈希、Host/Device private-ABI probe、public liburma ABI，以及 pinned
UMMU UAPI。下面两条都必须无输出：

```bash
readelf -d "$KVAF_DIRECT_PROVIDER_BUILD_DIR/libkvaf_urma_provider_host.so" \
  | grep -E 'NEEDED.*lib(urma|urma-udma|ummu|hcomm)\.so' || true
readelf -d "$KVAF_DIRECT_PROVIDER_BUILD_DIR/libkvaf_urma_provider_device.so" \
  | grep -E 'NEEDED.*lib(urma|urma-udma|ummu|hcomm)\.so' || true

grep -E \
  '^(urma_core_mode|urma_library_linkage|udma_provider_mode|ummu_provider_mode|ummu_uapi_header_sha256|ummu_uapi_abi|ummu_uapi_abi_sha256|ummu_uapi_probe_sha256|udma_teardown_contract)=' \
  "$KVAF_DIRECT_PROVIDER_BUILD_DIR/build.manifest"

"$KVAF_DIRECT_PROVIDER_BUILD_DIR/kvaf_ummu_uapi_probe"
```

期望 manifest 包含 `urma_core_mode=source-linked-no-dynamic-providers`、
`source-built-internal`、完整的结构体 offset/ioctl/capability
指纹和 `udma_teardown_contract=checked-retryable-v1`。runner 在任何硬件资源创建
前会重新执行并校验这份 probe；header、probe 或 manifest 任一个变化都会退出 2。

### 6.4 Docker：安装 AICPU provider

执行环境：`VLLM_ASCEND_CS_KVAF_DOCKER`

```bash
bash scripts/package_a5_direct_urma_aicpu.sh

tar -tzf \
  "$ASCEND_HOME_PATH/compat/liburma_perftest.tar.gz"
```

archive 名称保留 `liburma_perftest.tar.gz` 只是为了适配当前 CANN compat
loader；内容不再包含旧 `liburma_perftest.so`，精确白名单是：

```text
aicpu_kernels_device/libkvaf_urma_provider_device.so
aicpu_kernels_device/kvaf-direct-urma-provider.manifest
aicpu_kernels_device/kvaf-direct-urma-source.sha256
```

### 6.5 Docker：构建 AIV/SIMT smoke

执行环境：`VLLM_ASCEND_CS_KVAF_DOCKER`

```bash
if test -r \
  /data/c00982601/asc-devkit-9.1.0-20260722/include/simt_api/asc_simt.h
then
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

当前版本执行下面命令时会由 AICPU affinity gate 以退出码 2 停止；这是预期的
fail-closed 结果，不应通过设置环境变量继续。先完成第 5 节所述正式 runtime 契约
验证或长驻 owner 改造，再由代码评审修改源码门。保留命令仅用于确认门禁位置和为
后续实现准备参数。

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
export KVAF_H2D_BYTES=4096
export KVAF_H2D_TIMEOUT_SECONDS=120
export KVAF_DIRECT_CONTROL_PORT=22117
export KVAF_A5_RUN_TAG="npu7_checked_$(date +%Y%m%d_%H%M%S)_$$"

bash scripts/run_a5_direct_urma_h2d_smoke.sh
```

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
4. public `liburma`、`ubengine` 源码、完整 UMMU UAPI 与运行驱动同代。

四项均通过就不必仅为 direct URMA 升级 CANN；任一项缺失才需要补匹配组件
或升级完整套件，不能只替换某个 include 目录。

## 8. 仍未完成的生产化工作

当前实现用于 A5 正向 MVP 和性能验证，还不能宣称生产安全：

- private accessor 仍依赖 `PERFTEST_AIV_V1` 布局。最终应在实际
  `liburma-udma` provider 中新增 versioned
  `QUERY_DIRECT_QUEUE_INFO`，由 provider 自己返回 SQ/CQ 地址和 feature，
  彻底删除外部 private cast；
- source-built UDMA/UMMU 已替换上游会吞掉
  `ummu_ungrant`/`ummu_free_tid`、PLBI/PSYNC 与 userspace `munmap` 错误的销毁
  路径，并按 phase 保留 context/segment/token/Jetty/JFC/TID/queue/doorbell
  句柄。仍需在服务器 B 用当前驱动匹配的 full UAPI 完成 AArch64 构建和真机
  fault injection；本机全量单元测试不能替代这一步；
- provider 是进程级 singleton，仅支持一个连接；
- AICPU Open/Close 必须命中同一 KFC worker 和 SO 实例。当前 runner 在任何新硬件
  访问前 fail closed；同-stream PID/cookie/SO-base/generation 采样不是正式保证。
  解除该门需要长驻 AICPU ownership service/单 invocation 生命周期，或当前目标
  CANN 对同-stream worker affinity 的正式 runtime 契约；
- Host DVA 的 `non_pin` 注册和 AIV CQ cache 语义仍需当前 A5 固件实测；
- 当前逐 descriptor `doorbell -> CQ`，还没有批量 WQE/doorbell/CQ。

正确性通过后的性能工作依次是：长驻 context、Host DDR pool 一次注册、
SIMT 批量生成 WQE、一次 doorbell 提交一批、尾部 fence/CQE、多 SQ 分片。
绕过 HCOMM 主要减少控制面依赖与建链开销；稳态带宽仍取决于长驻资源和批量
提交。
