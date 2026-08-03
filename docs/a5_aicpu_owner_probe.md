# A5 AICPU bounded owner probe 与分阶段 owner 设计

> **HISTORICAL / `HARDWARE_EXECUTION_DISABLED`**
>
> 本文保留 AICPU/AIV 控制记录协议和 owner 生命周期分析，但不再提供动态
> probe 或资源版 owner 的真机操作。即使 resource-free probe 不创建 URMA/UMMU
> 资源，它仍会进入 ACL、AICPU 和 AIV，因此不属于当前允许的静态检查。

## 1. 结论

`kvaf_a5_aicpu_owner_probe` 是一个不创建 URMA/UMMU 资源的并发可见性探针。
它只回答一个较小的问题：一个有明确超时的 AICPU KFC invocation 保持运行时，
另一个 AIV stream 能否通过 GM 控制记录看到 `READY`，并把带 session/sequence 的
`RELEASE` 传回同一次 AICPU invocation。

**Probe PASS 不会解锁 direct URMA。** 它没有执行 provider Open、AIV 数据传输或
provider Close，因而不能证明 UDMA/UMMU 句柄在完整生命周期内由同一个 AICPU
进程和 SO instance 持有。历史实现曾使用
`candidate-epoch-owner-unvalidated-v1` staged wrapper；当前全局
`HARDWARE_EXECUTION_DISABLED` 门禁同时禁止动态 owner probe 和资源版矩阵，
不存在环境变量、probe 输出或专用 wrapper 绕过。

## 2. 探针验证的协议

控制协议使用两个 64 字节、64 字节对齐的记录。command 只有外部执行流写，status
只有 AICPU owner 写，从结构上避免两端同时改写同一 cache line：

```text
Host enqueue aicpu_kvaf_urma_owner_probe_v1 on owner_stream
  -> AICPU publishes READY(session_id)
  -> distinct aiv_stream polls READY with DCCI cache maintenance
  -> AIV publishes RELEASE(session_id, sequence=1)
  -> Host synchronizes aiv_stream
  -> AICPU observes RELEASE and publishes RELEASED(sequence=1)
  -> Host synchronizes owner_stream
  -> Host copies status to CPU and validates the complete record
```

稳定 ABI 为：

- entry symbol：`aicpu_kvaf_urma_owner_probe_v1`；
- command：`kvaf_urma_owner_probe_command_v1`；
- status：`kvaf_urma_owner_probe_status_v1`；
- KFC args：`kvaf_urma_aicpu_owner_probe_args_v1`；
- command：`NONE=0`、`RELEASE=1`；
- state：`EMPTY=0`、`READY=1`、`RELEASED=2`、`TIMED_OUT=3`、`FAILED=4`；
- release sequence 固定为 1。

owner poll、AIV poll 和用户可配置 timeout 都必须有界。Host 还会把 provider
deadline 加上一个小的 grace 后写入 `rtAicpuArgsEx_t.timeout`，作为 stuck KFC task
的第二道 runtime 退出边界。成功标记是：

```text
KVAF_A5_AICPU_OWNER_PROBE_OK logical_device_id=N timeout_seconds=S
```

任何失败都必须非零退出，并输出：

```text
KVAF_A5_AICPU_OWNER_PROBE_FAILED phase=...
```

## 3. Probe、MVP owner 与 production owner 的边界

### 3.1 Resource-free probe

当前 probe 不打开 EID，不创建 URMA context、Jetty、JFR、JFC、SQ、CQ、segment、
token 或 UMMU mapping。它验证的是 AICPU/AIV 并发调度与控制记录可见性，而不是
DMA 生命周期。

### 3.2 MVP session owner

候选 `MVP session owner` 指一次 smoke run 对应一个 session nonce、一个 owner
invocation 和一个 provider context。仓内的
`aicpu_kvaf_urma_epoch_owner_v1` 已按这个结构实现：

```text
Open -> publish READY -> wait for quiesced command -> checked Close/retry -> exit
```

这样 process-local UDMA/UMMU 指针不会跨 KFC launch。provider 的进程级 epoch
claim 还会拒绝 owner 之外的 Open/Close。Host 关闭时先冻结新提交，再把 shutdown
和全部 BatchGet 排在同一 control/data stream，以 FIFO 证明没有 data kernel
落在 Close 后面。MVP 仍只支持单 session；worker 被 `SIGKILL`、ACL context 丢失
或控制记录不可读时，不能由新进程接管旧 owner，必须继续保留
active/quarantine 语义。只有资源版 owner 的 Open-transfer-Close fault matrix
全部通过后，才能评审是否把正式 runner 的源码 gate 改为
`single-invocation-owner-v1`。历史 epoch-owner candidate wrapper 只保留作代码
审计，当前不能运行真机验证。

### 3.3 Production epoch owner

`production epoch owner` 是可跨多个 BatchGet 复用的事件驱动服务，并不等于为
每次 H2D 创建一个 AICPU task。这里的 epoch 是 owner 每次
重建、设备 reset 或硬件隔离后递增的代际号。命令必须同时匹配
`epoch + session + sequence`，因此旧进程遗留的命令不能误关新资源。默认建议每张
NPU 至多保留一个 owner。lease 到期不能在队列未 drain 时强制 Close 或退出；它
只能触发诊断/拒绝新提交，随后等待合作式 shutdown 或驱动级隔离。如果目标 CANN
没有受支持的 blocking、watchdog、取消和 worker 配额契约，就不能把普通 KFC
worker 长驻方案作为正式方案。

生产版本还需要多 session 隔离、租约和 heartbeat、合作式取消、配额、公平调度、
durable tombstone、进程重启恢复、可审计资源计数以及 epoch rollover 测试。这些能力
不属于当前 resource-free probe，也不应为了尽快跑通 MVP 而伪装成已实现。

## 4. 必须完成的 probe matrix

下表是解除真实 URMA owner 门禁之前的完整验收矩阵，不是当前二进制已经覆盖的功能
清单。当前 `kvaf_a5_aicpu_owner_probe` 只执行基本握手、独立 stream 和成功路径的
cache 可见性；process-local contract test 覆盖 timeout/非法参数，但真机负向模式、
soak、调度压力和异常退出仍是后续工作。

| 类别 | 必测内容 | 失败含义 |
|---|---|---|
| ABI | 64/64/128 字节、64 字节对齐、固定 enum 和 sequence | Host/AICPU/AIV 解释不一致 |
| 基本握手 | `EMPTY -> READY -> RELEASED`，session 和 sequence 精确匹配 | 控制协议不可用 |
| 独立 stream | owner 未完成时 AIV stream 仍能运行；先 sync AIV，再 sync owner | 长驻 owner 阻塞数据面 |
| 有界性 | timeout 取 1、5、10 秒；缺少 RELEASE 必须进入 `TIMED_OUT` | owner 可能永久占用 worker |
| Cache 可见性 | command/status 分 cache line，DCCI 后重复握手 | GM cache/coherency 契约不足 |
| 负向协议 | 错 session、sequence 0/2、未知 opcode、stale READY | stale 命令可能操作新资源 |
| 重复与 soak | 每卡至少 10,000 次短测和 30 分钟持续测试 | 调度或可见性存在低概率失败 |
| 设备覆盖 | 服务器 B 的 NPU 0–7 分别运行 | 结果只适用于某个 device |
| 版本覆盖 | 记录 CANN、driver、firmware、kernel 和 provider hash | PASS 无法绑定部署版本 |
| 失败注入 | AIV 不 release、AIV sync 失败、owner timeout、D2H 失败 | 错误路径可能误报 PASS |
| 调度压力 | 同卡存在普通 AIV/AICPU 工作时重复探针 | owner 可能饿死其他工作 |
| 进程边界 | 合作式终止与宿主进程异常退出 | 明确哪些情况仍需 fail closed |

PID、TID、SO base 和 generation 可以随结果记录，但它们只是诊断。即使 10,000 次
都相同，也不能推出下一次独立 KFC Open/Close 一定命中同一 worker。

## 5. 历史操作记录（不可执行）

旧验证流程会构建 provider、部署 AICPU package、构建 probe，然后在隔离 NPU 上
启动动态 owner runner。这条流程已撤下：package 部署会改变 CANN
注册状态，probe 随后会调用 `aclInit` 并进入 AICPU/AIV；两者都不属于当前静态审计
范围。保留的历史关系如下，故意使用非 shell 记录格式：

```text
HISTORICAL_OWNER_PROBE_RECORD (NOT SHELL)
  build provider: scripts/build_a5_direct_urma_provider.sh
  deploy AICPU package: scripts/package_a5_direct_urma_aicpu.sh
  build probe: scripts/build_a5_direct_urma.sh
  dynamic runner: scripts/run_a5_aicpu_owner_probe.sh
  former parameters: logical_device_id=7, timeout_seconds=5
  execution: HARDWARE_EXECUTION_DISABLED
```

历史 wrapper 会在 `aclInit` 之前持有 deploy/active 两把锁，并验证 provider v3、
direct build manifest、owner-probe 哈希、真实 `libascendcl.so`、独立 KVAF AICPU tar，
以及 `ascend_package_load.ini` 中唯一且精确的注册块。旧成功前置标记为：

```text
KVAF_A5_AICPU_OWNER_PROBE_PREFLIGHT_OK
AICPU_REGISTRATION=verified-exact
```

打包脚本使用
`opp/vendors/cust/op_impl/aicpu/kernel/cann-kvaf-direct-urma-compat.tar.gz`，不会覆盖
UB_MicroBenchmark 的 `compat/liburma_perftest.tar.gz`。仅有 tar 而缺少 CANN 白名单
注册时，wrapper 会在任何 ACL/NPU 操作前失败；这正是旧版本出现 `507018` 和
Device 侧 `So does not exist` 的原因。

旧 timeout 范围是 1–10 秒。即使已有历史 PASS，也不能据此运行 direct smoke 或
epoch-owner candidate；当前没有环境变量、旧 marker 或风险接受短语可以解除门禁。

## 6. 最小否证标准

出现以下任一结果，都应停止 owner 方案的硬件解锁评审：AIV stream 被 owner stream
阻塞、READY/RELEASED 偶发不可见、超时后 owner 不退出、错误 session 被接受、同步
失败却输出 PASS，或同一已固定软件栈在不同 NPU 上行为不一致。此时应先修正 runtime
与 cache 协议，不能把重复次数或 PID 日志当成安全证明。
