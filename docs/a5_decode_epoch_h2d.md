# A5 direct URMA Decode H2D epoch 设计

## 1. 结论

Decode offloading 不能为每个 token、每次 `BatchGet` 重新注册 DDR/HBM，也不能
重复创建 EID、TP、Jetty、JFC、SQ、CQ 和 MR。正确边界是一个 **transport
epoch**：KVAF 在 epoch 开始时注册稳定内存池并建链，随后用同一条 URMA 通道执行
多轮 H2D；只有 Runtime 退出、内存池地址变化、设备复位、链路失效或计数器接近
epoch 上限时才关闭并重建。

当前 direct URMA POC 使用 Device pull 模式：AIV 向 SQ 写 URMA READ WQE，从 Host
DDR 读取数据并写入 HBM。这样建链完成后，每轮热路径不需要 Host 再调用注册或
建链 API；Host 只需在固定容量的设备 Buffer 中更新本轮 AccessTable、索引、Mask、
输出偏移和有效条数。图模式下，这些更新也应由图内生产者或固定地址 Buffer 完成。

默认验证负载定义为：

```text
transfer_count        = 4096
bytes_per_transfer    = 1024
bytes_per_batch       = 4096 * 1024 = 4 MiB
batch_iterations      = 1（可调；大于 1 用于验证同一 epoch 复用）
descriptor_capacity   = 4096（图捕获时固定，runtime count 不得超过它）
```

旧的 `--bytes=4096` 表示“一批总共 4 KiB”；兼容模式把它映射成一个 4 KiB
descriptor。它不是 4096 个 1 KiB 请求。新参数不能和旧参数混用，避免静默改变
历史命令的含义。

runner 使用下面四个彼此独立的形状参数：

| 环境变量 | 含义 | 默认值 |
| --- | --- | --- |
| `KVAF_H2D_TRANSFER_COUNT` | 当前批有效 descriptor 数 | `4096` |
| `KVAF_H2D_BYTES_PER_TRANSFER` | 每个 descriptor 的统一长度 | `1024` |
| `KVAF_H2D_DESCRIPTOR_CAPACITY` | epoch 内固定的最大 descriptor 数 | `4096` |
| `KVAF_H2D_BATCH_ITERATIONS` | 同一建链/MR 上连续执行的批次数 | `1` |

device ABI 的 `entry_count` 可以在 `0..descriptor_capacity` 内逐批变化；当前 smoke
为了可重复对比，在一次 run 的所有 iteration 中使用同一个 transfer count。零项只
允许 candidate 验证，用于测试不发 WQE 的 Open/Close。当前 smoke 的每项长度相同，
但 device ABI 已经从 AccessTable 读取逐项 length；正式 Runtime 若要混合 KV block
大小，应直接填逐项 length，而不是为每种大小重建连接。
`KVAF_H2D_RUN_TIMEOUT_SECONDS` 只控制 Host/Device worker 的整次监督时间，不改变
provider 控制连接 timeout；长时间 soak 应单独把它调大。

## 2. 为什么初始化必须与 BatchGet 分离

一次 direct URMA 初始化包含 TCP 控制连接、EID/MR/Jetty/TP 交换、UMMU 映射、
Jetty/JFC/SQ/CQ 创建和 Host DDR/NPU HBM 注册。这些操作既慢，又改变资源所有权；
如果每轮 decode 都重复执行，会把控制面延迟放进 token 热路径，并显著扩大失败和
teardown 风险。

TCP 控制地址与 URMA EID 是不同地址域：runner 必须用
`KVAF_DIRECT_CONTROL_SERVER_ADDRESS` 显式提供数值 IPv4/IPv6 地址，client 可用
`KVAF_DIRECT_CONTROL_BIND_ADDRESS` 选择同族源地址；server 始终 wildcard listen。
EID 只参与 URMA endpoint 和交换结果校验，绝不能传给 TCP `-S/--bind_ip`。

因此生命周期应拆成三段：

| 阶段 | 执行动作 | 频率 |
| --- | --- | --- |
| Epoch 初始化 | 分配/注册 Host DDR 池和 HBM 池；创建并交换 EID、MR、TP、Jetty、SQ/CQ；发布 READY | Runtime/内存池代际一次 |
| Decode 热路径 | 更新固定地址的 AccessTable/indices/mask/count；AIV+SIMT 生成 descriptor；分 wave 写 SQ、敲 doorbell、收 CQ | 每轮 BatchGet |
| Epoch 关闭 | 停止新提交；等待 SQ/CQ 完全 drain；在原 owner 中 checked Close；注销 MR；最后释放 DDR/HBM | Runtime 退出或 epoch rollover 一次 |

在一个 epoch 内，KV block 可以在已注册池中复用不同 offset。只要 MR 覆盖范围、
token 和 backing address 不变，地址表内容变化不要求重新注册。以下变化必须重建：

- Host DDR 或 HBM 池被释放、扩容后迁移，或地址不再位于现有 MR；
- 对端 EID、设备或 provider generation 变化；
- TP/Jetty/SQ/CQ 报错且不能在原 owner 中证明安全恢复；
- 设备复位、容器/Runtime 重启导致旧资源所有权失效；
- 软件绝对队列计数接近约定的 epoch rollover 阈值。

## 3. 多轮 BatchGet 的数据面

图捕获时固定 `descriptor_capacity`、输出池容量和执行路径；图回放只修改
`entry_count`、地址、长度、索引、Mask 和 generation。一次 4096×1 KiB 批次应由
一次 AIV kernel 处理，而不是 Host 发起 4096 次 kernel launch。

SIMT 的 32 个 lane 以 stride 方式规划全部 descriptor。SQ/CQ 深度通常小于 4096，
所以标量 queue owner 按 `min(SQ 可用项, CQ 可用项)` 分 wave：先提交一波 WQE，
再回收对应 CQ，随后复用 ring slot。这样 descriptor 数量不再被硬件 queue depth
限制，也不会覆盖尚未完成的 SQ/CQ 项。

每轮完成必须同时满足：

```text
planned_count == submitted_count == completed_count == active_count
sq_head == sq_tail == cq_tail == wqe_count
completion.generation == request.generation
```

关闭前先在 Host 对资源对象设置 `frozen`，拒绝新的 BatchGet；随后把 shutdown
command kernel 排在同一个 control/data stream。该 stream 的 FIFO 和成功同步证明
所有已接受的 data kernel 都先完成。关闭命令还携带 epoch、session、sequence 和
最终四个队列计数；计数相等是额外的软件 drain evidence，不能代替 stream 顺序，
也不能被描述成硬件 PI/CI 证明。

## 4. 为什么使用 epoch AICPU owner

当前固定版本 UMDK/UMMU 把 context、UMMU mapping 和 teardown cursor 保存在创建
它们的进程/SO instance 中。公共 KFC launch ABI 不能保证两个独立 launch 落在同一
worker，因此“一个 launch Open、另一个 launch Close”不安全。

候选实现使用一个 AICPU invocation 完成整个 Device epoch：

```text
provider Open
  -> publish READY
  -> AIV 执行多轮 BatchGet（owner 不参与每个 WQE）
  -> AIV publish SHUTDOWN_QUIESCED
  -> 同一 invocation checked Close/retry
  -> publish CLOSED
```

这不是“每次 H2D 占一个 AICPU”。它是每张 NPU/每条通道一个资源 owner，热路径由
AIV 直接操作 SQ/CQ。POC 中保持一个 owner 是绕开 HCOMM 后保住 process-local
资源所有权的最小安全方案。

但普通 KFC worker 是否适合覆盖整个模型生命周期，仍取决于目标 CANN 对 worker
配额、watchdog、阻塞等待和取消的正式约束。在这些约束没有验证前，epoch owner
只能作为候选实现，不能声称已具备生产可用性。正式方案应优先使用受支持的专用
AICPU service/daemon；如果没有这种能力，需要评估基于目标 TGID 建队列的 Host
控制面或回到受支持的资源服务，而不是无限占用通用 worker。

资源 owner 的 lease 也不能粗暴超时退出。若没有收到带 queue-drain 证明的
`SHUTDOWN_QUIESCED`，超时强杀会再次丢失 UDMA/UMMU 所有权。候选 owner 将 lease
到期视为诊断/故障状态，并继续保留资源；只有合作式 shutdown 或硬件 reset/隔离
才能结束未知所有权。

## 5. 当前 POC 覆盖与仍缺能力

本阶段目标是验证一条顺序、多批、单通道的最小路径：一次注册/建链，多轮
`BatchGet`，每轮可配置条数和块大小。它还不是完整 KVAF Runtime，主要缺口包括：

- 当前每端只有一个连续 MR；完整 decode KV 池可能需要多 MR、超过 4 GiB 的分段
  注册和 MR table；
- 当前只有一个 Jetty/SQ/CQ owner，适合顺序 batch；并发 stream/多请求需要 queue
  pool、提交序列化或明确的多 producer 协议；
- smoke 由 Host 同步和检查每轮结果；正式 `BatchGetAsync` 需要图兼容 Event/Flag，
  不能在 replay 中执行 Host 编排；
- AccessTable 的生产、KV key 锁、页复用和 vLLM Tensor 生命周期尚未完整接线；
- owner 崩溃后的无复位接管没有安全证明。process-local UMDK state 不能由新 worker
  猜测接管；需要受支持的服务恢复或驱动级撤销能力；
- AIV 当前是单 queue owner。4096×1 KiB 的正确性通过后，还要测 doorbell 合并、
  wave 大小、CQ polling 和尾延迟，不能仅凭功能通过宣称“极致性能”。
- 当前 CQ poll budget 到期会冻结并保留 owner/MR/queue，避免错误销毁；要实现无需
  reset 的在线恢复，还需增量提交真实 tail 并增加可续跑的 pending-CQ drain kernel。

因此当前 `batch_iterations > 1` 证明的是“同一 MR/Jetty/SQ/CQ 能被多批复用”，而
不是 decode 热路径的最终性能。每轮 canary、Host memcpy、stream synchronize、D2H
回读和 provider Snapshot 都是 smoke 诊断动作，接入 Runtime 时必须移出图回放路径。

解除硬件门禁前，至少应按顺序验证：零 WQE Open/Close、一条 1 KiB、跨 queue
depth、多波 4096×1 KiB、多 batch iteration、错误 session、CQ timeout、Close
retryable、owner/进程异常和 30 分钟 soak。任一失败都应保持 backing memory 与
owner，不允许删除 quarantine 来伪造恢复。
