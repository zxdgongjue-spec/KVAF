# Gather：整批 ACK 下提前准备 WQE，再加入协作 chunk

当前保留整批 ACK 基线、连续前缀 chunk 对照，以及 WQE 提前准备与协作 chunk。
两步优化有独立开关；默认使用 128 KiB 连续前缀，PREPARE_AHEAD=0、
COOPERATIVE_CHUNKS=0，便于同一构建比较。没有目标双机性能结果。

## 第一步：整批 ACK + WQE 提前准备

`KVAF_AIV_GATHER_CHUNK_BYTES=0`、`KVAF_AIV_GATHER_PREPARE_AHEAD=1`：

1. 完整 descriptor、command、notify 的有序 WRITE 完成后，先发出异步 ACK READ。
2. ACK 查询在途时，用原 SIMT encoder 提前填充至多两个 UB WQE bank。
3. 观察到本轮最终成功 ACK 才将 WQE 经 MTE3 写入 SQ、敲 doorbell。
   started ACK、旧 ACK、WQE 构造完成都不能允许 payload READ。
4. 已提交 READ 与后续 WQE 准备继续重叠；仅在物理容量 wave 尾和整批尾等待数据 CQ。

预构造只发生在 UB，未就绪 WQE 不写 SQ；SQ published PI 只包含已确认 ready 的
完整 READ。预构造 SQ 位置按 WQEBB 预留，但预留不代表发布，也不增加软件在途预算。
UB 复用依赖本地 MTE3 完成，SQ 复用依赖物理 wave 的尾 CQ。

## 第二步：少量多线程协作的连续 chunk

`KVAF_AIV_GATHER_COOPERATIVE_CHUNKS=2/4/8` 启用。默认自动启用 prepare-ahead；
显式 `PREPARE_AHEAD=0` 拒绝初始化。`COOPERATIVE_CHUNKS=0` 关闭此功能。

每个非空 lane 根据本次实际有效行数计算：

```text
chunk_bytes = align_up(ceil(lane_payload_bytes / configured_chunks), 64)
实际 chunk 数 = ceil(lane_payload_bytes / chunk_bytes) <= configured_chunks
```

因此部分前缀、不同大小的 lane 都按实际 payload 分块，短请求可能只有一块。
块边界允许位于 token/WR 内，AIV 始终按完整 WR 的源结束位置判定可提交范围。

CPU 保留上述等长 chunk 的发布边界，块内复制改为动态领取小任务（tile），不再给
每个 helper 预留固定份额。全局原子游标按 staging 顺序分配任务，线程完成后继续
领取最早尚未分配的任务。线程数量仍按整批 payload 和已有线程预算选择。

多线程时 tile 按 64 B 对齐，大小为 `min(128 KiB, align64(ceil(chunk_payload /
(2 * threads))))`；较小的 chunk 保留约两轮调度机会，较大的 chunk 将单个已领取
任务限制在 128 KiB。单线程直接复制整个 chunk，末尾都精确处理有效字节。
tile 只决定 CPU 任务分配，不增加 NPU 进度粒度、WQE 或 READ 提交次数。

同一线程在一个 chunk 内完成的 tile 先本地累计，领取到后续 chunk 或任务耗尽时，
统一执行原有系统可见性 barrier，再对该块的剩余 tile 数做一次汇总原子递减。
每个参与线程/块最多一次 barrier 和完成 RMW；没有领到任务的晚启动 helper
不参与该块完成计数。最后完成某块的线程汇总已完整完成的连续块，并用单调 CAS
推进 `ready_bytes`；发布后保留系统 barrier。已领取但未完成的 tile 仍会阻止前缀
跨过该块，不能用后面已完成的块跳过空洞。
最多八个块的完成者参与发布，不引入专用通知线程。coordinator 正在复制下一块时，
helper 仍可发布前一块，不能让前缀推进依赖 coordinator 的拷贝检查点。

```text
CPU 任务领取：块0 tile0、tile1…… → 块1 tile0、tile1…… → …
CPU 完成者：         ready0               ready1
AIV：        query + 提前准备 WQE → 提交 ready 的连续 WR
NIC：                         READ0 ─── READ1 ─── …
```

AIV 使用 40 B descriptor READ 查询 sequence / ready_bytes，按连续前缀消费。
准备游标可以领先 ready 游标，只有两者
共同覆盖的完整 WR 才提交。一个预构造 bank 中只有少量 WR ready 时，也提交这部分
并保留其余内容，不能等待凑满 128 WR。每条 WQE 只构造一次；bank 和在途 READ 跨
前缀保留。已准备且 ready 的首批优先提交，后续快照不插在其提交前处理。

最后数据 post 发布后，在 READ 仍在途时查询最终 ACK，再确认数据 CQ。ready=total
不替代最终 ACK；即使所有 tile 已完成，CPU 仍须汇合本轮已调度的所有 helper 后才
返回最终 ACK，不允许提前复用 descriptor/staging。错误继续 poison 两条队列，
消费本地已发出的 VF/MTE3 事件，并沿用原有会话停机和资源保留规则。

这版只实施 CPU 动态领取任务，保留等长 2/4/8 个发布块，NPU 的 WQE 编码、双 bank、
数据队列与进度查询调度均不变。已有 V9/GDR4 部署只需重编、重启 K 侧 CPU 服务，
环境变量和参数保持原配置。首尾小、中间按 READ bank 发布的非等长布局，以及
NPU 进度快照消费时机的调整留作后续独立对照，避免同时改变多个因素。

## CPU helper 的等待策略

helper 完成本轮所有复制份额后，先判断整批复制是否完成。coordinator 或其他
helper 仍在复制时，已完成的 helper 直接进入条件变量等待，避免持续自旋与慢线程
竞争 CPU。块内的份额划分、完成计数和 ready 发布方式沿用上述实现。

整批复制完成后，最多允许 helper 1 短时自旋等待下一轮：`active` 上限 100 us，
`adaptive` 上限 25 us，`passive` 不自旋；其余 helper 继续休眠。该保温仅在 helper 1
决定等待时已观察到整批完成的情况下启用，不为保温唤醒已休眠的线程。
下一轮任务仍由 generation 发布和条件变量广播唤醒，spin 转 sleep 时在同一互斥锁下
检查 generation，避免丢失通知。线程数缩小后的非活跃 helper 也保持休眠。

coordinator 的 notify 轮询策略不变。这次等待策略调整只需重编、重启 K 侧 CPU
服务，可配合现有 V9 NPU 构建使用；没有新增开关、通信字段或 ABI 变更。
线程预算、绑核和内存 NUMA 放置本轮不作调整。

比较时固定线程数、chunk、SGL 和 NPU 配置，分别重测 `active` / `adaptive` 的
656 B、4096 B 请求。正式比较关闭 profile；单独采样 CPU `copy_ns`、`join_wait_ns`、
helper `start_delay_ns`，观察减少空转后慢线程是否更快，以及唤醒开销是否上升。
等待策略不能保证消除所有 AIV `wait_cpu`，实际收益需双机测试确认。

## 协议、ABI 与配置优先级

- kernel：`KvafGatherBatchGetAivKernelV9`，args 248 B；loader 拒绝 V1–V8。
- profile：PRF5，256 B；Python CSV schema 9。
- 第一步仍使用 GDR2 descriptor；第二步使用 gather 专用 GDR4 magic。
  GDR4 的 48 B header、16 B span、ready 前缀及最终 ACK 布局/含义与 GDR2 相同，
  magic 仅区分 CPU 任务调度。每 lane scratch 为 64 B，descriptor 后不附加进度向量。
- CPU 与 NPU 的 Host/Runtime/Python/kernel 均须同步重编部署。不支持 GDR4 的旧 CPU 会拒绝该请求，
  会返回错误 ACK，不能混合部署。NPU 运行日志应显示 V9 / args_bytes=248。
- 已删除 SHARD_PIPELINE、PACKED_POSTS、PROGRESS_MODE 及其专用代码、RPC、mailbox
  和 profile 字段。旧环境变量非空时明确报错，请先 unset；不再静默选择其他路径。
- 此时 CHUNK_BYTES 仅用于确认启用 chunk 模式，实际字节数按有效 payload 计算；
  显式 CHUNK_BYTES=0 与 cooperative 冲突并报错。
- prepare-ahead 可用于 whole-batch、普通 prefix 和 cooperative prefix。
  probe 及 token_bytes*SGL>65536 的拆 token 路径不支持新开关，显式请求会报错。
- 环境变量在 Runtime 初始化时捕获，切换组别须新建 Runtime。

## Python/Host 调用快路径

`trusted_gather_inputs=True` 仍默认启用：每轮读取当前七个 data_ptr、必要容量和当前
stream handle，复用 method/vectorcall 与 current_stream callable；不缓存跨调用的
指针、容量或 stream。调用方保证连续 NPU Tensor、int32 请求数组、int32[1] miss_count、
uint64 地址表和等长 uint8 valid、output 行字节数及生命周期合法。仅支持 eager。
`trusted_gather_inputs=False` 保留完整检查，C++ 可显式指定
`BatchGetExecutionOptions::trusted_npu_request=true` 使用相同合同。

同流异步提交、切流 fence、状态检查、MR/entity 缓存、错误后资源保留继续使用原有
生命周期；本轮没有改变 Python/Host 快路径。

## 三组硬件对照

固定 Python/Host 优化、线程/绑核、SGL、核数、输出布局及数据 seed。测试脚本在
同一目录，分别运行以下三组；环境变量不在不同进程之间遗留：

```bash
# 清除已删除的调度开关，避免继承上轮实验配置。
unset KVAF_AIV_GATHER_SHARD_PIPELINE KVAF_AIV_GATHER_PACKED_POSTS KVAF_AIV_GATHER_PROGRESS_MODE

# A：整批 ACK 基线
env KVAF_RUN_NPU_TESTS=1 KVAF_AIV_GATHER_COOPERATIVE_CHUNKS=0 \
  KVAF_AIV_GATHER_PREPARE_AHEAD=0 KVAF_AIV_GATHER_CHUNK_BYTES=0 \
  python -u tests/python/test_aiv_gather_batch_get_perf.py

# B：整批 ACK + 提前准备
env KVAF_RUN_NPU_TESTS=1 KVAF_AIV_GATHER_COOPERATIVE_CHUNKS=0 \
  KVAF_AIV_GATHER_PREPARE_AHEAD=1 KVAF_AIV_GATHER_CHUNK_BYTES=0 \
  python -u tests/python/test_aiv_gather_batch_get_perf.py

# C：提前准备 + 四个协作 chunk；随后单独比较 2 / 8
env KVAF_RUN_NPU_TESTS=1 KVAF_AIV_GATHER_COOPERATIVE_CHUNKS=4 \
  KVAF_AIV_GATHER_PREPARE_AHEAD=1 KVAF_AIV_GATHER_CHUNK_BYTES=131072 \
  python -u tests/python/test_aiv_gather_batch_get_perf.py
```

正式延迟比较使用 profile=off 的 E2E p50/p95 和 kernel 总时延。单独 profile 解释机制：

| 新字段 | 含义 |
| --- | --- |
| prepare_ahead | 本 lane 是否采用提前准备路径，空 lane 为 0 |
| cooperative_chunks | 配置的协作块数，空 lane 为 0；实际粒度见 pipeline_chunk_bytes |
| prepare_ahead_wrs | 构造时尚未被 AIV 确认 ready 的 WR 数，不是实际隐藏时长 |
| prepare_ahead_cycles | 上述构造区间的 Scalar 服务周期，含 VF 发起和必要的本地依赖等待；不等于 VF 硬件执行时间 |

`read_build_cycles` 现在可能包含整批 ACK 之前的准备工作，不能全部归入 ACK→READ
完成区间。`wait_cpu` 仍是 notify→最终 ACK 的 AIV 观察区间；协作 chunk 下与数据
READ 重叠，整批提前准备下与 WQE 工作重叠。不能将这些字段相加推算 kernel 时间，
也不能用 prepare_ahead_wrs 证明 CPU 当时尚未完成或实际隐藏了多少微秒。

## 本地验证范围

模型覆盖：ACK/首次 progress CQ 延迟到 VF 后才返回；ACK 前仅构造 WQE；完整 WR
就绪判断；128-WR bank 的部分提交；小队列容量、SQ 环回、尾 WQE、coalescing；
2/4/8 块、不同 token、多核及空 lane；重复调用；进度回退/越界/错误和 CQ/超时退出。
CPU 并发消费者逐字节检查 ready 前缀；暂停首块/后续块的已领取 tile，验证其他
线程可完成后续块且 ready 不跨空洞；暂停所有 helper 的首次领取，验证 coordinator
独自完成并发布全部前缀时仍须等待 helper 汇合。另覆盖变长 span、短尾、2/4/8 块、
多轮缩小/恢复线程数及停机。模型和 sanitizer 不证明设备 DMA 可见性、
CANN 编译结果或实际性能收益，仍需目标双机验收。
