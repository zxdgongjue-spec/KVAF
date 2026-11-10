# Gather 实现与配置

本文是 `gather_read` 当前实现的入口；构建、性能采集和排障见
[Gather 性能与排障](aiv_gather_serial_profile.md)。

## 数据路径与各阶段优化

```text
Python/Host 异步下发
  → AIV SIMT 构造源清单 → MTE3 写 HBM → 控制 WRITE 发到 CPU
  → CPU 源 DDR memcpy 到连续 staging → 发布 byte-prefix / 最终 ACK
  → AIV 构造 READ WQE → MTE3 写 SQ → doorbell
  → NIC 将 staging payload 直接 READ 到 output HBM → 数据 CQ
```

HBM workspace 只存清单和每 lane 64 B scratch；UB 存清单/WQE，不存 payload。
每核独占一条控制 Jetty、一条数据 Jetty 及持久化 SQ/CQ 状态，最多 8 核、65536 请求。

| 阶段 | 保留的优化及约束 |
| --- | --- |
| Python | 默认 `trusted_gather_inputs=True`；复用 method/vectorcall 和 current_stream callable。每次重读七个 data_ptr、容量和当前 stream，不缓存跨调用的 Tensor 指针或 stream。 |
| Runtime/Host | trusted 请求快速分发；同 stream 普通异步调用连续排队，不逐次 Drain 或 D2H status；切流、显式 Drain、BatchPut 前和 shutdown 仍 fence。output MR/entity 缓存保留到 transport shutdown。 |
| 初始化与输入 | workspace/queues 使用 `HUGE_FIRST` 并允许普通页回退；不逐次 Host 清零 status/profile。Scalar 只刷新请求数，SIMT 输入不做全量 Scalar Flush；大 token 回退只刷新有效 lane 的 output_indices。 |
| descriptor | 128 个 SIMT 线程生成清单，两个 24 KiB UB bank 交替；构造下一块与 MTE3 搬运上一块重叠。发送前等待 MTE3，无逐行 Scalar 写出/Flush。 |
| 控制发布 | descriptor 分段、command、notify 合为一个有序 WRITE wave，标准队列只用一次 doorbell/尾 CQ；容量不足时仍 drain。已知尾 WR 在编码时设置 order/completion，避免二次修改和 Flush。 |
| CPU | 线程复用、分 lane 预算和绑核；相邻源 span 合并 memcpy；固定 token 用除法定位，变长 span 保留前缀索引。旧 byte-prefix 与协作 chunk 调度均保留，详见下文。 |
| READ | SIMT 直接在 UB 编码变长多 SGE WQE，两 bank 复用，按 `MTE(n−1) → VF(n) → commit(n−2)` 重叠搬运、构造和提交；首个 ready post 尽早发布。 |
| 进度与提前准备 | ready 查询与 WQE 工作重叠；可提前准备两个 UB bank，按完整 WR 就绪范围部分提交；跨 chunk 保留 bank 和在途 READ。最终 ACK 查询与最后数据 READ 重叠。 |
| 诊断 | 保留 CPU/Host/AIV profile、CSV 和 ack/wqe/sq/first_read 探针；profile 关闭时不读诊断计时器。 |

trusted 调用方保证 Tensor 连续、设备/类型/容量正确及存活到 stream 完成；仅支持 eager。
`trusted_gather_inputs=False` 保留完整检查；C++ 通过
`BatchGetExecutionOptions::trusted_npu_request` 选择同一合同。

## 输出连续与 SGE 合并

性能脚本用 `KVAF_GATHER_OUTPUT_LAYOUT` 选择目标索引：

| 值 | 布局 |
| --- | --- |
| `random`（默认） | token 目标随机排列。 |
| `segmented` | 每条逻辑 READ 内连续，完整段之间随机排列，lane 的不足 SGL 尾段保留原位。 |
| `contiguous` | 全批输出索引递增，跨 WR 也连续。 |

三种布局保持同样的随机 CPU 源请求、有效字节及保护行；只改变目标索引。
`segmented` 要求 `token_bytes*SGL <= 65536`。

`KVAF_AIV_GATHER_COALESCE_SGES=1` 将每条逻辑 READ 的连续目标合为一个本地 SGE。
仍按配置 SGL 分组，READ 数、远端范围和 payload 不变，尾 WR 使用实际 token 数。
调用方须保证组内 output_indices 依次加 1；kernel 不额外扫描这个前置条件。
性能脚本只允许在 segmented/contiguous 布局下开启，并在计时前校验 payload/guard
及 `data_wrs == data_wqebbs`，识别旧 Host/kernel 忽略开关的情况。
离散保护行的功能/探针脚本拒绝合并开关；合并路径由性能脚本预检覆盖。

例如 8192 请求、2 核、4096 B、SGL=6，每核 683 个 READ：未合并 2048 WQEBB，
合并后 683 WQEBB；整批 ACK 基线每核均为 6 次数据 doorbell。

## 配置与模式

以下环境变量在 Runtime 初始化时捕获，切换配置必须新建 Runtime。

| 环境变量（前缀 `KVAF_AIV_GATHER_`） | 默认值及范围 |
| --- | --- |
| `CHUNK_BYTES` | 131072；0 关闭 CPU chunk 消费，其他值须为 uint32 范围内的 64 B 倍数。 |
| `PREPARE_AHEAD` | 普通模式默认 0，协作模式默认 1；显式值只接受 0/1。 |
| `COOPERATIVE_CHUNKS` | 0 关闭；均分模式接受 2/4/8/16/32；小首块模式表示后续块数 N，接受 1..32。 |
| `FIRST_CHUNK_DIVISOR` | 0 关闭小首块；D≥2 且不超过 UINT32_MAX，首块取每 lane 实际 payload 的 1/D。 |
| `COALESCE_SGES` | 0；1 开启上述合并，要求 token_bytes*SGL≤65536。 |
| `CHECK_STATUS` | 1 恢复异步逐批状态回收和复用前检查；正式性能脚本拒绝该设置。 |
| `PROBE` | 空值正常执行；ack/wqe/sq/first_read 仅用于同步诊断。 |

默认是旧 GDR2、128 KiB byte-prefix、PREPARE_AHEAD=0、COOPERATIVE_CHUNKS=0。
`CHUNK_BYTES=0` 仍保留 AIV 内部 READ 流水，使用整批 ACK 就绪条件。
prepare-ahead 可独立用于整批 ACK、普通 prefix 或协作 prefix。
协作模式要求非零 CHUNK_BYTES 和 PREPARE_AHEAD=1，此时实际块大小覆盖 CHUNK_BYTES。

probe 和 `token_bytes*SGL>65536` 自动回到整批 ACK；后者走 Scalar 拆 token 编码，
每条 READ 最多 64 KiB。两者不支持显式 prepare-ahead/cooperative；除 ack 外的
WQE/SQ/first_read 探针也要求快速路径。所有 SGL 共用通用 encoder，软件实验范围
1..13，默认 6；设备的 READ 实际 SGE 限制须按目标 SDK/硬件验证。

已移除的 SHARD_PIPELINE、PACKED_POSTS、PROGRESS_MODE 非空时明确报错，须 unset。
`batch_get_max_inflight_total` 只控制 direct_read；gather 没有每 128 WR 等 CQ 的软件窗口。

## CPU 分块与连续前缀

### 旧 byte-prefix

CPU 按整批 payload 和线程预算分配均衡静态任务，每轮每任务不超过 256 KiB，
与 ready 发布粒度解耦。例如 4096×656 B、16 线程得到 16 个 164 KiB 任务。
每个 writer 跨全局 chunk 边界或到任务末尾时，先完成 payload stores 的系统 barrier，
再更新自己独占 cache line 的部分进度；coordinator 汇总连续任务前缀并按 chunk
向下取整发布 ready_bytes，全部完成时发布实际尾长。等待 helper 时也继续推进前缀。
不能越过数据空洞，也不必等待整个 copy task 完成才发布首前缀。

### 协作块与小首块

每个非空 lane 根据本次实际有效字节 T 计算边界，64 B 对齐，最后精确截断：

```text
D=0：chunk = align64(ceil(T/N))
D≥2：first = min(T, align64(ceil(T/D)))
     tail  = align64(ceil((T-first)/N))
     边界  = first, first+tail, …, T
```

短请求可少于配置块数；首块覆盖全部 payload 时保留 64 B 非零 chunk 包络，不产生尾块。
`pipeline_chunk_bytes` 在混合模式记录尾块大小，配置的 N 表示尾块数，总块数最多 N+1=33。
CPU/NPU 使用 `common/include/gather_prefix_layout.h` 的同一组函数计算和校验边界。

线程数仍按整批 payload 和 lane 预算选择，不因 chunk 小而减少线程。
每个线程依次复制各块中按 64 B cache line 均衡划分的静态份额，份额可为空。
非空份额完成系统 barrier 后，各线程对该块独立完成计数原子递减；最后完成者标记
该块完成，并以单调 CAS 发布连续完整块前缀。线程直接进入下一块，无块间等待。
helper 可在 coordinator 复制后续份额时发布前块，不能依赖 coordinator 检查点。
最终 ACK 前仍汇合全部 helper。active/adaptive/passive 等待策略及线程绑核保留。

8192 行、2 核、D=16 的示例：

| N | 656 B：首块 / 后续每块 | 4096 B：首块 / 后续每块 |
| --- | --- | --- |
| 4 | 164 KiB / 615 KiB | 1 MiB / 3.75 MiB |
| 6 | 164 KiB / 410 KiB | 1 MiB / 2.5 MiB |
| 8 | 164 KiB / 307.5 KiB | 1 MiB / 1.875 MiB |

## READ 调度与完成条件

三条调度路径保留独立主循环，共用 READ 编码、尾 WR/WQEBB 计算和 SQ 搬运：

| 路径 | 调度 |
| --- | --- |
| 整批 ACK | 最终 ACK 后启动双 bank READ 流水，保留全部 probe 检查点。 |
| 旧 prefix | 整条 WR 的源结束位置不超过 ready_bytes 才编码/提交；首 post 在一次 VF/MTE3 后发布。后续进度查询可与已 ready WQE 工作重叠，不按快照重置流水。 |
| prepare-ahead | notify 完成后先发异步 ACK/ready 查询，在 UB 提前编码。准备游标可领先 ready；已有完整 WR 就绪时先提交，允许部分 bank 提交，剩余 WQE 保留且只构造一次。 |

块边界可以在 token/WR 内；第 i 条 WR 所需前缀为
`min((i+1)*SGL, lane_rows)*token_bytes`。准备完成不等于数据就绪，未就绪 WQE 不写 SQ。
SQ 预留位置不推进已发布 PI，也不增加软件在途预算。

UB 复用只等待相应 VF/MTE3 事件；SQ 容量复用等待物理 wave 尾 CQ。
wave 同时受实际 WQEBB 容量和逻辑 WR 上限约束，128 只是 post 大小。
跨 chunk 保留在途 READ，只在物理 wave 尾及整批尾等数据 CQ；completion/order
在发布前确定，禁止回写已发布 WQE。最后数据 post 发布后，在 READ 在途时确认最终 ACK。

成功及复用必须同时满足最终 ACK 和数据 CQ：started ACK（~sequence）、ready=total、
WQE 构造完成均不能替代最终完成。错误/超时停止补发，回收已发出的本地事件，poison
控制/数据队列；Shutdown 先停止 CPU lanes 并 quiesce transport，确认安全后才释放资源。
失败后须 shutdown/reinitialize；同步或 quiesce 失败时继续保留资源。

## ABI 与兼容性

- kernel `KvafGatherBatchGetAivKernelV9`，args 248 B；loader 拒绝 V1–V8。
- profile PRF5、256 B；CSV schema 9。profile 直接写每核独占的 GM 记录。
- GDR2：整批/普通 byte-prefix；GDR4：均分协作块；均为 48 B header + 16 B span 数组。
- GDR5：CPU 继续接受固定 1/32 首块 + 4 尾块的旧格式。
- GDR6：48 B header 后附加 16 B（D、N、零保留位），再接 span 数组。
- 所有格式的 sequence/ready 偏移不变，ready_bytes 位于 +40；控制查询 READ `[8,48)`
  到 scratch `[16,56)`，一次缓存行维护后校验 sequence、错误位、范围、单调性和边界。
- CPU 验证描述符/数据范围不重叠、配置、保留位及最多 33 个完成槽；变长 span 格式保留。
- K 侧 CPU 服务与 N 侧 Host/Runtime/Python/kernel 必须同步重编部署。
  旧 CPU 会拒绝不认识的 GDR magic/块数；仅改 D/N 无需重编，但必须新建 Runtime。

## 验证边界

本地测试覆盖生产 kernel 模型、Host/loader、Python 快路径、CPU 并发发布、profile/CSV：
空批/空 lane、重复调用、切 stream、所有 SGL/尾项、64 KiB 快速/Scalar 边界、SQ/CQ 环回、
大小队列、部分 bank 提交、进度 CQ/VF 延迟、ACK 与数据 CQ 重叠、协作块及 D/N 组合、
前缀空洞/回退/越界/旧序号、CPU/CQ 错误、超时和失败后资源保留。
关闭 profile 的模型同时检查无诊断时钟读取。

模型不能证明 CANN 指令生成、设备缓存/DMA 可见性、opcode 专用 SGE 支持或实际性能收益。
性能验收仍需目标 K/N 双机，方法见 [性能与排障](aiv_gather_serial_profile.md)。
