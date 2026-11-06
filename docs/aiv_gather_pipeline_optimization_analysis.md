# Gather 流水与 Python 调用开销分析

分析日期：2026-11-06。

本文保留实施前的基线分析；已落地的代码、V5/GDR3 合同和部署方式见
[Gather 分片流水与调用快路径](aiv_gather_shard_pipeline.md)。

对照当前工作区 `aiv_gather_yq@988e00f` 及已有未提交修改，与本机
`/Users/chelseyy/Working/main-test/KVAF` 中的 `origin/aiv-scatter@2c1f760`
（2026-11-03）。当前 gather 仓库自己的 `origin/aiv-scatter` 仍停在
`c140be9`，尚无此次要参考的 trusted-input 和分片消费实现。

用户观察为：656 B × 8192，CPU gather 约 37 us，流水后仍暴露约 30 us，
Host 到 Python E2E 另有约 40～50 us。本次未找到这组原始 CSV/CPU 日志，
以下区分源码事实、条件性估算与待验证方案；未运行目标 NPU 性能测试。

结论是：优先移植 scatter 的最小元数据读取与贯通 Runtime/Host 的 trusted-input
通路；流水优先改善首个可消费数据的产生与观察时间，再考虑取消单一连续前缀的
消费限制。现有双 bank、跨 chunk 在途 READ 和 ACK/CQ 重叠应保留。

## 1. 先统一两个开销的计时范围

当前性能脚本有两种不同合同：

| 路径 | 实际计时内容 |
| --- | --- |
| profile off | Python `batch_get(synchronize=False)`，再 `torch.npu.synchronize()`；含外部全设备同步，无逐批状态/profile D2H |
| profile on | Python `batch_get(synchronize=True)`；Host 内状态 D2H、profile D2H、stream 同步与校验；快照字典转换在计时外 |

位置：`tests/python/test_aiv_gather_batch_get_perf.py:514`、`:541`、`:560`。

`GatherBatchGet::Execute` 到获取锁、capture 查询、上次必要 Drain 之后，才记录
`profile_begin_ns`（`host/gather_batch_get.cpp:447`、`:472`）。因此同一条 profile
样本的 `call_e2e_us - host.total_us`，还包含：

- Python 包装与 pybind 重载派发、进入/返回边界；
- 七个 Tensor 的元数据提取、请求构造与 stream 解析；
- Runtime 的表句柄与布局验证；
- Host 获取锁、capture 查询与前置 Drain。

40～50 us 是这些部分的合计线索，不能直接全部命名为 Python 执行时间。
也不能用 profile-off E2E 减 profile-on Host 均值得到这一差值。

chunk 模式的 `wait_cpu` 是 notify 尾 CQ 后到 AIV 观察最终 ACK 的跨度；
`read` 从首次数据提交到数据完成，其中含后续进度等待。二者重叠，且当前最终
ACK 在所有数据 WQE 发布后查询，`ack_done_cycle` 不是 CPU 的实际结束时刻。
所以若 30 us 来自 `wait_cpu` 列，`37 - 30 = 7 us` 不能直接作为实际掩盖量。
应用同合同的流水关闭/开启 E2E 差值衡量净收益，并结合首 READ 时间解释。

## 2. Python 与 Host：scatter 已实现、gather 尚缺的优化

gather 的 `GetNpuTensorView` 每次对七个 Tensor 各执行六次 `hasattr`，再读取
连续性、device、data_ptr、numel、element_size 和 shape，并将 shape 转成
`std::vector`。合计有 42 次额外属性存在性查询、七份 shape vector 转换。
之后还在 binding、Runtime、Host Input 中重复验证布局。

对应位置：

- `bindings/src/npu_python_runtime.cc:61`、`:513`；
- `runtime/src/runtime.cc:251`；
- `host/sequential_batch_get_common.h:7`；
- `host/gather_batch_get.cpp:453`。

scatter 的可借鉴实现如下（行号以 `2c1f760` 为准）：

| scatter 实现 | gather 可采用的改动 | 范围 |
| --- | --- | --- |
| `TensorMetadataMethods`，binding:105 | 缓存方法名，用 CPython method vectorcall；缓存 current_stream callable，每次解析当前 stream/handle | 无需改变输入合同即可先移植 |
| `TrustedScatterRequest`，binding:145 | 每次只读取七个当前指针、indices/address 两个容量、output 首维；按配置推导其余字节数 | 需显式 caller-trusted 输入合同 |
| `RuntimeCore::BatchGet`，runtime:258 | trusted 标记贯通到 Runtime，跳过表查找/布局复验 | 同上；仅改 Python 层会留下重复工作 |
| scatter Host:285、:302 | trusted eager 通路略过 capture 查询，直接 FillInput | 明确仅 eager；跳过查询不产生 graph replay 支持 |
| `scatter_host_profile.h` 与 binding:586 | 补齐 binding/metadata/stream/runtime/lock/capture/Input/launch/sync 计时 | 诊断开关控制，baseline 不读取这些时钟 |

相关提交：`d852729`（Host 分段计时）、`f552248`（trusted 通路）、
`3533886`（方法调用与 stream 解析优化）。建议选择性迁移，避免整分支合并混入
CPU-push 协议、scatter 状态生命周期或其他算法修改。

保持普通 checked API，另提供显式 trusted 通路，适合已经保证格式的重复调用。
每次仍读取当前指针/容量/stream，不能缓存上一轮 Tensor 存储或 stream handle。
若以后加入 prepared context，应明确 Tensor 所有权、resize/替换后的失效规则。

gather 已删除逐次 status memset、普通异步 D2H 和不必要的同流 Drain；
这些不需要再移植。固定 output 的 MR/entity 已有缓存
（`host/batch_get_host.cpp:232`），逐次 Acquire 通常是缓存查询和引用增减，
不能先假设每轮重新注册 output。必要时单独计时，再评估实例级最近命中缓存。
scatter Host 目前每次 Execute 前 Drain，也不应覆盖 gather 已有的同流异步行为。

## 3. 默认 chunk 为什么可能只留下约 7～8 us 的重叠机会

目前 CPU 按线程均衡分配、每任务不超过 256 KiB，worker 处理
`worker, worker + active_threads, ...`。任务内部到全局 chunk 边界发布部分进度，
只有 coordinator 汇总并向 NPU 暴露一个连续 ready 前缀。

位置：`runtime/include/cpu_gather_executor.h:132`，
`runtime/src/cpu_gather_executor.cc:152`、`:177`。

以已有文档使用的配置举例：总请求 8192，2 个 AIV lane，总 CPU 线程 32，
每 lane 实际活跃 16 线程，则每 lane：

```text
4096 × 656 B = 2,686,976 B = 2.5625 MiB
16 个任务，每任务 167,936 B = 164 KiB
默认首 ready 需要最前面的线程复制到 128 KiB
128 / 164 ≈ 78.05%
```

尽管其他线程同时复制后面的数据，最前面 128 KiB 仍由一个线程完成。
首 ready 不是“整批 2.56 MiB 完成了 128 KiB 的比例”，而是首线程已完成自身
约 78% 工作时才产生。后面任一任务未完成，还会阻止连续前缀跨过去。

若仅作均匀拷贝估算，并把 37 us 全视为此轮并行 copy 时间：

| ready chunk | 首线程完成比例 | 首 ready 的拷贝时间估算 | 剩余 copy 窗口 | 首 ready 可支持的完整 READ WR，SGL=6 |
| --- | ---: | ---: | ---: | ---: |
| 128 KiB | 78.05% | 28.88 us | 8.12 us | 33 |
| 64 KiB | 39.02% | 14.44 us | 22.56 us | 16 |
| 32 KiB | 19.51% | 7.22 us | 29.78 us | 8 |

这是调度机制的条件性解释，不是硬件预测：37 us 可能包含 prepare/wakeup/join，
实际线程数、NUMA、调度、barrier 和 NIC 争用也会改变结果。
但 128 KiB 的估算与“约 30 us 暴露、仅约 7 us 掩盖”的现象接近，值得先验证。

首 ready 产生后，AIV 还需完成控制 READ/CQ、缓存维护、VF 构造、WQE MTE3 和
doorbell。该开销进一步压缩可重叠窗口。只扩大双 bank 不会提前 CPU 的首 ready。

最小实验先扫描 `KVAF_AIV_GATHER_CHUNK_BYTES=0/131072/65536/32768`。
不直接将最小粒度设为默认：小 chunk 会增加 `dsb sy`、查询、小 post 和 doorbell，
应同时看 CPU copy/total 与 E2E。也可比较更小的静态任务轮转，让多个线程更早
完成前缀内的不同任务；需保证 coordinator 及时扫描，不能又被它下一段长 copy 阻挡。

## 4. scatter 流水最值得借鉴的是独立分片消费

scatter `WaitReadyShard`（kernel:381）扫描当前 chunk 尚未消费的四路 Jetty；
哪路 ready 就先处理哪路，不要求最前一路先完成。
`ScatterPipeline`（:450）用 pending mask 保证每个分片只消费一次。
`RowCopyState`（:276）跨分片保留 bank 状态，仅复用 bank 或最终退出时等待。
其改进是当前 chunk 内独立 Jetty 分片消费，并非对全部 worker/chunk 任意乱序调度。

gather 则有两个连续性限制：

1. CPU `PublishReady` 遇到首个未完成任务就停止，即使更后面的任务已完成。
2. AIV `ReadGatherChunks` 只推进下一条逻辑 WR，等待覆盖该 WR 的连续前缀
   （kernel:594）。

建议的新方向是让 CPU 发布 per-shard 的 sequence/progress，AIV 一次批量查询多个
shard，先提交任何已完成 shard 的 READ，减少头部阻塞。保持实际源地址与
output_indices 的映射，请求提交顺序可以改变，但各输出范围仍须遵守唯一、不重叠合同。

这一改动需要：

- CPU 分片尽量对齐完整 SGL 分组和 staging cache line；例如 656 B/SGL=6，
  12 token 同时满足两种边界。尾部显式处理。当前按字节切的任务不能直接当完整 WR。
- 为每个 shard 记录本轮进度与已消费范围，处理部分完成、旧/未来序号、错误和尾部，
  防止跨 shard WR 提前读、重复读或漏读。
- READ 构造接收逻辑源区间，物理 SQ head 仍按实际提交顺序推进；completion/wave
  按实际提交量管理。保持两 bank 和跨分片在途 READ，不逐片等数据 CQ。
- 全部 READ 完成且最终 CPU ACK 成功后，才允许下一批复用 staging/descriptor。

这是比减小 chunk 更完整的方案，涉及 wire/progress 布局和两端调度，不适合只改
`ReadyPrefix` 的一个条件。可以先做少数 coarse shard，再判断是否需要逐线程进度。

## 5. 进度通知与 WQE 准备还有两处串行化

**控制 READ 往返。** gather 的 `ReadyPrefix::BeginQuery` 每次从 CPU DDR
READ 40 B 到 HBM scratch，再等控制 CQ 后读快照（kernel:478）。虽然已有
lookahead，等待新前缀时仍可能落到前台。CPU `dsb sy` 和 progress release
并不会直接把该值写入 NPU HBM。

scatter 的 Ready 是 CPU payload WRITE 链上的 8 B strong-order WRITE，直接落到
HBM；AIV 只刷新一条本地缓存行读取四路进度，省去逐次远端 READ 往返。
可为 gather 设计 CPU 主动 WRITE 到 HBM mailbox 的 ready/ACK 通知，但 gather
当前 CPU memcpy worker 不具备这条已有的 payload WRITE 链，必须增加/复用 CPU
URMA 通道、HBM mailbox 导入与注册、队列所有权和清理。不能视为免费的协议移植。
应在 `ready_queries/ready_wait` 与 E2E 数据证明值得之后再做。

**先等 ready 才构造 WQE。** 当前 kernel:596 的等待在 VF 构造（:646）之前。
WQE 所需的 staging 地址、output_indices、token/SGL 和队列位置，在 CPU payload
ready 前就已知。可在首个/后续查询在途时提前构造一至两个 bank，等对应源区间
ready 后立即 publish；将“可以准备 WQE”与“允许网卡执行 READ”分开。

预构造不等于提前敲 doorbell；也不能让已发布的 PI 跨过未 ready 的 WQE。
UB/SQ 预留、owner、event、signaled 尾 WR 和错误退出必须一起管理。
这是不修改 CPU wire 的候选，比完整 push-ready 改造更适合先实验。

## 6. 已有优化与进一步边界

当前 gather 已具备：

- 首个 eligible post 一次 VF/MTE3 后立即提交（kernel:666）；
- 查询 lookahead 和后续 VF/MTE3 工作交叠；
- 跨 ready 快照保留数据在途状态，仅在物理容量 wave 尾和整批尾等待 CQ；
- 最终 ACK 查询与最后一波数据 READ 重叠（:688）；
- SIMT 清单构造与 MTE3 双 bank。

等待新进度时 `FinishReadBanks` 仍会提交本地准备好的 post 并收回本地 bank 事件，
这是本地流水收尾，不等于排空网卡数据 CQ。进一步优化应保留已提交 READ 的在途性。

整份 descriptor 仍在 CPU 开始之前完成构造、WRITE 和 notify（kernel:845）。
如果这段已占较大比例，后续可考虑分段 descriptor 发布，重叠地址清单生成/传输与
CPU copy；当前 CPU 直接持有清单指针，需新增 descriptor-ready/所有权协议。
固定长度 token 的清单也可由每条 16 B span 评估压缩为 8 B address，但这是另一个
独立 wire 改动，不能复用旧 magic 而静默改变布局。

CPU helper 等待模式与拓扑也要核对：passive notify 轮询可 sleep 20 us
（`runtime/src/cpu_runtime.cc:418`），adaptive helper 闲置后可睡眠。
已有 `prepare/start_delay/join_wait/thread_cpu` 足以先区分唤醒、拷贝和尾部等待。
不能默认测量配置已 active，也不能靠增加线程数判断一定更快。

## 7. 建议实施顺序与验收

| 顺序 | 改动/实验 | 主要判断依据 |
| --- | --- | --- |
| 1 | 迁移 scatter 的 binding→Runtime→Host 分段计时；用同一条同步 profile 样本核算外层差值 | 40～50 us 具体集中在哪一层 |
| 2 | 方法名/vectorcall 优化，再贯通显式 trusted eager 通路 | 同合同 E2E 和 metadata/Runtime/Input 时间下降；changing pointers/stream 正确 |
| 3 | 原协议扫描 128/64/32 KiB，补首 ready/首 READ 相对 notify 的派生列 | 首 READ 提前且 E2E 净改善，CPU/barrier/查询成本没有抵消收益 |
| 4 | 查询在途时预构造 WQE，准备与发布解耦 | ready→首 doorbell 的剩余时间下降；无未 ready 源读取 |
| 5 | per-shard progress 与就绪分片优先 READ | 人为延迟首片仍能消费后片；完整 changing-data/guard 正确 |
| 6 | 有依据地实验 CPU push-ready，之后再评估 descriptor 流水/清单压缩 | 控制往返和启动开销下降超过新增 WR/协议成本 |

新增/派生诊断量建议：

```text
notify_to_first_ready = first_ready_cycle - notify_done_cycle
ready_to_first_submit = first_read_submit_cycle - first_ready_cycle
notify_to_first_submit = first_read_submit_cycle - notify_done_cycle
post_notify_wall = max(ack_done_cycle, read_done_cycle) - notify_done_cycle
```

CPU 侧另外记录相对本地 Copy 开始的首 ready 发布延迟，以及分片 ready 时间；
这些属于新增字段，现有 CPU 日志不能直接还原。AIV 首 ready 是查询观察时刻，
不能与 CPU 的绝对时间戳相减。`ready_wait_cycles` 是控制路径前台服务/等待累计量，
不包含已与 WQE 工作交叠的查询在途时间，也不能视为纯网络 RTT。

多核主区间按每条样本 `max(finish) - min(start)` 计算，同时展示 lane 分布；
不要用平均 lane total 代替整批完成跨度。比较流水净收益保持核数、CPU 总线程、
NUMA/绑核、页策略、SGL、表规模、输入/输出顺序、profile 和同步合同一致，
交替复测并比较全样本 p50/p95/p99。正式性能关闭 CPU/NPU/Host profile。

仅测相关正确性：首片延迟与后片先 ready、656 B 非对齐和分片尾项、changing
payload/指针/输出映射、stream 切换、重复批次、SQ 环回、最终 ACK/CQ 的两种先后、
错误/超时后的资源拒绝复用。保留完整 output/guard 验证。

理想流水只能将 CPU copy 与 NIC READ 的可重叠部分由相加压向取最大值；首片启动、
尾部、控制与共享 DDR 带宽仍有代价。如果独立 READ 时间短于 CPU copy，本批中也
没有足够通信工作完全覆盖 CPU。最终收益必须用完整 E2E 验证，不能预先承诺消除
全部 30 us 或 40～50 us。
