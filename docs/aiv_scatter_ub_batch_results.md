# AIV UB 批量 scatter：4 MiB 性能分析与下一步方案

> 历史实验记录：当前已固定SIMT融合清单构建和UB双缓冲，仅保留流水算法开关。
> 下文旧对照开关不再支持；运行方式和计时口径见 [当前性能测试说明](aiv_scatter_performance.md)。

分析日期：2026-10-31。当前代码：`aiv-scatter@4d1c732`。
数据来源：本轮附件 `70be83e2-6c81-40a5-89a9-4ce50832960f/已粘贴的文本.txt`。
本文仅分析数据和安排后续实现，不引入新的 kernel 修改或独立搬运性能测试。

## 1. 样本口径与核对

附件包含 100 条 `profiled` 样本、58 列，无表头；按当前性能脚本的列顺序还原。
iteration 为 0～99，sequence 为 1052～1151，均连续无重复。
所有样本的有效字节、ns→μs、带宽和时间线关系均通过算术核对：

```text
payload = miss_count * token_bytes
cpu_comm = cpu_post + cpu_cq_wait
aiv_wall = input_wall + build_notify + notify_to_done + scatter_tail
end_to_end = aiv_wall + call_outside_aiv
GB/s = payload_bytes / elapsed_ns
```

这些检查确认字段与计算一致，不替代时钟校准或完整结果校验。
附件没有 `baseline` 样本及 kernel SHA256，不能据此直接完成算法版本的正式 A/B 验收。

| 参数 | 数值 |
| --- | --- |
| 每个样本 | 一次同步 BatchGet，启用 CPU/AIV profile |
| AIV 核数 | 4 |
| 源 block 数 / tokens_per_block | 1024 / 4 |
| 请求容量 | 8192 |
| 实际有效 token | 1024 |
| token 长度 | 4096 B |
| 有效 payload | 4,194,304 B = 4 MiB |
| 输入模式 / 输出顺序 / 输出行步长 | changing / random / 1 |

本次是 full-blocks 工作负载中 1024/4096 的有效前缀，不是取满 4096 个 token，
也不等价于每个源 block 各取一个 token。CSV 的 random 是输出顺序；
源模式应从 JSON 配置核实，不能由这一列推出。

## 2. 全样本统计

带宽使用总有效字节/总耗时；百分位采用排序后的线性插值。

| 指标 | 平均 μs | P50 μs | P95 μs | 有效 GB/s |
| --- | ---: | ---: | ---: | ---: |
| 同步 BatchGet 端到端 | 530.297 | 531.635 | 551.463 | 7.909 |
| AIV 已打点主体跨度 | 384.899 | 385.529 | 397.262 | 10.897 |
| 输入准备 wall | 5.628 | 5.615 | 5.713 | — |
| 构建与 notify | 127.395 | 127.754 | 130.509 | — |
| notify→done | 168.723 | 169.506 | 180.778 | — |
| max(各核 scatter 时长) | 83.094 | 82.864 | 84.336 | 50.477 |
| scatter tail | 83.153 | 82.963 | 84.440 | — |
| 调用减去 AIV 主体的剩余量 | 145.398 | 146.672 | 158.732 | — |
| CPU WriteBatch | 153.066 | 153.965 | 165.690 | 27.402 |
| CPU 通信窗口 | 135.284 | 136.300 | 146.127 | 31.004 |
| CPU post | 6.503 | 6.395 | 7.612 | — |
| CPU CQ wait | 128.781 | 129.900 | 139.284 | — |
| snapshot，计时之外 | 105.519 | 105.160 | 114.463 | — |

按调用时间轴，均值可拆为：

```text
530.297 μs
  = 145.398  调用减去 AIV 已打点主体的剩余量      27.4%
  +   5.628  输入准备                            1.1%
  + 127.395  构建与 notify                      24.0%
  + 168.723  notify→done                        31.8%
  +  83.153  scatter tail                       15.7%
```

CPU header 2.612 μs、descriptor 3.648 μs、WriteBatch 153.066 μs 是 notify→done 内的
嵌套工作，不能再次加到上述总和。WriteBatch 减去 comm 为 17.782 μs，是软件准备等
工作的合计剩余量；notify→done 减去三项 CPU 耗时约 9.398 μs，也不能直接命名为单个函数耗时。

CPU 通信仍是包含提交、payload/ready WRITE 和尾 CQ 处理的有效完成窗口，
不是纯 DMA 硬件带宽。CQ wait 占 comm 的约 95.2%，说明主要耗时在 post 之后；
现有字段不能继续区分硬件传输、队列等待、完成上报和 CPU 调度。

`call_outside_aiv` 含 binding/Runtime/Host、设备未覆盖的开头收尾及状态同步等，
不能直接称为纯 Host 耗时。snapshot 在样本停止计时之后，不应从端到端中扣除或再次加上。

## 3. 收益和当前限制

### 3.1 本地 scatter 已明显缩短，但要区分历史对照与严格 A/B

仓库旧 [计时审计](aiv_scatter_timing_audit.md) 的 4 MiB 结果为：

| 均值 | 历史 4 MiB | 本轮 4 MiB |
| --- | ---: | ---: |
| 端到端 μs | 967.332 | 530.297 |
| 构建与 notify μs | 359.726 | 127.395 |
| scatter μs | 277.739 | 83.094 |
| CPU comm μs | 119.173 | 135.284 |

表面上端到端缩短 45.2%，scatter 缩短 70.1%、约为原来的 1/3.34。
但旧组 tokens_per_block=1，本轮为 4；即使有效字节和核数相同，源地址与 offset 分布也可能变化。
另外，从 `7a499b4` 到当前版本的修改没有直接改动 ValidateRows/Source 构建段，
该段却节省约 232.3 μs；scatter tail 节省约 194.5 μs。
所以不能把约 437.0 μs 的总改善全部归因于 UB copy 改动，必须使用相同参数和二进制指纹复测。

当前实现每核 256 行，4096 B 每 tile 8 行，共 32 个 tile/核。
整次调用输入搬运和写出完成等待各 128 次，输出写出命令仍有 1024 次。
这个机制与本地 scatter 改善的方向一致，但精确因果收益以匹配条件的 A/B 为准。

### 3.2 scatter 稳定，当前调用波动更多来自其他时段

scatter 范围为 81.889～85.226 μs，变异系数约 0.86%；
端到端范围为 479.100～574.020 μs，变异系数约 2.84%。
最慢样本 iteration=48 的 scatter 为 83.748 μs，仍处于常见范围；
其 AIV 主体外剩余量为 174.548 μs，notify/CPU 完成时段也更长。
因此继续削减 scatter 不能解决所有尾延迟。

### 3.3 后续收益按端到端预算排序

保持其余阶段不变的估算如下，均是预算分析，不是性能预测：

| 假设 | 估算端到端 μs | 比当前缩短 |
| --- | ---: | ---: |
| scatter tail 减半 | 488.721 | 7.84% |
| scatter tail 完全消除 | 447.144 | 15.68% |
| 构建与 notify 减半 | 466.600 | 12.01% |

因此应把构建和调用剩余量纳入主线；双缓冲仍值得做一个受控的端到端对照，
但不能承诺它会把整个调用再加速一倍。

## 4. 下一步实现顺序

### P0：固定当前 UB 版本，补齐相同条件的关闭打点结果

保持本轮参数、seed、源模式、CPU 服务、频率/时钟换算和请求容量，
保存当前二进制 SHA256。沿用完整 BatchGet、changing-input、后继 NPU 消费和 output/guard 校验。
主指标用 CSV 中 `phase=baseline` 的端到端 p50/p95 和带宽；profiled 只负责阶段归因。
每个候选与当前 UB 单 bank 分进程交替比较，不新增独立 staging→output 性能测试。

### P1：先减少清单构建的重复读取，再评估多核构建

现有 core 0 先 ValidateRows 全量校验，再 Source 全量读取并写清单。
先将两遍合为一遍，复用已经读取的 index、offset、row、valid 和 base，
逐项保留原校验。整批校验和清单刷新全部成功后才发布 notify；
中途失败时允许存在未发布的部分清单，不能让 CPU 访问它。

若剩余构建时间仍显著，再让各核并行校验、构建独立清单区间：

- descriptor 按完整 64 B cache line 分区，即每组 8 个地址，避免不同核清理写回同一条脏缓存行。
- 每核使用独立 cache line 的 sequence/result/ready 协调标志。
- 各核清单可见且所有校验成功后，core 0 才统一发布 header/notify。
- 保留错误传播、空分区、超时和重复调用序号校验，不将核间同步成本排除在端到端之外。

### P2：继续 UB 优化，逐项比较

1. 对 4096 B 这类长度满足 32 B 对齐的 token，比较一次连续 `DataCopy`
   与当前多块 `DataCopyPad` 搬入；此时 UB pitch 等于 token 长度，输入本身紧凑。
   656 B 仍沿用 padding 路径，不能使用该条件直接替换。
2. 双 bank 初始保持每 bank 32 KiB、每批 8 行，让下一 tile 的输入/准备与上一 tile 写出重叠。
   每 bank 独立事件，只在复用前等待对应写出完成，末尾完整 drain。
3. 加一组相同总 UB 预算的单 bank 64 KiB 对照，区分“大 tile 减少同步”与“真正重叠”的收益。
   所有配置均先核对目标机静态 UB 和编译资源预算。

如果端到端收益小于运行噪声，就保留更简单的单 bank 版本；
不因局部 DMA 时间下降而接受端到端变慢的候选。

### P3：只补少量聚合计时，定位 145.4 μs 的剩余量

按需记录 binding/Host 输入准备、memset/launch 提交、status D2H 提交、
同步等待和最终状态检查的聚合时段；关闭采集时不增加读时钟或报告传输。
Host 时段在自己的时间轴闭合；同步等待与设备执行重叠，不能将二者相加。
设备 profile/status 刷新和最后 PipeBarrier 也需要纳入定位，
不能直接以这 145.4 μs 为目标删除状态检查或错误处理。

### P4：更细 ready 流水及 SIMT，放到后续候选

本轮每 worker 只有 256 行，当前 kChunkRows=1024，因此各 worker 只有一个不足整块的 chunk。
仅使用现有 ready 仍是粗粒度；尝试 64/128 行才有每 worker 多 chunk 的机会。
CPU/NPU 必须一致修改 chunk 协议、marker 存储和计划，并测量额外强序 ready WRITE 的成本。
结束前仍需最终 done 和全部本地写回完成，流水后的重叠指标不能沿用旧线性拆解相加。

SIMT 可以作为减少随机输出命令的后续对照；当前 local scatter 只占 15.7%，
其优先级低于构建和调用剩余量的定位。所有候选最终均由完整端到端收益决定。

## 5. 本轮建议

下一项代码优化优先合并 ValidateRows/Source，另做对齐连续输入和 UB 双 bank 的小步 A/B；
同时用少量聚合打点定位调用剩余量。不要将较大的改动一次合入同一个候选，
否则构建、搬运、Host 和 CPU 时段的变化无法归因。

后续实施状态：单遍校验/清单构建及双 bank 已接入，可用两个独立运行时开关分别比较，
见 [四种版本的端到端运行步骤](aiv_scatter_performance.md)。本文的 100 条数据仍是
`4d1c732` 单 bank 版本的历史样本，83 μs 不代表新双 bank 的实测性能。
