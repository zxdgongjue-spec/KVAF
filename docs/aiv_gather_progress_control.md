# Gather V7：progress-only READ 与 CPU 进度 WRITE

本轮保留 V6 packed READ 的分片选择、WQE plan、双 bank、在途数据请求和尾 CQ。
优化范围是进度发现通路。没有引入 scatter 的 payload WRITE、done 状态或 staging
生命周期。GDR3 descriptor、generation-tagged completed-WR word 和 DDR 最终 ACK
的含义保持不变。

## 三种控制路径

`KVAF_AIV_GATHER_PROGRESS_MODE` 在 Runtime 初始化时读取，支持：

| 值 | 实际模式 | 每轮进度发现 |
| --- | --- | --- |
| `legacy` | 0 | V6 的 ACK READ + strong-tail progress READ |
| `pull`（默认） | 1 | 一条 progress READ，ACK 独立检查 |
| `push` | 2 | CPU 将进度快照 WRITE 到 NPU mailbox，NPU 本地观察 |

仅 `SHARD_PIPELINE=1`、`PACKED_POSTS=1` 且 chunk 非零时采用所选模式。
前缀、旧 shard 调度器、整批模式及 probe/大 token 回退的实际 progress_mode 为 0，
push setup 不执行。日志中的 `GATHER_OPTIONS progress=...` 和 profile 的实际模式
比 configured_progress_mode 更能说明执行路径。

`PROGRESS_MODE=legacy` 仍使用 packed READ，只对照控制路径；
`PACKED_POSTS=0` 则切回旧的连续 run 调度器，不能混为同一个实验变量。

### pull

普通查询只 READ 512 B progress vector，使用一条 WR 和一个完成 CQ。
避免在它之前附加 ACK READ，再用 strong tail 等待前一条 READ。
连续八次查询没有新进度时，独立检查一次 ACK/CPU error；有新进度后重置该计数。
因此 CPU 验证失败不会只表现为无限等候空进度。最终 ACK 必须单独确认，
其等待仍与最后一波 payload READ 的 NIC 执行重叠。

### push

Host 将现有 workspace 的 scratch mailbox 导出给 CPU，复用每 lane control
通道的反向连接，不新建 scatter session，不额外传输 payload。新增可选控制命令
`EnableGatherProgress`；原有控制协议 record 不变。旧 CPU 服务仍可运行 legacy/pull，
push 需要新的 CPU 服务，setup 失败会明确报错，不静默退回其他模式。

每 lane 的 CPU coordinator 是唯一发送者：自己的首前缀/后续发布后，以及 join
等待期间，读取 helper 已发布的 DDR progress，形成快照并异步提交 WRITE。
helper 不共享提交队列，也不等待网络 CQ。此版没有增加专用通知线程；如果
coordinator 的首块复制本身延迟，首个推送仍会受其影响，可从 CPU profile 检查。

最多两份注册的源快照在途，直到匹配 CQ 返回才允许覆盖。两个槽都在途时继续
CPU copy，后续把最新累计进度合并发送，不要求每个中间更新都送达。每份快照
是一条 strong-order WRITE，保证同一 mailbox 的快照不倒序覆盖。
关闭时输出 `GATHER_PROGRESS_STATS` 的 frames 和 busy_checks；后者统计源槽忙的
检查次数，不代表被丢弃的分片数量。

NPU 不清空每轮 mailbox，通过每个 64-bit word 的 generation 排除旧请求。
每次本地观察仍刷新缓存，并逐 word 检查进度不回退、不越界；不要求 512 B
整体原子更新。payload 仅在对应进度确认后 READ。长时间无本地进度时，每 256
次本地观察可发起一次异步 ACK/error READ，不阻塞已经 ready 的数据提交。

CPU 所有 copy helper 汇合后，补发最新快照并退休在途 WRITE，再发布原 DDR
最终 ACK；成功 ACK 不会被后续改写为错误。最后的 WRITE CQ 退休可能增加 ACK
尾部时间，需要确认能被在途 payload READ 掩盖。CPU profile 的 ack_ns 会包含
这段退休等待。

正常关闭和失败关闭都先同步/结束 NPU kernel，STOP CPU lane 并删除 CPU 发送
Jetty，确认 DMA 停止后才释放注册的快照和 NPU mailbox。关闭失败保留资源，
允许 Shutdown 重试。CPU 接受下一 generation 之前也已退休前一批源快照。

## 版本和采集

kernel 符号 `KvafGatherBatchGetAivKernelV7`；args 248 B，profile 320 B / PRF3。
Host、Runtime、Python binding 和 AIV kernel 配套重编，loader 拒绝 V1～V6。
push 还需重编并重启 CPU 服务。

CSV schema 7 新增 configured_progress_mode，以及每核实际计数：

| 字段 | 含义 |
| --- | --- |
| progress_mode | 实际 0/1/2 模式；空 lane 为 0 |
| ready_control_wrs | 进度发现及其错误检查使用的远端 READ WR 数，不含最终 ACK |
| ready_ack_checks | 无进展时独立发起的 ACK/error 检查次数 |
| ready_local_polls | push 的本地进度观察次数 |
| ready_local_updates | push 本地观察中实际增加 ready 库存的次数 |

ready_queries 仍只统计远端 progress 查询，所以 push 正常为 0；
ack_queries 包含实际 ACK READ，包括收尾。push 的 ready_wait_us 包含本地观察、
前台等待与控制完成检查，不能直接解释为网络 RTT。

wait_cpu 与 read_us 存在重叠；read_us 也包含后续进度等待。首 READ 启动更早后，
read_us 可能增长而 kernel/E2E 下降。评估以 profile 关闭的 E2E p50/p95/p99 为主，
诊断轮同时比较首读等待：

`total_us - input_us - descriptor_us - publish_us - read_us - finalize_us`

该式适用于此实现成功流水请求，最终 ACK 在 read_done 之前确认。

## 双机 A/B

使用同一份 V7 NPU 二进制和同一 CPU 服务，固定数据形状、线程/核数和 chunk；
每轮重新创建 Runtime。以下命令沿用现有 CPU 地址、设备和 kernel 环境：

```bash
export KVAF_AIV_GATHER_SHARD_PIPELINE=1
export KVAF_AIV_GATHER_PACKED_POSTS=1
export KVAF_AIV_GATHER_CHUNK_BYTES=65536
KVAF_AIV_GATHER_PROGRESS_MODE=legacy python -u tests/python/test_aiv_gather_batch_get_perf.py
KVAF_AIV_GATHER_PROGRESS_MODE=pull   python -u tests/python/test_aiv_gather_batch_get_perf.py
KVAF_AIV_GATHER_PROGRESS_MODE=push   python -u tests/python/test_aiv_gather_batch_get_perf.py
```

先用相同模式运行功能脚本 `test_aiv_gather_batch_get.py` 检查完整 output/guard。
建议分别测 656×8192、4096×8192，交替采集三种模式，保留 raw sample 和 CPU profile。
656 B 重点观察首次提交是否仍要等到 100% ready；4096 B 同时观察 ready_idle_waits、
READ 的供给连续性和最终 ACK 尾部等待。

本地模型验证了单 WR 查询计数、push 本地观察、延迟 shard 0、非连续 ready、
重复/空请求、generation 过滤、进度回退/越界、CPU error 和错误后的队列拒绝复用。
CPU 测试覆盖在途快照不可变、双槽合并和实际 copy-before-ready；可选 URMA 类型
测试执行生产发送器，覆盖延迟 CQ、partial post、错误 CQ、初始化失败和资源保留。
当前环境没有目标 CANN/NPU 或双机链路，不以模型时间宣称真机性能收益。
