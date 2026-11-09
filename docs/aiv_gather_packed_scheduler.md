# Gather ready 库存与离散 WR 合批调度

2026-11-09：默认使用 packed 调度器。Python 调用快路径保持不变；CPU wire 继续
使用 GDR3，sequence、最终 ACK、staging 复用和队列 poison 规则保持不变。

## 调度变化

- 每 lane 初始化一次 shard 边界和长度，维护 ready/consumed 数量、ready bitmap
  和未计划的 ready WR 库存。选片使用 bitmap，不再逐 post 全量重算边界。
- 初次没有 ready 时只查询进度，不构造猜测 WQE。查询已有完成结果时先读取结果，
  再按确定 ready 的 WR 生成计划。当前批只执行一次 VF；新进度交给下一批，
  不因候选或数量变化丢弃并重建当前批。
- 一个 post 可以打包不同 shard、逻辑上不连续的 ready WR，最多 128 条。
  每条 READ 的源区间与 SGL 分组不变，物理 SQ 连续排放。计划显式记录每条 WQE
  的实际 WQEBB 偏移，因此逻辑末条的短 SGE WQE 可以位于 post 中间，尾 CQ
  标记仍落在本物理 wave 的最后一条已提交 WQE。
- ready 库存低于或等于 128 WR 时才发起新的 lookahead 查询；有足够库存就继续
  构造和提交。控制 CQ 未返回时不阻塞已知 ready 的数据。没有可消费库存时先
  Commit 上一批 SQ，再进入前台进度等待。
- 两个 WQE bank 和 NIC 在途 READ 跨片保留；首批立即发布，后续 MTE3 与下一次
  VF 重叠。数据 CQ 仍只在物理容量 wave 尾和整批尾等待。

每个 bank 在 UB 增加一份 1024 B 的 WR ID/WQEBB 偏移计划，使用既有 raw UB
地址范围与显式 S→V 同步；不使用 TPipe，不增加 HBM plan、额外 payload 拷贝
或逐块数据 CQ。两 bank 共使用 66 KiB，仍在原有 160 KiB launch 预算内。
初次控制 READ 的往返仍在首批路径上；本轮通过消除无效构造和稳态 lookahead
降低其影响，未引入 CPU→HBM 的 ready WRITE 通道。

## CPU 发布变化

CPU 改为按线程连续分配 shard，减少不同线程对同一 progress cache line 的写入。
每个线程先发布一个较短的首前缀，之后按 chunk 控制的字节预算跨相邻 shard
复制；一个发布周期可以更新多个 shard 的 progress word。

每个周期仍执行 `copy → system barrier → release store progress → system barrier`。
没有删除或弱化 barrier，progress 仍然只覆盖已完成的 payload。最终 ACK 仍在
所有 helper 汇合之后发布。

以每 lane 4096×656 B、SGL=6、16 活跃线程、128 KiB chunk 为例，64 个 shard
分成每线程连续 4 片。首片约 40 KiB，后续约 124 KiB 可以合并发布；发布周期
由原来的 64 次降为 32 次，对应发布路径的 barrier 由 128 次降为 64 次。
64 个 progress word 仍各自更新，不要求消费者原子地观察整组 word。
这是发布次数计算，不能直接当作 CPU 延迟减半。

CPU thread profile 增加 `progress_publications`，统计该线程执行的发布周期数；
关闭 CPU profile 时不增加该诊断计数。

## 版本与配置

kernel 符号为 `KvafGatherBatchGetAivKernelV6`，args 仍是 240 B，profile 升为
256 B / PRF2 magic。Host、Runtime、Python binding 和 kernel 必须配套重编。
loader 拒绝 V1～V5，Host 也核对 profile 的实际 packed 模式。

CPU 服务需重编并重启才能获得新的跨片发布策略；旧 GDR3 CPU 服务仍能与 V6
kernel 通信。NPU 两种调度器可在同一份二进制上做 A/B，CPU 服务保持同一版本。

| 配置 | 执行方式 |
| --- | --- |
| `KVAF_AIV_GATHER_PACKED_POSTS=1`（默认） | GDR3 ready 库存、离散 WR 合批 |
| `KVAF_AIV_GATHER_PACKED_POSTS=0` | 保留 V5 的 GDR3 连续 run 调度器 |
| `KVAF_AIV_GATHER_SHARD_PIPELINE=0` | 原 GDR2 连续前缀流水，packed 配置不生效 |
| `KVAF_AIV_GATHER_CHUNK_BYTES=0` | 整批 ACK 后 READ，packed 配置不生效 |

这些环境变量在 Runtime 初始化时读取，改变后需重建实例。probe 和大 token
Scalar 回退继续自动关闭分片流水。

```bash
cmake --build build-cpu -j
# 使用原服务参数重启 CPU 服务。
cmake --build build-npu -j
export PYTHONPATH="$PWD/build-npu/python"
export KVAF_BATCH_GET_KERNEL="$PWD/build-npu/bin/kvaf_gather_batch_get_kernels.o"
KVAF_AIV_GATHER_PACKED_POSTS=1 python -u tests/python/test_aiv_gather_batch_get.py

# 固定其他负载、服务、核数、CPU 线程与 profile 设置，交替采集。
KVAF_AIV_GATHER_PACKED_POSTS=0 python -u tests/python/test_aiv_gather_batch_get_perf.py
KVAF_AIV_GATHER_PACKED_POSTS=1 python -u tests/python/test_aiv_gather_batch_get_perf.py
```

## 诊断与模型结果

CSV schema 为 6，记录 configured_packed_posts 和每核实际 packed_posts。原有
Host/core 阶段时间定义不变；新增以下计数，不进行跨 CPU/NPU 时钟相减：

| 字段 | 含义 |
| --- | --- |
| read_vf_calls | READ WQE VF 构造次数；包含 legacy 路径的重复构造 |
| ready_no_change_queries | packed 进度查询成功但 payload ready 数未增加的次数 |
| ready_query_skips | packed 库存充足、尚有未知进度，但无需新查询的 post 次数 |
| max_ready_backlog_wrs | packed 快照可用但尚未计划的 WR 库存峰值 |
| ready_idle_waits | packed 库存为空时前台确认进度查询的次数 |
| multi_run_posts | 一个 packed post 包含多个不连续逻辑 run 的次数 |
| read_plan_segments | 所有 packed post 中的逻辑 run 数之和 |

除 read_vf_calls 外，以上新计数只适用于 packed 调度器。read_vf_calls 应结合
data_doorbells 比较；正常 packed 执行每个 post 构造一次。

生产 kernel 模型复核，每 lane 4096×656 B、SGL=6，共 683 READ WR：

| ready 模式 | legacy VF / doorbell / query | packed VF / doorbell / query |
| --- | --- | --- |
| 全部 ready | 7 / 6 / 1 | 6 / 6 / 1 |
| 每次查询逆序就绪一片 | 127 / 64 / 64 | 32 / 32 / 64 |
| 奇数片先 ready，随后全部 ready | 11 / 9 / 2 | 6 / 6 / 2 |

以上均只有一次数据尾 CQ。模型验证调度次数和范围，不代表真机时间或收益比例。
覆盖分片 0 延迟、零散 ready、短尾 WQE 位于 post 中间、实际 WQEBB 环回、
小队列容量、重复/空请求、1/3/8 核、SGL 1/6/13、coalesced SGE、进度与数据 CQ
错误/超时、最终 ACK 和失败后的资源拒绝复用。SGL=9/13 的硬件支持范围仍需按
目标 SDK/设备验证。

目标机验收应先检查完整 output/guard，再比较 profile 关闭的 E2E p50/p95/p99；
诊断轮同时观察 VF、doorbell、query 和 CPU copy/publication，确认新增计划工作
没有抵消批次合并的收益。当前环境不提供目标 CANN 编译或双机性能测量。
