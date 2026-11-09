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

`KVAF_AIV_GATHER_COOPERATIVE_CHUNKS=2/4/8/16/32` 启用。默认自动启用 prepare-ahead；
显式 `PREPARE_AHEAD=0` 拒绝初始化。`COOPERATIVE_CHUNKS=0` 关闭此功能。

每个非空 lane 根据本次实际有效行数计算：

```text
chunk_bytes = align_up(ceil(lane_payload_bytes / configured_chunks), 64)
实际 chunk 数 = ceil(lane_payload_bytes / chunk_bytes) <= configured_chunks
```

因此部分前缀、不同大小的 lane 都按实际 payload 分块，短请求可能只有一块。
块边界允许位于 token/WR 内，AIV 始终按完整 WR 的源结束位置判定可提交范围。

CPU 不再让不同线程分别复制整批的不同大段；每个线程依次复制自己在块 0、块 1、
块 2……中的份额。块内按 64 B cache line 均衡划分，末尾精确处理有效字节。
线程数量仍按整批 payload 和已有线程预算选择，不因 chunk 变小而降低并行度。

每个线程复制一个非空份额后执行原有系统可见性 barrier，再对该块的独立完成
计数做一次原子递减，随后直接复制下一块，不等待其他线程。最后完成某块的线程
汇总已完整完成的连续块，并用单调 CAS 推进 `ready_bytes`；发布后保留系统 barrier。
最多 32 个块的完成者参与发布，不引入专用通知线程。coordinator 正在复制下一块时，
helper 仍可发布前一块，不能让前缀推进依赖 coordinator 的拷贝检查点。

```text
CPU 每个线程：块0的份额 → 块1的份额 → 块2的份额……
CPU 完成者：       ready0       ready1       ready2
AIV：       query + 提前准备 WQE → 提交 ready 的连续 WR
NIC：                       READ0 ─── READ1 ─── READ2
```

AIV 使用 40 B descriptor READ 查询 sequence / ready_bytes，按连续前缀消费。
准备游标可以领先 ready 游标，只有两者
共同覆盖的完整 WR 才提交。一个预构造 bank 中只有少量 WR ready 时，也提交这部分
并保留其余内容，不能等待凑满 128 WR。每条 WQE 只构造一次；bank 和在途 READ 跨
前缀保留。已准备且 ready 的首批优先提交，后续快照不插在其提交前处理。

最后数据 post 发布后，在 READ 仍在途时查询最终 ACK，再确认数据 CQ。ready=total
不替代最终 ACK，也不允许提前复用 descriptor/staging。错误继续 poison 两条队列，
消费本地已发出的 VF/MTE3 事件，并沿用原有会话停机和资源保留规则。

## 协议、ABI 与配置优先级

- kernel：`KvafGatherBatchGetAivKernelV9`，args 248 B；loader 拒绝 V1–V8。
- profile：PRF5，256 B；Python CSV schema 9。
- 第一步仍使用 GDR2 descriptor；第二步使用 gather 专用 GDR4 magic。
  GDR4 的 48 B header、16 B span、ready 前缀及最终 ACK 布局/含义与 GDR2 相同，
  magic 仅区分 CPU 任务调度。每 lane scratch 为 64 B，descriptor 后不附加进度向量。
- CPU 与 NPU 的 Host/Runtime/Python/kernel 均须同步重编部署。不支持 GDR4 的旧 CPU 会拒绝该请求，
  会返回错误 ACK，不能混合部署。NPU 运行日志应显示 V9 / args_bytes=248。
- 新增 16/32 档位保持 V9/GDR4 和所有字段布局不变，但 K 侧 CPU 服务与 N 侧
  Host/kernel 均须更新、重编并部署；旧 Host/kernel 不接受新配置，旧 CPU 的
  完成状态容量最多 8 块，收到超过 8 块的描述符会返回错误 ACK。
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

# C：提前准备 + 四个协作 chunk；随后单独比较 2 / 8 / 16 / 32
env KVAF_RUN_NPU_TESTS=1 KVAF_AIV_GATHER_COOPERATIVE_CHUNKS=4 \
  KVAF_AIV_GATHER_PREPARE_AHEAD=1 KVAF_AIV_GATHER_CHUNK_BYTES=131072 \
  python -u tests/python/test_aiv_gather_batch_get_perf.py
```

保持 K 侧线程数、adaptive 等配置不变，固定脚本中的 `TOKEN_BYTES`，按下列方式
比较各档位；随后将脚本的 `TOKEN_BYTES` 从 656 改为 4096，再运行同一组命令。
每档启动独立进程，Runtime 会捕获本次配置：

```bash
for chunks in 4 8 16 32; do
  env KVAF_RUN_NPU_TESTS=1 KVAF_AIV_GATHER_COOPERATIVE_CHUNKS="$chunks" \
    KVAF_AIV_GATHER_PREPARE_AHEAD=1 KVAF_AIV_GATHER_CHUNK_BYTES=131072 \
    python -u tests/python/test_aiv_gather_batch_get_perf.py
done
```

8192 行、2 个 AIV 核时，每 lane 的实际大小如下；协作模式下配置的
`CHUNK_BYTES=131072` 仍只用于启用 chunk，具体数值由实际 payload 与块数覆盖：

| 协作块数 | 656 B token 的实际 chunk | 4096 B token 的实际 chunk |
| --- | --- | --- |
| 4 | 656 KiB | 4 MiB |
| 8 | 328 KiB | 2 MiB |
| 16 | 164 KiB | 1 MiB |
| 32 | 82 KiB | 512 KiB |

CPU 活跃线程数继续按整批 payload 和现有 lane 预算选择，不随块数增加；每个线程
依次参与全部实际块。新增档位只扩大可选分块范围，未改变静态份额、线程等待策略、
WQE 编码、双 bank、控制查询或数据 CQ 规则。

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
2/4/8/16/32 块、不同 token、多核及空 lane；重复调用；进度回退/越界/错误和 CQ/超时退出。
CPU 并发消费者逐字节检查 ready 前缀，并暂停 coordinator 的后续份额验证 helper
发布，暂停首块份额验证前缀不跨空洞。模型和 sanitizer 不证明设备 DMA 可见性、
CANN 编译结果或实际性能收益，仍需目标双机验收。
