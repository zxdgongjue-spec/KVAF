# AICPU Get：单次 launch 的有界窗口

Gather 和 Scatter 不再以 65536 限制整个 BatchGet。Host 仍只提交一次 kernel，
各 worker 在该次调用内循环处理窗口，直到设备上的 `miss_count` 有效前缀全部完成。
输入/输出张量容量、int32 有效计数和 ABI 字段范围仍须有效。

## 配置

```python
gather = kvaf.GetExecutionConfig("urma", "aicpu", "gather")
scatter = kvaf.GetExecutionConfig("urma", "aicpu", "scatter")

get_resources = [
    (gather, kvaf.ExecutionResources(cores=5, gather_ddr_bytes=64 * 1024**2)),
    (scatter, kvaf.ExecutionResources(cores=5, scatter_hbm_bytes=64 * 1024**2)),
]
# 传给 NpuRuntime(get_resources=get_resources, ...)，其他初始化参数沿用现有配置。
```

`gather_ddr_bytes` 只用于 URMA AICPU Gather Get，预算覆盖额外 CPU DDR 注册工作区的
payload、源描述符、控制区和显式 HugeTLB backing 对齐。`scatter_hbm_bytes` 只用于 URMA
AICPU Scatter Get，预算覆盖连续 HBM 工作区的 payload、源地址数组和控制区，
以及 mailbox WRITE 的 AICPU 源缓冲区。
通信队列、运行库内部开销、独立状态/profile/协调缓冲区及最终 output 不属于这两项工作区预算。
HBM 预算约束的是传给 ACL 的申请字节数，不是驱动内部保留量。

两字段默认均为 0，表示按 token 大小和 worker 数使用现有 65536 条布局，绝非零字节。
显式非零值选择预算内最大的 4096 倍数窗口；不足以容纳 4096 条时初始化报错。
窗口可以大于 65536，实际尾窗可以小于 4096。`request_capacity` 对所有 Get 仅是预热提示，
图/输出绑定仍按调用方声明的最大张量容量校验；Put 的配置容量限制不变。
预算在资源初始化时生效，变更后需重建执行实例和捕获图，不能移动运行中的工作区。

默认 5 workers、656 字节/token 时，Gather 默认逻辑 DDR 工作区为 44043648 字节，
Scatter 默认连续 HBM 工作区为 43516416 字节；默认窗口仍是 65536 条。
资源初始化日志 `GATHER_WINDOW` / `SCATTER_WINDOW` 输出有效窗口、预算与申请大小。

`gather_read_wave_wrs` 控制每个 worker 每个 READ 波次最多提交多少个数据 WR，显式值范围为
1..4096。`ExecutionResources` 未设置（`None`）时，kernel 按本次调用设备上的总有效
`miss_count` 自动选择：`miss_count <= 8192` 使用 512 WR，否则使用 64 WR。
判断依据是总有效请求数，不是输入张量容量、单 worker 片段或内存窗口大小。
显式配置如 `gather_read_wave_wrs=128` 时始终使用该值，优先于自动策略；仅 URMA AICPU
Gather Get 接受该参数。它独立于 DDR 内存窗口、每次 post 的 32 WR 和 CPU 的 128 KiB ready 块。
每波末尾 CQE 确认后才复用该波 WR/SGE；硬件可能已经提前完成部分 WR。
例如 5 workers、每 WR 合并 6 条 token 时，64 WR 对应全局最多 320 个未确认数据 READ WR，
即最多 1920 条逻辑请求。工作区容量和总请求数不因该参数改变。

预算和 WR 上限都在 NPU 客户端的 `ExecutionResources` 配置，通过 `NpuRuntime` 的
`get_resources` 传入；DDR 预算会在初始化时协商给 CPU 服务，WR 上限仅传入 Gather kernel。
已捕获的图保留自动模式或显式固定值。自动模式在每次 replay 时读取当前设备 `miss_count`，
所以同一张图可以随计数切换 512/64，且一次调用内所有内存窗口使用同一上限。
调用方须在 replay 前按流顺序更新原计数张量内容，保持图绑定的地址有效；无需增加 kernel
节点或 Host 读取计数。改变自动/固定模式或显式固定值需重建执行资源和捕获图。

测试入口接受 `--gather-read-wave-wrs`，环境变量为 `KVAF_BATCH_GET_GATHER_READ_WAVE_WRS`：

```bash
bash tests/run_remote_cpu_batch_get_perf.sh build-npu \
  --backend gather_read --aicpu-workers 5 --gather-ddr-bytes 67108864 \
  --request-length 1000000 --miss-count 1000000 \
  --table-capacity 16384 --tokens-per-block 128 --token-bytes 656 \
  --execution graph --graph-replays 10 --warmup 5 --iterations 20 --kernel-profile \
  --output-dir reports/gather-wave-auto
```

上述命令省略 WR 参数以测试自动默认值；改为 `--miss-count 8192` 时自动选择 512。
同一构建中添加 `--gather-read-wave-wrs 32`、128、256 等即可固定上限对照，无需逐次改源码重编。
`BATCH_GET_PERF_CONFIG` 和 summary.csv 的 `gather_read_wave_wrs` 记录计时用 `miss_count`
对应的有效值，`gather_read_wave_wrs_mode` 记录 `auto` 或 `fixed`。
NPU 初始化执行资源时的 `GATHER_READ_CONFIG` 输出自动策略或固定值及窗口容量。

## 精确选择窗口

设 `C` 为全局窗口条数，`W` 为 worker 数，`B` 为 token_bytes，`A64` 表示向上 64 字节对齐。

Gather：

```text
L = ceil(C / W)
S = A64(L * B)
D = A64(64 + 16 * L)
逻辑 DDR 字节数 = W * (S + D + 64) + 128
```

CPU 服务根据自身页模式计算实际 backing 字节数；显式 HugeTLB 向上对齐到配置的页大小
（包括 2 MiB、1 GiB 及其他支持的大小），普通页/THP 使用申请的逻辑长度。
预算计算与 CPU 分配器共用页大小解析结果。CPU 在预算内搜索最大的 `C = 4096*k`，
并核对每 lane payload 可以用 uint32 表示，然后返回窗口和 workspace 的注册信息。
NPU 依据返回窗口申请源描述符 HBM，避免根据未知 CPU 页模式估算错误。

Scatter：

```text
逻辑 HBM 字节数 = A64(A64(192 + 64*W + 8*C) + B*C)
```

Host 搜索预算内最大的 `C = 4096*k`，并核对窗口 payload 可以用 uint32 表示。
两条路径都使用整数二分，边界同时满足布局、对齐和 ABI 条件，不采用浮点估算。
源块布局和窗口大小互相独立；65536 字节的单 WR 传输限制继续保留。

## 窗口推进和完成

Gather 的 lane 工作区互相独立。每个 worker 遍历全局窗口，处理该窗自己的请求片段，
生成局部描述符、通知 CPU、沿用 128 KiB ready-prefix 流水线 READ 到最终 output。
确认本窗 READ 已完成且 CPU 已终止访问 descriptor/staging 后才复用 lane。
worker 间不新增窗口屏障；只在初始化和外层 Host 执行生命周期中共享资源所有权。

Scatter 所有 worker 协同推进同一全局窗口：构建源地址列表、built 屏障、通知 CPU，
随后沿用每 worker 1024 条、四 Jetty ready 的生产/消费流水线。
所有 worker 消费完成且 CPU done 已发布后才切换下一窗；空尾片段也参与屏障。
同一次 launch 中复用 staging 的后续窗口，在 ready 确认后刷新对应源区间并执行屏障，
避免复用地址残留上一窗口的缓存数据；ready/done 查询及描述符/notify 发布也处理同一次
launch 内的 DMA 可见性。只维护内部工作区，单窗口路径及调用方张量的 CANN 交接保持原样。

输入下标使用 `window_base + local_index`，staging/描述符使用 local_index，输出仍依据
原始 `output_indices` 写入。一次 kernel 中只领取一次 worker ticket；状态中的序号仍标识
整次 launch，独立的窗口序号跨 launch/replay 持续递增，避免旧 ready/ACK 被误认为新窗口。
零请求也推进 launch 身份；Scatter 使用一个不发 CPU 通知的空窗口完成协调。
Scatter 的 32 位 wire 序号耗尽会拒绝执行并要求重建资源，不允许静默回绕。

Profile 累计所有窗口的 bytes/spans/时间。错误报告保留整次 launch 序号、已完成条数和
失败窗口的全局请求位置；失败后不再发新窗口，继续原有隔离/资源回收流程，不能清零复用。
Scatter 软件超时按窗口重新计时；CANN 单次 launch 的总时限仍有效，大批次耗时需要实机验证。

## 部署和验证

Gather 的 READ 波次参数使用入口 `KvafAicpuGatherReadExecutionV5`。
Scatter 的 mailbox 请求使用入口 `KvafAicpuCpuPushScatterExecutionV5`。
自动策略沿用 V5 布局，以内部 0 表示自动模式；部署本次默认值变更也须重编并更新 AICPU 包。
必须一起更新 CPU 服务、Host 库、Python binding 和 AICPU 包并重启进程，避免旧布局混用。
新增的 Gather 窗口协商消息和 Scatter setup 的窗口字段在初始化阶段传递，不增加逐窗 Host RPC
或 kernel launch。窗口边界仍有 CPU 通知、ACK 和屏障成本，不保证所有预算下吞吐相同。

`BUILD_TESTING=ON` 的 `aicpu_get_window` 检查精确预算、4096 边界和大页对齐；
`aicpu_gather_window` / `aicpu_scatter_window` 执行真实 kernel 控制流，覆盖小于/大于 65536 的
窗口、百万条总请求、动态/零请求回放、全局输出映射、尾片段和中间窗口故障。
模型以 memcpy 代替 DMA，不能替代 A5 的缓存可见性、并发时序和大批次图回放验证。
Gather 模型还检查不同 WR 上限、非 32 倍数、缩小已分配缓冲区后的逻辑波次限制、
8192/8193 自动切换、显式覆盖、跨内存窗口策略、末尾 CQ 检查点及百万请求回放。
