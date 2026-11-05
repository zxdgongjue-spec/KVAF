# Gather CPU 性能采集

目标是解释 AIV `wait_cpu_us` 中 CPU 执行的部分，尤其是
4096 B × 8192、2 个 AIV 核、SGL=1 的约 1.4～1.5 ms 等待。
CPU 阶段和线程埋点均由 `KVAF_GATHER_PROFILE=1` 开启；关闭时不读取这些计时器。
CPU 埋点本身不改变通信 ABI，所有 SGL 共用同一路径。当前
[chunk 调度与 NPU 查询优化](aiv_gather_chunk_pipeline.md)需要同步更新 K 侧
CPU 服务和 N 侧 gather kernel；部署方式以上述文档为准。

## 采集

在 K 侧仓库根目录启动服务，将原来的其他参数保持不变：

```bash
cmake --build build-cpu -j
KVAF_GATHER_PROFILE=1 ./build-cpu/runtime/kvaf_cpu_service \
  --listen 0.0.0.0 --port 34555 --gather-threads 32 \
  --gather-mode adaptive --gather-affinity on --memory-page-mode thp \
  > gather-cpu-4096-sgl1.log 2>&1
```

N 侧保持相同环境，确认性能脚本的 `TOKEN_BYTES=4096`、`REQUEST_COUNT=8192`、
`TABLE_CAPACITY=16384`、`TOKENS_PER_BLOCK=128`、`GATHER_SGL_SIZE=1`，然后运行：

```bash
DEVICE_ID=4 KVAF_NPU_CORE_COUNT=2 \
  python -u tests/python/test_aiv_gather_batch_get_perf.py
```

另一次运行将 token 改为 656、SGL 改为 6，并使用不同 K 侧日志名。
保留完整 K 日志、同一轮 N 侧输出及 `profile_core.csv`，按
`session/lane/sequence` 核对；CPU lane 对应 AIV core。
N 侧 CSV 暂无 CPU session 字段，必须人为确认来自同一轮测试，不能将不同
session 的相同 sequence 混合。优先比较 CSV 中 profile 样本的 sequence 范围，
不要将 CPU 的 preflight、warmup、baseline 请求混入 profile 均值。

每 lane 预分配 512 条请求记录，每条含本请求所有活跃线程的记录。
计时循环内不分配记录、不打印；正常关闭 N 侧 Runtime、停止 gather lane 或
释放 region 时，K 侧输出缓冲日志，**不需要停止整个 CPU 服务**。
`CPU_GATHER_PROFILE_BUFFER` 包含 `samples/capacity/dropped`；超过容量的请求
仍然执行，但不再保存记录。132 次左右的这组测试不会超过容量。
进程被强制终止时缓冲可能丢失，应让 N 侧测试正常退出。

同步采集 K 侧 CPU 拓扑和服务进程的 CPU 范围，便于核对两个 lane 是否各有
16 个可用 CPU、是否存在 CPU 重叠或 NUMA 不匹配。服务 PID 可以在另一终端查找：

```bash
lscpu -e=CPU,NODE,CORE
pgrep -af '[k]vaf_cpu_service'
```

取得实际 PID 后读取 `/proc/PID/status` 的 `Cpus_allowed_list`、
`/proc/PID/task/TID/status` 的 `Cpus_allowed_list` 及 `/proc/PID/numa_maps`。
日志中的 `tid/cpu` 可以对应每个 gather 线程；`cpu` 是拷贝开始时所在 CPU，
单个 CPU 采样不能证明整段执行期间没有迁移。内存 NUMA 分布可以结合已有
`numastat -p PID` 工具检查。

## 请求阶段

`CPU_GATHER_PROFILE` 保留原字段，并补充控制和同步数据。所有 `_ns` 为 ns，
除以 1000 转为 us。

| 字段 | 含义 |
| --- | --- |
| `observed_ns` / `done_ns` | CPU 观察到新 notify / 写最终 ACK 并完成 barrier 的 CPU steady-clock 时间戳 |
| `start_ack_ns` | 观察 notify 后，写 started ACK 并执行 barrier 的耗时 |
| `prepare_ns` | 校验描述、发布任务和唤醒 helper 的耗时；首次调用含创建线程 |
| `copy_ns` | coordinator 开始执行 Copy 到所有 helper 完成，包含唤醒、调度、拷贝和等待 |
| `dispatch_to_copy_ns` | 发布 helper generation 到 coordinator 开始 Copy 的间隔，与 prepare 部分重叠 |
| `join_wait_ns` | coordinator 完成本地任务后等待 helper 的时间，是 copy 的子项；chunk 优化版在此期间也推进 ready 前缀 |
| `ack_ns` | 写最终 ACK 并完成 barrier 的耗时 |
| `total_ns` | 从观察 notify 到最终 ACK barrier 完成，包含上述执行阶段及边界开销 |
| `notify_poll_window_ns` | 最近一次带时间戳的空轮询到观察新 notify 的间隔，仅在 valid=1 时有效 |

`notify_poll_window_ns` 是本机“notify 已对 CPU 可见到 CPU 观察到”时间的一个
宽松上界，包含每 256 次轮询采一个时间戳带来的窗口，也可能包含调度停顿。
它不代表 NPU 发布通知到 CPU 的传输延迟。请求开始前没有采到空轮询时
`notify_poll_window_valid=0`，不能将输出的 0 当作零延迟。
passive 模式每轮询可休眠 20 us，因此采样窗口尤其宽。

`total_ns` 起点在 CPU 观察 notify 之后。CPU 无法单独知道远端何时发出通知，
也不能把两端的绝对时间戳直接相减。两侧同一请求的**持续时间**可对照，但
`AIV wait_cpu_us - CPU total_ns/1000` 只能作为线索：其中混有通知传输/轮询、
ACK 读取与 CQ 等待，而且 CPU 可能在 AIV 的 publish 阶段尚未结束时就开始执行。
该差值可能为负，不是严格可加的阶段，也不能命名为“通知耗时”。

## 每线程记录

`CPU_GATHER_THREAD_PROFILE` 每个活跃线程一行，worker=0 为 coordinator。

| 字段 | 含义 |
| --- | --- |
| `ready_delay_ns` | 发布 helper generation 到该线程取得任务；coordinator 使用 Copy 开始时刻 |
| `start_delay_ns` | coordinator 开始 Copy 到该线程开始 ProcessSource 的延迟，含 phase 等待和调度 |
| `copy_ns` | 该线程的 ProcessSource 墙钟耗时；chunk 模式含任务内 barrier，coordinator 还包含前缀发布 |
| `barrier_ns` | ProcessSource 返回到末尾同步完成；整批模式含 dsb sy，chunk 模式已在任务内同步，不重复执行末尾 barrier |
| `work_ns` | 该线程 ProcessSource 加 barrier 的墙钟耗时 |
| `thread_cpu_ns` | 同一工作区间附近的线程 CPU 时间；`cpu_time_valid=0` 时不可使用 |
| `bytes` | 该线程实际拷贝的字节数；各活跃线程相加应等于请求 bytes |
| `wait_path` | 该任务是否经过条件变量等待分支；谓词可能立即满足，不等于确实睡眠 |
| `tid/cpu` | Linux 线程 ID 和开始拷贝时的 CPU 编号 |

按每个 sequence 找到 `done_ns` 最晚的线程，判断该 lane 的关键路径。
线程并行执行，不能把线程 copy/work 墙钟时间相加解释整批延迟。

- `start_delay_ns` 很大：helper 未及时开始，优先检查唤醒、绑核和调度。
- `copy_ns` 很大，`thread_cpu_ns` 与 `work_ns` 接近：CPU 执行/访存本身较慢，
  再对照 NUMA、内存带宽、缓存及页表情况；内存 stall 仍计入线程 CPU 时间。
- `work_ns` 明显大于 `thread_cpu_ns`：工作期间可能有较长的未运行区间或中断；
  这是调度线索，两个时钟的读取边界不同，不能用微小差值认定被抢占。
- `barrier_ns` 高：检查数据写入完成/系统 barrier，而不是只看 memcpy。
  chunk 模式的任务内 barrier 计入 copy_ns，不能由较低的 barrier_ns 推断其没有成本。
- `join_wait_ns` 高：coordinator 在等待其他线程，结合最晚线程的启动和工作时间定位。
- CPU `total_ns` 只有几十/几百 us，而同请求 AIV 等待仍有 1.5 ms：CPU 执行不足以
  解释等待，继续检查通知观察之前及 ACK 发布之后的通信与调度，不先归因于 memcpy。

计时和线程 CPU 时钟本身有开销。这轮用于归因，正式吞吐关闭
`KVAF_GATHER_PROFILE`，不要将诊断轮当作无打点基线。

## 批量汇总到一个 CSV

将日志放入同一目录，文件名包含 SGL，例如 `gather-cpu-656-sgl6.log`、
`gather-cpu-4096-sgl1.log`。脚本通过请求中的 `bytes/rows` 识别 token 大小，
SGL 从文件名读取；文件名没有 SGL 时可用 `--sgl 6` 为本次输入指定同一个值。

```bash
python3 tests/python/analyze_gather_cpu_profile.py cpu-logs/ -o cpu_summary.csv
```

也可传入多个文件或通配符。默认只解析 `CPU_GATHER_PROFILE` 请求汇总行，
统计其中所有 `_ns` 耗时字段；绝对时刻 `observed_ns/done_ns` 不作为耗时统计。
失败请求和 valid=0 的字段不参与统计。每个文件、token 大小、SGL、lane、rows、
active_threads 分别统计，每个耗时字段在 CSV 中占一行。

仅处理指定字段、指定 sequence 范围（例如此前 profile 的 123～132）：

```bash
python3 tests/python/analyze_gather_cpu_profile.py 'cpu-logs/*.log' -o cpu_summary.csv \
  --fields total copy prepare join_wait --first-sequence 123 --last-sequence 132
```

输出包含 samples、mean、p50/p95/p99、min/max、stddev、variance、CV。
耗时和标准差均为 us，方差为 us²，CV 为百分比；方差使用 n−1 的样本估计，
只有一个样本时标准差/方差/CV 留空。分位数使用线性插值。
脚本保留慢样本，默认包含全部 sequence；需要排除预热时指定 sequence 范围。

## 相关实现

- `runtime/src/cpu_runtime.cc`：notify 轮询、请求阶段、缓冲和日志输出。
- `runtime/include/cpu_gather_executor.h`：CPU 线程/profile 结构和任务发布。
- `runtime/src/cpu_gather_executor.cc`：每线程拷贝、barrier、CPU 时间及 join 计时。

本机开发测试只校验数据结果、profile 时序和线程同步，不代表目标 K/N 硬件性能。
