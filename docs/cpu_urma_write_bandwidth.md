# CPU URMA WRITE → HBM 带宽测试

入口：`tests/run_cpu_urma_write_bandwidth.sh`，Python 控制端为
`tests/python/test_cpu_urma_write_bandwidth.py`。CPU 数据面实现为
`runtime/src/cpu_write_benchmark.cc`，通过现有 CPU 服务运行。

测量窗口只执行 CPU 发起的 URMA WRITE、真实尾 CQE 等待。不会运行 AIV/AICPU kernel、
BatchGet、scatter、清单构建、CPU gather 或 payload memcpy。源数据填充、WR/SGL 计划构建、
注册、建链、TCP 控制、Python 派发和 HBM 回读校验均在窗口之外。
初始化复用已验证的 CPU-push 注册和 Jetty 路径，因而仍需要既有 NPU binding、scatter/BatchPut
kernel 对象。初始化会执行一次现有的 BatchPut 连通性探测，该探测不属于测量。

测试占用专用 Runtime 会话。开始测试时暂停并 join 普通 notify worker，由同一线程独占队列；
测试后直接 shutdown，不再在该会话中运行 BatchGet。生产 BatchGet 数据路径未增加分支。

## 构建与运行

CPU 服务新增了控制命令，需要重编并重启；既有命令格式和 NPU Python 接口没有变化。
CPU 构建需已有 URMA 路径配置（新构建目录还需 `-DKVAF_BUILD_REMOTE_CPU_SERVICE=ON`
及对应的 `KVAF_URMA_ROOT`）。

```bash
# CPU 端：使用已有 build-cpu 配置。
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j --target kvaf_cpu_service kvaf_cpu_push_transport_test
ctest --test-dir build-cpu -R kvaf_cpu_push_transport_test --output-on-failure
./build-cpu/runtime/kvaf_cpu_service --listen 0.0.0.0 --port 28990 --gather-threads 8
```

NPU 端在仓库根目录、正常的 CANN/torch_npu 环境中运行；A+K 时将服务地址改为远端 CPU IP。

```bash
# 预览默认 9 组：4/16/64 MiB × 1/2/4 Jetty，不访问硬件、不创建输出文件。
bash tests/run_cpu_urma_write_bandwidth.sh "$PWD/build-npu" --dry-run

# 单组连续源峰值测试，固定之前已验证可用的路由。
DEVICE_ID=7 KVAF_CPU_SERVICE_ADDR=127.0.0.1:28990 \
bash tests/run_cpu_urma_write_bandwidth.sh "$PWD/build-npu" \
  --sizes 64M --message-sizes 64K --queue-counts 4 \
  --source-modes contiguous --source-bytes 128M \
  --npu-eids 000000000000020000100000df00e100 --peer-ub-device udmac0d1e2 \
  --warmup 30 --iterations 100 --output-dir results/cpu-write-peak

# 对比连续大包和 656 B 随机源 SGL，目标 HBM 均连续。
DEVICE_ID=7 KVAF_CPU_SERVICE_ADDR=127.0.0.1:28990 \
bash tests/run_cpu_urma_write_bandwidth.sh "$PWD/build-npu" \
  --sizes 4M,16M,64M --message-sizes 64K --queue-counts 1,2,4 \
  --source-modes contiguous,random-sgl --fragment-bytes 656 --source-bytes 128M \
  --npu-eids 000000000000020000100000df00e100 --peer-ub-device udmac0d1e2 \
  --output-dir results/cpu-write-matrix
```

`contiguous` 每个 WR 使用一个连续 source SGE，实际 message 长度为请求值与设备上限的较小值。
`random-sgl` 按可复现的随机映射从整个 CPU 源池选取 fragment；每个 WR 受实际 max_sge 和
message 上限共同限制。例如 max_sge=13、fragment=656 时每个满 WR 仅 8528 B，
`--message-sizes 64K` 并不意味着每个随机 SGL WR 都能达到 64 KiB。
源池跨度会影响随机访问结果；与 BatchGet 对照时需对齐 source-bytes、fragment 大小和队列数量。

## 指标与校验

每个样本覆盖一次完整 payload 从首批 WRITE 提交到所有 payload 尾 CQE 到齐的时间；
包含 WQE 提交、CQ 轮询和多波之间的间隙。只有队列尾 WR 请求 CQE，并使用强放置排序，
全部队列完成后才返回。大请求继续受 SQ/SGE credit 约束，不能通过提前返回放大带宽。

- `comm_us`：上述完整传输窗口。
- `bandwidth_GBps`：`payload_bytes / comm_ns`，十进制 GB/s。
- `post_us`、`cq_wait_us`：提交与 CQ 等待；多波情况下还会有波间管理时间。
- `data_wrs`、`source_sges`、`waves`、`tail_cqes`：实际数据工作量与完成计数。
- summary 的 `effective_GBps`：总字节除以总耗时；`p50_GBps`：payload/p50 耗时；
  `max_observed_GBps`：payload/最快样本耗时，是最高观测值，不是理论链路带宽。

连续源的稳定大包结果用于判断已选 EID/链路的可达上限，随机 SGL 结果用于判断碎片读取损失。
两种模式都用同一个 CPU 提交线程，可扫描 1/2/4 Jetty；不聚合多个 UB 设备的带宽。
`optimized=1` 表示 CPU 数据面启用了编译器优化。性能对照应使用相同的 Release 配置、NUMA
内存放置和 CPU 亲和性，避免将运行环境差异当成链路能力。

CPU 直接填充具有位置特征的 64-bit 模式。每组先完整回读一次 preflight，再更换 seed 执行正式
warmup/测量，最后完整回读 HBM 与独立生成的期望值比较。回读按 4 MiB 分块，校验全部 payload；
不只检查头尾，不将回读耗时计入带宽。

输出目录可重复使用并覆盖同名文件：`samples.csv` 为逐样本 16 列结果，`results.json`
保留配置、实际设备上限、原始纳秒计时、校验 seed 和各组汇总；每完成一组即落盘。

## 参数

| 参数 | 默认 | 含义 |
|---|---|---|
| `--sizes` | `4M,16M,64M` | 每样本有效字节，可用 K/M/G（二进制）后缀；64 B 对齐，最大 256 MiB |
| `--message-sizes` | `64K` | 每 WR 最大字节数，8 B 对齐，最大 65536 |
| `--queue-counts` | `1,2,4` | 启用的 CPU Jetty 数量 |
| `--source-modes` | `contiguous` | 可选 `contiguous,random-sgl` |
| `--source-bytes` | `128M` | CPU 源池跨度；包含连续模式时自动扩到至少容纳最大 payload |
| `--fragment-bytes` | `656` | 随机源片段长度，需为 8 的倍数 |
| `--post-batch-wrs` | `128` | 每次 API 提交的 WR 数，1..8192 |
| `--warmup` / `--iterations` | `20` / `100` | 每组预热/正式次数，各最多 4096 |
| `--seed` | `20260903` | 数据模式和随机源映射 seed |
| `--cpu-service` | 环境变量或 `127.0.0.1:28990` | CPU 服务地址 |
| `--device` | `DEVICE_ID` 或 `0` | NPU 设备号 |
| `--npu-eids` / `--peer-ub-device` | 自动 | 固定已验证的单个 NPU EID 和 CPU UB 设备 |
| `--timeout` | `600` | 单次控制 RPC 的秒数，包含该组 CPU 准备与测量 |
| `--output-dir` | 带时间/PID 的 results 子目录 | CSV/JSON 输出目录 |
| `--dry-run` | 关闭 | 只打印计划，不访问硬件 |

单组准备最多允许 1,048,576 个 source SGE。HBM 固定注册约 256 MiB，CPU 仅分配所选源池
加少量控制区，不分配原 miss-count 场景的超大表。使用 NumPy 和 `libascendcl.so` 进行回读校验。

无硬件检查：

```bash
python3 -m unittest discover -s tests/python -p test_cpu_write_bandwidth_config.py
```

## 定位总数据量增大后的带宽下降

`--diagnostic-suite` 固定每个样本实际 WRITE 总量，分别控制 CPU 源工作集、HBM 目标工作集、
每次提交并等到完成的在途窗口。默认都是 64 MiB 实际流量、64 KiB/WR、4 Jetty，改变以下参数：

| variant | 源工作集 | HBM 目标工作集 | 在途窗口 |
|---|---|---|---|
| baseline | 64 MiB | 64 MiB | 原 SQ credit 约束，一波 |
| source_reuse | 4 MiB | 64 MiB | 与 baseline 相同 |
| flight_limited | 64 MiB | 64 MiB | 每 4 MiB 等待全部尾 CQE |
| flight_source_reuse | 4 MiB | 64 MiB | 每 4 MiB 等待全部尾 CQE |
| flight_target_reuse | 64 MiB | 4 MiB | 每 4 MiB 等待全部尾 CQE |
| flight_both_reuse | 4 MiB | 4 MiB | 每 4 MiB 等待全部尾 CQE |
| baseline_repeat | 64 MiB | 64 MiB | 与首组相同，检查时间漂移 |

4 MiB 窗口由 `--suite-window-bytes` 修改；每样本总流量由 `--sizes` 修改。
源窗口缩小仅复用 CPU 源地址，目标仍写满整个 64 MiB，用于隔离源工作集影响。
目标窗口缩小需在上一波真实完成后才能重用；所以判断目标工作集时，比较
`flight_target_reuse` 与同样有完成等待的 `flight_limited`，不能直接比较不同在途窗口。
原始流量按全部 WRITE 的实际长度求和并检查等于 payload，复用地址没有减少统计中的实际传输。

先重编 CPU 服务并重启，再在 NPU 端运行：

```bash
cmake --build build-cpu -j --target kvaf_cpu_service kvaf_cpu_push_transport_test
ctest --test-dir build-cpu -R kvaf_cpu_push_transport_test --output-on-failure

DEVICE_ID=7 KVAF_CPU_SERVICE_ADDR=127.0.0.1:28990 \
bash tests/run_cpu_urma_write_bandwidth.sh "$PWD/build-npu" \
  --diagnostic-suite --warmup 30 --iterations 100 \
  --npu-eids 000000000000020000100000df00e100 --peer-ub-device udmac0d1e2 \
  --output-dir results/cpu-write-diagnose
```

诊断使用独立控制命令，旧命令和常规 16 列 CSV 兼容；无需重编 NPU binding 或 kernel。
启用诊断时 CSV 增加 variant、窗口和线程观察字段，JSON 还保存环境和自动对照结果。
控制台 `CPU_WRITE_DIAGNOSIS` 与 JSON 的 `diagnosis` 给出各组最快 25% 样本的带宽比值；
比值用于定位方向，不自动断言缓存、IOMMU 或硬件的具体根因。首尾 baseline 相差超过 10%
时会提示先复测时间漂移，避免将系统负载变化错误归因于窗口设置。

诊断指标：

- `thread_cpu_us`：该线程实际使用的 CPU 时间，来自 `CLOCK_THREAD_CPUTIME_ID`。
- `off_cpu_estimate_us`：墙钟窗口减线程 CPU 时间的近似差值。诊断采样边界略宽，微小负值取 0；
  用于识别抢占/中断等时间损失，不用来修正带宽分母。
- `voluntary_switches` / `involuntary_switches`：线程上下文切换增量。
- `minor_faults` / `major_faults`：该 CPU 线程的页故障，零计数不能排除设备侧地址转换开销。
- `cpu_start` / `cpu_end`：样本前后的逻辑 CPU；相同 CPU 不代表中间没有迁移。
- `inter_sample_gap_us`：上一样本完成到本样本首个提交的间隔，包含诊断采样和显式 idle；首个样本无前值时为 0。
- JSON `diagnostics.environment`：源页 NUMA 抽样分布、CPU NUMA 节点、所在 VMA 的
  KernelPageSize/MMUPageSize/AnonHugePages、cgroup v2 配额和整组限流计数变化。
  VMA 大页信息不能当作设备 TLB 粒度的证明；限流计数包括 warmup，且不覆盖不可见的祖先限制。

NUMA 查询只查询页面位置，不迁移页面；权限或 seccomp 禁止查询时保留 errno，不中断测试。
接口含义见 [move_pages](https://man7.org/linux/man-pages/man2/move_pages.2.html)、
[线程 CPU 时钟](https://man7.org/linux/man-pages/man2/clock_gettime.2.html)、
[getrusage](https://man7.org/linux/man-pages/man2/getrusage.2.html)。诊断采样仅在显式启用时运行。
线程计数读取放在传输计时区间之外，会略微改变样本间隔；优先比较同一套诊断里的对照，
不要将它与普通模式的差异直接当成硬件加速。带宽分母始终保留完整墙钟传输窗口。

结果判断顺序：

1. 慢样本新增约 1 ms 且 off_cpu 同步增长：先固定空闲 CPU，排查调度/限额；保持完整完成窗口计时。
2. source_reuse 比 baseline 明显快：源工作集相关，继续区分内存放置、CPU 缓存和设备地址转换。
3. flight_limited 在相同完整源/目标工作集下更快：单次在途量或流控/观测节奏相关，不能解释为源缓存改善。
4. flight_target_reuse 比 flight_limited 明显快：目标工作集或 HBM 入站路径相关。
5. 所有变体都无改善且线程无明显 off_cpu：再结合端口实际速率、TP/重传/拥塞计数检查设备路径。

可加 `--cpu N` 绑定 CPU 服务端执行本次 RPC 的逻辑核，结束后恢复原线程亲和性；N 不是 NPU 的
DEVICE_ID。绑定不会迁移已分配的源内存，内存放置应在启动服务时用 NUMA 策略控制。
`--idle-us 100` 在样本间插入不计时的间隔，可对照连续压力和突发节奏，不能将它计入有效带宽。

单独对照也可用 `--diagnostics --source-window-bytes 4M`、
`--diagnostics --flight-bytes 4M` 等参数；随机 SGL 可以采线程指标，但窗口复用对照限定为 contiguous。
目标窗口复用时完整校验窗口内最后一轮写入的数据；所有样本仍等待全部 payload 尾 CQE。
