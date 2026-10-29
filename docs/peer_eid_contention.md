# NPU 单 CPU EID 基线与双 CPU EID 并发端口争用测试

`run_peer_eid_sweep.py` 接受一个或两个远端 CPU EID。一个 EID 时退化为单路
基线，所有 Jetty 都访问该 EID；两个 EID 时，两路 BatchGet 由同一
Runtime、同一 stream 上的同一次 AIV kernel 发出，各自使用独立 Jetty/SQ/CQ。
脚本名称保留 sweep，但不逐个 EID 串行跑性能测试。

## 构建和运行

CPU/NPU 控制协议已升级为 **v7**，必须同步更新两侧程序以及 NPU kernel 和
Python binding。在各自目标机器上重新构建：

```bash
./build.sh cpu --build-dir build-cpu
./build.sh npu --build-dir build-npu
```

CPU 侧只启动一个服务，让它枚举全部 UB device/EID：

```bash
./build-cpu/runtime/kvaf_cpu_service --listen 0.0.0.0 --port 29000
```

NPU 侧：

```bash
KVAF_CPU_SERVICE_ADDR=127.0.0.1:29000 \
DEVICE_ID=7 \
python3 tests/python/run_peer_eid_sweep.py "$PWD/build-npu" \
    --peer-eids 00000000003f020000100000df080b00,000000000000060000100000df080100 \
    --npu-eid 00000000003f030000100000df00eb00 \
    --cores 1 \
    --jettys 4 \
    --table-capacity 16384 \
    --output-dir results/peer-contention-001
```

若 CPU 服务在另一台机器上，`KVAF_CPU_SERVICE_ADDR` 使用该机器可达的 IP。
输出目录可不存在或为空；已有同名结果文件时拒绝覆盖。
指定 EID 未被 CPU 服务公布、EID 重复、建链失败时直接失败，不回退到其他 EID。
服务端若通过 `--ub-device` 限制为单个设备，可能无法公布这两个 EID。

单路基线使用同一脚本，只需给 `--peer-eids` 传一个值，例如：

```bash
KVAF_CPU_SERVICE_ADDR=127.0.0.1:29000 \
DEVICE_ID=7 \
python3 tests/python/run_peer_eid_sweep.py "$PWD/build-npu" \
    --peer-eids 00000000003f020000100000df080b00 \
    --npu-eid 00000000003f030000100000df00eb00 \
    --cores 1 --jettys 4 --table-capacity 16384 \
    --miss-count 2048 \
    --output-dir results/peer-baseline-001
```

单 EID 模式下可使用 `--jettys 1`，也允许小请求量或 `--miss-count 0`，不要求
每个 Jetty/core 都有流量。精确 EID 选择及数据正确性校验仍然保留。
与双路比较时，保持 `cores`、`jettys`、`miss-count`、token 大小和 inflight
等参数一致，并使用不同输出目录。EID 数量不会自动改变总请求量：单路全部搬运，
双路分担同一总量；例如 `--jettys 4 --miss-count 2048` 在单路是四个 Jetty
都指向同一个 EID，在双路是每个 EID 两个 Jetty。

## 参数和测量口径

| 参数 | 含义 |
| --- | --- |
| `--peer-eids` | 一个 CPU EID 为单路基线；两个不同 CPU EID 为并发测试，顺序对应 path 0、1 |
| `--npu-eid` | 一个本地 NPU EID；设备由 `DEVICE_ID` 选择 |
| `--cores` | 一次 kernel 使用的 AIV core 总数，1–48 |
| `--jettys` | **每 core 的 Jetty 总数**，单路 1–64，双路 2–64；单路全部使用 path 0，双路按 0、1、0、1 分配 |
| `--table-capacity` | 共享 DDR 表的条目数 |
| `--request-length` | 请求数组容量，默认 32768 |
| `--miss-count` | 每次 BatchGet 的有效请求总数；单路全部搬运、双路分担；默认等于 request-length |
| `--token-bytes` / `--tokens-per-block` | 默认 656 / 128 |
| `--warmup` / `--iterations` | 默认 3 / 100，warmup 不计入结果 |
| `--max-inflight-total` | 总 inflight 上限；单路默认 2048，与原性能测试一致；双路默认 `max(2048, cores * jettys * 256)` |
| `--timeout` | 完整测试超时秒数，默认 600 |
| `--kernel-profile-clock-mhz` | AIV 周期换算频率，默认读取 `KVAF_BATCH_GET_KERNEL_CLOCK_MHZ`，未设置时为 1000 MHz，与原 miss-count 脚本一致 |

双路时，`--cores 1 --jettys 4` 是 1 core、共 4 Jetty，每个 CPU EID 分到 2 个。
这不等同于两个各用 1 core、4 Jetty 的独立测试；比较基线时应匹配资源和流量。
奇数 Jetty 会使两路分配不对称，公平比较宜使用偶数。请求以每 post 最多 128 个
READ 的粒度轮转，尾部也可能使两路字节数不同，实际分配以 CSV 为准。

两个 CPU EID 对**同一块 DDR backing**分别注册，拥有各自的 token/segment/channel。
这样 BatchPut 只需通过第一条路径填充数据，两个 EID 读取的是同一张已验证的表。
`table-capacity=16384` 的默认 DDR 大小为 `16384 * 128 * 656 = 1375731712` 字节，
不是每个 EID 各分配这么多内存。这避免双份 DDR 分配，但内存工作集与旧的双服务
测试不同；测量仍同时包含 DDR、UB、NPU 等因素。

## 为什么不会按 EID 串行执行

1. CPU 的 TCP 请求只用于发现、注册和建链。建链完成后保留两条注册，READ
   数据直接经 UB 传输，CPU 的 TCP accept 循环不参与 BatchGet 数据搬运。
2. NPU 将每个 Jetty 的远端 EID、token、TP 和队列信息写入 entity table。
   每个 core 内 Jetty 按 peer 顺序轮转，单 core 也能向两个 EID 发流。
3. 每个提交组先向参与的 Jetty 发出 doorbell，再检查完成；不会先等待一个 EID
   的全部 BatchGet 完成再启动另一个。脚本检查请求数和 inflight 上限，确保首个组
   足够覆盖所有配置的 Jetty。两条 doorbell 指令之间仍有指令执行时间，并非同一周期。
4. kernel 原本对跨 Jetty 的远端 token 做相等校验，现允许每个注册使用独立凭证，
   同时继续校验所有路径的 DDR/HBM 地址范围一致。

## 查看结果

终端输出与 `tests/run_remote_cpu_batch_get_miss_count_perf.sh` 保持相同的主要结构：
运行时实时显示原生建链日志、测试阶段、`PERF_RESULT` 和每 Jetty 表，同时写入
`batch_get.log`；结束后打印三张 `BATCH_GET_MISS_SWEEP_STAGE_SUMMARY` 表：

| stage | 主要列 |
| --- | --- |
| `kernel_total` | token_bytes、miss_count、aiv_cores、jettys_per_core、mean_us、p50_us、p90_us、p99_us、p50_GB/s |
| `kernel_post` | 相同配置列，以及 mean_us、p50_us、p90_us、p99_us |
| `kernel_completion_wait` | 相同配置列，以及 mean_us、p50_us、p90_us、p99_us |

微秒数使用 `cycles / clock_mhz`，不从四舍五入后的毫秒数换算；
`p50_GB/s` 使用本次合计的 `miss_count * token_bytes` 除以 kernel_total 的 p50
时间，以十进制 GB/s 表示，与原脚本口径一致，不再额外乘以 EID 数量。
每次调用依然只有一个配置（`combinations=1`），传两个 EID 时在该配置内并发。
没有自动改成原 shell 脚本的 2048/4096/8192 三组请求量。需要对比某一组时，
在原命令末尾追加例如 `--miss-count 2048`；两路共同分担这 2048 个请求。

- `config.json`：启动参数、命令、设备和 CPU 地址。
- `batch_get.log`：完整终端日志，包含原生建链/路由、预热、校验、总吞吐、
  每 Jetty 性能、最后三张汇总表以及失败信息。超时或子进程失败时返回非零，
  保留已有日志，不生成成功汇总。
- `batch_get_miss_sweep_summary.csv`：与原脚本同名、同列顺序的一行性能汇总。
  `selected_ub_device` 留空，因为脚本按精确 EID 选择；具体 EID 见 config/peer CSV。
- `batch_get_jetty_summary.csv`：与原脚本同名、同 schema 的 Jetty 汇总。
- `jettys.csv`：测量迭代的每 Jetty 汇总，`eid_path` 为 CPU EID 列表下标。
- `peer_samples.csv`：每次迭代、每个 core、每个 CPU EID 的请求数、字节数、
  首末 doorbell/CQE 观测周期及 `first_wave_observed_overlap_cycles`。

双路成功的运行应包含 `BATCH_GET_PERF_CORRECTNESS_PASS`、两条 `PEER_EID_TRAFFIC`，
以及 `PEER_EID_CONCURRENCY ... execution=single_kernel`。
每个样本/core 都必须两路有流量，且两路首个 doorbell 均早于任一路首次观测到 CQE。
不满足时测试失败，不把仅启动了两条路径当作并发测试成功。

单路则输出一条 `PEER_EID_TRAFFIC` 和
`PEER_EID_BASELINE ... concurrency_check=not_applicable`，不输出双路并发结论。
同样生成三个性能汇总表及全部 CSV；`peer_samples.csv` 的
`first_wave_observed_overlap_cycles` 留空，空闲 core 的请求数和时间戳记为 0。
启动日志和 `config.json` 的 `mode` 区分 `single_eid_baseline` 与 `dual_eid_concurrent`。

这里的重叠量是 **doorbell 到观测到 CQE 的未确认请求窗口**；CQE 轮询有延迟，
它不能证明物理线上每一瞬间都存在两路流量。不同 core 的周期不直接比较。
判断是否争用你给定的共享物理端口，还需对照单 EID 基线以及端口计数器；
代码本身不推断 EID 到物理端口的拓扑。建议使用已有端口统计脚本辅助观察。

输出格式调整及单 EID 支持只改 Python 脚本，无需再次编译 CPU/NPU 二进制；首次启用双 EID
功能所需的 v7 构建要求仍适用。原有命令可继续使用，请换一个尚无同名结果的输出目录。

## 已完成的验证与目标机验收

本地 Linux 默认构建和 8 个 CTest 目标通过，包括模拟 TCP PeerLink 的精确 EID、
共享地址注册、每路径建链、部分释放失败重试；真实 kernel 源码的 CPU 模型验证了
两路不同远端 EID/token 的 WQE，以及四个 Jetty 均先提交再首次轮询完成。
Python 测试覆盖并发/串行观测、空闲路径、路由错误和参数限制。

当前开发环境无 CANN/UMDK/NPU，尚未完成两侧硬件构建及 NPU7 实测。
目标机运行上述命令后，先核对 `PEER_EID_SELECTED` 和 `BATCH_GET_PEER_ROUTE`
映射，再检查 payload 校验、两路 bytes 和 overlap 记录，最后比较同资源条件下
单路/双路带宽及端口计数。当前验证不代表硬件吞吐或端口争用已得到实测确认。
