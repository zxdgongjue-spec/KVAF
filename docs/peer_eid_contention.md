# NPU 单 CPU EID 基线与多 CPU EID 并发端口争用测试

`run_peer_eid_sweep.py` 接受一个或多个远端 CPU EID。一个 EID 时退化为单路
基线，所有 Jetty 都访问该 EID；多个 EID 且 Jetty 总数足够时，各路 BatchGet 由同一
Runtime、同一 stream 上的同一次 AIV kernel 发出，各自使用独立 Jetty/SQ/CQ。
多个 EID 且总共只有一个 Jetty（`cores * jettys == 1`）时，每组参数随机选择一个 EID，按单路基线执行。
脚本可以依次测试多组参数，每组都使用同一份 CPU EID 候选列表。

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
实际选中的 EID 未被 CPU 服务公布、EID 重复、建链失败时直接失败，不回退到其他 EID。
服务端若通过 `--ub-device` 限制为单个设备，可能无法公布全部指定 EID。

三路及以上仍使用逗号分隔的 `--peer-eids`，并保证 `cores * jettys` 不少于 EID 数量。
例如，将 `CPU_EID_0`、`CPU_EID_1`、`CPU_EID_2` 和 `NPU_EID` 设为实际 EID 后，
三路、每路两个 Jetty 的配置为：

```bash
KVAF_CPU_SERVICE_ADDR=127.0.0.1:29000 DEVICE_ID=7 \
python3 tests/python/run_peer_eid_sweep.py "$PWD/build-npu" \
    --peer-eids "$CPU_EID_0,$CPU_EID_1,$CPU_EID_2" \
    --npu-eid "$NPU_EID" --cores 1 --jettys 6 \
    --output-dir results/peer-contention-three-001
```

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
与多路比较时，保持 `cores`、`jettys`、`miss-count`、token 大小和 inflight
等参数一致，并使用不同输出目录。EID 数量不会自动改变总请求量：单路全部搬运，
多路分担同一总量；例如 `--jettys 4 --miss-count 2048` 在单路是四个 Jetty
都指向同一个 EID，在双路是每个 EID 两个 Jetty。

也可以保留两个候选 EID，使用 `--cores 1 --jettys 1`：

```bash
KVAF_CPU_SERVICE_ADDR=127.0.0.1:29000 DEVICE_ID=7 \
python3 tests/python/run_peer_eid_sweep.py "$PWD/build-npu" \
    --peer-eids "$CPU_EID_0,$CPU_EID_1" --npu-eid "$NPU_EID" \
    --cores 1 --jettys 1 --output-dir results/peer-random-baseline-001
```

仅总共一个 Jetty 时，每组参数从候选列表中等概率选择一个 EID，只选择一次，预热和测量迭代
都使用该 EID。不会在每个请求或迭代之间重新选择；再次运行可能选中相同的 EID。
此时使用单路的请求量限制和默认 inflight 上限 2048，不做多 EID 并发校验。
日志 `BATCH_GET_PEER_EID_SELECTION` 记录候选列表和选中项；`config.json` 的
`requested_peer_eids` 保留候选列表，`peer_eids` 和 `peer_samples.csv` 使用实际选中的 EID，
该 EID 的 path 下标为 0。直接运行 `test_npu_batch_get_perf.py --aiv-cores 1 --jettys-per-core 1`
时也支持相同选择规则。要复现固定路径，可只传入日志中选中的那个 EID。

两个 EID 配合 `--cores 2 --jettys 1` 时，总共两个 Jetty，保留两个 EID：

```text
core 0 / Jetty 0 -> peer EID 0
core 1 / Jetty 0 -> peer EID 1
```

每核 Jetty 数不少于 EID 数 N 时，保持 `jetty_slot % N` 的原有映射。
每核不足、但总数足够时，使用 `(jetty_slot * cores + core) % N` 跨 core 分配，
先分配各 core 的第一个 Jetty，再分配第二个。总数大于 1、但仍少于 EID 数时直接报错。

## 参数和测量口径

### 一次测试多个组合

`--token-bytes`、`--miss-count`、`--cores`、`--jettys` 各接受一个整数或一组逗号分隔的整数。
脚本按这四个参数的顺序遍历全部组合，最内层先切换 `--jettys`。下面的命令依次测试
2 × 3 × 2 × 2 = 24 组，每组都由同一个 kernel 并发访问两个 CPU EID。

```bash
KVAF_CPU_SERVICE_ADDR=127.0.0.1:29000 DEVICE_ID=7 \
python3 tests/python/run_peer_eid_sweep.py "$PWD/build-npu" \
    --peer-eids 00000000003f020000100000df080b00,000000000000060000100000df080100 \
    --npu-eid 00000000003f030000100000df00eb00 \
    --token-bytes 656,1024 --miss-count 2048,4096,8192 \
    --cores 1,2 --jettys 2,4 \
    --output-dir results/peer-sweep-001
```

也可沿用旧脚本的空格分隔环境变量 `KVAF_BATCH_GET_SWEEP_TOKEN_BYTES`、
`KVAF_BATCH_GET_SWEEP_MISS_COUNTS`、`KVAF_BATCH_GET_SWEEP_AIV_CORES`、
`KVAF_BATCH_GET_SWEEP_JETTYS_PER_CORE`。命令行优先于对应环境变量。两者都省略时，
默认只测一组，使用 656 字节、`miss-count=request-length`、1 core、每核 4 个 Jetty。
脚本启动 benchmark 前会校验全部组合。运行中某组失败或超时会立即停止，已运行各组的明细保留。
例如两个候选 EID 配合 `--cores 1,2 --jettys 1,2,4` 时，只有 `1 core × 1 Jetty`
的组合随机选择一个 EID，其余组合都访问两个 EID；每组结果按实际生效的 EID 数量标记模式。

### 参数说明

| 参数 | 含义 |
| --- | --- |
| `--peer-eids` | 一个 CPU EID 为单路基线；多个候选 EID 在 Jetty 总数为 1 时随机选一个，否则须有足够的总 Jetty 访问全部 EID，顺序对应 path 0 到 N−1 |
| `--npu-eid` | 一个本地 NPU EID；设备由 `DEVICE_ID` 选择 |
| `--cores` | 一次 kernel 使用的 AIV core 总数，1–48 |
| `--jettys` | **每 core 的 Jetty 总数**，1–64；总数为 `cores * jettys`，除总数为 1 的随机单路模式外，总数须不少于 EID 数量 N |
| `--table-capacity` | 共享 DDR 表的条目数 |
| `--request-length` | 请求数组容量，默认 32768 |
| `--miss-count` | 每次 BatchGet 的有效请求总数；单路全部搬运、多路分担；默认等于 request-length |
| `--token-bytes` / `--tokens-per-block` | 默认 656 / 128 |
| `--warmup` / `--iterations` | 默认 3 / 100，warmup 不计入结果 |
| `--max-inflight-total` | 总 inflight 上限；单路默认 2048，与原性能测试一致；多路默认 `max(2048, cores * jettys * 256)` |
| `--timeout` | 每个组合的 benchmark 超时秒数，默认 600 |
| `--kernel-profile-clock-mhz` | AIV 周期换算频率，默认读取 `KVAF_BATCH_GET_KERNEL_CLOCK_MHZ`，未设置时为 1000 MHz，与原 miss-count 脚本一致 |

双路时，`--cores 1 --jettys 4` 是 1 core、共 4 Jetty，每个 CPU EID 分到 2 个。
这不等同于两个各用 1 core、4 Jetty 的独立测试；比较基线时应匹配资源和流量。
每核轮转时每核 Jetty 数、跨核分配时总 Jetty 数，宜为 EID 数量的整数倍，避免各路分配不对称。
请求以每 post 最多 128 个 READ 的粒度轮转，尾部也可能使各路字节数不同，实际分配以 CSV 为准。
多路测试要求 `miss-count >= cores * jettys * 128`，以保证所有 Jetty 都有流量；
默认请求量不足时需同步增大 `--request-length` 和 `--miss-count`。

所有 CPU EID 对**同一块 DDR backing**分别注册，拥有各自的 token/segment/channel。
这样 BatchPut 只需通过第一条路径填充数据，所有 EID 读取的是同一张已验证的表。
`table-capacity=16384` 的默认 DDR 大小为 `16384 * 128 * 656 = 1375731712` 字节，
不是每个 EID 各分配这么多内存。这避免多份 DDR 分配，但内存工作集与旧的双服务
测试不同；测量仍同时包含 DDR、UB、NPU 等因素。

## 为什么不会按 EID 串行执行

1. CPU 的 TCP 请求只用于发现、注册和建链。建链完成后保留各路注册，READ
   数据直接经 UB 传输，CPU 的 TCP accept 循环不参与 BatchGet 数据搬运。
2. NPU 将每个 Jetty 的远端 EID、token、TP 和队列信息写入 entity table。
   每核 Jetty 足够时按 peer 顺序轮转；不足时在各 core 之间分配路径。
3. 每个 core 的提交组先向参与的 Jetty 发出 doorbell，再检查完成；不会先等待一个 EID
   的全部 BatchGet 完成再启动另一个。脚本检查请求数和 inflight 上限，确保首个组
   足够覆盖所有配置的 Jetty。各条 doorbell 指令之间仍有指令执行时间，并非同一周期。
4. kernel 原本对跨 Jetty 的远端 token 做相等校验，现允许每个注册使用独立凭证，
   同时继续校验所有路径的 DDR/HBM 地址范围一致。

## 查看结果

终端输出与 `tests/run_remote_cpu_batch_get_miss_count_perf.sh` 保持相同的主要结构：
运行时实时显示原生建链日志、测试阶段、`PERF_RESULT` 和每 Jetty 表，并写入该组的
`batch_get.log`；全部组合成功后打印三张 `BATCH_GET_MISS_SWEEP_STAGE_SUMMARY` 表：

| stage | 主要列 |
| --- | --- |
| `kernel_total` | token_bytes、miss_count、aiv_cores、jettys_per_core、mean_us、p50_us、p90_us、p99_us、p50_GB/s |
| `kernel_post` | 相同配置列，以及 mean_us、p50_us、p90_us、p99_us |
| `kernel_completion_wait` | 相同配置列，以及 mean_us、p50_us、p90_us、p99_us |

微秒数使用 `cycles / clock_mhz`，不从四舍五入后的毫秒数换算；
`p50_GB/s` 使用本次合计的 `miss_count * token_bytes` 除以 kernel_total 的 p50
时间，以十进制 GB/s 表示，与原脚本口径一致，不再额外乘以 EID 数量。

单组合沿用原目录结构，全部文件直接写入 `--output-dir`。多组合时，根目录的
`batch_get.log` 记录计划、各组进度和最终汇总。查看单组原生建链、预热、校验及每 Jetty 性能时，
打开对应子目录，例如 `token_bytes_656_miss_count_2048_aiv_cores_1_jettys_per_core_2/batch_get.log`。

| 输出文件 | 内容 |
| --- | --- |
| 根目录 `batch_get_miss_sweep_summary.csv` | 每个组合一行，顺序与执行顺序一致。列顺序与旧脚本相同，`selected_ub_device` 留空，精确 EID 见各组明细 |
| 根目录 `batch_get_jetty_summary.csv` | 合并全部组合的每 Jetty 测量结果，可按 `token_bytes`、`miss_count`、`aiv_cores`、`jettys_per_core` 筛选 |
| 每组 `config.json` | 候选与实际 EID 列表、实际启动参数、命令、设备、CPU 地址及该组 inflight 上限 |
| 每组 `jettys.csv` | 测量迭代的每 Jetty 汇总，`eid_path` 为实际 CPU EID 列表下标 |
| 每组 `peer_samples.csv` | 每次迭代、每个 core 实际分配的 CPU EID 的请求数、字节数、首末 doorbell/CQE 观测周期及核内 `first_wave_observed_overlap_cycles` |

多路成功的运行应包含 `BATCH_GET_PERF_CORRECTNESS_PASS`、每个 EID 一条 `PEER_EID_TRAFFIC`。
每个样本中，所有配置的 Jetty 和 EID 都必须有完成的流量。对于同一 core 上的多条路径，
所有路径的首个 doorbell 均须早于该 core 任一路首次观测到 CQE，否则测试失败。
每个 core 都覆盖全部 EID 时，输出 `PEER_EID_CONCURRENCY ... execution=single_kernel`。
跨 core 分配时，输出 `PEER_EID_MULTI_CORE_TRAFFIC ... cross_core_overlap=not_measured`：
它证明同一 kernel 的各 core 分别访问了指定 EID，不证明不同 core 的传输时间重叠。
只分到一个 EID 的 core，其 `first_wave_observed_overlap_cycles` 留空。

单路则输出一条 `PEER_EID_TRAFFIC` 和
`PEER_EID_BASELINE ... concurrency_check=not_applicable`，不输出多路并发结论。
同样生成三个性能汇总表及全部 CSV；`peer_samples.csv` 的
`first_wave_observed_overlap_cycles` 留空，空闲 core 的请求数和时间戳记为 0。
启动日志和 `config.json` 的 `mode`：单路为 `single_eid_baseline`，双路保留
`dual_eid_concurrent`，三路及以上为 `multi_eid_concurrent`。扫描同时包含单路和多路组合时，
根日志的配置模式为 `mixed`，每组日志和 `config.json` 仍记录该组实际模式。

这里的重叠量是 **doorbell 到观测到 CQE 的未确认请求窗口**；CQE 轮询有延迟，
它不能证明物理线上每一瞬间都存在所有路径的流量。不同 core 的周期不直接比较。
判断是否争用你给定的共享物理端口，还需对照单 EID 基线以及端口计数器；
代码本身不推断 EID 到物理端口的拓扑。建议使用已有端口统计脚本辅助观察。

跨 core 分配修改了 NPU runtime 校验和 BatchGet host 建链逻辑，需重新运行
`./build.sh npu --build-dir build-npu`，并更新本次 Python 脚本。首次启用多 EID 功能所需的
两侧 v7 构建要求仍适用。原有命令可继续使用，请换一个尚无同名结果的输出目录。

## 已完成的验证与目标机验收

本地 Linux 默认构建和 8 个 CTest 目标通过，包括模拟 TCP PeerLink 的精确 EID、
共享地址注册、每路径建链、部分释放失败重试；真实 kernel 源码的 CPU 模型验证了
两路不同远端 EID/token 的 WQE，以及四个 Jetty 均先提交再首次轮询完成。
Python 测试覆盖并发/串行观测、空闲路径、路由错误和参数限制。

本次总 Jetty 数与跨 core 分配修改在 Windows 上通过 43 项 Python 测试和 kernel CPU 模型验证，
覆盖 `2 core × 1 Jetty`、`4 core × 1 Jetty`、跨核三 EID 路由、原有每核轮转及对应 WQE/payload。
CPU 模型按顺序模拟各 core，不用于证明跨核时间重叠。

当前开发环境无 CANN/UMDK/NPU，尚未完成两侧硬件构建及 NPU7 实测。
目标机运行上述命令后，先核对 `PEER_EID_SELECTED` 和 `BATCH_GET_PEER_ROUTE`
映射，再检查 payload 校验、两路 bytes 和 overlap 记录，最后比较同资源条件下
单路/双路带宽及端口计数。当前验证不代表硬件吞吐或端口争用已得到实测确认。
