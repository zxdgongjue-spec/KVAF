# NPU 到两个 CPU EID 的并发端口争用测试

`run_peer_eid_sweep.py` 一次运行两条远端 EID 路径。两路 BatchGet 由同一
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

## 参数和测量口径

| 参数 | 含义 |
| --- | --- |
| `--peer-eids` | 恰好两个不同的 CPU EID，顺序对应 path 0、1 |
| `--npu-eid` | 一个本地 NPU EID；设备由 `DEVICE_ID` 选择 |
| `--cores` | 一次 kernel 使用的 AIV core 总数，1–48 |
| `--jettys` | **每 core 的 Jetty 总数**，2–64，按 0、1、0、1 分配给两个 EID |
| `--table-capacity` | 共享 DDR 表的条目数 |
| `--request-length` | 每次 BatchGet 的总请求数，默认 32768，两路共同分担 |
| `--token-bytes` / `--tokens-per-block` | 默认 656 / 128 |
| `--warmup` / `--iterations` | 默认 3 / 100，warmup 不计入结果 |
| `--max-inflight-total` | 总 inflight 上限；默认 `max(2048, cores * jettys * 256)` |
| `--timeout` | 完整测试超时秒数，默认 600 |

因此 `--cores 1 --jettys 4` 是 1 core、共 4 Jetty，每个 CPU EID 分到 2 个。
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

- `config.json`：启动参数、命令、设备和 CPU 地址。
- `batch_get.log`：原生建链/路由日志、预热、校验、总吞吐及每 Jetty 性能输出。
- `jettys.csv`：测量迭代的每 Jetty 汇总，`eid_path` 为 CPU EID 列表下标。
- `peer_samples.csv`：每次迭代、每个 core、每个 CPU EID 的请求数、字节数、
  首末 doorbell/CQE 观测周期及 `first_wave_observed_overlap_cycles`。

成功的运行应包含 `BATCH_GET_PERF_CORRECTNESS_PASS`、两条 `PEER_EID_TRAFFIC`，
以及 `PEER_EID_CONCURRENCY ... execution=single_kernel`。
每个样本/core 都必须两路有流量，且两路首个 doorbell 均早于任一路首次观测到 CQE。
不满足时测试失败，不把仅启动了两条路径当作并发测试成功。

这里的重叠量是 **doorbell 到观测到 CQE 的未确认请求窗口**；CQE 轮询有延迟，
它不能证明物理线上每一瞬间都存在两路流量。不同 core 的周期不直接比较。
判断是否争用你给定的共享物理端口，还需对照单 EID 基线以及端口计数器；
代码本身不推断 EID 到物理端口的拓扑。建议使用已有端口统计脚本辅助观察。

## 已完成的验证与目标机验收

本地 Linux 默认构建和 8 个 CTest 目标通过，包括模拟 TCP PeerLink 的精确 EID、
共享地址注册、每路径建链、部分释放失败重试；真实 kernel 源码的 CPU 模型验证了
两路不同远端 EID/token 的 WQE，以及四个 Jetty 均先提交再首次轮询完成。
Python 测试覆盖并发/串行观测、空闲路径、路由错误和参数限制。

当前开发环境无 CANN/UMDK/NPU，尚未完成两侧硬件构建及 NPU7 实测。
目标机运行上述命令后，先核对 `PEER_EID_SELECTED` 和 `BATCH_GET_PEER_ROUTE`
映射，再检查 payload 校验、两路 bytes 和 overlap 记录，最后比较同资源条件下
单路/双路带宽及端口计数。当前验证不代表硬件吞吐或端口争用已得到实测确认。
