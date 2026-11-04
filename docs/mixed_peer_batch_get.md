# 两路独立配置的 BatchGet 并发与基线对比

入口为 `tests/python/run_mixed_peer_batch_get.py`，复用现有两路并发启动器和
BatchGet sweep，使用相同的 `--flow1/2-device`、`--flow1/2-priorities` 和
`--flow1/2-args` 接口。两个独立客户端进程在同一主机启动，可使用相同或不同
NPU；每路分别选择 CPU DDR 或 NPU HBM peer，不要求一条必须是 NPU、另一条必须是 CPU。
下面以 **NPU6 → NPU5，加上 NPU4 → 远端 CPU** 为例：

| 流量 | 发起端 | peer 类型 | 服务地址 | CPU UB 设备 |
| --- | --- | --- | --- | --- |
| `flow1` | NPU6 | `npu` | `127.0.0.1:28999`（服务使用 NPU5） | 不适用 |
| `flow2` | NPU4 | `cpu` | `CPU_HOST:29000` | `udmac0d1e3` |

`--flow1-device` / `--flow2-device` 分别指定发起端逻辑卡号，必须显式传入；
目标卡 NPU5 由服务启动参数选择。
`udmac0d1e3` 是远端 CPU 的 UB 设备名，不是 IP 地址。TCP 用于建链，
BatchGet 数据经 UB 传输。两路分别填充和校验自己的远端内存池。

此前专用入口的 `--device`、`--npu-*`、`--cpu-*` 参数已替换为上述 flow 接口；
日志目录也改为按 flow 编号区分。已有 `run_concurrent_batch_get.py` 的参数和
输出结构不变，它本身也支持这些 peer 组合；本入口额外提供共享参数和基线对比。

## 1. 启动两端服务

客户端和两个服务使用本分支构建的产物。NPU 主机先加载 CANN 环境并执行
`./build.sh npu`；CPU 服务器执行 `./build.sh cpu`。构建细节见
[NPU peer](npu_peer.md)。以下命令均从各自主机的项目根目录执行。

在 NPU 主机的一个终端启动 NPU5 服务，保持运行：

```bash
./build-npu/runtime/kvaf_npu_service \
  --listen 0.0.0.0 --port 28999 --device-id 5
```

在远端 CPU 服务器启动服务，保持运行：

```bash
./build-cpu/runtime/kvaf_cpu_service \
  --listen 0.0.0.0 --port 29000 --ub-device udmac0d1e3 --eid-index 0
```

`--eid-index` 按该服务器实际配置调整。容器内使用各进程可见的逻辑 NPU 编号；
本机服务的 `127.0.0.1` 需要客户端与服务共享网络命名空间，否则换成可达地址。

## 2. 分别配置两个 flow

在 NPU 主机另开终端，设置 CPU 服务器的实际 IP：

```bash
CPU_HOST=192.0.2.10  # 替换成实际 CPU 服务器 IP
python3 tests/python/run_mixed_peer_batch_get.py "$PWD/build-npu" \
  --flow1-device 6 --flow2-device 4 \
  --flow1-priorities 15 --flow2-priorities 15 \
  --flow1-args="--peer-device npu --peer-service-addr 127.0.0.1:28999" \
  --flow2-args="--peer-device cpu --peer-service-addr $CPU_HOST:29000 --peer-ub-device udmac0d1e3" \
  --common-args="--batch-get-mode lookup --request-length 8192 --miss-count 8192 --table-capacity 16384 --tokens-per-block 128 --token-bytes 656 --aiv-cores 2 --jettys-per-core 2 --warmup 10 --iterations 1000" \
  --output-dir results/mixed-npu6-npu4
```

若两路都从 NPU6 发起，只需改为 `--flow2-device 6`。每个 flow 的
`--peer-device`、`--peer-service-addr` 和 `--peer-ub-device` 都在自己的
`--flow*-args` 中设置；两个 flow 也可交换 peer 类型，或都配置为 `npu` / `cpu`。
NPU peer 不接受 `--peer-ub-device`。CPU 实验应固定 UB 设备或在服务端固定过滤器，
本入口与原启动器均不允许自适应 UB 校准。

两路各完成建链、BatchPut 填充、读回校验和预热后，在第一次计时前等待共同
起跑时刻。终端输出分别带 `[flow1]`、`[flow2]` 前缀；参数中的
`aiv-cores` 和 `jettys-per-core` 是每一路的配置。示例每个远端池约需
1.28 GiB，客户端还需要填充和输出等临时内存。

两路默认都启用 kernel/逐 Jetty profiling。`--flow1-priorities` 和
`--flow2-priorities` 分别控制两路本地 BatchGet Jetty 的 priority，可传
`3,13` 等列表，范围 0–15；未传时沿用 `KVAF_BATCH_GET_JETTY_PRIORITIES`，
该环境变量也未设置时为 15。SL 由设备的 priority→SL 映射决定。

## 3. 自动执行单路基线与并发对比

在上述命令中增加 `--with-baselines` 并使用一个新输出目录，即依次执行：

1. 仅 `flow1`（示例：NPU6 → NPU5）。
2. 仅 `flow2`（示例：NPU4 → CPU `udmac0d1e3`）。
3. 两个 flow 同时运行。

同一路在单路/并发阶段使用相同参数与 profiling 设置。输出结构为：

```text
results/mixed-npu6-npu4/
  baseline_flow1/flow1/     # 仅 --with-baselines 时生成
  baseline_flow2/flow2/     # 仅 --with-baselines 时生成
  concurrent/flow1/
  concurrent/flow2/
  comparison.csv           # 仅 --with-baselines 且三阶段全部成功时生成
```

每路目录包含原有 shape 日志、逐 Jetty CSV、`batch_get_miss_sweep_summary.csv`
和 `batch_get_jetty_summary.csv`。`comparison.csv` 每路一行，记录单路与
并发的 `mean_us`、`p50_us`、`p99_us`、`p50_GBps`，以及各指标的
`并发值 / 单路值`。耗时比值大于 1 表示变慢，带宽比值小于 1 表示下降；
基线为零时比值留空。这些指标沿用 sweep 的 AIV kernel 计时及有效字节带宽，
不是端口总带宽，也不包含初始化和预热耗时。

## 分别配置两路流量

`--common-args` 使用现有 benchmark 参数。`--flow1-args` / `--flow2-args`
追加各自的参数，同名的普通单值参数以后者为准，例如：

```bash
  --flow1-args="--peer-device npu --peer-service-addr 127.0.0.1:28999 --batch-get-mode raw --miss-count 4096 --iterations 2000" \
  --flow2-args="--peer-device cpu --peer-service-addr $CPU_HOST:29000 --peer-ub-device udmac0d1e3 --batch-get-mode no-pipeline --miss-count 8192 --iterations 1000"
```

固定客户端 EID 时，在 `--flow1-args` 字符串末尾追加
`--npu-eids <NPU6_EID列表>`，在 `--flow2-args` 中追加
`--npu-eids <NPU4_EID列表>`。每个列表属于该 flow 的发起端，不能混用两张卡的
EID。NPU5 服务的 EID 用服务端 `--npu-eid` 配置。每个 `--flow*-args` 传一个
完整字符串；重复传同一个 `--flow*-args` 时，后面的字符串会整体替换前面的。

同步只发生在首轮测量前，两路随后独立执行各自的迭代次数。它保证进程准备
完成后共同起跑；实际 kernel 是否重叠取决于设备调度。两路结束时间可能不同，
较慢一路末尾可能包含另一流已经结束的样本。应根据单路耗时调整各自迭代次数，
使测量持续时间接近，并结合现有 [端口采样工具](../tests/diagnostics/README.md)
确认流量重叠。若需比较稳定性，可更换输出目录重复实验。

输出根目录必须不存在，脚本拒绝覆盖已有实验。`KVAF_REMOTE_CPU_PERF_TIMEOUT`
默认 600 秒，分别限制每个阶段的准备与测量；任一阶段失败会停止后续阶段，
保留日志并返回非零状态。Ctrl+C/终止信号沿用现有启动器的子进程组清理。
脚本不负责启停两个服务。

## 无硬件验证

```bash
python3 -B tests/python/test_mixed_peer_batch_get.py -v
python3 -B tests/python/test_concurrent_batch_get.py -v
```

Linux 测试以替身替代 NPU benchmark，实际运行 Bash sweep、同步门和 CSV
汇总，检查同卡/异卡、NPU+CPU、CPU+NPU、NPU+NPU、CPU+CPU 的路由隔离、
基线顺序、参数继承和比较值。
真实 NPU/CPU 连通性与并发性能需要在目标设备执行上述实验。
