# 四种 Get 的多卡带宽测试

多卡入口支持 `aiv`、`ubmem`、`gather_read`、`cpu_push_scatter`。
CPU service 须先启动，各卡使用相同构建与已安装的 AICPU 包。
例如在 CPU 端运行：

```bash
./build-cpu/runtime/kvaf_cpu_service \
  --listen 0.0.0.0 --port 21111 \
  --gather-threads 8 --gather-mode active
```

在 NPU 仓库根目录测 device 1、2 的 Gather：

```bash
KVAF_CPU_SERVICE_ADDR=127.0.0.1:21111 \
bash tests/run_remote_cpu_batch_get_multicard_perf.sh build-npu \
  --backend gather_read --devices 1 2 --aicpu-workers 5 \
  --request-length 32768 --miss-count 4096 --table-capacity 4096 \
  --token-bytes 656 --tokens-per-block 128 \
  --source-access-pattern random --output-access-pattern contiguous \
  --no-sort-input-indices --seed 20260903 \
  --warmup 10 --seconds 10
```

设备号由 `--devices` 指定；每卡启动一个进程。脚本先逐卡测独立带宽，再同时启动各卡
测并发带宽。所有阶段使用已有 CPU service，脚本不会启动或重启服务。
每卡单独初始化 DDR 数据区；测试的是这些访问对共享 CPU、内存与通信资源的竞争。

## 每张卡连接独立的 CPU service

使用可重复的 `--device-peer DEVICE=HOST:PORT` 按设备号指定服务，映射与 `--devices`
中的顺序无关。可以连接同一主机不同端口的服务，也可以连接不同 CPU 主机。
例如先在 CPU 主机分别启动两个服务进程：

```bash
./build-cpu/runtime/kvaf_cpu_service \
  --listen 0.0.0.0 --port 21111 \
  --gather-threads 8 --gather-mode active >cpu-service-21111.log 2>&1 &
./build-cpu/runtime/kvaf_cpu_service \
  --listen 0.0.0.0 --port 21112 \
  --gather-threads 8 --gather-mode active >cpu-service-21112.log 2>&1 &
```

在 NPU 端指定 device 1、2 各自的服务（将 `cpu-host` 替换为 CPU 主机地址）：

```bash
bash tests/run_remote_cpu_batch_get_multicard_perf.sh build-npu \
  --backend gather_read --devices 1 2 --aicpu-workers 5 \
  --device-peer 1=cpu-host:21111 --device-peer 2=cpu-host:21112 \
  --request-length 32768 --miss-count 4096 --table-capacity 4096 \
  --token-bytes 656 --tokens-per-block 128 \
  --warmup 10 --seconds 10 --output reports/multicard-per-service
```

未设置 `--device-peer` 的卡使用 `--peer`；省略 `--peer` 时沿用
`KVAF_CPU_SERVICE_ADDR`，再回退到 `127.0.0.1:28888`。例如
`--devices 1 2 3 --peer cpu-host:21111 --device-peer 3=cpu-host:21112`
让 1、2 卡共享服务，3 卡使用另一个服务。同一设备重复配置、配置未参与测试的设备，
以及无效的 `DEVICE=HOST:PORT` 都会在启动 worker 前报错。

独立、并发和并行度扫描的所有阶段使用相同映射，四种后端均支持。
每个服务可独立配置线程数、内存页模式与外部 CPU/NUMA 亲和性，
设置方式见 [CPU service 说明](../runtime/README.md#cpu-runtime-服务与-npu-link)。
独立服务进程仍可能共享主机的 CPU、内存或链路资源；需要区分这些资源的竞争时，
应在启动服务时选择对应亲和性或主机。

## 持续 Graph 测量

每卡捕获一张 ACL 图，默认图内串行执行 256 个 Get。各 Get 使用独立的请求方案，
共用输出缓冲区。只有需要改变图大小时才传 `--graph-calls N`。
预热和测量都每轮回放一次、同步一次。测量前后按最后一个 Get 的请求检查对应输出行。
有效字节数为 `已完成的回放次数 × graph_calls × miss_count × token_bytes`，
带宽按实际测量时间计算。

在原多卡命令中加 `--kernel-profile` 并指定新的 `--output` 目录，可同时查看独跑和并发阶段每张卡的 kernel 结果。
设备在录图前启用 profiling。每次正式回放同步完成后，worker 读取图中最后一个 Get 的 profile。
默认图中有 256 个 Get，每轮产生 1 个 kernel 样本。校验和预热回放不计入样本。
每个 Get 的耗时取最早 worker 开始至最晚 worker 结束。AIV 按 `--kernel-profile-clock-mhz` 将周期换算为时间，
默认 1000 MHz，可按设备实际频率设置。AICPU 直接使用设备纳秒时间戳。

`summary.csv` 中的 `concurrent_kernel_mean_GBps` 用采样 Get 的有效字节总和除以这些 Get 的设备耗时之和，
`concurrent_kernel_p50_GBps` 用一次 Get 的有效字节除以设备耗时 P50。`isolated_kernel_*` 给出对应独跑结果。
带宽单位为十进制 GB/s。AIV 另有 `concurrent_kernel_max_core_p50_GBps`，每个样本先取各核自身耗时的最大值，
再取跨样本 P50 换算带宽，可与 peer-eid 的 `p50_GBps` 按相同定义对照。各卡的阶段耗时和 worker 明细见 `concurrent/device_<id>.log`；
AIV 的各核记录含 Jetty 轮询与等待数据。启用采样时，原 `concurrent_GBps` 的主机时间也包含每轮回读开销。

## 四种后端使用相同负载

| backend | 并行度参数 | 通信参数 |
|---|---|---|
| `aiv` | 本机 2 核，远端 A+K 4 核 | 本机每核 2 个 Jetty，远端每核 1 个，`--max-inflight-total 2048` |
| `ubmem` | `--aiv-cores 16` | 不使用 Jetty / READ 在途预算 |
| `gather_read` | `--aicpu-workers 5` | 使用算法自身的通信布局 |
| `cpu_push_scatter` | `--aicpu-workers 5` | 使用算法自身的通信布局 |

依次测试全部后端：

```bash
bash <<'BASH'
set -euo pipefail
report_root="reports/multicard-$(date +%Y%m%d-%H%M%S)"
for backend in aiv ubmem gather_read cpu_push_scatter; do
  case "$backend" in
    aiv) resources=(--max-inflight-total 2048) ;;
    ubmem) resources=(--aiv-cores 16) ;;
    *) resources=(--aicpu-workers 5) ;;
  esac
  KVAF_CPU_SERVICE_ADDR=127.0.0.1:21111 \
  bash tests/run_remote_cpu_batch_get_multicard_perf.sh build-npu \
    --backend "$backend" --devices 1 2 "${resources[@]}" \
    --request-length 32768 --miss-count 4096 --table-capacity 4096 \
    --token-bytes 656 --tokens-per-block 128 \
    --source-access-pattern random --output-access-pattern contiguous \
    --no-sort-input-indices --seed 20260903 \
    --warmup 10 --seconds 10 \
    --output "$report_root/$backend"
done
BASH
```

多卡入口的 `--aiv-cores 1 2 4` 或 `--aicpu-workers 1 2 4 5` 可以扫描并行度。
旧 `--requests N` 继续表示请求容量，省略 `--miss-count` 时全部请求有效。
现在也可分别配置容量与有效请求数、源/输出访问模式、输入排序、EID 和源数据填充批次，
含义与单卡入口一致。

## 单卡与多卡复用数据路径，保留各自计时方式

| 入口 | 测量方式 | profiling |
|---|---|---|
| `run_remote_cpu_batch_get_perf.sh` | eager/graph 逐次采样，输出延迟分位数和带宽 | 默认采集算子时间；`--kernel-profile` 增加分段明细 |
| `run_remote_cpu_batch_get_multicard_perf.sh` | 持续回放一张 ACL 图，每次回放同步一次，按时间窗口计算独立/并发吞吐 | 默认关闭，`--kernel-profile` 采集每轮最后一个 Get 的阶段及 worker 明细 |

多卡控制器调用单卡入口的持续回放 worker；两者复用负载准备、建表和数据校验。
公共参数/校验在 `tests/python/utils/get_options.py`，图输入更新、捕获与租约释放在
`tests/python/utils/get_graph.py`。多卡协调 worker 位于 `utils/multicard.py`，
控制器只负责设备进程、起跑屏障、测量窗口、扫描和报告。

单卡入口支持动态输入校验；多卡入口固定各节点的请求，校验在测量窗口外。
单卡默认 Torch Graph；与多卡对照时可在单卡命令上指定 `--graph-api acl`。
持续吞吐包含回放、等待及循环控制成本，与单卡 `scope=kernel` 的带宽口径不同。

## 查看帮助和结果

```bash
bash tests/run_remote_cpu_batch_get_multicard_perf.sh build-npu --backend gather_read --help
bash tests/run_remote_cpu_batch_get_perf.sh build-npu --backend gather_read --help
```

选定 backend 后，帮助只显示适用的资源参数；查看帮助不要求构建产物或 NPU。

结果目录包含 `config.json`、各阶段/各设备日志与原始 JSON，以及：

- `summary.csv`：各卡 CPU service（`peer`）、图数、独立/并发带宽、保留比例和下降比例。
- `aggregate.csv`、`aggregate.json`：总吞吐、窗口重叠率与起跑偏差；CSV 额外记录 `device_peers` 映射。
- `report.md`：配置、每卡服务地址和结果表；AICPU 单独标记 worker 数。

`config.json` 的 `device_peers` 保存所有参与设备的实际服务地址，原始 worker JSON
也记录 `peer`。重新汇总时使用保存的映射，不受当前环境变量影响；旧的单服务结果目录仍可汇总。

并发总吞吐使用所有卡的有效字节之和除以最早开始至最晚结束的实际窗口。
窗口重叠率低于 95% 时保留报告并返回失败，需检查日志后再判断竞争。
扫描时各并行度存入 `cores_N/` 或 `workers_N/`，顶层汇总全部配置。

重新生成已有目录的汇总，无需设备：

```bash
bash tests/run_remote_cpu_batch_get_multicard_perf.sh build-npu --summarize reports/某次测试
```
