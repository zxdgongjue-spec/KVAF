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
  --graph-api torch --graphs 8 --graph-replays 5 --warmup 10 --seconds 10
```

设备号由 `--devices` 指定；每卡启动一个进程。脚本先逐卡测独立带宽，再同时启动各卡
测并发带宽。所有阶段使用已有 CPU service，脚本不会启动或重启服务。
每卡单独初始化 DDR 数据区；测试的是这些访问对共享 CPU、内存与通信资源的竞争。

`--graphs N` 控制每卡同时保留的独立图数，默认 8。每张图捕获一个 Get，输入、输出和
图租约各自独立，复用同一张卡的 AccessTable 和通信资源。每轮在同一 stream 上依次
提交全部 N 张图，再统一同步一次；多张卡仍由各自进程并发运行。
例如 `--graphs 8` 每轮执行 8 次 Get，`--graphs 1` 使用原来的逐次回放方式。
`--warmup 10` 表示预热 10 轮，每轮也回放全部图。

测量后逐图校验完整输出。`calls = batches × graphs`，有效字节数为
`calls × miss_count × token_bytes`；只统计已同步完成的整轮，测量时间使用实际起止时间。
多图会增加 NPU 上的输入、输出及图资源占用，CPU DDR 数据区每卡仍只分配一份。

## 四种后端使用相同负载

| backend | 并行度参数 | 通信参数 |
|---|---|---|
| `aiv` | `--aiv-cores 2` | `--jettys-per-core 2 --max-inflight-total 2048` |
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
    aiv) resources=(--aiv-cores 2 --jettys-per-core 2 --max-inflight-total 2048) ;;
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
    --graph-api torch --graphs 8 --graph-replays 5 --warmup 10 --seconds 10 \
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
| `run_remote_cpu_batch_get_perf.sh` | eager/graph 逐次采样，输出延迟分位数和带宽 | 默认关闭；`--kernel-profile` 开启设备采集、回读、kernel 指标及详细输出 |
| `run_remote_cpu_batch_get_multicard_perf.sh` | 多张图成组回放、每组同步一次，按时间窗口计算独立/并发吞吐 | 关闭设备 profiling，不在循环内回读 profile |

多卡控制器调用单卡入口的持续回放 worker；两者复用负载准备、建表和数据校验。
公共参数/校验在 `tests/python/utils/get_options.py`，图输入更新、捕获与租约释放在
`tests/python/utils/get_graph.py`。多卡协调 worker 位于 `utils/multicard.py`，
控制器只负责设备进程、起跑屏障、测量窗口、扫描和报告。

两个入口的图模式都在测量前验证动态输入、部分请求和零请求，多卡会逐图验证。
进入测量后固定每张图的请求，
循环中只有整组 replay、一次完成等待和中止检查。
测后执行完整输出校验，Gather/Scatter 还检查设备完成状态。
多卡测试每张图只捕获一个 Get 节点，`--graph-calls` 保持为 1，图数使用 `--graphs` 设置。

单卡默认 Torch Graph，多卡保留原 ACL Graph 默认值；对照时可显式统一 `--graph-api`。
持续吞吐包含回放、等待及循环控制成本，与单卡 `scope=kernel` 的带宽口径不同。

## 查看帮助和结果

```bash
bash tests/run_remote_cpu_batch_get_multicard_perf.sh build-npu --backend gather_read --help
bash tests/run_remote_cpu_batch_get_perf.sh build-npu --backend gather_read --help
```

选定 backend 后，帮助只显示适用的资源参数；查看帮助不要求构建产物或 NPU。

结果目录包含 `config.json`、各阶段/各设备日志与原始 JSON，以及：

- `summary.csv`：各卡图数、独立/并发带宽、保留比例和下降比例。
- `aggregate.csv`、`aggregate.json`：总吞吐、窗口重叠率与起跑偏差。
- `report.md`：配置和结果表；AICPU 单独标记 worker 数。

并发总吞吐使用所有卡的有效字节之和除以最早开始至最晚结束的实际窗口。
窗口重叠率低于 95% 时保留报告并返回失败，需检查日志后再判断竞争。
扫描时各并行度存入 `cores_N/` 或 `workers_N/`，顶层汇总全部配置。

重新生成已有目录的汇总，无需设备：

```bash
bash tests/run_remote_cpu_batch_get_multicard_perf.sh build-npu --summarize reports/某次测试
```
