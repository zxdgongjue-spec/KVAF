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

每卡捕获一张 ACL 图，默认图内串行执行 256 个 Get。各 Get 使用独立的请求方案，
共用输出缓冲区。只有需要改变图大小时才传 `--graph-calls N`。
预热和测量都每轮回放一次、同步一次。测量前后按最后一个 Get 的请求检查对应输出行。
有效字节数为 `已完成的回放次数 × graph_calls × miss_count × token_bytes`，
带宽按实际测量时间计算。

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
| `run_remote_cpu_batch_get_multicard_perf.sh` | 持续回放一张 ACL 图，每次回放同步一次，按时间窗口计算独立/并发吞吐 | 关闭设备 profiling，不在循环内回读 profile |

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
