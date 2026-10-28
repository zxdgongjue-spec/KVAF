# BatchGet 小 post 多 Jetty 实验

本实验只改变少量请求时的 post 粒度，验证增加参与 Jetty 的收益能否覆盖
额外 WQE 提交、doorbell 和 CQE 回收成本。默认关闭。

## 策略

构建参数 `KVAF_BATCH_GET_SMALL_POST_MAX_REQUESTS` 为**整次 BatchGet 的有效
READ 数阈值**，不是每核阈值，也不是每个 post 的条数。值为 0 时保留
128 条/post 的原有调度。

非零且 `miss_count <= 阈值`、每核 Jetty 数大于 1 时：

```text
core_requests = 本核的有效 READ 数
post_width = min(128, ceil(core_requests / jettys_per_core))
```

无请求核仍使用原来的空请求路径。每个 post 仅提交实际条数，不添加填充
READ；极小或不能整除的请求区间可能无法用满全部 Jetty。

例如阈值 256、2 核、每核 2 Jetty：

| miss_count | 每核请求数 | 基线每核提交 | 实验每核提交 |
| ---: | ---: | --- | --- |
| 32 | 16 | Jetty0: 16 | Jetty0: 8, Jetty1: 8 |
| 64 | 32 | Jetty0: 32 | Jetty0: 16, Jetty1: 16 |
| 128 | 64 | Jetty0: 64 | Jetty0: 32, Jetty1: 32 |
| 256 | 128 | Jetty0: 128 | Jetty0: 64, Jetty1: 64 |
| 257 | 128 / 129 | 原有 128 条/post 调度 | 回到相同调度 |

UB 双 bank 容量、128 线程 SIMT 调用、112 字节 Kernel 参数、公开 Runtime /
Python 接口和在途预算均保持不变。post 数、请求偏移、SQ/CQ 容量检查、
跨组 Jetty 轮转、尾部 CQE 回收和 profile 统一使用实际 `post_width`。

阈值在构建时固定；每次 Kernel 根据设备上的 `miss_count` 动态选择策略，
因此同一 captured graph 可以在阈值两侧切换。它不是运行时环境变量或
benchmark 的 CLI 选项。

## 构建

在已配置匹配 CANN 的 NPU Linux 环境中，从仓库根目录分别构建：

```bash
./build.sh npu --build-dir "$PWD/build-npu-baseline" \
  --batch-get-small-post-max-requests 0
./build.sh npu --build-dir "$PWD/build-npu-small-post" \
  --batch-get-small-post-max-requests 256
```

也可直接向 CMake 传入
`-DKVAF_BATCH_GET_SMALL_POST_MAX_REQUESTS=256`。不传参数时，新构建默认
为 0；已有构建保留 CMake cache 中的值。关闭实验须显式设置为 0 并重编。

保持两个构建的编译器、CANN、优化级别及其他选项一致。每次运行使用对应
构建目录下配套的 Kernel 和 Python binding。

## 性能对照示例

以下命令供目标 NPU 环境执行。固定 CPU 服务、NPU EID、UB 设备、表容量、
访问分布、在途预算和随机种子。示例表容量须与要复现的基线保持一致。
若使用自动选路，应从日志确认两种构建使用同一路径。

```bash
export KVAF_CPU_SERVICE_ADDR=CPU_IP:28999
mkdir -p logs/small-post

for round in 1 2 3; do
  variants=(baseline small-post)
  if ((round % 2 == 0)); then
    variants=(small-post baseline)
  fi
  for n in 32 64 128 255 256 257 512 1024; do
    for variant in "${variants[@]}"; do
      bash tests/run_remote_cpu_batch_get_perf.sh "$PWD/build-npu-$variant" \
        --request-length 32768 --table-capacity 16384 \
        --tokens-per-block 128 --token-bytes 656 \
        --miss-count "$n" --aiv-cores 2 --jettys-per-core 2 \
        --max-inflight-total 2048 \
        --source-access-pattern random --output-access-pattern contiguous \
        --no-sort-input-indices --seed 20260903 \
        --warmup 20 --iterations 1000 --kernel-profile \
        > "logs/small-post/r${round}_n${n}_${variant}.log" 2>&1 || exit 1
    done
  done
done

python3 tests/diagnostics/summarize_small_post.py logs/small-post
```

另测 `token_bytes=1024`、每核 Jetty 数为 4，分别使用新的日志目录。
每核 Jetty 数为 1 时应作为策略不变的对照。

先核对 `KERNEL_PROFILE_CORE`：32 条请求、2 核 × 2 Jetty 时，每核
`total_posts_mean` 和 `completion_count_mean` 应从基线的 1 变成 2，
`active_requests_mean` 仍为 16。大于 256 条的调用回到原有调度。

主要比较 `batch_get_kernel_total`、`batch_get_kernel_wall` 的 P50/P99，
同时观察 `kernel_post`、`kernel_completion_wait` 和各 Jetty polls。
`kernel_total` 是最大单核历时；`kernel_wall` 是最早核开始到最晚核结束，
需要跨核公共时钟。换算微秒时使用原始 cycles 并确认设备时钟频率。
阶段统计不能直接相加减；传输会与 posting 重叠。

选出配置后可关闭 profile，另测端到端耗时，以评估采样本身的影响。

## CSV 汇总

已有上述循环生成的日志时，无需重跑测试，只需执行：

```bash
python3 tests/diagnostics/summarize_small_post.py logs/small-post
```

脚本只使用 Python 标准库，不需要 CANN、NPU 或 PyTorch。默认在日志目录
生成以下文件，也可通过 `--output-dir reports/small-post` 指定输出目录：

| 文件 | 每行含义 |
| --- | --- |
| `small_post_runs.csv` | 一次 benchmark：轮次、版本、请求配置、时间、带宽、post/CQE 数及日志路径 |
| `small_post_pairs.csv` | 同轮、同请求形状的 baseline/small-post 配对结果及提升比例 |
| `small_post_summary.csv` | 相同完整配置下多轮配对结果的中位数汇总，附实际轮数和轮次列表 |

脚本递归查找 `r<轮次>_n<请求数>_baseline.log` 和
`r<轮次>_n<请求数>_small-post.log`。上述 3 轮、8 个请求数的完整实验应得到
48 行运行结果、24 行配对结果；完整配置固定时为 8 行汇总结果。
每核 Jetty 数、token 大小等不同的实验仍建议放在不同日志目录，避免覆盖
相同轮次/请求数的文件名。

各核时间使用日志中的 `*_cycles / clock_mhz` 换算微秒，避免原有三位小数
毫秒输出的舍入误差。`p50_GBps` 使用十进制 GB/s，公式为
`miss_count * token_bytes / p50_us / 1000`，不是日志里的 `useful_GiBps`。
`posts_per_call`、`completions_per_call` 是各核相应均值之和。

配对表中的对比字段：

- `p50_speedup = baseline_p50_us / small_post_p50_us`，大于 1 表示加速。
- `p50_latency_reduction_pct`、`p99_latency_reduction_pct`：延迟降低百分比，正值表示改善。
- `p50_bandwidth_gain_pct`：带宽提升百分比，正值表示改善。
- `post_p50_change_us`、`completion_wait_p50_change_us`：实验减基线的时间差，正值表示该阶段增加开销。

汇总表的 `median_*` 均是各轮对应值的中位数；提升比例先逐轮配对计算，
再取中位数。`median_baseline_p99_us` 等字段不是合并所有样本后重新计算的
P99，不能当作合并样本分位数。

脚本在写 CSV 前检查日志完成标记、采样数、profile、重复运行、缺失配对，
并检查同轮两种版本的访问模式、容量、种子、在途预算、时钟及设备等已记录
配置是否相同。不完整或不一致会报错，不会静默混合结果。
`selected_ub_device` 为空表示日志未记录实际设备；两个空值相同不能证明
两次运行实际走了同一设备。不同轮次的已记录配置不同时，会分组汇总。

CSV 使用 UTF-8 BOM，便于在 Excel 中打开。重复执行会更新这三个 CSV，
原始日志保持不变。

## 本地检查范围

`kvaf_batch_get_schedule_test` 覆盖基线/小 post、奇数尾部、请求数少于
核或 Jetty 数、阈值边界以及与在途预算交互的跨组轮转和容量计算。

`kvaf_batch_get_kernel_model_test` 使用当前构建阈值；新增的
`kvaf_batch_get_small_post_kernel_model_test` 固定阈值为 256。两者直接
执行生产 `.asc` 控制流，检查 payload、实际 doorbell/profile 数、所有
参与 Jetty 的 CQ 回收、SQ/CQ 环回、replay 和错误后的队列禁用。

CPU 模型不验证 AIV 时序、DMA/cache 可见性、真实硬件保序或 CANN 编译，
也不能用于给出性能结论。
