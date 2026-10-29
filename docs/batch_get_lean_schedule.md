# BatchGet：baseline post 分配与按需 Jetty 管理

当前固定使用 baseline 的最多 **128 READ/post** 分配。主动拆小 post 的
small-post 策略、fused 小 post 构造和 `--batch-get-small-post-max-requests`
构建参数已移除，不再需要 `post_limit`。

## 为什么移除 small-post

多 Jetty 可能缩短传输窗口，但拆分也增加 SQ 提交、doorbell 和 CQE 回收开销。
只有关键路径上的传输、排队时间节省超过新增成本，总延迟才会下降。这不是每个小 batch
都满足的条件，128 READ/post 也是实现粒度，不是所有设备的物理吞吐阈值。

已有单轮 round-robin profile 数据，token_bytes=656，同核数和配置 Jetty 数比较：

| miss_count | 核数 | 配置 Jetty/核 | baseline P50/us | small-post P50/us | 延迟变化 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 128 | 1 | 2 | 13.512 | 10.806 | 下降 20.0% |
| 256 | 1 | 4 | 15.0485 | 12.9325 | 下降 14.1% |
| 128 | 2 | 2 | 9.929 | 9.9435 | 基本持平 |
| 128 | 2 | 8 | 12.055 | 14.6505 | 增加 21.5% |

这些结果证明部分配置可能受益，未证明跨请求规模、核数和重复运行的稳定收益；
带 profile 的数据也不能直接代表关闭诊断后的生产端到端收益。
因此按“需要稳定收益证据才保留策略”的标准移除，不能将结论解释成多 Jetty
在理论上永远无益。

## 当前分配规则

以设备端当前 miss_count=N、核数=C、核编号=c 为准，本核实际 READ 数为：

```text
R = floor(N * (c + 1) / C) - floor(N * c / C)
```

整批能放进在途预算时，每核只有一个 group，从 Jetty 0 开始按 post 轮转。
实际参与数为 `min(配置 Jetty 数, ceil(R / 128))`，R=0 时为 0。例如：

| 本核 R | 配置 Jetty 数 | 实际参与数 | 分配 |
| ---: | ---: | ---: | --- |
| 32 | 4 | 1 | Jetty0: 32 |
| 128 | 4 | 1 | Jetty0: 128 |
| 129 | 4 | 2 | Jetty0: 128，Jetty1: 1 |
| 256 | 4 | 2 | Jetty0/1 各 128 |
| 512 | 2 | 2 | 每个 Jetty 两个 post，各承担 256 READ |

这里依据实际 R，而不对 N/C 的平均值直接取整。比如 N=257、C=2 时，
两个核分别处理 128 和 129，实际参与数可以不同。

- configured：预创建资源数，始终作为 `[core][jetty]` 表的步长。
- active：本次实际发布 READ 的不同 Jetty 数，由逐 Jetty profile 证明。
- managed：本次检查、初始化和管理的资源数。单 group 使用实际参与前缀；
  多 group 的预算窗口路径保留配置宽度和原有跨组轮转。

需要更多 Jetty 的下一次调用，会在首次使用前检查和初始化相应队列。
提交后的超时或 CQ 错误仍禁用整套配置队列，包括本次未初始化的队列。

整批能放进在途预算（默认 2048），或预算为 0 时，非空核直接提交、回收一个
group，跳过双槽调度器和旧 group 的在途量管理。只有一个 post 时还跳过双 bank
调度循环和全部 SQ head 快照。需要多个窗口 group 时保留原调度器。

## 构建与 A/B

`KVAF_BATCH_GET_LEAN_SCHEDULE` 新构建默认 ON。`build.sh` 不传参数会保留
已有 CMake 缓存；A/B 请明确传 `--batch-get-lean-schedule 0/1`。
旧 small-post CMake 缓存会被清理，内核不再读取它。

```bash
./build.sh npu --build-dir "$PWD/build-npu-baseline" --batch-get-lean-schedule 0
./build.sh npu --build-dir "$PWD/build-npu-lean" --batch-get-lean-schedule 1
```

本轮只比较这两个版本：同样的 baseline post 分配，区别为按需管理和单 group
路径的开关。两侧使用相同核数、配置 Jetty 数、数据、预算和时钟设置。

以下命令由使用者在 NPU 环境运行，构建目录需分别包含对应 host、binding 和
kernel。runner 会从传入目录选择匹配文件。

```bash
export KVAF_CPU_SERVICE_ADDR=127.0.0.1:29991
export DEVICE_ID=7
log_dir=logs/batch-get-lean-serial
mkdir -p "$log_dir"
for round in 1 2 3; do
  variants=(baseline lean)
  if ((round % 2 == 0)); then variants=(lean baseline); fi
  for n in 32 64 128 255 256 257 512 1024; do
    for core in 1 2; do
      for jetty in 1 2 4 8; do
        for variant in "${variants[@]}"; do
          bash tests/run_remote_cpu_batch_get_perf.sh "$PWD/build-npu-$variant" \
            --request-length 32768 --table-capacity 16384 \
            --tokens-per-block 128 --token-bytes 656 --miss-count "$n" \
            --aiv-cores "$core" --jettys-per-core "$jetty" \
            --max-inflight-total 2048 \
            --source-access-pattern random --output-access-pattern contiguous \
            --no-sort-input-indices --seed 20260903 \
            --warmup 20 --iterations 1000 --kernel-profile \
            > "$log_dir/r${round}_n${n}_c${core}_j${jetty}_${variant}.log" 2>&1 || exit 1
        done
      done
    done
  done
done
python3 tests/diagnostics/summarize_batch_get.py "$log_dir"
```

主 A/B 使用 serial profile。完成观测时间线可另开目录加
`--kernel-profile-jetty-timeline`；关闭 profile 的真实端到端验证另开目录使用
`--no-kernel-profile`。CSV 工具分析带 profile 的日志，缺少测量不会补零。

## 日志与 CSV

例如 1 核、配置 8 Jetty、128 READ，实验组应输出：

```text
KERNEL_PROFILE_SCHEDULE ... post_width_min=128 post_width_max=128 schedule_metadata_version=1 lean_schedule_enabled=1 configured_jettys=8 active_jettys_per_call=1.000000 managed_jettys_min=1 managed_jettys_max=1 managed_jettys_per_call=1.000000 single_group_fast_path_samples=1000
```

对照组的 active 同样为 1，managed 为 8，fast_path_samples 为 0。
两侧逐 Jetty READ/post/CQE 数应一致。优先看总延迟和带宽，再用 setup、first
doorbell、post、finalize 定位减少的管理时间。传输与完成等待仍需硬件测量。

`summarize_batch_get.py` 验证版本开关和固定 post 策略后生成六张表，配对列使用
`baseline_`/`lean_` 前缀，移除已退役 small-post 策略字段：

- `batch_get_runs.csv`：每轮每版本的配置、时间、带宽和管理计数。
- `batch_get_pairs.csv`：同轮同配置的 baseline/lean 配对结果。
- `batch_get_summary.csv`：上述配对的多轮中位数。
- `batch_get_jettys.csv`：逐核、逐 Jetty 的 READ/post/CQE 和观测时间。
- `batch_get_jetty_pairs.csv`：相对同轮同核 baseline 1 Jetty 的比较。
- `batch_get_jetty_summary.csv`：上述 Jetty 比较的多轮汇总。

`managed_jettys_per_call` 和 `single_group_fast_path_calls` 按所有核求和。
原 `summarize_small_post.py` 仅保留用于历史 baseline/small-post 日志。
内核 V3 ABI 中的旧 small-post/fused 字段保留位置并写 0，以维持旧日志解析兼容。

CPU 模型分别覆盖 lean OFF/ON、128/129 边界、配置步长、空核、活跃范围变化、
队列回绕、预算窗口、profile 关闭和错误禁用；不替代 CANN 编译或硬件验证。
