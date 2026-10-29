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

UB 双 bank 容量、128 线程 SIMT 调用、112 字节 Kernel 参数和在途预算均保持不变。
逐 Jetty 诊断扩展了 status/profile；Kernel 使用 V3 符号，必须一起重编 host、
Python binding 和 kernel，不能混用旧构建。post 数、请求偏移、SQ/CQ 容量检查、
跨组 Jetty 轮转、尾部 CQE 回收和 profile 统一使用实际 `post_width`。

第二版对单核、单组不超过 128 条的拆分请求使用低开销发射路径，且要求
每个参与 Jetty 只有一个 post：

1. 标量侧读取各 Jetty 独立的 SQ PI、token、EID 和通道字段，将快照写入
   空闲 UB bank；通过 S → V 事件交接后，SIMT 不再直接读取标量拥有的 GM 队列索引。
2. 用一次 128 线程 SIMT 调用构造整组 WQE，WQE 按各 post 分段保存在另一
   bank 中。owner、completion 及路由字段仍按各 Jetty 的实际上下文填写。
3. 每段 SQ 拷贝完成后立即敲该 Jetty 的 doorbell，然后再拷贝下一段；首个
   doorbell 不再等待第二段 SQ 拷贝。

例如 32 条、2 核 × 2 Jetty：每核仍是两个 8 条 post、两个 CQE，但 SIMT
构造调用由每核两次减为一次。post/CQE 数不会因构造合并而减少。
单组超过 128 条、只有一个 post 或同一组需要重复使用某个 Jetty 时，继续
使用原有双 bank 发射路径；预算检查和 CQ 回收保持一致。

两种发射路径共用 WQE 打包函数，以避免协议位随路径分叉。
该 helper 使用 `__simt_callee__`，遵循
[Ascend C 的 SIMT 函数调用约束](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMT-API/SIMD_SIMT_hybrid_programming_intro/extended_syntax/function_execution_space_qualifier_142.md)。

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
mkdir -p logs/small-post-v3-serial

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
        > "logs/small-post-v3-serial/r${round}_n${n}_${variant}.log" 2>&1 || exit 1
    done
  done
done

python3 tests/diagnostics/summarize_small_post.py logs/small-post-v3-serial
```

另测 `token_bytes=1024`、每核 Jetty 数为 4，分别使用新的日志目录。
每核 Jetty 数为 1 时应作为策略不变的对照。
V3 采用新的日志目录，避免覆盖旧日志或混合不同版本的实验。

先核对 `KERNEL_PROFILE_CORE`：32 条请求、2 核 × 2 Jetty 时，每核
`total_posts_mean` 和 `completion_count_mean` 应从基线的 1 变成 2，
`active_requests_mean` 仍为 16。大于 256 条的调用回到原有调度。

主要比较 `batch_get_kernel_total`、`batch_get_kernel_wall` 的 P50/P99，
同时观察 `kernel_post`、`kernel_completion_wait` 和各 Jetty polls。
新增 `batch_get_kernel_first_doorbell` 为每核 Kernel 开始到首个 doorbell
软件提交记录点的时间，再按各核最大值汇总；无请求核记为 0。它不是硬件
收包或数据开始传输的时间。`batch_get_kernel_transfer_window` 从该记录点
量到所有参与 Jetty 完成回收，仍会与后续提交重叠。
`kernel_total` 是最大单核历时；`kernel_wall` 是最早核开始到最晚核结束，
需要跨核公共时钟。换算微秒时使用原始 cycles 并确认设备时钟频率。
阶段统计不能直接相加减；传输会与 posting 重叠。

选出配置后可关闭 profile，另测端到端耗时，以评估采样本身的影响。
若 512/1024 等回退调度的对照仍出现明显漂移，先用同一 baseline 构建按
两个版本的顺序做 A/A，并核对实际 EID/UB 设备及系统负载，再判断发射策略
收益。1000 次采样不能替代多轮独立运行；均值、P50、P99 应同时观察。

## 验证策略触发与逐 Jetty 完成观测

普通 `--kernel-profile` 新增以下日志，仍采用串行 CQ 等待：

- `KERNEL_PROFILE_SCHEDULE`：由实际运行的 kernel 返回编译上限、选中的
  `post_width_min/max`、`small_post_active_samples`、`small_post_groups_per_call`、
  `fused_groups_per_call` 和 `wqe_build_calls_per_call`。
- `KERNEL_PROFILE_JETTY`：每核、每 Jetty 的实际 `reads_per_call`、
  `posts_per_call`、`completions_per_call`、payload 和软件时间戳统计。
  READ/post 计数在对应 SQ doorbell 提交后更新，不只根据策略公式推算。

例如总 32 READ、2 核 × 2 Jetty、上限 256 时，每核应打印：

```text
KERNEL_PROFILE_SCHEDULE ... small_post_max_requests=256 post_width_min=8 post_width_max=8 small_post_active_samples=1000 small_post_groups_per_call=1.000000 fused_groups_per_call=1.000000 wqe_build_calls_per_call=1.000000
KERNEL_PROFILE_JETTY ... jetty=0 ... reads_per_call=8.000000 posts_per_call=1.000000 completions_per_call=1.000000 ...
KERNEL_PROFILE_JETTY ... jetty=1 ... reads_per_call=8.000000 posts_per_call=1.000000 completions_per_call=1.000000 ...
```

这证明策略触发、请求实际拆到两个 Jetty，并使用一次合并构造。性能收益仍
需通过整次延迟验证。基线应为每核 Jetty0 16 READ、Jetty1 0 READ，
`small_post_groups_per_call=0`、`fused_groups_per_call=0`。
总 1024 READ 时每核 512 READ、128 条/post，两个版本都使用原发射路径，
`fused_groups_per_call=0`；不要用这组日志验证小 post 是否触发。

为了比较 Jetty 的完成观测时间，在原 benchmark 命令末尾同时添加：

```bash
--kernel-profile --kernel-profile-jetty-timeline
```

该开关需要 `--kernel-profile`。在每个 group 回收阶段，它轮流探测所有未完成
Jetty，且轮换每轮的首个探测对象；每个 Jetty 的最后一个 CQE 都回收成功后
才归还该 group 的预算。CQ owner/CI、错误处理及队列禁用规则保持一致。
Python 也可用 `runtime.start_batch_get_kernel_profile(jetty_timeline=True)`。

关注以下以 **us** 为单位、仅对该 Jetty 活跃的样本统计的字段：

| 字段 | 含义 |
| --- | --- |
| `first_doorbell_offset_us_mean` / `last_doorbell_offset_us_mean` | 本核开始到该 Jetty 首次/最后一次 doorbell 的软件记录点 |
| `first_cqe_seen_offset_us_mean` / `last_cqe_seen_offset_us_mean` | 本核开始到该 Jetty 首个/最后一个 checkpoint 的 CQE 首次被软件观察到 |
| `first_db_to_last_cqe_us_mean/p50/p99` | 该 Jetty 本次调用首次发射到最后一个 checkpoint 被观察到 |
| `last_db_to_last_cqe_us_mean/p50/p99` | 该 Jetty 本次调用最后一次发射到最后一个 checkpoint 被观察到 |
| `probe_active_us_mean` | round-robin 模式实际执行该 Jetty CQ 探测/回收函数的累计时间，仍不是 Jetty 运行时间 |

READ/post/CQE 数与 `polls_per_call` 则按全部调用求均值。无请求 Jetty 的时间
打印 `n/a`，CSV 留空。多个 group 时，上述跨度覆盖整次调用，不是每个 group
的平均 RTT；比较小 post 时先用在途预算内的单 group 样本。

时间戳是软件观察值，不是硬件完成时间。探测在 group 提交完后进行，包含
排队、后续 posting、CQ 可见性与轮询采样误差。若 CQE 在开始探测前就已就绪，
无法从这个日志恢复其更早的真实完成时刻。先在同一核内比较 Jetty；跨核
offset 各自以本核开始为零，不能据此宣称硬件完成时刻一致。

`KERNEL_PROFILE_CONTRACT completion_poll_mode=serial/round_robin` 标明模式。
串行模式的 `jetty*_wait_ms_mean` 仍是轮到该 Jetty 后的剩余等待时间，不能
比较执行均衡；round-robin 模式不打印这些字段，使用独立的 probe-active 指标。
额外探测和计时会改变诊断模式的开销，不能把它自动视为性能优化。

两个版本都要使用相同模式与同一版本的构建。新日志分别写到
`logs/small-post-v3-serial`、`logs/small-post-v3-timeline`，不要覆盖或混入旧版日志。
用详细模式确认调度与时序，再关闭详细模式和 profile 评估端到端收益。
最终 profile 写回位于 `kernel_finish_cycle` 之后，不计入 `kernel_total`，
但会计入 host 到 stream 完成的耗时；新 contract 显式标记这一点。
V3 status 为 5376 字节/profile 为 5312 字节，queue state 仍为 64 字节，
Kernel 参数仍为 112 字节；loader 要求 `KvafExecuteBatchGetAivKernelV3`。

## 扫描 AIV 核数与每核 Jetty 数

128 是每个 post 的 READ 上限，分配规律应结合**每核 READ 数**观察，而不是
只看整次 `miss_count`。例如 1 核时 128 READ 是候选边界，2 核时对应总数
256，4 核时对应总数 512。baseline 在单组内按 post 轮转 Jetty；它并不限制
一个 Jetty 只能处理 128 READ。同一 Jetty 可以承接多个 post。

先固定 token 大小、访问分布、在途预算和设备，扫描总 READ 数 × 核数 ×
每核 Jetty 数。既比较固定总 READ 数下的最低延迟，也沿着每核 READ 数
相近的配置观察分配规律；不同核数的结果保留为不同配置，不合并采样。

若要探索 512/1024 的 small-post 分配，实验构建的编译阈值必须覆盖这些值：

```bash
./build.sh npu --build-dir "$PWD/build-npu-baseline" \
  --batch-get-small-post-max-requests 0
./build.sh npu --build-dir "$PWD/build-npu-small-post" \
  --batch-get-small-post-max-requests 1024
```

上限 256 的构建也能完成扫描，但大于 256 时实验版本使用 baseline 调度。
上限 1024 也不保证每种配置都改变分配：1 核、256 READ、2 Jetty 时两者
都是 128 + 128；4 Jetty 的 small post 则是 64 × 4。单核单组超过 128
READ 时，拆分可能生效，但不使用一次 SIMT 合并构造，需要分别判断分配
收益和发射成本。运行表会保留实际上限、post 宽度与合并构造次数。

下面的循环覆盖 1/2/4/8 核和每核 1/2/4/8 Jetty。文件名同时记录核数和
Jetty 数，避免多核扫描时覆盖结果。串行与 round-robin 模式、不同编译
上限的实验应使用不同日志目录。

```bash
export KVAF_CPU_SERVICE_ADDR=127.0.0.1:29991
export DEVICE_ID=7
log_dir=logs/small-post-core-jetty-1024-timeline
mkdir -p "$log_dir"

for round in 1 2 3; do
  variants=(baseline small-post)
  cores=(1 2 4 8)
  jettys=(1 2 4 8)
  if ((round % 2 == 0)); then
    variants=(small-post baseline)
    cores=(8 4 2 1)
    jettys=(8 4 2 1)
  fi
  for n in 128 256 512 1024; do
    for core in "${cores[@]}"; do
      for jetty in "${jettys[@]}"; do
        for variant in "${variants[@]}"; do
          bash tests/run_remote_cpu_batch_get_perf.sh "$PWD/build-npu-$variant" \
            --request-length 32768 --table-capacity 16384 \
            --tokens-per-block 128 --token-bytes 656 \
            --miss-count "$n" --aiv-cores "$core" --jettys-per-core "$jetty" \
            --max-inflight-total 2048 \
            --source-access-pattern random --output-access-pattern contiguous \
            --no-sort-input-indices --seed 20260903 \
            --warmup 20 --iterations 1000 --kernel-profile --kernel-profile-jetty-timeline \
            > "$log_dir/r${round}_n${n}_c${core}_j${jetty}_${variant}.log" 2>&1 || exit 1
        done
      done
    done
  done
done

python3 tests/diagnostics/summarize_small_post.py "$log_dir"
```

这组完整实验有 384 次运行、192 对固定核数/Jetty 数的 A/B 配对，及 384
条相对同轮同核 baseline 1 Jetty 的比较（包含 baseline 1 Jetty 自身作为
零收益参照）。它用于初筛，再对表现接近的候选增加独立轮次，补测 32/64
以及 `128 * aiv_cores` 附近的请求数，例如边界前后一条。对多 Jetty 收益
同时看 P50、均值、P99 和各轮一致性，不仅看跨轮中位数。
补测边界时，编译上限也必须覆盖新增请求数；例如 8 核的 1025 READ
超过了示例上限。还应固定足够大的在途预算，使每核请求仍能容纳于单组，
避免同时跨越 group 容量边界。示例扫描至 1024 时预算 2048 满足该条件；
扩展至 1025 可另用上限/预算更大的独立实验，并重跑相应参照点。

round-robin 日志用于解释实际分配与完成观测。选出候选后，用普通串行
profile 复核，并关闭 profile 确认单次 BatchGet 的 host 端到端延迟。
当前 CSV 需要 kernel profile，不解析关闭 profile 的日志；其中主要延迟
与带宽字段是 kernel total 口径，不能直接当作 host 端到端结果。

## CSV 汇总

已有上述循环生成的日志时，无需重跑测试，只需执行：

```bash
python3 tests/diagnostics/summarize_small_post.py logs/small-post-v3-serial
```

脚本只使用 Python 标准库，不需要 CANN、NPU 或 PyTorch。默认在日志目录
生成以下文件，也可通过 `--output-dir reports/small-post` 指定输出目录：

| 文件 | 每行含义 |
| --- | --- |
| `small_post_runs.csv` | 一次 benchmark：轮次、版本、请求配置、时间、带宽、post/CQE 数及日志路径 |
| `small_post_pairs.csv` | 同轮、同请求形状的 baseline/small-post 配对结果及提升比例 |
| `small_post_summary.csv` | 相同完整配置下多轮配对结果的中位数汇总，附实际轮数和轮次列表 |
| `small_post_jettys.csv` | 每轮、版本、核、Jetty 的实际分配、策略证明和完成观测时间；旧日志仅生成表头 |
| `small_post_jetty_pairs.csv` | 每轮每种核数/Jetty 数/版本，相对同轮同核 baseline 1 Jetty 的比较 |
| `small_post_jetty_summary.csv` | 上述 Jetty 比较的多轮中位数、P50 获益轮数、P99 退化轮数和最差轮收益 |

脚本递归查找日志，兼容三种命名（版本均为 `baseline` 或 `small-post`）：

- `r<轮次>_n<请求数>_<版本>.log`：原有格式。
- `r<轮次>_n<请求数>_<Jetty数>_<版本>.log`：只扫描 Jetty 数时使用。
- `r<轮次>_n<请求数>_c<核数>_j<Jetty数>_<版本>.log`：核数 × Jetty 扫描推荐格式。

核数与 Jetty 数仍以日志中的 `BATCH_GET_PERF_CONFIG` 为准；文件名显式
记录的值必须匹配，否则报错。相同运行的不同文件名不是独立样本，重复
的 round/shape/variant 仍会报错。

上文固定 2 核 × 2 Jetty 的 3 轮、8 个请求数实验应得到
48 行运行结果、24 行配对结果；完整配置固定时为 8 行汇总结果。
核数/Jetty 数可放在同一目录，通过推荐文件名区分。不同 token 大小仍应
使用独立子目录，避免相同文件名被覆盖。

运行/配对/汇总表新增 `reads_per_core_min/max/mean`（逐核 READ 均值的
最小值、最大值和平均值）及 `configured_jettys_total`（核数 × 每核配置
Jetty 数）。例如总数 257、2 核时每核是 128/129 READ，不能只用 128.5
掩盖边界差异。运行表还包含实际 `small_post_max_requests`、
`post_width_min/max` 和 `active_jettys_per_call`；最后一项来自每个 Jetty
的 active samples 合计除以迭代数，旧日志缺失时留空，不用配置数代替。

两个新的 Jetty 比较表统一以同轮、同 token 大小、同总 READ 数、同核数
的 **baseline 1 Jetty/core** 为参照，`reference_*` 是该参照，
`candidate_*` 是当前行 `variant` / `jettys_per_core` 的结果。
因此可以同时比较 baseline 4 Jetty 与 baseline 1 Jetty，以及 small-post
4 Jetty 与 baseline 1 Jetty。原有 A/B 表继续比较固定核数、固定 Jetty 数
的两种构建，两类比较回答的问题不同。

除 Jetty 数外，已记录配置必须一致，包括时钟、实际 UB 设备和 profile
模式，否则在写 CSV 前报错。缺少某轮某核的 baseline 1 Jetty 时，该组
不生成 Jetty 比较，原有 A/B 表仍正常导出，不借用其他核数或其他轮次。
控制台 `SMALL_POST_JETTY_REFERENCE` 报告匹配和缺少参照的运行数。

`small_post_jetty_summary.csv` 中：

- `paired_rounds` / `round_ids`：实际配对轮次。
- `median_p50_latency_reduction_pct`：逐轮相对 baseline 1 Jetty 的延迟收益中位数，正值改善。
- `p50_win_rounds`：P50 严格改善的轮数。
- `p99_regression_rounds`：P99 严格退化的轮数；不设置隐含容差。
- `worst_p50_latency_reduction_pct` / `worst_p99_latency_reduction_pct`：最差一轮的收益，负值表示退化。

对固定 miss_count，先在相同核数下比较 Jetty 数，再比较不同核数的配置。
缺少的轮次不补零；收益比例先逐轮配对计算再取中位数，不用两个中位数
相除代替。不同编译 small-post 上限按独立组汇总，不能把两种策略的轮次
合成同一收益。获益轮数只是描述结果，不自动认定统计显著或生产策略。

各核时间使用日志中的 `*_cycles / clock_mhz` 换算微秒，避免原有三位小数
毫秒输出的舍入误差。`p50_GBps` 使用十进制 GB/s，公式为
`miss_count * token_bytes / p50_us / 1000`，不是日志里的 `useful_GiBps`。
`posts_per_call`、`completions_per_call` 是各核相应均值之和。

配对表中的对比字段：

- `p50_speedup = baseline_p50_us / small_post_p50_us`，大于 1 表示加速。
- `p50_latency_reduction_pct`、`p99_latency_reduction_pct`：延迟降低百分比，正值表示改善。
- `p50_bandwidth_gain_pct`：带宽提升百分比，正值表示改善。
- `post_p50_change_us`、`completion_wait_p50_change_us`：实验减基线的时间差，正值表示该阶段增加开销。
- `first_doorbell_p50_change_us`、`transfer_window_p50_change_us`：实验减基线的首个 doorbell 延迟及传输窗口差值，负值表示缩短。

运行表新增 `first_doorbell_*_us`、`transfer_window_*_us`；配对表分别
带有 `baseline_`、`small_post_` 前缀。旧日志仍可使用同一工具汇总，缺少的
时间字段留空，不会被当作 0。只有双方都具备同一时间指标时，该轮才参与
相应配对与汇总；`first_doorbell_rounds`、`transfer_window_rounds` 标明
各指标实际有效的配对轮数。

另外输出 `setup_*_us`、`finalize_*_us` 及相应差值/有效轮数，便于定位小 batch
固定开销。运行/配对/汇总表还包含实际 small-post/fused group 数及 WQE 构造
调用数。逐 Jetty 表的时间采用日志中的 us，不从已舍入的毫秒字段恢复。
不同 profile 版本、串行和 round-robin 模式不能互相配对；脚本检查逐 Jetty READ/post/CQE 合计
与每核统计一致，并拒绝缺失或重复的 V3 诊断行。

汇总表的 `median_*` 均是各轮对应值的中位数；提升比例先逐轮配对计算，
再取中位数。`median_baseline_p99_us` 等字段不是合并所有样本后重新计算的
P99，不能当作合并样本分位数。

脚本在写 CSV 前检查日志完成标记、采样数、profile、重复运行、缺失配对，
并检查同轮两种版本的访问模式、容量、种子、在途预算、时钟及设备等已记录
配置是否相同。不完整或不一致会报错，不会静默混合结果。
`selected_ub_device` 为空表示日志未记录实际设备；两个空值相同不能证明
两次运行实际走了同一设备。不同轮次的已记录配置不同时，会分组汇总。

CSV 使用 UTF-8 BOM，便于在 Excel 中打开。重复执行会更新这六个 CSV，
原始日志保持不变。

若旧日志出现 `n/a[KVAF_AIV]` 等粘连，通常是 Python stdout 缓冲块与
C++ stderr 经 `2>&1` 合并后交错。脚本会提取完整、以换行结束的
`[KVAF_AIV]` 原生日志，并拼回其前后被打断的 stdout 字段；原生日志中的
UB 设备选择仍保留用于配置校验。恢复成功时输出 `SMALL_POST_LOG_RECOVERED`，
无需修改原日志或重跑 benchmark。未知前缀、损坏数值、丢失字段或无法
确定边界的交错仍会报错，不把它们补成零或 `n/a`。runner 使用 `python3 -u`
减少后续采样日志中的缓冲交错；这不改变请求、调度或内核计时范围。

## 本地检查范围

`kvaf_batch_get_schedule_test` 覆盖基线/小 post、奇数尾部、请求数少于
核或 Jetty 数、阈值边界以及与在途预算交互的跨组轮转和容量计算。

`kvaf_batch_get_kernel_model_test` 使用当前构建阈值；新增的
`kvaf_batch_get_small_post_kernel_model_test` 固定阈值为 256。两者直接
执行生产 `.asc` 控制流，检查 payload、实际 doorbell/profile 数、所有
参与 Jetty 的 CQ 回收、SQ/CQ 环回、replay 和错误后的队列禁用。
第二版另外记录模拟 SIMT 调用与 SQ copy/doorbell 顺序，检查一次 SIMT
构造多个 post、首个 doorbell 早于第二段 SQ 拷贝，并验证不同 Jetty 的
TP/token/EID、独立 SQ head 和跨 owner 边界的 WQE。
逐 Jetty 诊断用例还覆盖实际 SQ 提交量、inactive/replay 清零、延迟 CQ 时对另一
Jetty 的探测、窗口组回收及 round-robin 模式下的超时/错误禁用。

CPU 模型不验证 AIV 时序、DMA/cache 可见性、真实硬件保序或 CANN 编译，
也不能用于给出性能结论。
