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
默认尾部门槛为 0，实际参与数为 `min(配置 Jetty 数, ceil(R / 128))`，R=0 时为 0。例如：

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

## 小尾部复用：可调的 Jetty 启用门槛

新增构建参数 `--batch-get-min-tail-reads N`，对应 CMake
`KVAF_BATCH_GET_MIN_TAIL_READS`，范围 0～128。新构建默认 0，省略参数保留
CMake 缓存；0 关闭尾部复用，1 与原分配等价。该门槛只在 lean 开启、整批能放入
在途预算（或预算为 0）时生效，依据每核实际 R，单位为 READ 数。

令 `P=ceil(R/128)`、`T=R%128`、配置数为 J。当
`1 < P <= J` 且 `0 < T < N` 时，只启用 P-1 个 Jetty，最后一个 post
按轮转复用 Jetty 0；其他情况保持默认分配。T 恰好达到门槛时启用新 Jetty。
如果 post 数已超过配置数，尾部原本就在复用已有 Jetty，无须再减少活跃数。

门槛为 32、配置 4 个 Jetty 的例子：

| 每核 READ 数 | 默认活跃数 | 新活跃数 | 新分配 | post 数 | CQ checkpoint 数 |
| ---: | ---: | ---: | --- | ---: | ---: |
| 128 | 1 | 1 | 128 | 1 | 1 |
| 130 | 2 | 1 | 128+2，均发到 Jetty0 | 2 | 1 |
| 159 | 2 | 1 | 128+31，均发到 Jetty0 | 2 | 1 |
| 160 | 2 | 2 | 128 / 32 | 2 | 2 |
| 255 | 2 | 2 | 128 / 127 | 2 | 2 |
| 256 | 2 | 2 | 128 / 128 | 2 | 2 |
| 257 | 3 | 2 | 129 / 128 | 3 | 2 |
| 288 | 3 | 3 | 128 / 128 / 32 | 3 | 3 |
| 320 | 3 | 3 | 128 / 128 / 64 | 3 | 3 |

每个 post 仍最多 128 READ；复用队列只在其最后一个 post 产生完成 checkpoint。
130 READ 仍提交两次 doorbell，不会合成一个 130-READ post。收益来源是少管理
一条队列并少回收一个 checkpoint，也避免极短尾部独占一条队列。整除 128 的场景、
配置资源表步长和多 group 预算窗口不受此参数影响。

现有单轮数据中，每核 129～130 READ 使用 1 Jetty 更好，160 READ 使用 2 Jetty
已有 P50 收益，257 READ 多开第三条队列有损失，而 320 READ 使用 3 Jetty 有收益。
因此 32 是供 A/B 验证的候选门槛，尚未证明是跨核数、token_bytes 和 P99 的最优值。
这次修改不宣称已测得收益，也不将候选门槛设为默认值。

### 只验证 Jetty 启用策略

两侧都保留相同 lean 管理和单 group 路径，避免混入其他优化。构建目录名称中
baseline 表示门槛 0 的对照组，lean 表示门槛 32 的候选组：

```bash
./build.sh npu --build-dir "$PWD/build-npu-tail-baseline" \
  --batch-get-lean-schedule 1 --batch-get-min-tail-reads 0
./build.sh npu --build-dir "$PWD/build-npu-tail-lean" \
  --batch-get-lean-schedule 1 --batch-get-min-tail-reads 32

export KVAF_CPU_SERVICE_ADDR=127.0.0.1:29991
export DEVICE_ID=7
log_dir=logs/batch-get-tail-32
mkdir -p "$log_dir"
for round in 1 2 3 4 5 6; do
  variants=(baseline lean)
  if ((round % 2 == 0)); then variants=(lean baseline); fi
  for core in 1 2; do
    for reads in 128 129 130 144 152 159 160 190 255 256 257 272 287 288 320 380 512; do
      n=$((reads * core))
      for jetty in 1 2 3 4; do
        for variant in "${variants[@]}"; do
          bash tests/run_remote_cpu_batch_get_perf.sh "$PWD/build-npu-tail-$variant" \
            --request-length 32768 --table-capacity 16384 \
            --tokens-per-block 128 --token-bytes 656 --miss-count "$n" \
            --aiv-cores "$core" --jettys-per-core "$jetty" \
            --max-inflight-total 2048 \
            --source-access-pattern random --output-access-pattern contiguous \
            --no-sort-input-indices --seed 20260903 \
            --warmup 100 --iterations 2000 --kernel-profile \
            > "$log_dir/r${round}_n${n}_c${core}_j${jetty}_${variant}.log" 2>&1 || exit 1
        done
      done
    done
  done
done
python3 tests/diagnostics/summarize_batch_get.py "$log_dir" --reference-lean
```

这是每核 READ 数扫描，N=reads×core；另可补 N=257、C=2 验证 128/129
不均匀切分。更换门槛时使用不同日志目录。serial profile 主测 P50/P99；
时间线诊断和关闭 profile 的端到端验证各用独立目录。

`--reference-lean` 要求 baseline 和 lean 标签均实际开启 lean；未传该选项时，
仍要求 baseline=lean OFF、lean=lean ON。工具逐侧校验单 group 实际 READ/post/CQE
分配，允许门槛引起的活跃数和 CQ 数变化；多窗口仍要求两侧分配相同。

新增证明字段：

- `KERNEL_PROFILE_SCHEDULE min_tail_reads=32 tail_jetty_reuse_samples=2000`：
  该核本轮 2000 个样本均避免为短尾部启用新 Jetty；0 表示未触发。
- CSV `min_tail_reads`：实际生效门槛；lean OFF 记为 0。旧日志缺少字段时留空。
- CSV `tail_jetty_reuse_calls`：各核触发次数之和除以迭代数；可能大于 1。
- 逐 Jetty 日志/CSV 中，R=130 应出现 Jetty0 READ=130、post=2、CQ=1，
  其他 Jetty READ/post/CQ=0。R=257、配置 4 时应为 129/128/0/0、CQ 总数 2。

六张 CSV 文件名保持不变，汇总按两侧门槛分组。Jetty scaling 表的参考始终是
同轮同核、baseline 标签的配置 1 Jetty，具体 lean 开关和门槛同时保存在参考列中。
`p50_GBps` 使用内核 total P50，完整调用延迟另看 wall 指标。

profile 扩展复用 V3 保留字，设备记录大小保持不变；使用新增字段时应整体重建
host、binding 和 kernel。CPU 模型增加 32/128 门槛目标，覆盖尾部复用、实际 CQ 数、
SQ 回绕、资源增长/收缩、serial/RR、profile 关闭和出错禁用；CSV 用例覆盖新旧日志
及误分配拒绝。测试代码已补充，需由使用者在相应环境执行。
