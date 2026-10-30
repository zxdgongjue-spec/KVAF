# BatchGet 快路径隔离 A/B

本轮定位无闲置 Jetty 时的退化，保持 CQ 轮询、WQE 构造、128 READ/post 上限和
Jetty 分配算法相同。新增两个独立构建参数：

- `--batch-get-single-group-fast-path 0|1|auto`：绕过通用 group 调度器。
- `--batch-get-single-post-fast-path 0|1|auto`：单 post 时走已有的简化提交路径。

对应 CMake 缓存为 `KVAF_BATCH_GET_SINGLE_GROUP_FAST_PATH` 和
`KVAF_BATCH_GET_SINGLE_POST_FAST_PATH`。新构建默认 AUTO，跟随 lean；显式 0/1
可独立设置，省略参数保留缓存。本次未改变默认启用行为。

| 构建后缀 | 裁剪（lean） | 单 group | 单 post | 尾部门槛 |
| --- | ---: | ---: | ---: | ---: |
| trim，对照 | 1 | 0 | 0 | 0 |
| group | 1 | 1 | 0 | 0 |
| post | 1 | 0 | 1 | 0 |
| both | 1 | 1 | 1 | 0 |

主测配置没有闲置资源，因此实际管理宽度相同。所有 miss 数不超过预算 2048，每核
只有一个 group。多 post 配置也运行 post 版本：此时开关为 1、实际触发应为 0，
有助于区分执行快路径的影响和编译后布局的影响。

## 完整构建与运行命令

在 NPU 环境的仓库根目录执行，完整重建匹配的 host、binding 和 kernel。
构建会生成测试目标，不运行测试。

```bash
for variant in trim group post both; do
  group_fast=0
  post_fast=0
  if [[ "$variant" == group || "$variant" == both ]]; then group_fast=1; fi
  if [[ "$variant" == post || "$variant" == both ]]; then post_fast=1; fi
  bash build.sh npu --build-dir "$PWD/build-npu-fast-$variant" \
    --batch-get-lean-schedule 1 --batch-get-min-tail-reads 0 \
    --batch-get-single-group-fast-path "$group_fast" \
    --batch-get-single-post-fast-path "$post_fast" || exit 1
done

export KVAF_CPU_SERVICE_ADDR=127.0.0.1:29991
export DEVICE_ID=7
export KVAF_FAST_PATH_ROUNDS=6
export KVAF_FAST_PATH_WARMUP=100
export KVAF_FAST_PATH_ITERATIONS=2000
export KVAF_FAST_PATH_CLOCK_MHZ=1000
bash tests/run_remote_cpu_batch_get_fast_path_ab.sh "$PWD" \
  "$PWD/logs/fast-path-ab-$(date +%Y%m%d-%H%M%S)"
```

时钟参数应使用设备实际 GetSystemCycle 频率，上述 1000 与此前测试设置一致。
runner 不启动或修改 CPU 服务、不编译、不自动重试失败测试；目录已含日志时拒绝覆盖。
三个对比分别保存到 `group/`、`post/`、`both/`。每个目录的 baseline 标签表示 trim
构建，lean 标签表示该候选。每轮交换两侧顺序，各对比独立采集自己的对照。

覆盖 `(miss_count, 核数, Jetty/核)`：

```text
(32,1,1) (64,1,1) (128,1,1)
(256,1,2) (512,1,2) (512,1,4) (1024,1,4) (1024,1,8)
(64,2,1) (128,2,1) (256,2,1) (512,2,2) (1024,2,2)
```

开始前检查 CMake 缓存是否显式匹配四种组合。首对日志采集后，先核验设备实际输出的
开关、触发次数、无闲置及 READ/post/CQE 分配，再继续扫描。每个对比生成六张标准
CSV，位于其 `csv/` 下。手工重新导出示例：

```bash
python3 tests/diagnostics/summarize_batch_get.py logs/你的运行目录/group \
  --fast-path-ab --candidate-fast-paths group --output-dir logs/你的运行目录/group/csv
```

`--fast-path-ab` 要求两侧 lean=1、尾部门槛=0、管理数和活跃数均等于配置数，对照的
两个快路径均关闭；候选至少开启一个且整套实验组合一致。
`--candidate-fast-paths group|post|both` 核对候选标签，`--validate-only` 只校验。
不要对总日志根目录导出，三个对比各有独立配对。

## 归因顺序与证明字段

先看各目录的 `batch_get_summary.csv`，再看 `batch_get_pairs.csv` 的逐轮结果，
最后用 `batch_get_jettys.csv` 核对分配。比较各候选相对自己的对照，避免直接比较
不同目录中候选的绝对时间。

- 只有 group 稳定退化：排查单 group 分支及其编译产物。
- 只有 post 在单 post 配置稳定退化：排查单 post 分支及其编译产物。
- 单独开启无退化、both 才退化：排查组合影响。
- post 开启但实际触发为 0 时仍退化：先排除波动，再检查编译布局差异。
- first doorbell 基本不变、transfer window 大幅波动：主要变化在发送之后，不能
  直接记为调度器管理开销，需要重复运行和关闭 profile 的验证。

`KERNEL_PROFILE_SCHEDULE` 新字段示例：

```text
schedule_metadata_version=2
single_group_fast_path_enabled=0
single_post_fast_path_enabled=1
single_group_fast_path_samples=0
single_post_fast_path_calls_per_call=1.000000
```

开关表示编译配置，samples / calls 表示实际触发。single-post calls 按每核每次
调用的单 post group 数统计，窗口调度可能多次触发。CSV 的
`single_post_fast_path_calls` 按所有核求和，2 核各触发一次记为 2；单 group 的
calls 同样按核求和。

调度 metadata 从版本 1 扩展到 2，复用已有 64-bit 保留字，设备 ABI 大小和 V3
入口不变。新 host 可解码旧版本；旧日志缺失的独立开关与次数留空。
CPU 模型补充独立组合、尾部、预算窗口、空核、资源步长、回绕和错误禁用用例。

## 完整调用时间

`p50_us` / `p50_GBps` 仍是内核 total，`wall_p50_us` 也是内核跨度。
新增 `host_p50_us` / `host_p99_us` 来自 `operation=batch_get`，包含主机提交至流
同步完成；`host_submit_p50_us`、`host_synchronize_p50_us` 分别对应两段主机计时。
`host_p50_GBps` 使用完整调用的 P50，单位为十进制 GB/s。

新日志额外输出六位小数微秒，CSV 直接读取；旧日志只有三位小数毫秒时留空，不恢复
不存在的精度。host 指标排除请求生成和 profile D2H 读取，但包含内核 profile 写回。

本轮先用 serial profile 定位。关闭 profile 的复核示例使用相同构建：

```bash
for variant in trim group post both; do
  bash tests/run_remote_cpu_batch_get_perf.sh "$PWD/build-npu-fast-$variant" \
    --request-length 32768 --table-capacity 16384 \
    --tokens-per-block 128 --token-bytes 656 --miss-count 128 \
    --aiv-cores 1 --jettys-per-core 1 --max-inflight-total 2048 \
    --source-access-pattern random --output-access-pattern contiguous \
    --no-sort-input-indices --seed 20260903 --warmup 100 --iterations 2000 \
    --no-kernel-profile > "fast-path-no-profile-${variant}.log" 2>&1 || exit 1
done
```

这组日志查看 `operation=batch_get` 的 `p50_us` / `p99_us`，不要混入 profile CSV
导出目录；选出候选后继续多轮交错复测。
