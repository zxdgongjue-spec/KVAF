# BatchPut 分段计时与有效带宽

默认关闭 profiling。重新构建 NPU Host、Python binding 和 BatchPut kernel 后，
`test_npu_batch_put.py` 支持以下环境变量：

| 变量 | 含义 |
| --- | --- |
| `KVAF_BATCH_PUT_PROFILE=1` | 启用 Host 分段计时和 AIV cycle 打点 |
| `KVAF_BATCH_PUT_PROFILE_CLOCK_MHZ` | 经平台确认的 `GetSystemCycle` 计数器频率，必须提供 |
| `KVAF_BATCH_PUT_PROFILE_JSONL` | 逐次记录路径，默认 `batch_put_profile.jsonl`，每次运行覆盖 |

计数器频率不等于 NPU 核心频率。[CANN 9.1 的 GetSystemCycle 文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/tool_interface/system_resources_and_variables/GetSystemCycle_ISASI.md)
规定 Ascend 950PR/950DT 使用 **1000 MHz**，Atlas A2/A3 使用 **50 MHz**。
必须按实际型号设置，不能跨平台套用同一数值。
多 core 的全局时间窗口还要求 `GetSystemCycle` 计数器在各 core 间共享时基。

在已有 CANN 环境、CPU service、kernel 路径和 `PYTHONPATH` 配置下，
Ascend 950PR/950DT 示例为：

```bash
bash build.sh npu
KVAF_RUN_NPU_TESTS=1 KVAF_NPU_REQUESTS=1024 \
KVAF_NPU_CORE_COUNT=32 KVAF_NPU_JETTYS_PER_CORE=2 \
KVAF_BATCH_PUT_PROFILE=1 \
KVAF_BATCH_PUT_PROFILE_CLOCK_MHZ=1000 \
KVAF_BATCH_PUT_PROFILE_JSONL=/tmp/batch_put_profile.jsonl \
  python3 tests/python/test_npu_batch_put.py
```

打点在预热后启用；读取快照、JSONL 写入、统计输出和 BatchFree 均在每次计时窗口外。
打点本身仍有开销，因此带 `profiling=1` 的结果应与关闭打点的基线分别记录。
关闭时不读取 Host 时钟或 AIV cycle，不额外增加 ACL copy/synchronize。
状态仍为每 core 384 字节，Kernel 请求与 launch envelope 大小保持不变；必须将 Host
与 kernel 一起重建，旧 kernel 的空白 profile 会被明确拒绝。

## Host 阶段

`PERF_RESULT operation=batch_put_host_*` 输出均值和 P50/P90/P99，单位 ms：

- `binding_metadata`：Python metadata 转换、Tensor 检查和 stream 解析。
- `request_validation`、`prepare_put`、`commit_put`：Runtime 校验和元数据事务。
- `operator_validation`、`metadata_reserve`、`metadata_pack`：Operator 校验、扩容和打包。
- `local_buffer_acquire`、`arguments`、`local_buffer_release`：MR 缓存获取、参数准备和释放引用。
- `metadata_h2d_enqueue`、`status_seed_enqueue`、`kernel_launch_enqueue`、
  `status_d2h_enqueue`：Host 提交耗时，不能解释为设备 DMA 耗时。
- `sync_validation`、`sync_lock_wait`、`stream_wait`、`status_check`：执行参数检查、
  launch 锁等待、最终 stream 等待及完成状态检查。

`binding_total`、`runtime_total`、`operator_total`、`execute_sync` 标记为
`scope=aggregate`，已经包含子阶段，不能再与子阶段相加。`stream_wait` 是提交完成后的
剩余等待，包括元数据 H2D、Kernel、状态 D2H 和此前同 stream 未完成的任务，不是纯通信时间。

## AIV 阶段与带宽

每个 core 记录开始、初始化完成、首个 doorbell、提交完成、CQ 完成和结束的 cycle，
以及 active request、post 和 Jetty 数量。逐次原始值保存在 JSONL 的 `profile.cores` 中。

| 输出 | 时间范围 |
| --- | --- |
| `batch_put_kernel_wall` | 所有 core 最早开始到最晚结束 |
| `batch_put_communication_window` | 有请求的 core 中，最早 doorbell 到最后 CQ 完成 |
| `batch_put_kernel_setup_max_core` | 各 core 初始化耗时的最大值 |
| `batch_put_kernel_post_max_core` | 各 core 提交阶段耗时的最大值 |
| `batch_put_kernel_cq_wait_max_core` | 各 core 提交完到 CQ 完成耗时的最大值 |

`max_core` 各行可能来自不同 core，不能相加得到 Kernel wall time。通信窗口包含
WQE 后续提交和 CQ 观察开销，并与传输重叠，表示有效通信带宽，不是物理链路线速。

有效 payload 使用 `sum(lengths)`，不含输入 buffer padding、DDR slot padding、元数据或状态。
每次记录保存端到端、Kernel wall 和通信窗口三个口径的有效 GB/s：

```text
duration_ms = cycles / counter_clock_mhz / 1000
useful_GBps = payload_bytes / duration_ms / 1e6
```

汇总带宽按总 payload / 总时间计算，避免逐次带宽平均产生偏差。
JSONL 同时记录 Host ns、原始 cycle、计数器频率、三种有效带宽和各次序号。

Python API 为 `start_batch_put_profile()`、`batch_put_profile()`、
`stop_batch_put_profile()`。仅成功且完整提交的 BatchPut 才能读取快照；失败不会复用旧数据。
