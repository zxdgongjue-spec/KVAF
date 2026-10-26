# BatchGet 总在途请求控制

`batch_get_max_inflight_total` 默认 2048，控制一次 BatchGet 调用中所有 AIV 核
合计的“已提交、尚未确认完成”的 READ 数。它是保守的软件上限，不是硬件实时
在途计数。多个独立 Runtime 的预算会叠加；同一个 Runtime 的共享队列仍须由调用方
串行使用。设为 0 可恢复整批提交基线。

## 提交和回收

- `miss_count <= 总预算`：每核仍提交完整请求区间，只在各参与 Jetty 的最后
  一条 READ 请求 CQE。预算为 0 时所有请求都使用此路径。
- 大请求：每核预算为 `W / C + (core < W % C)`；按两个 FIFO 槽位拆成
  `floor(Wc / 2)`、`ceil(Wc / 2)`，两个槽位交替复用。非零预算必须至少为 `2*C`。
- 先提交两组，确认最老组在所有参与 Jetty 上完成后再补一组；各核独立推进，
  不增加全局屏障。四核、总预算 2048 时，每核至多 512 条，每组 256 条。
- 每个检查点沿用原来的 completion 和 `place_order=2`。每个组的每个参与
  Jetty 都生成一个 CQE；不能用某一个 Jetty 的完成释放整个组。
- 每次 Doorbell 最多 128 条，Jetty 路由跨组轮转；UB 双 bank 使用组内索引。
  小组可能少于 128 条，也可能不能同时用满全部 Jetty，性能分析需结合实际 post 数。
- 每次提交前按该 Jetty 的未回收组与新增组检查 SQ 容量和 CQ 容量。不会因为
  整个 batch 超过 SQ 容量就拒绝一个可以分组执行的请求。
- 最后按组消费全部 CQE，尾组按实际条数计。SQ PI、CQ CI 跨组和 replay 连续，
  `completed_batch_count` 仍按成功的整次调用计数。
- CQ 错误、超时或提交中途容量错误会停止补发。若仍有未回收的组，相关核的
  队列被标记为不可复用；之后的调用/replay 不再在这些队列发请求。必须同步、
  关闭并重建 transport/Runtime 后恢复，不能只清零软件索引。错误仍沿用现有
  device status/诊断通道，不新增每次调用的 Host 同步。

## 配置与 ABI

```python
runtime = kvaf.NpuRuntime(
    kernel_path,
    batch_get_token_bytes=656,
    batch_get_tokens_per_block=128,
    npu_core_count=4,
    jettys_per_core=4,
    batch_get_max_inflight_total=2048,  # 0: baseline
)
```

C++ 使用 `NpuRuntimeOptions` / `AccessOperatorConfig` 的同名字段。底层直接
构造 `KvafExecuteBatchGetKernelArgs` 的调用方须显式填写该字段，零初始化表示基线。
Kernel 参数仍为 112 字节，预算字段占用原尾部 padding，queue 大小保持不变。
新增每 Jetty 测量后，profile 为 5760 字节、每核 status 为 5824 字节。
Kernel 符号为 `KvafExecuteBatchGetAivKernelV3`，旧
Host/Kernel 混用会在加载符号时失败。必须整体重编 Host、Python binding 和 Kernel。
预算按值 capture，改变预算需重建 Runtime/graph；`miss_count` 内容仍可动态更新。

性能入口接受 `--max-inflight-total`，环境变量为
`KVAF_BATCH_GET_MAX_INFLIGHT_TOTAL`，默认 2048。原 miss sweep 脚本会透传 CLI
参数。比较预算时固定核数、Jetty 数、请求分布、token 大小和 EID/UB 设备。

```bash
tests/run_remote_cpu_batch_get_perf.sh "$PWD/build-npu" \
  --request-length 8193 --miss-count 8193 --aiv-cores 4 --jettys-per-core 4 \
  --max-inflight-total 2048 --kernel-profile
# 同一配置再运行 --max-inflight-total 0；另测 miss-count 2047/2048/2049。
```

## 统计

Profile 增加 `submitted_group_count`、`completed_group_count`、`completion_count`、
`post_active_cycles`、`drain_wait_cycles`、`peak_submitted_not_drained`。
每 Jetty 的 probe 活跃时间和 poll 次数也累计，poll 累计使用 64 位。
Profiling 模式轮询全部在途 Jetty 和两组检查点，只有最老组全部完成后才回收额度；
未开启 profiling 时仍逐 Jetty 阻塞等待。每 Jetty 的实际请求量、提交及观察完成
时间通过新增统计表和独立 CSV 输出，详见
[每 Jetty 测量说明](../access_operator/demo/aiv_batch_get/README.md#per-jetty-measurements-and-csv)。

`kernel_post` 使用累计提交时间，`kernel_completion_wait` 包含组间 probe 和 drain 时间。
`post_done_cycle` 仍表示最后一组提交结束时刻，不能再用它与 setup 的差值代表纯
提交时间。各核峰值相加只是一种保守上界，并非同一时刻的全局硬件峰值。新增
CQE、检查点和标量调度都有成本，性能改善需在 A5 上验证。

## 验证

本地 CTest 增加 `kvaf_batch_get_schedule_test`、`kvaf_batch_get_kernel_model_test`、
`kvaf_batch_get_inflight_config_python` 和 `kvaf_batch_get_jetty_profile_python`，无需 CANN。
调度测试直接使用 Kernel 的
分组策略，检查独立推进的多核总预算、奇数预算、尾组、64 Jetty 轮转和边界计数；
Kernel CPU 模型直接包含实际 `.asc` 源码，以同步 UB copy/SIMT 和延迟生成 CQE
执行控制流，验证完整 payload、SQ/CQ owner 环回、索引持久化、尾组、多 Jetty
及超时/CQ 错误后的复用拒绝。模型使用较小 CQ 深度加速环回；它不能验证真实
AIV pipeline、DMA 可见性、硬件保序或 CANN 编译兼容性。

在 A5 上整体重编后，开启现有 eager/graph 集成测试的边界扫描：

```bash
./build.sh npu
KVAF_CPU_SERVICE_ADDR=CPU_IP:28999 \
KVAF_NPU_TABLE_CAPACITY=8193 KVAF_NPU_REQUESTS=8193 KVAF_NPU_OUTPUT_ROWS=8193 \
KVAF_NPU_TOKEN_BYTES=656 KVAF_NPU_TOKENS_PER_BLOCK=1 \
KVAF_NPU_CORE_COUNT=1 KVAF_NPU_JETTYS_PER_CORE=16 \
KVAF_NPU_GRAPH_REPLAYS=2 KVAF_BATCH_GET_INFLIGHT_CHECKS=1 \
KVAF_BATCH_GET_MAX_INFLIGHT_TOTAL=2048 \
tests/run_remote_cpu_e2e.sh "$PWD/build-npu"
```

边界扫描在相同 SQ/CQ 上交替执行 0、小请求、2048/2049、尾组及大请求，并在
同一 captured graph 中改变 miss_count，检查完整 payload、未写入行和 profile
预算/组回收。继续覆盖以下配置：

- 核/Jetty/预算为 `4/4/2048`、`3/3/2049`：多核及预算余数。
- `1/64/2048`：跨组使用后续 Jetty。
- `1/1/2`：8193 请求的多轮边界扫描推动 SQ 和 CQ owner 环回。
- `1/16/0`：关闭限流，对照数据及性能。

CQ 错误和超时需目标环境故障注入，检查停止补发及下一次 replay 拒绝复用；
CPU 模型覆盖了错误路径，但不能替代硬件检查。当前基线中被注释的请求地址校验维持原状。
