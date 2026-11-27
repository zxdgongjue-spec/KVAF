# BatchGet 总在途请求控制

`batch_get_max_inflight_total` 默认 2048，控制一次 BatchGet 调用中所有 AIV 核
合计的“已提交、尚未确认完成”的 READ 数。它是保守的软件上限，不是硬件实时
在途计数。多个独立 Runtime 的预算会叠加；同一个 Runtime 的共享队列仍须由调用方
串行使用。普通 URMA AIV Direct Get 设为 0 时，使用 SQ/CQ 容量窗口，不额外施加
全局软件预算。独立 raw-address Pull 的零预算语义保持原样。

## 提交和回收

- `miss_count <= 总预算`：每核仍提交完整请求区间，只在各参与 Jetty 的最后
  一条 READ 请求 CQE。零预算时，只有各队列能整体容纳的区间才使用此路径。
- 大请求：每核预算为 `W / C + (core < W % C)`；按两个 FIFO 槽位拆成
  `floor(Wc / 2)`、`ceil(Wc / 2)`，两个槽位交替复用。非零预算必须至少为 `2*C`。
- 零预算大请求：每条 SQ 每组容量取 `floor((sq_depth - 1) / (2 * 128)) * 128`，
  使用最小队列容量乘以每核 Jetty 数作为组大小。当前深度 32768 时，每 Jetty 每组
  16256 条。本机默认 3 核、每核 1 个 Jetty，两组的全局保守窗口为 `2 × 3 × 16256 = 97536` 条。
  远端 A+K 默认 4 核、每核 1 个 Jetty，对应窗口为 130048 条。
  每次仍按实际轮转分配核对 SQ 和 CQ 额度，空间不足先回收最老组再重试。
  CQ 只能容纳一组时自动串行提交/回收；不会因整个批次超过 SQ 容量而直接拒绝。
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
    batch_get_max_inflight_total=2048,  # 0: SQ/CQ capacity windows
)
```

C++ 使用 `NpuRuntimeOptions` / `AccessOperatorConfig` 的同名字段。底层直接
构造 `KvafExecuteBatchGetKernelArgs` 的调用方须显式填写该字段，零初始化表示容量窗口。
Kernel 参数仍为 112 字节，新增字段占用原尾部 padding；profile/queue 大小也保持
不变，但字段语义已更新。当前 Kernel 符号为 `KvafExecuteBatchGetAivKernelV3`，在
V2 的窗口控制基础上增加规范地址到会话注册基址的转换。旧
Host/Kernel 混用会在加载符号时失败。必须整体重编 Host、Python binding 和 Kernel。
预算按值 capture，改变预算需重建 Runtime/graph；`miss_count` 内容仍可动态更新。

AIV Direct Get（URMA/UBMEM）移除 65536 条 provider 总量限制，`request_capacity` 配置仅作
预热提示。实际调用仍受输入张量、int32 `miss_count` 以及 ABI 字段宽度约束；图和输出绑定
必须按最大调用容量预热/准备。URMA 的临时 WQE bank、UBMEM 的地址批次与 MTE 缓冲区
保持固定大小，不随总请求数增长。Gather/Scatter 也移除批次总量限制，并在一次 launch 内
循环使用由字节预算决定的工作区，见 [AICPU 窗口](aicpu_get_windows.md)。Put 容量限制保持原样。

性能入口接受 `--max-inflight-total`，环境变量为
`KVAF_BATCH_GET_MAX_INFLIGHT_TOTAL`，默认 2048。通过统一性能入口扫描该参数。比较预算时固定核数、Jetty 数、请求分布、token 大小和 EID/UB 设备。

```bash
tests/run_remote_cpu_batch_get_perf.sh "$PWD/build-npu" \
  --request-length 8193 --miss-count 8193 --aiv-cores 4 --jettys-per-core 4 \
  --max-inflight-total 2048 --kernel-profile
# 同一配置再运行 --max-inflight-total 0；另测 miss-count 2047/2048/2049。
```

## 统计

Profile 增加 `submitted_group_count`、`completed_group_count`、`completion_count`、
`post_active_cycles`、`drain_wait_cycles`、`peak_submitted_not_drained`。
每 Jetty 的等待时间和 poll 次数也累计，poll 累计使用 64 位。

`kernel_post` 和 `kernel_completion_wait` 分别使用累计提交和累计 drain 时间。
`post_done_cycle` 仍表示最后一组提交结束时刻，不能再用它与 setup 的差值代表纯
提交时间。各核峰值相加只是一种保守上界，并非同一时刻的全局硬件峰值。新增
CQE、检查点和标量调度都有成本，性能改善需在 A5 上验证。

## 性能与边界检查

在 A5 上整体重编后，使用统一性能入口，固定源池、请求布局及核/Jetty 数，
扫描预算两侧的有效请求数：

```bash
bash <<'BASH'
set -euo pipefail
for misses in 2047 2048 2049 8193; do
  KVAF_CPU_SERVICE_ADDR=127.0.0.1:21111 DEVICE_ID=2 \
  bash tests/run_remote_cpu_batch_get_perf.sh build-npu \
    --backend aiv --execution both \
    --request-length 8193 --table-capacity 8193 --miss-count "$misses" \
    --token-bytes 656 --tokens-per-block 1 \
    --workers 4 --jettys-per-core 4 --max-inflight-total 2048 \
    --warmup 10 --iterations 100 --kernel-profile
done
BASH
```

改为 `--max-inflight-total 0` 测容量窗口；使用 `3/3/2049` 的核/Jetty/预算组合
观察预算余数。检查完整输出和 profile 中的组数、CQE 数及峰值在途统计。
图性能采样之前还会验证动态输入、部分请求和零请求回放。

`BUILD_TESTING=ON` 提供 `batch_get_window_model` 和 `batch_get_capacity` 测试。
前者执行真实 URMA/UBMEM kernel 控制流，覆盖百万条请求、SQ/CQ 回绕、单 CQ 槽位、
动态计数及错误/超时后的队列隔离；后者检查客户端和 CPU 服务端的容量策略及 ABI 收窄。
Python 参数检查可运行 `PYTHONPATH=tests/python python3 -m unittest test_batch_get_window_options`。
模型不模拟真实 DMA/cache 可见性和流水线时序；CANN 编译及 A5 大批次 eager/graph、
CQ 故障注入和耗时测试仍是部署前的硬件验证项。
