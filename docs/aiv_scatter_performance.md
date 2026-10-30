# AIV CPU-push scatter 性能测试

测试入口为 `tests/python/test_aiv_scatter_perf.py`，包装脚本为
`tests/run_remote_cpu_scatter_perf.sh`。固定选择 `cpu_push_scatter`，使用真实
CPU 服务、URMA WRITE、AIV GM→UB→output，不走 mock。

## 构建与运行

首次使用计时需要同时更新 CPU 服务、NPU Host/Python binding 和 scatter kernel。
CPU 服务应在空闲时用当前分支产物重启；旧服务可以执行业务传输，但无法返回
本次新增的 CPU 计时报告，测试会明确失败，不会把缺失计时当成零耗时。

CPU 机器，沿用现有 UMDK 配置：

```bash
cmake --build build-cpu -j
./build-cpu/runtime/kvaf_cpu_service --listen 0.0.0.0 --port 28990 --gather-threads 8
```

NPU 机器，在仓库根目录，沿用已验证的 CANN/PyTorch 环境及 NPU EID：

```bash
set -euo pipefail
cmake --build build-npu -j
export KVAF_CPU_SERVICE_ADDR=127.0.0.1:28990
export DEVICE_ID=7
export KVAF_NPU_EIDS=000000000000020000100000df00e100
mkdir -p results
bash tests/run_remote_cpu_scatter_perf.sh "$PWD/build-npu" \
  --cores 1 --requests 8193 --token-bytes 656 \
  --blocks 1024 --tokens-per-block 4 --plans 2 \
  --warmup 20 --iterations 100 --profile-warmup 5 --profile-iterations 20 \
  --json results/scatter-c1-n8193-t656.json \
  --csv results/scatter-c1-n8193-t656.csv \
  2>&1 | tee results/scatter-c1-n8193-t656.log
```

CPU 位于另一台机器时，地址改为实际 CPU_IP。脚本默认总超时 600 秒，可通过
`KVAF_SCATTER_PERF_TIMEOUT` 覆盖。硬件测试未 opt in 时，Python 入口返回 77；
包装脚本会显式设置 opt in。使用 `--help` 查看完整参数。

## 测试过程

1. 预先构造确定性的 CPU 源数据，经 BatchPut 写入服务端 DDR，并准备 access table。
   预分配多个随机源请求计划、output 和 stream，计时区间内不创建请求 Tensor。
2. 在测量前逐计划执行一次全输出/guard 校验。
3. `baseline` 轮关闭 CPU/AIV 细分打点：预热后测量同步 `BatchGet`。
4. `profiled` 轮开启打点，单独预热后采样。每次调用完成后，在计时区间外读取
   对应 sequence 的每核报告，并验证 magic、顺序、字节数和 CPU 报告存在性。
5. 每轮先校验最后一次被测调用留下的实际 output，再逐计划做回归校验。最终
   Shutdown 成功后才写 JSON/CSV 并打印 PASS。

源数据、初始化、BatchPut、请求生成及正确性回读不计入 BatchGet 样本。
`setup_ms` 单独记录初始化和 BatchPut。默认 strided output 是随机奇数行，
保留空洞/首尾 guard；`--output-pattern dense` 使用随机连续行集合。
每轮请求计划循环复用，CPU WR plan 可命中缓存，这是稳态复用场景。
默认 `--source-pattern random`；可改成 `sequential` 比较 CPU DDR 源访问局部性。

纯基线测试使用 `--profile-iterations 0`。这时不要求 profile API、不读取时钟
打点、不回传 CPU 报告。端到端 Python 计时仍然有效。

## 指标定义

设有效 payload 为 `requests * token_bytes`。带宽统一为十进制 GB/s，
`GB/s = payload_bytes / elapsed_ns`。不是 GiB/s，也不把 HBM 读写计作双倍有效载荷。
多样本带宽用 `sum(payload_bytes) / sum(elapsed_ns)`，不平均各次带宽。

| JSON 指标 | 计时区间 | 带宽含义 |
| --- | --- | --- |
| `baseline.end_to_end_ns` | Python 调用同步 BatchGet，直到传输、scatter、状态 D2H/同步及业务状态核验完成 | 正常完整通路有效带宽，作为主要性能结果 |
| `profiled.end_to_end_ns` | 相同区间，启用计时报告 | 包含打点和报告传输开销，用于对照 |
| `cpu_comm_ns` | CPU 首次调用提交 payload WRITE 前 → 最后一波 payload/ready 尾 CQ 处理完成 | CPU DDR→HBM staging 的有效通信带宽 |
| `cpu_write_ns` | CPU WriteBatch 入口 → 全部 payload/ready CQ 完成 | 包括 WR plan 准备、源地址刷新等 CPU 开销的有效带宽 |
| `cpu_header_ns` / `cpu_descriptor_ns` | CPU READ 请求头 / 地址清单 | 控制面读耗时，只报告延迟 |
| `cpu_post_ns` / `cpu_cq_wait_ns` | 各波 post / CQ 等待耗时之和 | CPU 阶段延迟，不单独作为 payload 带宽 |
| `aiv_body_ns` | 各核入口打点 → 本核 scatter 完成打点，取每核时长最大值 | AIV 主体有效带宽；不是整个 stream 的 event 时间 |
| `aiv_input_ns` | 各核输入/cache 准备时间，取最大值 | 输入准备延迟 |
| `aiv_build_notify_ns` | core 0 输入准备完成 → 描述符构建、header 刷新、notify 发布完成 | 请求构建/发布延迟 |
| `aiv_notify_to_done_ns` | core 0 notify 发布完成 → 观察到本轮 done | 包含 CPU 接收/读取、WRITE、报告回传及 done 的往返等待 |
| `aiv_scatter_ns` | 各核看到 done → 本核 GM→UB→output 完成，取每核时长最大值 | 本地 scatter 的有效带宽，包含地址/cache/同步开销 |

`cpu_comm_ns` 包含 CPU posting、四路 Jetty 的 ready 控制 WRITE、波次间间隔和
完成轮询处理；不是纯链路时延或理论线速。CPU 只在全部 payload/ready CQ 完成
后发布 done。AIV 的 notify→done 不能替代 CPU→HBM 通信窗口。

CPU、Host、AIV 分别在自己的时钟域计算差值，不跨域相减，也不把重叠阶段相加
解释为端到端耗时。AIV 原始周期保存在 JSON `workers` 中。Ascend 950PR/950DT
的 GetSystemCycle 计数器按 1 GHz 换算，默认 `--aiv-clock-mhz 1000`，依据
[CANN GetSystemCycle 文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/tool_interface/system_resources_and_variables/GetSystemCycle_ISASI.md)。
这是计数器频率，不能用随意填写的 AI Core 工作频率代替。其他产品需按其文档覆盖。

## 输出和采样开销

控制台每轮结束后输出 `SCATTER_PERF_RESULT`，带有效载荷的指标包含
mean/min/p50/p95/max 毫秒和 `effective_gbps`。JSON 保存配置、Get/Put SHA-256、
汇总、每次原始样本、CPU ns 和每核 AIV cycle。CSV 保存扁平样本，方便绘图。

`loop_effective_gbps` 使用整轮 wall time，包含 Python 循环开销；profiled 轮还
包含逐次 snapshot 的 D2H/转换开销。它与逐调用时间之和得到的带宽口径不同。

profiled 轮新增 CPU 本地计时、每核 128 B 报告和一个 56 B 控制 WRITE/CQ 等待，
后者在 done 之前完成以保证报告先于完成标记可见，避免依赖 64 B WRITE 原子性。
该控制 WRITE 不计入 `cpu_comm_ns`，但会增加 AIV 等待和该轮端到端时间。
snapshot D2H 不计入单次计时。`instrumentation_overhead_included=true` 标记该轮，
不要把它作为关闭打点后的基线性能。热路径不打印逐请求计时日志。
`snapshot_ns` 单独记录回读/转换耗时，CSV 同时记录 sequence 便于对应原始请求。

## 扩展组合

```bash
set -euo pipefail
for cores in 1 3 8; do
  for requests in 19 4096 8193 65536; do
    name="scatter-c${cores}-n${requests}-t656"
    bash tests/run_remote_cpu_scatter_perf.sh "$PWD/build-npu" \
      --cores "$cores" --requests "$requests" --token-bytes 656 \
      --json "results/${name}.json" --csv "results/${name}.csv" \
      2>&1 | tee "results/${name}.log"
  done
done
```

大 token 的分块边界测试可使用 `--requests 19 --blocks 16 --token-bytes 16385`
或 65535，避免把边界测试变成大内存分配测试。配置日志提供 source/output/staging
字节数。记录 CPU 服务构建方式、线程绑核和两端是否存在其他负载后再比较结果。

## 在自己的程序中采集

```python
runtime.start_scatter_profile()  # initialize 之后；会先 drain 既有调用
runtime.batch_get(view, indices, offsets, count, rows, output, synchronize=True)
records = runtime.scatter_profile()  # 校验 sequence 并回读，放在计时区间外
runtime.stop_scatter_profile()
```

仅 scatter 支持此接口。旧 kernel/CPU、缺失报告、错误 sequence 或业务执行异常
均直接失败。空请求不产生 CPU 通信，API 可以返回空工作记录；性能入口要求非空
payload，避免生成无意义的 0/0 带宽。
