# AIV CPU-push scatter 性能测试

测试入口为 `tests/python/test_aiv_scatter_perf.py`，包装脚本为
`tests/run_remote_cpu_scatter_perf.sh`。固定选择 `cpu_push_scatter`，使用真实
CPU 服务、URMA WRITE、AIV GM→UB→output，不走 mock。

**一个样本就是一次同步 BatchGet，不是多次调用的总耗时，也不是单 token 耗时。**
默认 `1024 blocks × 4 tokens/block × 656 B`，每次取满 4096 个 token，
有效载荷为 **2,686,976 B**。`--request-length` 只控制输入 Tensor 容量，
`--miss-count` 控制这次调用实际处理的 token 前缀，两者独立。
例如容量 8192、miss_count 4096，一次只传输 4096 个 token；无效后缀填 -1 并检查不被访问。

测试设置参考本地 `aicpu-urma` 分支的
`access_operator/demo/aicpu_batch_get/python/test_batch_get.py`：每轮改变源 payload、
源索引/偏移和输出映射，计时前 BatchPut/准备表，计时后校验整个 output/guard，
并先由后继 NPU 算子消费结果。默认输出为随机排列的相邻行。
参考脚本每个 block 只随机取一个 token，不会自动取满 tokens-per-block；
严格复现这个请求语义使用 `--workload one-token-per-block`。

| workload | 默认每次有效请求数（1024 blocks、4 tokens/block） | 每次有效字节数 |
| --- | ---: | ---: |
| `full-blocks`（默认） | 4096，每个 block 的四个 token 各取一次 | 2,686,976 |
| `one-token-per-block`（AICPU 参考语义） | 1024，每个 block 随机取一个 token | 671,744 |
| `sampled-tokens` | 默认等于 request-length，允许重复采样 | miss_count × 656 |

`--active-percent 75` 或显式 `--miss-count` 可以测量部分前缀。
旧 `--requests` 现为 `--request-length` 的别名，不再隐式决定有效前缀。
旧脚本的 `--requests 8194` 则确实代表一次调用完成 8194 项，其结果不能直接当成新默认形状的性能。

## 清单构建与 UB 流水版本对比

当前默认启用单遍校验/清单构建，以及两个各 32 KiB 的静态 UB bank，总计 64 KiB。
656 B 的 UB 行距为 672 B，每批最多 32 行；4096 B 每批 8 行。
每批一次多块 GM→UB，按映射连续发出精确长度的 UB→output。
只在 bank 复用前等待该 bank 写完，使下一批输入与上一批写出重叠；末尾等待两个 bank 全部完成。
尾批按实际行数搬运，超过 32 KiB 的单行使用原有精确分段路径。
清单构建复用已校验的 index/offset/base，全部成功并刷新清单后才发布 notify。
仍在整批 CPU done 后开始 scatter，现有 CPU/AIV 计时边界可以直接对比。

两项改动使用**测试时开关**，默认均为 1，同一 Host 和 kernel 二进制支持四种组合：

| 版本 | `--fused-build` | `--double-buffer` |
| --- | --- | --- |
| reference：两遍清单、单 bank | 0 | 0 |
| fused：单遍清单、单 bank | 1 | 0 |
| double：两遍清单、双 bank | 0 | 1 |
| both：单遍清单、双 bank | 1 | 1 |

测试参数覆盖同名环境变量 `KVAF_SCATTER_FUSED_BUILD`、`KVAF_SCATTER_DOUBLE_BUFFER`，
未传参数时使用环境变量，否则默认 1；只接受 0/1。
Host 在 `initialize()` 读取一次，之后修改环境变量不影响当前实例；需 shutdown 后重新初始化。
初始化日志应包含 `SCATTER_OPTIONS version=1 fused_build=... double_buffer=...`，确认实际选项。
flags 通过现有 176 B 参数的 reserved 字传入，使用新版标签并独立保留 profile bit；
不改 Python/C++ 公共接口、CPU URMA 协议或计时窗口。
首次需同时重编译 NPU Host 和 kernel，此后切换组合不需重编译。
旧 Host 的 zero/profile-only 参数仍选择两遍清单、单 bank；新 Host 配旧 kernel 会验证失败。

所有运行时组合均保留最多 64 KiB 的静态 UB 地址范围，单 bank 只使用前 32 KiB。
因此 reference 是同一二进制下的算法对照，不等价于重编译一份仅声明 32 KiB UB 的旧二进制。
如需评估最终代码体积、编译布局或 UB 资源差异，另保留旧版本独立比较。

验收使用同一完整端到端测试。各组合用独立进程、相同参数比较。
示例沿用下文已配置好的 CANN/PyTorch、服务地址、EID、DEVICE_ID 和 PYTHONPATH：

```bash
set -euo pipefail
KVAF_AB_DIR="$PWD/results/scatter-ab-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$KVAF_AB_DIR"
cp build-npu/bin/kvaf_scatter_batch_get_kernels.o "$KVAF_AB_DIR/prior.o"
cmake --build build-npu -j

for variant in reference fused double both; do
  case "$variant" in
    reference) fused=0; double=0 ;;
    fused)     fused=1; double=0 ;;
    double)    fused=0; double=1 ;;
    both)      fused=1; double=1 ;;
  esac
  bash tests/run_remote_cpu_scatter_perf.sh "$PWD/build-npu" \
      --fused-build "$fused" --double-buffer "$double" \
      --cores 4 --workload full-blocks --input-mode changing \
      --blocks 1024 --tokens-per-block 4 --token-bytes 4096 \
      --request-length 8192 --miss-count 1024 \
      --source-pattern random --output-row-step 1 --output-order random --seed 1234 \
      --warmup 50 --iterations 1000 --profile-rounds 100 \
      --json "$KVAF_AB_DIR/${variant}.json" --csv "$KVAF_AB_DIR/${variant}.csv"
done
```

该例固定为本轮 4 MiB 形状（1024 个有效 token），源模式采用 random；
复现已有数据时还应与其 JSON 中的 source-pattern 一致。
各文件中 `phase=baseline` 都表示关闭细分打点，不表示旧算法。
通过 CSV/JSON 的 `fused_build`、`double_buffer` 区分算法组合，kernel SHA256 应完全一致。
先比较这部分的端到端 p50/p95、
有效带宽，再用 `profiled` 的 `aiv_scatter_ns` 和 `cpu_comm_ns` 归因。
四个版本均须通过完整 output/guard 校验，再交替重跑对照检查波动。
656 B、非整除前缀和其他核数也需回归，不仅依据一个形状判断。

已有 CPU kernel 模型覆盖四种开关组合、尾批和大行 fallback、旧 sequence 复用、
非法输入与中途构建失败。模型按每个事件之前的 DMA 命令确认完成，
校验双 bank 的输入/输出交错及 UB 复用安全，不模拟真实硬件性能或外部 DMA 可见性。
可在有 C++ 编译器的 Linux 环境运行：

```bash
cmake -S . -B build-model -DBUILD_TESTING=ON -DKVAF_BUILD_PYTHON=OFF \
  -DKVAF_ENABLE_AIV_BATCH_GET=OFF
cmake --build build-model -j --target \
  kvaf_sequential_batch_get_kernel_model_test kvaf_sequential_host_test
ctest --test-dir build-model -R 'kvaf_sequential_(batch_get_kernel_model|host)_test' --output-on-failure
```

一个 kernel 模型可执行文件在运行时遍历四种组合，并覆盖旧参数兼容与非法 flags。
Host 测试验证默认值、环境变量快照、重新初始化、profile 开关保留算法选择和非法环境值。

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
  --cores 4 --workload full-blocks --input-mode changing \
  --blocks 1024 --tokens-per-block 4 --token-bytes 656 \
  --request-length 8192 --miss-count 4096 \
  --output-row-step 1 --output-order random \
  --warmup 50 --iterations 1000 --profile-warmup 1 --profile-rounds 100 \
  --json results/scatter-c4-b1024-t4.json \
  --csv results/scatter-c4-b1024-t4.csv \
  2>&1 | tee results/scatter-c4-b1024-t4.log
```

CPU 位于另一台机器时，地址改为实际 CPU_IP。脚本默认总超时 600 秒，可通过
`KVAF_SCATTER_PERF_TIMEOUT` 覆盖。硬件测试未 opt in 时，Python 入口返回 77；
包装脚本会显式设置 opt in。使用 `--help` 查看完整参数。
未提供输出路径时，包装脚本自动在 `results/aiv-scatter-时间-进程号.{json,csv}` 写入两种结果。
`KVAF_SCATTER_PERF_RESULT_PREFIX` 可覆盖默认结果前缀。

与 AICPU 的每 block 单 token 测试比较时，保持相同 worker 数、输出布局和轮次，改为：

```bash
bash tests/run_remote_cpu_scatter_perf.sh "$PWD/build-npu" \
  --cores 4 --workload one-token-per-block --input-mode changing \
  --blocks 1024 --tokens-per-block 4 --token-bytes 656 \
  --request-length 1024 --miss-count 1024 \
  --output-row-step 1 --output-order random \
  --warmup 50 --iterations 1000 --profile-rounds 100
```

## 测试过程

1. 预分配并复用设备 source、请求 Tensor、output 和 stream。默认 changing 模式
   每轮更新 payload 和请求，将 output 重置为 guard，经 BatchPut 写入 DDR，准备新的 access table。
   这些操作及同步都在该次计时开始前完成。
2. 每个样本只调用一次 `batch_get(..., synchronize=True)`，等待有效前缀全部传输、
   scatter 及业务状态检查完成后停止计时。
3. `baseline` 轮关闭 CPU/AIV 细分打点：预热后测量同步 `BatchGet`。
4. `profiled` 轮开启打点，单独预热后采样。每次调用完成后，在计时区间外读取
   对应 sequence 的每核报告，并验证 magic、顺序、字节数和 CPU 报告存在性。
5. 每个预热/采样调用结束后先由 NPU 消费 output，再回读并校验完整 output/guard。
   changing 模式随后释放该轮 access table 和 keys，下轮重新填充。
   最终 Shutdown 成功后才写 JSON/CSV 并打印 PASS。

源数据、初始化、BatchPut、请求生成及正确性回读不计入 BatchGet 样本。
`setup_ms` 记录初始化；steady 模式额外记录初次 BatchPut。
默认 `--output-row-step 1 --output-order random`：随机排列相邻输出行；
`--output-row-step 2` 留输出空洞，`--output-order sequential` 按行顺序输出。
`--output-pattern dense/strided` 是 row-step 1/2 的兼容别名。
`--input-mode steady --plans 2` 复用源 payload、access table 和两个请求计划，
仍逐次重置并校验 output；这是额外的固定输入对照，不能混同 changing 结果。
默认 `--source-pattern random`：full-blocks 随机排列不重复 token，one-token-per-block
随机排列不重复 block 并各选一个随机 token；sampled-tokens 才允许重复源 token。
改为 `sequential` 可以比较源访问顺序。

纯基线测试使用 `--profile-iterations 0`。这时不要求 profile API、不读取时钟
打点、不回传 CPU 报告。端到端 Python 计时仍然有效。

## 指标定义

设有效 payload 为 **`miss_count * token_bytes`**，与请求数组容量无关。带宽统一为十进制 GB/s，
`GB/s = payload_bytes / elapsed_ns`。不是 GiB/s，也不把 HBM 读写计作双倍有效载荷。
多样本带宽用 `sum(payload_bytes) / sum(elapsed_ns)`，不平均各次带宽。

| JSON/CSV 指标 | 计时区间 | 带宽含义 |
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

CSV 每一行还直接提供对应的 `_us` 和有效载荷带宽字段：
`end_to_end_gbps`、`cpu_comm_gbps`、`cpu_write_gbps`、`aiv_body_gbps`、
`aiv_wall_gbps`、`aiv_scatter_gbps`。header、post、CQ 等子阶段只给延迟，不拿全量 payload
除以这些子阶段耗时生成误导性的带宽。baseline 只包含调用级计时和带宽，CPU/AIV 列留空。

`calls=1`、`source_blocks`、`tokens_per_block`、`request_length`、`miss_count`、
`payload_bytes`、`workload`、`input_mode` 和输出布局逐行记录，避免把源表、容量、有效请求混为一谈。

多核还提供以下调用级时间线。所有 AIV 核使用同一个系统计数器：

| 字段 | 定义 |
| --- | --- |
| `aiv_wall_ns` | 最早核 start → 最晚核 finish，包含核启动差异 |
| `aiv_input_wall_ns` | 最早核 start → core 0 输入准备完成 |
| `aiv_build_notify_ns` | core 0 输入准备完成 → notify |
| `aiv_notify_to_done_ns` | core 0 notify → core 0 观察到 done |
| `aiv_scatter_tail_ns` | core 0 观察到 done → 最晚核 finish；含其他核观察 done 的差异 |
| `aiv_launch_skew_ns` | 最晚核 start − 最早核 start，诊断启动偏斜 |
| `call_outside_aiv_ns` | Python 单次同步调用时长 − aiv_wall；包含 binding、launch、设备末尾收尾及同步/状态检查，不是纯 Host 时间 |

其中四个不重叠的 AIV 时间段严格满足：

```text
aiv_wall = input_wall + build_notify + notify_to_done + scatter_tail
end_to_end = call_outside_aiv + aiv_wall
```

`call_outside_aiv` 是两个已分别计算好的时长之差，不对 Host/device 绝对时间戳做相减；
若 AIV wall 大于同步调用时长，测试报错，需检查计数器换算。
旧 `aiv_body`/`aiv_scatter` 保留“最大单核区间”的含义，不能将各阶段单独取最大值后相加。

`cpu_comm_ns` 包含 CPU posting、四路 Jetty 的 ready 控制 WRITE、波次间间隔和
完成轮询处理；不是纯链路时延或理论线速。CPU 只在全部 payload/ready CQ 完成
后发布 done。AIV 的 notify→done 不能替代 CPU→HBM 通信窗口。

CPU、Host、AIV 分别在自己的时钟域计算时长，不跨域相减绝对时间戳，也不把重叠阶段相加
解释为端到端耗时。AIV 原始周期保存在 JSON `workers` 中。Ascend 950PR/950DT
的 GetSystemCycle 计数器按 1 GHz 换算，默认 `--aiv-clock-mhz 1000`，依据
[CANN GetSystemCycle 文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/tool_interface/system_resources_and_variables/GetSystemCycle_ISASI.md)。
这是计数器频率，不能用随意填写的 AI Core 工作频率代替。其他产品需按其文档覆盖。

## 输出和采样开销

控制台每轮结束后输出 `SCATTER_PERF_RESULT`，带有效载荷的指标包含
mean/min/p50/p95/max 毫秒和 `effective_gbps`。JSON 保存配置、Get/Put SHA-256、
汇总、每次原始样本、CPU ns 和每核 AIV cycle。CSV 保存扁平样本，方便绘图。

`loop_effective_gbps` 使用整轮 wall time，包含 Python 循环开销；profiled 轮还
包含逐次 snapshot 的 D2H/转换开销。整轮还包含输入准备、BatchPut、表准备、
NPU 消费和完整校验，因此只是整个测试程序的 wall 吞吐，不是 BatchGet 带宽。
业务性能比较使用 `end_to_end_ns.effective_gbps`（汇总）和 `end_to_end_gbps`（每次）。

执行算法也要区分：当前 AIV 先观察整批 done，再开始本地 scatter；AICPU 参考路径按
1024 行 chunk 的共同 ready 前缀流水消费，CPU WRITE 可以与 AICPU scatter 重叠。
对齐测试输入不会自动消除这项实现差异。同步调用完成始终表示整个有效前缀已完成。

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
  for blocks in 256 1024 4096; do
    name="scatter-c${cores}-b${blocks}-t4"
    bash tests/run_remote_cpu_scatter_perf.sh "$PWD/build-npu" \
      --cores "$cores" --blocks "$blocks" --tokens-per-block 4 --token-bytes 656 \
      --json "results/${name}.json" --csv "results/${name}.csv" \
      2>&1 | tee "results/${name}.log"
  done
done
```

大 token 的分块边界测试可使用
`--workload sampled-tokens --request-length 19 --blocks 16 --token-bytes 16385`
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
