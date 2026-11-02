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

当前默认启用**多核 UB 单遍清单构建**，以及两个各 32 KiB 的静态 UB bank，总计 64 KiB。
656 B 的 UB 行距为 672 B，每批最多 32 行；4096 B 每批 8 行。
每批一次多块 GM→UB，按映射连续发出精确长度的 UB→output。
只在 bank 复用前等待该 bank 写完，使下一批输入与上一批写出重叠；末尾等待两个 bank 全部完成。
尾批按实际行数搬运，超过 32 KiB 的单行使用原有精确分段路径。
清单构建按 8 条地址（64 B）分组分给各核，避免两个核写同一缓存行。
每核按 256 条一批把 index/offset 搬入 UB 并生成源地址；检查路径另加载 output row。
再一次 DMA 写回这批 descriptor。1024 项、4 核时每核构建 256 项。
地址表不超过 4096 项时，每个参与核将 address 表批量载入 UB（检查路径另加载 valid）；更大的表保留
安全的标量表查找，但请求输入和 descriptor 仍批量搬运。构建最多使用 41 KiB UB，
之后与 payload bank 复用同一 UB 地址范围。

省略第二遍 Source 读取、逐条 GM descriptor 写、对应的逐缓存行清单刷新，
以及每核对整个 request capacity 的索引/偏移/输出行号缓存刷新。
不变量（block 大小、源区域最大合法偏移）移出逐项循环，索引/offset/输出行号的负值与上界判断合并。
只处理有效前缀；空请求直接完成。Python CPU-push scatter 沿用可信输入契约，
调用方保证有效前缀的索引、offset、源地址、valid 和唯一输出行号合法；该路径省略逐项检查，
不再加载 valid 和用于检查的 output row。普通 C++ 检查路径保留全部内容检查。
地址表 DMA 与第一批请求输入共用一次等待，避免先等表、再等首批输入。
payload 的输出行号仍通过标量 GM 读取，只刷新本核实际使用的行号片段。
staging payload 仅通过 DMA 读取并禁用 L2 cache，去掉搬运前逐 64 B 的标量 DCache 刷新；
这个公共清理同时应用于 fused/reference，CPU done、DMA 完成等待和标量控制字的缓存维护保留。
依据见 [DataCacheCleanAndInvalid 的 Scalar/DMA 路径说明](https://asc.gitcode.com/api/SIMD-API/basic_api/cache_control/DataCacheCleanAndInvalid.html)。

每核 descriptor DMA 完成后在独立缓存行发布当前 sequence；core 0 等待全部核就绪，
再写请求头和 notify。任一核校验失败或构建等待超时都不发布本轮 notify。
这些构建完成标记位于 CPU layout 后的私有尾部，不能省略。
仍在整批 CPU done 后开始 scatter，现有 CPU/AIV 计时边界可以直接对比。

两项改动使用**测试时开关**，默认均为 1，同一 Host 和 kernel 二进制支持四种组合：

| 版本 | `--fused-build` | `--double-buffer` |
| --- | --- | --- |
| reference：两遍清单、单 bank | 0 | 0 |
| fused：多核 UB 单遍清单、单 bank | 1 | 0 |
| double：两遍清单、双 bank | 0 | 1 |
| both：多核 UB 单遍清单、双 bank | 1 | 1 |

测试参数覆盖同名环境变量 `KVAF_SCATTER_FUSED_BUILD`、`KVAF_SCATTER_DOUBLE_BUFFER`，
未传参数时使用环境变量，否则默认 1；只接受 0/1。
Host 在 `initialize()` 读取一次，之后修改环境变量不影响当前实例；需 shutdown 后重新初始化。
初始化日志包含 `SCATTER_OPTIONS version=1 fused_build=... double_buffer=... build=parallel_ub`
（关闭时为 `build=serial_two_pass`），确认实际选项。
flags 通过现有 176 B 参数的 reserved 字传入，使用新版标签并独立保留 profile bit；
不改 Python/C++ 方法签名或计时窗口。
本次需同时重编译 NPU Host 和 kernel：Host 为每核新增一个私有完成缓存行。
之后切换组合不需重编译。下述控制路径优化需要同时重编译并重启 CPU 服务。
旧 zero/profile-only 参数仍选择两遍清单、单 bank；构建/测试时应始终使用配套 Host 和 kernel。

所有运行时组合均保留最多 64 KiB 的静态 UB 地址范围；payload 单 bank 使用前 32 KiB，
开启 fused build 时构建阶段会复用前 41 KiB。
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
有效带宽，再用 `profiled` 的 `aiv_build_notify_us`、`aiv_body_us`、`aiv_wall_us`、
`aiv_scatter_us` 和 `cpu_comm_us` 归因。JSON 保留对应的原始 ns 字段。
四个版本均须通过完整 output/guard 校验，再交替重跑对照检查波动。
656 B、非整除前缀和其他核数也需回归，不仅依据一个形状判断。

CPU kernel 模型使用线程执行并发 scatter 核，覆盖四种开关组合、256 条构建边界、
4096/4097 项地址表缓存边界、尾批和大行 fallback、旧 sequence 复用、
非法输入与中途构建失败。模拟 CPU 会检查全部核已发布本轮构建完成标记，才消费清单。
模型按每个事件之前的 DMA 命令确认完成，
校验双 bank 的输入/输出交错及 UB 复用安全，不模拟真实硬件性能或外部 DMA 可见性。
可在有 C++ 编译器的 Linux 环境运行：

```bash
cmake -S . -B build-model -DBUILD_TESTING=ON -DKVAF_BUILD_PYTHON=OFF \
  -DKVAF_ENABLE_AIV_BATCH_GET=OFF
cmake --build build-model -j --target \
  kvaf_sequential_batch_get_kernel_model_test kvaf_sequential_host_test \
  kvaf_runtime_batch_get_contract_test
ctest --test-dir build-model -R 'kvaf_(sequential_(batch_get_kernel_model|host)|runtime_batch_get_contract)_test' --output-on-failure
```

一个 kernel 模型可执行文件在运行时遍历四种组合，并覆盖旧参数兼容与非法 flags。
Host 测试验证默认值、环境变量快照、重新初始化、profile 开关保留算法选择和非法环境值。

## 构建与运行

### 调用外围与 Host 分段诊断

`call_outside_aiv_us = end_to_end_us - aiv_wall_us`。它覆盖 Python/Pybind 包装与
Tensor 元数据读取、Host 请求准备、提交/排队、状态回拷和同步返回，
也覆盖 kernel 的计时 start 之前和 finish 之后的工作。它不是某一个 API 的计时，
也不能直接全部归为 Python 执行时间。
BatchPut、准备表、输入刷新、output guard 校验和单独读取 profile 快照均不在该单次计时内。

每次同步调用原先提交 `status memset → scatter kernel → 256 B status D2H（4 核）`，
然后执行一次 stream synchronize 并校验各核 magic/sequence/result/phase。
现在去掉每轮 status memset：状态区只在初始化时清零一次，每次 kernel 完整写回状态，
旧状态由递增 sequence 检出。

Python `cpu_push_scatter` 的 BatchGet 默认使用调用方保证输入格式的通路：
只读取七个当前 data_ptr、indices/address 两个容量和 output 的首维行数，
其余大小由配置的 token_bytes 和 int32/64 位地址/uint8 格式推导。
去掉连续性、device、element_size、维数、长度匹配、溢出等逐 Tensor 校验，
并跳过 Runtime 的 Host AccessTable 查找/生命周期/布局复验、Host Input 复验和 capture 查询。
不跨调用缓存指针，原有 changing 测试仍每轮取得当前 Tensor 存储。

调用方须提供连续 NPU Tensor：三个 int32 请求数组等长，miss_count 为 int32[1]，
地址表为 64 位地址、valid 为等长 uint8；output 每行字节数等于配置 token_bytes。
容量和生命周期由调用方保证，使用 eager stream；该通路不验证也不支持 graph capture。
Python 签名不变；其他模式及 C++ 默认调用保留原验证行为。
C++ 调用者可通过 `BatchGetExecutionOptions::trusted_npu_request=true` 使用同一合同。
完成等待、异步 workspace 复用前的 drain、设备结果和 API 错误处理仍执行。

profiled 阶段默认采集 Host 分段，结果写进 JSON，并在 CSV 原 27 列后追加五列，合计 32 列：
`host_tensor_metadata_us`、`host_launch_us`、`host_status_copy_enqueue_us`、
`host_stream_sync_us`、`host_python_dispatch_us`，统一保留三位小数。
关闭采样的行留空，不以 0 代替未采集值。完整端到端入口示例：

```bash
cmake --build build-npu -j
bash tests/run_remote_cpu_scatter_perf.sh "$PWD/build-npu" \
    --host-profile --cores 4 --fused-build 1 --double-buffer 1 \
    --blocks 1024 --tokens-per-block 4 --token-bytes 4096 \
    --request-length 8192 --miss-count 1024 --input-mode changing \
    --warmup 50 --iterations 1000 --profile-rounds 100 \
    --json results/scatter-host-profile.json --csv results/scatter-host-profile.csv
```

也可在直接使用绑定时设置 `KVAF_SCATTER_HOST_PROFILE=1`，然后调用现有
`start_scatter_profile()`。测试脚本以 `--host-profile/--no-host-profile` 为准覆盖环境变量。
脚本在 profiled 阶段默认启用，可用 `--no-host-profile` 关闭；baseline 不读取 Host 时钟。
`--profile-rounds 0` 默认同时关闭 Host 计时。
本次需重编译绑定、Runtime 和 NPU Host；CPU 服务与 AIV kernel 协议不变。

`samples.profiled` 中新增原始 `host_*_ns` 及对应 `host_*_us`，`summary.profiled`
保留各分段的 `mean_us/p50_us/p95_us/min_us/max_us` 等统计。
字段的包含关系如下，各层存在嵌套，不能全部相加：

```text
end_to_end
├─ host_python_dispatch           调用总时长减 binding：包装、重载派发和返回差值
└─ host_binding                   BatchGetImpl 整体
   ├─ host_tensor_metadata        必需的指针/容量读取与请求组装
   ├─ host_stream_resolve         stream 对象转 ACL handle
   └─ host_native                 Runtime::BatchGet 整体
      ├─ host_runtime_validation  初始化状态与通路分发；完整复验已跳过
      └─ host_operator            ScatterBatchGet::Execute 整体
         ├─ host_mutex_wait       获取实例锁
         ├─ host_capture_query    该通路省略，JSON 中为 0
         ├─ host_prior_drain      收尾上一笔请求；同步稳态下无上一笔待完成任务
         ├─ host_input            kernel 参数构造，无布局复验
         ├─ host_launch           kernel 提交 API
         ├─ host_status_copy_enqueue  状态 D2H 的提交 API
         ├─ host_stream_sync      等待设备、D2H 完成及同步返回
         └─ host_status_validate  各核状态校验
```

上述名字在 JSON 中以 `_ns`/`_us` 结尾；父层还包含少量分段之间的工作。
若一次调用需要 drain 上一笔异步请求，`host_prior_drain` 内还包含相应的 copy/sync/validate，
因此这些字段也不能简单相加当成严格的互斥分区。
`host_stream_sync` 包含尚未完成的 AIV/CPU 通信时间，不能整段算成额外开销，
也不能直接用它减完整 aiv_wall：设备可能在 Host 进入同步 API 之前已经开始执行。
Host 诊断增加计时开销，只用于定位；最终收益用默认关闭诊断的 baseline 比较。

该通路按 eager stream 合同直接提交；同步接口仍等待真实完成。
状态校验仍用于发现 kernel 正常返回但某核结果为 INVALID/TIMEOUT/REMOTE_ERROR 的情况。
设置 `synchronize=False` 只改变返回时机，不能据此宣称同步端到端耗时降低。

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
`GB/s = payload_bytes / elapsed_ns = payload_bytes / (elapsed_us * 1000)`。
不是 GiB/s，也不把 HBM 读写计作双倍有效载荷。
多样本带宽用 `sum(payload_bytes) / sum(elapsed_ns)`，不平均各次带宽。

| JSON 原始指标（CSV 核心耗时改用同名 `_us`） | 计时区间 | 带宽含义 |
| --- | --- | --- |
| `baseline.end_to_end_ns` | Python 调用同步 BatchGet，直到传输、scatter、状态 D2H/同步及业务状态核验完成 | 正常完整通路有效带宽，作为主要性能结果 |
| `profiled.end_to_end_ns` | 相同区间，启用计时报告 | 包含打点和报告传输开销，用于对照 |
| `cpu_comm_ns` | CPU 首次调用提交 payload WRITE 前 → 最后一波 payload 尾 CQ 处理完成（旧进度模式另含 ready） | CPU DDR→HBM staging 的有效通信带宽 |
| `cpu_write_ns` | CPU WriteBatch 入口 → 全部 payload CQ 完成 | 包括 WR plan 准备、源地址刷新等 CPU 开销的有效带宽 |
| `cpu_header_ns` / `cpu_descriptor_ns` | CPU 合并 READ 请求头/上次 count 前缀 / count 增长时补读清单尾部 | 控制面读耗时，只报告延迟；旧版分别计 header / 全清单 |
| `cpu_post_ns` / `cpu_cq_wait_ns` | 各波 post / CQ 等待耗时之和 | CPU 阶段延迟，不单独作为 payload 带宽 |
| `aiv_body_ns` | 各核入口打点 → 本核 scatter 完成打点，取每核时长最大值 | AIV 主体有效带宽；不是整个 stream 的 event 时间 |
| `aiv_input_ns` | 各核输入/cache 准备时间，取最大值 | 输入准备延迟 |
| `aiv_build_notify_ns` | core 0 输入准备完成 → 描述符构建、header 刷新、notify 发布完成 | 请求构建/发布延迟 |
| `aiv_notify_to_done_ns` | core 0 notify 发布完成 → 观察到本轮 done | 包含 CPU 接收/读取、WRITE、报告回传及 done 的往返等待 |
| `aiv_scatter_ns` | 各核看到 done → 本核 GM→UB→output 完成，取每核时长最大值 | 本地 scatter 的有效带宽，包含地址/cache/同步开销 |

CSV 使用固定的 **32 列**，baseline-only、profiled-only 和混合采样的表头/顺序一致。
所有耗时只写 `_us`，保留 3 位小数；带宽为 GB/s，保留 6 位小数。
内部计时、计时边界及带宽分子保持原定义；JSON 保留原始 ns、完整指标、每核报告和完整测试配置。

| CSV 分组 | 保留列 |
| --- | --- |
| 标识与规模（10 列） | `phase`、`iteration`、`sequence`、`cores`、`fused_build`、`double_buffer`、`request_length`、`miss_count`、`token_bytes`、`payload_bytes` |
| 核心耗时（11 列） | `end_to_end_us`、`cpu_comm_us`、`aiv_build_notify_us`、`aiv_scatter_us`、`cpu_write_us`、`aiv_body_us`、`aiv_wall_us`、`aiv_input_wall_us`、`aiv_notify_to_done_us`、`aiv_scatter_tail_us`、`call_outside_aiv_us` |
| 有效带宽（6 列） | `end_to_end_gbps`、`cpu_comm_gbps`、`aiv_scatter_gbps`、`cpu_write_gbps`、`aiv_body_gbps`、`aiv_wall_gbps` |
| Host 耗时（新增 5 列） | `host_tensor_metadata_us`、`host_launch_us`、`host_status_copy_enqueue_us`、`host_stream_sync_us`、`host_python_dispatch_us` |

端到端、通信和 scatter 的耗时/带宽列相邻放置。`calls=1` 和重复的 `requests` 不再逐行输出。
workload、input_mode、source_blocks、tokens_per_block 和输出布局统一查看 JSON `config`。
header/descriptor/post/CQ、最大单核 input、launch skew、snapshot 等诊断耗时只保留在 JSON，
CSV 不输出这些字段或任何 `_ns` 列。baseline 的 CPU/AIV/Host 和 sequence 列留空，不写成 0。
仍然每行只对应一次同步 BatchGet，不把数组容量当作有效请求数。

读取数据应使用 `csv.DictReader` 等方式按表头取值。旧 CSV 的列顺序由实际包含的采样字段决定，
不要把旧表头或固定列号套到另一个版本上。例如双优化 4 MiB 样本首行的原始值是：

```text
cpu_descriptor_us    =   3.790   # CPU READ 1024 个源地址，共 8192 B；仅在 JSON 保留
cpu_comm_us          = 133.960   # CPU WRITE 4 MiB payload 及 ready 的通信窗口
aiv_build_notify_us = 103.302   # AIV 校验、清单生成和通知
aiv_scatter_us      =  59.863   # HBM staging -> UB -> output
cpu_comm_gbps       =  31.310122
```

约 4 us 和 103 us 在该样本中分别属于地址清单 READ 和构建阶段，不能当作 CPU payload 通信和 scatter。
如果套用了新增 `fused_build`、`double_buffer` 之前的旧表头，后续指标会错位两列：
旧 `cpu_comm_us` 恰好读到新 `cpu_descriptor_us`，旧 `aiv_scatter_us` 恰好读到新 `aiv_build_notify_us`。
如果另一个新样本确实按表头显示 4 MiB 的 `cpu_comm_us=4`，其带宽应为 1048.576 GB/s，
与上述约 31 GB/s 的通信窗口不一致，需要连同表头、payload 和计时原始记录检查。

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

`cpu_comm_ns` 包含 CPU posting、波次间间隔和完成轮询处理；旧进度模式另含四路 Jetty 的
ready 控制 WRITE。它不是纯链路时延或理论线速。CPU 只在全部 payload 尾 CQ 完成
后发布 done。AIV 的 notify→done 不能替代 CPU→HBM 通信窗口。

CPU、Host、AIV 分别在自己的时钟域计算时长，不跨域相减绝对时间戳，也不把重叠阶段相加
解释为端到端耗时。AIV 原始周期保存在 JSON `workers` 中。Ascend 950PR/950DT
的 GetSystemCycle 计数器按 1 GHz 换算，默认 `--aiv-clock-mhz 1000`，依据
[CANN GetSystemCycle 文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/tool_interface/system_resources_and_variables/GetSystemCycle_ISASI.md)。
这是计数器频率，不能用随意填写的 AI Core 工作频率代替。其他产品需按其文档覆盖。

## 输出和采样开销

控制台每轮结束后输出 `SCATTER_PERF_RESULT`，带有效载荷的指标包含
mean/min/p50/p95/max 毫秒和 `effective_gbps`。JSON 保存配置、Get/Put SHA-256、
汇总、每次原始样本、CPU ns 和每核 AIV cycle。CSV 保存固定 32 列样本，方便绘图。

`loop_effective_gbps` 使用整轮 wall time，包含 Python 循环开销；profiled 轮还
包含逐次 snapshot 的 D2H/转换开销。整轮还包含输入准备、BatchPut、表准备、
NPU 消费和完整校验，因此只是整个测试程序的 wall 吞吐，不是 BatchGet 带宽。
业务性能比较使用 `end_to_end_ns.effective_gbps`（汇总）和 `end_to_end_gbps`（每次）。

执行算法也要区分：当前 AIV 先观察整批 done，再开始本地 scatter；AICPU 参考路径按
1024 行 chunk 的共同 ready 前缀流水消费，CPU WRITE 可以与 AICPU scatter 重叠。
对齐测试输入不会自动消除这项实现差异。同步调用完成始终表示整个有效前缀已完成。

profiled 轮新增 CPU 本地计时、每核 128 B 报告和一个 56 B 控制 WRITE。
当前将报告 WRITE 与 8 B done WRITE 链接后一次提交，仅等待尾部 CQ；done 使用强放置顺序，
保证报告先于完成标记可见，避免依赖 64 B WRITE 原子性，也避免单独等待报告的一次往返。
该控制链不计入 `cpu_comm_ns`，但计入 AIV 等待和该轮端到端时间。
snapshot D2H 不计入单次计时。`instrumentation_overhead_included=true` 标记该轮，
不要把它作为关闭打点后的基线性能。热路径不打印逐请求计时日志。
`snapshot_ns` 在 JSON 中单独记录回读/转换耗时；CSV 记录 sequence 便于对应 JSON 原始请求。

## 小包控制与 Host 路径优化

当前 scatter 在 Request 的已有 reserved 位声明 `kBatchDoneOnly`，CPU 保持原有 workspace
偏移，但不再写 AIV 不读取的逐 worker/chunk/Jetty Ready 进度。CPU 直接将有效前缀
分配给四条 Jetty，每条尾 WR 强放置排序并产生 CQE；所有 payload 尾 CQE 到齐后才提交
最终 done。构建 ready、错误传播、超时、整批 done 和 scatter DMA 完成等待均保留。
未声明该位的消费者仍使用原有进度协议，写计划缓存同时区分 count 与该模式。

1024 项、4 核、max_sge=16、message_bytes=65536 时，656 B 和 4096 B 均为 64 条
payload WR：移除 16 条 Ready WR 后，总数从 80 减到 64，post 仍为 4 次，payload 尾 CQE
仍为 4 个。WR 减少不代表带宽按比例增加，需看完整调用实测。

CPU 在观察到新 notify 后，将 header 和上次 count 对应的 descriptor 前缀合并 READ。
相同/更小 count 时不再单独读 descriptor；首次或 count 增长时补读缺少的尾部，较大
请求仍按设备 message_bytes 分段。每次都重新读取地址内容，不缓存地址；支持每轮
更换源和同指针内容变化。notify 轮询仍保留。稳态小包由 notify/header/descriptor
三步变为 notify/合并前缀两步，不会把整个 capacity 的 65536 个地址都读回来。

计时字段沿用原有 32 列：合并 READ 计入 JSON 的 `cpu_header_ns`；`cpu_descriptor_ns`
仅记录 count 增长时的补读，稳定 count 时为 0。CPU 通信仍为 payload 首次 post 至末尾
CQE 的完整窗口；Ready 在新模式中不再参与。profile 报告和最终 done 在该窗口之外，
但仍在 `aiv_notify_to_done_us` 和端到端计时之内。

Host 复用 Python 方法名，使用 CPython method vectorcall 读取当前 Tensor 指针/尺寸，
避免重复构造绑定方法与调用参数元组；显式 stream 只取一次 `npu_stream`。不缓存
Tensor 指针或尺寸，不依赖固定输入对象，不增加 PyTorch C++ 依赖。方法签名不变。
kernel 提交、各核 status 回拷、一次 stream 同步和执行结果检查仍是同步调用的必要步骤。

本次需重编译 NPU binding/Host/kernel 和 CPU service，并重启 CPU service。
旧 Host/新 kernel 的 trusted 位、旧 CPU 服务的 Ready 行为会影响对照，测试使用配套产物。
Linux/URMA 环境新增 `kvaf_cpu_push_transport_test`，执行真实 Transport 的延迟 URMA
替身：检查 count 增长/缩小/不变、地址变化、65536 项多波次、旧/新进度模式切换、
精确有效前缀、报告/done 顺序及 CQ 错误；不用于宣称硬件可见性或性能已通过。

仍用完整端到端测试对比旧版和新版，保留相同 CPU 服务、EID、核数和输入设置：

```bash
cmake --build build-cpu --target kvaf_cpu_service kvaf_cpu_push_transport_test -j
ctest --test-dir build-cpu -R kvaf_cpu_push_transport_test --output-on-failure
cmake --build build-npu -j
ctest --test-dir build-npu -R 'kvaf_sequential_(host_test|batch_get_kernel_model_test)' --output-on-failure
# 重启更新后的 CPU service，然后运行已有端到端测试；65535 B/多核边界另跑正确性回归。
for token in 656 4096; do
  bash tests/run_remote_cpu_scatter_perf.sh "$PWD/build-npu" \
    --cores 4 --blocks 1024 --tokens-per-block 4 --token-bytes "$token" \
    --request-length 8192 --miss-count 1024 --input-mode changing \
    --fused-build 1 --double-buffer 1 --iterations 100 --profile-rounds 100 \
    --json "results/compact-${token}.json" --csv "results/compact-${token}.csv"
done
```

优先比较 baseline 的端到端结果，再用 profiled 的 `cpu_write_us`、`cpu_comm_us`、
`aiv_build_notify_us`、`call_outside_aiv_us` 与五个 Host 字段归因。只看快样本时，
预先固定和旧数据相同的筛选规则；同时保留原始样本，不改变有效字节数或计时边界。

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
