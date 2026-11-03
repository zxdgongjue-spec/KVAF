# AIV CPU-push scatter 性能测试

测试入口为 `tests/python/test_aiv_scatter_perf.py`，包装脚本为
`tests/run_remote_cpu_scatter_perf.sh`。固定选择 `cpu_push_scatter`，使用真实
CPU 服务、URMA WRITE、AIV GM→UB→output，不走 mock。

临时 HBM staging 所属的持久 workspace 使用 `ACL_MEM_MALLOC_HUGE_FIRST` 分配，
优先使用大页，大页不足时允许回退普通页；分配成功不代表一定获得了大页。

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

当前默认启用**多核 SIMT 单遍清单构建**，以及两个各 32 KiB 的 payload UB bank。
混合二进制的 loader 通过 `dynUBufSize=65536` 预留这 64 KiB 共享 UB；
`LocalTensor`/`get_imm` 只是地址视图，不能替代 SIMT 启动时的共享内存预留。
656 B 的 UB 行距为 672 B，每批最多 32 行；4096 B 每批 8 行。
每批一次多块 GM→UB，按映射连续发出精确长度的 UB→output。
只在 bank 复用前等待该 bank 写完，使下一批输入与上一批写出重叠；末尾等待两个 bank 全部完成。
尾批按实际行数搬运，超过 32 KiB 的单行使用原有精确分段路径。
清单构建按 8 条地址（64 B）分组分给各核，避免两个核写同一缓存行。
SIMT 路径每核启动 256 个线程并行读 index/offset、查地址表、加偏移，直接生成紧凑的 UB 清单，
每批最多 4096 条。8192 项、4 核时，每核 2048 项，只需一次 VF 调用和一次 16 KiB descriptor DMA。
输入通过 `asc_ldcg` 绕过核私有 SIMT DCache，防止同地址请求/表内容更新后读到旧值；
该接口从共享 L2 或 GM 加载，见 [asc_ldcg](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta2/API/ascendcopapi/atlasascendc_api_07_10556.html)。
检查路径仍先检查 index/offset/output row，再读 valid 和源地址，线程失败标志归并后不发布 notify。
可信输入路径不读 valid/output row、不归并失败标志；VF→MTE3 与清单写回完成等待始终保留。
SIMT 构建使用 32 KiB 清单槽和 1 KiB 检查槽，复用预留的 64 KiB 共享 UB。

`--simt-build 0` 保留上一版 UB/DMA 构建作为同二进制对照：
每核按 256 条一批把 index/offset 搬入 UB 并生成源地址；检查路径另加载 output row。
再一次 DMA 写回这批 descriptor。1024 项、4 核时每核构建 256 项。
地址表不超过 4096 项时，每个参与核将 address 表批量载入 UB（检查路径另加载 valid）；更大的表
按本批 index 发起精确 8 B 的地址条目 DMA，每项使用独立 32 B UB 槽，全部发出后集中等待一次。
检查路径先检查整批 index/offset/row，再发起 address/valid DMA，避免后续非法索引留下未完成搬运。
地址与 valid 均从 UB 读取，不再逐项 DCCI 后串行读取 GM；每次调用重新搬入条目，支持表原址更新。
GM→UB 的非对齐填充规则见 [DataCopyPad](https://asc.gitcode.com/api/SIMD-API/basic_api/memory_vector_compute/data_move/DataCopyPad_GMToUB.html)。
构建最多使用 41 KiB UB，
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
`--pipeline 0` 在整批 CPU done 后开始 scatter；默认 `--pipeline 1` 按 Ready 分块消费，
与后续 CPU WRITE 重叠。它独立于 UB 双缓冲开关；具体时序与计时含义见文末“通信与 scatter 重叠”。

三项选择均使用**测试时开关**，默认均为 1，同一 Host 和 kernel 二进制支持下列组合：

| 版本 | `--fused-build` | `--simt-build` | `--double-buffer` |
| --- | --- | --- | --- |
| reference：两遍清单、单 bank | 0 | 忽略 | 0 |
| double：两遍清单、双 bank | 0 | 忽略 | 1 |
| UB 单遍清单、单 bank | 1 | 0 | 0 |
| UB 单遍清单、双 bank | 1 | 0 | 1 |
| SIMT 单遍清单、单 bank | 1 | 1 | 0 |
| SIMT 单遍清单、双 bank（默认） | 1 | 1 | 1 |

测试参数覆盖同名环境变量 `KVAF_SCATTER_FUSED_BUILD`、`KVAF_SCATTER_DOUBLE_BUFFER`、`KVAF_SCATTER_SIMT_BUILD`，
未传参数时使用环境变量，否则默认 1；只接受 0/1。
Host 在 `initialize()` 读取一次，之后修改环境变量不影响当前实例；需 shutdown 后重新初始化。
初始化日志包含 `SCATTER_OPTIONS ... simt_build=... build=parallel_simt`，
UB 对照为 `build=parallel_ub`，关闭 fused 时为 `build=serial_two_pass`。
flags 通过现有 176 B 参数的 reserved 字传入，使用新版标签并独立保留 profile bit；
不改 Python/C++ 方法签名或计时窗口。
本次需同时重编译 NPU Host 和 kernel，以识别 reserved 中的 SIMT 位。
之后切换组合不需重编译；本次 SIMT 改动不要求更新 CPU 服务。
旧 zero/profile-only 参数仍选择两遍清单、单 bank；构建/测试时应始终使用配套 Host 和 kernel。

针对 8192 requests、4 核的大表场景，直接在完整 E2E 中比较 SIMT 和现有 UB 构建。
`simt_build` 写入 JSON config 和 plan 中的命令，样本 CSV 仍保持 32 列；A/B 使用不同输出目录。
build_notify 的起止点仍是 inputs_ready 到真实 notify 发布，包含 VF、清单写回和各核完成等待。
以 build_notify 的 p50/p99 是否低于 50 us、以及 E2E 是否降低验收，不能只依据指令数判断收益。

```bash
# 重编 Host 与 kernel；两轮对照使用同一份二进制、同一 CPU 服务配置。
cmake --build build-npu -j
for simt in 0 1; do
  DEVICE_ID=7 KVAF_CPU_SERVICE_ADDR=127.0.0.1:28990 \
  bash tests/run_remote_cpu_scatter_miss_count_perf.sh "$PWD/build-npu" \
    --token-sizes 656,4096 --miss-counts 8192 --core-counts 4 \
    --fused-build 1 --double-buffer 1 --simt-build "$simt" \
    --warmup 100 --iterations 100 --case-timeout 1800 \
    --npu-eids 000000000000020000100000df00e100 --peer-ub-device udmac0d1e2 \
    --output-dir "results/scatter-build-simt-${simt}"
done
```

所有运行时组合均由 loader 预留相同的 64 KiB 动态共享 UB；payload 单 bank 使用前 32 KiB，
开启 fused build 时构建阶段会复用前 41 KiB。
因此 reference 是同一二进制下的算法对照，不等价于重编译一份仅声明 32 KiB UB 的旧二进制。
如需评估最终代码体积、编译布局或 UB 资源差异，另保留旧版本独立比较。

### SIMT 预检异常与 UB 启动契约

PID 8066 的异常位于 `BuildSourcesSimt<true>_simt_entry+0xf8`，
多个核均报告 `VEC_ERROR=0x200000`、`MTE_ERROR=0`，返回 507035（Vector Core 执行异常）。
CPU 当时仍是 last_sequence=0，随后控制 READ 报 status=9（ACK 重试超时）；应先定位设备端 VF，
不能仅将 CPU 的错误当作 payload 通信故障。

加入 SIMT 时，旧纯 SIMD loader 的 `dynUBufSize=0` 未同步修改，导致手工地址访问的 UB
没有对应的共享内存预留。现由 Host/kernel 共用 `kScatterLocalMemoryBytes=64*1024`，
loader 固定申请 65536 B，包括 `--simt-build 0` 在同一混合二进制中的对照路径。
SIMT 还使用 UB 中的编译器预留区和 DCache，内存划分见
[毕昇编译器内存约束](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/programug/Ascendcopdevg/asc-devkit/docs/zh/guide/programming_guide/compilation_and_execution/operator_compilation/bisheng_compiler.md)。
这个修正处理了明确的启动配置遗漏，具体故障指令和硬件恢复仍需以设备日志和复测确认。

必须重编并部署 NPU Host/binding 与 kernel；只替换 kernel .o 不会修复 loader 配置。
新 Python 进程应打印 `SCATTER_KERNEL_READY ... dyn_ubuf_bytes=65536`，
以及 `SCATTER_OPTIONS ... build=parallel_simt`。
预检改为同步 BatchGet，使首个错误带上 Native mode/sequence/phase；正式样本的计时方式不变。
先用小源表完成同样的完整 CPU→HBM→scatter 正确性检查，避免每次排障重新准备百 GB 源表：

```bash
cmake --build build-npu -j
DEVICE_ID=7 KVAF_CPU_SERVICE_ADDR=127.0.0.1:28990 \
bash tests/run_remote_cpu_scatter_miss_count_perf.sh "$PWD/build-npu" \
  --single --token-bytes 656 --miss-count 8192 --aiv-cores 4 \
  --table-capacity 8192 --tokens-per-block 4 --simt-build 1 \
  --warmup 1 --iterations 3 --case-timeout 180 \
  --npu-eids 000000000000020000100000df00e100 --peer-ub-device udmac0d1e2 \
  --output-dir results/scatter-simt-ub-check
```

该配置用于快速确认完整正确性；性能验收继续使用原始大表配置。
Linux loader 回归会实际调用生产 loader 并检查动态 UB，使用不依赖 CANN 的模型构建：

```bash
cmake -S . -B build-model -DBUILD_TESTING=ON -DKVAF_BUILD_PYTHON=OFF \
  -DKVAF_ENABLE_AIV_BATCH_GET=OFF
cmake --build build-model -j --target kvaf_sequential_loader_test
ctest --test-dir build-model -R kvaf_sequential_loader_ --output-on-failure
```

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
      --fused-build "$fused" --double-buffer "$double" --simt-build 0 \
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
4096/4097 项地址表缓存边界、1,277,952 项大表的稀疏访问和原址更新、尾批和大行 fallback、旧 sequence 复用、
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

profiled 阶段默认采集 Host 分段，结果写进 JSON，CSV 保留五列 Host 分段：
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

1. 预分配并复用设备 source、请求 Tensor、output、stream 和 changing 模式的 CPU 源缓冲区。
   CPU Tensor 直接引用这份缓冲区；每轮按 16 MiB 分块生成随机字节并原地覆盖，
   避免为整个源数据构造巨大的 Python int/bytes/bytearray 临时副本。
   分块按 4 B 对齐，保持相同 seed/epoch 下与原单次 randbytes 相同的字节流。
   默认 changing 模式
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
JSON config 的 `source_generation_chunk_bytes` 记录随机数据生成块大小，
它与 URMA 单条 WR 大小无关，不改变通信分片或带宽分子。

例如 `--blocks 4096 --tokens-per-block 128 --token-bytes 4096` 的源数据为 2 GiB，
而 `--miss-count 4096` 的单次有效载荷为 16 MiB。旧版在 prepare 中调用
`randbytes(2 GiB)`，Python 3.11 内部将 17179869184 位传给 C int 参数的 getrandbits，
触发 `OverflowError: Python int too large to convert to C int`；这发生在 BatchGet 之前。
分块修复保留全部源数据刷新、seed、请求规模和逐轮校验，不靠缩小测试规模规避。
大源数据重复更新仍有实际 setup 成本，运行较多轮时可按需要调大原 wrapper 的
`KVAF_SCATTER_PERF_TIMEOUT`（整个进程的总超时，默认 600 秒）。

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
| `aiv_scatter_ns` | 各核搬运提交、bank 复用等待及最终 MTE 收尾时长累计的最大值；排除 Ready/done 轮询 | scatter 服务时间对应的有效带宽；异步 MTE3 也可在轮询时执行，此值不是硬件 DMA 忙碌时长 |

CSV 使用固定的 **34 列**，baseline-only、profiled-only 和混合采样的表头/顺序一致。
相对旧 32 列新增 `pipeline` 和 `aiv_ready_wait_us`，原始周期信息留在 JSON。
所有耗时只写 `_us`，保留 3 位小数；带宽为 GB/s，保留 6 位小数。
通信计时与带宽分子保持原定义；scatter 改为累计实际搬运区间，适配流水模式。
JSON 保留原始 ns、完整指标、每核报告和完整测试配置。

| CSV 分组 | 保留列 |
| --- | --- |
| 标识与规模（11 列） | `phase`、`iteration`、`sequence`、`cores`、`fused_build`、`double_buffer`、`pipeline`、`request_length`、`miss_count`、`token_bytes`、`payload_bytes` |
| 核心耗时（12 列） | `end_to_end_us`、`cpu_comm_us`、`aiv_build_notify_us`、`aiv_scatter_us`、`cpu_write_us`、`aiv_body_us`、`aiv_wall_us`、`aiv_input_wall_us`、`aiv_notify_to_done_us`、`aiv_scatter_tail_us`、`aiv_ready_wait_us`、`call_outside_aiv_us` |
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
`aiv_body` 是最大单核执行区间；`aiv_scatter` 是最大单核实际搬运累计时长。
不能将各阶段单独取最大值后相加；流水模式尤其不能把 scatter 加到 notify→done 上。

`cpu_comm_ns` 包含 CPU posting、波次间间隔和完成轮询处理；流水模式另含四路 Jetty 的
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
汇总、每次原始样本、CPU ns 和每核 AIV cycle。CSV 保存固定 34 列样本，方便绘图。

`loop_effective_gbps` 使用整轮 wall time，包含 Python 循环开销；profiled 轮还
包含逐次 snapshot 的 D2H/转换开销。整轮还包含输入准备、BatchPut、表准备、
NPU 消费和完整校验，因此只是整个测试程序的 wall 吞吐，不是 BatchGet 带宽。
业务性能比较使用 `end_to_end_ns.effective_gbps`（汇总）和 `end_to_end_gbps`（每次）。

执行算法也要区分：`pipeline=0` 的 AIV 先观察整批 done 再 scatter；`pipeline=1` 和
AICPU 参考路径均按 CPU 的 1024 行 chunk Ready 消费，允许 CPU WRITE 与 scatter 重叠。
两者的具体执行开销仍应实测比较。同步调用完成始终表示整个有效前缀已完成。

profiled 轮新增 CPU 本地计时、每核 192 B V2 报告和一个 56 B 控制 WRITE。
当前将报告 WRITE 与 8 B done WRITE 链接后一次提交，仅等待尾部 CQ；done 使用强放置顺序，
保证报告先于完成标记可见，避免依赖 64 B WRITE 原子性，也避免单独等待报告的一次往返。
该控制链不计入 `cpu_comm_ns`，但计入 AIV 等待和该轮端到端时间。
snapshot D2H 不计入单次计时。`instrumentation_overhead_included=true` 标记该轮，
不要把它作为关闭打点后的基线性能。热路径不打印逐请求计时日志。
`snapshot_ns` 在 JSON 中单独记录回读/转换耗时；CSV 记录 sequence 便于对应 JSON 原始请求。

## 小包控制与 Host 路径优化

`pipeline=0` 的 scatter 在 Request 的已有 reserved 位声明 `kBatchDoneOnly`，CPU 保持原有 workspace
偏移，但不再写 AIV 不读取的逐 worker/chunk/Jetty Ready 进度。CPU 直接将有效前缀
分配给四条 Jetty，每条尾 WR 强放置排序并产生 CQE；所有 payload 尾 CQE 到齐后才提交
最终 done。构建 ready、错误传播、超时、整批 done 和 scatter DMA 完成等待均保留。
未声明该位的消费者仍使用原有进度协议，写计划缓存同时区分 count 与该模式。

整批模式下，若 `max_sge×token_bytes <= message_bytes`，每个 token 对应一个 source SGE，
直接顺序读取当前 descriptor 并刷新 SGE 地址，省去逐项 Source 映射表与二次索引。
大 token 的分段和旧进度模式继续使用完整映射。每次调用都更新地址，不缓存请求内容。
payload WR 按提交批次轮转四条 Jetty，使其他队列尽早开始；每次默认提交 128 条 WR，
仍只在每条队列的最后一条 WR 上请求完成，全部尾 CQE 到齐后才发布 done。
每条队列在本轮第一次提交前刷新自己的 SGE，随后立即提交，让先提交队列的 DMA
与后续队列的地址刷新重叠。已提交队列的 SGL 在完成前保持不变，不再先刷新完四条队列才开始传输。
`cpu_comm_ns` 仍从实际首批提交前计到最后 payload CQE，期间后续队列的刷新计入
`cpu_post_ns` 和 `cpu_comm_ns`。因此 `cpu_write_ns - cpu_comm_ns` 不再代表全部 SGE 刷新工作，
对比时以端到端、`cpu_write_ns` 和 notify→done 的总耗时判断收益，不能只看该差值降低。

可在启动 CPU 服务时设置 `KVAF_CPU_PUSH_POST_BATCH_WRS`（1..8192，默认 128），
不改变 kernel ABI 或 NPU Python 接口。修改后需重启服务；该变量应设置在 CPU 服务进程上。
例如 8192 项、max_sge=13 的 656/1024 B 请求都是 632 条数据 WR：batch=32 时 20 次 post，
batch=128 时 8 次，batch=256 时 4 次。WR/SGE 的硬件上限与 SQ credit 预算没有放宽。
减少提交次数不等于同比提高带宽，需对比端到端结果和 JSON 的 cpu_post_ns/cpu_cq_wait_ns。
`CPU_PUSH_CAP` 记录实际 max_sge、message_bytes、post_batch_wrs 和 optimized；
性能运行应为 optimized=1，使用常规 Release 构建，显式 Debug 配置不会被程序强制覆盖。

直接 SGL 路径另提供实验开关 `KVAF_CPU_PUSH_PAYLOAD_COMP_ORDER=all|tail`，
由 CPU service 在初始化会话时读取，默认 `all` 保持原行为。`tail` 仅清除普通 payload
WR 的 `comp_order`；这些 WR 仍为 `RELAX_ORDER`，每队列最后一条 WR 仍为
`STRONG_ORDER + comp_order=1 + complete_enable=1`，所有尾 CQE 到齐后才发布 done。
旧进度模式的强序 Ready WR、控制 READ、profile/done WRITE 均保持原完成顺序。
这不启用 Jetty 的 outorder_comp，也不改为 NO_ORDER；强序尾对前序 RELAX/STRONG 的等待保留。
语义依据为 [URMA API Guide](https://docs.openeuler.org/zh/docs/24.03_LTS_SP3/unifiedbus/unifiedbus/urma/URMA%20API%20Guide.ch.html)
的 `place_order/comp_order` 说明。该位能否改善随机 SGL 的设备调度必须在硬件上 A/B 验证，
不能依据本地替身测试认定吞吐提高。传输测试额外逆序执行无完成顺序约束的 payload WR，
验证强序边界、完整数据、suffix guard、ready/report/done 和错误处理。

纯 WRITE 单测和 BatchGet 共用该开关；CPU 日志 `CPU_PUSH_CAP` 记录实际
`payload_comp_order`。A/B 只切换 CPU service 上的该变量并重启，保持相同输入、CPU/NUMA、
post_batch_wrs 和 NPU 产物，并保留对应服务日志。只需重编 CPU service，无需修改 kernel、
协议或 payload 布局，也不引入 CPU gather。默认 `all` 不变，`tail` 通过端到端和完整数据校验
后再决定是否采用；`cpu_comm`、`cpu_write` 与 AIV 计时边界不变。

```bash
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j --target kvaf_cpu_service kvaf_cpu_push_transport_test
ctest --test-dir build-cpu -R kvaf_cpu_push_transport_test --output-on-failure
KVAF_CPU_PUSH_POST_BATCH_WRS=128 \
  ./build-cpu/runtime/kvaf_cpu_service --listen 0.0.0.0 --port 28990 --gather-threads 8
```

1024 项、4 核、max_sge=16、message_bytes=65536 时，656 B 和 4096 B 均为 64 条
payload WR：移除 16 条 Ready WR 后，总数从 80 减到 64，post 仍为 4 次，payload 尾 CQE
仍为 4 个。WR 减少不代表带宽按比例增加，需看完整调用实测。

CPU 在观察到新 notify 后，将 header 和上次 count 对应的 descriptor 前缀合并 READ。
相同/更小 count 时不再单独读 descriptor；首次或 count 增长时补读缺少的尾部，较大
请求仍按设备 message_bytes 分段。每次都重新读取地址内容，不缓存地址；支持每轮
更换源和同指针内容变化。notify 轮询仍保留。稳态小包由 notify/header/descriptor
三步变为 notify/合并前缀两步，不会把整个 capacity 的 65536 个地址都读回来。
分段 READ 现在链接后批量提交，仅尾 WR 请求 CQE 并使用强放置顺序、完成顺序；
一批最多使用 SQ depth 数量的 WR，等该批全部完成后才复用 SGE 或处理下一批。
8192 项、4 核、64 KiB message 时，合并前缀占 65984 B，两个 READ 一起提交，
由原来两次串行 CQ 等待改为一次。首次/count 增长时仍在检查 header 后补读缺少的地址。

合并 READ 计入 JSON 的 `cpu_header_ns`；`cpu_descriptor_ns`
仅记录 count 增长时的补读，稳定 count 时为 0。CPU 通信仍为 payload 首次 post 至末尾
CQE 的完整窗口；Ready 在新模式中不再参与。profile 报告和最终 done 在该窗口之外，
但仍在 `aiv_notify_to_done_us` 和端到端计时之内。

Host 复用 Python 方法名，使用 CPython method vectorcall 读取当前 Tensor 指针/尺寸，
避免重复构造绑定方法与调用参数元组；显式 stream 只取一次 `npu_stream`。不缓存
Tensor 指针或尺寸，不依赖固定输入对象，不增加 PyTorch C++ 依赖。方法签名不变。
默认 stream 路径缓存 `torch.npu.current_stream` 查询函数，省去逐次 import/属性预检；
每次仍调用查询函数并读取返回对象的当前 `npu_stream`，支持 current-stream 上下文切换。
kernel 提交、各核 status 回拷、一次 stream 同步和执行结果检查仍是同步调用的必要步骤。

这轮固定开销优化需重编并重启 CPU service，同时重编 NPU binding；AIV kernel 和协议 ABI 不变。
原生 Transport 测试覆盖分段 READ 的一次尾完成、超过 SQ depth 的分批、地址刷新与提交交错、
变化地址的数据结果以及 CQ 失败后禁止复用。端到端正确性测试另覆盖默认/显式 stream 切换。

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

## 与 direct-READ miss-count 扫描对齐的独立测试

新增 `tests/run_remote_cpu_scatter_miss_count_perf.sh` 和
`tests/python/test_aiv_scatter_miss_count_perf.py`。原有 changing-data 测试
`tests/run_remote_cpu_scatter_perf.sh` / `test_aiv_scatter_perf.py` 保留。
新入口走现有 CPU SGL WRITE→HBM staging→AIV scatter，没有新增传输实现。

新入口复用 `test_npu_batch_get_perf.py` 的参数默认值、环境变量、排序函数和请求生成函数。
direct-READ 原入口只将原有请求生成代码提取为共享函数，随机数调用顺序、分布和测试行为不变。
对照源为 `tests/run_remote_cpu_batch_get_miss_count_perf.sh`：

| 场景项 | 新测试与参考测试共同设置 |
| --- | --- |
| request-length | 32768，独立于有效 miss-count |
| table-capacity | 16384×78 = 1277952 个有效 block |
| tokens-per-block | 128 |
| 源数据 | 一次性分块 BatchPut 填满表；复用同一确定性 token pattern |
| 请求 | 每次重新生成整个容量的索引/offset；只执行 miss-count 前缀 |
| 默认源分布 | 从整个 table 随机取 block、随机取 block 内 token |
| 默认输出 | 连续行；也支持 random 输出和源地址排序 |
| warmup / iterations / seed | 3 / 100 / 20260903 |
| token 扫描 | 128, 256, 512, 656, 1024, 2048, 4096 B |
| miss-count 扫描 | 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |

远端表不因请求量较小而自动缩减。默认 656 B 配置需要 107307073536 B DDR，
4096 B 需要 670014898176 B；NPU 端只保留固定大小的填充 chunk 和请求/output。
`--dry-run` 会打印每个配置的确切 DDR 大小，不导入 torch/kvaf，也不创建结果目录。
如指定较小 `--table-capacity`，direct-READ 与 scatter 两边应使用同一个值。

拓扑约束单独记录：本实现的 CPU payload 固定 4 条 Jetty，支持 1..8 个 AIV 核，
新入口默认扫描 1/2/4/8 核，共 7×9×4 = 252 个配置。参考中的 16/32 核不支持；
其每核 Jetty 1..64 的轴不能等价映射成这条 CPU-push 通路。
新测试固定 Host 的 `--jettys-per-core=2`，该参数用于初始化/BatchPut，
每行汇总另写 `cpu_jettys=4`，避免误解为 `aiv_cores×jettys_per_core` 条 payload 队列。
仅支持单 EID；可以指定 `--peer-ub-device` 固定 CPU 设备，不做 adaptive UB 校准。
`max-inflight-total` 是 direct READ 参数，新报告明确记录其不约束 CPU WRITE 通路。

默认 `--completion-mode reference` 使用与参考相同的调用与计时顺序：
`begin → batch_get(六个位置参数，默认 stream/异步返回) → submitted → torch.npu.synchronize() → completed`。
`end_to_end` 是 begin→completed，包含真实 kernel 和全设备同步；请求生成/预同步、
status 验证和 snapshot 在窗口外。每轮测量后必须 drain 并检查所有核 status，
最后一轮 warmup 也在首个测量前 drain，避免上轮状态回拷进入下一轮计时。
profiled 用现有 `scatter_profile()` 的 drain+验证契约；baseline 用
`stop_scatter_profile()` 的 drain-before-disable 契约，此时打点始终关闭。
验证失败直接失败，不产生成功样本，不继续扫描。

`--completion-mode validated` 保留一次 `batch_get(..., synchronize=True)` 的完整同步口径，
将 status 回拷/检查计入 end_to_end，不再额外做全设备同步。此口径用于与原 changing-data
测试比较，不能和 reference 口径混算。原 changing-data 入口的计时及默认行为不变。
JSON、plan 和汇总均记录 completion_mode。输出/count 与设备地址表的 HBM 分配顺序也按
参考保留：先分配 output/count，再填表，最后创建 NPU 地址表。

报告同时提供 `kernel_total`（最大单核主体时长）和 `kernel_wall`（跨核最早 start→最晚 finish），
便于对照设备区间；它们各自遵循两个 kernel 原有的打点边界。direct 的 finish 在
业务 status 写入之后，scatter 的 finish 在报告/status 刷新之前，因此这里只是同样的
max-per-core/wall 聚合，不能把两个内部阶段当作完全相同的执行窗口。
CPU post/CQ 的耗时保留在 JSON，不把它们重命名成 direct AIV 的 post/completion 阶段。

每个配置先做确定性 active-prefix/guard 校验，正式采样结束后再校验最后一轮随机 payload；
逐轮改变源数据和全 output/guard 的强回归仍由原有 changing-data 测试覆盖。
默认与参考扫描一样开启 kernel profile；scatter 路径另采集 CPU 时间并在 done 前传回报告，
因此两种实现的打点成本不相等。JSON 记录 instrumentation_overhead_included 和
cpu_profile_report_before_done；比较无打点的业务性能时两边均运行 baseline。
Host 额外 RAII 打点默认关闭；需要归因时显式传 `--host-profile`。
`--no-kernel-profile` 可运行 baseline；参考的 miss-count shell 会
强制开启 kernel profile，baseline 应改用它的单场景 `run_remote_cpu_batch_get_perf.sh` 对照。
`reference` 模式的 JSON `host_call_ns` 表示 enqueue 调用，`device_sync_ns` 表示其后
的全设备同步，两者相加等于 end_to_end_ns；汇总对应 host_call/device_sync 两个阶段。
`host_python_dispatch` 按 host_call 减 binding 时长，不会把设备等待误算成 Python 派发。
CSV 的 `host_stream_sync_us` 仍只表示 Native 内部 stream 同步：Host 打点关闭时留空，
reference 模式开启 Host 打点时该值为 0，不能与外部 device_sync 混淆。
窗口外的验证代价记录为 snapshot_ns 或 status_validation_ns。

沿用已配置的 CPU 服务、EID 和 CANN/PyTorch 环境：

```bash
# 先查看计划和内存规模；不访问设备。
bash tests/run_remote_cpu_scatter_miss_count_perf.sh "$PWD/build-npu" \
  --dry-run --token-sizes 656,4096 --miss-counts 32,1024 --core-counts 4

# 单配置：保持参考的大表和 128 tokens/block 默认值。
bash tests/run_remote_cpu_scatter_miss_count_perf.sh "$PWD/build-npu" \
  --single --token-bytes 656 --miss-count 1024 --aiv-cores 4 \
  --json results/aligned-t656-m1024.json --csv results/aligned-t656-m1024.csv

# 全部 252 个支持的配置；允许复用目录并覆盖同名结果。
bash tests/run_remote_cpu_scatter_miss_count_perf.sh "$PWD/build-npu" \
  --output-dir results/scatter-miss-aligned

# 相同场景的 baseline；也可同时使用 --single 或缩小扫描轴。
bash tests/run_remote_cpu_scatter_miss_count_perf.sh "$PWD/build-npu" \
  --no-kernel-profile --token-sizes 656,4096 --miss-counts 1024 --core-counts 4 \
  --output-dir results/scatter-miss-aligned-baseline

# 完整同步 API 口径及额外 Host 归因；与默认 reference 结果分开分析。
bash tests/run_remote_cpu_scatter_miss_count_perf.sh "$PWD/build-npu" \
  --single --token-bytes 656 --miss-count 1024 --aiv-cores 4 \
  --completion-mode validated --host-profile
```

端点默认值沿用参考 wrapper：`KVAF_CPU_SERVICE_ADDR=127.0.0.1:28999`、`DEVICE_ID=0`，
已有环境变量会覆盖。原有 scatter wrapper 的默认端口/设备不同，运行前传入同一个服务和设备。
系统计数器频率与参考 shell 扫描使用同样优先级：`KVAF_BATCH_GET_KERNEL_CLOCK_MHZ`
（缺省 1000）覆盖单场景 PERF 时钟环境，避免两边周期换算使用不同频率。
shell 也像参考扫描一样在命令末尾固定该时钟值；通过此环境变量设置两边的频率，
传给 shell 的 `--kernel-profile-clock-mhz` 不会覆盖它。单独运行 Python 入口时仍可用该 CLI 参数。
默认扫描轴与参考一样不被 token/miss-count/core 的单值环境变量缩窄；可显式用列表或
CLI 单值筛选。profile 默认值也按参考正式扫描强制开启，明确的 `--no-kernel-profile` 才关闭。
`KVAF_REMOTE_CPU_PERF_TIMEOUT` / `--case-timeout` 控制每个配置的超时，默认 600 秒；
单配置和扫描均使用独立子进程。失败、超时或不完整报告会停止后续配置，保留已经完成的汇总。
这个预算覆盖 initialize、完整表 population、preflight、warmup、测量和 teardown，不能将
`SCATTER_ALIGNED_SWEEP_FAILED timeout=600` 直接解释为一次 BatchGet 通信超时。
子进程启用无缓冲输出，日志记录各阶段 begin/end，填表每 5 秒报告一次进度；
超时/失败会回显日志末尾 40 行。这些日志均在单次 BatchGet 的计时窗口之外。

默认大表在 4096 B、128 tokens/block 时分配 `670014898176 B`（624 GiB）CPU DDR，
656 B 时为 `107307073536 B`。THP 模式会先 memset 整个区域，URMA 注册随后执行；
CPU 日志的 `REGION_ALLOCATE_BEGIN`、`REGION_MEMORY_READY` 和 `REGION_ALLOCATED` 可区分
内存准备与注册，后者记录 memory_us/register_us/total_us。此为测试准备，未计入 E2E 样本。
自动 EID 的失败候选可能导致释放后再分配整张表；已验证可用的路由应明确固定。
例如当前 Device 7 与 udmac0d1e2 的组合可按下列配置复测（重编 scatter kernel；
CPU 新增准备耗时日志还需重编并重启 CPU 服务）：

```bash
cmake --build build-npu -j --target kvaf_scatter_batch_get_kernel
DEVICE_ID=7 KVAF_CPU_SERVICE_ADDR=127.0.0.1:28990 \
bash tests/run_remote_cpu_scatter_miss_count_perf.sh "$PWD/build-npu" \
  --token-sizes 656,4096 --miss-counts 1024 --core-counts 4 \
  --npu-eids 000000000000020000100000df00e100 --peer-ub-device udmac0d1e2 \
  --case-timeout 1800 --output-dir results/scatter-sparse-table
```

1800 秒是扩大整例准备预算，不改变 URMA/AIV 的内部超时，也不保证通信故障会自行恢复。
保持完整大表需要足够的 CPU 内存；若需缩表，对照分支也必须使用相同 table-capacity。
大表 fused build 只 DMA 读取本次有效请求用到的地址条目，不扫描完整表，不逐项等待标量 GM 读取。
reference（fused_build=0）保留原实现用于对照。指令和刷新数量的下降不能直接当作实测加速比，
收益应比较相同配置的新旧 E2E 和 build_notify 样本。

每配置保存 `.log`、原始 `.json`、固定格式的 `.csv`；扫描额外保存 `plan.json` 和
按配置/阶段逐行排列的 `summary.csv`。汇总为 15 列：配置、completion_mode、stage、mean/p50/p90/p99_us、
effective_GBps、p50_GBps 和原始 JSON 路径。带宽分子始终为 `miss_count×token_bytes`；
effective_GBps 按总字节/总耗时，p50_GBps 按有效字节/p50 耗时。
CPU/AIV 细分字段仅在 profiled 样本存在；baseline 不生成虚构的阶段耗时。
按总队列数对照时，direct 侧应选 1核×4 Jetty、2核×2 Jetty 或 4核×1 Jetty，
分别对照 scatter 相同核数的固定 4 条 CPU Jetty；8 核 direct 至少 8 条队列，无法等量匹配。
plan/JSON 的 queue_matched_direct_jettys_per_core 给出这个匹配值（不适用时为 null）。
这仅对齐队列数量，CPU owner 与 AIV 发起线程的执行位置和调度仍由实现决定。

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

## 通信与 scatter 重叠

默认 `KVAF_SCATTER_PIPELINE=1`。两个现有性能入口均支持 `--pipeline 0|1`，
命令行优先于环境变量，Host 在初始化时读取；无需重新编译来切换。
它与 `--double-buffer` 独立：前者重叠 CPU→HBM 通信与 scatter，后者重叠 AIV 内部
GM→UB 和 UB→output 的两条搬运流水线。业务 BatchGet 接口和 176 B 启动参数不变。

```text
pipeline=0:
  构建清单/notify → CPU 写完整批 staging、全部尾 CQ、最终 done → 全量 scatter → 返回
pipeline=1:
  构建清单/notify → CPU 各 Jetty 写 payload、分别 Ready → 后续 chunk → 全部尾 CQ、done
                                 AIV 按已 Ready 分片消费，跨片保持 UB bank → 等 done 与 MTE3 → 返回
```

复用现有 CPU V3 协议，chunk 为每个 CPU worker 最多 1024 行。每条 Jetty 写完自己负责的
chunk payload 后，用 STRONG 放置顺序写该 worker 的 Ready 字；AIV 扫描当前 chunk 中尚未消费的
Jetty，哪路 Ready 的 sequence/进度匹配，就先读哪路的精确 staging 分片。
不等待其他三路，也不固定先等 queue 0；一次缓存行刷新覆盖四个 Ready 字。
1024 行完整 chunk 在四路上各 256 行，因此消费粒度缩小，无需增加 Ready WR。
Ready 可跳过多个已完成 chunk，
不会退回去等待已经被后续进度覆盖的值；旧 sequence 被忽略，未来 sequence 和错误 done 拒绝。
CPU 的 worker 分区按 staging 缓存行对齐，AIV 保持原有均衡分区，对两种分区求交后消费，
因此 1/19/8193 等不整齐请求数与 idle core 也适用。

所有 payload 尾 CQ 和最终 done 仍保留。每个分片 EnqueueRows 只在 bank 复用前等待该 bank，
跨分片、跨 chunk 保留 bank 状态；等待下一分片 Ready 或最终 done 时，先前的 MTE3 可以继续执行。
在成功返回或异常退出前统一等待仍未回收的 MTE3，未完成的 bank 不能被重新用作输入。
最后一个 Ready 不能替代最终成功结果；错误完成不会被当成 PASS。空请求继续直接完成。
现有 CPU 服务已经支持该进度协议，CPU 不增加 gather、数据打包或额外 payload 拷贝。
Ready 是链内 8 B 强序 WRITE，无额外逐块 Host 往返或逐块 CQ 等待；最终 CQ/报告计时不移出样本。
Ready 增加 WR 和强序边界，因此通信耗时可能变化，收益以相同配置的端到端结果为准。

CPU 端也恢复流水模式的快速 SGE 刷新：每个连续 payload 区间只保存一个
`{local_sge, first_descriptor_row, count}` 段，提交前顺序刷新整段 SGE 地址；Ready SGE 不参与刷新。
大 token 需要跨 WR 分段时继续保留精确的 row/offset 映射。只复用写计划，每次调用都重新读取
descriptor 地址，不能复用上轮地址内容；首条队列提交后继续刷新后续队列，保持 SGL 生命周期。

必须重建 AIV kernel 和 CPU service，并重启 CPU service 才能同时获得两端优化。
656/4096 B 常见配置的 CPU 初始化日志应出现 `source_refresh=contiguous_runs_v1`；
大 token 回退时显示 `source_refresh=fragment_map`。仅更新 NPU kernel 不会修复旧 CPU 服务的刷新路径。
从上一版流水升级时启动 ABI、192 B V2 profile 和 CPU V3 wire ABI 均不变；
更早的 128 B profile 仍需配套重建 Host/binding，旧 magic 会被拒绝。
workspace 的 request/Ready/descriptor/staging 偏移未变。

CSV/JSON 的新含义：

- `pipeline`：实际启用模式；CSV 由每核报告核对，baseline 记录测试配置。
- `aiv_scatter_us`：每核各分片提交/搬运依赖等待与最终 MTE3 收尾的时长累计后取最大值，
  排除 Ready/done 轮询；`aiv_scatter_gbps` 用整次有效 payload 除以该服务时间。
  异步 MTE3 可以在轮询期间继续执行，这不是完整 DMA 忙碌时长，也不是物理 HBM 总读写带宽。
- `aiv_ready_wait_us`：每核等待下一个可消费 Jetty 分片以及最终 done 的累计时长，取最大值。
  idle core 也可能参与这个最大值；这是诊断量，不能直接加到墙钟分解上。
- `aiv_notify_to_done_us`：core 0 notify 到观察到最终 done；期间允许 scatter。
  `done_cycle` 在等待及 Jetty 分片边界采样，反映观察时刻，无法精确还原搬运期间发生的 done。
- `aiv_scatter_tail_us`：core 0 观察 done 到最后一个核 finish 的剩余尾段。
  其中可以只剩其他核的搬运，也可能主要是控制收尾；不能当成全量 scatter 时长。
- JSON `workers` 另有 `scatter_start_cycle`、`scatter_finish_cycle`、`scatter_cycles`、
  `scatter_chunks`、`ready_wait_cycles`。`scatter_chunks` 现在计实际消费的 Jetty 分片数，
  不是 CPU 1024 行 chunk 数；首尾时间跨度包含中间的 Ready/done 等待。
  JSON config 用 `pipeline_consumer=jetty_ready_persistent_ub` 标识本版流程；CSV 列数保持 34。
  CPU 与 AIV 绝对时间戳不可跨时钟域相减。

墙钟仍按下式拆解。`cpu_comm` 与完整 `aiv_scatter` 均为重叠子区间，不能再额外相加：

```text
end_to_end = call_outside_aiv + aiv_wall
aiv_wall = aiv_input_wall + aiv_build_notify + aiv_notify_to_done + aiv_scatter_tail
```

在 NPU 机器上保留现有 CPU service、EID、UB device 和工作负载参数，先跑现有正确性测试，
再在同一份二进制上对照两个模式（其余参数与原测试保持一致）：

```bash
cmake --build build-cpu -j --target kvaf_cpu_service kvaf_cpu_push_transport_test
ctest --test-dir build-cpu -R '^kvaf_cpu_push_transport_test$' --output-on-failure
cmake --build build-npu -j
# 重建 CPU 服务后，使用原有启动参数重启该服务，再运行下面的 NPU 测试。
KVAF_SCATTER_PIPELINE=1 python -u tests/python/test_aiv_scatter_batch_get.py

for p in 0 1; do
  bash tests/run_remote_cpu_scatter_miss_count_perf.sh "$PWD/build-npu" \
    --single --token-bytes 656 --miss-count 8192 --aiv-cores 4 \
    --pipeline "$p" --json "results/overlap-p${p}.json" --csv "results/overlap-p${p}.csv"
done
```

用相同 token/miss-count/核数比较各自 E2E 最快 25% 样本的 `end_to_end_us`、`aiv_wall_us`、
`aiv_scatter_tail_us`，同时检查 `cpu_comm_us` 是否因 Ready 序列明显增加。
同时比较 `cpu_write_us-cpu_comm_us` 的准备/收尾差值是否下降，避免流水恢复后丢掉 CPU 快速路径。
656 B 与 4096 B 分别比较；不能把通信带宽提高视为本次优化的前提或验收结论。
现有 changing-data E2E 和 reference-aligned 测试均保留完整输出校验，未新增独立 scatter 性能测试。

CPU 原生模型检查可在有 C++ 编译器的 Linux 机器运行：

```bash
cmake -S . -B build-model -DBUILD_TESTING=ON -DKVAF_BUILD_PYTHON=OFF -DKVAF_ENABLE_AIV_BATCH_GET=OFF
cmake --build build-model -j --target kvaf_sequential_host_test kvaf_sequential_batch_get_kernel_model_test
ctest --test-dir build-model -R 'kvaf_sequential_(host_test|batch_get_kernel_model_test)' --output-on-failure
```

模型逐条 Jetty 发布 Ready，检查 AIV 的每次 payload DMA 只读取对应 Jetty 已就绪的分片，
并故意等到 AIV 消费本片才发布下一路 Ready，同时测试 queue 0 最晚到达。
检查跨 Ready 边界仍有 MTE3 在途，以及缺失 Ready/未来序号/最终错误时仍回收全部 DMA；
现有完整输出与 guard 校验保留。CPU transport 模型另在 Ready WR 执行前校验该路分片已写完，
并覆盖连续段刷新、changing descriptor、多波次和分段 token 的数据正确性。
它验证控制流与范围，硬件 DMA 可见性、CANN 编译和性能仍需上机验证。
