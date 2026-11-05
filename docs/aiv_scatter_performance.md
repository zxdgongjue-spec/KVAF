# AIV CPU-push scatter 性能测试

CPU 通过四条 Jetty 发起 URMA SGL WRITE，将数据写入 NPU HBM staging；AIV 将 staging
中的 token scatter 到输出行。固定采用 SIMT 多核融合清单构建和 UB/MTE stage ring。
唯一算法开关是流水掩盖：`--pipeline 0|1`，对应环境变量 `KVAF_SCATTER_PIPELINE`，
未设置时默认开启。初始化后不再读取开关，不允许改变在途请求的算法。

`--fused-build`、`--double-buffer`、`--simt-build` 及其同名环境变量已移除。
串行两遍构建、UB/DMA 构建对照和单缓冲算法已删除；JSON 用
`source_list_builder=parallel_simt`、`scatter_copy=ub_mte_stage_ring` 描述固定实现。
计时开关与请求规模、队列、源分布等测试参数继续保留。

## A5 stage 调度

参考 [DeepEP-Ascend A5 dispatch copy epilogue](https://github.com/deepseek-ai/DeepEP-Ascend/blob/main/deep_ep/include/deep_ep/impls/ep/dispatch_copy_epilogue.hpp)
的预填充、环形 refill 和 MTE3→MTE2 复用依赖。payload 继续使用 MTE2/MTE3；
随机输出仍逐行发出精确长度的 MTE3 命令，不依赖输出行连续，也不使用 SIMT 搬 payload。

- 每批最多 32 行，先预取可用 stage，再消费并 refill。stage 复用由 MTE3→MTE2
  事件排在 DMA 流水线上，Scalar 不再逐批等待 MTE3→S。
- 保持 64 KiB UB 和原有每批行数；能容纳三个 payload bank 及各自 128 B 行号缓存时
  使用三 stage，否则使用双 stage。656 B token 的 UB pitch 为 672 B：
  `3 × 32 × 672 + 3 × 128 = 64896 B`。
- UB 有余量时，output_indices 每批 DMA 到独立行号缓存，在 UB 上读取；
  仅等待该批行号到达 Scalar，payload 到达由 MTE2→MTE3 事件约束。
  行号走 DMA 的路径不再 Flush 整个输出索引前缀。4096 B 等占满双 bank 的行宽保留
  Scalar GM 行号读取；超过 32 KiB 的行仍走精确分块 fallback。
- `pipeline=1` 只在已发布 Ready 的 Jetty shard 内预取，状态跨 shard 保留。
  返回前仍等待最终 done，并通过 MTE3→MTE2→S 排空所有 bank；异常退出也排空。

JSON 记录 stage 上限、UB 大小和复用事件方向，CSV 及计时口径不变。
这些改动减少 Scalar 等待和 GM 行号读取，不能据模型推断具体硬件加速比。
验收继续使用两套完整 BatchGet；先对比 pipeline=0，再对比 pipeline=1，
固定 CPU/UB 端口、table-capacity、请求计划及核数，656/4096 B 都需要测试。
业务随机输出须显式指定 `--output-access-pattern random`（changing 入口为
`--output-order random`），避免默认连续输出掩盖真实开销。
在同轮最快 25% 的端到端样本上比较 `end_to_end_us`、`aiv_scatter_us`、
`aiv_scatter_tail_us` 和 `cpu_comm_us`，不要分别挑各阶段最快值。

## 构建与运行

CPU 服务、NPU Host 和 AIV kernel 必须一起重编并重启服务（PeerLink 协议升级为 V7，
OpenCpuPushGet 响应增加 CPU mailbox segment/token）；混合 SIMD/SIMT kernel 的 loader 始终申请 65536 B 动态 UB，
kernel 类型必须为 AIV。接口参数仍为 176 B，生产输入沿用 trusted-input 契约。
CPU 源内存默认保留 THP 分配；NPU workspace 保留 `ACL_MEM_MALLOC_HUGE_FIRST`，
大页不足时可回退普通页。显式 HugeTLB 模式、CPU tail-only completion-order 试验、
额外启动 READ/WRITE 探测、ACK 调参和专用诊断 RPC 已删除。

```bash
cmake --build build-cpu -j --target kvaf_cpu_service
cmake --build build-npu -j
```

重新启动 CPU 服务和 NPU 进程后测试。指定 C1 端口可在 CPU 服务上使用
`--ub-device udmac1d1e2`，测试侧使用 `--peer-ub-device udmac1d1e2`。
CPU 核和内存 NUMA 位置由启动服务时的 `numactl` 策略控制，peer 候选编号不代表物理 CPU。

## 两套端到端场景

保留 changing-data 回归入口：

```bash
DEVICE_ID=7 KVAF_CPU_SERVICE_ADDR=127.0.0.1:28990 \
bash tests/run_remote_cpu_scatter_perf.sh "$PWD/build-npu" \
  --cores 4 --blocks 1024 --tokens-per-block 4 --token-bytes 656 \
  --pipeline 1 --iterations 100 --profile-rounds 100 \
  --json results/scatter.json --csv results/scatter.csv
```

一个样本是一次同步 BatchGet。默认 full-blocks 为 1024 block × 4 token/block × 656 B，
有效 payload 2686976 B。`--request-length` 是输入容量，`--miss-count` 是本次有效 token 数；
`--workload one-token-per-block` 复现每 block 一个 token 的请求语义。
changing 模式在计时外更新源数据、请求和表，计时后验证完整输出与 guard。
`--input-mode steady` 复用表；`--profile-iterations 0` 关闭可选打点。
`--source-pattern random|sequential`、`--output-order random|sequential` 和
`--output-row-step 1|2` 控制访问分布；随机大源表按块生成，避免 Python randbytes 溢出。

参考 direct-READ miss-count 场景的稳态入口：

```bash
DEVICE_ID=7 KVAF_CPU_SERVICE_ADDR=127.0.0.1:28990 \
bash tests/run_remote_cpu_scatter_miss_count_perf.sh "$PWD/build-npu" \
  --single --token-bytes 656 --miss-count 8192 --request-length 32768 \
  --tokens-per-block 128 --table-capacity 1277952 --aiv-cores 4 \
  --pipeline 1 --warmup 30 --iterations 100 \
  --json results/scatter-aligned.json --csv results/scatter-aligned.csv
```

该入口共享 `test_npu_batch_get_perf.py` 的参数和请求生成器，对齐
`tests/run_remote_cpu_batch_get_miss_count_perf.sh`。
默认从全表随机取 block、再随机取 block 内 token；每轮重新生成请求。
源表一次性填满，正式计时不含初始化、BatchPut、表准备、请求生成及预同步。
默认 table-capacity=16384×78、tokens-per-block=128；
656 B 源区为 107307073536 B（约 99.94 GiB），每次 miss-count=8192 仅传 5373952 B。
缩减 table-capacity 同时改变 CPU 源访问范围，对照实现应保持相同配置。

常用对齐参数：

| 参数 | 默认/限制 |
|---|---|
| `--request-length` | 32768，输入 Tensor 容量 |
| `--miss-count` | 有效 token 数；不超过 request-length，最大65536 |
| `--table-capacity` | 1277952 个 block |
| `--tokens-per-block` | 128 |
| `--source-access-pattern` | random、contiguous、shuffled-contiguous |
| `--output-access-pattern` | 默认 contiguous，也支持 random |
| `--sort-input-indices` | 按解析后的源地址排序；排序在 BatchGet 计时外 |
| `--aiv-cores` | 1..8；CPU payload 始终4条 Jetty |
| `--jettys-per-core` | 固定2，仅用于 Host setup/BatchPut |
| `--pipeline` | 默认1；设0比较全批 done 后 scatter |
| `--kernel-profile/--no-kernel-profile` | 对齐入口默认开启 AIV/CPU 打点 |
| `--host-profile/--no-host-profile` | 默认关闭 Host 细分打点 |
| `--host-submit` | reference（默认原调用）或 prepared（固定 Tensor 存储和 stream） |
| `--completion-mode` | reference 或 validated |
| `--case-timeout` | 整个 case，包含初始化、填表、采样和拆除 |

不带 `--single` 时扫描 token-sizes、miss-counts、core-counts，在 output-dir 下输出分组 CSV/JSON
与 summary.csv；`--dry-run` 只打印计划，不访问硬件。
只支持单 NPU EID 和固定 peer-ub-device；不做 adaptive UB 设备校准。
固定4条 CPU Jetty不能等价解释为 aiv-cores×jettys-per-core 条 payload 队列。

## 固定 Host 提交

固定存储的业务调用可以在初始化后准备一次无参数提交函数：

```python
submit = runtime.prepare_scatter_batch_get(
    view, indices, offsets, miss_count, output_indices, output,
    stream=stream, synchronize=False,
)
# 每次先在同一个 stream 上更新输入内容，再提交。
submit()
torch.npu.synchronize()
```

准备阶段解析七个 Tensor 指针、容量并选定 stream 对象，返回函数持有 Runtime、Tensor 和
stream 的引用。逐次提交不再调用 Tensor 方法、构造当前 stream 对象或派发多参数重载，
仍走原有 Runtime/算子/kernel 提交路径。每次提交仍读取固定对象的 `npu_stream` 属性：
torch_npu 的该 getter 会排空框架 Host 任务队列，保证此前的 `zero_`、`copy_` 等
操作先于裸 ACL kernel 提交。不能把这一步当成普通字段访问而缓存成原始句柄。
该顺序保证沿用框架的 dispatch fence，不增加逐次全设备同步。
`miss_count`、请求数组和表内容可原地更新；
Tensor 存储和尺寸须保持固定，提交函数及表保持有效直到完成。更换存储、表或 stream 时重新准备；
shutdown/reinitialize 后旧提交函数失效。原有 `batch_get` 继续逐次解析当前输入和 stream。

对齐性能入口增加 `--host-submit prepared`，原 reference 模式和默认参数保留。
prepared 模式仍使用相同随机请求生成器，每轮将新计划原地写入固定输入 buffer，
更新与预同步在计时外；计时内仍只有一次提交及原来的全设备同步，snapshot/校验位置不变。
准备耗时记录为 JSON `setup_us.prepare_submit`，不将准备或多次调用冒充一次提交。
JSON 用 `host_submit`、`input_storage`、`stream_selection` 和
`host_timing_matches_reference` 明确区分调用方式，CSV 保持32列。
开启 Host profile 后 prepared 的 `host_tensor_metadata_us` 为0；
`host_stream_resolve_us` 仍记录必要的 Host 任务队列 fence 和句柄取得，不能要求其为0。
JSON 的 `stream_selection=fixed_stream_per_call_task_queue_fence` 标识这一实现。
对应机制见 torch_npu 的 [npu_stream 属性](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/npu/Stream.cpp#L78)
及 [NPUStream::stream 队列处理](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/core/npu/NPUStream.cpp#L442)。

使用同一组参数分别运行 `--host-submit reference --pipeline 0`、
`--host-submit prepared --pipeline 0`、`--host-submit prepared --pipeline 1`，
保持 `--completion-mode reference --host-profile`，每组写入各自 JSON/CSV。
先比较 `host_call_us`、`host_python_dispatch_us`、`host_launch_us` 和
`call_outside_aiv_us`，再比较端到端及 scatter 尾部；比较同组最快25%的样本。

## 流水和完成语义

pipeline=0：所有 payload 尾 CQE完成，CPU 发布 done，随后 AIV scatter。
pipeline=1：CPU 按每核1024行 chunk 组织四条 Jetty，分别发布各 Jetty shard 的 Ready；
AIV 优先消费已就绪 shard，UB 双缓冲跨 Ready 边界保持 MTE3 搬运在途。
两种模式都必须等待最终匹配序号的 done，并完成全部输出 DMA 后才成功返回。
Ready 表示某个 shard 已可读，不代表整次请求成功。

上述流水覆盖本次 CPU WRITE 与 scatter；本次提交前的 Host 工作仍在启动关键路径上。
请求N运行时可提前准备请求N+1的 Host 参数；进一步让N+1的CPU传输与N的scatter
并行，则需要独立 workspace、状态/完成序号和输入输出生命周期。当前逐次 Drain
会串行化提交，因此不能直接取消逐次同步或把多次提交的平均耗时称为单次 BatchGet
延迟。固定 Host 提交不改变这项约束。即使请求内重叠充分，最后一个 Ready shard
的scatter及最终完成收尾仍可能形成尾部。

CPU payload 恢复全 WR `comp_order=1`；每条链保留强序尾 WR 和真实尾 CQE，
ready/done 的必要强序保证继续保留。不得删除最终完成等待或提前发布成功。
CPU 的 WR/SGL 计划复用，每次刷新真实源地址。四队列按提交批次交错启动，
仅刷新当前批次的 SGE/Ready 源字，已经提交的部分保持不变直到 CQ 完成。
`KVAF_CPU_PUSH_POST_BATCH_WRS`（CPU 服务环境变量，默认128）现在按 payload WR 计数，
对应 Ready WR 附在该批次尾部，不单独占用预算。
例如 max_sge=16、8192×656 B 时，每队列128个 payload WR +8个 Ready 只提交一次，
四队列从8次 post 降为4次；实际 WR 数仍受硬件 max_sge、消息上限和 SQ wave 约束。

## 请求控制通路

AIV 多核完成清单构建后，由核0利用双向 Jetty 0 的 USER_CTL SQ 发送：

```text
WRITE 64 B 请求头 + WRITE 有效源地址清单（按协商消息上限分片）
→ 同一 SQ 的强序 WRITE 发布 8 B notify
→ CPU acquire 轮询本地 notify，直接读取本地头和清单
→ CPU SGL WRITE payload / Ready → 最终报告 / done → AIV 排空 scatter
```

头、清单、notify 分成互不重叠的传输，不覆盖 CPU 拥有的 done/Ready/marker 区。
小请求通常一批提交三个控制 WR，只确认该批强序尾 CQE；较大清单按有界批次发送。
清单和控制 SQ 编码共用已保留的 UB，发送完成后再交给 payload stage ring。
CPU 空闲轮询和每次请求解析不再发出远端 READ，减少原来的请求发现及清单拉取往返。
仍保留队列完成、错误、超时和最终 done；无流水和流水共用此通路，不新增算法开关。

## 计时和带宽口径

CSV 为32列；删除恒定的 fused_build、double_buffer 两列，耗时统一 µs，带宽为十进制 GB/s。
JSON保留原始纳秒、worker cycle、CPU post/CQ等待和配置。
有效带宽始终为 `payload_bytes / elapsed_ns`，没有额外乘8，也不将本地读+写流量翻倍。

| 字段组 | 含义 |
|---|---|
| phase、iteration、sequence、cores、pipeline | 样本标识和执行配置 |
| request_length、miss_count、token_bytes、payload_bytes | 请求容量、有效token数、token字节、实际有效载荷 |
| end_to_end_us / end_to_end_gbps | 一次 BatchGet 的完整调用/同步窗口 |
| cpu_comm_us / cpu_comm_gbps | 首个 payload WRITE 提交到最后尾 CQE 被观测 |
| cpu_write_us / cpu_write_gbps | CPU WriteBatch 整体，包含计划复用处理和地址刷新 |
| aiv_build_notify_us | 输入就绪到核0提交最后 notify 所在批次，包含多核构建、汇合及控制 WR 编码/提交；大清单还含先前控制批次等待 |
| aiv_notify_to_done_us | 最后控制批次提交到核0观测最终 done，包含 mailbox 传输/尾CQ、CPU准备、payload传输和完成发布 |
| aiv_scatter_us / aiv_scatter_gbps | 最大单核scatter活跃耗时，流水模式下排除等待Ready的区间 |
| aiv_body_us / aiv_body_gbps | 最大单核主体耗时 |
| aiv_wall_us / aiv_wall_gbps | 跨核最早start到最晚finish的设备跨度 |
| aiv_input_wall_us | 最早start到最晚inputs-ready的跨度 |
| aiv_scatter_tail_us | 全局done之后仍未完成的scatter尾部跨度 |
| aiv_ready_wait_us | 最大单核等待Ready/done耗时，与通信等设备阶段存在重叠 |
| call_outside_aiv_us | end-to-end减AIV wall，包括提交、包装和设备窗口之外的同步等 |
| host_tensor_metadata_us | Tensor元数据解析/必要参数绑定 |
| host_launch_us | kernel提交 |
| host_status_copy_enqueue_us | 状态回拷提交 |
| host_stream_sync_us | Host同步调用；包含尚未完成的设备执行 |
| host_python_dispatch_us | 绑定计时外的包装、派发与返回差值 |

`cpu_comm` 还包含首个提交后后续队列的地址刷新/提交及完成观测延迟，不能直接等同物理线速。
`cpu_header_ns`、`cpu_descriptor_ns` 为兼容历史 JSON 保留，mailbox 路径没有控制 READ，值为0。
控制开销转入上述 AIV 窗口，不能将这两个零值理解为控制通信不耗时。
`aiv_notify_to_done_us - cpu_comm_us` 仍是控制、CPU准备和完成发布的残差，
不能把以前约38 µs全部当成可删除时间，也不能只看该残差变小就认定端到端受益。
Host和设备时钟不能跨域相减；各核计时不能求和。
pipeline=1时CPU通信、Ready等待、scatter活跃区间重叠，不能将这些列直接相加。
做关闭流水的近似层级拆解可用：

```text
端到端
├─ 调用外围
└─ AIV wall
   ├─ 输入
   ├─ 构建清单并通知
   ├─ 通知→done（内部包含CPU通信）
   └─ done之后的scatter尾部
```

对齐入口 reference：异步 BatchGet后全设备同步，完成后在采样外drain/验证status。
validated：同步API，状态回拷/检查计入端到端，不额外全设备同步。
profile snapshot及最终payload验证均在采样外；打点本身的报告写回开销仍在设备流程中。
Host profile用于解释外围；其stream-sync耗时不能再与AIV时长叠加。
最终收益应同时比较无打点端到端与相同打点配置下的阶段变化。

## 校验

保留preflight有效前缀/guard验证、最后一轮随机payload验证与正常CQ/超时/失败处理。
changing-data入口继续覆盖逐轮改变源数据的完整回归。

```bash
python3 -m unittest discover -s tests/python -p 'test_scatter*py'
python3 -m unittest discover -s tests/python -p test_cpu_write_bandwidth_config.py
ctest --test-dir build-npu -R 'kvaf_sequential_' --output-on-failure
ctest --test-dir build-cpu -R kvaf_cpu_push_transport_test --output-on-failure
```

独立纯CPU WRITE带宽入口保留，参见 [CPU WRITE 带宽测试](cpu_urma_write_bandwidth.md)。
旧的优化探索和计时审计文档属于历史记录，其已移除开关不再适用于当前运行命令。
