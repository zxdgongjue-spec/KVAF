# AIV BatchGet 第一阶段实现与验证

Gather 当前实现与配置见 [Gather 实现与配置](aiv_gather_prepared_prefix.md)，
性能与排障见 [Gather 性能与排障](aiv_gather_serial_profile.md)。
部署使用最新 Gather V10 / args 256 B，并同步更新 CPU 服务、NPU Host 和 kernel。

## 数据通路

默认 BatchGet 模式为 `direct_read`。总体设计见
[三阶段设计](aiv_batch_get_scatter_gather_design.md)。

| 模式 | 执行顺序 |
| --- | --- |
| `cpu_push_scatter` | AIV 多核 SIMT 生成清单并 WRITE 到 CPU mailbox → 多条 Jetty WRITE payload → AIV 按 Ready scatter 到目标 HBM → 等待最终 done 和 DMA 排空 |
| `gather_read` | AIV 流式发布清单、command、notify → CPU 按连续前缀 gather → AIV 随 ready prefix 提交多 SGE READ → 最终 ACK 与 CQ 完成 |

Scatter 默认按 Jetty 分片的 Ready 提前消费。设置 `KVAF_SCATTER_PIPELINE=0` 后，CPU 省略
分片 Ready，AIV 等到整批 done 成功再开始搬运。两种模式下，可信请求超过 1024 个有效行时，
内核都会先发布首段清单，再合批发布剩余清单。
N 端可用 `--overlap-head-cq 1`（`KVAF_SCATTER_OVERLAP_HEAD_CQ=1`）让 core 0 在首段
WRITE 在途时构造剩余清单。首段 CQE 成功后才发布后缀。默认值 `0` 先等首段 CQE，
再构造剩余清单。对照时固定 gather、Jetty cursor 和请求分布，比较 AIV 墙钟、
CPU `WriteBatch`、`suffix_wait_ns` 及端到端延迟。
Gather 默认使用 128 KiB byte-prefix；可信输入超过每核 1024 行时，在首段清单发布后通知 CPU，
剩余清单通过 suffix 通知完成。`KVAF_AIV_GATHER_CHUNK_BYTES=0` 保留整批 ACK 对照。
AIV 不调用 UMDK；两端通过注册内存和现有 CPU 控制协议通信。

在 `pipeline=1` 下，设置 `KVAF_SCATTER_JETTY_CURSORS=1` 或为性能脚本传入
`--jetty-cursors 1`，可让每核在所覆盖的 worker 范围内按 Jetty 独立推进消费。
默认值 `0` 按 chunk 消费。同一 chunk 内相邻且已就绪的分片仍可合并搬运。
两组可直接对比端到端耗时和 AIV 墙钟。profiled CSV 中的 `aiv_cross_chunk_shards`
统计本批跨过尚未消费的早期 chunk 提交的分片数。

Scatter 默认在每个会话创建 4 条共享 payload Jetty。N 端在初始化前设置
`KVAF_SCATTER_JETTYS_PER_CORE`，或给两个 Scatter 性能脚本传入 `--jettys-per-core`，
可按 `AIV 核数 × jettys-per-core` 设置共享 Jetty 总数，上限为 32。
例如 `--aiv-cores 4 --jettys-per-core 8` 会创建 32 条。每个 worker 的 Ready 区按实际
Jetty 数分配 1～4 条 cache line。非 4 条配置使用扩展建链协议，需要配套部署 K 端服务、
N 端 Runtime 和 Scatter 内核。两端的 `payload_jettys` 日志和性能输出的 `cpu_jettys`
可用于确认测试配置。

### Scatter 的 CPU Gather 回推

启动 NPU 进程时设置 `KVAF_SCATTER_CPU_GATHER=1`，并使用
`aiv_batch_get_mode="cpu_push_scatter"`。该开关默认关闭，在 Runtime 初始化时读取。
NPU 通过专用的会话打开命令选择 CPU gather，同一个 CPU 服务可为不同会话选择不同的回推路径。
启用时需部署支持该命令的 CPU 服务和 NPU Host。
CPU 按请求行顺序将离散 DDR 源复制到连续的注册 DDR staging，再通过会话的
payload Jetty 将连续 DDR 区间 WRITE 到 HBM staging 的对应连续区间。每个数据 WR
使用一个源 SGE 和一个目标 SGE，按设备报告的消息及 WRITE 长度上限拆分。
AIV 复用现有的 Ready/Done 和 scatter。CPU 启用分波且收到首段通知时，
CPU 先 gather 并回推首段，再等后缀清单。CPU 线程协作拷贝每个最多 1024 行的块，
块就绪后由服务线程提交 WRITE，其他线程继续拷贝后续块。
`pipeline=1` 时每个 AIV 分片的 Ready 随该分片的最后一部分 WRITE 发布。
`pipeline=0` 时 AIV 等最终 Done，CPU 仍可用 `KVAF_CPU_GATHER_INTERLEAVE=1`
分波 gather 和 WRITE。两端都关闭时，CPU 等完整清单并完成整批 gather 后提交 WRITE。

CPU gather 使用服务的 `--gather-threads` 和 gather 绑核、等待配置。
每个会话的额外 pinned DDR staging 按观察到的最大有效批次字节数增长，
以 4 KiB 对齐并复用到会话关闭。可分别用开关 0 和 1 启动 NPU 进程，沿用下方的
正确性回归和性能脚本对照。

Scatter 使用 64 KiB UB stage ring 执行 GM→UB→GM，通过 MTE3→MTE2 事件安全复用 bank。
精确长度的 `DataCopyPad<uint8_t>` 用于读取 packed staging 和逐行写入目标。
包含 SIMT VF 的当前二进制由 loader 设置 `dynUBufSize=65536` 预留共享 UB；
旧纯 SIMD loader 的零动态 UB 配置不适用于这个混合二进制。
GM→UB 普通模式恢复、MTE2/MTE3 事件同步及退出屏障保留。

## 独立二进制与类型声明

以下路径相对于 `access_operator/demo/aiv_batch_get/`：

| 层次 | Scatter | Gather |
| --- | --- | --- |
| Host owner | `host/scatter_batch_get.cpp` | `host/gather_batch_get.cpp` |
| Loader | `host/execute_scatter_batch_get_loader.cpp` | `host/execute_gather_batch_get_loader.cpp` |
| 参数 ABI | `include/scatter_batch_get_abi.h`，176 B | `include/gather_batch_get_abi.h`，256 B |
| AIV 源码 | `kernel/execute_scatter_batch_get_aiv.asc` | `kernel/execute_gather_batch_get_aiv.asc` |
| Symbol | `KvafScatterBatchGetAivKernelV1` | `KvafGatherBatchGetAivKernelV10` |
| 二进制 | `kvaf_scatter_batch_get_kernels.o` | `kvaf_gather_batch_get_kernels.o` |

Scatter 参数末尾使用版本化选项字段。176 B 参数和 64 B 业务状态之外，计时使用独立的
每核 192 B 区域。AIV Scatter mailbox 的打开、启动和停止命令使用 V8，握手响应为 392 B。
CPU Gather 回推使用单独的 V8 打开命令，复用 Scatter 的握手结构及启动、停止命令。
V8 由 AIV 把地址清单和通知 WRITE 到 CPU DDR mailbox，CPU 从本地读取。
AICPU Scatter 使用 V6，握手响应为 300 B，CPU 从 NPU HBM READ 通知、请求头和地址清单。
两条路径分别通过 `aiv_batch_get_mode` 和 `aicpu_batch_get_mode` 选择。Gather 和公共控制命令
使用 V6。部署 Scatter 时同步重编 CPU 服务、NPU Host 和 Scatter kernel。

原 `direct_read` 继续使用 `kvaf_execute_batch_get_kernels.o`。三个模块独立编译，
避免重复链接 SDK 全局符号。顺序模式接收标准 direct/旧 sequential 路径时，
按模式选择同目录的新 `.o`；显式自定义路径按原样加载。缺失目标模块或 symbol
直接失败，不回退到其他模式。

Scatter 必须保留函数级 AIV 类型声明，方式见
[CANN 独立 AIV 下发示例](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/commlib/commopdev/docs/en/comm_op_dev_guide/aiv_comm_op_dev/op_dispatch.md)。
`.ascend.meta.KvafScatterBatchGetAivKernelV1` 中应包含
`tag=1, length=4, value=3 (K_TYPE_AIV)`，对应小端字节 `01000400 03000000`。
仅有 section 名不能证明核类型存在。检查实际产物：

```bash
python tests/python/aiv_kernel_metadata.py build-npu/bin/kvaf_scatter_batch_get_kernels.o
```

检查器无需 CANN，支持 NOTE/PROGBITS。scatter 硬件测试也会在导入 NPU 后端前
执行该检查，避免再次加载缺少类型声明的产物。

## 使用约束

下面的初始化示例使用性能测试的 block 布局，每个源 block 存 128 个 token，每个 token
为 656 B，slot 大小为 83,968 B。`batch_get_tokens_per_block` 是源数据布局参数，
Scatter 流水仍按每个 worker 的 1024 行 chunk 切分。

```python
token_bytes = 656
tokens_per_block = 128
slot_size = token_bytes * tokens_per_block

runtime = kvaf.NpuRuntime(
    get_kernel_path,
    batch_get_token_bytes=token_bytes,
    batch_get_tokens_per_block=tokens_per_block,
    batch_put_kernel_object_path=put_kernel_path,
    npu_core_count=4,
    aiv_batch_get_mode="cpu_push_scatter",
)
runtime.link_to(cpu_service_address)
runtime.initialize(ddr_bytes, slot_size)  # ddr_bytes 为 slot_size 的整数倍
```

- 支持 1～8 核和单 EID。Scatter 支持 eager 和由 `prepare_scatter_batch_get` 管理的 ACL Graph。
  请在调用方图捕获之外准备和提交 Scatter。
- 请求容量 1～65536，固定 token 长度 1～65535 B；`miss_count=0` 是合法空批次。
- 仅搬运有效前缀，允许重复源和不连续输出行；输出行必须唯一，offset 不能越过源 block。
- Gather 每核两条 Jetty，SGE 数为 1～6；`--gather-threads` 至少覆盖其 AIV 核数。
- 同一 Runtime 复用 workspace，下一次 Get/Put 先 drain 上一批；
  `synchronize=False` 只表示入队成功，Tensor 和 stream 必须存活到执行完成。
- Scatter 在 Drain/Shutdown 检查设备状态；Gather 默认在同步调用或关闭时检查状态，
  异步逐批检查可通过 `KVAF_AIV_GATHER_CHECK_STATUS=1` 启用。执行失败后拒绝继续提交；无法确认 DMA
  停止时保留注册资源，允许重试关闭。错误输出不能作为有效结果消费。
- done 轮询有次数上限，loader 的 kernel 超时为 180 秒；轮询次数不是毫秒。
- Scatter HBM workspace 按最大 65536 请求配置，包括 `65536 * token_bytes` staging、
  512 KiB 地址清单和控制区。大 token 回归使用较小请求容量以缩小测试输出，workspace 仍按上限分配。

### Scatter ACL Graph

使用同一条 NPU stream 捕获和重放，更新固定地址 Tensor 的内容即可发送新请求。
设置 profiling 后调用 `prepare_scatter_batch_get(..., stream=stream, acl_graph=True)`，返回的
无参 callable 每次调用提交一次 replay。AIV 在 kernel 内推进序号并通知 K 端，K 端执行 Gather
和 payload WRITE，AIV 按原有 Ready/Done 协议执行 Scatter。图持有输入、输出、AccessTable 的
Tensor 和 stream，运行时会检查完成状态。用 `synchronize=True` 可在每次调用内验证结果，
异步调用可以同 stream 连续入队，在 `scatter_profile()` 或图销毁时检查状态。

prepared callable 通过专用入口提交已捕获的图，每次回放会读取一次 `npu_stream`，确保调用前
已交给 Torch 的同 stream 任务先入队。启用 Host profiling 后，`host_native_us` 记录原生提交，
包含 `host_operator_us` 和 `host_launch_us`。`host_runtime_validation_us` 记录通用 BatchGet
入口的校验，专用回放入口跳过该入口并检查图所属 Runtime 的状态。

使用 `reset_scatter_batch_get_graph()` 等待并销毁图后，可以释放 AccessTable、改变 profiling
选项或重新准备图。比较性能时，N 端性能脚本两组都用 `--completion-scope stream`，其中一组
额外加 `--acl-graph`。捕获与结果验证在样本计时窗口外。

对照 Host 完成开销时，可以把 `--completion-scope stream` 改成 `--completion-scope native-stream`。
原生 prepared callable 在同一次 Python 调用里提交并等待同一条 stream，CSV 的 `host_call_us`
此时包含等待。开启 Host profiling 时可用 `host_launch_us` 和 `host_stream_sync_us` 分开查看。
两种完成方式的单请求端到端都计到同一条 stream 执行完成。

单次 replay 的优化可用 N 端 `KVAF_SCATTER_GRAPH_MAILBOX=0/1` 做对照，默认 `0`。设为
`1` 时，首段清单发布后，Suffix 复用本次请求已校验的 mailbox。Scatter pipeline 开启时，
core 0 在 Suffix 提交后即可消费已经 Ready 的分片，在等 Ready 的间隔检查最后一个控制 CQE。
成功返回前仍确认该 CQE 和整批 Done。分段发布用于可信入口中超过 `1024` 行的请求。
`aiv_suffix_cq_blocked_us` 记录最后一个 Suffix CQE 的阻塞等待。CQE 已在 Ready 空档回收时，
该指标为零。端到端对照可先关闭 profiling，再开 profiling 对比这个计时和 `aiv_wall_us`。

### 将 Scatter 捕获到调用方的图

先在图外完成 Runtime 初始化、填表和 AccessTable 准备。如果已经提交过 Scatter，先完成状态校验。
调用方持有图和 stream，KVAF 将一个异步 Scatter 节点提交到正在捕获的 stream。图中依赖输出的
NPU 操作可以在同一 stream 排在该节点后面。

```python
capture_batch_get = runtime.prepare_scatter_batch_get(
    view, indices, offsets, miss_count, output_indices, output,
    stream=stream, capture_into_graph=True,
)
graph = torch.npu.NPUGraph()
with torch.npu.graph(graph, stream=stream):
    capture_batch_get()
    # 在这里提交现有推理图中依赖 output 的操作。

# 可以按 stream 顺序原地更新固定地址的输入。
with torch.npu.stream(stream):
    graph.replay()
runtime.check_scatter_graph_replay()  # 整图提交后同步并校验本次 KVAF 状态。

stream.synchronize()
graph.reset()
runtime.detach_scatter_batch_get_graph()
```

每个 Runtime 支持在同一张图的同一 stream 挂接多个 Scatter 节点。各节点可以使用不同的输入
Tensor、AccessTable 和输出，但节点间按 stream 顺序执行，共享该 Runtime 的传输工作区。
保留 prepared callable、Runtime、stream、Tensor 和
AccessTable，直到调用方完成回放、销毁图并 detach。每次 replay 后调用一次
`check_scatter_graph_replay()`，再开始下一轮。检查失败时丢弃该次推理结果并结束会话。
框架若在检查前写入持久状态，需要在设备侧门控这些写入，或提供回滚。
启用 Scatter profiling 时，在状态检查后调用 `scatter_profile()` 读取 AIV/K 端数据。

性能脚本使用 `--caller-acl-graph --completion-scope stream` 验证调用方捕获、回放和校验，
CSV 标记为 `caller_acl_graph`。脚本只捕获 BatchGet，状态校验在样本计时窗口外。
共享提交的收益应在包含该节点的实际推理图里测量。

### 多次 BatchGet 共用一次 Graph 提交

使用 `--caller-acl-graph --graph-calls 8 --completion-scope stream`，一次捕获 8 个
BatchGet kernel，每次 replay 统一等待整图完成。`--iterations 20` 对应 20 次 replay，正式
样本总计 160 次 BatchGet。当前实测在 8 个节点附近已取得主要摊销收益，继续增大收益较小。
`--graph-calls` 默认仍为 1，大于 1 时使用调用方图路径，可在不同设备和请求尺寸下重新对照。

每个节点有独立的源索引、偏移和输出行清单。随机请求模式下，在样本计时外刷新所有节点的
输入。输出分配按 stream 顺序复用，读回最终节点的输出。捕获前逐节点验证请求数据及 guard，
整图执行后的状态检查累计所有节点的序号，设备侧失败状态会保留到后续节点。

CSV 中 `graph_total_us` 是一次 replay 到统一等待返回的总耗时，`end_to_end_us` 是它除以
`calls_per_sample` 的每调用平均耗时。`graph_payload_bytes` 是整图有效字节数。
`graph_total_gbps = graph_payload_bytes / graph_total_ns`，与 `end_to_end_gbps` 相等。
Host 提交和同步时间也按调用数摊销。此时 P50 表示各次 replay 的每调用平均耗时的中位数。

先用 `--no-kernel-profile --no-host-profile` 测吞吐。诊断时可启用 kernel profiling，现有
设备 profile 缓冲区记录最后一个节点，CSV 用 `last_call_*` 字段输出它的 CPU/AIV 阶段。
多节点模式的 `aiv_wall_us` 和 `call_outside_aiv_us` 留空，完整的末次 profile 存入 JSON 的
`last_call_profile`。整图 Host 总耗时与单节点的 AIV profile 分别用于吞吐统计和末次请求诊断。

## 构建与完整回归

CPU 端使用与部署环境匹配的 UMDK 重编并启动服务：

```bash
cmake -S . -B build-cpu -DKVAF_BUILD_REMOTE_CPU_SERVICE=ON -DKVAF_URMA_ROOT=/path/to/umdk
cmake --build build-cpu -j
./build-cpu/runtime/kvaf_cpu_service --listen 0.0.0.0 --port 28990 --gather-threads 8
```

NPU 端沿用支持 A5 `dav-c310-vec` 的 CANN 环境，首次配置启用
`KVAF_ENABLE_AIV_BATCH_GET`、`KVAF_ENABLE_AIV_BATCH_PUT`、`KVAF_BUILD_PYTHON`。
在仓库根目录执行 Scatter 正确性回归。当前脚本固定使用 16 个源 block，每个 block
包含 4 个 token，并覆盖重复源、输出 guard 和 workspace 复用。上面的 128 token/block
对应性能测试使用的布局。

```bash
set -euo pipefail
cmake --build build-npu -j
export PYTHONPATH="$PWD/build-npu/python${PYTHONPATH:+:$PYTHONPATH}"
export KVAF_RUN_NPU_TESTS=1
export KVAF_CPU_SERVICE_ADDR=127.0.0.1:28990
export KVAF_BATCH_GET_KERNEL="$PWD/build-npu/bin/kvaf_scatter_batch_get_kernels.o"
export KVAF_BATCH_PUT_KERNEL="$PWD/build-npu/bin/kvaf_execute_batch_put_kernels.o"
# 沿用已验证的 KVAF_NPU_EIDS；CPU 在远端时将地址改为实际 CPU_IP。
mkdir -p results
DEVICE_ID=7 KVAF_NPU_CORE_COUNT=1 KVAF_NPU_TOKEN_BYTES=656 KVAF_NPU_REQUESTS=8193 \
  python -u tests/python/test_aiv_scatter_batch_get.py \
  2>&1 | tee results/scatter-e2e-c1-t656-n8193.log
```

验收要求退出码为 0，最后打印：

```text
PASS: mode=cpu_push_scatter cores=1 token_bytes=656 counts=[0, 1, 7, 19, 8193, 8193, 19, 7, 1, 0]
```

测试包含真实 BatchPut、access table、BatchGet，回读比较全部有效输出及 guard，
并覆盖重复源、倒序输出行、A/B/A 缓冲、同步/异步两种调用、两个 caller stream
连续提交和最终 Shutdown。请使用普通 Python 执行，保留 assert 校验。

建议扩展如下组合，每次使用独立进程并保存日志：

| 核数 | token 字节 | 请求容量 | 目的 |
| --- | --- | --- | --- |
| 1/3/8 | 656 | 8193 | 多核、非均匀分工、大小批次及复用 |
| 8 | 1/13/32/4096/16384/16385/65535 | 19 | 尾部、16 KiB 分块边界及最大 token |
| 8 | 32 | 65536 | 最大请求容量 |

Gather 使用 `test_aiv_gather_batch_get.py` 和对应 gather `.o`。兼容入口
`test_aiv_sequential_batch_get.py` 根据 `KVAF_AIV_BATCH_GET_MODES` 为各模式启动
独立进程。未启用 `KVAF_RUN_NPU_TESTS=1` 返回 77，属于跳过。

## Scatter 性能回归

miss-count 脚本固定使用 prepared 提交，提交后调用 `torch.npu.synchronize()`，
并将 profile 快照放在计时区间之外。以下示例使用每 block 128 个 token。

开启 CPU Gather 回推时，`cpu_comm_us` 仍从首个 payload WR 提交计到最后的
payload CQE。CPU 分波时这段时间可包含后缀 gather，`--pipeline 1` 还包含 Ready WRITE。
用 `--pipeline 0` 和 `KVAF_CPU_GATHER_INTERLEAVE=0` 可测完整 gather 后的提交到完成窗口，
并在 CPU 日志中查看 `CPU_PUSH_CAP` 的 `payload_write_bytes`。

Scatter 的 CPU gather 复用了 `aiv_gather_yq` 的整批线程预算和分段清单消费方式。
流水模式下只调度一次：各线程先拷贝首段，走到第 1024 行时才检查 suffix；
连续前缀就绪后可以立即提交 WRITE。第一个 Jetty 分片的结束位置对齐 CPU
cache line 时，会先提交这个分片，沿用原有 Ready 信号。每批仍重新读取源清单，
只复用与请求形状有关的 WRITE 计划。

在 N 端保持 `KVAF_SCATTER_CPU_GATHER=1`，用以下组合对照。K 端开关在启动服务时设置。

| N 端 `--pipeline` | K 端 `KVAF_CPU_GATHER_INTERLEAVE` | CPU 提交 | AIV scatter |
| --- | --- | --- | --- |
| 0 | 0 | 整批 gather 后提交 | 等 Done |
| 0 | 1 | 首段提前提交，交错 gather，后缀按 Jetty 独立提交 | 等 Done |
| 1 | 0 | 按连续行分波 | 按 Ready |
| 1 | 1 | 首段提前提交，交错 gather，后缀按 Jetty 独立提交 | 按 Ready，可启用 Jetty cursor |

K 端统一使用 `KVAF_CPU_GATHER_INTERLEAVE`，默认值为 `0`。设为 `1` 对应原来的
Interleave 与 Jetty supply 同时开启，独立于 N 端的 Scatter pipeline。旧的
`KVAF_CPU_GATHER_JETTY_SUPPLY` 已移除，启动脚本改为设置 `KVAF_CPU_GATHER_INTERLEAVE=1`。
`CPU_PUSH_CAP` 的 `gather_interleave` 输出配置值，Gather profile 的 `interleaved` 和
`jetty_supply` 输出本次请求实际采用的调度方式。

交错模式保留首段提前提交，后续每波优先为不同 AIV 核和 Jetty 各准备一个分片。
分片跨越首段边界或内部边界未按 cache line 对齐时，CPU 回退到按连续行分波。
CPU profile 的 `pipeline` 表示 AIV 模式，`gather_pipeline` 表示 CPU 是否实际分波，
`interleaved` 表示是否采用交错顺序，`ready_wrs` 记录逐分片 Ready WRITE 数。
同一条记录也输出 payload WR 数、提交次数和回收的 tail CQE 数。用 `post_window_ns`
（首条提交前到最后一次提交返回）和 `post_tail_ns`（最后一次提交返回到完成回收）
拆解 `cpu_comm_us`。分波时提交窗口包含供数间隔。

K 端设置 `KVAF_CPU_GATHER_INTERLEAVE=1` 会启用交错调度，后缀分片按 Jetty 独立提交。
满足 cache-line 对齐且活跃 gather 线程数至少等于 Jetty 数时，每条 Jetty 有独立的 gather 线程组，
日志记录 `jetty_supply=1`。每条 Jetty 按自身队首分片的就绪顺序提交，保留有序 Ready。
最终 Done 在全部 payload CQE 回收后发布。N 端使用 `--pipeline 1 --jetty-cursors 1` 按 Jetty 消费 Ready。
`waves` 记录 WRITE 计划提交次数，一个 gather 波可对应多个提交计划。

CQ 回收保留队列信用不足时的回收和最终排空。已删除空档 CQ 扫描实验，等待后缀或 gather
工作线程时只推进已经就绪的提交。预填 staging 和额外 Ready refresh 实验也已删除，每个
请求均读取当前源数据并正常 gather。

`max_jetty_post_gap_ns` 是同一 Jetty 相邻提交之间的最大间隔，包含在途数据仍可搬运的时间。
对照 `post_calls`、`tail_cqes`、`post_window_ns`、`post_tail_ns`、`write_ns` 和端到端 P50/P90。
Scatter profiled CSV 的 `sequence` 可与 CPU 日志对应。`aiv_tail_core` 是最后结束的核，
`aiv_tail_to_last_enqueue_us`、`aiv_tail_to_local_done_us` 和 `aiv_tail_after_local_done_us`
依次记录 core 0 观察 Done 后，最后结束的核在最后一次 scatter 入队、本核观察 Done、收尾
之间的可见时间。若入队或本核 Done 在此之前完成，对应区间为零。收尾区间含 MTE 排空和状态准备。

用 `--kernel-profile` 采集时，CPU 服务在会话关闭后输出 `CPU_PUSH_GATHER_PROFILE`。
`prepare_ns` 记录 gather 任务准备，`copy_ns` 记录任务发布、各线程的源地址校验、拷贝和汇合，
`suffix_wait_ns` 记录协调线程等待后缀清单的时间。`gather_pipeline=1` 的 `copy_ns`
包含这段等待及拷贝期间的 WRITE 提交，不能与 `suffix_wait_ns` 相加。
单独比较 CPU gather 耗时用两端开关都关闭的组合，此时 suffix 在拷贝计时前等完。
每会话保留前 512 条，`active_threads` 记录单批实际参与的 CPU 线程数。
Scatter 的一条记录覆盖整批，`aiv_gather_yq` 的记录覆盖单个 lane。比较时先对齐
总字节数、CPU 线程总预算、绑定、源表和请求分布；只有复现参考分支的固定请求基准时，
才加 `--fixed-request-plan`。

日志还记录协调线程完成自己拷贝后的 `join_wait_ns`，以及最晚完成线程的
`slow_worker`、`slow_tid`、`slow_cpu`、`slow_bytes`、`slow_ready_delay_ns`、`slow_start_delay_ns`、
`slow_copy_ns`、`slow_barrier_ns` 和 `slow_thread_cpu_ns`。
`slow_ready_delay_ns` 从协调线程进入拷贝阶段计到该线程取得任务。
`slow_start_delay_ns` 还计入该线程开始拷贝前的采样开销。
`slow_cpu_time_valid=1` 时可对照工作墙钟和线程 CPU 时间判断该线程的运行区间。
`gather_pipeline=1` 的协调线程区间包含就绪块提交，块内 barrier 也包含在 `slow_copy_ns` 中。

```bash
DEVICE_ID=7 KVAF_CPU_SERVICE_ADDR=127.0.0.1:28990 \
  bash tests/run_remote_cpu_scatter_miss_count_perf.sh "$PWD/build-npu" \
    --single --token-bytes 656 --tokens-per-block 128 --table-capacity 16384 \
    --request-length 32768 --miss-count 8192 --aiv-cores 4 --pipeline 1 \
    --warmup 3 --iterations 100 \
    --json results/scatter-aligned.json --csv results/scatter-aligned.csv
```

加 `--dry-run` 可查看配置。另一个入口 `tests/run_remote_cpu_scatter_perf.sh` 用于
请求和数据随样本变化的端到端性能回归。

### Jetty 时间戳诊断

在 K 端启动服务时设置 `KVAF_CPU_PUSH_JETTY_TRACE_SAMPLES=8`，采集每会话前 8 个带 profiling 的请求，
包含 profile warmup。默认值为 0。会话关闭后，`CPU_PUSH_JETTY_TRACE` 按 `session / sequence / jetty / plan`
输出每条 Jetty 的 WRITE 计划，按 `sequence` 关联同一会话的 `CPU_PUSH_GATHER_PROFILE` 和 N 端 CSV。

- `gather_ready_ns` 记录最后一个 Gather 写入线程完成的时间。整波计划共用同一就绪时间。
  `suffix_seen_ns` 在该请求的 Gather 日志里，记录 K 端首次采样到后缀标记的时间。无后缀时为 0。
- `post_begin_ns` 和 `post_return_ns` 分别记录该 Jetty 的首个 post 开始和最后一个 post 返回。
  `cqe_seen_ns` 是 K 端实际轮询到该计划末尾 CQE 的时间，包含等待轮询的时间。
- `pending_fences_before_post` 统计该 Jetty 提交前尚未回收的此前计划，含已完成但尚未轮询的 CQE。
  所有时间戳使用同一 K 端单调时钟。`trace_write_begin_ns` 可作请求内时间原点。

Trace 沿用正常的 gather 路径，无需固定请求清单。以下示例固定源数据和一份请求计划，方便逐条
对照提交间隔；也可使用 miss-count 脚本采集每次刷新的随机清单。

```bash
python tests/python/test_aiv_scatter_perf.py --input-mode steady --plans 1 \
  --workload sampled-tokens --blocks 1024 --tokens-per-block 128 \
  --request-length 32768 --miss-count 8192 --token-bytes 656 --cores 8 \
  --pipeline 1 --jetty-cursors 1 --overlap-head-cq 1 \
  --warmup 3 --iterations 100 --profile-iterations 20 --csv results/scatter-jetty.csv
```

从 trace 比较 `suffix_seen → gather_ready`、`gather_ready → post_begin`、`post_return → cqe_seen`。
用脚本的无 profiling 阶段对比端到端 P50/P90，从 profiled CSV 选取未记录 Jetty 时间戳的样本比较
`cpu_comm_us` 与 `cpu_write_us`。

## Gather 合并接口与回归

构造参数保留公共 Runtime 的位置参数契约（包括 `transport`）；`aicpu_batch_get_mode`、
`aiv_batch_get_mode`、`aiv_gather_sgl_size` 和 `trusted_gather_inputs` 使用关键字。
Gather SGL 范围为 1～6。`trusted_gather_inputs=True` 为默认快速路径，调用方负责正确的
Tensor dtype、连续布局、索引、唯一输出行、生命周期和 eager stream；设为 False 使用带校验路径。
普通 Gather 每次重新读取 Tensor 指针和当前 stream，不缓存跨调用的存储地址。

```python
runtime = kvaf.NpuRuntime(
    get_kernel_path, 656, 128,
    batch_put_kernel_object_path=put_kernel_path,
    transport="urma", aiv_batch_get_mode="gather_read",
    aiv_gather_sgl_size=6, trusted_gather_inputs=True,
)
```

Scatter 的 `prepare_scatter_batch_get`、profiling 和已有 `batch_get(profile=False)` 接口保留。
`profile=True` 仍只用于 AICPU 且要求同步；Gather 使用独立的
`start_gather_batch_get_profile` / `gather_batch_get_profile` / `stop_gather_batch_get_profile`。

正确性回归使用 `test_aiv_gather_batch_get.py`，建议覆盖 1/3/8 核以及 SGL 1/5/6。
性能脚本 `test_aiv_gather_batch_get_perf.py` 保留最新 NPU source buffer 复用、128 entries
批量填表、分阶段初始化日志和独立计时样本；具体环境及输出见 Gather 文档。
