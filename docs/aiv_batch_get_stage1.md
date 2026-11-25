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
Gather 默认使用 128 KiB byte-prefix；可信输入超过每核 1024 行时，在首段清单发布后通知 CPU，
剩余清单通过 suffix 通知完成。`KVAF_AIV_GATHER_CHUNK_BYTES=0` 保留整批 ACK 对照。
AIV 不调用 UMDK；两端通过注册内存和现有 CPU 控制协议通信。

Scatter 使用 64 KiB UB stage ring 执行 GM→UB→GM，通过 MTE3→MTE2 事件安全复用 bank。
精确长度的 `DataCopyPad<uint8_t>` 用于读取 packed staging 和逐行写入目标。
包含 SIMT VF 的当前二进制由 loader 设置 `dynUBufSize=65536` 预留共享 UB；
旧纯 SIMD loader 的零动态 UB 配置不适用于这个混合二进制。
GM→UB 普通模式恢复、MTE2/MTE3 事件同步及退出屏障保留。

### 连续 staging 与流水供数

N 端设置 `KVAF_SCATTER_CPU_GATHER=1`，CPU 将离散源数据 gather 到连续注册 DDR staging，
再通过 payload Jetty WRITE 到连续 HBM staging，AIV scatter 到请求指定的输出行。
完整流程如下：

1. AIV 读取请求的源索引、offset 和输出行索引，SIMT 生成源地址清单。core 0 将首段清单与
   notify WRITE 到 K 端 mailbox，随后发布后缀清单；K 端可以在后缀清单到齐前开始处理首段。
2. K 端 Gather 线程将离散 CPU 源数据复制到连续注册 DDR staging。分片复制完成并发布
   就绪状态后，提交线程可以提交该分片已准备好的 payload WRITE。
3. 各 payload Jetty 将 CPU staging 写入连续 HBM staging。同一 Jetty 内按顺序发布
   payload 和对应 Ready；提交线程发送已就绪分片时，Gather 线程可以继续复制后续分片。
4. AIV 根据 Ready 消费已到达的 HBM 分片，经 GM→UB→GM 搬运到离散输出行。
   启用独立 Jetty 游标后，一个 Jetty 等待下一分片不会阻止其他 Jetty 消费已就绪分片。
5. K 端回收全部 payload CQE 后发布整批 Done。AIV 检查 Done 成功且完成所有输出 DMA
   后返回；Ready 用于提前消费，Done 用于确认整批请求结果。

K 端设置 `KVAF_CPU_GATHER_INTERLEAVE=1`，统一开启交错 Gather 和每 Jetty 独立供数。
该开关默认 0，与 N 端 `--pipeline` 独立；线程不足或边界未对齐时保留回退路径。
Gather→WRITE 的交错发生在不同就绪分片之间；一个分片的 payload WRITE 在该分片
Gather 完成后提交。三个阶段可以同时处理不同分片。

| Scatter pipeline | CPU interleave | 行为 |
| --- | --- | --- |
| 0 | 0 | 整批 Gather 后 WRITE，AIV 等 Done 再 scatter |
| 0 | 1 | Gather 与 WRITE 重叠，AIV 等 Done 再 scatter |
| 1 | 0 | 连续行分波供数，AIV 按 Ready scatter |
| 1 | 1 | 交错 Gather、每 Jetty 独立提交，AIV 按 Ready scatter |

`--jetty-cursors 1` 让各 Jetty 的消费游标独立推进，`--overlap-head-cq 1` 让后缀清单构造
与首段控制 WRITE 的 CQ 等待重叠。二者默认 0。
Scatter 默认使用 4 条共享 payload Jetty；`--jettys-per-core N` 或
`KVAF_SCATTER_JETTYS_PER_CORE=N` 将总数设为 AIV 核数乘以 N，上限 32。
CPU Gather 使用服务的 `--gather-threads`、绑核和等待配置，staging 按需增长并复用。
`KVAF_SCATTER_CPU_GATHER=0` 时，CPU 使用源地址 SGL 直接 WRITE 到 HBM staging，
省去 CPU staging Gather；Scatter pipeline、Jetty 游标、head CQ 重叠和 ACL 图仍可单独配置。
两种通路的 HBM staging 均使用 `ACL_MEM_MALLOC_HUGE_FIRST`。

## 独立二进制与类型声明

以下路径相对于 `access_operator/algorithms/get/direct/urma_aiv/`：

| 层次 | Scatter | Gather |
| --- | --- | --- |
| Host owner | `host/scatter_batch_get.cpp` | `host/gather_batch_get.cpp` |
| Loader | `host/execute_scatter_batch_get_loader.cpp` | `host/execute_gather_batch_get_loader.cpp` |
| 参数 ABI | `include/scatter_batch_get_abi.h`，176 B | `include/gather_batch_get_abi.h`，256 B |
| AIV 源码 | `kernel/execute_scatter_batch_get_aiv.asc` | `kernel/execute_gather_batch_get_aiv.asc` |
| Symbol | `KvafScatterBatchGetAivKernelV1` | `KvafGatherBatchGetAivKernelV10` |
| 二进制 | `kvaf_scatter_batch_get_kernels.o` | `kvaf_gather_batch_get_kernels.o` |

Scatter 参数末尾使用版本化选项字段。176 B 参数和 64 B 业务状态之外，计时使用独立的
每核 192 B 区域。默认四条 payload Jetty 的 AIV Scatter mailbox 使用 V8 打开，握手响应为
392 B；显式配置非四 Jetty 数量时使用 V9 扩展打开，响应为 2296 B。启动和停止命令使用 V8。
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

- 支持 1～8 核和单 EID。普通 Scatter 入口使用 eager 执行；图捕获使用下面的
  通用 Get graph lease。
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

## Scatter ACL 图模式

性能脚本统一使用 `--acl-graph`，每张图的 `--graph-calls` 默认 8。
一次捕获多次 BatchGet，一次 replay 后统一同步；同一 stream 上按序执行，共享传输工作区。
每个节点有独立输入清单，每次 replay 前在计时窗口外刷新；输出按顺序复用，保留末次结果。
不启用图模式时，每个样本仍只执行一次 BatchGet，`--graph-calls` 不参与计时。
图模式在捕获结束后直接绑定 `graph.replay` 和原生 stream 同步方法；样本内通过
`aclmdlRIExecuteAsync` 重放，不再逐次解析 Python stream 或经过 Python 转发函数。
`--completion-scope stream` 使用 `aclrtSynchronizeStream`，`device` 保留原有整设备同步。

`graph_total_us` 是整图提交到同步返回的总耗时，`end_to_end_us` 是总耗时除以
`calls_per_sample`；带宽使用整图有效字节数除以整图总耗时。例如每图 8 次调用、20 个计时
样本执行 160 次 BatchGet，不含预检和 warmup。P50 表示每图平均耗时的中位数，
不等于单独提交一次 BatchGet 的 P50。图准备、输入刷新和状态检查在样本计时之外。
启用 kernel profiling 时，多节点图的阶段数据保存在 `last_call_*` 和 JSON 的
`last_call_profile`，只描述末次调用；平均值的 `aiv_wall_us` 和 `call_outside_aiv_us` 留空。

推理框架可通过 `runtime.graph_resources()` 把 Scatter 节点接入调用方已有的图。
显式管理节点时，使用 `begin_get_graph(...)` 预备每个节点的 lease，
再将 `graph_lease=lease` 传给 `batch_get` 或 `prepare_scatter_batch_get`。
准备发生在 capture 之前。同一 Runtime 的 Scatter 节点共享工作区，须使用同一张图和同一个 stream。

每次 replay 后通过 `check_batch_get_status(stream, graph_lease=lease)` 检查整张 Scatter 图的
累计序号和最终状态。任一节点失败会传递到后续节点。图、stream、Tensor 和 AccessTable 的存储须保持
有效且地址不变。销毁 ACL 图后释放 lease，失败的图使用 `discard_result=True` 清理。
profiling 开关须在准备图资源前设置。

普通 prepared Scatter 支持 `wait_stream=True`，在调用内等待 DMA 完成并推迟状态校验。
它与 `synchronize=True` 互斥。图节点捕获时两者均为 false。
无图性能对照可选 `--completion-scope native-stream`，图模式使用 `stream` 或 `device`。

最小框架接入流程如下。`input_plans` 为 8 组固定地址的输入，Scatter Runtime、表、输出和 stream 已初始化。

```python
def forward():
    for indices, offsets, output_indices in input_plans:
        runtime.batch_get(
            view, indices, offsets, count, output_indices, output, stream=stream)

resources = runtime.graph_resources()
graph = kvaf.AclGraph()
discard = True
try:
    with resources.prepare(stream):
        forward()
    with resources.capture(stream):
        graph.capture_begin(stream)
        forward()
        graph.capture_end()
    for _ in range(iterations):
        refresh_inputs_in_place()
        graph.replay()
        graph.synchronize()
        runtime.check_batch_get_status(stream, graph_lease=resources.leases[-1])
    discard = False
finally:
    if graph.capturing:
        graph.capture_end()
    if graph.ready:
        graph.synchronize()
        graph.reset()
    resources.close(discard_result=discard)
```

图只捕获一次，随后多次执行 `aclmdlRIExecuteAsync`；每次重放后同步并检查整图状态。
无参原生入口要求 replay 前输入生产任务已提交并建立正确的 stream 依赖；性能脚本通过
计时窗口外的 `torch.npu.synchronize()` 保证输入就绪。需要保留原有 stream getter 提交屏障时，
仍可使用 `graph.replay(stream)`。捕获的 stream 对象由图持有，reset 后无参入口会拒绝重放。

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

miss-count 脚本默认使用 prepared 提交；`--acl-graph` 启用多节点图，完成后按所选范围同步，
并将 profile 快照放在计时区间之外。

### 当前实测最优配置

下面是当前测试使用的 N 端配置：4 核 AIV、每图 8 次 BatchGet、pipeline、独立 Jetty
游标和 head CQ 重叠，同时启用 kernel/host profiling。CPU 服务地址按实际部署设置。
该配置是当前环境的实测选择，其他机器上的核数、Jetty 数和 Gather 线程数需结合实测确定。

```bash
export PYTHONPATH="$PWD/build-npu/python"
export KVAF_RUN_NPU_TESTS=1
export DEVICE_ID=0
export KVAF_SCATTER_CPU_GATHER=1
export KVAF_CPU_SERVICE_ADDR=141.62.38.71:28890
export KVAF_BATCH_GET_KERNEL="$PWD/build-npu/bin/kvaf_scatter_batch_get_kernels.o"
export KVAF_BATCH_PUT_KERNEL="$PWD/build-npu/bin/kvaf_execute_batch_put_kernels.o"

bash tests/run_remote_cpu_scatter_miss_count_perf.sh "$PWD/build-npu" \
  --single --token-bytes 656 --miss-count 8192 --aiv-cores 4 \
  --table-capacity 16384 --tokens-per-block 128 \
  --acl-graph --graph-calls 8 \
  --completion-scope stream \
  --pipeline 1 \
  --jetty-cursors 1 \
  --overlap-head-cq 1 \
  --json results/aligned-t656-test2.json \
  --csv results/aligned-t656-test2.csv
```

Gather 与 WRITE 的交错需要在 **K 端启动 CPU 服务前** 设置
`export KVAF_CPU_GATHER_INTERLEAVE=1`；仅在 N 端设置不会改变远端服务的配置。
Gather 线程数量、亲和性和等待方式由 K 端服务参数决定。
上述命令未指定 payload Jetty 数，因此在没有 `KVAF_SCATTER_JETTYS_PER_CORE` 环境覆盖时
使用 4 条共享 payload Jetty。

每次 BatchGet 的有效数据为 `8192 * 656 = 5,373,952 B`，每图为 `42,991,616 B`。
端到端耗时按整图耗时除以 8 计算，带宽按整图有效数据量除以整图耗时计算。
kernel profiling 只保留末节点的阶段数据，不能将末节点 `last_call_aiv_wall_us` 从整图单次
平均耗时中相减，得到平均 `call_outside_aiv_us`。阶段耗时定位使用末节点 profile；
单次 eager 的完整耗时拆解使用下面的无图对照。

### 单次 eager 对照

保留相同数据布局、Gather、流水、核数和 profiling 配置，只移除 `--acl-graph --graph-calls 8`，
使用同一 stream 完成范围：

```bash
bash tests/run_remote_cpu_scatter_miss_count_perf.sh "$PWD/build-npu" \
  --single --token-bytes 656 --miss-count 8192 --aiv-cores 4 \
  --table-capacity 16384 --tokens-per-block 128 \
  --completion-scope stream \
  --kernel-profile --host-profile \
  --pipeline 1 --jetty-cursors 1 --overlap-head-cq 1 \
  --json results/aligned-t656-eager.json \
  --csv results/aligned-t656-eager.csv
```

加 `--dry-run` 可查看配置。另一个入口 `tests/run_remote_cpu_scatter_perf.sh` 用于
请求和数据随样本变化的端到端性能回归。


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
