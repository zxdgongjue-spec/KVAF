# Python Bindings

应用接口以[实现与使用](../docs/cache_transport_execution_implementation.md)为详细入口：
[main 接口迁移、签名及最小示例](../docs/cache_transport_execution_implementation.md#python-migration)，
[Get 图模式完整示例](../docs/cache_transport_execution_implementation.md#get-graph-example)，
[vLLM/模型框架接入位置](../docs/cache_transport_execution_implementation.md#framework-graph-integration)。
下文是绑定布局与使用约束摘要，不要求应用阅读测试脚本才能构造请求。

本目录是 KVAF C++ 核心与 Python 之间的可选绑定层，使用 Pybind11 构建 `_kvaf` 原生扩展。

`./build.sh npu` 默认将 AIV URMA、UBmem、AICPU Gather/Scatter 和 CPU/NPU Pull 构建进同一个
`build-npu/python/kvaf` 包，并自动安装 AICPU 设备包。运行时按 `transport_mode`、默认 Put/Get 配置及请求的
`execution` 选择实现；省略 AICPU 使用 `--without-aicpu`；只需 CPU Pull 或 NPU raw Pull 时分别使用
`--cpu-pull-only` / `--raw-pull-only`，不会构建普通 DDR 算子的设备内核。

`include/` 存放 Binding 内部共享头，`src/` 存放扩展实现；Python 包与构建后导入校验脚本保留在本目录。
`build.sh` 默认使用 `RelWithDebInfo`，每次构建同步 Python 模块并校验扩展导入。
运行时将所选构建目录的 `python/` 加入 `PYTHONPATH`，见 [Python 包构建说明](../python/README.md)。

当前 Binding 用于验证以下链路：

```text
C++ 编译 → Python 原生扩展 → Python 导入
```

启用 `KVAF_BUILD_PYTHON` 的便携构建提供进程内 Mock Runtime；`./build.sh cpu` 只构建
CPU 服务，不生成 Python 包。启用 `KVAF_ENABLE_AIV_BATCH_GET`、`KVAF_ENABLE_AICPU_URMA`、`KVAF_ENABLE_CPU_OFFLOAD`
或 `KVAF_ENABLE_NPU_PULL` 时，额外提供
`NpuRuntime` 与 `AclGraph`：后者是 Python-owned 的 ACL Runtime Graph 薄封装，直接
暴露 capture / replay / synchronize / destroy 生命周期，用于 Python 集成测试或上层
Adaptor 的图编排；它不保存 KVAF 请求、Tensor 或 Operator 状态。

`AclGraph.copy_host_to_device_async(destination, destination_bytes, source, stream)` 是图
回放前的原生 ACL H2D 输入更新：调用方在已 capture 的同一 stream 上，把 host staging
buffer 写入地址稳定的 NPU tensor，再调用 `replay`。`source` 的所有权仍属于 Python，必须
至少保持到同一 stream 的 `synchronize` 返回。该方法不记录 Tensor、不会分配 buffer，也不
属于 Runtime / AccessOperator 生命周期。

`NpuRuntime.access_table_view(handle, device="npu")` 返回直接可用的
`DeviceAccessTable`，其中保存 NPU `addresses` / `valid` Tensor；
`NpuRuntime.prepare_access_table` 同时接受 Python key 序列或一维、连续的 CPU
`torch.int64` Tensor；大表应使用后者，避免 Python 逐元素转换。
`NpuRuntime.batch_get(device_access_table, indices, offsets, miss_count,
output_indices, output, stream=None, synchronize=False)` 直接接收 NPU Tensor，也支持
`batch_get(addresses, valid, indices, offsets, miss_count, output_indices, output, ...)`
重载，以使用调用方持有的设备表 Tensor。它不暴露 data pointer、字节数、
output 行字节数或 AIV 的内部 status/queue buffer；output 行字节数由 output Tensor
第 0 维之后的连续元素推导，并必须与配置的 `batch_get_token_bytes`
一致。`miss_count` 必须是单元素 `torch.int32` NPU Tensor，仅请求数组的
`[0, miss_count)` 前缀参与搬运。

```text
NpuRuntime(batch_get_token_bytes=None, batch_get_tokens_per_block=None, *,
           device_id=0, transport_mode="urma", default_put=None, default_get=None,
           put_resources=None, get_resources=None, npu_eids=[], peer_ub_device="",
           cpu_pull_options=None, npu_pull_options=None)
```

两个 token 布局参数可同时省略，以使用独立 Pull；普通缓存 Get 的 provider 首次准备时要求有效布局。
显式布局必须同时为正数，每个 AccessTable entry 的大小为两者乘积。
其余参数仅接受关键字。`transport_mode` 可为 `"urma"`、`"ubmem"` 或 `"urma+ubmem"`，
默认算法由 `PutExecutionConfig` / `GetExecutionConfig` 指定，单次调用的 `execution`
可覆盖默认配置。构建支持且符合 transport 模式的全部 provider 自动可选，无 enable 列表。
未指定默认配置时优先选当前构建可用的实现；显式选择不可用实现会在首次使用时失败。

`put_resources/get_resources` 是 `(execution_config, ExecutionResources(...))` 列表。
预算按 provider 配置。`cores=0` 时，URMA AIV Put 用 2 核、Get 在本机用 2 核、远端 A+K 用 4 核，UBMEM AIV
Put/Get 用 16 核，AICPU Put 用 1 个 worker，Gather/Scatter Get 用 5 个。URMA AIV Get 每核 Jetty 数默认为本机 2、远端 A+K 1（支持 1..64），
Put 每核默认 2 个（支持 1..2）。仅 URMA AIV Get 接受 `max_inflight_total`，默认 2048，显式 0 按 SQ/CQ 容量自动分组，非零须至少为
`2 * cores`。`request_capacity` 默认 65536；所有 Get 仅把它作为预热提示，
总请求量由调用方张量和 ABI 决定，不受此配置限制。Put 仍受配置容量及 65536 上限约束。
图/输出绑定仍须预先准备足够的请求容量。Gather 可额外指定 `cpu_threads`。
Gather 可设置 `gather_ddr_bytes`，Scatter 可设置 `scatter_hbm_bytes`；均为字节预算，包含
对应工作区的元数据和控制区，自动选择预算内最大的 4096 倍数窗口。默认 0 保留现有
65536 条布局的内存占用，所有窗口在一次 kernel launch 内处理。见 [AICPU 窗口配置](../docs/aicpu_get_windows.md)。
Gather 还可设置 `gather_read_wave_wrs=64`，限制每 worker 每波的数据 READ WR 数；支持
1..4096。默认 `None` 在每次 kernel 执行时按设备总 `miss_count` 自动选择：不超过 8192 用
512 WR，否则用 64 WR；同一张图可随计数变化自动切换。显式值始终覆盖自动策略。
该参数独立于内存预算，改变自动/固定配置后需重建执行资源和捕获图。
这些字段均配置在 NPU 侧 `ExecutionResources`，由 `NpuRuntime(get_resources=...)` 接收。
AIV/UBmem 的 `kernel_path=""` 使用构建目录的默认产物：Get 为
`bin/kvaf_execute_batch_get_kernels.o`，Put 为 `bin/kvaf_execute_batch_put_kernels.o`。
`./build.sh npu` 默认使用 `build-npu`；自定义 `--build-dir` 时自动记录该目录的绝对路径，
不依赖应用启动目录。资源项中非空的 `kernel_path` 覆盖默认值；移动构建产物后须重新构建
或显式指定新路径。不使用的算法不检查 kernel 文件。
AICPU 首次准备时从 CANN 安装目录定位包，不接受资源项 kernel 路径。
预算变化需新建 Runtime/graph，详见[在途控制](../docs/batch_get_inflight_control.md)。

`runtime.initialize()` 只初始化基础设备，不连接 CPU、不分配 DDR。普通缓存或 CPU Pull
需 `link_to("CPU_IP:port")` 后 `initialize_ddr(ddr_bytes, slot_size)`；兼容入口
`initialize(ddr_bytes, slot_size)` 依次完成基础设备和 DDR 初始化。DDR 初始化只建立缓存区域与
元数据，不执行探测 Put 或启动带宽测速；算法及通信资源在首次使用或 warmup 时准备。
`shutdown_ddr()` 只关闭 DDR 模块，NPU raw Pull 连接继续可用。
`npu_eids=[]` 使用首个自动发现的候选；显式列表不回退到自动发现，AIV Get 按自身的
`cores * jettys` 预算使用整组 EID 条带，Put/AICPU 使用首项。`peer_ub_device` 可固定
CPU 侧设备。远端 URMA AIV Direct Get 默认选两个不同的 CPU FE，可在 NPU 进程设置
`KVAF_DEFAULT_PEER_FE_COUNT=1` 改为一个，也可用 `peer_ub_device="devC,devA,devB"`
按顺序显式选路。每次 `batch_get` 仍提交整个请求。详见[远端多 FE](../runtime/README.md#远端-ak-的多-fe-batchget)。
UBmem 不使用 EID，CPU 内存分配依赖 Ascend HAL 与驱动。

`batch_put(keys, packed_input, input_offsets, lengths, stream=None)` 接收调用方的
NPU packed input 和每条 block 的 CPU 整数 metadata，同步写入及事务提交完成后返回。
AIV Put 使用默认或显式指定的可读 kernel，不再回退到 ACL D2H mock。

`AclGraph` 与 `NpuRuntime.batch_get` 可接收调用方持有的 `torch.npu.Stream`；后者省略
stream 时使用 `torch.npu.current_stream()`。KVAF 保留执行所需的 stream 引用，但不创建或销毁调用方的 stream；
图 capture、replay 和生命周期仍由上层框架负责。同一 stream 上的后续 Torch/ACL 操作
自然按序；跨 stream 依赖由调用方建立。

`CpuPullOptions(timeout_ms=30000)` 配置 CPU Pull 默认超时；`batch_cpu_pull()` 同步完成
READ 与元数据提交，`prepare_put / submit_cpu_pull / query_cpu_pull / wait_cpu_pull /
commit_put / abort_put` 提供显式事务。它们使用与普通 Put 相同的键写入事务规则，独立于
Put/Get provider 的通信资源。通过 `cpu_pull_options=` 传入选项，无 `enable_cpu_offload`
或 `cpu_candidate` 构造参数；CPU 设备选择使用 `peer_ub_device`。

`NpuPullOptions(kernel_path="", cores=2, jettys=2, max_inflight_total=0,
max_output_bindings=64, diagnostics=False)` 通过 `npu_pull_options=` 传入；raw 的设备
inflight 预算和 output 注册缓存预算与普通 Get 的 `get_resources` 独立。
`batch_get_raw_addr(ready, src_offsets, dst_offsets, lengths, output, stream=None,
profile=False, validate_ranges=False, synchronize=False)` 默认只入队，不隐式等待。
应用负责保留三组 metadata Tensor 及其 storage、stream 和源，直到执行完成；图模式则
保持到全部 replay 完成并销图。KVAF 将 output Tensor 及其实际 storage 保留到 Runtime
成功关闭，应用仍不能提前修改 metadata、resize 或重新绑定注册内存。

raw 图先对相同源 MR/output 预热并外部同步，再用框架 capture 普通 raw 调用；不需要
额外的 KVAF Begin/End 接口或 graph lease。应用负责 replay 后的 stream 同步、销毁框架图，
然后才能关闭源/Runtime；关闭 Runtime 不代替这些同步和销图步骤。图回放固定源 MR 与
缓冲地址，不重新执行 Host ReadyToken 检查。应用确认消费完成后才能通知生产端复用。
`pending_raw_addr` 仅表示未决失败隔离，不是正常异步工作的完成状态或计数；只有本次调用
返回完成状态未知时，KVAF 才额外保留它的 metadata/storage/stream，直到 Runtime 安全关闭。
完整握手见 [NPU raw READ](../docs/hbm_raw_addr_read.md)。

Binding 不负责 vLLM 适配。后续只在 Runtime、Meta Manager 和 KV Access Operator 的 C++ 接口稳定后，将需要暴露的类和方法注册到 Python。

CacheOS-vLLM Adaptor 属于 vLLM 侧实现，不放在本目录。

异步 Get 的设备完成和框架回收通过 `drain_get()` 完成；外部 stream 同步不会自动
归还执行租约及引用。下一次普通 Get 会先完成前一次 Get。Graph 可使用
`resources = runtime.graph_resources()`，在一次真实模型预热的 `resources.prepare(stream)`
scope 中自动收集 Get 节点，再在 `resources.capture(stream)` scope 中进行外部 Torch/ACL
捕获；模型仍调用普通 `batch_get`，无需传节点租约。保留底层显式 begin/end_get_graph 接口。
单张图保持固定 stream，不同图可使用不同 stream，调用方负责与普通 Get 串行执行；
KVAF 不代理外部 replay 或自动安排跨 stream event。停止 replay、同步并销毁关联 graph
后调用 `resources.close()`；失败保留依赖，未回放或捕获失败可用 `discard_result=True`。
状态及 profile 的图查询须传 `graph_lease`（由 `resources.leases` 取得），不传则查询 eager；
profiling 在申请该实例图租约前开启，全部结束后关闭。

### 重复 Get 的输出绑定

URMA AIV Direct / AICPU Gather 可使用 `begin_get_output(execution, output,
request_capacity, stream=None)`，将返回 ID 作为后续 `batch_get(..., output_lease=id)`
的参数。`drain_get()` 完成本次调用但保留注册；`end_get_output(id)` 等待并注销。
成功 shutdown 也会释放绑定。输出绑定期间不得 resize/set_ 或通过别名替换、释放
底层 storage；Python 引用保活不等于 storage 不可变。

每种 Get 配置可持有一个输出绑定，固定该配置和输出分配；不同配置的绑定可共存，
Get 仍串行完成，Put 不受影响。未绑定调用按临时注册规则完成并退役自身引用；匹配的注册若仍被图持有则保留。
完整例子、错误处理和性能计时方式见
[实现与使用](../docs/cache_transport_execution_implementation.md#显式输出绑定)。
