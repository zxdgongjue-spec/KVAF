# Python Bindings

本目录是 KVAF C++ 核心与 Python 之间的可选绑定层，使用 Pybind11 构建 `_kvaf` 原生扩展。

`./build.sh npu` 默认将 AIV URMA、UBmem 和 AICPU Gather/Scatter 构建进同一个
`build-npu/python/kvaf` 包，并自动安装 AICPU 设备包。运行时按 `transport_mode`、默认 Put/Get 配置及请求的
`execution` 选择实现；仅需 AIV/UBmem 时使用 `--without-aicpu`。

`include/` 存放 Binding 内部共享头，`src/` 存放扩展实现；Python 包与构建后导入校验脚本保留在本目录。

当前 Binding 用于验证以下链路：

```text
C++ 编译 → Python 原生扩展 → Python 导入
```

CPU 构建提供进程内 Mock Runtime。启用 `KVAF_ENABLE_AIV_BATCH_GET=ON` 或 `KVAF_ENABLE_AICPU_URMA=ON` 时，额外提供
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
NpuRuntime(batch_get_token_bytes, batch_get_tokens_per_block, *,
           device_id=0, transport_mode="urma", default_put=None, default_get=None,
           put_resources=None, get_resources=None, npu_eids=[], peer_ub_device="")
```

两个 token 布局参数必须提供；每个 AccessTable entry 的大小为两者乘积。
其余参数仅接受关键字。`transport_mode` 可为 `"urma"`、`"ubmem"` 或 `"urma+ubmem"`，
默认算法由 `PutExecutionConfig` / `GetExecutionConfig` 指定，单次调用的 `execution`
可覆盖默认配置。构建支持且符合 transport 模式的全部 provider 自动可选，无 enable 列表。
未指定默认配置时优先选当前构建可用的实现；显式选择不可用实现会在首次使用时失败。

`put_resources/get_resources` 是 `(execution_config, ExecutionResources(...))` 列表。
预算按 provider 配置：`cores` 默认为 2；URMA AIV 的 `jettys` 默认为 2，Get 支持 1..64、
Put 支持 1..2；仅 URMA AIV Get 接受 `max_inflight_total`，默认 2048，显式 0 关闭限制，非零须至少为
`2 * cores`。`request_capacity` 默认 65536，Gather 可额外指定 `cpu_threads`。
AIV/UBmem 的 `kernel_path` 在相应资源项中提供；不使用的算法不要求提供路径。
AICPU 首次准备时从 CANN 安装目录定位包，不接受资源项 kernel 路径。
预算变化需新建 Runtime/graph，详见[在途控制](../docs/batch_get_inflight_control.md)。

必须先 `runtime.link_to("CPU_IP:port")` 再 `initialize()`。初始化只建立缓存区域与
元数据，不执行探测 Put 或启动带宽测速；算法及通信资源在首次使用或 warmup 时准备。
`npu_eids=[]` 使用首个自动发现的候选；显式列表不回退到自动发现，AIV Get 按自身的
`cores * jettys` 预算使用整组 EID 条带，Put/AICPU 使用首项。`peer_ub_device` 可固定
CPU 侧设备。UBmem 不使用 EID，CPU 内存分配依赖 Ascend HAL 与驱动。

`batch_put(keys, packed_input, input_offsets, lengths, stream=None)` 接收调用方的
NPU packed input 和每条 block 的 CPU 整数 metadata，同步写入及事务提交完成后返回。
AIV Put 必须在其资源项中提供可读 kernel，不再回退到 ACL D2H mock。

`AclGraph` 与 `NpuRuntime.batch_get` 可接收调用方持有的 `torch.npu.Stream`；后者省略
stream 时使用 `torch.npu.current_stream()`。KVAF 不创建、不保存或销毁调用方的 stream；
图 capture、replay 和生命周期仍由上层框架负责。同一 stream 上的后续 Torch/ACL 操作
自然按序；跨 stream 依赖由调用方建立。

Binding 不负责 vLLM 适配。后续只在 Runtime、Meta Manager 和 KV Access Operator 的 C++ 接口稳定后，将需要暴露的类和方法注册到 Python。

CacheOS-vLLM Adaptor 属于 vLLM 侧实现，不放在本目录。

异步 Get 的设备完成和框架回收通过 `drain_get()` 完成；外部 stream 同步不会自动
归还执行租约及引用。下一次普通 Get 会先完成前一次 Get。Graph 需显式申请图租约，
一个租约只允许捕获一个 Get 节点；可为多个节点分别申请租约，并在同一 stream 上串行回放多张图。
停止 replay、同步并销毁关联 graph 后结束对应租约。同算法有多个节点时，状态及 profile
查询须传 `graph_lease` 选择节点；profiling 在申请图租约前开启，结束租约后关闭。

### 重复 Get 的输出绑定

URMA AIV Direct / AICPU Gather 可使用 `begin_get_output(execution, output,
request_capacity, stream=None)`，将返回 ID 作为后续 `batch_get(..., output_lease=id)`
的参数。`drain_get()` 完成本次调用但保留注册；`end_get_output(id)` 等待并注销。
成功 shutdown 也会释放绑定。输出绑定期间不得 resize/set_ 或通过别名替换、释放
底层 storage；Python 引用保活不等于 storage 不可变。

每种 Get 配置可持有一个输出绑定，固定该配置和输出分配；不同配置的绑定可共存，
Get 仍串行完成，Put 不受影响。未绑定调用按次完成后注销输出。完整例子、错误处理和性能计时方式见
[实现与使用](../docs/cache_transport_execution_implementation.md#显式输出绑定)。
