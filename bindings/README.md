# Python Bindings

本目录是 KVAF C++ 核心与 Python 之间的可选绑定层，使用 Pybind11 构建 `_kvaf` 原生扩展。

`include/` 存放 Binding 内部共享头，`src/` 存放扩展实现；Python 包与构建后导入校验脚本保留在本目录。

当前 Binding 用于验证以下链路：

```text
C++ 编译 → Python 原生扩展 → Python 导入
```

CPU 构建提供进程内 Mock Runtime。启用 `KVAF_ENABLE_AIV_BATCH_GET=ON` 时，额外提供
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

`NpuRuntime(batch_get_kernel_object_path, batch_get_token_bytes,
batch_get_tokens_per_block, device_id=0, npu_core_count=1,
batch_put_kernel_object_path="", jettys_per_core=2, npu_eids=[],
batch_get_stream_count=1)` 要求显式传入单 token KV
字节数和每个 KV block 中的 token 数，两个参数均无默认值；
每个 AccessTable entry 大小为两者乘积。
`jettys_per_core` 配置 BatchGet 每个 AIV core 使用的 Jetty 数量，
可选 1 到 64 个独立 Jetty。BatchPut 保持原有实现：该值为 1 或 2 时
使用相同数量，BatchGet 配置为 3 到 64 时 BatchPut 仍使用 2 个。
`batch_get_stream_count` 可选 1（默认）或 2。`npu_core_count` 表示**每个 stream**
的 AIV block 数，总 BatchGet block 数为两者乘积，当前不能超过 48。
双 stream 使用同一个 transport：调用方 stream 执行分片 0，内部 stream 执行分片 1；
各自使用独占的 SQ/CQ 和持久 queue state。分片范围按设备上的实际 `miss_count`
计算，不回传 CPU。内部 stream 在返回前通过 event 汇合到调用方 stream，
因此调用方 stream 的后续任务及同步包含两路读取。
`batch_get_kernel_profile()` 按 `[stream][core]` 返回所有 block 的 profile；
`core` 是全局编号，`stream` 和 `stream_core` 分别标识路编号和路内编号。
BatchPut 仍为单 stream、`npu_core_count` 个 block。
远程 CPU DDR 只增加 `runtime.link_to("CPU_IP:port")`，必须在 `initialize` 前调用。
`npu_eids=[]` 时 EID、UMDK device 与候选 RMA 验证由 C++ Runtime 自动完成；
显式传入多个 32 位十六进制 EID 时，Runtime 为每个 EID 创建独立
BatchGet 路径，并把 core/Jetty worker 轮转分配到这些路径。BatchPut
保持原有单 EID 实现，使用列表第一项。该列表是严格配置，不会静默回退。
`batch_put(keys, packed_input, input_offsets, lengths, stream=None)` 接收 caller-owned
NPU packed input 和每条 block 的 CPU 整数 metadata；同步写入完成后才返回。未传
BatchPut kernel 时，该方法保留 ACL D2H mock，便于只验证 BatchGet 的构建。

`AclGraph` 与 `NpuRuntime.batch_get` 可接收调用方持有的 `torch.npu.Stream`；后者省略
stream 时使用 `torch.npu.current_stream()`。KVAF 不创建、不保存或销毁调用方的 stream；
图 capture、replay 和生命周期仍由上层框架负责。同一 stream 上的后续 Torch/ACL 操作
自然按序；跨 stream 依赖由调用方建立。
对同一 Runtime 的 BatchGet 调用及 graph replay 必须保序，不能在无依赖的调用方
stream 上重叠执行。切换调用方 stream 前，需要同步前一次调用或建立外部依赖。
双 stream 的 stream/event 在初始化时创建，capture 记录完整的分叉和汇合；
capture 前仍需预热 output 注册。释放 Runtime 前应完成工作并销毁引用它的 graph。
若部分提交失败，完成调用方 stream 后才能释放输入输出；若错误指出汇合失败，
则需要同步整个 device。失败的双 stream 执行组需重新初始化才能继续提交。

Binding 不负责 vLLM 适配。后续只在 Runtime、Meta Manager 和 KV Access Operator 的 C++ 接口稳定后，将需要暴露的类和方法注册到 Python。

CacheOS-vLLM Adaptor 属于 vLLM 侧实现，不放在本目录。
