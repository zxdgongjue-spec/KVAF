# Python Bindings

本目录是 KVAF C++ 核心与 Python 之间的可选绑定层，使用 Pybind11 构建 `_kvaf` 原生扩展。

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
`NpuRuntime.batch_get(device_access_table, indices, offsets, mask, output_indices,
output, stream=None, synchronize=False)` 直接接收 NPU Tensor，也支持以
`batch_get(addresses, valid, indices, offsets, mask, output_indices, output, ...)`
传入调用方持有的设备表 Tensor。它不暴露 data pointer、字节数、
output 行字节数或 AIV 的内部 status/queue buffer；output 行字节数由 output Tensor
第 0 维之后的连续元素推导，也不返回伪装成完成时间的数值。

`mask` 保留原来的 `torch.uint8[N]` 语义，也直接接受原生 MTP LIDU 输出的
`torch.int32[1]` 总 miss 数：后者表示 `indices/offsets/output_indices` 的有效紧凑
前缀，形状仍为固定容量 `N`。调用签名、656B payload、当前 stream、Direct-HCCP
队列和输出注册不变；不在 Python 展开 mask、不在 Host 读取 count、不新增传输路径。
`kvaf.AIV_BATCH_GET_COMPACT_COUNT_ABI == 1` 是支持该路径的显式能力标记；调用方
必须检查它，旧扩展或 CPU-only 扩展返回 0。NPU host/kernel 启动参数从 104B 扩展
到 112B，必须与 fused 源码消费者一起重新编译，不得混用旧 kernel 二进制。

LI/LIDU 必须在同一 stream 上先于 `batch_get` 完成，或由调用方添加跨 stream event。
调用方持有包括 count 在内的所有 Tensor，直到 Get 和后续 QSFA 最终读者结束；图回放
必须保持这些地址稳定，可改变 count 的内容（包括变为 0）。count 超出 `[0,N]` 时，
设备记录错误且不提交任何 READ。内核按有效前缀分配 AIV 工作，沿用每 AIV 16384 条
active descriptor 的上限以及现有 SQ 容量检查、32-WQE 分批提交和 CQ 完成等待。

`NpuRuntime(batch_get_kernel_object_path, device_id=0, npu_core_count=1,
batch_put_kernel_object_path="", jettys_per_core=2)` 的 `jettys_per_core` 参数同时
配置 BatchGet 和 BatchPut 每个 AIV core 使用的 Jetty 数量，可选 1 或 2 个独立 Jetty。
`batch_put(keys, packed_input, input_offsets, lengths, stream=None)` 接收 caller-owned
NPU packed input 和每条 block 的 CPU 整数 metadata；同步写入完成后才返回。未传
BatchPut kernel 时，该方法保留 ACL D2H mock，便于只验证 BatchGet 的构建。

`AclGraph` 与 `NpuRuntime.batch_get` 可接收调用方持有的 `torch.npu.Stream`；后者省略
stream 时使用 `torch.npu.current_stream()`。KVAF 不创建、不保存或销毁调用方的 stream；
图 capture、replay 和生命周期仍由上层框架负责。同一 stream 上的后续 Torch/ACL 操作
自然按序；跨 stream 依赖由调用方建立。

Binding 不负责 vLLM 适配。后续只在 Runtime、Meta Manager 和 KV Access Operator 的 C++ 接口稳定后，将需要暴露的类和方法注册到 Python。

CacheOS-vLLM Adaptor 属于 vLLM 侧实现，不放在本目录。
