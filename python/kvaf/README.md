# `kvaf` Python 包

该目录提供 Python 侧的 `kvaf` 包入口和 Runtime 的直接 pybind API。

构建 Pybind11 扩展后，使用方式为：

```python
import kvaf

runtime = kvaf.Runtime()  # 当前是确定性的进程内 CPU Mock operator
runtime.initialize(ddr_bytes=256, slot_size=64)
payload = b"payload"
runtime.batch_put(
    keys=[101], input=payload, input_offsets=[0], lengths=[len(payload)]
)

table = runtime.prepare_access_table([101])
output = bytearray(64)  # 调用方拥有 BatchGet 的输出 buffer
runtime.batch_get(
    table,
    indices=[0],
    output_indices=[0],
    output=output,
    output_block_bytes=64,
)
runtime.release_access_table(table)
runtime.batch_free([101])
runtime.shutdown()
```

`Runtime` 直接执行真正的 Runtime、MetaManager 和 Host DDR 生命周期，
不会启动子进程。`batch_put(keys, input, input_offsets, lengths)` 接收一个
连续打包的 buffer；第 `i` 个 key 的数据位于
`input[input_offsets[i] : input_offsets[i] + lengths[i]]`。它支持同批变长
block，不会隐式重打包输入。`batch_get` 目前要求可写 `bytearray` 作为 CPU
Mock 输出。`access_table_view(table)` 返回地址、长度和有效位的 Python
memoryview，便于测试代码断言。

CPU Mock 仅用于本地功能验证；它不等同于 A5/CANN 的设备内存和 Direct-HCCP
传输。AIV operator 的直接 Python binding 会作为独立后端提供，不能用 MPS
代替 NPU 集成验证。vLLM 的 CacheOS-vLLM Adaptor 不在此处实现。

启用 AIV CANN 构建时，额外导出 `kvaf.NpuRuntime`、`kvaf.DeviceAccessTable` 与
`kvaf.AclGraph`。使用 NPU BatchGet 时先调用
`runtime.access_table_view(table, device="npu", capacity=N)` 得到
`DeviceAccessTable`，再调用：

```python
runtime.batch_get(device_table, indices, offsets, mask, output_indices, output)
# 或直接传入 DeviceAccessTable 中的两个 Tensor：
runtime.batch_get(
    device_table.addresses,
    device_table.valid,
    indices,
    offsets,
    mask,
    output_indices,
    output,
)
```

其中所有数据参数均为连续 NPU Tensor；输出行字节数从 `output.shape[0]` 推导，默认
使用 `torch.npu.current_stream()` 且只提交不主动同步。AIV 内部 status/queue 不属于
Python API。直接传入 `addresses` / `valid` 时，调用方需自行保证表项指向
`128 * 656B` 的远端区域，并保持 Tensor 有效直到 stream 执行完成。
`AclGraph` 由 Python 调用方持有，直接封装 ACL Runtime Graph 的
capture / replay / synchronize / reset；图对象不属于 Runtime 或 Operator。
