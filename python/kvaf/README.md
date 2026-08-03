# `kvaf` Python 包

该目录提供 Python 侧的 `kvaf` 包入口和同步 Runtime 的直接 pybind API。

构建 Pybind11 扩展后，使用方式为：

```python
import kvaf

runtime = kvaf.Runtime()  # 当前是确定性的进程内 CPU Mock operator
runtime.initialize(ddr_bytes=256, slot_size=64)
runtime.batch_put(keys=[101], blocks=[b"payload"])

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
不会启动子进程。`batch_put` 的 payload 接受 `bytes`、`bytearray` 或任何
Python buffer 对象；`batch_get` 目前要求可写 `bytearray` 作为 CPU Mock
输出。`access_table_view(table)` 返回地址、长度和有效位的 Python 列表，便于
测试代码断言。

CPU Mock 仅用于本地功能验证；它不等同于 A5/CANN 的设备内存和 Direct-HCCP
传输。AIV operator 的直接 Python binding 会作为独立后端提供，不能用 MPS
代替 NPU 集成验证。vLLM 的 CacheOS-vLLM Adaptor 不在此处实现。
