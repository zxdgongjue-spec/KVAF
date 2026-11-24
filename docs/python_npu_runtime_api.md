# KVAF Python API · NpuRuntime

本文适用于通过 `kvaf.NpuRuntime` 向远端 CPU DDR 写入 KV block，并按 token 读回 NPU 的调用方。构建和部署见[项目 README](../README.md#构建)，Python 扩展和资源管理分别见 [Binding README](../bindings/README.md) 和 [Runtime README](../runtime/README.md)。

Python 参数名、默认值和重载以 pybind 注册为准。Tensor 布局与字节范围以绑定代码和 `RuntimeCore` 校验为准。对应入口见[构造函数注册](../bindings/include/npu_runtime_constructor.h#L15)、[Python 方法注册](../bindings/src/npu_python_runtime.cc#L623)、[RuntimeCore 声明](../runtime/include/runtime.h#L65)和[请求结构](../common/include/requests.h#L13)。

## 构造与初始化

```text
kvaf.NpuRuntime(
    batch_get_kernel_object_path,
    batch_get_token_bytes,
    batch_get_tokens_per_block,
    device_id=0,
    npu_core_count=2,
    batch_put_kernel_object_path="",
    jettys_per_core=2,
    npu_eids=[],
    adaptive_ub_device=False,
    adaptive_ub_miss_count=2048,
    peer_ub_device="",
    adaptive_peer_ub_devices=[],
    batch_get_max_inflight_total=2048,
    transport="urma",
    *,
    aicpu_batch_get_mode="",
    allocation_mode="append_only",
)
```

前三个参数必填。`aicpu_batch_get_mode` 和 `allocation_mode` 只接受关键字传参。其余参数也可通过对应关键字传入。

| 参数 | 类型 | 默认值 | 含义与约束 |
| --- | --- | --- | --- |
| `batch_get_kernel_object_path` | `str` | 必填 | AIV URMA 和 UBmem 传 BatchGet kernel 的有效 `.o` 路径。AICPU 传 `""`，由 Runtime 从 CANN 安装目录定位设备包。 |
| `batch_get_token_bytes` | `int` | 必填 | 单个 token 的 KV 字节数，须大于 0。等于 BatchGet 的输出行字节数。 |
| `batch_get_tokens_per_block` | `int` | 必填 | 每个 KV block 的 token 数，须大于 0。与 `batch_get_token_bytes` 相乘得到 BatchGet 期望的表项长度。 |
| `device_id` | `int` | `0` | NPU 设备编号。先用 `torch.npu.set_device(device_id)` 选择设备，在同一设备上创建调用所用的 Tensor 和 stream。 |
| `npu_core_count` | `int` | `2` | AIV 核数或 AICPU worker 数。构造校验范围分别为 `1..48` 和 `1..8`，实际配置还须满足所用设备和后端的资源约束。 |
| `batch_put_kernel_object_path` | `str` | `""` | AIV URMA 在 `link_to()` 时要求配置 Put kernel，用于初始化时的链路探测。UBmem 在调用 Put 时需要配置。AICPU Put 使用 CANN 设备包。 |
| `jettys_per_core` | `int` | `2` | AIV URMA BatchGet 的每核 Jetty 数，范围 `1..64`。AIV URMA BatchPut 每核最多使用 2 个。 |
| `npu_eids` | `list[str]` | `[]` | URMA 的 NPU 本端 EID 列表，每项为 32 个连续十六进制字符。空列表使用自动发现。AIV URMA 的显式列表用于 Get 条带化，Put 使用首项，列表长度至多为 `npu_core_count * jettys_per_core`。AICPU 按候选顺序尝试，Get 和 Put 共用选中的 EID。 |
| `adaptive_ub_device` | `bool` | `False` | 开启 AIV URMA 的启动带宽选优。CPU 服务已固定设备或候选列表为空时跳过测速。 |
| `adaptive_ub_miss_count` | `int` | `2048` | 启动测速的 BatchGet 请求数，开启测速时须在 `1..2147483647`。 |
| `peer_ub_device` | `str` | `""` | 固定 CPU 对端的 UB 设备名，可复用此前选出的设备。AIV URMA 下与 `adaptive_ub_device=True` 互斥。 |
| `adaptive_peer_ub_devices` | `list[str]` | `[]` | 启动测速时允许选择的 CPU 对端 UB 设备名。 |
| `batch_get_max_inflight_total` | `int` | `2048` | AIV URMA 单次 Get 所有核合计的在途 READ 预算。`0` 使用整批提交，非零值至少为 `2 * npu_core_count`。详见[在途控制](batch_get_inflight_control.md)。 |
| `transport` | `str` | `"urma"` | 传输类型，可选 `"urma"` 或 `"ubmem"`。 |
| `aicpu_batch_get_mode` | `str` | `""` | 选择配套的 Get 和 Put 实现。可选值与 `transport` 的组合见下表。 |
| `allocation_mode` | `str` | `"append_only"` | 按对象真实长度追加，多个 Key 可共用一个 slot。依赖 `key == slot_id` 的调用方显式传 `"fixed"`。 |

| `transport` | `aicpu_batch_get_mode` | BatchPut | BatchGet |
| --- | --- | --- | --- |
| `"urma"` | `""` | AIV URMA | AIV URMA |
| `"urma"` | `"gather_read"` | AICPU UMDK | CPU Gather + AICPU READ |
| `"urma"` | `"cpu_push_scatter"` | AICPU UMDK | CPU Push + AICPU Scatter |
| `"ubmem"` | `""` | AIV UBmem | AIV UBmem |

| 方法 | 参数和调用顺序 |
| --- | --- |
| `link_to(address)` | 先指定 CPU 服务地址，格式为 `"host:port"`。 |
| `initialize(ddr_bytes, slot_size)` | 建链后申请并初始化远端 CPU DDR。两个参数均以字节计，须为正整数，且 `ddr_bytes % slot_size == 0`。可用 slot 数为 `ddr_bytes // slot_size`。 |
| `shutdown()` | 调用结束后释放 Runtime 的 DDR、元数据和传输资源。存在异步 Get 时，先等待相关 stream 完成。 |

构造参数在 C++ 中对应 [NpuRuntimeOptions](../runtime/include/npu_runtime.h#L16)。后端、构造参数和初始化约束见 [Runtime 实现](../runtime/src/npu_runtime.cc#L513)，EID 与启动测速的使用说明见 [Runtime README](../runtime/README.md#可选启动-ub-device-选优)。

## BatchPut

将连续 NPU Tensor 中的字节范围写入远端 DDR。成功返回 `None` 时，写入和元数据提交都已完成。

```python
runtime.batch_put(keys, source, input_offsets, lengths, stream=None)
```

| 参数 | 类型 | 含义与约束 |
| --- | --- | --- |
| `keys` | Python 整数序列，或一维连续 CPU `torch.int64` Tensor | 同一批内 Key 须唯一，且每个 Key 须未占用。`append_only` 的 Key 独立于 slot；`fixed` 的 Key 同时是 slot ID，范围为 `[0, ddr_bytes // slot_size)`。 |
| `source` | 非空、连续的 NPU Tensor | 整批数据的源 buffer。绑定将其视为连续字节，`torch.uint8` 便于直接按字节构造数据。 |
| `input_offsets` | Python 整数序列，或一维连续 CPU `torch.int64` Tensor | 每块在 `source` 内的起始字节偏移，须大于等于 0。 |
| `lengths` | Python 整数序列，或一维连续 CPU `torch.int64` Tensor | 每块的字节长度，须满足 `0 < lengths[i] <= slot_size`。同一批可写入不同长度的块。 |
| `stream` | `torch.npu.Stream` 或 `None` | 默认使用 `torch.npu.current_stream()`。在该 stream 上等待 Put 完成后提交元数据。 |

`keys`、`input_offsets` 和 `lengths` 必须非空且等长。Runtime 将 `source` 的字节区间 `[input_offsets[i], input_offsets[i] + lengths[i])` 写入为第 `i` 个 Key 预留的连续地址，该区间须完整落在源 buffer 内。搬运成功后才让映射对 Get 可见，搬运失败则放弃预留。

Python 关键字用 `source`，它映射到 C++ `BatchPutRequest::input`。对应校验见 [BatchPut 绑定](../bindings/src/npu_python_runtime.cc#L338)、[BatchPutRequest](../common/include/requests.h#L73)、[RuntimeCore::BatchPut](../runtime/src/runtime.cc#L344)和 [MetaManager slot 校验](../meta_manager/src/in_memory_meta_manager.cc#L157)。

## BatchGet

从 AccessTable 选择已写入的 KV block，按给定字节偏移读取一个 token，原地写入调用方预分配的输出 Tensor。

### 准备与释放 AccessTable

| 方法 | 参数、返回值与生命周期 |
| --- | --- |
| `prepare_access_table(keys)` | 接受非空的扁平 Python 整数序列或一维连续 CPU `torch.int64` Tensor。按输入顺序锁定已提交的 Key 并返回 `AccessTable` handle。缺失 Key 会触发 `KeyError`。 |
| `access_table_view(access_table, device="cpu", capacity=0)` | CPU 模式返回含 `addresses`、`lengths`、`valid` 和 `size` 的字典，前三项为依赖 handle 生命周期的只读 memoryview。指定 `device="npu"` 时返回 `DeviceAccessTable`，包含 `addresses: torch.int64[N]` 和 `valid: torch.uint8[N]` 两个 NPU Tensor。`capacity=0` 使用实际表长，也可指定更大的设备表容量，补齐位置的 `valid=0`。 |
| `release_access_table(access_table)` | 释放 handle 和 Key 锁。调用方须等所有使用该表的 Get 完成后再调用。 |
| `batch_free(keys)` | 释放已提交的 Key。共享 slot 在最后一个对象释放后回收。参数类型与建表相同，同一批中 Key 须唯一。仍被 AccessTable 锁定时返回 `kBusy`，Python 抛出 `ValueError`。 |

实现入口见 [AccessTable 绑定](../bindings/src/npu_python_runtime.cc#L364)、[CPU view 字段](../bindings/include/access_table_python.h#L14)和 [MetaManager::BatchFree](../meta_manager/src/in_memory_meta_manager.cc#L386)。

### 调用签名与参数

```python
runtime.batch_get(
    device_access_table, indices, offsets, miss_count, output_indices, output,
    stream=None, synchronize=False, profile=False,
)

runtime.batch_get(
    addresses, valid, indices, offsets, miss_count, output_indices, output,
    stream=None, synchronize=False, profile=False,
)
```

所有数据 Tensor 均须非空、连续并位于 Runtime 对应的 NPU 上。表容量 `N`、请求容量 `R` 和输出行数分别由各自的 Tensor 决定。

| 参数 | 类型 | 含义与约束 |
| --- | --- | --- |
| `device_access_table` | `kvaf.DeviceAccessTable` | `access_table_view(handle, device="npu")` 返回的表。Runtime 校验原 handle 是否有效，并检查有效表项的长度。 |
| `addresses`、`valid` | NPU `torch.int64[N]`、`torch.uint8[N]` | 裸 Tensor 重载使用等容量的地址表和有效位。调用方负责保证底层地址、表项长度和 slot 的生命周期。 |
| `indices` | 一维 NPU `torch.int32[R]` | 请求使用的表内行号。用 `[7, 3]` 建表时，`indices[i] = 0` 对应 Key `7`。 |
| `offsets` | 一维 NPU `torch.int32[R]` | 选中 block 内的**字节偏移**。每次读 `batch_get_token_bytes` 字节，读取范围须完整落在该 block 内。 |
| `miss_count` | 单元素 NPU `torch.int32` Tensor | 值为 `0..R`，只处理三个请求数组的 `[0, miss_count)` 前缀。AICPU 的请求容量 `R` 上限为 65536。 |
| `output_indices` | 一维 NPU `torch.int32[R]` | 某条请求写入的输出行号，须落在 `[0, output.shape[0])`。 |
| `output` | 至少二维的 NPU Tensor | 第 0 维是输出行数。一行的字节数 `output.numel() * output.element_size() // output.shape[0]` 必须等于 `batch_get_token_bytes`。例如字节输出可用 `torch.uint8[行数, batch_get_token_bytes]`。 |
| `stream` | `torch.npu.Stream` 或 `None` | 默认使用 `torch.npu.current_stream()`。跨 stream 消费输出时由调用方建立依赖。 |
| `synchronize` | `bool` | 默认 `False`，入队后返回。`True` 在本次调用内等待 stream。 |
| `profile` | `bool` | 默认 `False`，返回 `None`。`True` 仅支持 AICPU，要求 `synchronize=True`，调用返回阶段耗时字典。 |

每条有效请求从 `addresses[indices[i]] + offsets[i]` 读取一个 token，写入 `output[output_indices[i]]`。用 `DeviceAccessTable` 调用时，整个表的有效条目长度都必须等于 `batch_get_token_bytes * batch_get_tokens_per_block`。因此如果 Put 后要通过同一张表做 NPU Get，写入长度应按这个固定 block 字节数准备。

同一 stream 上的后续工作按提交顺序执行。调用方须保持输入、输出和设备表有效，直到关联 Get 完成。使用 Graph 时，在 capture 前建好表、预热固定输出，并在后续 replay 中保持 Tensor 地址稳定。AICPU capture 使用 `synchronize=False, profile=False`，在图外调用 `check_batch_get_status(stream)` 检查执行状态。

校验和请求结构见 [BatchGet 绑定](../bindings/src/npu_python_runtime.cc#L526)、[BatchGetRequest](../common/include/requests.h#L51)和 [RuntimeCore::BatchGet](../runtime/src/runtime.cc#L248)。AICPU 容量限制见 [GatherRead](../access_operator/demo/aicpu_batch_get/host/batch_get_host.cc#L194)和 [CPU Push Scatter](../access_operator/demo/aicpu_batch_get/host/cpu_push_host.cc#L233)，图模式见[统一测试用例](../tests/python/test_npu_batch_get_perf.py#L503)。

## Python 示例 · 写入两个 block 并读取两个 token

此例使用 AIV URMA。运行前按[构建说明](../README.md#构建)准备 NPU 包和 kernel 文件，并启动 [CPU 服务](../runtime/README.md#cpu-runtime-服务与-npu-link)。从仓库根目录运行，Python 环境需能导入 `build-npu/python` 下的 `kvaf`。

```python
import os
from pathlib import Path

import torch
import torch_npu
import kvaf

device_id = int(os.environ.get("DEVICE_ID", "0"))
torch.npu.set_device(device_id)
device = f"npu:{device_id}"
kernel_dir = Path("build-npu/bin").resolve()
token_bytes = 656
tokens_per_block = 128
block_bytes = token_bytes * tokens_per_block
keys = [7, 3]

runtime = kvaf.NpuRuntime(
    batch_get_kernel_object_path=str(kernel_dir / "kvaf_execute_batch_get_kernels.o"),
    batch_get_token_bytes=token_bytes,
    batch_get_tokens_per_block=tokens_per_block,
    device_id=device_id,
    batch_put_kernel_object_path=str(kernel_dir / "kvaf_execute_batch_put_kernels.o"),
)

try:
    runtime.link_to(os.environ.get("KVAF_CPU_SERVICE_ADDR", "127.0.0.1:28999"))
    runtime.initialize(ddr_bytes=32 * block_bytes, slot_size=block_bytes)

    source_cpu = torch.zeros((2, tokens_per_block, token_bytes), dtype=torch.uint8)
    source_cpu[0, 1].fill_(11)
    source_cpu[1, 0].fill_(20)
    source = source_cpu.to(device)
    runtime.batch_put(
        keys=keys,
        source=source,
        input_offsets=[0, block_bytes],
        lengths=[block_bytes, block_bytes],
    )

    table = runtime.prepare_access_table(keys)
    stream = torch.npu.current_stream()
    try:
        device_table = runtime.access_table_view(table, device="npu")
        indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
        offsets = torch.tensor([token_bytes, 0], dtype=torch.int32, device=device)
        miss_count = torch.tensor([2], dtype=torch.int32, device=device)
        output_indices = torch.tensor([1, 0], dtype=torch.int32, device=device)
        output = torch.empty((2, token_bytes), dtype=torch.uint8, device=device)

        runtime.batch_get(
            device_table, indices, offsets, miss_count, output_indices, output,
            stream=stream,
        )
    finally:
        stream.synchronize()
        runtime.release_access_table(table)

    result = output.cpu()
    assert (result[0] == 20).all().item()
    assert (result[1] == 11).all().item()
    runtime.batch_free(keys)
finally:
    runtime.shutdown()
```

仓库的硬件集成测试分别演示了[变长 BatchPut](../tests/python/test_npu_batch_get.py#L266)、[定长写入与建表](../tests/python/test_npu_batch_get.py#L298)和 [BatchGet Tensor 构造及读回校验](../tests/python/test_npu_batch_get.py#L324)。
