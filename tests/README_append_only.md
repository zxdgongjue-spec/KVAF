# append_only：每个 tensor 一个 Key，多 Key 共用 slot

默认仍为 `allocation_mode="fixed"`。新增 `allocation_mode="append_only"`，在创建 `NpuRuntime` 时选择，一次 Runtime 生命周期内保持不变。该选项与 AIV/AICPU 后端选择独立；本次变长端到端验收针对 **AIV + URMA + batch_get_ranges**，没有新增 AICPU range Get 能力。

## 分配与回收规则

| 行为 | fixed（默认） | append_only（可选） |
| --- | --- | --- |
| Key 与 slot | Key 就是 slot_id，范围为 `[0, slot_count)` | Key 独立于 slot，可用 uint64 ID；Key 数可多于 slot 数 |
| 地址 | `base + key * slot_size` | `base + slot_id * slot_size + 对象在 slot 内的 byte offset` |
| 单 Key 长度 | `0 < length <= slot_size` | 相同；一个 Key 的数据连续存放且不跨 slot |
| 放置 | 每个 Key 独占自己的 slot | 按请求顺序，以真实长度追加到当前 slot |
| 当前 slot 尾部放不下 | 对应 Key 超长时拒绝 | 换一个完全空闲的 slot，当前对象从新 slot 的 offset=0 开始 |
| 删除 Key | 该 Key 的 slot 可重用 | 删除映射并减少 slot 引用；只有 slot 内最后一个预留或已提交对象消失后，整个 slot 才回收 |
| 零碎空间 | 无包内分配 | 不复用已删除对象的空洞，也不回头使用已关闭 slot 的剩余尾部 |

例如 slot_size=4096，按顺序 Put 三个独立 Key，长度为 656、1968、1968：

```text
slot 0: [Key 101: 656B][Key 202: 1968B][剩余 1472B，关闭后暂不再用]
slot 1: [Key 303: 1968B][当前可继续追加的尾部 2128B]
```

Key 202 的实际地址是 `base + 656`，但 Get 的 `source_offsets=0` 已经指向它自身的起点。**不要在 Get 时再加一次 slot 内偏移**，也不要按 Key 值推算地址。

删除 Key 202 后，其 1968B 空洞暂不重用；再删除 Key 101，slot 0 才能整体重用。Key 删除后可再次 Put 同一个 Key，新的地址由当前分配状态决定。没有空闲 slot、当前尾部也放不下时，返回 OutOfMemory，即使其他位置还留有空洞。模式切换需要释放旧 Runtime 后重新创建、初始化并重新 Put 数据。

分配器不对每个 tensor 自动补齐到 64B/512B，当前是按真实长度紧密追加。输入 source 中加入 padding 不会自动转化成远端对象之间的 padding：`input_offsets` 只决定从哪里读 source，`lengths` 决定远端预留与写入多少字节。如果把 padding 计入 length，则它属于该 Key 的存储长度；真实 tensor 长度需要调用方另行保存。

## 调用方式

只增加一个构造参数和一个只读属性：

```python
rt = kvaf.NpuRuntime(..., allocation_mode="append_only")
assert rt.allocation_mode == "append_only"
# 省略 allocation_mode 或传 "fixed"，继续使用原固定映射。
```

`initialize(ddr_bytes, slot_size)` 不变，DDR 仍一次分配整片区域，ddr_bytes 必须是 slot_size 的非零整数倍。自动换 slot 是这片区域内的地址管理，不会重新建链、增加 DDR 容量或为每个 tensor 创建 Runtime。

Put/Get/free 接口不变；每条 Put 元数据现在可以直接描述一个独立 tensor：

```python
rt.batch_put(keys, source, input_offsets, lengths, stream=None)
handle = rt.prepare_access_table(keys)
table = rt.access_table_view(handle, device="npu")
rt.batch_get_ranges(table, indices, source_offsets, lengths,
                    request_count, output_offsets, output,
                    stream=None, synchronize=False, read_chunk_bytes=0)
rt.release_access_table(handle)
rt.batch_free(keys)
```

| 参数 | 每 tensor 独立 Key 时的含义 |
| --- | --- |
| Put `keys` | 每个 tensor 的业务 Key，按本批 Put 顺序排列，不是 slot_id；批内不能重复，也不能覆盖已存在或预留中的 Key |
| Put `source` | 调用方整理好的连续 NPU 数据缓冲区；本版仍采用这一个 source 的接口 |
| Put `input_offsets` | CPU 字节偏移列表，每个 tensor 在 source 中的起点 |
| Put `lengths` | CPU 长度列表，每个 Key 实际写入字节数，可不相等；每项不得超过 slot_size |
| Get `table` | 按 prepare 的 Key 顺序建立地址、真实长度、有效标记快照，并 pin 对应 Key |
| Get `indices` | NPU int32 数组，值为 table 行号，不是业务 Key，也不是 slot_id |
| Get `source_offsets` | NPU int64 字节偏移数组，相对每个 Key 自身数据起点；读取整个 tensor 时均为 0 |
| Get `lengths` | NPU int64 长度数组，每条请求的读取字节数；整份读取时为该 Key 的真实长度 |
| Get `request_count` | NPU int32[1]，有效请求前缀的条数 |
| Get `output_offsets` | NPU int64 字节偏移数组，每条结果放到 output 的位置；输出区间不能重叠 |
| Get `output` | 连续的一维 NPU uint8 缓冲区，容量覆盖所有输出区间 |
| `stream` | 使用已有的同一 NPU stream；跨 stream 仍需调用方用 Event 建立依赖 |
| `read_chunk_bytes` | 原 range Get 的 READ 分段选项，默认 0；不参与 slot 分配，也不决定 Key 的大小 |

多个独立 Key 合批并不要求“一批所有对象必须在一个 slot”。分配器逐个检查放置位置，可以在同一 batch 中自动换 slot；每个 Key 自身不跨 slot。旧定长 `batch_get` 仍要求其原有 token 布局，变长整份读取使用 `batch_get_ranges`。

下面在同一个 source 中拼接两个 uint8 tensor，以不同 Key Put，再按相反顺序整份读回。shape/dtype 仍由适配层保存。

```python
import os
import torch
import torch_npu
import kvaf

device_id = 7
torch.npu.set_device(device_id)
device = f"npu:{device_id}"
stream = torch.npu.current_stream()
rt = kvaf.NpuRuntime(
    os.environ["KVAF_BATCH_GET_KERNEL"], 656, 1,
    device_id=device_id, npu_core_count=2, jettys_per_core=2,
    batch_put_kernel_object_path=os.environ["KVAF_BATCH_PUT_KERNEL"],
    transport="urma", allocation_mode="append_only")
handle = None
try:
    rt.link_to(os.environ["KVAF_CPU_SERVICE_ADDR"])
    rt.initialize(2 * 1024**2, 1024**2)  # 两个 1MiB slot
    a = torch.arange(656, dtype=torch.int32, device=device).to(torch.uint8)
    b = torch.full((1968,), 0x5A, dtype=torch.uint8, device=device)
    source = torch.cat([a, b])
    rt.batch_put([101, 202], source, [0, 656], [656, 1968], stream=stream)

    handle = rt.prepare_access_table([202, 101])
    table = rt.access_table_view(handle, device="npu")
    def tensor(values, dtype):
        return torch.tensor(values, dtype=dtype, device=device)
    output = torch.empty((2624,), dtype=torch.uint8, device=device)
    rt.batch_get_ranges(
        table, tensor([0, 1], torch.int32),        # table 的两行：202、101
        tensor([0, 0], torch.int64),              # 各自 Key 的起点
        tensor([1968, 656], torch.int64),
        tensor([2], torch.int32),
        tensor([0, 1968], torch.int64), output,
        stream=stream, synchronize=True)
    assert torch.equal(output, torch.cat([b, a]))
    rt.release_access_table(handle)
    handle = None
    rt.batch_free([202])  # 只删除 b，a 仍可读取；b 的空洞暂不重用
    rt.batch_free([101])  # slot 中无其他对象，现在可整 slot 回收
finally:
    stream.synchronize()
    if handle is not None:
        rt.release_access_table(handle)
    rt.shutdown()
```

`synchronize=True` 会等待完成并检查 range 结果。使用默认 False 时，同 stream 后续 NPU 算子可按序消费 output；CPU 必须等待该 stream 上相关任务完成、检查状态后，才能 release table 和 free Key。table 存续时独立删除被 pin 的 Key 会返回 Busy；追加其他 Key 不使已持有 table 的地址失效。

## 事务与实现边界

新分配器独立于原 fixed 实现，负责 Key→地址/长度、slot 追加游标、slot 引用计数以及 pending/active/pin 状态。Put、range Get、DDR 注册、URMA transport 和 `.asc` kernel 均沿用当前实现。

PreparePut 先验证整批 Key/length。空间不足或中途元数据分配失败时，恢复本批地址预留、slot 游标和空闲池，不会留下半批 Key。成功预留后，Runtime 在 WRITE 成功时提交整批映射；写入路径失败则 Abort。Abort 不回退其他后续预留之后的游标，失败对象占过的区域只在整 slot 空闲后回收。pending 对象也计入 slot 引用，因此不会提前回收仍有未提交对象的 slot。

真实 transport 错误仍遵循现有 Runtime 的完成与清理约定：若设备任务/DMA 是否结束尚不确定，应完成清理、重建 Runtime 后重试；元数据 Abort 本身不是传输完成屏障。此次没有改动 transport 的故障恢复协议。

本版没有对象跨 slot、空洞复用、数据搬移整理、live Key 原地覆盖或动态扩容。它也不改变小请求 range kernel 的调度性能。

## 构建与功能验收

在目标机沿用现有 CMake 配置，重新构建 Host 库和 Python binding（例如 `cmake --build build-npu --parallel`），并使用同一构建目录。CPU service 的分配协议保持不变；本次没有修改 kernel 源文件。

无 NPU 的 Host 检查：

```bash
bash tests/run_append_only_host_tests.sh
```

要求退出 0，且出现 `APPEND_ONLY_META_PASS`、`APPEND_ONLY_RUNTIME_PASS`、`APPEND_ONLY_HOST_PASS`。覆盖多 Key 合槽、自动换槽、精确尾部容量、空间不足整批回滚、交错预留/Abort、table pin、哈希扩容后快照、Key 重用、分配异常注入、payload 与邻居数据，以及原 fixed 的 11 项测试。可通过 `CXX` 选编译器，通过 `KVAF_APPEND_HOST_BUILD_DIR` 指定编译产物目录。

真实 AIV + URMA 功能验收，CPU service 先按原方式启动：

```bash
KVAF_CPU_SERVICE_ADDR=127.0.0.1:34555 DEVICE_ID=7 \
  bash tests/run_remote_cpu_append_only.sh "$PWD/build-npu"
```

目标机脚本先检查省略参数仍为 fixed，再用 append_only 验证非对齐小对象紧密存放、两 slot 内多于两个 Key、逐 Key 整份变长 Get、自动换槽、pin 期间追加、OOM 回滚、删除空洞不重用、整 slot 回收以及邻居 payload 完整性。每次 Get 均比较完整字节并检查输出 guard。只有退出 0 且末尾出现 `APPEND_ONLY_HARDWARE_PASS fixed_default=1 append_only=1` 才算硬件验收通过；`APPEND_ONLY_HARDWARE_FAIL`、超时或缺少末尾 PASS 均不通过。wrapper 默认超时 300 秒，可用 `KVAF_APPEND_TEST_TIMEOUT` 调整。未设置硬件运行开关而直接运行 Python 时返回 77，并输出 `APPEND_ONLY_SKIP`。

2026-11-23 本地 Host 检查通过；两个新 C++ 测试通过 AddressSanitizer/UndefinedBehaviorSanitizer；52 项相关 Python 测试和 6 项构造参数兼容性测试通过。本机没有 CANN/NPU，尚未执行目标机编译与硬件验收，不据此提供带宽结论。

## 与 fixed 对比性能

现有性能脚本新增 `--allocation-mode`，默认 fixed。配置、CSV/JSONL 和日志目录名均记录模式；输出默认仍为 KVAF 同级 `result`，按配置和时间命名。分别运行：

```bash
KVAF_CPU_SERVICE_ADDR=127.0.0.1:34555 DEVICE_ID=7 \
  bash tests/run_remote_cpu_batch_get_ranges_perf.sh "$PWD/build-npu" \
  --tensor-bytes 262144 --repeat 1 --pack-count 4 --slot-size 1048576 \
  --cores 2 --jettys 2 --inflight 2048 --read-chunk-bytes 0 \
  --allocation-mode fixed

# 其余参数相同，将 --allocation-mode 改为 append_only 再运行。
```

这个例子每包仅一个 256KiB tensor，因此四个包就是四个独立 Key。fixed 使用四个 slot；append_only 的四个 Key 可以共用一个 1MiB slot。脚本为公平配置仍分配 `slot_size * pack_count` 的 4MiB DDR，不会按模式自动缩减初始化容量。功能脚本则用两个 slot、多个独立 Key 验证真实共享能力。

`--tensor-bytes 656 1968 --repeat 8192 --pack-count 1` 仍表示一个 Key 包含很多逻辑 tensor，不会自动变成每 tensor 一个 Key。需要每 tensor 独立 Key 时，使用上面的单 tensor/包配置，或在适配层按独立 Key 列表调用 Put。`--alignments` 仍只控制输入包内部与 output 的布局，不是远端分配器的对象对齐选项。性能计时边界和 chunk 解释见 [range 性能说明](README_batch_get_ranges_perf.md)，比较时固定同一 scope 和其余配置。
