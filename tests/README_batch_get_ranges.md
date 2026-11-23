# 方案 B 可变长读取使用与验收

本版在 `dynamicIO_yq` 分支实现 AIV + URMA 的方案 B：上层把多个 tensor 拼成一个连续包，使用一个 Key 写入一个 slot；读取时，按包内的 byte offset 和真实 length 分别取出完整 tensor。原来的 `batch_put` 和定长 `batch_get` 保留，新增 `batch_get_ranges`。

性能测试、FP16 完整调用示例和分段性能解释见 [Packed Put 和 Get range 性能测试与调用方式](README_batch_get_ranges_perf.md)。

可选的“一 tensor 一 Key、多 Key 共用 slot、只追加分配”见 [append_only 分配模式](README_append_only.md)。默认仍为 fixed，Put/Get range 接口保持不变。

当前版本支持 eager 执行、同包多次读取、多包混合读取、不同长度、乱序请求和独立的输出 byte offset。AICPU gather_read、UBmem、Graph capture、range Get 的 stage/kernel profile 尚未适配。CPU 模型检查功能正确性，真实 NPU 上的 CANN 编译、URMA 连通性、缓存可见性及性能需要另行验收。

## 1 数据布局和接口变化

以三个 uint8 tensor 为例：A 为 656B，B 为 1968B，C 为 1024B，合用 Key 3。

| 布局 | A 的 offset 和 length | B 的 offset 和 length | C 的 offset 和 length | Put 整包长度 |
| --- | --- | --- | --- | --- |
| 紧密拼接 | 0，656 | 656，1968 | 2624，1024 | 3648B |
| 每个 tensor 起点按 64B 对齐 | 0，656 | 704，1968 | 2688，1024 | 3712B |
| 每个 tensor 起点按 512B 对齐 | 0，656 | 1024，1968 | 3072，1024 | 4096B |

padding 只改变下一个 tensor 的起点。读取 A 仍传 `length=656`，不会把 padding 当作 A 的有效数据。对齐是适配层的可选布局策略，Runtime 不会自动补齐；这版也没有据此宣称某种对齐更快。

两种 offset 的基准不同：`batch_put.input_offsets` 指向整个 source 中各个**包**的起点；`batch_get_ranges.source_offsets` 指向相应 Key 的包中各个 **tensor** 的起点。读取字节地址为：

```text
远端 DDR 地址 = device_access_table.addresses[indices[i]] + source_offsets[i]
本地 NPU 地址 = output.data_ptr() + output_offsets[i]
搬运长度     = lengths[i]
```

### Put 保持原接口

```python
runtime.batch_put(keys, source, input_offsets, lengths, stream=None)
```

| 参数 | 含义和要求 |
| --- | --- |
| `keys` | 每个包的存储 Key。Python 整数序列或连续的一维 CPU int64 tensor。当前 Key 也是 slot 序号，必须在初始化得到的 slot 数量范围内。 |
| `source` | 调用方拼好的连续 NPU tensor。建议用一维 uint8 表达原始字节。Runtime 不帮忙拼接，也不保存 tensor 的 shape/dtype。 |
| `input_offsets` | 每个包在 source 中的 byte offset，CPU 整数序列或一维 CPU int64 tensor。 |
| `lengths` | 每个包实际写入的总字节数，包括包内 padding。与 keys 和 input_offsets 数量相同；大于 0，不超过 slot_size，也不能越过 source。AIV Put 的单包 WQE 长度还要求不超过 UINT32_MAX，超过会拒绝，不能静默截断。 |
| `stream` | 执行 Put 的调用方 torch.npu.Stream，默认当前 stream。当前 AIV Put 完成该 stream 的同步等待后返回；成功后才提交包的地址和长度元数据。 |

一个 tensor 超过包容量时，要在上层拆包或增大 slot_size。方案 B 没有自动跨 slot，也没有多个独立存储 Key 共享 slot 的分配器。多个 tensor 共享的是一个包 Key，删除、覆盖和 slot 复用均以整个包为单位。

### Get 新增接口

```python
runtime.batch_get_ranges(
    device_access_table, indices, source_offsets, lengths,
    request_count, output_offsets, output,
    stream=None, synchronize=False, profile=False, read_chunk_bytes=0,
)
runtime.check_batch_get_ranges_status(stream=None)
```

下表中的 NPU tensor 都必须连续、一维、非空，位于该 Runtime 的同一 NPU。R 是请求数组容量，N 是 device table 容量；即使 request_count=0，也保留至少 1 行请求容量和非空 output。

| 参数 | 类型和形状 | 含义 |
| --- | --- | --- |
| `device_access_table` | `DeviceAccessTable` | 由有效 Host table handle 构建的设备查询表，含 addresses:int64[N]、valid:uint8[N]、新增的 lengths:int64[N]。lengths 是每个已提交包的真实总长度，不是 slot_size，也不是各个 tensor 长度。不要修改表中内容。 |
| `indices` | NPU int32[R] | 请求所读包在 table 中的行号，**不是 Key**。若 prepare 的顺序是 `[7, 3]`，Key 3 对应 index=1。多条请求可以重复同一行。 |
| `source_offsets` | NPU int64[R] | 在该包内的 byte offset，必须非负。无需统一对齐。 |
| `lengths` | NPU int64[R] | 每条请求真实读取的字节数，必须大于 0，可以互不相同。offset + length 不得超过表中该包的实际长度。 |
| `request_count` | NPU int32[1] | 本次有效请求数，范围 0..R；只消费各请求数组的 `[0, request_count)` 前缀。不是长度，也不是 Put 的包数。未使用后缀可以含无效值。 |
| `output_offsets` | NPU int64[R] | 每条结果放入 output 的 byte offset，必须非负，且 offset + length 不越过 output。顺序可以离散。**调用方保证所有有效输出区间不重叠，本版不检测重叠。** |
| `output` | NPU uint8[总容量] | 调用方分配的连续字节输出区。仅写有效请求对应的区间，间隙和尾部不写入。 |
| `stream` | torch.npu.Stream | 默认当前 stream。Get kernel 完成所有 READ 的 CQ 确认后结束，同 stream 后续 NPU 算子按序消费结果。跨 stream 通过 Event 建立依赖。 |
| `synchronize` | bool，默认 False | False 时本次调用提交后返回；True 时 CPU 等待该 stream 此前提交的任务完成，并检查每个 core 的 range Get 设备状态。对 NPU 的按序执行和对 CPU 的返回时机应分别理解。 |
| `profile` | bool，当前必须 False | 当前不支持 range Get profiling。设 True 会报错。旧的 kernel profile 接口不能用来解释 range Get 的耗时。 |
| `read_chunk_bytes` | Python int，默认 0，只能通过关键字传入 | 单条物理 READ 的软件分段大小，范围 0..UINT32_MAX。0 取消人为分段，单条 READ 仍受 uint32 长度字段和实际设备能力限制；非零时按指定大小切分。调用方应根据已确认的硬件能力设置，本版不自动查询设备并裁剪该值。 |

返回值为 `None`；失败抛异常。`check_batch_get_ranges_status(stream)` 是带设备状态检查的同步等待，必须使用最近一次 range Get 的执行 stream。仅 `stream.synchronize()` 可以等待执行结束，但**不会把 range kernel 的逻辑校验失败自动转换为 Python 异常**。

已去掉固定 64KiB 分段。默认 read_chunk_bytes=0，在长度不超过 UINT32_MAX 时，每个逻辑请求生成一条 READ。显式设为 1MiB、4MiB 等正数时，才按该大小拆分，最后一段读取实际剩余字节，不截断，也不强制读满 padding。

| 配置 | 读取 1MiB tensor 的 READ 数 | 读取 32MiB tensor 的 READ 数 |
| --- | --- | --- |
| 默认 0 | 1 | 1 |
| 65536，即旧 64KiB 策略 | 16 | 512 |
| 1048576，即 1MiB | 1 | 32 |
| 4194304，即 4MiB | 1 | 8 |

这只是软件 WQE 生成规则，不能据此推断硬件已支持 32MiB 的单条 READ。当前 Direct HCCP 封装没有暴露单条 READ 上限的查询，UINT32_MAX 也只是长度字段上限。大 tensor 整包 Put 同样要满足设备实际 WRITE 能力。拿到已确认的单条传输能力后，可选择不分段或指定合适的 read_chunk_bytes；分段大小不要求与 AICPU 路径相同。性能优劣仍需实测。

`batch_get_max_inflight_total` 仍对物理 READ 数量生效；0 时 range kernel 按每 Jetty 最多 128 个 READ 的有界 wave 执行。增大单条 READ 后，同一数量预算可能对应更多在途字节。变长路径的 CQ 轮询预算按每个 post 的总字节量扩大并饱和到 uint32 上限，避免沿用小包的固定等待预算；它是轮询次数预算，不是按秒校准的超时。

当前实现沿用共享状态区和队列，后续同一 Runtime 的 range Get、定长 Get、Put 或 profiling 配置会先等待并检查上一次尚未检查的 range Get。因此不能据默认 False 推断它支持同一 Runtime 上多个并发的 Get；由单个调度方按序提交。

## 2 最小调用示例

下面假定已在 NPU 机器上导入 torch、torch_npu、kvaf，且 CPU 服务已启动。完整可运行示例见 `python/test_npu_batch_get_ranges.py`，其中 `pack()` 是参考打包适配层。

```python
torch.npu.set_device(0)
dev = "npu:0"
runtime = kvaf.NpuRuntime(
    "build-npu/bin/kvaf_execute_batch_get_kernels.o",
    batch_get_token_bytes=656, batch_get_tokens_per_block=1,
    device_id=0, npu_core_count=2, jettys_per_core=2,
    batch_put_kernel_object_path="build-npu/bin/kvaf_execute_batch_put_kernels.o",
    transport="urma", aicpu_batch_get_mode="",  # 选择 AIV
)
runtime.link_to("CPU_HOST_IP:28999")
runtime.initialize(ddr_bytes=16 * 1024 * 1024, slot_size=1024 * 1024)
# 旧 constructor 的 token_bytes/tokens_per_block 只约束旧定长 Get，
# 不要求新 range Get 的长度为 656，也不要求包大小为二者乘积。

a = torch.arange(656, dtype=torch.int32, device=dev).to(torch.uint8)
b = torch.full((1968,), 7, dtype=torch.uint8, device=dev)
c = torch.full((1024,), 9, dtype=torch.uint8, device=dev)
source = torch.cat([a, b, c])
stream = torch.npu.current_stream()
runtime.batch_put([3], source, [0], [3648], stream)

handle = runtime.prepare_access_table([3])
table = runtime.access_table_view(handle, device="npu")
indices = torch.tensor([0, 0], dtype=torch.int32, device=dev)
starts = torch.tensor([656, 0], dtype=torch.int64, device=dev)   # 先 B 后 A
lengths = torch.tensor([1968, 656], dtype=torch.int64, device=dev)
count = torch.tensor([2], dtype=torch.int32, device=dev)
targets = torch.tensor([0, 1968], dtype=torch.int64, device=dev)
output = torch.empty(2624, dtype=torch.uint8, device=dev)
runtime.batch_get_ranges(table, indices, starts, lengths, count, targets, output,
                         stream=stream, synchronize=False)
consumer_result = output.clone()  # 同 stream：排在 Get kernel 后
runtime.check_batch_get_ranges_status(stream)  # CPU 等待并检查错误
assert torch.equal(consumer_result[:1968].cpu(), b.cpu())
assert torch.equal(consumer_result[1968:].cpu(), a.cpu())

runtime.release_access_table(handle)  # 完成之前不能 release/free
runtime.batch_free([3])
runtime.shutdown()
```

上层自行保存 `(逻辑 tensor ID -> 包 Key、offset、length、dtype、shape)`。Runtime 只保存包级元数据，不会自动推断 tensor 边界或恢复 shape。非 uint8 tensor 可在适配层转成连续原始字节；取回后按原 dtype 和 shape 解释。只要提交正确的真实 length，每个 tensor 就完整取回，不会因同批其他 tensor 更短而截断。

### Table 和资源生命周期

`prepare_access_table` 构建“table 行号到包地址、实际长度、有效标记”的查询表，并锁住相关 Key；它不复制包内容。`access_table_view(..., device="npu")` 将查询信息物化为设备 tensor。设备表扩容的额外行 valid 和 lengths 均为 0。

保留 Host handle、设备表、所有请求 tensor 和 output，直到 Get 执行完成；执行期间不要并发改写请求数组、设备表或输出区。建议先调用 completion check，再 release table。仅丢弃 Python 的 DeviceAccessTable 对象不会代替 `release_access_table`。即使 Get 已经完成，table 未 release 时，`batch_free` 仍会报 Busy。

如果消费者位于其他 stream，还要等消费者使用 output 完成才能复用/释放 output；跨 stream 的标准顺序是 `producer.record(ready) -> transfer.wait_event(ready) -> Get -> transfer.record(done) -> consumer.wait_event(done)`。测试脚本有完整写法。stream 是 Runtime/上下文管理的执行队列，需要显式建立跨 stream 依赖。

## 3 无硬件本地测试

在仓库根目录执行，只需要支持 C++17 的编译器：

```bash
bash tests/run_batch_get_ranges_host_tests.sh
```

可用 `CXX=clang++` 选择编译器，或设置 `KVAF_RANGE_HOST_BUILD_DIR=/tmp/kvaf-range-host` 保留指定位置的二进制。脚本默认在临时目录保留编译产物，并打印路径。CMake 环境也可构建并运行 `kvaf_batch_get_ranges_runtime_test`、`kvaf_batch_get_ranges_operator_test` 和 `kvaf_batch_get_kernel_model_test`。

必须退出码为 0，且依次出现：

```text
RANGE_GET_MODEL_PASS: ...
BatchGet kernel CPU model: ... passed
RANGE_GET_RUNTIME_PASS: ...
RANGE_GET_OPERATOR_PASS: ...
RANGE_GET_HOST_PASS
```

| 检查层 | 通过表示什么 |
| --- | --- |
| Runtime | 实际 MetaManager 提交包长度，支持不同请求 length、乱序范围、active prefix、边界校验；table pin/free/reuse 正确；未适配 backend 返回 NotImplemented。 |
| Host operator | 实际 operator 代码在模拟 ACL/transport 下正确区分 enqueue 与等待，映射设备错误，拒绝 capture，串行复用状态，启动/等待失败时保留清理所需资源。 |
| AIV kernel CPU 模型 | 直接编译生产 .asc 控制流，校验 1MiB/32MiB 级 tensor 默认单 READ、可配置分段的精确 READ 数量、guard 字节、1/2/4 core 和 1/2/4 Jetty、不同预算、跨 128-WQE post、SQ/CQ owner 回绕、超过旧轮询预算的延迟完成和 CQ 错误隔离；还执行原定长 Get 模型回归。 |

CPU shim 不模拟 NPU 缓存一致性、真实 SIMT 调度、pipeline 时序或网络，因此这些 PASS 不能替代下面的硬件验收。

## 4 真实 AIV 加 URMA 端到端验收

CPU 和 NPU 两端都使用本分支代码。首次准备平台 CANN、URMA/HCCP、驱动和网络后执行：

```bash
# CPU 主机
bash build.sh cpu --build-dir build-cpu
./build-cpu/runtime/kvaf_cpu_service --listen 0.0.0.0 --port 28999

# NPU 主机：先 source 当前 CANN 的 set_env.sh
bash build.sh npu --without-aicpu --build-dir build-npu
KVAF_CPU_SERVICE_ADDR=CPU_HOST_IP:28999 DEVICE_ID=0 \
  bash tests/run_remote_cpu_batch_get_ranges.sh "$PWD/build-npu"
```

不要复用旧 `.o` 或旧 Python binding；取消固定分段后的 entry 为 `KvafExecuteBatchGetRangesAivKernelV2`，旧 range V1 二进制会被拒绝。缺失该符号会明确提示重新构建。原定长 Get 的 kernel entry 保持不变。此测试只创建一次 Runtime、一次 link/initialize，随后复用它测试所有 padding 和大小。

默认 `--cores 2 --jettys 2 --inflight 2048 --alignments 1 64 512 --iterations 3 --read-chunk-bytes 0`，使用原小 tensor 集合做快速检查。加 `--large-tensors` 会增加 1MiB 和 32MiB tensor，slot_size 改为 64MiB，16 个 slot 共申请 1GiB 远端 DDR。比较默认不分段和显式 1MiB/4MiB 分段：

```bash
(
set -e
for chunk in 0 1048576 4194304; do
  KVAF_CPU_SERVICE_ADDR=CPU_HOST_IP:28999 \
    bash tests/run_remote_cpu_batch_get_ranges.sh "$PWD/build-npu" \
    --large-tensors --read-chunk-bytes "$chunk" --iterations 3
done
)
```

这组脚本检查功能，不统计性能；每次运行都应独立出现 HARDWARE_PASS。大包整读仍包含包内 padding。单条传输超过设备能力时，应按实际设备限制调整布局/参数，再重新验收。若要验证单个 tensor 跨 wave，可使用：

```bash
KVAF_CPU_SERVICE_ADDR=CPU_HOST_IP:28999 \
  bash tests/run_remote_cpu_batch_get_ranges.sh "$PWD/build-npu" \
  --device 0 --cores 4 --jettys 4 --inflight 8 --iterations 20 \
  --large-tensors --read-chunk-bytes 1048576
```

这里 `--inflight 8` 配合 4 core，每 core 最少预算为 2，可验证长 tensor 跨 wave；也建议再以 2048 和 0 各跑一次。显式选 NPU EID 时沿用环境变量 `KVAF_NPU_EIDS=32位hexEID[,32位hexEID]`；每条 EID 必须非零、不同，数量不超过 core×Jetty。

### 硬件结果判断

必须同时满足：进程退出码为 0，所有组出现 `RANGE_GET_CASE_PASS`，最后出现 `RANGE_GET_HARDWARE_PASS`。默认最终统计为 49 个验收组，具体的预期异常检查还会单独打印 CASE_PASS。

| 验收组 | 判断方法 |
| --- | --- |
| Put 和 metadata | 超过 slot_size 的 Put 被拒绝；两个不同包存入 Key 3/7，table 顺序改成 7/3，Host/Device 长度和 padding 行精确匹配；整包读回含 padding 的所有字节匹配。 |
| 变长和乱序 | 默认每包包含 1、7、656、1968、65535、65536、65537、153600B 的逻辑 tensor；large 模式再加入 1MiB、32MiB。整份取回、重复读取、跨包混排，逐字节与 pattern 比较，4KiB 页头含独立编号以检测错误重复读取前一段。 |
| 0、1、5、全量前缀 | 后缀故意填负数；只允许有效前缀写 output。对所有 output 字节比较，间隙和尾部必须仍为 0xa5。 |
| 同 stream 和跨 stream | 同 stream 的 clone 消费结果正确；producer/transfer/consumer 通过 Event 协同后，consumer 结果完全一致。 |
| 无效参数 | dtype/数组容量/负 offset/零长度/int64 极大值/无效 table 行/越过 output/count 越界，必须抛预期异常；随后合法请求仍能成功。 |
| 生命周期 | table 未释放时 free 必须失败；release 后旧 table 读取必须失败；free 后相同 Key 写新内容，使用新 table 读出新数据。 |
| 兼容性 | 最后用旧定长 batch_get 读取独立 656B 包，内容仍正确。 |

直接运行 Python 脚本而未设置 `KVAF_RUN_NPU_TESTS=1` 会打印 `RANGE_GET_SKIP`，退出 77，CTest 显示 Skipped。wrapper 会自动开启硬件测试；此时依赖缺失、NPU 不可用、网络失败、数据不符都会返回非零并打印 `RANGE_GET_HARDWARE_FAIL`，不会当作通过。

设备参数错误通过 ValueError 报告，并附 core/result/request；例如 result=4 为源查询/范围错误，8 为输出范围错误，1 为结构/count 错误。CQ timeout、CQ error、队列已污染等传输错误会报 RuntimeError，需重建 Runtime/链路后重试。某 core 参数失败时其他 core 可能已写部分输出，失败后应丢弃整次输出；Get 不保证失败时原子回滚。wrapper 在有 GNU timeout 的系统上默认设 300 秒上限，可用 `KVAF_RANGE_TEST_TIMEOUT=600` 调整；超时通常退出 124，判为未通过。CTest 同样设 300 秒上限。保留两端日志和 CANN plog 排查。

## 5 本地验证记录

2026 年 11 月 21 日，macOS 本机执行 `bash tests/run_batch_get_ranges_host_tests.sh`，kernel 模型、Runtime 和 Host operator 三组测试均通过。更新后的模型对 1MiB 和 32MiB+17B tensor 验证默认一条 READ，以及 64KiB、1MiB、4MiB、UINT32_MAX 分段设置；检查实际 READ 数量和完整字节输出，也验证延迟完成超过旧固定轮询预算后仍成功。Python 和 shell 语法已检查；未开启硬件开关时返回 77，符合“未执行”的约定。

三组新增测试还通过 AddressSanitizer 和 UndefinedBehaviorSanitizer。现有 Runtime 清理重试、backend 配置、MetaManager 的 11 项检查以及 42 项 Python 配置/脚本回归均通过。Python 回归复现命令为：

```bash
cd tests/python
python3 -m unittest test_build_script test_dsa_keys test_batch_get_inflight_config \
  test_batch_get_unified test_batch_get_multicard_perf test_npu_eids
```

本机没有 CANN、torch_npu 和 NPU，尚未完成真实 CANN 编译和硬件端到端验收。此处不提供延迟、带宽或 padding 效率结论；拿到硬件测试最后的 `RANGE_GET_HARDWARE_PASS` 后，才算完成 AIV + URMA 功能验收。

## 6 功能日志与旧性能脚本的判断

目标机提供的运行日志末尾为 `RANGE_GET_HARDWARE_PASS cases=49 ... large_tensors=False`，说明默认小 tensor 集合在三种 alignment 下已通过功能验收。标为 `RANGE_GET_CASE_PASS ... expected=ValueError/KeyError` 的行属于故意输入非法参数的负例通过，不是未处理的运行失败。这份日志尚未覆盖 1MiB 和 32MiB，需要带 `--large-tensors` 重新运行；它也不包含旧性能测试的带宽数据。

`run_remote_cpu_batch_get_miss_count_perf.sh` 沿用定长 `batch_get`，每条 READ 搬运 token_bytes。656B token 不触发 range 分段，8192 是该 sweep 的有效请求数量。`read_chunk_bytes` 只用于 `batch_get_ranges`，不能用于调节旧性能脚本的带宽。当前旧性能脚本及其参数默认值与方案 B 改动前相同。

为排除共享 helper 导致的编译变化，定长 kernel 的源文件恢复为方案 B 改动前的版本；变长的动态 CQ 轮询预算由独立 helper 处理。此隔离不代表已经确认性能下降原因或验证带宽恢复，仍需目标机对比。聚焦 656B、8192 个请求、保留原 core/Jetty 扫描，可运行：

```bash
KVAF_CPU_SERVICE_ADDR=127.0.0.1:34555 DEVICE_ID=7 \
KVAF_BATCH_GET_TOKEN_BYTES=656 KVAF_BATCH_GET_MISS_COUNTS=8192 \
KVAF_BATCH_GET_MISS_SWEEP_LOG_DIR=/tmp/kvaf-656-8192-after \
  bash tests/run_remote_cpu_batch_get_miss_count_perf.sh "$PWD/build-npu" \
  --warmup 5 --iterations 30
```

如果原命令还带有后端、graph、访问顺序或设备选择参数，复测时继续保留。默认在途预算仍为 2048；可用 `--max-inflight-total 0` 与 `--max-inflight-total 2048` 单独比较 whole-batch 和窗口调度，但它是独立实验，不能把切换预算后的结果当作同配置代码回归的证明。

每个配置的日志会保存 `BATCH_GET_PERF_CONFIG`、`BATCH_GET_RESULT` 及 kernel total/post/completion_wait 三类 profile。比较同一 core、Jetty、EID/UB device、时钟换算值和相同统计口径；区分 GB/s 与 GiB/s、kernel 带宽与端到端带宽。先对比原始 cycles 和 post/CQ 等待的变化，再判断是否是提交或网络完成等待增加。
