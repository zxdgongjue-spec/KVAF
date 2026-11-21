# Packed Put 和 Get range 性能测试与调用方式

新增 `run_remote_cpu_batch_get_ranges_perf.sh` 测量 AIV + URMA 的方案 B。默认每个包拼接 1MiB 和 32MiB 两个逻辑 tensor，共两个包、四条读取请求；同一次 Runtime 内比较不分段、64KiB、1MiB、4MiB 分段。功能日志中 `large_tensors=True` 的 49 组 PASS 已证明目标机通过相应功能用例，但不代表任何带宽结论。

## 运行性能测试

CPU 服务沿用已启动的实例；NPU 机器使用包含 range Get 的构建。新增内容是 Python 和 shell 脚本，无需为它们修改 kernel：

```bash
KVAF_CPU_SERVICE_ADDR=127.0.0.1:34555 DEVICE_ID=7 \
  bash tests/run_remote_cpu_batch_get_ranges_perf.sh "$PWD/build-npu" \
  --tensor-bytes 1048576 33554432 --pack-count 2 \
  --cores 2 --jettys 2 --inflight 2048 \
  --read-chunk-bytes 0 65536 1048576 4194304 \
  --warmup 5 --iterations 30 --output-dir /tmp/kvaf-range-perf
```

默认远端 DDR 约 66MiB，HBM 和 CPU 还需存放 source、output、预期字节及框架缓存；实际总内存占用高于 DDR 大小。`--slot-size 0` 自动取能覆盖最大包的 4096B 对齐容量，DDR 总容量是 slot_size×pack_count。初始化和建链按每个 core/Jetty/inflight 组合进行一次，随后复用 Runtime 测所有 alignment 和 chunk。

无 NPU 也能先看分段数、请求数和内存规格：

```bash
python3 tests/python/batch_get_ranges_perf.py --dry-run
```

默认四条请求合计 66MiB 有效数据。计划中的物理 READ 数量如下，这是软件计算值，不是硬件计数器：

| read_chunk_bytes | 总 READ WQE 数 |
| --- | ---: |
| 0 | 4 |
| 65536 | 1056 |
| 1048576 | 66 |
| 4194304 | 18 |

## 扫描参数

| 参数 | 默认与含义 |
| --- | --- |
| `--tensor-bytes` | `1048576 33554432`，每个包内的逻辑 tensor 字节长度列表，可混合长度。 |
| `--repeat` | 1，在同一包内重复上述列表；不是跨 stream 并发提交数。 |
| `--pack-count` | 2，包数量，也是 Put 的 Key 数量。 |
| `--alignments` | 1，可传 `1 64 512` 扫描包内 tensor 起点及 output 起点的相对对齐；真实 length 不补齐。 |
| `--read-chunk-bytes` | `0 65536 1048576 4194304`，逐个测试，0 表示取消软件分段。该参数不改变 Put。 |
| `--cores`、`--jettys`、`--inflight` | 各接受整数列表，扫描笛卡尔积；默认分别是 2、2、2048。非零 inflight 必须不小于 2×core 数。Put 使用独立 transport，Jetty 数为 min(Get Jetty 数, 2)。 |
| `--request-order` | 默认 reverse，可选 sequential/random；输出结果保持自然 tensor 顺序，随机排列使用固定 seed。 |
| `--guard-bytes` | 默认 0，输出默认紧密排列；可设为 16 等值插入不应写入的间隙做边界检查，间隙不计入有效 payload。alignment 造成的空隙也会检查。 |
| `--warmup`、`--iterations` | 默认 5、30。首次注册、预先验证、warmup 都不计入正式样本。 |
| `--no-events` | 只测 CPU 调用到完成，不采集 NPU Event；适合运行环境不支持 timing Event 的情况。 |
| `--output-dir` | CSV 和原始样本目录，不指定则创建临时目录并打印路径；同目录再次运行会覆盖文件。 |

例如扫描 core 与 Jetty：

```bash
KVAF_CPU_SERVICE_ADDR=127.0.0.1:34555 DEVICE_ID=7 \
  bash tests/run_remote_cpu_batch_get_ranges_perf.sh "$PWD/build-npu" \
  --cores 1 2 4 --jettys 1 2 4 --inflight 2048 \
  --read-chunk-bytes 0 1048576 4194304
```

仅有少量逻辑请求时，增加 core 未必有收益；当前 kernel 按逻辑请求划分 core，不会将一个大 tensor 的物理分段自动分给多个 core。可增加 pack-count 或 repeat，提供更多独立请求。

模拟 8192 个 656B tensor 共用一个包：

```bash
KVAF_CPU_SERVICE_ADDR=127.0.0.1:34555 DEVICE_ID=7 \
  bash tests/run_remote_cpu_batch_get_ranges_perf.sh "$PWD/build-npu" \
  --tensor-bytes 656 --repeat 8192 --pack-count 1 --read-chunk-bytes 0
```

这与旧定长脚本的随机跨表访问、kernel 计时口径不同，不能仅凭“656/8192”就直接比较带宽。若研究 padding，选用有非对齐尾部的长度，例如 `--tensor-bytes 656 1968 1048589 33554449 --alignments 1 64 512`。默认 1MiB/32MiB 均为 512 的倍数，源包布局不会因这些 alignment 设置产生差别。

## 结果和计时边界

成功要求退出码为 0，并在最后出现 `RANGE_PERF_PASS`。`RANGE_PERF_FAIL`、超时退出和没有 PASS 的部分 CSV 均不能算完成验收。`--dry-run` 只打印 PLAN，不会生成硬件 PASS。未授权运行时 Python 脚本退出 77；wrapper 自动设置运行开关，缺依赖或不可用 NPU 会报错。wrapper 默认 1200 秒超时，可用 `KVAF_RANGE_PERF_TIMEOUT` 调整。

输出 `summary.csv`、`samples.jsonl` 和 `RANGE_PERF_RESULT` 行：

| operation / scope | 测量内容 |
| --- | --- |
| put / call_to_completion | CPU 调用 batch_put 到同步返回。含本次 Key 预留、元数据处理、WRITE、完成/状态检查及提交元数据；不含 source 打包、上传、提前注册和 free。当前不能覆盖 live Key，每次 Put 前在计时外 free，未持有 table。 |
| get_ranges / call_to_completion | CPU 调用 Get-range 到 stream.synchronize 返回。单独采样，不插入 Event。含 Python 绑定、查询检查、缓存注册查找、下发及执行等待，不含后续 completion check 的状态 D2H。 |
| get_ranges / submit | CPU 下发调用的耗时；返回时未必已搬完数据，因此不计算带宽。 |
| get_ranges / stream_event | 另一次采样中，围绕 Get 记录两个 NPU timing Event。包含前置状态清零和 range kernel，也可能包含 CPU 准备/下发造成的 stream 空闲间隙。它不是 kernel 内部时间，不能直接等同旧脚本的 kernel_total。 |

所有请求、tensor 和输出缓冲区保持固定并复用，测试同一已驻留数据，是 warm/repeated-access 基准，不代表随机大工作集或并发 vLLM 负载。输入生成、打包、上传、建表和逐字节校验均在计时外；每个 chunk 在采样前后验证有效数据及存在的输出间隙，每次 Get 都在计时外检查设备状态，错误不会被当作高速结果。

带宽按一次有效 payload 计算：`GB/s = useful_bytes / elapsed_seconds / 1e9`，另列 GiB/s。Put 的 padding 不计入 useful_bytes，但会包含在 storage_bytes 和真实 WRITE 中；Get 的 useful_bytes 是各请求真实 length 之和。WQE、CQE 和链路协议开销不计入有效字节。原始 ns 样本保存在 JSONL，可复算 mean/p50/p90/p99；不把提交耗时、最短样本或不同计时口径混作完成带宽。

## 当前 Put 与 Get range 接口

```python
rt.batch_put(keys, source, input_offsets, lengths, stream=None)
rt.batch_get_ranges(table, indices, source_offsets, lengths,
                    request_count, output_offsets, output,
                    stream=None, synchronize=False, profile=False,
                    read_chunk_bytes=0)
rt.check_batch_get_ranges_status(stream=None)
```

Put 的 `keys/input_offsets/lengths` 是 CPU 整数序列或连续的一维 CPU int64 tensor，每项描述一个包；source 是调用方拼好的连续 NPU 字节数据。input_offsets 从整个 source 起点算，lengths 是各包总长度，含包内 padding，不能超过 slot_size。Put 成功返回才提交包级地址与长度。已有 Key 要先解除 table 的锁并 free 后才能再写。

Get 的 table 由 `prepare_access_table(keys)` 和 `access_table_view(handle, device="npu")` 构建。indices 是 table 行号，NPU int32[R]；source_offsets/lengths/output_offsets 都为 NPU int64[R]，单位为字节；request_count 为 NPU int32[1]，只消费有效前缀；output 是连续 NPU uint8 字节数组。所有设备 tensor 连续、一维、位于该 Runtime 的同一 NPU。输出区间不得重叠。

下面是 FP16 tensor 按一个包 Put，再按 B、A 顺序 Get 的完整例子；tensor shape/dtype 由适配层保留，Runtime 只管理包。

```python
import torch
import torch_npu
import kvaf

torch.npu.set_device(7)
device, stream = "npu:7", torch.npu.current_stream()
rt = kvaf.NpuRuntime(
    "build-npu/bin/kvaf_execute_batch_get_kernels.o", 656, 1,
    device_id=7, npu_core_count=2, jettys_per_core=2, transport="urma",
    batch_put_kernel_object_path="build-npu/bin/kvaf_execute_batch_put_kernels.o",
)
rt.link_to("127.0.0.1:34555")
rt.initialize(ddr_bytes=64*1024*1024, slot_size=64*1024*1024)
a = torch.full((512, 1024), 1, dtype=torch.float16, device=device)   # 1MiB
b = torch.full((4096, 4096), 2, dtype=torch.float16, device=device) # 32MiB
na, nb = a.numel()*a.element_size(), b.numel()*b.element_size()
source = torch.cat([a.contiguous().view(torch.uint8).reshape(-1),
                    b.contiguous().view(torch.uint8).reshape(-1)])
rt.batch_put([0], source, [0], [na+nb], stream=stream)
handle = rt.prepare_access_table([0])
table = rt.access_table_view(handle, device="npu")
indices = torch.tensor([0, 0], dtype=torch.int32, device=device)
starts = torch.tensor([na, 0], dtype=torch.int64, device=device)
lengths = torch.tensor([nb, na], dtype=torch.int64, device=device)
count = torch.tensor([2], dtype=torch.int32, device=device)
targets = torch.tensor([0, nb], dtype=torch.int64, device=device)
output = torch.empty(na+nb, dtype=torch.uint8, device=device)
rt.batch_get_ranges(table, indices, starts, lengths, count, targets, output,
                    stream=stream, read_chunk_bytes=0)
restored_b = output[:nb].view(torch.float16).reshape(b.shape)
restored_a = output[nb:].view(torch.float16).reshape(a.shape)
rt.check_batch_get_ranges_status(stream)  # CPU 等待并检查设备结果
assert torch.equal(restored_a.cpu(), a.cpu())
assert torch.equal(restored_b.cpu(), b.cpu())
rt.release_access_table(handle)
rt.batch_free([0])
rt.shutdown()
```

默认 Get 只提交，NPU 同 stream 按序执行；跨 stream 仍需 Event。请求 tensor、设备表、Host handle 和 output 必须保留至执行完成。`synchronize=True` 可以代替本例末尾的显式 completion check。`profile=True`、Graph capture、AICPU 和 UBmem 的 range Get 当前不支持。

## read chunk bytes 对性能的影响

它只影响 Get 的物理 READ 拆分，不影响 Key、slot 分配、Put 包大小和返回 tensor 的真实长度。较小的分段增加 WQE 构造、提交和 CQ 管理次数；较大分段减少这些开销，但不保证更快。

当前每个 core 的一个 wave 从 Jetty 0 开始，每个 Jetty 最多提交 128 条 READ。单个 core 的 READ 少于首个 Jetty 的可提交额度时，后面的 Jetty 可能没有工作。默认案例 chunk=0 时每 core 只有两条 READ；chunk=64KiB 时每 core 有 528 条，可以使用更多 Jetty 和 wave。小在途预算也可能让 Jetty 0 就用完该 wave 的额度。因而分段会同时改变提交开销、Jetty 利用和在途字节量，不能只根据 READ 数量做带宽预测。

`read_wqes_expected`、`core_read_wqes_expected` 和 `core_payload_bytes_expected` 是按请求及 core 分配规则计算的数量，便于解释请求数与字节负载，不是实测硬件计数。建议固定 cores、Jettys、inflight、布局和请求顺序后比较 chunk；再用相反 chunk 顺序复测，检查缓存和运行先后影响。实际单条 READ/WRITE 能力仍以目标硬件为准，不能把 uint32 长度字段上限当作设备保证。

## 本地验证

本机没有 torch_npu/NPU，不能提供真实带宽数值。新增 host 单元测试验证布局与 guard、READ 数量、时延/带宽换算、非法参数、计时外状态检查和 Key 回收，以及模拟设备下的完整测量流程与 CSV/JSONL 输出。可执行：

```bash
python3 -m unittest discover -s tests/python -p test_batch_get_ranges_perf.py
```

模拟设备测试只验证脚本流程与记账，不证明真实 Event 时序、CANN 行为或硬件性能。需要目标机跑出 `RANGE_PERF_PASS` 后，使用对应 CSV 中同一 scope 的结果比较。
