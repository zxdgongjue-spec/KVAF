# NPU 直接读取对端 HBM：batch_get_raw_addr

同一个 NpuRuntime 可访问 CPU DDR，并通过独立 fullmesh 链路读取另一张卡的 HBM。本路径不使用 AccessTable、keys 或 DDR MetaManager。B 可分别注册一块供 CPU Pull 的 HBM 和一块供 NPU Pull 的 HBM。

## 生产端 B

```python
source = kvaf.NpuPullHbmSource(device_id=7, npu_eid=b_fullmesh_eid)
ticket = source.register(hbm_for_a, stream=producer_stream)
port = source.listen("127.0.0.1:0")
# 应用把可达的控制地址与 ticket 发给 A，等待 A attach 完成。
# 每批：应用先完成生产同步，再把 ready 与业务元数据发给 A。
producer_stream.synchronize()
ready = source.ready()
# 收到 A 应用的完成通知后：
source.release_batch(ready)
```

register 保留连续 NPU Tensor，初始化实际注册在 attach 时完成。Tensor 起点无需页对齐：底层向下对齐 4 KiB 并覆盖整个 Tensor，逻辑源起点仍为 Tensor 的实际地址。注册至探测完成期间保持数据稳定。

ready 不携带范围、不等待生产 stream；应用负责同步。release_batch(ready) 只在读取完成/已安全取消后成功，cancel_batch(ready) 只允许取消尚未被读取方获取的批次。两者都核验源身份和 sequence。注销前先释放批次、由 A detach，再 source.shutdown()。

## 读取端 A

```python
runtime.attach_hbm(b_address, ticket, local_eid=a_fullmesh_eid)
# 收到 ready 后，应用计算三个等长、连续的一维 int64 NPU Tensor。
with torch.npu.stream(a_stream):
    src = torch.tensor([7, 100003, 250019], dtype=torch.int64, device=output.device)
    dst = torch.tensor([100, 500, 4096], dtype=torch.int64, device=output.device)
    sizes = torch.tensor([1, 656, 100001], dtype=torch.int64, device=output.device)
runtime.batch_get_raw_addr(ready, src, dst, sizes, output, stream=a_stream)
# 成功后由应用通知 B 释放批次；所有批次结束后 runtime.detach_hbm(ticket)。
```

源偏移相对于注册 HBM，目的偏移相对于 output，均以字节计。output 和三数组必须在 Runtime 所在 NPU，连续且存储不重叠。与 BatchGet 默认路径一致，调用方负责保证源/目的范围合法、目的段互不重叠，并保证每条长度为 1..UINT32_MAX 且不超过设备支持的单次 READ 上限。每条请求直接生成一个 WQE，不拆分；默认不预扫描长度或源/目的范围。`validate_ranges=True` 可启用正长度、32 位长度字段和源/目的边界诊断，但不检测目的重叠或查询设备能力。形状、数量及元数据存储/output 别名检查保留。应用保证元数据在传入 stream 上就绪，并保持到完成/恢复结束。

当前只保留该 NPU Tensor 重载。成功返回 `{"state": "done", "sequence": ready.sequence}`，包含 READ 完成和生产端完成确认。output 按 `(地址, 字节数)` 复用 BatchGet transport 的注册缓存，每批只 Release 引用，detach/shutdown 时统一注销；状态和队列缓冲也跨批复用。Python 绑定保留各缓存 output 的 Tensor 引用直到对应 link 成功 detach 或 runtime 成功 shutdown；期间可读写内容，但不能 resize 或替换底层 storage。C++ 调用方负责保持这些缓冲区的生命周期。同一连接使用越多不同 output，缓存占用也越大，固定 output 的稳定态只注册一次。

KVAF 按源身份保存初始化的连接，每批从 `ready.source` 自动查找；应用无需维护本地连接编号。同一 Runtime 可以绑定多个源，分别持有通信资源。关闭时原样传回初始化的 ticket；attach 失败但清理未完成时，同样用 `detach_hbm(ticket)` 重试清理，不解析报错中的内部编号。未决 DMA 仍会阻止关闭。

未决时用原批次的 `resolve_raw_addr(ready)` 重试控制或清理，绝不重发 READ；源身份或批次号不符会被拒绝。`pending_raw_addr` 表示仍需保留资源。若获取批次的响应丢失而尚未发 READ，恢复会取消并返回错误，不能将 output 视为有效。

可选 `profile=True` 在成功结果中增加 `profile` 字典，单位均为微秒：

```python
result = runtime.batch_get_raw_addr(
    ready, src, dst, sizes, output, stream=a_stream, profile=True)
print(result["profile"])
```

| 字段 | 计时范围 |
| --- | --- |
| `host_total_us` | 原生 HbmReader::Read 全流程；不含 Python 绑定的参数处理和结果字典构造 |
| `validation_us` | Host 设备、源身份、数组形状和元数据/输出重叠检查 |
| `acquire_us` | 向生产端获取当前批次的控制请求 |
| `buffers_us` | 首次状态/队列分配、每批状态清零、首次计时事件创建；Host 状态数组复用 |
| `register_output_us` | 获取缓存的 output 注册和通信实体，首次遇到该地址/长度时注册 |
| `launch_us` | Host 提交 raw READ 算子调用 |
| `synchronize_us` | Host 等待传入 stream 完成 |
| `kernel_event_us` | 同一 stream 中紧邻算子前后两个设备事件之间的时间 |
| `status_us` | 最终状态回读与检查 |
| `cleanup_us` | 释放 output 注册引用；MR、通信实体和状态缓冲保留到 detach/shutdown |
| `complete_us` | 向生产端确认本批完成 |
| `event_record_us` / `event_query_us` | Host 提交计时事件 / 查询计时结果的开销 |
| `host_other_us` | 原生总时间减去上述 Host 阶段的余量 |

`kernel_event_us` 使用 `ACL_EVENT_TIME_LINE` 和 `aclrtEventElapsedTime`，毫秒转微秒。
它覆盖 raw READ 算子中的可选长度/边界诊断、WQE 生成与提交和 CQ 等待，属于设备 stream 的事件区间，
并非仅网络传输时间或各核时间的累加；也可能包含事件与设备调度开销。
该区间与 Host 的 launch/synchronize 重叠，不能再加到 Host 阶段总和中。
用法依据 [CANN 设备事件计时接口](https://www.hiascend.com/doc_center/source/en/canncommercial/800/apiref/appdevgapi/aclcppdevg_03_0090.html)。
事件记录或结果查询失败不改变已成功的数据传输结果；此时 `kernel_event_us=None`，
`kernel_event_error` 提供 ACL 返回码。事件创建失败发生在发起 READ 之前，接口返回异常。
默认 `profile=False`，不创建/记录计时事件；计时事件按 reader 复用，detach 时销毁。

双方必须显式配置 fullmesh EID：B 构造源时的 npu_eid 和 A attach_hbm 的 local_eid。CPU↔NPU 路由独立配置，自动 CLOS/pair-info 发现不能替代 fullmesh。

## 算子与资源

raw READ V2 在同一次算子中直接生成 WQE、提交并等待 CQ。WQE 总数等于输入请求数，各核按请求数量划分任务；默认不预扫描长度，也不执行逐段源/目的边界诊断；显式启用诊断时，各核独立检查整批，在任何 READ 提交前拒绝越界输入。目的区间两两比较已删除。三数组不回传 Host，没有独立准备算子。所有请求均按下标直接索引，无分段前缀表或查找；沿用 BatchGet 的 WQE 编码、双 bank、多 Jetty、inflight 窗口和 CQ 处理。Host 只读取最终状态，设备状态每批清零以排除上批结果。

仅建链的 all-lane 探测仍使用内部 Host 绝对地址描述符模式，不再作为公开数据读取接口。探测失败必须确认 kernel 已退出且 initiator 队列成功销毁后才能换候选。

KVHR v4 控制协议只传源/连接身份和批次号，不传三数组或 payload。A 的 HbmReader 保存连接/导入 MR、USER_CTL Jetty/CQ、队列和未决任务；B 的 NpuPullHbmSource 保存 Tensor、导出 MR、目标 Jetty、客户端身份和批次状态。各通道分别协商 TP，CPU Pull 的共享 TP 路径独立。

同一个 reader 的控制请求复用 TCP 连接，双方启用 TCP_NODELAY；服务端轮询多连接，空闲连接不阻塞其他请求。服务端通过 v4 Response 的保留字段声明连接复用能力，旧服务端自动回退为单请求连接。通信结果不确定时关闭连接，由既有 resolve 流程恢复，不自动重发请求。Acquire/Complete 和同步返回继续保留，用于保证 B 的 ready/release_batch 复用时序。

每批最多 4096 段，对应最多 4096 个 WQE，每段长度由 WQE 的 32 位字节数字段表示（1..UINT32_MAX），实际还受设备单次 READ 能力限制；每个源绑定一个 reader，同时一个活动批次，默认 2 核 × 2 Jetty，总计最多 4 个 Jetty。接口同步，不支持同一 reader 的并发调用或 graph capture。

使用 `KvafExecuteBatchGetRawAddrAivKernelV2`，旧 V1 object 不兼容。NPU 运行库和 raw kernel object 需一起重编译。底层 provider 返回的内存身份与路由 EID 分开保留；不手动改写 MR/Jetty 的导出身份。

## 编译与硬件验证

CPU service 与 NPU 扩展需同时重编译并重启，以使用 KVRP/KVHR v4。
在 Linux 测试机已有的 CANN、torch_npu 和 liburma 环境中更新源码，然后重新编译 NPU 侧：

```bash
cmake -S . -B build-shared-ddr \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DKVAF_BUILD_REMOTE_CPU_SERVICE=OFF \
  -DKVAF_BUILD_PYTHON=ON \
  -DKVAF_ENABLE_CPU_OFFLOAD=ON \
  -DKVAF_ENABLE_AIV_BATCH_GET=ON \
  -DKVAF_ENABLE_AIV_BATCH_PUT=ON \
  -DBUILD_TESTING=OFF \
  -DPython_EXECUTABLE="$(command -v python)"
cmake --build build-shared-ddr -j8
export PYTHONPATH="$PWD/build-shared-ddr/python${PYTHONPATH:+:$PYTHONPATH}"
python tests/python/test_hbm_read_raw_addr.py \
  --cpu 127.0.0.1:28999 --device-a 6 --device-b 7 --rounds 3 \
  --npu-eid-a-raw '<A_FULLMESH_EID>' \
  --npu-eid-b-raw '<B_FULLMESH_EID>'
```

运行前将两个占位符替换为对应卡的实际 fullmesh EID。

脚本自动启动 B 子进程和 B 的临时控制监听端口。在同一个 A runtime 中逐轮验证：

1. A BatchPut 写入 DDR 的 marker 保持正确。
2. CPU Pull B HBM₁ 到共享 DDR，A BatchGet 读回并逐字节比较。
3. A 直接 READ B HBM₂ 的 1/656/100001 字节三段，检查目的位置间隔、padding 和 NPU 消费者可见性。
   HBM₂ 使用页内偏移为 `0xc00` 的连续 Tensor view，覆盖非页对齐的注册场景。
4. B 两块缓冲区分别收到确认后复用，内容每轮变化。
5. 先释放批次、A detach、B shutdown，最后关闭 A runtime。

可独立指定 `--npu-eid-a`（DDR）、`--npu-eid-a-raw`（A—B，A 的 fullmesh）、
`--npu-eid-b-cpu`（CPU—B）、`--npu-eid-b-raw`（A—B，B 的 fullmesh）。
成功输出 `HBM_RAW_PASS two_b_buffers=1 shared_runtime=1 no_access_table_for_hbm=1`。
出现 unresolved 时测试会保留进程/缓冲区，需先检查两端日志和确认传输停止。
若 attach 已明确失败且没有未解决的传输，A 会通知 B 关闭源，避免 B 等待 attached 超时。

本地无 Ascend 硬件时，`kvaf_batch_get_raw_addr_kernel_model_test` 运行实际 AIV 源码的 CPU 模型；
`kvaf_hbm_read_test` 使用真实 TCP/source/client 与假 ACL/HCCP 验证状态、每段一个 WQE、丢 ACK 和清理。
这些测试不替代 Linux CANN 编译和真实 A—B READ 验收。

## 随机离散包性能测试

新增 `tests/python/bench_hbm_read_raw_addr.py`，脚本中的 A 为读取端，B 为生产端。
先按上文重新编译 NPU 扩展和 raw kernel，启动 CPU 服务，再执行：

```bash
export PYTHONPATH="$PWD/build-shared-ddr/python${PYTHONPATH:+:$PYTHONPATH}"
python tests/python/bench_hbm_read_raw_addr.py \
  --cpu 127.0.0.1:28999 --device-a 6 --device-b 7 \
  --npu-eid-a-raw 000000000047050000100000df00d806 \
  --npu-eid-b-raw 000000000047050000100000df00f807 \
  --packets 4096 --packet-bytes 656 --source-space-bytes 268435456 \
  --warmup 10 --iterations 100 --profile-warmup 1 --profile-iterations 20 \
  --raw-kernel build-shared-ddr/bin/kvaf_execute_batch_get_raw_addr_kernels.o
```

EID 对应此前测试机的 fullmesh 配置，其他机器需替换。CPU 服务仅用于当前 NpuRuntime
初始化；数据直接从 B HBM 读取到 A HBM，不经过 CPU DDR。

- `--packets`：包数量，1..4096；兼容 `--blocks`。
- `--packet-bytes`：每包字节数，所有包相同；兼容 `--block-bytes`。每包直接生成一个 WQE，长度须为 1..UINT32_MAX 且满足设备单次 READ 上限。
- `--source-space-bytes`：B 注册的 HBM 空间大小，至少容纳所有包。以包大小划分槽位，随机抽取不重复槽位，源地址按随机顺序读取；不足一个槽位的尾部不使用。
- `--seed`：默认 123，可复现布局。目的 output 连续排列，首尾各留 64 字节 guard。
- `--aiv-cores` / `--jettys-per-core`：默认 2/2，总 Jetty 数不超过 4。
- `--max-inflight-total`：默认 2048，含义与 BatchGet 一致。
- `--profile-warmup` / `--profile-iterations`：单独的内部计时轮次，默认 1/20；设为 `--profile-iterations 0` 即只运行未插桩的带宽测试。
- `--validate-ranges`：打开可选正长度、32 位长度字段及源/目的边界诊断，默认关闭；不包含目的重叠检测。日志标记 `validate_ranges` 和 `output_registration_cache=1`，首次 output 注册计入 warmup（warmup 为 0 时计入测量）。

输出 `NPU_HBM_BENCH_PASS` 的吞吐和分位数只使用 `--iterations` 轮的
Python `batch_get_raw_addr()` 完整调用时间，未打开内部计时。
另外运行带 `profile=True` 的轮次，输出 `NPU_HBM_INTERNAL_PROFILE_STAGE`，
包括 `python_total`、上述全部 Host 阶段和 `kernel_event` 的 mean/P50/P95/P99/min/max。
内部计时轮次不混入带宽统计；各分位数按 `(N-1)*p` 线性插值，不能将不同阶段的 P99 相加。
`NPU_HBM_INTERNAL_PROFILE_INFO` 列出有效设备计时样本数和计时错误码，缺失的设备计时不会作为零值统计。

源注册、连接探测、源写入、三组 NPU int64 Tensor 准备、output 清空、应用 IPC、数据校验均在传输计时之外。
每轮交替输入内容，并检查 NPU 消费者和完整回读数据（含 guard）。
`NPU_HBM_BENCH_PROFILE_STAGE` 另列读取端接口/校验时间及生产端 ready/release_batch 时间。
源与目的进程的时钟不相减。两端初始化与清理时间不纳入稳定态逐批统计。
