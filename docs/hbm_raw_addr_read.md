# NPU 直接读取对端 HBM：batch_get_raw_addr

同一个 NpuRuntime 可访问 CPU DDR，并通过独立 fullmesh 链路读取另一张卡的 HBM。纯 NPU Pull 不需要 CPU service 或 DDR 初始化，也不使用 AccessTable、keys 或 DDR MetaManager。B 可为 CPU Pull 和 NPU Pull 分别注册 tensor 列表；两套注册与连接独立，不要求是同一批底层数据。

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

ready 不携带范围、不等待生产 stream；应用负责同步。release_batch(ready) 只在读取完成/已安全取消后成功，cancel_batch(ready) 只允许取消尚未被读取方获取的批次。两者都核验源身份、sequence 和 mr_id。注销前先释放批次、由 A detach，再 source.shutdown()。

## 多 MR 共用 NPU 连接

```python
# B：固定列表中的每个 tensor 独立注册；列表共用连接。
source = kvaf.NpuPullHbmSource(device_id=7, npu_eid=b_fullmesh_eid)
ticket = source.register(tensors, stream=producer_stream)
port = source.listen("127.0.0.1:0")
# Connector 转交地址和 ticket，等待 A attach 完成。
# produce(tensors[i])，并等待实际生产 stream 完成后：
ready = source.ready(tensors[i])
# Connector 把 ready 与业务元数据发给 A；不查询 MR 或通信句柄。
# A 的调用与下文一致，src_offsets 相对于本次选中的 tensors[i]。
# A 完成后应用通知 B：
source.release_batch(ready)
```

- `register()` 接收单 tensor，或 1～4096 个 tensor 的 list/tuple。每个 tensor 连续、非空、在源对象指定的 NPU 上；逻辑范围不能重叠。列表固定到 shutdown，Python 保留所有 tensor 引用；注册期间不能 resize、释放或替换底层 storage。
- `ready(tensor)` 按设备、起点和长度匹配注册项；不同范围的子视图不被接受。单 MR 仍可调用 `ready()`；多 MR 必须显式选择。一次调用的全部段属于同一 MR，不跨 MR。
- 一个源同时一个未释放批次；同一 tensor 重复 Ready 返回该批次，选择其他 tensor 被拒绝。sequence 在源内统一递增。未选中的 tensor 可由应用继续生产，选中的 tensor 直到完成确认和 release 之前保持稳定。
- B 使用一个 context、每个 tensor 一个 MR/token ID，以及共用的一组 4 个目标 Jetty。A 使用一个 context、逐个导入的 MR 和共用的 1～4 个发起 Jetty/CQ；TP 保留现有逐 Jetty 协商方式。
- attach 的 Open 返回首 MR 和数量，Memory 请求取得其余 MR；注册/导入不在运行阶段重复执行。每个 MR 都做覆盖全部启用 Jetty 的小块 READ 校验，全部通过才返回成功。初始化部分失败按整组清理，清理不确定时保留原 ticket。
- A 以 `source_id + source_epoch` 找到连接，以 `ready.mr_id` 选择远端 MR。源描述符的 `mr_bytes`/`mr_count` 是只读信息，`source_bytes` 仍指首 tensor 的大小，不是所有 tensor 的地址包络或长度之和。应用原样转交即可。
- output 的本地 MR 仍按地址/长度缓存。按“MR ID + output 注册”另外缓存轻量通信描述；它们引用同一 SQ/CQ/Jetty/TP，远端基址、长度、token 和内存 EID 各不相同。首次组合生成并上传一次，后续复用；SQ/CQ 计数与队列状态跨 MR 延续。raw READ 算子及其 ABI 不变。
- 新 ticket 为 version 2，增加 MR 长度列表；可解析旧 version 1 单 MR ticket。ReadyToken 的 pickle 包含 mr_id，旧单 MR 状态按 mr_id=1 解析。线协议 KVHR 则要求两端均为 v5，不与 v4 互通。
- CPU Pull 与 NPU Pull 不共用 MR 或状态。若应用给两者注册同一底层 tensor，覆写前应等待两条路径的消费者都完成。

## 读取端 A

```python
runtime = kvaf.NpuRuntime(
    device_id=6,
    batch_get_raw_addr_kernel_object_path=raw_kernel,
    npu_core_count=2, jettys_per_core=2,
)
runtime.initialize()
assert runtime.initialized and not runtime.ddr_initialized
runtime.attach_hbm(b_address, ticket, local_eid=a_fullmesh_eid)
# 收到 ready 后，应用计算三个等长、连续的一维 int64 NPU Tensor。
with torch.npu.stream(a_stream):
    src = torch.tensor([7, 100003, 250019], dtype=torch.int64, device=output.device)
    dst = torch.tensor([100, 500, 4096], dtype=torch.int64, device=output.device)
    sizes = torch.tensor([1, 656, 100001], dtype=torch.int64, device=output.device)
runtime.batch_get_raw_addr(ready, src, dst, sizes, output, stream=a_stream)
# 成功后由应用通知 B 释放批次；所有批次结束后 runtime.detach_hbm(ticket)。
```

`initialize()` 只检查所启用功能的配置、选择本卡设备并标记基础就绪；不访问 CPU、不申请 DDR、不建立远端链路。
实际 raw kernel 加载、MR/Jetty/TP 建链与探测在 `attach_hbm()` 中完成。
纯 NPU Pull 只配置 raw kernel，不要求普通 BatchGet 的 token 参数或 `enable_cpu_offload=True`。
混合使用时仍配置普通 BatchGet/BatchPut kernel；raw 路径为空时可从 BatchGet kernel 同目录推导，二者都为空时 attach 报错。

需要 DDR 时，在基础初始化后调用 `link_to(cpu_address)` 和 `initialize_ddr(ddr_bytes, slot_size)`。
可在 attach HBM 前后初始化 DDR，两条路径的资源相互独立。
旧 `link_to(...) → initialize(ddr_bytes, slot_size)` 仍可用，内部依次完成基础和 DDR 初始化。
`initialized` 只表示基础就绪，`ddr_initialized` 才表示 DDR 就绪；DDR 初始化失败可能保留基础就绪状态。
`shutdown_ddr()` 保留 HBM 链路、输出注册缓存和 CPU 目标，可用于重试失败的 DDR 清理及随后重新初始化 DDR。
它仍要求 CPU Pull 的任务及源租约已退役。`shutdown()` 关闭所有模块；未决 DMA 不会被强制释放，恢复接口不重发 READ。

源偏移相对于注册 HBM，目的偏移相对于 output，均以字节计。output 和三数组必须在 Runtime 所在 NPU，连续且存储不重叠。与 BatchGet 默认路径一致，调用方负责保证源/目的范围合法、目的段互不重叠，并保证每条长度为 1..UINT32_MAX 且不超过设备支持的单次 READ 上限。每条请求直接生成一个 WQE，不拆分；默认不预扫描长度或源/目的范围。`validate_ranges=True` 可启用正长度、32 位长度字段和源/目的边界诊断，但不检测目的重叠或查询设备能力。形状、数量及元数据存储/output 别名检查保留。应用保证元数据在传入 stream 上就绪，并保持到完成/恢复结束。

当前只保留该 NPU Tensor 重载。成功返回 `{"state": "done", "sequence": ready.sequence}`，包含 READ 完成和生产端完成确认。output 按 `(地址, 字节数)` 复用 BatchGet transport 的注册缓存，每批只 Release 引用，detach/shutdown 时统一注销；状态和队列缓冲也跨批复用。Python 绑定保留各缓存 output 的 Tensor 引用直到对应 link 成功 detach 或 runtime 成功 shutdown；期间可读写内容，但不能 resize 或替换底层 storage。C++ 调用方负责保持这些缓冲区的生命周期。同一连接使用越多不同 output，缓存占用也越大，固定 output 的稳定态只注册一次。

KVAF 按源身份保存初始化的连接，每批从 `ready.source` 自动查找；应用无需维护本地连接编号。同一 Runtime 可以绑定多个源，分别持有通信资源。关闭时原样传回初始化的 ticket；attach 失败但清理未完成时，同样用 `detach_hbm(ticket)` 重试清理，不解析报错中的内部编号。未决 DMA 仍会阻止关闭。

未决时用原批次的 `resolve_raw_addr(ready)` 重试控制或清理，绝不重发 READ；源身份、批次号或 MR ID 不符会被拒绝。`pending_raw_addr` 表示仍需保留资源。若获取批次的响应丢失而尚未发 READ，恢复会取消并返回错误，不能将 output 视为有效。

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
| `register_output_us` | 获取 output 注册及选定 MR 的通信描述；首次 output 注册、首次 MR/output 组合上传描述，之后均复用 |
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

KVHR v5 运行阶段控制消息只传源/连接身份、MR ID 和批次号，不传三数组或 payload；初始化另行交换各 MR 的导出信息。A 的 HbmReader 保存连接/导入 MR、USER_CTL Jetty/CQ、队列和未决任务；B 的 NpuPullHbmSource 保存 Tensor、导出 MR、目标 Jetty、客户端身份和批次状态。各通道分别协商 TP，CPU Pull 的共享 TP 路径独立。

同一个 reader 的控制请求复用 TCP 连接，双方启用 TCP_NODELAY；服务端轮询多连接，空闲连接不阻塞其他请求。服务端通过 Response 的保留字段声明连接复用能力，同版本未声明复用的服务端使用单请求连接；不同线协议版本拒绝通信。通信结果不确定时关闭连接，由既有 resolve 流程恢复，不自动重发请求。Acquire/Complete 和同步返回继续保留，用于保证 B 的 ready/release_batch 复用时序。

每批最多 4096 段，对应最多 4096 个 WQE，每段长度由 WQE 的 32 位字节数字段表示（1..UINT32_MAX），实际还受设备单次 READ 能力限制；每个源绑定一个 reader，同时一个活动批次，默认 2 核 × 2 Jetty，总计最多 4 个 Jetty。接口同步，不支持同一 reader 的并发调用或 graph capture。

使用 `KvafExecuteBatchGetRawAddrAivKernelV2`，旧 V1 object 不兼容。NPU 运行库和 raw kernel object 需一起重编译。底层 provider 返回的内存身份与路由 EID 分开保留；不手动改写 MR/Jetty 的导出身份。

## 编译与硬件验证

此次多 MR NPU Pull 需同时重编译、重启 A/B 的 NPU 运行库与 Python 扩展（KVHR v5）。CPU Pull 的 KVRP v5 未改，已有匹配的 CPU service 无需因此更新。raw READ kernel ABI 保持 V2。
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
2. CPU Pull 本轮选中的 B tensor 到共享 DDR，A BatchGet 读回并逐字节比较。
3. A 直接 READ 同一个 B tensor 的 1/656/100001 字节三段，检查目的位置间隔、padding 和 NPU 消费者可见性。
   两条路径独立注册相同的 tensor 列表；tensor 使用页内偏移为 `0xc00` 的连续 view，覆盖非页对齐的注册场景。
4. 每轮只生产一次数据，两条路径分别对照同一份期望值验证。A 在 CPU Pull 和 NPU Pull 均完成且校验通过后通知 B，B 释放两条路径的批次后才复用源；内容每轮变化。
5. 先释放批次、A detach、B shutdown，最后关闭 A runtime。

可独立指定 `--npu-eid-a`（DDR）、`--npu-eid-a-raw`（A—B，A 的 fullmesh）、
`--npu-eid-b-cpu`（CPU—B）、`--npu-eid-b-raw`（A—B，B 的 fullmesh）。
`--hbm-mrs` 同时控制两条路径的源 tensor 数量。设置 78 时，B 只分配 78 个不同大小、非页对齐的 tensor，由 CPU Pull 和 NPU Pull 分别注册，因此各有 78 个 MR、各自共用一组连接；设置 1 时两条路径注册同一个 tensor。每轮两条路径选择相同索引，NPU 读取复用同一个 output。不指定 rounds 时默认 `max(3, 2 * hbm_mrs)`，78 MR 对应 156 批。
成功输出 `HBM_RAW_PASS npu_mrs=78 cpu_mrs=78 shared_source_tensors=1 shared_runtime=1 no_access_table_for_hbm=1 connector_api=1`（单 MR 时两项均为 1）。
出现 unresolved 时测试会保留进程/缓冲区，需先检查两端日志和确认传输停止。
若 attach 已明确失败且没有未解决的传输，A 会通知 B 关闭源，避免 B 等待 attached 超时。

本地无 Ascend 硬件时，`kvaf_batch_get_raw_addr_kernel_model_test` 运行实际 AIV 源码的 CPU 模型；
`kvaf_hbm_read_test` 在关闭 CPU Pull 构建能力时，使用真实 NpuRuntime/TCP/source/client 与假 ACL/HCCP 验证纯 NPU Pull、每段一个 WQE、丢 ACK 和清理。
`kvaf_hbm_read_with_ddr_test` 另外验证两种初始化顺序、DDR 失败/关闭/重新初始化不影响 raw READ，以及旧 PutHandle 被拒绝。
上述两个集成测试还覆盖 78 MR 注册/探测、选择错误、部分注册失败及多 MR 丢失完成通知恢复。
`kvaf_hbm_read_transport_test` 运行生产 Host 注册/缓存代码，验证 78 份不同 token/EID 的描述共用 output MR 和队列、缓存命中、上传失败、部分导入和重试清理。
这些测试不替代 Linux CANN 编译和真实 A—B READ 验收。

## 随机离散包性能测试

新增 `tests/python/bench_hbm_read_raw_addr.py`，脚本中的 A 为读取端，B 为生产端。
先按上文重新编译 NPU 扩展和 raw kernel，再执行；此脚本不需要 CPU 服务：

```bash
export PYTHONPATH="$PWD/build-shared-ddr/python${PYTHONPATH:+:$PYTHONPATH}"
python tests/python/bench_hbm_read_raw_addr.py \
  --device-a 6 --device-b 7 \
  --npu-eid-a-raw 000000000047050000100000df00d806 \
  --npu-eid-b-raw 000000000047050000100000df00f807 \
  --packets 4096 --packet-bytes 656 --source-space-bytes 268435456 \
  --warmup 10 --iterations 100 --profile-warmup 1 --profile-iterations 20 \
  --raw-kernel build-shared-ddr/bin/kvaf_execute_batch_get_raw_addr_kernels.o
```

EID 对应此前测试机的 fullmesh 配置，其他机器需替换。脚本已删除 `--cpu`、`--cpu-candidate` 参数和占位 DDR 分配。
初始化日志包含 `ddr_initialized=0 cpu_service_required=0`；数据直接从 B HBM 读取到 A HBM。
纯 NPU 部署可用 `KVAF_ENABLE_AIV_BATCH_GET=ON`、`KVAF_ENABLE_AIV_BATCH_PUT=OFF`、
`KVAF_ENABLE_CPU_OFFLOAD=OFF`、`KVAF_BUILD_REMOTE_CPU_SERVICE=OFF` 构建。
上面的混合功能验收脚本仍需 CPU service 和 CPU Pull/BatchPut 构建能力。

- `--packets`：包数量，1..4096；兼容 `--blocks`。
- `--packet-bytes`：每包字节数，所有包相同；兼容 `--block-bytes`。每包直接生成一个 WQE，长度须为 1..UINT32_MAX 且满足设备单次 READ 上限。
- `--source-space-bytes`：B 每个注册 tensor 的 HBM 空间大小，至少容纳本批所有包。以包大小划分槽位，随机抽取不重复槽位，源地址按随机顺序读取；不足一个槽位的尾部不使用。
- `--source-count`：注册多少个独立 tensor/MR（默认 1，范围 1～4096），共用连接，逐批轮换；HBM 总空间约为 source-count × source-space-bytes。多 MR 时预热至少覆盖全部 MR/output 组合，实际 warmup 会打印。各 MR 的数据不同，同一 MR 再次使用时改变数据；期望值生成和校验不计入传输时间。
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

### 78 MR 的纯 NPU 验收与性能测量

无需启动 CPU service。以下沿用测试机曾通过的 device/fullmesh EID 映射：

```bash
python tests/python/bench_hbm_read_raw_addr.py \
  --device-a 7 --device-b 6 \
  --npu-eid-a-raw 000000000047050000100000df00f807 \
  --npu-eid-b-raw 000000000047050000100000df00d806 \
  --source-count 78 --source-space-bytes 1048576 \
  --packets 1024 --packet-bytes 656 \
  --warmup 78 --iterations 156 --profile-iterations 78 \
  --validate-ranges \
  --raw-kernel build-shared-ddr/bin/kvaf_execute_batch_get_raw_addr_kernels.o
```

注册成功日志包含 `HBM_READ_READY ... jettys=4 mrs=78 probe=all_mrs_all_lanes_pass`。
通过标志为 `NPU_HBM_BENCH_PASS ... source_count=78`。功能验收可保留 validate-ranges，
比较默认快速路径性能时去掉此开关，并使用相同包数/包大小/空间大小对照 source-count=1。
