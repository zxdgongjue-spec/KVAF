# NPU 直接读取对端 HBM：batch_get_raw_addr

D（读取端，以下示例 A）每张卡的 NpuRuntime 提供一个控制监听入口。P（生产端，以下示例 B）主动连接 D，一次 register 完成注册、元数据交换、硬件建链和逐 MR READ 探测。应用不再转交 export_ticket，也不调用 source.listen 或 runtime.attach_hbm/detach_hbm。

## 初始化：D 先监听，P 一次注册

```python
# D / A：不需要 CPU service 或 DDR 初始化。
runtime = kvaf.NpuRuntime(
    device_id=6,
    batch_get_raw_addr_kernel_object_path=raw_kernel,
    npu_core_count=2, jettys_per_core=2,
)
runtime.initialize()
receiver = runtime.listen_hbm(
    listen_address="0.0.0.0:29000",
    advertise_address="192.168.1.20:29000",  # P 可访问的 D 地址
    local_eid=a_fullmesh_eid,
)
# 应用发现/Connector 将 receiver 原样提供给 P；不解析内部字段。

# P / B：register 成功返回 None，表示 D 已经保存连接并完成全部探测。
source = kvaf.NpuPullHbmSource(device_id=7, npu_eid=b_fullmesh_eid)
source.register(tensors, receiver=receiver, stream=producer_stream)
# tensors 也可以是单个连续 NPU Tensor。
```

receiver 是可序列化字典，包含 version=1、control_address、receiver_id 和 receiver_epoch。绑定 0.0.0.0 时必须给 advertise_address；本机测试可用 listen_hbm("127.0.0.1:0", local_eid=...) 自动选择端口。此端口只交换控制消息，HBM payload 通过 NPU READ 搬运。

NPU↔NPU 两端分别配置 fullmesh EID；TCP 地址与 EID 独立，CPU 路由的 CLOS/pair-info EID 不能替代 fullmesh。

D 自动保存 `(source_id, source_epoch) → 连接`，包括控制 TCP、MR 表、Jetty/CQ/TP 和批次状态。D 的监听线程串行处理初始化，运行时复用每个源的长连接；P 的源对象运行后台控制响应线程，无监听端口。一次接入只绑定一个 D，多 D 使用各自的源对象，通信资源独立。

register 内部同步传入的生产 stream。应用在注册和探测期间保持所有源数据稳定。初始化失败时 Python 仍保留源 Tensor 引用，调用 source.shutdown() 重试清理；不能用新对象覆盖未决资源。D 只有基本 initialize 就能提供该功能，CPU DDR 可以在 NPU 连接建立前后独立初始化或关闭。

## 多 MR 和生命周期

- register 接收 1～4096 个 Tensor（单 Tensor 或 list/tuple）。每个连续、非空、属于源 NPU，逻辑范围不重叠。整个注册期间底层 storage 地址和大小不得改变。起点可以不按页对齐，底层注册覆盖向下对齐的区间，逻辑偏移仍相对 Tensor 起点。
- 每个 Tensor 独立 MR；同一个 Source 中的全部 MR 共用一组通信资源。生产端使用四个目标 Jetty，读取端按配置使用 1～4 个发起 Jetty，各通道仍分别协商 TP。
- 初始化时通过 Open/Memory/Channel 交换全部 MR 和通道信息，每个 MR 探测全部启用通道。全部成功后 Confirm，并向 register 调用返回最终结果。
- ready(tensor) 按设备、基址和大小匹配 MR；单 MR 可用 ready()。每个 Source 同时只有一个已发布批次，全部段属于该批次选中的一个 MR。同一 Tensor 重复 ready 返回原批次。
- CPU Pull 和 NPU Pull 的注册、连接和状态独立。如果共享同一底层 Tensor，应用必须等所有消费者完成后再覆盖。

## 运行阶段：读取就绪即返回，应用负责通知

```python
# P：生产完成并同步后发布。
producer_stream.synchronize()
ready = source.ready(tensors[i])
connector.send(ready, business_metadata)  # 应用自己的业务消息

# D：根据业务元数据准备 NPU 上三个等长连续的一维 int64 Tensor。
with torch.npu.stream(a_stream):
    src = torch.tensor([7, 100003, 250019], dtype=torch.int64, device=output.device)
    dst = torch.tensor([100, 500, 4096], dtype=torch.int64, device=output.device)
    sizes = torch.tensor([1, 656, 100001], dtype=torch.int64, device=output.device)
runtime.batch_get_raw_addr(ready, src, dst, sizes, output, stream=a_stream)
# 返回时 output 已可用；Runtime 不发送或等待批次完成通知。
connector.send(("reuse", ready))  # D 的应用通过自己的通道通知 P。
# P 收到并核对本批的 reuse 消息后：
source.release_batch(ready)  # 在 P 侧调用
```

ready 发布本批读取许可，P 保持该 Tensor 内容稳定，直到应用确认消费者完成并调用 release。D 本地检查源、MR、批次和数组布局，直接启动 raw READ 算子；成功表示 stream 已完成、所有核的 status 校验通过。每批不发送 AcquireRaw、Complete 或 Cancel 控制请求，Runtime 不提供完成通知接口。三数组留在读取端 NPU，不经过控制 TCP。

`release_batch(ready)` 和源端 `cancel_batch(ready)` 只核对批次身份并释放本地冻结状态，不核验远端是否结束。应用先通过自己的消息通道确认读取成功，或确认读取端已经禁止该批次再发起 READ，然后才能释放并覆盖源。bench 使用已有 Pipe：A 读取并校验成功后发 `reuse`，B 核对轮次后 release；通知在读取计时之外。通知丢失或 DMA 不确定时，应用继续保留源数据和 output。

`src_offsets` 相对本批源 Tensor，`dst_offsets` 相对 output，单位均为字节。数组和 output 必须在 D 本卡、连续、存储不重叠。调用方保证合法范围、目的区间不重叠、长度在 1..UINT32_MAX 且符合设备单次 READ 上限。一段直接生成一个 WQE，无 64 KiB 拆分；validate_ranges=True 可选检查长度及边界，不检查目的重叠或设备能力。

成功返回 `{"state": "done", "sequence": ready.sequence}`。output 的本地 MR 按地址/大小缓存，按 MR ID/output 组合复用轻量通信描述，SQ/CQ 状态跨批次与 MR 延续。连接关闭时注销硬件缓存，Python 保守持有 output Tensor 到 Runtime shutdown；C++ 调用方保持 output 到连接关闭。缓存期间不要 resize 或替换 storage。

### 取消、断线和关闭

- 跳过一个已发布批次时，D 调用 `runtime.cancel_raw_addr(ready)`，只在本地记录该批次不可再读，不发送通知。应用将取消结果通知 P，再由 P 调用 cancel_batch/release_batch。P 的源对象无法自行判断 D 是否还可能读取。
- 同一源的 READ 和取消串行；迟到或重复的已完成/已取消批次不会再次启动 READ。未决任务先 resolve_raw_addr，不能直接取消或覆盖。
- `resolve_raw_addr` 只处理同一 ReadyToken 的本地清理，不发送完成通知、不重放 READ；DMA 仍不确定时继续返回 InDoubt。应用消息重试、去重和批次释放由应用负责。
- 初始化后，控制 TCP 只用于注册、重连和关闭，不阻塞正常数据读取。P 后台重新建立 TCP 时，D 校验同一接收端代次、源身份和 MR 清单，复用原硬件连接。
- DMA 状态不确定仍保留资源并返回 InDoubt，TCP 断开不代表 DMA 已结束。接收端重启或会话退役后，旧连接不能自动创建一套新硬件资源。
- 正常关闭：应用停止任务 → P release_batch → P source.shutdown() → D runtime.shutdown()。source.shutdown 内部通知 D 关闭发起端资源，再关闭 P 导出资源并确认退役，无需应用持有 export_ticket。失败继续保留对象并重试；D 全局 shutdown 也会关闭所有源连接。
- NPU 控制协议升级为 KVHR v8，要求 A/B 两端一起更新，旧版本拒绝连接。CPU Pull 的 KVRP v6 和完成协议保持不变。

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
| `buffers_us` | 首次设备状态/队列分配与清零、Host 固定页状态缓冲分配、首次计时事件创建；稳定态复用，不再每批清零 |
| `register_output_us` | 获取 output 注册及选定 MR 的通信描述；首次 output 注册、首次 MR/output 组合上传描述，之后均复用 |
| `launch_us` | Host 提交 raw READ 算子调用 |
| `synchronize_us` | Host 等待传入 stream 中的 kernel 和状态异步回读都完成 |
| `kernel_event_us` | 同一 stream 中紧邻算子前后两个设备事件之间的时间 |
| `status_us` | 提交状态异步回读的 Host 开销，加上同步后检查每核启动编号、执行结果和请求数量的开销；不含设备拷贝等待 |
| `cleanup_us` | 释放 output 注册引用；MR、通信实体和状态缓冲保留到 detach/shutdown |
| `event_record_us` / `event_query_us` | Host 提交计时事件 / 查询计时结果的开销 |
| `host_other_us` | 原生总时间减去上述 Host 阶段的余量 |

绑定层另提供以下字段，同样只在 `profile=True` 时读取时钟：

| 字段 | 计时范围 |
| --- | --- |
| `binding_checks_us` | 入口准备、源类型、三数组数量及调用前 pending 检查 |
| `binding_src_us` / `binding_dst_us` / `binding_lengths_us` | 分别提取三个描述 Tensor 的地址、数量、元素大小和 shape，检查连续性、dtype 和设备；不回读数组内容 |
| `binding_output_us` | 提取 output Tensor 信息并检查设备 |
| `binding_stream_us` | 获取/检查 stream 并提取 ACL stream 指针；显式传入 stream 时不获取默认 stream |
| `binding_retain_us` | 查询/更新 output 引用缓存、保留本轮 Tensor/stream 引用和准备 profile 结构 |
| `binding_native_us` | 整个 NpuRuntime::BatchGetRawAddr 调用，包含 `host_total_us` |
| `binding_release_us` | 原生返回后检查 pending、释放本轮保留引用、检查返回状态 |
| `binding_result_us` | 构造正常返回的 state/sequence 字典 |
| `binding_profile_us` | 构造原有原生计时字典并挂到返回结果；仅 profile 调用产生 |
| `binding_report_us` | 将新增绑定阶段计时写入字典的开销；仅 profile 调用产生 |
| `binding_total_us` | 绑定函数入口至写完上述阶段字段；包含原生调用及计时探针开销，不含最后写入 report/total 两个字段、函数退栈和 pybind 返回过程 |

这些阶段以相邻时间戳划分，互不重叠，`binding_checks_us` 合并了入口和数组数量/pending 两段检查。
`binding_total_us` 等于各 `binding_*` 阶段之和（不再把 total 自身相加）；
`binding_native_us` 已包含原生总时间与各原生子阶段，`kernel_event_us` 也不能额外加到 Host 总和。
计时和返回计时字典会增加 profile 调用的开销，带宽仍使用未插桩轮次。

`kernel_event_us` 使用 `ACL_EVENT_TIME_LINE` 和 `aclrtEventElapsedTime`，毫秒转微秒。
它覆盖 raw READ 算子中的可选长度/边界诊断、WQE 生成与提交和 CQ 等待，属于设备 stream 的事件区间，
并非仅网络传输时间或各核时间的累加；也可能包含事件与设备调度开销。
该区间与 Host 的 launch/synchronize 重叠，不能再加到 Host 阶段总和中。
用法依据 [CANN 设备事件计时接口](https://www.hiascend.com/doc_center/source/en/canncommercial/800/apiref/appdevgapi/aclcppdevg_03_0090.html)。
事件记录或结果查询失败不改变已成功的数据传输结果；此时 `kernel_event_us=None`，
`kernel_event_error` 提供 ACL 返回码。事件创建失败发生在发起 READ 之前，接口返回异常。
默认 `profile=False`，不创建/记录计时事件；计时事件按 reader 复用，detach 时销毁。

状态通过 `aclrtMemcpyAsync` 回读到初始化时由 `aclrtMallocHost` 分配的固定页内存，
与 kernel 使用同一条 stream。顺序为 kernel → 结束计时事件（仅 profile）→ 状态回读 → 一次 stream 同步 → Host 校验。
`kernel_event_us` 不包含状态拷贝；`synchronize_us` 现在包含状态拷贝等待，因此比较优化前后时要看完整调用时间，不能只比较 `status_us`。
拷贝提交失败仍尝试同步已启动的 kernel；提交或同步失败均保留未决状态，应用不得据此通知源复用。
固定页缓冲跨批次复用，只在连接安全关闭时释放；同步失败时继续保留，防止未完成拷贝写入已释放的 Host 内存。

双方必须显式配置 fullmesh EID：B 构造源时的 npu_eid 和 A listen_hbm 的 local_eid。CPU↔NPU 路由独立配置，自动 CLOS/pair-info 发现不能替代 fullmesh。

## 算子与资源

raw READ V3 在同一次算子中直接生成 WQE、提交并等待 CQ。WQE 总数等于输入请求数，各核按请求数量划分任务；默认不预扫描长度，也不执行逐段源/目的边界诊断；显式启用诊断时，各核独立检查整批，在任何 READ 提交前拒绝越界输入。目的区间两两比较已删除。三数组不回传 Host，没有独立准备算子。所有请求均按下标直接索引，无分段前缀表或查找；沿用 BatchGet 的 WQE 编码、双 bank、多 Jetty、inflight 窗口和 CQ 处理。

每个 reader engine 为每次 kernel 启动（包括建链探测）分配一个非零、递增的 64 位编号。每个核写完最终结果后，把该编号写进结果结构原有的预留位置；Host 在解释成功或诊断失败前，先检查所有核的编号属于本轮。缺少任何核的新结果都会保留未决状态，应用继续保留源和 output。状态缓冲只在首次分配后清零一次，稳态不再调用同步 `aclrtMemset`；每核结果仍为 1280 B，HBM 分配量和状态回读量不增加。

仅建链的 all-lane 探测仍使用内部 Host 绝对地址描述符模式，不再作为公开数据读取接口。探测失败必须确认 kernel 已退出且 initiator 队列成功销毁后才能换候选。

NPU ReadyToken.source 只包含 source_id 和 source_epoch，ReadyToken 另含 sequence 与 mr_id。完整 MR 长度列表保存在 KVAF 内部，单 MR 也使用非空列表；初始化及 TCP 重连仍检查完整清单。连接描述与 MR 描述分别存储，新增 MR 的 Memory 请求/响应不再携带 Jetty/TP 描述。

KVHR v8 只负责初始化、硬件资源交换、重连和关闭，不承担每批完成或取消通知。A 的 HbmReader 保存连接/导入 MR、USER_CTL Jetty/CQ、队列和本地防重放状态；B 的 NpuPullHbmSource 保存 Tensor、导出 MR、目标 Jetty、客户端身份和应用管理的冻结批次。各通道分别协商 TP，CPU Pull 的共享 TP 路径独立。

初始化和关闭控制请求复用 TCP 连接，双方启用 TCP_NODELAY。A 按当前命令的回复长度接收，B 先读取请求共有的 72 B 前缀，再按消息种类读取尾部；分片、粘连、版本和身份校验仍保留。完成通知由应用自己的 Pipe/Connector 等通道传递。

每批最多 4096 段，对应最多 4096 个 WQE，每段长度由 WQE 的 32 位字节数字段表示（1..UINT32_MAX），实际还受设备单次 READ 能力限制；每个源绑定一个 reader，同时一个活动批次，默认 2 核 × 2 Jetty，总计最多 4 个 Jetty。接口同步，不支持同一 reader 的并发调用或 graph capture。

使用 `KvafExecuteBatchGetRawAddrAivKernelV3`，参数从 80 B 增至 88 B，旧 V1/V2 object 会拒绝加载。移除完成通知没有改动 kernel 和参数 ABI，已有匹配 V3 object 可继续使用。底层 provider 返回的内存身份与路由 EID 分开保留；不手动改写 MR/Jetty 的导出身份。

## 编译与硬件验证

本次更新需重编译 A/B 两端 NPU 运行库和 Python 扩展，并重启进程。控制协议为 KVHR v8，NPU receiver ticket 仍为 version=1；已有 V3 raw kernel 和匹配 KVRP v6 的 CPU service 无需因移除通知而更新。
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
`kvaf_hbm_read_test` 在关闭 CPU Pull 构建能力时，使用真实 NpuRuntime/TCP/source/client 与假 ACL/HCCP 验证纯 NPU Pull、每段一个 WQE、应用释放和清理。
`kvaf_hbm_read_with_ddr_test` 另外验证两种初始化顺序、DDR 失败/关闭/重新初始化不影响 raw READ，以及旧 PutHandle 被拒绝。
上述两个集成测试还覆盖 78 MR 注册/探测、选择错误、部分注册失败、每批零控制 RPC、控制 TCP 断开时仍能完成数据读取，以及关闭回复丢失后的恢复。
假 ACL 将 kernel 和状态拷贝延迟到 stream 同步时按顺序执行，验证 Host 不会提前读取结果；
还覆盖固定页内存申请/释放失败、跨批次复用，以及异步回读提交或同步失败时保留源、output 和状态缓冲。
`kvaf_hbm_read_transport_test` 运行生产 Host 注册/缓存代码，验证 78 份不同 token/EID 的描述共用 output MR 和队列、缓存命中、上传失败、部分导入和重试清理。
`kvaf_hbm_read_control_test` 验证初始化/关闭控制帧粘连、头/正文分片、截断、旧版本/错误长度/身份及远端错误返回。
Python bench 的无硬件测试还验证匹配的 reuse 消息才释放源；错误轮次、通知丢失或 Pipe 断开时保留冻结批次。
这些测试不替代 Linux CANN 编译和真实 A—B READ 验收。

## 随机离散包性能测试

`tests/python/bench_hbm_read_raw_addr.py` 中的 A 为读取端，B 为生产端。
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

- `--packets`：等长模式的包数量，默认 4096，1..4096；兼容 `--blocks`。
- `--packet-bytes`：等长模式的每包字节数，默认 656；兼容 `--block-bytes`。每包直接生成一个 WQE，长度须为 1..UINT32_MAX 且满足设备单次 READ 上限。
- `--packet-group COUNT:BYTES`：混合大小模式，可重复；兼容 `--block-group`。全部组在同一次调用中读取，总包数须在 1..4096，不能同时指定等长模式的数量/大小参数。
- `--source-space-bytes`：B 每个注册 tensor 的 HBM 空间大小，至少容纳本批所有包。等长模式按包大小划分槽位，随机抽取不重复槽位；混合模式按各组数据量比例划分同一 MR 的不重叠源区域，再在各区域抽取槽位。不足一个槽位的尾部不使用。
- `--source-count`：注册多少个独立 tensor/MR（默认 1，范围 1～4096），共用连接，逐批轮换；HBM 总空间约为 source-count × source-space-bytes。多 MR 时预热至少覆盖全部 MR/output 组合，实际 warmup 会打印。各 MR 的数据不同，同一 MR 再次使用时改变数据；期望值生成和校验不计入传输时间。
- `--seed`：默认 123，可复现布局。目的 output 连续排列，首尾各留 64 字节 guard。
- `--aiv-cores` / `--jettys-per-core`：默认 2/2，总 Jetty 数不超过 4。
- `--max-inflight-total`：默认 2048，含义与 BatchGet 一致。
- `--profile-warmup` / `--profile-iterations`：单独的内部计时轮次，默认 1/20；设为 `--profile-iterations 0` 即只运行未插桩的带宽测试。
- `--validate-ranges`：打开可选正长度、32 位长度字段及源/目的边界诊断，默认关闭；不包含目的重叠检测。日志标记 `validate_ranges` 和 `output_registration_cache=1`，首次 output 注册计入 warmup（warmup 为 0 时计入测量）。

输出 `NPU_HBM_BENCH_PASS` 的吞吐和分位数只使用 `--iterations` 轮的
Python `batch_get_raw_addr()` 数据就绪返回时间，未打开内部计时；`return_condition=data_ready completion_notification=application` 标明新口径。
与旧版包含 Complete 往返的调用相比，当前接口把通知职责交还应用，应用 IPC 仍在计时之外；`profile` 不再包含 `complete_us`。
另外运行带 `profile=True` 的轮次，输出 `NPU_HBM_INTERNAL_PROFILE_STAGE`，
包括 `python_total`、上述全部 Host 阶段和 `kernel_event` 的 mean/P50/P95/P99/min/max。
绑定阶段也输出在同一组日志中。bench 对每个 profile 调用另外计算：

- `binding_native_other = binding_native_us - host_total_us`：原生包装层及内外计时边界之间的余量。
- `python_and_binding_other = python_total - binding_total_us`：Python 调用、pybind 参数转换、最后两个计时字段写入、绑定局部对象析构、返回值交接及替换上一轮 result 的释放开销。

差额保留正负号，便于发现计时问题；按同一次调用计算后再统计，不能用未插桩轮次减去 profile 轮次。
逐轮可按 `python_total = binding_total_us + python_and_binding_other` 对账，
`binding_native_us = host_total_us + binding_native_other`；各分位数仍不能相加。
读取端函数在原生调用期间继续持有 GIL，不存在该边界的 GIL 释放/重新获取耗时。
内部计时轮次不混入带宽统计；各分位数按 `(N-1)*p` 线性插值，不能将不同阶段的 P99 相加。
`NPU_HBM_INTERNAL_PROFILE_INFO` 列出有效设备计时样本数和计时错误码，缺失的设备计时不会作为零值统计。

绑定层每次调用只获取一次各 Tensor 属性，复用同一个 device 对象完成设备检查。
raw READ 对 torch.Size 只提取维数并校验维度值，避免构造 shape vector；普通 BatchGet 仍获取完整 shape。
这些信息每次重新读取，不按 Python 对象身份缓存地址或布局；output 注册缓存和未决任务的引用保留规则不变。

如果 `binding_src` 明显高于另外两个描述 Tensor，可单独对照三种解析顺序：

```bash
# 构建测试辅助扩展；配置时需 KVAF_BUILD_PYTHON=ON、BUILD_TESTING=ON。
cmake --build build-shared-ddr --target _kvaf_binding_metadata_test -j
PYTHONPATH="$PWD/build-shared-ddr/python${PYTHONPATH:+:$PYTHONPATH}" \
python tests/python/bench_npu_binding_metadata.py --device 6 --iterations 100
```

该诊断调用与生产绑定相同的元数据提取函数，分别先解析 src、dst、lengths，输出 `NPU_HBM_METADATA_PROFILE`。
它只测属性提取，不进行 READ，也不计入主 bench 的吞吐结果。若较高耗时跟着第一个解析的位置走，说明需要继续追踪首次访问成本；
若没有复现，则需结合主 bench 的实际调用环境排查。优化效果仍以原 bench 未插桩的整次调用时间为准。

源注册、连接探测、源写入、三组 NPU int64 Tensor 准备、output 清空、应用 IPC、数据校验均在传输计时之外。
每轮交替输入内容，并检查 NPU 消费者和完整回读数据（含 guard）。
`NPU_HBM_BENCH_PROFILE_STAGE` 另列读取端接口/校验时间及生产端 ready/release_batch 时间。
源与目的进程的时钟不相减。两端初始化与清理时间不纳入稳定态逐批统计。

### 一次读取两组不同大小的 block

下面一次 `batch_get_raw_addr()` 读取 1024 个 `128 × 128 B` block 和 1024 个 `128 × 4 B` block：

```bash
export PYTHONPATH="$PWD/build-shared-ddr/python${PYTHONPATH:+:$PYTHONPATH}"
python tests/python/bench_hbm_read_raw_addr.py \
  --device-a 6 --device-b 7 \
  --npu-eid-a-raw 000000000047050000100000df00d806 \
  --npu-eid-b-raw 000000000047050000100000df00f807 \
  --block-group 1024:16384 --block-group 1024:512 \
  --source-space-bytes 268435456 \
  --warmup 10 --iterations 100 --profile-warmup 1 --profile-iterations 20 \
  --raw-kernel build-shared-ddr/bin/kvaf_execute_batch_get_raw_addr_kernels.o
```

每批共 2048 段、17,301,504 B（16.5 MiB），只调用一次 Runtime。默认 source-count=1，两组都来自同一个源 Tensor/MR，
输出也在同一个 Tensor。读取描述按大块、小块交错排列，lengths 为 `[16384, 512, 16384, 512, ...]`，
让默认两个核都处理 512 个大块和 512 个小块。output 的前 16 MiB 存放大块，随后 0.5 MiB 存放小块，首尾各有 64 B guard。
两组使用不同的数据模式，同一组逐轮改变内容；每轮仍校验完整 output 和 guard。

日志应包含 `packets=2048 packet_groups=1024x16384,1024x512 payload_bytes=17301504`。
带宽分子使用两组数据量之和，计时范围仍是同一次调用至数据就绪；源写入、期望值生成和校验都在计时之外。
此场景只修改 Python bench，已安装当前绑定和 raw kernel 时，拉取脚本更新即可运行。

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
