# AICPU GatherRead：构建、调用与验证

本页保留 Gather 算法数据流和兼容接口示例。按调用选择算法、驻留实例、输出绑定与
Graph 租约的完整用法见 [实现与使用](cache_transport_execution_implementation.md)；
四算法统一验收和性能口径见 [验证步骤](cache_transport_execution_validation.md)。

已有的 `NpuRuntime.batch_get` 可以选择 UMDK 后端。一次 launch 启动 N 个 AICPU 实例，
各自从本轮 NPU Tensor 的输入区间生成源清单，请求 CPU gather 到自己的 DDR staging 区间，
查询自己的就绪前缀、提交 READ，并等待自己的完成通知和 CPU ACK。
indices、offsets、miss_count、output_indices 和数据可以每轮变化，不缓存上一轮内容。
本页使用 `GetExecutionConfig("urma", "aicpu", "gather")` 选择 Gather，Runtime 的
`transport_mode` 须允许 URMA。Put 使用独立的执行配置，未显式选择时按构建能力确定默认值。
Host direct HCCP 建链的 `cpu_push_scatter` 路径见 [CPU 回推说明](aicpu_cpu_push_scatter.md)。
与顶层 BatchGet 一致，`int32[1] miss_count` 指定输入数组的有效前缀长度；
只处理前 `miss_count` 项，取值范围为 0 到输入数组容量。
AICPU 读取这个计数，将有效前缀均分给各 worker；READ 直接使用调用方的 `output_indices` 组 SGL。
源地址由 `addresses[indices[i]] + offsets[i]` 得到规范地址，再转换到当前 session 的注册地址；调用方保证有效条目的索引、
块内偏移、地址有效性及输出行号正确，输入元数据在本轮完成前保持不变。

## 构建和启动

当前控制协议使用版本 6，CPU 服务和 NPU 客户端使用配套版本构建并重启。
AICPU 使用独立的输入 ABI；重新安装设备包后再启动客户端。

以下命令均在仓库根目录执行。CPU 端构建服务：

```bash
./build.sh cpu
```

NPU 端先加载实际安装的 CANN `set_env.sh`，再统一构建：

```bash
./build.sh npu
./build.sh npu --check-kernel
export PYTHONPATH="$PWD/build-npu/python${PYTHONPATH:+:$PYTHONPATH}"
```

默认包含 AIV URMA、UBmem、AICPU Put/Gather/Scatter，生成一份 Python 包，
构建成功后自动安装 AICPU 设备包，无需 `--install-kernel`。
安装需要对所选 CANN 的包目录和加载配置有写权限。

仅需 AICPU Gather/Put 时，可改用 `./build.sh aicpu-urma`，同样自动安装；
此时将本文的 `build-npu` 换为 `build-aicpu-urma`，安装检查使用
`./build.sh aicpu-urma --check-kernel`。该裁剪构建不包含 Scatter，也不要求其 securec 依赖。
`./build.sh npu --without-aicpu` 仅构建 AIV/UBmem，不能用于本页的 AICPU 测试。

CPU 服务使用新二进制重启：

```bash
./build-cpu/runtime/kvaf_cpu_service \
  --listen 0.0.0.0 --port 28990 \
  --gather-threads 8 --gather-mode active
```

同一服务也可供 AIV、UBmem 和 Scatter 使用。`--gather-threads` 是每个 Gather
session 的 CPU 线程总预算，默认 10，须不小于客户端 AICPU worker 数；等待模式默认
`active`，使用主动等待。上例显式将线程预算覆盖为 8。
这两个参数不改变其他后端的执行。Gather 线程继承外部 CPU 亲和性；需要限定 NUMA 节点时，
使用 `numactl --cpunodebind=N --membind=N` 启动服务，或使用 `taskset -c CPU_LIST` 指定 CPU。
示例使用设备 7 和本机服务 `127.0.0.1:28990`，跨主机部署时替换为实际设备号和 CPU 地址。

NPU 进程先加载 CANN 环境，再使用现有 Tensor 接口：

```python
import torch
import torch_npu
import kvaf

torch.npu.set_device(7)
put = kvaf.PutExecutionConfig("urma", "aicpu")
get = kvaf.GetExecutionConfig("urma", "aicpu", "gather")
runtime = kvaf.NpuRuntime(
    batch_get_token_bytes=656,
    batch_get_tokens_per_block=1,
    device_id=7,
    transport_mode="urma", default_put=put, default_get=get,
    put_resources=[(put, kvaf.ExecutionResources(cores=1))],
    get_resources=[(get, kvaf.ExecutionResources(cores=1))],
)
runtime.link_to("127.0.0.1:28990")
runtime.initialize(ddr_bytes=656 * 8192, slot_size=656)

keys = list(range(4096))
source = torch.randint(0, 256, (4096, 656), dtype=torch.uint8, device="npu:7")
runtime.batch_put(keys, source, [i * 656 for i in keys], [656] * 4096)
table = runtime.prepare_access_table(keys)
device_table = runtime.access_table_view(table, device="npu")
indices = torch.arange(4096, dtype=torch.int32, device="npu:7")
offsets = torch.zeros_like(indices)  # 表项内的字节偏移。
miss_count = torch.tensor([4096], dtype=torch.int32, device="npu:7")
output_indices = indices.clone()
output = torch.empty_like(source)
runtime.batch_get(device_table, indices, offsets, miss_count, output_indices, output,
                  synchronize=True)
assert torch.equal(output, source)
runtime.release_access_table(table)
runtime.shutdown()
```

UMDK device/index 由 Host EID 匹配，不再作为 `NpuRuntime` 参数提供。
AICPU 的 `ExecutionResources` 不提供 kernel 路径；`cores` 配置各 Put/Get 的 worker 数。
原来的 `addresses/valid` Tensor 重载也可直接使用。`default_get` 决定省略 `execution`
时使用的 Get 配置，单次调用可显式选择另一个已构建且符合 transport 模式的配置。
AICPU kernel JSON 由 Runtime 从选定 CANN 根目录的
`opp/vendors/cust/op_impl/aicpu/config/kvaf_aicpu_urma.json` 加载，无需传路径。
环境变量 `ASCEND_HOME_PATH` 和 `ASCEND_CANN_PACKAGE_PATH` 若同时设置，必须指向同一真实目录。
只有一个非空时使用它；都未设置时，依次检查 `/usr/local/Ascend/ascend-toolkit/latest`
和 `/usr/local/Ascend/cann/aarch64-linux/asc`。显式无效目录直接报错；根目录一旦确定，
缺包也不会转到另一套 CANN 安装中搜索。
Runtime 在首次准备 AICPU 时检查设备 tar 包及 `conf/ascend_package_load.ini` 注册配置，
并加载内核、检查所需入口；未使用 AICPU 时不触发包准备。缺包、缺入口或未构建后端时直接报错，不回退到 AIV。
AICPU-only 构建也使用执行配置选择算法；公开 JSON 路径参数已删除。

如果调用方已经持有 NPU 地址表，直接调用现有 Tensor 重载，不必再创建 Host AccessTable 或重新上传地址表：

```python
runtime.batch_get(addresses, valid, indices, offsets, miss_count, output_indices, output,
                  stream=stream, synchronize=True)
```

调用方管理这些地址对应的远端数据生命周期。使用 `prepare_access_table` 的调用仍保留公共元数据接口，
Host 地址表通过整段 H2D 上传；每次 BatchGet 本身不遍历或复制 Host 地址表。

## 与 AIV 共用 Host EID 发现

GatherRead 和 CPU Push Scatter 都由 `NpuRuntime` 的现有
EID 发现流程选路。本机自动使用该物理卡的 Pair Info remote_eid，缺失时回退到首个有效 CLOS；
远端自动使用首个有效 CLOS。显式 NPU EID 优先。

Host 将选定 EID 传入 UMDK Prepare，设备端只匹配该 EID 对应的 UB device/index 并创建 context；
没有匹配项或准备失败时直接报错，不改用其他 EID。
UMDK device/index 完全由选定 EID 自动匹配，不再提供单独的设备/索引过滤参数。
统一性能入口通过 `--npu-eids` 或 `KVAF_NPU_EIDS` 指定严格 EID 列表；不指定时自动发现。
GatherRead 和 CPU Push Scatter 均只使用列表首项；AIV Get 的显式多 EID 条带语义不变。
`KVAF_NPU_EIDS` 由测试脚本解析后传给 Runtime，Runtime 本身不读取该环境变量。

同一个 Runtime 的 GatherRead/BatchPut 共用选定 EID；CPU Push Scatter 的 HCCP 回推通道
与其 UMDK BatchPut 也绑定同一个 Host EID，各自管理自己的通信资源。
Host/设备控制参数已增加 EID 字段，仍使用单一 `KvafAicpuUmdkControl` 入口。
升级时必须一起重建 Host 库并重新安装设备算子包，然后重启 NPU 进程：

```bash
./build.sh npu
./build.sh npu --check-kernel
export PYTHONPATH="$PWD/build-npu/python${PYTHONPATH:+:$PYTHONPATH}"
```

AICPU 初始化输出 `UMDK_EID_SELECTED source=host device=... eid_index=... eid=...`，
应与 Runtime 选定的 Host 候选一致。该改动不包含 TP 实验、请求邮箱协议变更或诊断清零开关。

## 公开输入模式

公开 Runtime 只提供查表输入，`gather_read_direct` 不再是合法 mode，
统一性能入口不提供 `--source-mode direct`。裸 `addresses/valid` Tensor 重载
仍表示地址表，源地址为 `addresses[indices[i]] + offsets[i]`，不能传入逐请求最终地址替代。

## Python 分段采样

使用下一节的统一性能入口比较 worker 数和 eager/graph。
若需在应用中分析单次同步 AICPU 调用，可使用原生 profile 返回值：

```python
profile = runtime.batch_get(device_table, indices, offsets, miss_count, output_indices, output,
                            synchronize=True, profile=True)
print(profile["host"])
print(profile["workers"])
```

`profile` 默认关闭，保持原来的返回值 `None`；开启时仅支持同步 AICPU 调用，返回字典。
关闭时不读取新增的阶段时钟，也不分配或回传 profile 缓冲区。开启时额外采集时间和回传结果，
所以用正常测量轮次报告带宽，用独立的 profile 轮次定位耗时。

| 输出 | 含义 |
| --- | --- |
| Python `binding_prepare_us` / `native_call_us` | 绑定层准备（Tensor 信息、参数检查、stream 解析和 profile 对象分配）/ 原生 Runtime 调用；合计为 `binding_total_us` |
| Python `python_other_us` | 外部 Python 计时减去 `binding_total_us`，含绑定分派与 profile 字典转换 |
| Host | 准备、launch、状态及 profile D2H 入队、等待 stream、收尾；各项之和为 Host total |
| Device | 最早 worker 进入至最后 worker 主体退出，使用同一设备时钟；不含最后的 profile 导出 |
| Worker | 相对进入/退出、总时长、清单构建、源清单 WRITE、通知 CPU、流水 READ、最终 ACK |
| Within request | 清单交给 URMA DMA 前的缓存维护；已包含在请求准备中 |
| Within READ | 工作区准备、WR/SGL 构建、等待就绪前缀、provider 提交、最后 CQE 等待；已包含在 READ 中 |

各 worker 互相重叠，不能把它们的阶段最大值相加为端到端时间。
`ready_wait` 包含与 CPU gather 重叠的等待，并非 CPU gather 本身的计时。
provider 提交包含编码、写 SQ 和 doorbell，不能单独当作写 SQ 耗时。
Host wait 包含排队的设备执行和 D2H；Host 与设备的绝对时间戳不相减。

## 使用统一性能入口

CPU 服务启动并加载 CANN 环境后，运行四种 Get 共用的入口：

```bash
KVAF_CPU_SERVICE_ADDR=127.0.0.1:21111 DEVICE_ID=2 \
bash tests/run_remote_cpu_batch_get_perf.sh build-npu \
  --backend gather_read --aicpu-workers 5 --execution both \
  --request-length 32768 --table-capacity 4096 --miss-count 4096 \
  --token-bytes 656 --tokens-per-block 128 \
  --seed 20260903 --no-sort-input-indices \
  --source-access-pattern random --output-access-pattern contiguous \
  --warmup 10 --iterations 100 --kernel-profile
```

`BATCH_GET_RESULT scope=end_to_end` 报告 Host 调用到完成的延迟和带宽；
`scope=kernel` 报告最早 worker 进入到最晚 worker 退出的设备时间和带宽。
设备时间包含算子中的 CPU/网络等待及 profile 打点。
`BATCH_GET_PROFILE` 保留各 worker 阶段的统计，阶段可嵌套和重叠，不能相加为设备总时间。
数据初始化、请求生成、预热、输出校验和 profile 快照在正式性能采样之外。
去掉 `--kernel-profile` 仍报告端到端和算子时间、带宽，不输出 worker 分段明细。

源池大小由 `--table-capacity` 和块布局显式决定，不再由独立后端脚本分配 100 GiB。
扫描不同 miss 数、token 大小或 worker 数时，循环调用同一个入口。
详细口径及其他后端参数见 [统一 Get 性能测试](batch_get_performance.md)。

## 执行和资源约定

1..8 个 AICPU worker 共用一对 TP，每个 worker 有独立数据 Jetty/CQ 和控制 Jetty/CQ。
CPU 侧每条通道也各有独立 Jetty/RQ/CQ。CPU 与 AICPU 共用端点枚举、队列创建和 TP/Jetty 扩展代码；
AICPU 进程内的 liburma 动态加载与会话由 AICPU 自己维护。

每个 worker 负责有效前缀中的一个连续区间，直接按顺序构建源清单和 READ。
CPU 直接引用收到的源清单，使用到本轮拷贝结束、发布 ACK 为止，不再逐条复制清单或检查源地址范围。
BatchGet 在描述头中提供固定片段长度 `span_bytes`（例如 656 B）；CPU 用
`staging_offset / span_bytes` 和 `staging_offset % span_bytes` 定位源片段及片段内偏移，
不再扫描清单生成累计偏移表。相邻源地址在当前拷贝分块内合并成一次 memcpy；零散地址仍按片段拷贝。
旧的变长清单使用 `span_bytes=0`，仍按长度构建偏移表。清单头保持 48 B，进度字段位置保持不变。
启用固定长度描述头的版本需要同时更新 CPU 服务与 AICPU 包，并重启 CPU 服务。
单 worker 直接使用编号 0；多 worker 从 AICPU 进程内的循环编号器各领一个编号。
每次 launch 的 N 个实例恰好覆盖全部 N 个编号；上一轮结束后才允许复用资源。
Host 每轮直接提交算子，不再分配 HBM 编号缓冲或提交清零任务。
描述头、源清单、command、notify、ACK、ready_bytes 和 CPU gather 处理对象均按 worker 独立。
CPU 服务的 `--gather-threads` 是全部 worker 的总预算，须不小于 AICPU worker 数；
预算在 worker 之间均分，余数从 worker 0 开始分配。每个 worker 每完成 128 KiB 连续前缀即可开始 READ。

每次调用的总量不再限制为 65536；全部窗口在同一次 kernel launch 内执行。
`gather_ddr_bytes` 选择预算内最大的 4096 倍数窗口，默认仍为 65536。
token 大小由构造参数指定，每个 lane 的窗口载荷须能用 uint32 表示，见 [窗口与预算](aicpu_get_windows.md)。
首次 Gather 准备时，CPU 在独立工作区中按 worker 数分配 staging、源描述符和控制区，
每个 staging 区间可容纳 `ceil(65536 / worker_count)` 个 token，并单独对齐到 64 字节。
主缓存池无需尾部预留，这些空间不进入 KV slot 分配；工作区在该 Gather session 关闭时释放。
HBM 仅保留发给 CPU 的源描述符和控制 scratch，输出 Tensor 无需额外尾部空间。
READ 使用 SGL 6、每次提交 32 条 WR；每波最多 4096 条 WR，只请求最后一条的有序 CQE。
WR/SGL 工作区按本轮请求的上界增长，完成后复用容量；每轮仍重新填写全部有效请求。
会话用独立数组保存数据队列和控制队列，worker 0 与其他 worker 使用相同的存储和访问方式。
AICPU 只将输出地址交给 URMA，不直接读写输出 payload，因此 READ 前后不再逐段清理输出的 AICPU 缓存。
GatherRead 只在以下 kernel 内部 URMA DMA 交接处保留缓存维护：

| 位置 | 写入方 → 读取方 | 保留理由 |
| --- | --- | --- |
| 非空源清单发送前 | AICPU → URMA WRITE DMA | 清单刚由 AICPU 填写，DMA 随即在同一 kernel 内读取其 HBM 内容 |
| 通知发送前 | AICPU → URMA WRITE DMA | 发布刚写入的请求编号和描述符指针 |
| 进度 READ 完成后 | URMA READ DMA → AICPU | DMA 覆盖复用的 scratch，AICPU 在同一 kernel 内读取新进度 |
| ACK READ 完成后 | URMA READ DMA → AICPU | DMA 覆盖 ACK 槽，AICPU 在同一 kernel 内判断完成状态 |

这些交接不经过 kernel 完成边界，CQ 表示传输完成；它本身不构成 CANN 对本地 AICPU 缓存的交接。
已删除输入 Tensor、miss_count、status 和 profile 回传前的手工刷新。
输入在 kernel 启动前就绪，status/profile 由同一 stream 上排在 kernel 之后的 D2H 回传，
与现有 UMDK 控制 kernel 和 CPU-push kernel 的处理一致。空请求不发送清单，也不执行清单刷新。
`input_cache_ns` 的 ABI 槽位保留为零；不再执行该阶段或在测试输出中展示它。
每个 worker 的 scratch 独占一条 64 B 缓存行：通知发送前清理一次，此后 AICPU 只读取查询结果，
不再在每次 ready/ACK READ 前重复清理；收到 CQE 后仍清理再取值，避免读到旧缓存。
不清零 scratch 中的 ACK 槽，查询完成后直接使用远端覆盖的值。
单个不超过 64 KiB 的控制传输直接填写一个 WR/两端 SGE，保留在队列对象中至完成或队列销毁，
省去通用分段构建及 vector 操作；清单大于 64 KiB 和多段通知仍走通用路径。
连续性、设备、元素宽度、长度以及队列完成/错误检查仍在相应入口执行。

BatchPut 同步完成；BatchGet 默认只 enqueue，`synchronize=True` 等待完成并读取错误状态。
同一 Get 方向的下一次普通调用会先完成上一笔；也可以显式 `drain_get()`。
跨 stream 依赖由调用方负责。完成或资源释放失败时保留相关资源，调用方须继续保留
Tensor 和 stream，按实现文档执行恢复或关闭重试。
普通请求的 Tensor 保持至完成；显式输入/输出绑定保持至绑定结束，图资源保持至图销毁并结束图租约。
当前 AICPU 进程支持多个 session，Put/Get 分别持有自己的 session/队列。
Gather 已支持 ACL Graph capture/replay，可用 `graph_resources().prepare/capture` 自动收集节点，
也保留 `begin_get_graph/end_get_graph` 底层接口；
设备 ticket 在每次 replay 推进序号，图内不执行 Host 注册或同步。
每个 Get 节点使用独立租约和报告，单图固定 stream；不同图与 eager 由调用方串行安排，
可使用不同 stream。图 profile 或状态查询始终指定 `graph_lease`，省略则查询 eager。多图验证可在统一性能入口加 `--multi-graph-check`。

本地验证使用 ACL/provider 替身；本分支的 NPU 兼容性与性能需要服务器验证。

## 实现位置

| 目录/文件 | 职责 |
| --- | --- |
| `access_operator/providers/urma_aicpu/` | Runtime 适配、资源 owner；Put/Get 按执行方向和配置隔离 |
| `access_operator/algorithms/get/gather/urma_aicpu/host/` | Gather workspace、参数和 kernel 提交 |
| `access_operator/algorithms/get/common/aicpu/host/` | Gather/Scatter 共享参数构造、加载和 capture 辅助代码 |
| `access_operator/algorithms/get/gather/urma_aicpu/kernel/gather_read.cc` | 当前流水算法、READ WR/SGL 组装和调度 |
| `access_operator/algorithms/put/urma_aicpu/` | 原有同步 BatchPut 的 Host/设备实现 |
| `access_operator/include/aicpu_urma/` | Host 会话、算子加载、Host/设备控制 ABI、设备通信接口和缓存工具 |
| `access_operator/transports/aicpu_urma/host/` | Host 侧会话组织和 kernel 加载 |
| `access_operator/transports/aicpu_urma/kernel/session.cc` | AICPU 加载 liburma、创建/连接队列、注册内存和资源释放 |
| `access_operator/transports/aicpu_urma/kernel/transfer.cc` | READ/WRITE WR 构建、提交和 CQ 完成处理 |
| `access_operator/transports/aicpu_urma/kernel/control_kernel.cc` | Host 控制命令的设备入口和分发 |
| `access_operator/transports/aicpu_urma/package/` | 算子 JSON、UMDK 库打包和安装脚本 |

BatchGet 还提供 Scatter，通过 `GetExecutionConfig("urma", "aicpu", "scatter")` 设置
默认或本次执行配置；两条路径使用相同的输入前缀和输出地址语义。
两个算法模块共用一个设备包，按首次使用的算法解析所需入口。Gather 和 Put
使用独立 UMDK session，资源按执行方向和配置隔离。Scatter Get 使用 Host HCCP 通道，AICPU Put
使用 UMDK。进程级 API 环境和 CPU DDR 注册的共享范围见实现文档。
