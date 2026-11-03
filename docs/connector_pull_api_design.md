# 面向应用 Connector 的 CPU Pull 与 NPU Pull 接口设计

状态：第一版已实现，待双卡硬件验收。本文记录当前接口边界、命名和调用流程；Python 接口与恢复方法见第 13 节。

应用在 A、B 两台机器上分别运行 Connector。A 生产 HBM 数据并发送业务元数据，B 根据元数据组织源和目的地址，调用 KVAF 拉取。KVAF 向 B 返回结果，后续业务通知由两个 Connector 自行处理。

本文统一使用 **A 为源端、B 为拉取发起端**。现有 [共享 DDR 文档](shared_ddr_remote_offload.md) 和 [HBM 裸地址读取文档](hbm_raw_addr_read.md) 描述已实现接口，使用的 A/B 角色名称与本文相反；实际数据方向以本文定义为准。

## 1. 已确认的设计约束

- CPU Pull 的目的槽位通过 `PreparePut` 分配，与普通 BatchPut 共用同一个 MetaManager、DDR 池和写事务机制。
- 应用在 PreparePut 返回的预留范围内计算目的 offset，不设置独立的应用 DDR 分区，不提供 `reserve_raw_ddr()`。
- 数据是否已准备好、读取期间是否保持不变，由应用保证。`ReadyToken` 不包含 `readable_ranges`，A 不再向 KVAF 发布一份读取范围。
- 保留 `source_id`、`region_id`、`region_epoch` 等身份信息，由 KVAF 生成、保存和校验。
- 保留两种源对象，将 `HbmSource` 更名为 `CpuPullHbmSource`，将 `HbmReadSource` 更名为 `NpuPullHbmSource`。
- B Connector 自己选择 CPU Pull 或 NPU Pull，并计算源 offset、目的 offset、长度。
- 两条读取接口接收三个连续、等长数组，相同下标表示同一段传输。CPU Pull 与 NPU Pull 都使用 `src_offsets`、`dst_offsets`、`lengths`，目的偏移的基址分别是 DDR 池与 output 缓冲。
- KVAF 的传输完成结果返回 B Connector。A/B Connector 的业务元数据和业务完成消息使用应用自己的通信通道。
- 保留现有已验证的数据面：CPU 四 Jetty 批量 READ，以及独立的 `batch_get_raw_addr` 算子。

## 2. 组件与职责

| 组件 | 负责的事情 |
|---|---|
| A Connector | 调用注册和建链接口、接收初始化票据、生产数据、保证源数据就绪和稳定、发送业务元数据、接收完成通知 |
| A 的 `CpuPullHbmSource` | 管理供 CPU 拉取的 HBM 注册、CPU 对端绑定、批次身份及源生命周期 |
| A 的 `NpuPullHbmSource` | 管理供 B NPU 拉取的 HBM 注册、主动连接读取端及批次状态 |
| B Connector | 解析业务元数据、确定 key 和长度、调用 PreparePut、组织三个连续数组、发起拉取、处理结果并通知 A |
| B 的 `NpuRuntime` | 管理共享 DDR、MetaManager、源绑定和任务，提供两条读取路径及写事务接口 |
| CPU service | 分配并注册 DDR，保存 CPU Pull 源连接，执行 READ，提供任务状态和完成确认 |

B 的普通 BatchPut、BatchGet、本卡 Offload 和跨卡 CPU Pull 共用一个 RuntimeCore、MetaManager 和 DDR Region。NPU 直接拉取 A HBM 的目的地是 B 的输出 HBM，不经过 DDR 元数据管理。

本设计不增加第二个 OffloadRuntime，也不为应用计算的地址另外分配一块 DDR。

## 3. 两种源对象

| 对比 | `CpuPullHbmSource` | `NpuPullHbmSource` |
|---|---|---|
| 原类名 | `HbmSource` | `HbmReadSource` |
| 所在位置 | A 源端进程 | A 源端进程 |
| 主动读取方 | CPU service | B 的 NPU |
| 数据路径 | A HBM → CPU DDR | A HBM → B HBM |
| 初始化关联 | 指定的 CPU DDR region | NPU reader 连接 |
| 目的内存确定时机 | 初始化时确定 DDR 池，每批通过 PreparePut 确定槽位 | 每次读取时由 B 提供 output 缓冲 |
| 完成条件 | READ 完成后，B 提交或撤销写事务并完成 CPU 确认 | NPU READ 结束并完成传输确认 |
| 路由 | CPU↔NPU 路由 EID | NPU↔NPU fullmesh EID |

两个对象共用底层 `Exporter` 的 HBM 注册、地址对齐、资源创建和清理代码，但分别持有注册、连接和状态。CPU 与 NPU 对端的描述符格式、TP 绑定和控制协议仍分别处理，不合并资源实例。

两个对象可以分别注册不同的 HBM，也可以独立注册同一份 tensor 列表；注册和连接实例仍各自持有。共享底层 tensor 时，应用等待两条路径均完成并释放批次后才能覆写它。对象本身留在生产端进程内，不能通过 Connector 序列化发送。

## 4. 描述符、批次和地址结构

以下为应用可见的逻辑结构。Connector 原样传递 KVAF 返回的 ReadyToken（内含源描述符），不自行组装 MR、Jetty 或 TP 数据。Python 的两个源描述符及 ReadyToken 为只读对象，支持 pickle 跨进程交换；本地 PutHandle 不支持跨进程传递。

### 4.1 每批源描述

```cpp
struct CpuPullSourceDesc {
    uint64_t source_id;
    uint64_t region_id;
    uint64_t region_epoch;
    uint64_t mr_length; // 本批 ReadyToken.mr_id 对应的逻辑长度
};

struct NpuPullSourceDesc {
    uint64_t source_id;
    uint64_t source_epoch;
};
```

| 字段 | 含义与用途 |
|---|---|
| `source_id` | 一组 HBM 源注册和连接的编号；CPU/NPU Pull 都可包含多个 MR，不是物理卡号 |
| `mr_length` | CPU Pull 本批选中 tensor 的逻辑长度，用于提交前范围检查；不是首个 MR 长度或跨 tensor 的地址范围 |
| `region_id` | CPU service 分配的整个 DDR 池编号，不是 key 或 slot 编号 |
| `region_epoch` | CPU service 实例代次，拒绝重启前遗留的旧请求；当前实现由服务创建时生成，同一实例的 region 共用 |
| `source_epoch` | NPU 直读源身份的代次，用于拒绝旧的源描述符 |

CPU Pull 的源在初始化时绑定指定 DDR 池，因此包含 region 身份。NPU Pull 源描述的是 A 上被读取的 HBM，不绑定 B 的某块输出 HBM。

上面的结构只携带本批所需身份，以及 CPU Pull 所需的选中 MR 长度。完整 MR 清单保存在 KVAF 内部，不随 ReadyToken 反复传递。NPU 源的访问凭据、HBM 基址、segment 和通道描述由 KVAF 在 register(receiver=...) 的建链连接中自动交换。应用不再转发 export_ticket 或维护 hbm_link。

CPU 源的连接和 MR 保存在 CPU service，B Runtime 无需另存一份初始化源绑定。每批直接读取 ReadyToken.source，先在本地检查描述符格式和 DDR region/epoch，再通过已有的提交前 Query 核对 mr_length 与选中 MR 的真实长度及批次状态，CPU 在 Submit 时校验实际注册范围。NPU 源自动接入 B 的监听 Runtime，按 source_id/source_epoch 保存连接；每批按 mr_id 查询内部 MR 清单。内部长度列表对单 MR 和多 MR 都非空，不再另存首个 MR 长度；初始化及重连仍核对完整清单和访问凭据。

### 4.2 ReadyToken

```text
ReadyToken {
    source: CpuPullSourceDesc 或 NpuPullSourceDesc
    sequence: 本次生产的数据批次编号
    mr_id: 本批选择的 MR，由 ready(tensor) 自动填写
}
```

`ReadyToken` 不包含 `readable_ranges`、源/目的地址列表、完整 MR 长度列表、业务 key 或实际数据。这里的 Token 是批次身份描述，不是 URMA 内存访问 token。Python pickle 使用 v2 类型标记并拒绝旧格式，Connector 仍只需原样传递对象。

单 MR 使用 `source.ready()`，两种源的多 MR 都使用 `source.ready(tensor)` 选择已注册的 tensor。接口登记批次并返回身份信息。应用应在确认生产写入完成后调用它；该接口不再接收 offset/length 列表，也不承担业务层的数据可读性判断。

CPU Pull 的 Ready 状态登记在 CPU service；NPU Pull 的 Ready 状态登记在 A 的源对象中。批次身份用于将读取、完成、取消和释放关联起来。源未完成/未释放时不能发布新批次；Ready 控制请求响应丢失时，重试必须沿用原 sequence，不能另建批次。

A Connector 向 B 发送的是：

```text
应用业务元数据：逻辑块、数据布局、业务请求标识等
KVAF ReadyToken：源身份、sequence 和 mr_id；CPU Pull 另带本批 mr_length
```

业务元数据由应用解释，KVAF 只校验资源和批次身份。

### 4.3 三个连续、等长数组

对外接口直接接收三个数组，不再传入逐段结构体列表，也不要求额外构造批次包装对象：

```text
src_offsets[N]
dst_offsets[N]
lengths[N]

第 i 段：从源偏移 src_offsets[i] 读取 lengths[i] 字节，写入对应目的位置
```

所有偏移和长度均以字节为单位。连续指 Tensor 输入及内部三个元数据数组各自在内存中连续，不要求它们相邻，也不要求其描述的源或目的数据区间连续。

| 数组元素 | CPU Pull | NPU Pull |
|---|---|---|
| `src_offsets[i]` | 相对 A 注册 HBM 的逻辑起点 | 相对 A 注册 HBM 的逻辑起点 |
| 目的位置 | `dst_offsets[i]`：PreparePut 返回的 address 减 DDR base，可加槽内偏移 | `dst_offsets[i]`：相对 B 的 output 缓冲起点 |
| `lengths[i]` | 本段实际传输长度 | 本段实际传输长度 |

两条入口分别采用与执行位置一致的元数据形式：

| 接口 | Python 三数组类型 | C++ 三数组类型 |
|---|---|---|
| `submit_cpu_pull()` | 一维、连续 CPU `torch.int64` Tensor；也接受 list、tuple、一维整数 NumPy 数组，与 `batch_put()` 共用转换规则 | Host `Uint64ArrayView` |
| `batch_get_raw_addr()`（ReadyToken 新入口） | Runtime 当前 NPU 上的一维、连续 `torch.int64` Tensor；不隐式转换 CPU Tensor、list 或 NumPy | `NpuInt64ArrayView`，Host 不解引用其中的设备指针 |

三个数组的元素数量必须相同，范围为 1～4096，偏移非负，每个 `lengths[i]` 大于零。CPU Pull 在 Host 检查数值并保存连续 `uint64_t` 快照，然后通过控制协议发送到 CPU service；接口返回或抛出异常后，应用可以复用这三组 CPU 元数据。

NPU Pull 在 Python 入口检查类型、形状、连续性、设备及元素数量，Host 另检查元数据存储不与整个 output 分配范围重叠。随后只启动一次 raw READ V2 算子，每个输入段直接生成一个 WQE，填写时计算基址加偏移并读取对应长度。默认不预扫描长度，由应用保证每条长度为 1..UINT32_MAX 且符合设备单次 READ 上限、源/目的范围合法、目的段不重叠；`validate_ranges=True` 可开启正长度、32 位长度字段及源/目的边界诊断，不检查目的重叠或查询设备能力。最多 4096 个输入段，对应最多 4096 个 WQE，无 64 KiB 拆分或前缀表。

各核直接按输入请求数量划分任务，单个请求完整交给一个核处理；双 bank、多 Jetty、inflight 窗口和 CQ 调度沿用原实现。没有独立准备算子、中间状态回传或展开描述符的 HBM 缓冲，三数组不经 Host 中转。Host 在同一次算子完成后读取最终状态，检查各核总 WQE 数等于输入请求数，以及各自已提交/处理数量。该实现的真实性能仍需硬件验证。

NPU 三数组不得与整个 output 分配范围重叠，且必须与 output 一起在传入 stream 上就绪；如果应用在其他 stream 生成它们，应先建立事件依赖。接口执行期间不得修改或释放它们。Python 绑定保留三数组和 stream 到正常完成或恢复确认安全；output 硬件注册缓存保留到连接关闭，Python Tensor 引用保守保留到 Runtime shutdown 成功。如果 `pending_raw_addr` 为真，应用仍不得修改或复用本批存储。C++ 调用者自行保持存储有效。已启用的设备内容检查失败时不会发 READ，KVAF 向生产端确认取消该已发布批次；生产端释放该批次后才能发布新的 ReadyToken 重试。

应用不再逐段创建对象，原 READ 的 WQE 编码、多 Jetty 调度及 CQ 处理规则保持不变。公开入口只保留 NPU Tensor 三数组形式；建链探测内部保留专用 Host 描述符路径。

CPU Pull 的地址转换为：

```text
提取时计算 dst_offsets[i] = PreparePut 返回的 address - DDR base + 槽内偏移
CPU READ 源地址          = 源描述符中的 HBM 基址 + src_offsets[i]
CPU READ 目的地址        = DDR base + dst_offsets[i]
```

减基地址放在提取 PreparePut 地址时完成：便利入口内部处理；显式调用者使用
`runtime.remote_region["base"]` 计算目的偏移。`submit_cpu_pull` 直接接收并保存偏移快照，
不再转换目的数组。Runtime 根据 Put 的真实预留范围检查偏移归属，CPU 控制协议仍使用原来的相对偏移格式。既有范围、溢出、精确覆盖和重复提交一致性检查保留；这属于 CPU Pull 显式事务的检查，不表示普通 BatchPut 已提供相同检查。

NPU Pull 的地址转换为：

```text
源地址   = KVAF 根据 ReadyToken.source 找到的 A HBM 导出基址 + src_offsets[i]
目的地址 = B output 本地地址 + dst_offsets[i]
```

B 不解引用 A 的本地指针。注册对齐、导出地址转换和必要的本地内存注册仍由 KVAF 处理。

## 5. 控制服务地址

控制服务地址是 KVAF 用于交换建链信息、查询任务和确认传输状态的 `IP:端口`，与数据面的 EID 和 HBM 地址分别管理。

| 地址 | 用途 |
|---|---|
| `cpu_address` | CPU service 的控制地址，用于分配 DDR、注册 CPU Pull 源、提交/查询任务和完成确认 |
| B 的 NPU 读取端控制地址 | NpuRuntime.listen_hbm 建立唯一监听，所有生产端主动连接；交换注册、建链和完成确认 |
| Connector 业务地址 | 应用自己的消息通道，由应用选择和管理 |

```python
# B（读取端 D）
receiver = runtime.listen_hbm("0.0.0.0:29001", advertise_address="10.0.0.2:29001",
                              local_eid=b_fullmesh_eid)
# Connector 将 receiver 原样交给 A（生产端 P）
npu_source.register(tensors, receiver=receiver, stream=producer_stream)
```

此初始化调用完成全部双端建链与逐 MR 探测，成功返回 None。0.0.0.0 仅用于绑定，advertise_address 必须能从生产端访问。应用无需再次 connect、发送 export_ticket 或 attach_hbm。数据走 NPU READ，TCP 只传控制信息。

NPU 源不提前绑定 B 的目标 HBM，是因为 `batch_get_raw_addr()` 每次接收 `output`。B Runtime 对该缓冲执行范围检查和必要的本地注册，并保持它的生命周期。CPU Pull 当前则在初始化时借用并注册指定 DDR 池；这是两条路径的资源组织差异。

## 6. 初始化接口

下表列出当前接口。旧类名、带范围的 Ready 以及旧 raw READ 重载已删除。

| 调用方 | 接口 | 返回与作用 | 变更 |
|---|---|---|---|
| B Connector | `runtime.initialize()` | 选择本卡设备、校验配置；不访问 CPU 或申请 DDR | 基础初始化 |
| B Connector | `runtime.link_to(cpu_address)` | 配置 CPU service 目标；可在基础初始化前后调用 | DDR 可选依赖 |
| B Connector | `runtime.initialize_ddr(ddr_bytes, slot_size)` | 在基础初始化后分配共享 DDR、初始化 MetaManager 和启用的 DDR 路径 | 独立 DDR 初始化 |
| B Connector | `runtime.initialize(ddr_bytes, slot_size)` | 依次执行基础和 DDR 初始化，需要先 link_to | 保留组合入口 |
| B Connector | `runtime.shutdown_ddr()` | 安全释放 DDR 相关资源，保留 NPU Pull 连接及 CPU 目标 | 独立 DDR 清理 |
| B Connector | `runtime.offload_region()` | 返回 DDR ticket，由应用发送给 A | 保留 |
| A Connector | `CpuPullHbmSource(device_id, npu_eid)` | 创建 CPU Pull 源对象 | 原 `HbmSource` 更名 |
| A Connector | `cpu_source.register(ddr_ticket, hbm_cpu, stream)` | 注册源、建立 CPU→A 链路并探测，返回 None | 一次初始化，无需读取端 attach |
| A Connector | `NpuPullHbmSource(device_id, npu_eid)` | 创建 NPU Pull 源对象 | 原 `HbmReadSource` 更名 |
| A Connector | `npu_source.register(hbm_npu, receiver=receiver, stream=stream)` | 自动连接、注册、建链、探测，返回 None | 一次初始化 |
| B Connector | `runtime.listen_hbm(listen_address, advertise_address="", local_eid="")` | 返回 receiver ticket，自动接入各生产端 | 每 Runtime 一个监听 |

CPU 源在 register 内完成与 CPU 的注册、建链和全部 MR 探测，CPU 已保存真实源信息；不再向 B 单独发送 source_desc，也不要求 B 调用 attach_cpu_source。ReadyToken.source 在运行阶段携带提交所需身份。NPU 源在 register 内主动连接 B，随后由 KVAF 完成 Open/Memory/Channel/Confirm 和全部 MR 探测，返回时可直接 ready。

纯 NPU Pull 使用 initialize → listen_hbm，仅需 raw kernel，无 CPU service 或 DDR 依赖。initialized 表示基础就绪，ddr_initialized 表示 DDR 就绪。普通 BatchGet/BatchPut、MetaManager 和 CPU Pull 要求 DDR 就绪；shutdown_ddr 不销毁 NPU Pull 链路。两端 fullmesh EID 分别配置在 A 的源对象和 B 的 listen_hbm 上。

应用只转发 receiver ticket，不转发 NPU export_ticket。TCP 重连必须匹配原接收端代次、源身份及 MR 清单，不重建硬件或重放 READ。注册/探测期间保持源数据稳定，失败保留源对象并重试 source.shutdown。

初始化时应用交换的内容：

```text
B Connector → A Connector：DDR ticket
B Connector → A Connector：NPU Pull receiver ticket
A Connector → B Connector：应用数据布局；源注册和建链信息由 KVAF 自动交换
```

源对象、MR、Jetty 和 TP 保存在 KVAF 内部；应用交换的只是票据与业务元数据。

`runtime.offload_region()` 返回给生产端的 DDR ticket 使用 version=2，仅包含：

| 字段 | 用途 |
|---|---|
| `version` | 票据格式版本，当前为 2 |
| `address` | CPU service 控制地址 |
| `region_id`、`epoch` | 绑定指定 DDR 池并拒绝旧实例的请求 |
| `token`、`eid` | 注册、建链时使用的访问凭据与 CPU 路由身份 |

DDR 的 base、bytes、slot_size、candidate 等完整布局仍由读取端 Runtime 和 CPU service 保存。生产端不需要这些字段；读取端应用通过 `runtime.remote_region` 取得 base、length、slot_size、device，目的地址提取流程不变。旧 version=1 DDR ticket 不再接受。NPU receiver ticket 仍使用 version=1。

### 6.1 内部连接描述与 MR 描述

初始化报文和资源存储将共享连接与单个 MR 分开：

| 内部结构 | 内容 | 保存与交换时机 |
|---|---|---|
| `ConnectionDesc` | 路由 EID、TP、PSN、连接 token、四个 Jetty key 及 key 长度 | 每个源连接保存；CPU Attach、NPU Open/Channel 时交换 |
| `MemoryDesc` | MR 逻辑 base/bytes、内存 token、导出 segment 及长度 | 每个 MR 独立保存；CPU AddMemory、NPU Memory 只交换 MR 描述 |

新增 MR 不再复制或发送整份连接描述。MR 的导入段、token ID、注册句柄仍逐块保存；实际注册范围和逻辑可读范围的区别也保留。NPU 各通道独立协商 TP、CPU 共享 TP，以及两条路径独立注册的方式不变。

CPU Pull 的初始化和对接点如下（D=读取端、P=生产端、K=CPU service）：

```mermaid
sequenceDiagram
    participant D as D Connector / Runtime
    participant P as P Connector / CpuPullHbmSource
    participant K as K CPU service
    D->>K: initialize_ddr：分配共享池
    D->>P: 应用通道：DDR ticket
    P->>K: register 内部：注册信息、建链、逐 MR 探测
    K-->>P: 注册就绪
    Note over D,P: 无单独 source_desc 消息，无 attach_cpu_source
    P->>P: 生产并同步，ready(tensor)
    P->>K: 登记批次身份
    P->>D: 应用通道：ReadyToken 和业务元数据
    D->>D: PreparePut、地址组织及本地校验
    D->>K: Query：核对选中 MR 和批次；Submit：提交读取
    K-->>D: 完成状态
    D->>K: Commit/Abort 确认
    D->>P: 应用通道：完成通知
    P->>K: release_batch 检查确认
```


### 6.2 CPU Pull：多个 tensor 注册，共用连接

```python
# 生产端：每个 tensor 各自连续，tensor 之间可以不连续。
source = kvaf.CpuPullHbmSource(device_id=producer_device)
source.register(ddr_ticket, tensors, stream=producer_stream)  # 返回 None
# 注册信息由 KVAF 与 CPU 自动交换，读取端不需要另一次 attach。

# 每批只读一个 tensor；源偏移都相对此 tensor 的起点。
selected = tensors[layer_id]
# 应用在 selected 上生产数据，并确保写入完成。
producer_stream.synchronize()
ready = source.ready(selected)
# Connector 转发 ready 和业务元数据；读取端计算三数组后：
put = runtime.prepare_put(keys, lengths)
ddr_base = runtime.remote_region["base"]
dst_offsets = [entry.address - ddr_base for entry in put.entries]
runtime.submit_cpu_pull(ready, put, src_offsets, dst_offsets, lengths)
runtime.wait_cpu_pull(put)
runtime.commit_put(put)
# 应用通知生产端可以复用；生产端：
source.release_batch(ready)
```

- `register()` 接收一个 tensor 或 1～4096 个 tensor 的 list/tuple。每个 tensor 必须在同一张 NPU 上连续且非空；列表中范围不能重叠。注册列表固定到 `shutdown()`，更换列表需先关闭。
- 每个 tensor 保留独立的注册范围、导出 segment 和 token ID。列表共用一个 HCCP context、一组 4 Jetty 及其 TP。CPU service 在一个 source_id 下导入全部 MR；DDR 的 pinned 接收 MR 由同一 Region 统一持有，不随 MR 数量增加。
- 建链时对每个 MR 做最多 64 字节的 scratch READ 校验，全部成功后 register 才成功返回；探测期间应用保持所有源数据稳定，不覆盖业务 DDR。候选失败时清理整组注册和连接后再回退。
- `ready(tensor)` 按注册时相同的基址、长度和设备查找 MR；子视图或未注册的 tensor 被拒绝。单 tensor 注册仍支持无参数 `ready()`。应用不需要查找或传入 MR 句柄。
- CPU 从 `source_id + mr_id` 选取本批的导入段、导出基址和长度。每批选择一次，所有分段共享它；不跨 MR。切换 MR 清除旧 WQE 计划缓存，队列和完成序号继续复用。
- 同一源一次只有一个冻结批次。重复 `ready()` 必须选择同一个 tensor；重复 Submit 必须保持 MR、sequence 和三个数组一致。Read 未决时保留整组 MR、连接和 DDR lease，禁止换 MR 或销毁源。
- 注册期间所有 tensor 的分配必须存活、地址和长度不能变化。正常运行只要求当前冻结 tensor 不被改写，其余 tensor 可由应用生产。Python 绑定保存独立的 tensor 引用列表，不依赖应用保留原 list 对象。
- NPU Pull 已独立支持多 MR，使用 `NpuPullHbmSource.register(tensors, receiver=receiver)` 和 `ready(tensor)`；不共享 CPU Pull 的 MR。详见下节。

### 6.3 NPU Pull：多个 tensor 注册，共用连接

多 MR NPU Pull：生产端 register(tensors, receiver=receiver) 固定 1～4096 个 Tensor，自动连接读取端并导入全部 MR、完成逐 MR 全通道探测。ready(tensor) 选中本批唯一源，batch_get_raw_addr 的 src_offsets 相对该 Tensor，dst_offsets 相对 output，三组 NPU int64 数组不变。

每个 NPU 源只有一组 Jetty/CQ/TP，各 MR 保留自己的内存身份。AIV Host 缓存“MR ID + output 注册”对应的轻量通信描述，复用 output 的本地 MR 和队列计数；现有 raw READ 算子及 ABI 不变。单 tensor 使用 register(tensor, receiver=receiver)/ready()。内部注册清单包含 mr_bytes，KVHR v7 在控制消息中携带 mr_id；应用不查询或传递 MR 句柄。注册约束、失败保留和硬件测试见 [NPU 多 MR 说明](hbm_raw_addr_read.md)。

## 7. A 端批次接口

两个源对象提供一致形式的接口：

| 接口 | 语义 |
|---|---|
| `source.ready() → ready_token` | 在应用保证生产完成后登记本批身份；不接收地址范围，不发起 READ |
| `source.release_batch(ready_token)` | 检查对应批次已经安全结束，解除本批占用；不注销长期 HBM 注册 |
| `source.cancel_batch(ready_token)` | CPU 源仍只取消未获取的批次；NPU 源必须先由读取端 cancel_raw_addr 确认，不能单方面撤销 ready |
| `source.shutdown()` | 在无未决批次和对端引用后释放连接与注册 |

`release_batch()` 和 `cancel_batch()` 显式匹配源、sequence 及 mr_id，避免旧的应用通知作用于新批次。

```python
# A Connector；应用自行实现生产和业务通知。
produce(hbm)
wait_until_production_complete()  # 应用保证生产写入已经完成
ready = source.ready()
connector.send_to_b(business_metadata, ready)

# 等待应用通道的对应批次完成消息。
completion = connector.receive_completion()
source.release_batch(completion.ready)
# 成功后可以覆写下一批；释放底层缓冲还需要 shutdown 成功。
```

应用负责从数据就绪到读取结束期间不覆写、不释放源。KVAF 可以拒绝提前注销、错误取消和释放错误批次，但不能阻止应用直接通过 Tensor/指针修改内存。

## 8. B 端 CPU Pull 接口


### 8.1 接口与完成边界

| 接口 | 返回与作用 |
|---|---|
| `batch_cpu_pull(keys, ready, src_offsets, lengths, timeout=None) → None` | 一 key 一段，封装 Prepare/Submit/Wait/Commit；本卡与非本卡一致，见第 11 节 |
| `resolve_cpu_pull(timeout=None) → None` | 恢复便利入口内部的原事务，不重发 READ；仅成功提交返回 None |
| `prepare_put(keys, lengths) → put` | 包装现有 MetaManager 的 PreparePut，预留槽位并返回目标地址，此时本批新数据尚未发布为可读 |
| `submit_cpu_pull(ready, put, src_offsets, dst_offsets, lengths) → None` | 使用 ready 携带的源身份，校验并提交 CPU READ，记录在原 put 中；不代表数据完成 |
| `query_cpu_pull(put) → status` | 查询原 put 的状态，不重发 READ；未提交时返回 PREPARED |
| `wait_cpu_pull(put, timeout) → result` | 成功返回 DONE 或 COMMITTED；只等待数据完成，不自动 Commit；尚未提交时立即报错 |
| `commit_put(put) → result` | 检查关联传输完成，提交 MetaManager，并完成 CPU 侧批次确认 |
| `abort_put(put)` | 在元数据未提交且确认无在途 DMA 时撤销事务 |

`put` 是应用调用 `runtime.prepare_put(keys, lengths)` 得到的 **PutHandle 写事务句柄**。它代表本次为这些 key 预留的 DDR 槽位及后续提交责任，不包含待搬运的数据，也不是一次已经执行完毕的 BatchPut。传入 `submit_cpu_pull()` 不会再次分配槽位。

Runtime 保存真实事务记录，应用得到该记录的本地句柄和只读分配描述：

```text
PutHandle
    事务编号
    所属 Runtime
    DDR region 身份
    entries：每个 key 的 address 和有效数据长度 length
    state：最近一次已知事务状态
    submitted：是否已记录本次提交请求
```

目标槽位仍采用当前 MetaManager 的 DSA 规则；接口公开不改变 key 到槽位的分配策略。应用不能自行构造一个 put 句柄来绕过预留。

三个数组只说明“从哪里搬到哪里、搬多少字节”，`put` 还提供目的槽位的归属和提交上下文。提交时 KVAF 根据它确认 `dst_offsets` 指向本次预留的有效范围，将请求快照与传输状态保存在同一事务中，防止其他任务覆盖槽位或在 DMA 未完成时释放槽位。读取成功后 `commit_put(put)` 发布这批 key 的新数据；安全撤销时 `abort_put(put)` 回收本次预留。

`put` 只在创建它的 B Runtime 中使用，不作为对象发给 A 或 CPU service；CPU service 通过控制协议中的 region、源和任务身份执行读取。NPU 直接读取路径的目的地由 `output` 指定，因此不需要 put。

提交请求在首次 RPC 前记录到原 put，并设置 `submitted=True`。它只表示本地已有请求记录，不代表 CPU 已接受或 DMA 已完成。提交响应丢失时直接用原 put 查询、等待及恢复，不另取 task，也不能重新 PreparePut 替代未决操作。

### 8.2 地址组织和检查

B Connector 根据业务元数据确定 key、有效长度以及源位置，在 PreparePut 返回的范围内组织传输：

```python
import torch

metadata, ready = connector.receive_from_a()
keys, key_lengths, source_offsets = organize_data(metadata)

put = runtime.prepare_put(keys, key_lengths)
src_offsets = torch.tensor(source_offsets, dtype=torch.int64, device="cpu")
ddr_base = runtime.remote_region["base"]
dst_offsets = torch.tensor(
    [entry.address - ddr_base for entry in put.entries],
    dtype=torch.int64, device="cpu",
)
lengths = torch.tensor(key_lengths, dtype=torch.int64, device="cpu")

runtime.submit_cpu_pull(
    ready, put, src_offsets, dst_offsets, lengths,
)
result = runtime.wait_cpu_pull(put, timeout)
runtime.commit_put(put)

# 传输和提交已完成；此后的业务处理及通知由应用决定。
connector.send_completion_to_a(ready, result)
```

示例展示成功路径；失败时按第 10 节保留或撤销资源，不能在任意异常后无条件 Abort。

示例中一个 key 对应一段传输，因此 `lengths` 与 `key_lengths` 相同。应用也可以把一个 key 拆成多段：PreparePut 的 lengths 按 key 定义有效数据长度，提交接口的 lengths 按传输段定义长度，两者数量不必相同；提交接口的三个数组之间必须等长。

KVAF 检查源身份和 sequence，检查源区间位于已注册 HBM 中，检查长度非零及地址加法不溢出。它不判断业务元数据所描述的数据是否已经生产完成。

B Runtime 根据 MetaManager 的真实预留记录检查目的范围：

- 所有目的区间属于本次 put，不能只检查它们位于整个 DDR 池内。
- 目的区间不能相互重叠。
- 三个数组描述的全部目的区间合起来完整覆盖本次准备提交的有效数据长度，不能留下未写入的空洞。
- 一个 key 可以拆成多段，源区间可以重排；槽内偏移也必须满足上述检查。

CPU service 保留 region 身份、注册范围、目的不重叠和任务一致性检查。MetaManager 在 B 端，事务归属和覆盖检查在 B Runtime 中完成，不在 CPU 端创建第二套 MetaManager。

### 8.3 CPU Pull 时序

```mermaid
sequenceDiagram
    participant AC as A Connector（应用）
    participant AS as A KVAF 源
    participant BC as B Connector（应用）
    participant BR as B Runtime
    participant CPU as CPU service

    AC->>AC: 生产数据并确认写入完成
    AC->>AS: ready()
    AS-->>AC: ReadyToken
    AC->>BC: 应用消息：业务元数据和 ReadyToken
    BC->>BR: prepare_put(keys, lengths)
    BR-->>BC: put 和目标地址描述
    BC->>BC: 计算三个连续、等长数组
    BC->>BR: submit_cpu_pull(ready, put, src_offsets, dst_offsets, lengths)
    BR->>CPU: 提交读取任务
    BR-->>BC: None（请求保存在原 put）
    CPU->>AS: READ A 的 HBM
    CPU->>CPU: 写入 DDR，等待各队列尾部完成 CQE
    BC->>BR: wait_cpu_pull(put)
    BR-->>BC: 数据完成
    BC->>BR: commit_put(put)
    BR->>BR: 提交 MetaManager
    BR->>CPU: 确认批次结束
    BR-->>BC: 提交完成
    BC->>AC: 应用自己的业务完成通知
    AC->>AS: release_batch(ready)
    AS-->>AC: 允许复用源缓冲
```

`commit_put()` 包含库内部 CPU 批次确认，不再要求应用调用额外的 `finish_cpu_pull()`。

## 9. B 端 NPU Pull 接口

保留 `batch_get_raw_addr` 名称，调整为由 B 显式提供三个连续、等长数组：

```python
result = runtime.batch_get_raw_addr(
    ready, src_offsets, dst_offsets, lengths, output, stream,
)
```

| 接口 | 作用 |
|---|---|
| `batch_get_raw_addr(...) → result` | 按三个数组描述的区间读取 A HBM 到 B output，第一版同步完成 |
| `resolve_raw_addr(ready)` | 校验源身份及批次号后恢复控制确认或资源清理，不重发 READ |
| `cancel_raw_addr(ready)` | 在读取端记录批次不可再读，并向生产端确认取消；响应丢失用 resolve_raw_addr 恢复 |

KVAF 在 Host 检查数组布局、元数据/output 不重叠和批次身份；NPU raw READ V2 直接从三个 Tensor 为每条请求生成一个 WQE；完整批次的长度/边界诊断仅在显式启用时执行。不会先生成展开的设备描述符数组。

该路径不需要 PreparePut、CommitPut 或 AccessTable。A 的源描述符无需绑定 B 的目标 HBM；B 每次传入 output，KVAF 管理读取所需的本地注册和缓冲引用。第一版成功返回时，输出已完成且传输控制确认已结束。

### 9.1 output 的含义

`output` 是 **B 应用分配的本地接收 HBM 缓冲**，Python 接口中为位于 B 卡、内存连续的 NPU Tensor。它是输入参数中的目标缓冲，调用会直接写入它；接口返回的 `result` 表示完成状态，数据本身位于 output 中。读取按字节搬运，不进行 Tensor dtype 转换。

每一段的目的地址为 `output.data_ptr() + dst_offsets[i]`，目的范围必须位于 `output.numel() * output.element_size()` 字节以内。偏移相对本次传入 Tensor 的逻辑起点，KVAF 在注册时处理底层对齐。多个目的区间可以有间隔，未覆盖的字节不属于本次读取结果。

例如，B 将 A 上两段不连续的 4 KiB 数据读入一个连续的 8 KiB 缓冲：

```python
import torch

with torch.npu.stream(stream):
    output = torch.empty(8192, dtype=torch.uint8, device=b_device)
    src_offsets = torch.tensor([4096, 12288], dtype=torch.int64, device=b_device)
    dst_offsets = torch.tensor([0, 4096], dtype=torch.int64, device=b_device)
    lengths = torch.tensor([4096, 4096], dtype=torch.int64, device=b_device)

result = runtime.batch_get_raw_addr(
    ready, src_offsets, dst_offsets, lengths, output, stream,
)
# 成功返回后，output[0:4096] 和 output[4096:8192] 分别包含上述两段数据。
```

output 在读取完成前不得并发覆写。其注册按地址/长度缓存到连接关闭，Python 保守保留 Tensor 引用到 Runtime shutdown，C++ 调用方负责保活；缓存期间不得调整大小或替换底层 storage。若接口因未决传输而失败，应用不得复用该缓冲，直到恢复流程确认可以安全使用。调用方负责范围合法、目的区间不重叠以及长度为 1..UINT32_MAX 且符合设备单次 READ 上限；默认关闭长度/边界诊断，可用 `validate_ranges=True` 显式开启，目的区间两两重叠检查已删除。

| 参数 | 表示什么 | 数据最终写入哪里 |
|---|---|---|
| CPU Pull 的 `put` | PreparePut 返回的写事务及槽位归属 | CPU DDR 中本次预留的槽位，由相对 DDR base 的 dst_offsets 定位 |
| NPU Pull 的 `output` | B 应用提供的本地 NPU Tensor | 该 Tensor 的 HBM 范围，由 dst_offsets 定位 |

### 9.2 NPU Pull 时序

```mermaid
sequenceDiagram
    participant AC as A Connector（应用）
    participant AS as A KVAF 源
    participant BC as B Connector（应用）
    participant BR as B Runtime

    AC->>AC: 生产数据并确认写入完成
    AC->>AS: ready()
    AS-->>AC: ReadyToken
    AC->>BC: 应用消息：业务元数据和 ReadyToken
    BC->>BC: 准备 output HBM 和三个连续、等长数组
    BC->>BR: batch_get_raw_addr(ready, src_offsets, dst_offsets, lengths, output, stream)
    BR->>BR: 一次 raw READ：可选长度/边界诊断、每段生成一个 WQE、等待 CQ
    BR->>BR: Host 检查各核最终状态及 WQE 数量
    BR->>AS: 库内部传输完成确认
    BR-->>BC: 输出就绪，返回结果
    BC->>AC: 应用自己的业务完成通知
    AC->>AS: release_batch(ready)
    AS-->>AC: 允许复用源缓冲
```

NPU ready 即发布不可单方面撤销的读取许可；不再发送 AcquireRaw，D 本地检查后直接 READ，完成后发送 Complete。三数组只供本地 NPU 使用。P 保持数据稳定直到结束确认和 release。取消必须由 D 调用 cancel_raw_addr，记录禁止迟到通知发起 READ 的状态，再确认给 P。CPU Pull 仍通过 Submit 将三数组发送给 CPU service。

Connector 的业务通知由应用实现，KVAF 的 Complete/Cancel 不替代业务消息。未决控制确认用原 ReadyToken 恢复，不重发 READ。

## 10. 事务、重试和资源生命周期

| 情况 | 允许的处理 |
|---|---|
| PreparePut 后尚未提交 READ | 可以 AbortPut |
| READ 全部完成、尚未 CommitPut | 可以 Commit，或在确认 DMA 结束后撤销写事务 |
| 提交响应丢失、超时或 DMA 状态不确定 | 保留原 put、目标槽位和源缓冲，用原 put 查询；不能释放并复用地址 |
| Meta 已提交，但 CPU 确认响应丢失 | 重试 commit_put 的确认阶段，不重复 Commit，不改为 Abort |
| NPU READ 完成但控制确认未完成 | 用 resolve_raw_addr 恢复确认，不重读数据 |
| A 收到延迟或重复的业务完成消息 | 按源和 sequence 匹配；不能释放另一个批次 |

重复提交必须关联原任务且参数一致。已完成请求返回原结果或明确的已完成状态，不能再次产生 READ；冲突参数必须拒绝。未知 DMA 状态不自动重建链路或重放请求。

公开的 put 状态与现有 Offload 的 pending 保护共用，避免拆出接口后普通 BatchPut、BatchFree 或 Shutdown 绕过未决资源保护。句柄超时、退出作用域或异常都不代表 DMA 已结束。

生产端注册持有底层缓冲直到 shutdown 成功。批次 release 只允许覆写下一批，不代表已经可以释放整块注册内存。

正常关闭顺序为：停止新任务并处理完已有批次，A 释放对应批次；NPU Pull 由 A source.shutdown 内部请求 B 关闭链路并确认退役；CPU Pull 由 A 关闭 CPU 源以释放 CPU service 上的源连接和 DDR lease；最后 B 关闭 Runtime 和共享 DDR。业务协调仍由 Connector 执行。

数据 checksum 保持独立诊断，不纳入传输计时。若应用需要校验，应在目的数据释放/覆写、下一次 Ready 或源注销前完成；校验结束后再通知 A 复用可直接沿用当前验收流程。

## 11. 现有代码的调整范围

CPU Pull 当前使用 `src_offsets + dst_offsets`，在提取 PreparePut 的 address 时减去 DDR base；Submit 不再转换目的数组。NPU 源绑定由 Runtime 管理；CPU 源连接由 CPU service 管理，Runtime 直接使用 ReadyToken.source 提交，保存本次事务以支持恢复。CPU Pull 的公开事务进一步统一为 PutHandle：Submit 成功返回 None，Query/Wait/Commit/Abort 均使用原 put，已删除 CpuPullTask 和 put.task。基础初始化与 DDR 初始化现已分离，纯 NPU Pull 不再依赖 DDR。本次描述符精简不改变 Raw READ 算子、批次同步与恢复语义，但更新了 ReadyToken 序列化格式、DDR ticket 和 KVRP/KVHR 报文；CPU service、两端 NPU 运行库和 Python 绑定需一起重编译、重启。

| 当前实现 | 目标调整 |
|---|---|
| `HbmSource` | 更名为 CpuPullHbmSource，register 返回 None，ReadyToken 内携带源描述符 |
| `HbmReadSource` | 更名为 NpuPullHbmSource，保留独立 NPU 控制协议与绑定实现 |
| `ready(offsets, lengths, stream)` | 新接口 ready() 只登记批次；应用负责生产完成和数据稳定 |
| Ready 携带最终源范围，Submit/Acquire 必须逐项匹配 | 具体范围来自 B 的三个数组；CPU Pull 保存 Host 请求快照，NPU Pull 在同一次 READ 算子中按需诊断并直接生成 WQE，控制面只获取批次读取权 |
| PreparePut/CommitPut 隐藏在 RuntimeCore 的完整 Offload 中 | 增加受 Runtime 状态管理的公开事务接口 |
| CPU Submit 隐藏在 RemotePullOperator | 增加 submit/query/wait 接口，由同一 put 管理预留、传输与提交 |
| CPU 目的地址必须落在 slot 起点 | 支持本次预留有效范围内的拆分和槽内偏移，保持覆盖、归属及不重叠检查 |
| batch_get_raw_addr 的源地址列表来自 Ready batch | 扩展入口由 B 提供三个连续、等长数组，保留原算子和通信实现 |

CPU Pull 控制协议 KVRP 使用 v6，NPU 直读控制协议 KVHR 使用 v7。本次精简需更新 CPU service、两个 NPU 应用及运行库。协议版本不符时拒绝通信。两种 Ready 都发布源身份、批次号及本批 MR 身份。

已删除：旧类名别名、`ready(offsets, lengths, stream)`、无 token 的 release/cancel、`offload_from()`、`offload()`、`resolve_offload()`、`release_offload_source()`、旧 raw READ 重载、独立 OffloadRuntime 及 KVOF 服务、直接传本卡 tensor 的 batch_cpu_pull 重载。

统一便利入口为（本卡、非本卡使用相同形式）：

```python
runtime.batch_cpu_pull(keys, ready, src_offsets, lengths, timeout=None, profile=False)
```

源端先通过 `CpuPullHbmSource.register(ticket, tensor_or_list)` 注册，再在每批生产同步后
调用 `ready(tensor)`。消费端直接使用收到的 ReadyToken，不需要初始化 attach 或逐 MR InspectSource RPC。原来的
`batch_cpu_pull(keys, tensor, input_offsets, lengths, stream=...)` 已删除，不再隐式注册本卡源。

便利入口与 `batch_put()` 一样同步返回；默认成功返回 `None` 表示数据搬运及元数据提交完成。
可选 `profile=True` 成功返回 `{"profile": {...}}`，包括绑定参数转换、C++ 总时间、校验、
PreparePut、目的偏移组织、Submit、Wait、Commit 和 native_other 的 Host 耗时（微秒）。
失败仍抛异常，不返回可用的性能样本；默认关闭计时，恢复不合并跨调用的样本。

```python
result = runtime.batch_cpu_pull(keys, ready, src_offsets, lengths, profile=True)
print(result["profile"]["wait_cpu_pull_us"])
```

内部执行流程：

```text
ReadyToken 携带源和 MR 身份，校验 DDR 归属 → PreparePut → 提取 entry.address 并减 DDR base，生成 dst_offsets
→ SubmitCpuPull → WaitCpuPull → CommitPut → 返回 None
```

PreparePut 仍返回绝对地址，普通 BatchPut 不变。显式 `submit_cpu_pull` 的第四个参数现为
`dst_offsets`，旧的绝对地址调用须改为 `entry.address - runtime.remote_region["base"]`。
便利入口签名不变；当前 CPU 控制协议为 KVRP v6，应用和 CPU service 需要配套重编译。

`keys`、`src_offsets`、`lengths` 是三个等长的 Host 整数序列或连续一维 CPU int64 Tensor，
每个 key 对应一个源段，顺序一一对应。偏移和长度以字节计，源偏移相对 Ready 选中的 tensor；
每个长度必须大于零且不超过目标槽位容量。最多 4096 段，一批只读取一个 MR。
`timeout` 单位为毫秒，作用于完成轮询等待，与 `wait_cpu_pull` 相同；不改变各控制 RPC 的网络超时。
`None` 使用 Runtime 的 `offload_timeout_ms`（默认 30000），0 表示查询一次、不等待轮询。

应用无需持有内部 Put 或组装 `dst_offsets`。一 key 多段、槽内偏移和自定义目标布局继续使用
显式 PreparePut/Submit/Wait/Commit 接口，两者共享同一个 MetaManager 和 CPU 数据面。
预注册的 1..4096 个 MR 都适用，便利入口不按批次注册、同步生产 stream 或释放生产端批次。
成功后由消费端 Connector 通知生产端，生产端调用 `release_batch(ready)`。
本卡也按同样的生命周期使用；注册和 tensor 存储由 `CpuPullHbmSource` 持有。

参数/提交前失败会撤销已预留的槽位；明确的终止传输失败会安全 Abort 后抛异常。
超时或丢失提交响应时保留内部 Put 和请求，`pending_offload=True`，应用保持源数据不变。
恢复入口无需任务句柄：

```python
runtime.resolve_cpu_pull(timeout=30000)
```

恢复只查询、等待原任务；确认完成后 Commit，确认失败后 Abort，绝不重新发起 READ。
只有成功 Commit 才返回 `None`，确认失败或仍未决都抛异常；可通过 `pending_offload`
区分已安全撤销和仍需保留资源。Commit/Abort 确认丢失时保留原决定，不能反向处理。
没有内部未决批次时调用 resolve 会报错；显式 Put 仍使用原句柄恢复，resolve 不接管它。
相同参数再次调用 `batch_cpu_pull()` 表示新操作，不能用于恢复，未决时会拒绝新批次。

`shutdown()` 也会尝试查询一次并完成内部事务，不重发 READ、不等待在途 DMA。
源的 Release/Cancel/Shutdown 仍归生产端；关闭消费端 DDR 前须先注销所有生产端源。

## 12. 第一版范围

- 一个源同时一个活动批次；一个 Ready 批次对应一次读取消费。
- 一个 Put 事务对应一个 CPU Pull 任务，不引入多任务拼接提交。
- 同一 DDR 池一次只有一个未完成或未确认的 CPU Pull。
- 同一 Runtime 的这些控制接口按顺序调用，不新增并发调用保证。
- 保留现有单个 NPU 源绑定一个 reader 的范围；同一 Runtime 可绑定多个源，通过 ReadyToken 的源身份区分，连接由 KVAF 内部保存。
- 应用负责 A/B 的业务元数据交换、业务完成通知、生产同步和源数据稳定。
- KVAF 负责注册、建链、地址与事务检查、数据搬运、完成状态和安全清理。

## 13. 实现对接与验收

### 13.1 Python 返回值与恢复入口

- `cpu_source.register(...)` 成功返回 `None`，无需向读取端额外发送 source_desc 或调用 attach_cpu_source；CPU Pull 线协议为 KVRP v6。
- CPU 源描述符保留在 `ReadyToken.source` 中；本地格式、DDR 归属及 Put 覆盖检查失败时不发 RPC，提交前 Query 和 CPU 端 Submit 仍校验选中 MR 与批次。取消了对所有 MR 的初始化 InspectSource 查询。
- `source.ready()` 返回 `ReadyToken`；应用调用前同步生产 stream。
- `prepare_put(...)` 返回 `PutHandle`：`entries` 提供只读的 `key/address/length`，`state` 为最近一次已知状态，`submitted` 表示是否已记录提交请求。`state` 不会自动轮询远端。
- `submit_cpu_pull(...)` 成功返回 `None`，失败抛异常。响应丢失时请求仍保存在原 put 上，继续通过它 Query/Wait；不得重新 PreparePut 替代未决任务。`submitted=True` 不表示 CPU 已接受或完成读取。
- `query_cpu_pull(put)` 返回 `CpuPullState`，需要时查询远端并更新 put.state；尚未提交时返回 `PREPARED`。`wait_cpu_pull(put, timeout=30000)` 的 timeout 单位为毫秒，成功返回 `DONE` 或已提交的 `COMMITTED`，不会自动 Commit；未提交即调用 Wait 会立即报错。
- `commit_put(put)` 返回 `COMMITTED`，`abort_put(put)` 返回 `ABORTED`。提交响应丢失后重试原 commit，不能反向 Abort。PreparePut 后尚未提交任务时，可以直接 Abort。
- 新 `batch_get_raw_addr(...)` 成功返回 `{"state": "done", "sequence": ready.sequence}`，实际数据直接写入 output。其未决恢复使用原批次的 `resolve_raw_addr(ready)`，不会重发 READ，也不能用其他批次的 ReadyToken 完成确认。
- 可选 `profile=True` 会在成功返回值中增加 `profile` 字典，提供 Host 大阶段和设备事件测得的算子区间；默认关闭，详见 [raw READ 计时与基准](hbm_raw_addr_read.md)。

显式事务使用 `query_cpu_pull`、`wait_cpu_pull`、`commit_put`、`abort_put` 恢复。统一 `batch_cpu_pull()` 内部管理事务，异常恢复使用 `resolve_cpu_pull()`；生产端源生命周期由应用管理。当前版本从 PreparePut 起保护这笔事务，完成或撤销前会拒绝同一 Runtime 的其他 DDR 访问操作。

如果提交前的参数校验失败，`put.submitted` 为 false，可以修正数组后继续提交，或撤销 put。请求记录后即使 CPU 尚未接受也保持 submitted 为 true，使用原 put 查询：`DONE` 可提交或撤销，`FAILED` 可撤销，`RUNNING` / `IN_DOUBT` 保留资源。已记录请求的参数不能更换；重复提交只确认与原请求一致，不重发 READ。未成功通知 CPU 的失败可能仍让生产批次处于 Ready，应用协调源端调用 `cancel_batch(ready)`；源端会拒绝取消已被获取的批次。

C++ 的共享类型与数组视图在 `common/include/connector_types.h`，事务句柄与 RuntimeCore 接口在 `runtime/include/runtime.h`，NPU 包装接口在 `runtime/include/npu_runtime.h`。CPU 源仍通过 `runtime/include/remote_offload.h` 引入，NPU 源通过 `runtime/include/hbm_read.h` 引入。C++ 调用者在未决期间自行保持源和输出内存有效，Python 绑定保留对应 Tensor 引用。

### 13.2 硬件验收入口

继续使用现有构建选项，重新构建并重启两端程序。脚本现在默认只使用新接口，已移除 `--connector-api` 开关：

```bash
python tests/python/test_remote_cpu_offload.py \
  --cpu 127.0.0.1:28999 --device-a 6 --device-b 7

python tests/python/test_hbm_read_raw_addr.py \
  --cpu 127.0.0.1:28999 --device-a 6 --device-b 7 \
  --npu-eid-a-raw 000000000047050000100000df00d806 \
  --npu-eid-b-raw 000000000047050000100000df00f807 \
  --batch-get-kernel build-shared-ddr/bin/kvaf_execute_batch_get_kernels.o \
  --batch-put-kernel build-shared-ddr/bin/kvaf_execute_batch_put_kernels.o \
  --raw-kernel build-shared-ddr/bin/kvaf_execute_batch_get_raw_addr_kernels.o
```

上述 EID/设备对应此前同机验收配置，构建目录按实际路径替换。两个旧测试脚本的参数名称继续沿用 **device-a 为读取端、device-b 为生产端**，与本文设计中的 A/B 角色命名相反。

CPU 验收覆盖显式 PreparePut/Submit/Wait/Commit、重复提交、DDR checksum 和批次复用确认。组合验收同时覆盖共享 DDR 的普通 BatchPut/BatchGet、CPU Pull 和三数组 NPU 直读，包括非页对齐源、输出间隔及 NPU 消费者校验。组合脚本的 `--hbm-mrs` 同时控制 CPU/NPU Pull 独立注册的同一份 tensor 列表，每轮选择同一个 tensor，等待两条路径完成后复用。统一便利入口的本卡场景另外通过 `python tests/python/test_cpu_offload.py --cpu 127.0.0.1:28999 --device 7` 验证；该脚本无需 AIV 算子。跨卡便利入口使用 `test_remote_cpu_offload.py --pull-api batch`，可同时传入 `--tensors 78 --rounds 156`。

需要同时重新编译 NPU 运行库和 raw-address kernel object。当前入口为 `KvafExecuteBatchGetRawAddrAivKernelV2`，使用扩展的参数与最终状态 ABI；旧 V1 object 会拒绝加载。已删除独立准备算子及其状态 ABI。

### 13.3 性能入口

- `tests/python/bench_remote_cpu_offload.py`：统一 batch_cpu_pull 封装，测量整个 Python 调用的总时间；
  默认开启可选 profile，输出绑定转换及内部 PreparePut、地址组织、Submit、Wait、Commit，
  校验/释放仍在总时间之外。`--no-profile` 关闭内部计时。CPU 服务通过 `KVAF_CPU_PULL_PROFILE=1` 汇总每个源的 READ 阶段。命令与统计口径见 [CPU Pull 使用说明](cpu_pull_offload_usage.md)。
- `tests/python/bench_hbm_read_raw_addr.py`：两卡 NPU Pull，可配置等大包数量、包大小及随机散布的源空间大小。分别执行完整调用带宽轮次和开启内部阶段/设备事件的诊断轮次，校验均在传输计时外。命令见 [NPU raw READ](hbm_raw_addr_read.md)。
