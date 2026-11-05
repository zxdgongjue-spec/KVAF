# AIV BatchGet Scatter / Gather 三阶段设计

> 2026-11-05：第3.1节已更新为当前 mailbox 控制通路；其余阶段保留最初设计过程。
> 当前固定优化、流水参数及计时口径见 [性能测试说明](aiv_scatter_performance.md)。

## 1. 文档状态与基线

- 日期：2026-10-28。
- 工作分支：`codex/aiv-urma-scatter-gather`。
- 基线：`origin/main`，提交 `c4375e13d7a9519d907ab6a5d11227795f207286`。
- 参考：`origin/aicpu-urma`，提交 `8bc5a2a3a1840b40f0e5fc80ee4f361cb62ffcba`。
- 状态：第一阶段代码已落地，本地模型与 Host 生命周期测试通过；CANN 编译和双机硬件验收待完成。第二、三阶段仍为设计。
- 第一阶段的代码入口、构建与测试说明见 [第一阶段实现说明](aiv_batch_get_stage1.md)。

目标是在 main 的 AIV 后端上支持 `cpu_push_scatter` 和 `gather_read` 两种
BatchGet 路径，复用参考分支的 CPU 服务、数据布局和控制协议。

实施顺序固定为：

1. **无流水顺序版本：先保证功能正确。**
2. **流水版本：加入分块流水及 AIV 内部流水，不增加软件在途控制。**
3. **在途控制：最后加入预算、完成检查点和窗口补发策略。**

本设计不要求移植完整 AICPU 后端。AIV 不能调用 UMDK 接口；设备侧通信通过
手工构造 USER_CTL WQE、写 SQ、敲 doorbell 和轮询 CQE 实现。

## 2. 接口和数据语义

`NpuRuntime` 已增加初始化选项：

```python
aiv_batch_get_mode="direct_read"       # 当前 AIV 路径，保持默认行为
aiv_batch_get_mode="cpu_push_scatter"  # CPU 回推，AIV 本地 scatter
aiv_batch_get_mode="gather_read"       # CPU gather，AIV READ 到输出
```

保留现有 DeviceAccessTable 和 addresses/valid 两种 Tensor 重载。三种模式采用
相同的有效前缀和输出映射：

```text
N = miss_count[0]
src[i] = addresses[indices[i]] + offsets[i]
dst[i] = output + output_indices[i] * token_bytes
仅处理 i ∈ [0, N)，每项搬运 token_bytes 字节。
```

空前缀不通知 CPU、不修改输出。允许重复源索引，输出行继续遵循现有的不重复
约定。调用方负责有效索引、偏移、地址和输出映射，以及相关资源的生命周期。
入口保留形状、容量、有效前缀长度及整数溢出检查。

新模式初始范围与参考实现对齐：单 EID、1～8 个 AIV 核、最多 65536 项，
最大 payload 字节数须能用协议的 uint32 字段表示。先实现单核，再补齐多核。
scatter 的四路 CPU Jetty 与 AIV 核数独立；gather 初始采用每核一条数据队列和
一条控制队列，明确区分核数与不同用途的队列数量。

模式选择、流水开关和在途预算应分别表达。前两个阶段的新路径不接入
`batch_get_max_inflight_total`，也不因现有默认值而隐式启用窗口调度。
已有 direct_read 的配置和行为仍由其原实现定义。

| 模式 | Payload 数据路径 | 额外搬运 | 额外 staging |
| --- | --- | --- | --- |
| direct_read | 离散 DDR → 离散 output | 无 | 无 |
| cpu_push_scatter | 离散 DDR → 连续 HBM → 离散 output | AIV 本地搬运 | HBM |
| gather_read | 离散 DDR → 连续 DDR → 离散 output | CPU memcpy | DDR |

## 3. AIV 如何传递地址清单和 notify

notify 是内存中的请求序号，不是一次 UMDK 函数调用，也不是 SQ doorbell。
SQ doorbell 通知通信硬件处理队列；notify 通知 CPU 软件处理新的 BatchGet。

### 3.1 Scatter：AIV 推送 CPU mailbox

当前实现由 NPU Host 注册 HBM workspace、CPU 服务注册可远端写入的 mailbox，
通过 V7 OpenCpuPushGet 交换两端 segment/token 和四路 Jetty。Jetty 0 同时提供
AIV 可发起控制 WRITE 的 USER_CTL SQ；CPU 四条 payload 队列保持不变。

```text
AIV                                      CPU worker
多核生成 sources[]，写入 HBM
核0等待构建完成，写 request header
WRITE header + 有效 sources[] 到 CPU mailbox
同一 SQ 强序 WRITE mailbox.notify
                                         acquire 轮询本地 notify
                                         校验本地 request，直接使用本地 sources[]
                                         更新 SGL 并开始回推 payload
```

头、清单和 notify 分开发送，绝不覆盖 CPU 拥有的 done/Ready/marker 源区。
每个有界控制批次只等待强序尾 CQE；最终 notify 排在全部清单之后。
`ReadRequest()`/`ReadDescriptor()` 名称沿用原接口，热路径不再发出 URMA READ。
具体 DMA 可见性与性能仍须在目标设备验证，当前说明见
[性能测试说明](aiv_scatter_performance.md)。

### 3.2 Gather：AIV 手写控制 WRITE WQE

初始化时 Host 建立控制 Jetty，注册 HBM 描述符和 scratch，导入 CPU DDR，
将 SQ/CQ 地址、token、TP 等实体信息交给 AIV。

每轮通过下列操作把清单发送给 CPU：

```text
AIV 构造 WRITE WQE → 写入 USER_CTL SQ → 等待 SQ 写入完成
    → st_dev 写 SQ doorbell → 硬件执行 WRITE → AIV 检查 CQE
```

具体发布顺序为：

1. WRITE 描述头和源清单到 CPU DDR descriptor，并确认有序完成。
2. WRITE descriptor 的远端地址到 CPU DDR command。
3. 强序 WRITE 本轮 sequence 到 CPU DDR notify。
4. CPU 轮询本地 notify，根据 command 定位清单并开始 gather。

清单和 notify 传输只需要单 SGE WRITE，较大清单按支持的传输长度分段。
main 的 AIV BatchPut 已提供手写 WRITE WQE 和 SQ doorbell 的实现基础。
多 SGE WQE 的新增工作用于后续“连续 DDR staging → 离散 output”的 READ。

## 4. 第一阶段：无流水顺序版本

本阶段先跑通单核，再扩展多 SGE 和多核。请求准备、远端处理、数据传输和
本地搬运按阶段确认完成。阶段内部允许队列容量范围内的批量提交，不能额外
加入“每提交固定数量就等待”的软件限流窗口。

### 4.1 单核 Scatter

```text
AIV 生成整批源地址清单
    ↓
确保清单可见，发布 notify
    ↓
CPU 拉取完整清单
    ↓
CPU 将整批 payload WRITE 到 HBM staging
    ↓
CPU 收齐全部数据完成，发布 done(sequence, result)
    ↓
AIV 等待本轮 done 成功，确保 staging 可读
    ↓
AIV 将整批 staging 按 output_indices 搬到 output
    ↓
等待本地写出完成，写状态并退出
```

- 复用 CPU 四路 Jetty、prepared WRITE 和现有协议。
- CPU 可以继续发布 chunk ready，但本阶段 AIV 不消费 ready，只等待整批 done。
- 四路 CPU Jetty 可以并行传输；CPU 传输与 AIV scatter 两个阶段不重叠。
- AIV 本地搬运使用单缓冲：读入、写出、等待完成，再处理下一块。
- 对 656 B 等非对齐行长精确处理有效字节，不能覆盖相邻输出行或 padding。

### 4.2 单核 Gather

```text
AIV 生成完整 GatherSpan 清单
    ↓
WRITE 描述符和清单到 CPU DDR，确认完成
    ↓
发布 command / notify
    ↓
CPU 将整批源数据 gather 到连续 DDR staging
    ↓
CPU 发布本轮最终 ACK
    ↓
AIV 等到 ACK 成功，再 READ staging → output
    ↓
收齐全部 READ 完成，写状态并退出
```

- 设置 `pipeline_chunk_bytes=0`，不使用 ready 前缀推动数据 READ。
- 必须等待本轮最终 ACK，不能把 started ACK 当成 gather 已完成。
- 首先使用现有单 SGE READ WQE，按 token 或必要的分段写入对应 output 行。
- 在物理队列容量允许时批量提交数据 READ，阶段末统一确认完成；不启用双组
  在途窗口，也不人为规定每组提交后必须排空。

### 4.3 无流水条件下补齐多 SGE 和多核

单核两条路径正确后，先完成 gather 多 SGE READ，再扩展多核。多 SGE READ
以连续 DDR 为一个远端源，对应多个离散本地 output SGE。

起步参考 SGL=6、单 WR 不超过 64 KiB，实际参数受目标设备和队列 ABI 限制。
需要独立验证：

- 多 SGE WQE 编码、实际 WQEBB 占用、SQ 环回和 owner 位。
- doorbell producer index 的单位及推进量。
- CQ 完成与对应 SQ 空间的回收。
- 大 token 跨 WR 分段，以及最后一条不满 SGL 的请求。

main 当前的单 SGE READ 使用固定 64 B WQE，不能直接将其“一条 WQE 占一个
WQEBB”的记账套用到多 SGE 格式。具体格式需要目标 HCCP/URMA ABI 验证。

多核 Scatter 由 0 号核生成整批清单并通知 CPU，各核等待整批 done，再搬运
各自区间。多核 Gather 为每核分配独立 lane，每个 lane 内仍严格执行
“清单发送完成 → CPU gather 完成 → 数据 READ 完成”。核间允许并行，本阶段
不引入 lane 内的 chunk 流水。核编号直接使用 `GetBlockIdx()`。

### 4.4 第一阶段验收

| 类别 | 覆盖内容 |
| --- | --- |
| 请求语义 | 空、单条、部分前缀、满容量；每轮改变地址、offset 和输出映射 |
| 数据布局 | 随机和顺序输出、输出间隔、重复源、未写区域 guard |
| 搬运边界 | 656 B 等非对齐长度、尾块、传输分段、多 SGE 尾项 |
| 多核和队列 | 非整除分区、空 lane、SQ/CQ 回绕、连续多轮 |
| 错误处理 | 旧序号不能误完成；CPU 错误、CQ 错误、超时和复用拒绝 |
| 可见性 | 同 stream 后继 NPU 算子直接消费 output，逐字节结果正确 |
| 调用方式 | 两种 Tensor 重载、同步调用、异步调用后的完成与状态回收 |

完成后保留可切换的顺序路径和性能基线，作为第二、三阶段的回归对照。

## 5. 第二阶段：流水版本，不做在途控制

本阶段将流水拆成两步，分别验证。数据 ready 后即可继续提交，不引入软件
在途预算、双组在途窗口或按预算进行的等待和补发。

### 5.1 分块流水

Scatter 沿用每个 worker 的 1024 行 chunk。CPU 每路 Jetty 在其数据之后强序
发布 ready；AIV 只消费四路共同完成的连续 chunk 前缀：

```text
CPU：WRITE chunk 0 → ready 0 → WRITE chunk 1 → ready 1 → ...
AIV：                         scatter chunk 0 ───────→ scatter chunk 1
```

AIV 无需等待整批 done 才开始 scatter，但 kernel 结束前仍必须确认本轮 done。
ready 需要检查 sequence、范围和进度回退；空区间不等待不存在的 chunk。

Gather 开启 CPU 128 KiB 连续前缀发布：

```text
CPU：gather chunk 0 → ready → gather chunk 1 → ready → ...
AIV：                        READ chunk 0 ─────────→ READ chunk 1
```

独立控制队列查询 `ready_bytes`，数据队列可以同时保留尚未完成的 READ。
只有某条 READ 覆盖的全部源字节都 ready，才允许提交该 READ；不能只检查起点。
CPU gather 的 chunk 边界可以落在 token 内部，AIV 必须按字节范围判断就绪。

此步先沿用简单的本地搬运实现，专门验证 ready 协议、错误传播和阶段重叠。
不为了凑固定提交组而等待整个下一组数据就绪。

### 5.2 AIV 内部流水

- Scatter 引入 UB 双缓冲，重叠下一块 staging 读入与当前块 output 写出。
- Gather 引入 WQE 构造和 SQ 写入/doorbell 提交流水。
- UB/WQE bank 只有在对应本地搬运完成后才能复用；这属于本地缓冲依赖。
- **不引入双组在途窗口，不接入 `batch_get_max_inflight_total`。**

UB/WQE 双缓冲不等于远端请求窗口。前者解决本地计算和搬运重叠，后者限制
已提交但尚未确认完成的通信量，属于第三阶段。

### 5.3 第二阶段验收和边界

- 所有第一阶段用例继续通过，并与顺序模式逐字节比较。
- 覆盖 ready 延迟、跳跃、回退、旧序号、未来序号以及 CPU 中途失败。
- 验证后续 chunk 传输/聚合与当前 chunk 消费确实重叠，不能只根据总耗时推断。
- 本轮源清单仍先完整准备：scatter 为整批清单，gather 为每个 lane 的清单。
  清单生成与该清单对应的 payload 传输暂不流水化。
- 不同 BatchGet 之间暂不重叠；跨批流水需要独立的多槽 workspace 和协议设计。

## 6. 第三阶段：在途控制

在正确性和无软件限流的流水性能基线稳定后，再引入主动限制提交压力的策略。

首先针对 Gather READ 设计：

1. 明确定义预算单位，分别记录 READ WR 数、实际占用 WQEBB 数和在途字节数。
   一个多 SGE READ 不能按一行 payload 估算流量。
2. 设置总软件预算并分配到各核，加入中间完成检查点。
3. 达到预算后等待对应完成，回收额度，再补发已 ready 的数据。
4. 可以评估双组 FIFO 窗口；只有该组所有相关 Jetty 的完成都确认，才回收整组。
5. 预算关闭时回到第二阶段行为，用于性能对照。

如果后续需要控制 Scatter 的提交压力，应在 CPU WRITE 侧单独定义预算和
回收规则，不能直接把 AIV READ 预算套到四路 CPU WRITE 队列上。

验收需要同时验证预算不超限、尾组和奇数预算处理、多核公平性、错误后停止
补发，以及相对第二阶段的吞吐和延迟变化。软件在途计数是保守的提交/完成
记账，不等于硬件实时执行中的请求数。

## 7. 三个阶段共同遵循的正确性要求

### 7.1 物理队列容量与软件在途预算的区别

第一、二阶段始终保留 SQ/CQ 容量检查、回绕处理和完成回收，不能覆盖仍在
使用的队列空间。当真实容量不足时，必须回收已完成空间或等待必要的完成。
这种等待是正确性要求，不属于第三阶段用于性能调节的软件限流策略。

不要求移除参考 CPU transport 为安全复用 WR/SGL 和真实队列容量而采用的
wave 排空；但前两个阶段不能额外加入可调预算或人为缩小的固定窗口。

### 7.2 完成和可见性

- Scatter：全部本地 output 写出完成，并确认本轮 CPU done 成功。
- Gather：全部数据 READ 完成，并确认本轮 CPU 最终 ACK 成功。
- 不能把 notify 已发布、doorbell 已写入、CPU started ACK 或某一路完成当成
  整批请求完成。
- 在 target hardware 上验证 AIV/远端 DMA 的可见性和保序，不能直接照搬
  AICPU 的缓存一致性假设，也不能仅用 CPU 模型测试证明设备可见性。
- ready/done 等控制区域明确写入方；轮询和缓存维护不能把旧值回写覆盖新值。

### 7.3 Workspace、异步调用与异常

Scatter 的 HBM workspace 约为 `65536 * token_bytes + 512 KiB 地址清单 + 控制区`。
调用方 output 由 AIV 本地写入，不必作为此路径的 URMA 目标注册。

Gather 在用户 DDR 数据区之外预留 staging、描述符和控制区，并注册所需 HBM
描述符/scratch 及 output。远端 workspace 不进入 KV slot 分配。

workspace 初始化时分配并复用，首版每个 Runtime 同时只允许一轮新模式
BatchGet 使用它。下一次 Get/Put 复用相关资源前先 Drain 并检查设备状态。

`synchronize=False` 表示入队成功；同步调用需要检查 stream 和自定义 kernel
status。异步调用的状态必须在后续 Drain 中检查，不能被下一轮覆盖。

超时、CQ 错误或部分提交失败后，相关会话进入不可复用状态。Scatter 释放 HBM
之前必须确认 CPU DMA 已停止；Gather 也须确认本端 DMA 和 CPU gather 已停止
使用对应资源。无法确认时保留资源，不能仅清零软件索引后继续执行。

请求序号按协议递增，耗尽时要求重建会话。新模式首版采用 eager，Graph replay
所需的设备侧序号生成和资源复用另行设计。

## 8. 代码组织与验证方式

第一阶段已按以下层次实现，后续阶段继续扩展：

| 层次 | 职责 |
| --- | --- |
| Runtime / Python binding | 模式配置、workspace 规划、执行选择和状态回收 |
| CPU Runtime / PeerLink | 移植 gather executor、push transport 及 enable/open/start/stop 控制协议 |
| AIV Host | transport、注册资源、workspace、kernel 参数和生命周期 |
| AIV kernel | Scatter / Gather 入口、请求生成、本地搬运和 USER_CTL 队列原语 |
| 公共 ABI / loader | 公共布局、版本化参数及独立 kernel symbol 加载 |

CPU 公共代码应与 AICPU 设备接口解耦，AIV 构建不依赖 AICPU kernel JSON 或
设备端 liburma。尽量复用参考分支 wire layout；扩展协议时明确版本和能力检查，
不假设仅靠现有主协议版本号即可识别新增能力。

主要参考位置：

- main：`access_operator/demo/aiv_batch_get/` 的 READ、CQ 和现有 ABI。
- main：`access_operator/demo/aiv_batch_put/kernel/execute_batch_put_aiv.asc` 的
  单 SGE WRITE 和 SQ 提交。
- aicpu-urma：`docs/aicpu_cpu_push_scatter.md`、`docs/aicpu_gather_read.md`。
- aicpu-urma：`runtime/src/cpu_push_get_transport.cc`、`cpu_push_get_worker.cc`、
  `cpu_gather_executor.cc` 及对应公共布局。

每个步骤单独验证后再进入下一步。协议状态、布局和队列回绕可使用本地模型
测试；CANN 编译、远端 DMA 保序、AIV 可见性及性能必须在目标服务器验证。

性能比较固定请求数、有效 payload、输出映射、核数、Jetty 数和 EID/设备配置。
端到端统计覆盖调用至完成；阶段 profile 单独采样，不把重叠阶段耗时相加。

建议实施顺序：

```text
阶段一：单核顺序 Scatter → 单核顺序 Gather → 顺序多 SGE / 多核
阶段二：chunk 流水 → AIV 本地搬运 / WQE 构造提交流水
阶段三：软件在途预算 → 完成检查点与窗口补发 → 参数和性能评估
```
