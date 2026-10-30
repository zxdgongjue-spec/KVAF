# CPU Pull / Offload 实现设计

本文记录最初的独立 OffloadRuntime。后续整合版本见
[共享 DDR 与跨卡 CPU Pull](shared_ddr_remote_offload.md)：NpuRuntime 统一 Meta/DDR，支持 A 提交、CPU 读取 B。

## 基线与范围

本分支 `codex/cpu-pull-hbm-reference` 从 `main@c4375e1` 创建，传输参考
`aicpu-urma@8bc5a2a` 的 `cpu_push_get_transport.cc`、`cpu_push_get_worker.cc`
和 `cpu_push_host.cc`。实现独立的 NPU HBM → CPU DDR Offload；不接入 Decode。

保留 BatchPut 的元数据事务：`PreparePut → 搬运 → CommitPut`。搬运由 CPU 主动
提交 URMA READ，不启动 AIV BatchPut kernel。Python `offload()` 同步返回；CPU
收到 Submit 后在工作线程执行 READ。

以下三个文件从参考提交原样引入，避免重新定义传输 ABI 和队列参数：

- `common/include/urma_transport.h`：32 字节 SGE、CTP、Jetty 导入及队列辅助函数。
- `common/include/urma_device_abi.h`：部署环境的 496 字节设备属性结构。
- `access_operator/include/direct_hccp/target_channel.h`：NPU target CQ/Jetty 生命周期。

控制协议、PreparePut/CommitPut 和 Python 源 Tensor 保留机制沿用前一版 Offload
的接口设计；CPU 数据面及 NPU 建链按上述参考重新实现。

## 参考参数

| 项目 | 当前实现 |
| --- | --- |
| NPU HBM 注册 | `MemRegFlags(true)`，值 `0x29c1` |
| CPU 导入 HBM | NON_CACHEABLE、REMOTE_READ \| REMOTE_WRITE、NOMAP |
| CPU DDR | 4096 字节对齐，注册 access=1、non_pin=0、显式 token ID |
| 认证 token value | CPU 分配 region token，HBM、DDR、Jetty 导入使用同一值；各端 token ID 独立 |
| NPU PSN | 1 |
| CPU PSN | 服务端递增序列，首个值 2 |
| Jetty 数量 | 双端各 4 路，互相导入全部 key |
| NPU 队列 | SQ/RQ=1024、CQ=16384、priority=2、rnr_retry=7、err_timeout=16、error_suspend=1 |
| CPU 队列 | SQ/RQ=8192、CQ=16384、max_sge=min(16,设备上限)、max_rsge=1、lock_free=0、error_suspend=0 |
| CPU retry 参数 | 使用参考辅助函数中的 URMA_MAX_PRIORITY、URMA_TYPICAL_RNR_RETRY、URMA_TYPICAL_ERR_TIMEOUT；公共 24.12 头文件对应 15、7、17 |
| READ | queue 0、单源/目的 SGE、STRONG_ORDER、comp_order=1、每条 WR 请求完成通知 |
| 单次 READ 大小 | min(64 KiB, 设备 max_msg_size)，更大 span 分段提交 |
| CQ | 每次最多取 8 条，校验 status 与 user_ctx，单条 READ 最长等待 30 秒 |

`access=1` 使用参考实现的数值约定，公共 24.12 头文件称为 LOCAL_WRITE；不将此位
混入 NPU 远端访问权限。HBM 权限按参考保留 READ/WRITE/ATOMIC，但本功能只提交 READ。

配置 EID 用于选择 CTP 路由；MR 和 Jetty key 内的 EID 是 provider 导出的身份。
二者可以不同，所有导出字节原样交换，不改写为配置 EID。

## 初始化与 Offload 流程

1. `link_to()` 保存 CPU 服务地址。`initialize()` 默认枚举 CPU 候选和当前物理 NPU 的
   EID，按 CPU 外层、EID 内层依次尝试。每组请求分配独立 DDR 池，得到地址、
   长度、候选 EID、服务 epoch 和 region token。DDR 此时已分配，注册延后到第一次 Open。
2. NPU 创建 Direct HCCP context、token ID，注册源 HBM。查询 CTP 后创建四路
   TargetChannel，发送 Open，携带原始 segment、四个 key、路由 EID、TP、PSN。
3. CPU 查询设备能力，导入 HBM，再注册接收 DDR，查询 CTP，创建四路队列并导入
   NPU Jetty；Open 响应返回 CPU 四个 key、TP 和 PSN。
4. NPU 导入全部 CPU Jetty。等待生产者 stream 上的 ACL event，确保数据已生成。
5. 初始化使用独立的 Probe 命令：分配 4096 字节 HBM、填充 0xa5，在 CPU 服务线程
   从 HBM+64 读取 min(64,slot_size) 字节到 DDR slot 0。此步骤对应参考 Worker::Start
   的首次同步 READ。确认 CQ、释放源连接并检查 CPU DDR 的 FNV-1a 校验和后，初始化完成。
6. 用户调用 `offload(keys, source, offsets, lengths)`：Runtime 校验输入，调用
   PreparePut 预留目标 slots，按步骤 2–4 注册并连接本次源 Tensor，然后 Submit。
7. CPU 校验 span 范围、slot 边界及目标不重叠，启动工作线程。每段用 queue 0
   提交 READ，确认 CQ 后处理下一段。全部成功将状态设为 Done。
8. NPU Query 看到 Done 后，要求 CPU ReleaseSource，确认 CPU 释放源引用和队列；
   NPU 随后注销 HBM、销毁通道，最后 CommitPut，返回成功。CPU DDR 池保留到 Shutdown。

NPU EID 发现与 NpuRuntime 共用 `npu_eid_discovery.h`，顺序为 HCCL CLOS EID →
asdrv pair_info 的活动 device EID，去重并保留来源；共用同一 selector 文件锁。
CPU 候选数量通过现有 KVCP GetCandidateCount 查询，KVOF wire ABI 不变。
显式指定的 CPU 候选/EID 分别固定对应维度。成功后记录选中的组合，payload 不自动换路。

每次候选尝试包含独立 MetaManager、RuntimeCore、PullOperator 和探测 HBM。
探测失败且确认无在途 DMA 后，释放源、关闭 DDR 池、释放探测 HBM，才创建下一次尝试。
候选全部失败时返回尝试总数及最后错误；各次失败原因记录在两端日志。

## 与参考场景的必要适配

- 参考读取固定 HBM workspace 的通知和描述符；这里读取调用者传入的连续 Tensor，
  `offsets` 和 `lengths` 都以字节计，不通过额外 HBM staging buffer 搬运用户数据。
- 参考首次 READ 的目的地址是 CPU 控制区对应偏移；这里使用 DDR slot 0 起始处。
- 四路连接与参考一致，但当前 payload READ 仅使用 queue 0；未做多路并行性能优化。
- CPU DDR 池长期保留；每次任务单独注册源 HBM并重建连接，以支持不同 Tensor。
- 使用 TCP Submit/Query 传递任务，不引入参考的 CPU WRITE、AICPU scatter、HBM 轮询通知
  或 Decode 数据返回流程。
- NPU 异步 CTP 查询参数放在 Exporter 成员中；超时仍保留，避免异步接口访问失效栈内存。

因此这是复用参考的 ABI、注册、建链和 READ 数据面，并非原样移植整个 CPU Push/Decode 功能。

## 协议与生命周期

KVOF 使用版本 2，与主线 KVCP 协议分开。V2 Endpoint 交换四个 Jetty key，旧版
单 Jetty Offload 服务不兼容，部署必须同时更新 CPU 服务与 Python 扩展。
控制协议使用部署端相同的小端结构 ABI，不跨端传递可解引用的本地句柄。

命令为 Allocate、Open、Submit、Query、ReleaseSource、Close、Checksum、Probe。
`region + epoch + task` 标识任务；同一 task 的重复 Submit/Probe 必须携带相同 spans
及命令类型，重复请求不提交第二次 READ。Probe 限定一个不超过 64 字节的 span。

成功路径仅在 CQ 确认后发布元数据。注册/建链等未提交 READ 的错误可释放资源并 AbortPut。
payload READ 提交错误、CQ 错误、超时或完成关联异常进入 InDoubt：保留队列、MR、DDR、源 Tensor
和 pending 元数据，不允许复用目标 slots。TCP 响应丢失同样先保留源，随后通过 Query 恢复。
`resolve_offload()` 查询已有任务，不重发 READ；transport 已隔离时不会自动解除隔离。

启动 Probe 的特殊处理：只有一次 poll 返回唯一且 user_ctx 匹配的发送错误 CQE，
才能确认该唯一在途 READ 已结束；随后必须确认 CPU Jetty、JFR、CQ、HBM 导入全部销毁成功，
才将任务报告为 Failed 并允许候选回退。任何未知完成、提交/轮询错误、超时或销毁失败仍然
InDoubt，并立即停止候选遍历。TCP ACK 丢失也停止遍历，保留原 region/task 用于恢复。
Allocate ACK 丢失时，Shutdown 用同一 region ID 重试幂等 Allocate，然后关闭该池。

CPU 释放顺序沿用参考：本地 Jetty → 导入 Jetty → JFR → CQ → 导入 HBM。
关闭池时再注销 DDR、释放 token ID 和内存。NPU TargetChannel 使用参考 Shutdown。
部分连接创建失败也走同一清理路径。清理失败则继续保留相关资源。

Python binding 持有 pending 源 Tensor 的强引用；C++ 调用者负责保留源。单 Runtime
只支持串行调用。初始化探测未决时内部 HBM 同样保留；恢复成功也不自动跳过校验将 Runtime
标记 ready，应 Shutdown 后重新初始化。与主线元数据管理一致，每个内部 RuntimeCore
只注册一次池；后续候选或重新初始化使用新的 RuntimeCore/MetaManager。

## 验证边界

主机测试覆盖事务可见性、重复请求、四路 key/参数、部分建链清理、分段 READ、校验和、
生产者 event、初始化线程顺序、响应丢失和 CQ 失败隔离。自动选择测试覆盖发现顺序与去重、
跨 CPU/EID 回退、显式固定、校验和失败、候选耗尽，以及未知状态/清理失败停止切换。
URMA/ACL/HCCP 使用测试替身，
只能证明编排和传输描述符符合约定，不能证明真实 UB 链路可用。

真实硬件使用 `tests/python/test_cpu_offload.py`：先检查 64 字节启动探测，再验证三轮
不同数据、非连续源偏移和超过 64 KiB 的 payload。构建与运行见
[cpu_pull_offload_usage.md](cpu_pull_offload_usage.md)。

2026-10-29 用户确认，device=7、CPU candidate=0、NPU EID 为
`000000000000020000100000df00e100` 时，当前分支及原 `codex/cpu-pull-hbm` 分支
均在测试机器上成功。此前首次 READ 失败与固定选择另一个 NPU 路由 EID 有关；
参考分支的自动候选回退是复现成功配置时必须考虑的差异。详细配置见使用文档。
