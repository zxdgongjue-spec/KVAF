# KV Access Operator

## 定位

`AccessOperator` 是 Runtime 的同步数据面接口。Runtime 负责 DDR、
MetaManager、AccessTable 和 Key 锁的生命周期；Operator 只把已经准备好的
caller-owned buffer 与远端 DDR 组织为一次数据搬运。它不拥有 Key 到地址的映射，
也不分配或释放调用方的输入、输出和状态 buffer。

`AccessOperator` 是抽象接口，供本地 Mock 和硬件实现注入。当前 CANN 构建会提供
`DefaultAccessOperator`；它是唯一的 NPU 实现入口，不向上层暴露 AIV/AICPU 等
路径选择。

## 当前接口与能力

| 接口 | 当前行为 | 目的 |
| --- | --- | --- |
| `Initialize(remote_region)` | 已支持 | 加载 BatchGet kernel，并为 Runtime-owned Host DDR 创建可复用 transport。初始化不接收 output HBM。 |
| `ExecuteBatchGet(request)` | 已支持，默认同步 | 使用 request 内 caller-owned NPU output/status/queue 和 stream，Acquire 对应的 local HBM buffer，launch 后按 execution 选项同步等待。相同 `(output 地址, bytes)` 由底层 transport 缓存复用。 |
| `ExecuteBatchPut(request)` | 已支持，测试 mock | 对 caller-owned NPU input block 执行同步 `aclrtMemcpy(DeviceToHost)` 到 MetaManager 预留的 Host DDR 地址，然后由 Runtime CommitPut。它只用于当前 BatchGet 集成测试，不是最终 BatchPut 数据面。 |
| `Shutdown()` | 已支持 | 释放 transport 及其缓存的 local HBM MR。调用方必须先完成所有外部 stream/graph 工作。 |

当前 BatchGet 是固定大小 block 的同步 POC：NPU 侧 `indices`、`mask`、
`output_indices`、device AccessTable、`output`、`status` 和 `queue_state` 都由
调用方创建并通过 `BatchGetRequest` 传入。没有公开的 `BindBatchGetOutput` API。

## 生命周期

```text
Runtime.Initialize
  -> DefaultAccessOperator.Initialize(remote DDR)
  -> TransportCreate(remote DDR)             # EID / CTP / Jetty / SQ / CQ

Runtime.BatchGet(request with output A)
  -> LocalBufferAcquire(A)
  -> launch + stream synchronize
  -> LocalBufferRelease(A)

Runtime.BatchGet(request with output B)
  -> LocalBufferAcquire(B)                   # 同一 transport，不重建链路
  -> launch + stream synchronize
  -> LocalBufferRelease(B)

Python / torch_npu external graph
  -> warm up Runtime.BatchGet(output A) outside capture
  -> torch.npu.graph capture(pre-op -> Runtime.BatchGet(stream, synchronize=false) -> post-op)
  -> torch_npu owns replay / graph destruction / stream completion

Runtime.BatchPut(request with NPU input)
  -> aclrtMemcpy(DeviceToHost) per reserved block  # temporary test mock
  -> Runtime.CommitPut

Runtime.Shutdown
  -> TransportDestroy                         # 释放缓存的 local HBM MR
```

`LocalBufferAcquire/Release` 是 demo host 库的内部协作接口，Operator 上层只调用
`Runtime.BatchGet`。默认同步调用会在 Release 前完成 stream；外部 graph capture
只能使用已预热、因而命中 transport cache 的 output，并由 Python/框架保证 graph
replay 完成后才释放 tensor、AccessTable 和 Runtime。KVAF 不创建、保存或 replay
任何图对象。

## 构建范围

不带 CANN 时只构建接口和 CPU Mock，用于本地 Python 测试。启用
`KVAF_ENABLE_AIV_BATCH_GET=ON` 时构建 `DefaultAccessOperator` 和现有的 BatchGet
demo；名称中的 AIV 只保留在 demo/kernel 实现目录，非 Runtime 的公开路径选择。

Selector / Executor / Template 分层、通用图调度/与上游框架图的整合、异步执行与真实
BatchPut 数据面均不属于当前范围。CANN Adapter 只提供可被上游固定地址图 capture
的 BatchGet 提交，不管理图对象。

## 后续待决事项：BatchGet / BatchPut Transport

当前 `DefaultAccessOperator` 只接入 BatchGet demo 的 transport。BatchPut 数据面
合入时，必须由两个 demo 的接口与 ABI 共同确认 EID、CTP、Jetty、SQ/CQ 是否能
安全共享；不能仅因两者都使用 Direct-HCCP 就假设可复用。

- 若控制面资源、token 语义和队列并发模型兼容，则提取共享 transport，并在一次
  `Initialize` 中创建。
- 若任一项不兼容，则 BatchGet 与 BatchPut 分别持有各自 transport，并在
  `Initialize` 中各自初始化、在 `Shutdown` 中按各自完成状态释放。

在 BatchPut 接口和硬件实测就绪前，保持当前 BatchGet-only transport，不提前抽象。
