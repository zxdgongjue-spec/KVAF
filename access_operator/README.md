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
| `ExecuteBatchGet(request)` | 已支持，同步 | 使用 request 内 caller-owned NPU output/status/queue，Acquire 对应的 local HBM buffer，launch 后同步等待，再 Release。相同 `(output 地址, bytes)` 由底层 transport 缓存复用。 |
| `ExecuteBatchPut(request)` | `kNotImplemented` | 接口已由 Runtime 正常调用；当前没有 BatchPut 数据面，Runtime 会 AbortPut，避免 MetaManager 提交半完成数据。 |
| `Shutdown()` | 已支持 | 同步 stream 后释放 transport 及其缓存的 local HBM MR。 |

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

Runtime.Shutdown
  -> TransportDestroy                         # 释放缓存的 local HBM MR
```

`LocalBufferAcquire/Release` 是 demo host 库的内部协作接口，Operator 上层只调用
`Runtime.BatchGet`。当前 Runtime 是同步的，因此每次 Release 前已完成 stream
同步；如果未来加入异步接口，MR 的 Release/Destroy 必须改为等待对应 completion
event，不能沿用当前同步时序。

## 构建范围

不带 CANN 时只构建接口和 CPU Mock，用于本地 Python 测试。启用
`KVAF_ENABLE_AIV_BATCH_GET=ON` 时构建 `DefaultAccessOperator` 和现有的 BatchGet
demo；名称中的 AIV 只保留在 demo/kernel 实现目录，非 Runtime 的公开路径选择。

Selector / Executor / Template 分层、图捕获管理、异步执行与 BatchPut 数据面均不
属于当前范围。
