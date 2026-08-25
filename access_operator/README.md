# KV Access Operator

## 定位

`AccessOperator` 是 Runtime 的数据面接口。Runtime 负责 DDR、
MetaManager、AccessTable 和 Key 锁的生命周期；Operator 只把已经准备好的
caller-owned buffer 与远端 DDR 组织为一次数据搬运。它不拥有 Key 到地址的映射，
也不分配或释放调用方的输入和输出 buffer。

`AccessOperator` 是抽象接口，供本地 Mock 和硬件实现注入。当前 CANN 构建会提供
`DefaultAccessOperator`；它是唯一的 NPU 实现入口，不向上层暴露 AIV/AICPU 等
路径选择。

## 当前接口与能力

| 接口 | 当前行为 | 目的 |
| --- | --- | --- |
| `Initialize(remote_region)` | 已支持 | 加载 kernel，并先取得按物理 NPU 共享的 Direct-HCCP RA/HDC session；随后分别创建可复用的 BatchGet、BatchPut transport。初始化不接收 caller HBM。 |
| `ExecuteBatchGet(request)` | 已支持，默认仅提交 | 使用 request 内 caller-owned NPU Tensor 与 stream，Acquire 对应的 local HBM buffer 并 launch。`status` / `queue_state` 是 Operator 内部 AIV ABI buffer；仅显式同步时读取并记录设备状态。相同 `(output 地址, bytes)` 由底层 transport 缓存复用。 |
| `ExecuteBatchPut(request)` | 已支持，同步 eager | 使用 caller-owned packed NPU `input`、每条 CPU `input_offsets/lengths` 和 Runtime `PreparePut` 返回的地址。Operator 将 metadata 上传到 NPU，调用 AIV Direct-HCCP WRITE，并在成功后由 Runtime CommitPut；失败则 Runtime AbortPut。未配置 BatchPut kernel 时保留 ACL D2H mock 供本地/BatchGet-only 构建使用。 |
| `Shutdown()` | 已支持 | 释放两个独立 transport 及其缓存的 local HBM MR；最后一个 transport 才释放共享 RA/HDC session。调用方必须先完成所有外部 stream/graph 工作。 |

当前 BatchGet 是固定大小 block 的 POC：NPU 侧 `indices`、`miss_count`、
`output_indices`、device AccessTable 与 `output` 都由调用方提供；`status` 和
`queue_state` 由 Operator 内部申请。Python 通过 `DeviceAccessTable` 和 Tensor
调用，不传裸指针/字节数。没有公开的 `BindBatchGetOutput` API。

## 生命周期

```text
Runtime.Initialize
  -> DefaultAccessOperator.Initialize(remote DDR)
  -> DirectHccpSessionAcquire(physical NPU)  # RaInit / HDC service，仅首次
  -> BatchGetTransportCreate(remote DDR)     # Get 的 RaCtx / CTP / Jetty / SQ / CQ
  -> BatchPutTransportCreate(remote DDR)     # Put 的 RaCtx / CTP / Jetty / SQ / CQ

Runtime.BatchGet(request with output A)
  -> LocalBufferAcquire(A)
  -> launch on caller stream
  -> LocalBufferRelease(A)

Runtime.BatchGet(request with output B)
  -> LocalBufferAcquire(B)                   # 同一 transport，不重建链路
  -> launch on caller stream
  -> LocalBufferRelease(B)

Python / torch_npu external graph
  -> warm up Runtime.BatchGet(output A) outside capture
  -> external graph captures(pre-op -> Runtime.BatchGet(stream) -> post-op)
  -> framework owns replay / graph destruction / stream completion

Runtime.BatchPut(request with packed NPU input + offsets/lengths)
  -> MetaManager.PreparePut
  -> upload offsets / lengths / prepared destination addresses to NPU
  -> BatchPut LocalBufferAcquire(input) + Direct-HCCP AIV WRITE + stream sync
  -> success: Runtime.CommitPut; failure: Runtime.AbortPut

Runtime.Shutdown
  -> BatchGetTransportDestroy / BatchPutTransportDestroy
  -> DirectHccpSessionRelease                 # 最后一个 transport 才 RaDeinit
```

`LocalBufferAcquire/Release` 是 demo host 库的内部协作接口，Operator 上层只调用
`Runtime.BatchGet/BatchPut`。默认 BatchGet 提交路径不替调用方同步；外部 graph capture 只能使用已预热、
因而命中 transport cache 的 output，并由 Python/框架保证 graph replay 完成后才释放
tensor、AccessTable 和 Runtime。KVAF 不创建、保存或 replay 任何图对象。

## 构建范围

不带 CANN 时只构建接口和 CPU Mock，用于本地 Python 测试。启用
`KVAF_ENABLE_AIV_BATCH_GET=ON` 时构建 `DefaultAccessOperator` 和 BatchGet demo；
`KVAF_ENABLE_AIV_BATCH_PUT=ON`（默认随前者启用）时额外构建 BatchPut demo 与
Direct-HCCP Runtime adapter。名称中的 AIV 只保留在 demo/kernel 实现目录，非 Runtime
的公开路径选择。

Selector / Executor / Template 分层、通用图调度/与上游框架图的整合、异步执行均不
属于当前范围。CANN Adapter 只提供可被上游固定地址图 capture 的 BatchGet 提交，
不管理图对象；BatchPut 是同步 eager 路径，不支持图模式。

## Direct-HCCP session 与 Transport 边界

BatchGet / BatchPut 的 demo 现在共享按 `{physical device, hdc type}` 索引的进程级
RA/HDC session。它只拥有 `RaInit`、`rtOpenNetService`、`RaDeinit` 与引用计数，避免
第二个 demo 重复 `RaInit`（HCCP `328002`）或在失败路径错误释放第一个 demo 的 RA。

两个数据面 transport 仍然独立：各自创建 `RaCtx`、token、MR、CQ、Jetty、TP 与
kernel entity。不能仅因两者共用 session 就假设 EID、CTP、Jetty、SQ/CQ 可共享。

- 硬件验收应确认第二个 transport 可以跨过 `RaCtxInit`、CTP、Jetty 创建，并完成
  Get/Put 数据校验；这验证的是两个数据面 transport 并存。
- 只有在控制面资源、token 语义和队列并发模型均确认兼容后，才考虑进一步共享某些
  data-plane resource；当前不提前合并它们。

当前 Runtime 始终保留独立 transport，不在 Get/Put 之间销毁、重建或切换链路。
