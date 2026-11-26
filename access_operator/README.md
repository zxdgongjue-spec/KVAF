# KV Access Operator

## 定位

`AccessOperator` 是 Runtime 的数据面接口。Runtime 负责 DDR、
MetaManager、AccessTable 和 Key 锁的生命周期；Operator 只把已经准备好的
caller-owned buffer 与远端 DDR 组织为一次数据搬运。它不拥有 Key 到地址的映射，
也不分配或释放调用方的输入和输出 buffer。

`AccessOperator` 是抽象接口，供本地 Mock 和硬件实现注入。Runtime 的执行调度器按
`GetExecutionConfig` / `PutExecutionConfig` 选择算法。每种 Get 配置懒初始化并保留
资源与执行状态；调用方可在应用运行期间交替使用已启用的算法。

## 目录

```text
access_operator/
  algorithms/
    get/
      direct/
        urma_aiv/         # URMA Direct Host、ABI、AIV kernel
        ubmem_aiv/        # UBMEM Direct AIV kernel
        common/           # 两种 Direct kernel 的组合编译单元
      gather/            # urma_aiv/、urma_aicpu/ 的 Gather Host、ABI 和 kernel
      scatter/           # urma_aiv/、urma_aicpu/ 的 Scatter Host、ABI 和 kernel
      common/aiv/        # AIV Gather/Scatter 共用的状态 ABI 和设备辅助函数
      common/aicpu/      # AICPU 共用 ABI、loader、设备入口和测试
    put/
      urma_aiv/           # URMA AIV Put
      urma_aicpu/         # URMA AICPU Put
      ubmem_aiv/          # UBMEM AIV Put kernel
      common/aiv_kernel/  # 两种 AIV Put kernel 的组合编译单元
  providers/
    urma_aiv/             # 跨 Put/Get 的 Runtime adapter、资源 owner
    urma_aicpu/           # 跨 Put/Gather/Scatter 的 adapter、资源 owner
    ubmem_aiv/            # 跨 Put/Get 的 adapter、资源 owner
    common/               # 请求参数、capture/profile 辅助实现
  transports/
    direct_hccp/          # HCCP session 与传输封装
    aicpu_urma/           # AICPU 会话、通信实现、设备包构建与安装
    ubmem/                # HAL 生命周期与 DDR 设备映射
  include/                # 稳定的公开接口、资源视图与传输头文件
  cmake/                  # AIV 编译工具链
```

`algorithms/` 按数据搬运算法分类；`providers/` 保留目前跨 Put/Get 的适配器和
资源 owner 组合，负责将 Runtime 请求接到具体算法。目录分层与对象的生命周期
分开管理，具体所有权和复用边界见实现文档。

算法目录的 `common/` 只保留已有的共享 ABI、loader、设备入口或组合编译单元，
不表示合并不同算法的 session、Jetty、队列、workspace 或完成状态。
例如 URMA Direct 和 UBMEM Direct 仍使用原有组合 `.o`，各自 kernel 实现分目录存放。
UBMEM provider 和公共映射代码分别位于 `providers/ubmem_aiv/` 与 `transports/ubmem/`。

CMake target、库名和导出符号保持不变。AIV 产物仍在 `build-npu/bin/`；
AICPU 构建包位于 `build-npu/access_operator/transports/aicpu_urma/`，统一构建与
安装脚本已使用该路径。验证入口仍在仓库根目录的 `tests/run_*.sh`。

AIV Gather/Scatter 的 provider 工厂位于 `providers/urma_aiv/`。两者各自把通信资源 owner
交给框架，分别在 Invocation 中绑定 eager 和图节点状态。CPU 侧按 `runtime/cpu_service/gather/`
和 `runtime/cpu_service/push/` 组织。独立 Pull 仍使用 `runtime/src/` 的控制和注册路径，
NPU raw READ kernel 位于 `algorithms/get/raw_addr/urma_aiv/`。

Get 的显式图租约、常驻实例和资源边界见
[实现与使用](../docs/cache_transport_execution_implementation.md)，运行方式见
[功能与性能验证](../docs/cache_transport_execution_validation.md)。Put 保持同步调用。

## Legacy AIV 接口与能力

以下两节保留旧适配器的历史说明，不作为当前统一 Runtime 的调用契约。当前 AIV/UBMEM eager Get
不回读检查设备状态；普通注册、显式绑定和图节点租约的生命周期以[实现与使用](../docs/cache_transport_execution_implementation.md)为准。

| 接口 | 当前行为 | 目的 |
| --- | --- | --- |
| `Initialize(remote_region)` | 已支持 | 加载 kernel，并先取得按物理 NPU 共享的 Direct-HCCP RA/HDC session；随后分别创建可复用的 BatchGet、BatchPut transport。初始化不接收 caller HBM。 |
| `ExecuteBatchGet(request)` | 已支持，默认仅提交 | 使用 request 内 caller-owned NPU Tensor 与 stream，Acquire 对应的 local HBM buffer 并 launch。`status` / `queue_state` 是 Operator 内部 AIV ABI buffer；不逐次清零 status，显式同步等待 stream 后回读诊断日志，设备 status 不转换为调用失败。相同 `(output 地址, bytes)` 由底层 transport 缓存复用。 |
| `ExecuteBatchPut(request)` | 已支持，同步 eager | 使用 caller-owned packed NPU `input`、每条 CPU `input_offsets/lengths` 和 Runtime `PreparePut` 返回的地址。Operator 将 metadata 上传到 NPU，调用 AIV Direct-HCCP WRITE，并在成功后由 Runtime CommitPut；确定失败则 Runtime AbortPut，完成状态未决时保留事务供恢复。未配置 BatchPut kernel 时保留 ACL D2H mock 供本地/BatchGet-only 构建使用。 |
| `Shutdown()` | 已支持 | 释放两个独立 transport 及其缓存的 local HBM MR；归还共享 RA/HDC session 引用，进程级 bootstrap 保持驻留。调用方必须先完成所有外部 stream/graph 工作。 |

BatchGet 按固定 token 大小搬运：NPU 侧 `indices`、`offsets`、`miss_count`、
`output_indices`、device AccessTable 与 `output` 都由调用方提供；`status` 和
`queue_state` 由 Operator 内部申请。Python 通过 `DeviceAccessTable` 和 Tensor
调用，不传裸指针/字节数。per-call 模式可通过 `begin_get_output/end_get_output` 管理显式输出绑定。

## Legacy AIV 生命周期

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
  -> success: Runtime.CommitPut; definite failure: Runtime.AbortPut; unresolved: retain transaction

Runtime.Shutdown
  -> BatchGetTransportDestroy / BatchPutTransportDestroy
  -> DirectHccpSessionRelease                 # 归还引用，RA/HDC 保持进程驻留
```

`LocalBufferAcquire/Release` 是算法 Host 库的内部协作接口，Operator 上层只调用
`Runtime.BatchGet/BatchPut`。默认 BatchGet 提交路径不替调用方同步；外部 graph capture 只能使用已预热、
因而命中 transport cache 的 output，并由 Python/框架保证 graph replay 完成后才释放
tensor、AccessTable 和 Runtime。KVAF 不创建、保存或 replay 任何图对象。

## 构建范围

不带 CANN 时只构建接口和 CPU Mock，用于本地 Python 测试。启用
`KVAF_ENABLE_AIV_BATCH_GET=ON` 时构建 `DefaultAccessOperator` 和 AIV BatchGet 算法；
`KVAF_ENABLE_AIV_BATCH_PUT=ON`（默认随前者启用）时额外构建 AIV BatchPut 算法与
Direct-HCCP Runtime adapter。`KVAF_ENABLE_CPU_OFFLOAD` 启用 CPU Pull，
`KVAF_ENABLE_NPU_PULL` 启用独立 NPU raw READ；两者均使用共享的 Direct-HCCP 会话基础。
`KVAF_ENABLE_AICPU_URMA` 启用 AICPU Gather/Put；
`KVAF_ENABLE_AICPU_CPU_PUSH` 进一步启用 Scatter。

图对象由调用方创建、capture、replay 和销毁。执行调度器只保留图使用的资源租约，
不管理图对象；BatchPut 是同步 eager 路径，不支持图模式。

## Direct-HCCP session 与 Transport 边界

BatchGet / BatchPut 的算法实现共享按 `{physical device, hdc type}` 索引的进程级
RA/HDC session。它只拥有 `RaInit`、`rtOpenNetService`、`RaDeinit` 与引用计数，避免
第二个算法重复 `RaInit`（HCCP `328002`）或在失败路径错误释放第一个算法的 RA。

CPU 服务在同一缓存 allocation 内共享 URMA context 与 DDR MR；AICPU 会话共享
进程级动态库/API 环境。各算法仍保留自己的队列、完成状态和执行 workspace，
NPU context 与 HBM MR 也保持独立。共享范围不包含跨算法复用 Jetty、SQ/CQ 或
kernel entity。

- 硬件验收应确认第二个 transport 可以跨过 `RaCtxInit`、CTP、Jetty 创建，并完成
  Get/Put 数据校验；这验证的是两个数据面 transport 并存。
- 后续若共享更多资源，仍需先确认 token 语义、访问权限和队列并发模型兼容；
  当前共享边界详见 [实现与使用](../docs/cache_transport_execution_implementation.md)。

当前 Runtime 始终保留独立 transport，不在 Get/Put 之间销毁、重建或切换链路。
