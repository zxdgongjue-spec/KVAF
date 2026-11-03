# Python 包目录

本目录存放 KVAF 的 Python 包入口，用于加载可选的 C++ 原生扩展，并向 Python 侧提供统一的 `kvaf` 命名空间。

当前 Python 包只承担调用入口和模块导出，不承担 KVAF Runtime、Meta Manager 或 KV Access Operator 的业务实现。

## 共享池 CPU Pull

启用 `KVAF_ENABLE_CPU_OFFLOAD` 后，`NpuRuntime(enable_cpu_offload=True)` 提供同步的本卡 `batch_cpu_pull()`，以及跨卡 `attach_cpu_source / prepare_put / submit_cpu_pull / wait_cpu_pull / commit_put / abort_put` 显式事务接口。
`CpuPullHbmSource` 在生产卡注册 HBM，使用 `ready()`、`release_batch(ready)`、`cancel_batch(ready)` 管理批次。
NPU 直读由读取端 `runtime.listen_hbm()` 监听，生产端 `NpuPullHbmSource.register(tensors, receiver=receiver)` 一次完成注册建链；不需要转发 export_ticket 或调用 attach_hbm。运行时无需 AcquireRaw，已发布批次只能由读取端 `cancel_raw_addr(ready)` 安全取消。使用 `NpuPullHbmSource` 和只接受三个连续 NPU int64 Tensor 的 `batch_get_raw_addr()`。
旧类名、旧重载及独立 OffloadRuntime 已删除。完整接口见 [Connector 设计](../docs/connector_pull_api_design.md)。
