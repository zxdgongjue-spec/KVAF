# Python 包目录

本目录存放 KVAF 的 Python 包入口，用于加载可选的 C++ 原生扩展，并向 Python 侧提供统一的 `kvaf` 命名空间。

当前 Python 包只承担调用入口和模块导出，不承担 KVAF Runtime、Meta Manager 或 KV Access Operator 的业务实现。

## 共享池 CPU Pull

启用 `KVAF_ENABLE_CPU_OFFLOAD` 后，`NpuRuntime(enable_cpu_offload=True)` 提供
`offload()`、`offload_region()`、`offload_from()` 和 `resolve_offload()`。
`HbmSource` 在生产卡注册固定 HBM，支持 `ready()` / `release_batch()` 复用。
完整时序、地址规则和双卡测试见 [共享 DDR 与跨卡 CPU Pull](../docs/shared_ddr_remote_offload.md)。
