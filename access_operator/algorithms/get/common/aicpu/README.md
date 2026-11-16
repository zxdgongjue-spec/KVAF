# AICPU Get 公共入口

本目录保留 Gather 和 Scatter 现有的共同 ABI、Host loader、设备导出入口和 Python 验证负载。目录迁移不改变导出符号、Host library target 或设备安装包。

- `include/batch_get_abi.h`：现有 Gather/Scatter 参数与状态布局。
- `include/kvaf_aicpu_batch_get.h`：现有 Host API 声明。
- `host/execute_batch_get_loader.cc`：按原入口提交 Gather/Scatter。
- `kernel/batch_get.cc`：导出原设备入口。
- `python/benchmark_gather_read.py`：两种算法共用的性能测试实现。
- `python/test_batch_get.py`：两种算法共用的数据与图回放验证。

算法实现分别位于 [Gather](../../gather/urma_aicpu/README.md) 和 [Scatter](../../scatter/urma_aicpu/README.md)。通信基础位于 `access_operator/transports/aicpu_urma`。

`CMakeLists.txt` 汇集各算法源文件，继续构建 `kvaf_aicpu_batch_get_host`，并向 AICPU 设备包导出源文件列表；Scatter 实现仍由 `KVAF_ENABLE_AICPU_CPU_PUSH` 控制。

完整构建和测试见 [Gather 使用说明](../../../../../docs/aicpu_gather_read.md) 与 [Scatter 使用说明](../../../../../docs/aicpu_cpu_push_scatter.md)。
