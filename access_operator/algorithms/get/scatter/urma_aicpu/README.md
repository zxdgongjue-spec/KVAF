# URMA AICPU Scatter Get

`host/cpu_push_host.cc` 管理 CPU push 通道、HBM staging 和请求状态；`kernel/cpu_push_scatter.cc` 执行本地 scatter。`include/cpu_push_transport.h` 声明此算法的资源接口。

共同 ABI、Host loader和设备导出入口保留在 [Get 公共目录](../../common/aicpu/README.md)。本算法仍由 `KVAF_ENABLE_AICPU_CPU_PUSH` 控制，构建目标与安装包保持不变。

`python/benchmark_cpu_push_scatter.py` 调用公共性能测试负载。完整说明见 [Scatter 使用说明](../../../../../docs/aicpu_cpu_push_scatter.md)。
