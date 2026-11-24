# KVAF 开发文档

本目录存放 KVAF 的架构与开发设计文档。

当前文档主要描述 POC 阶段的 Runtime、Meta Manager 和 KV Access Operator，以及它们的接口边界、调用关系和后续扩展方向。

- [KVAF Python API · NpuRuntime](python_npu_runtime_api.md) 包含构造参数、BatchPut、BatchGet 和端到端调用示例。
- [Host RoCE 读写方案](host_roce_runtime_design.md) 包含传输设计、HBM 注册和验证入口。
