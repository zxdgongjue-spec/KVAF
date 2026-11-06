# KVAF 开发文档

本目录存放 KVAF 的架构与开发设计文档。

- [CPU gather 与 payload READ 分块流水：实现、配置和验证](aiv_gather_chunk_pipeline.md)
- [Gather 分片进度流水与 Python/Host 调用快路径](aiv_gather_shard_pipeline.md)
- [Gather 串行阶段计时、脚本配置和性能对比](aiv_gather_serial_profile.md)
- [Gather 必需开销与 direct 对齐](aiv_gather_direct_alignment.md)
- [Gather CPU 性能采集与等待归因](aiv_gather_cpu_profile.md)

当前文档主要描述 POC 阶段的 Runtime、Meta Manager 和 KV Access Operator，以及它们的接口边界、调用关系和后续扩展方向。
