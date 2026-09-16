# KVAF 开发文档

本目录存放 KVAF 的架构与开发设计文档。

当前文档主要描述 POC 阶段的 Runtime、Meta Manager 和 KV Access Operator，以及它们的接口边界、调用关系和后续扩展方向。

测试记录：

- [BatchGet 双进程连续吞吐测试汇总](batch_get_two_process_throughput_results.md)：13 组配置的结果总表、端点映射、计时口径、对照结论与退出异常。
- [BatchGet 建链读回与 DFX 排查](batch_get_link_diagnostics.md)：两端资源映射、诊断开关、只读快照及运行期硬件计数的采集方法。
