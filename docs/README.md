# KVAF 开发文档

当前分支的执行框架以以下三篇文档为入口。Get 图模式分别纳入设计契约、接口用法和验证步骤，不单独维护演进文档。

| 文档 | 回答的问题 |
| --- | --- |
| [缓存访问执行框架设计](cache_transport_execution_design.md) | 支持哪些算法；资源由谁持有和共享；Put/Get 驻留、并发、Graph 与错误恢复的边界是什么。 |
| [实现与使用](cache_transport_execution_implementation.md) | 代码如何分层；资源和调用如何流转；如何使用输入/输出绑定、四种 Get 图租约、状态检查与 profiling。 |
| [验证步骤](cache_transport_execution_validation.md) | 如何部署、运行当前性能入口及数据预检，并按相同版本、API 路径和完成口径比较性能；说明已移除的独立测试覆盖。 |

调用方从 main 升级时直接读[接口迁移与最小示例](cache_transport_execution_implementation.md#python-migration)。
vLLM/模型框架适配见[两个 Get 与普通算子混合捕获的示例](cache_transport_execution_implementation.md#get-graph-example)
和[图 runner、层内 adaptor、回放及图缓存淘汰的接入位置](cache_transport_execution_implementation.md#framework-graph-integration)。

设计文档末尾保留的资源共享与上下文驻留草稿不是全量已实现清单；其当前落地范围以正文的状态对照为准。

其他文档按主题补充：

- [CPU Pull 构建与验证](cpu_pull_offload_usage.md)：共享 DDR 写事务、源注册、单 RPC READ、恢复和阶段性能统计。
- [NPU raw READ](hbm_raw_addr_read.md)：独立 HBM 建链、应用保活与同步、图捕获和双机测试；[建链流程](connection_handshake_flows.md)汇总各条路径的注册及握手顺序。
- [原始 POC 架构设计](kvaf_architecture_design.md) 与 [Meta Manager 地址管理](meta_manager_address_management_design.md)：背景、目标和元数据设计；当前执行 API 以三篇主文档及源码为准。
- [AICPU Gather Read](aicpu_gather_read.md) 与 [CPU Push Scatter](aicpu_cpu_push_scatter.md)：算法数据流、性能入口与部署说明。
- [AIV BatchGet 在途控制](batch_get_inflight_control.md)：设备侧窗口和队列预算。
- [AccessOperator 目录](../access_operator/README.md) 与 [Runtime 目录](../runtime/README.md)：模块索引和兼容接口示例。

- [统一 Get 性能测试](batch_get_performance.md) 与 [多卡带宽测试](batch_get_multicard_performance.md)：统一入口、计时口径及 profiling 开关。
- [Gather 实现与配置：各阶段优化、byte-prefix、协作 chunk 与 ABI](aiv_gather_prepared_prefix.md)
- [Gather 性能采集与排障：对照、Host/AIV/CPU 计时与 CSV](aiv_gather_serial_profile.md)

当前文档主要描述 POC 阶段的 Runtime、Meta Manager 和 KV Access Operator，以及它们的接口边界、调用关系和后续扩展方向。
