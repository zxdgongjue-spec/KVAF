# KVAF 开发文档

本目录存放 KVAF 的架构与开发设计文档。

当前文档主要描述 POC 阶段的 Runtime、Meta Manager 和 KV Access Operator，以及它们的接口边界、调用关系和后续扩展方向。

- [建链与数据传输交互](connection_handshake_flows.md)：opt 当前的 DDR、CPU Pull 单 RPC、NPU Pull 异步／图模式、应用通知及资源关闭时序。
- [NPU HBM 裸地址读取](hbm_raw_addr_read.md)：多 MR、异步提交、图回放、诊断及跨机验收。

- [CPU Pull / Offload 设计](cpu_pull_offload_design.md)：通过 PreparePut、CPU 主动 READ 和 CommitPut 实现独立的 HBM 到 DDR 卸载。
- [CPU Pull / Offload 构建与运行](cpu_pull_offload_usage.md)：统一 NpuRuntime、统一 batch_cpu_pull、跨卡验收与安全清理。

- [共享 DDR 与跨卡 CPU Pull](shared_ddr_remote_offload.md)：A 统一管理 Meta 和 DDR，CPU Pull B 注册的 HBM；接口、生命周期、双卡验收及性能实测记录。
- [面向应用 Connector 的拉取接口设计](connector_pull_api_design.md)：已实现的 Connector 接口；公开 PreparePut/CPU Pull 事务接口，明确两类 HBM 源、裸地址读取与应用对接边界。
