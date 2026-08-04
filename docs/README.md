# KVAF 开发文档

本目录存放 KVAF 的架构与开发设计文档。

当前文档主要描述 POC 阶段的 Runtime、Meta Manager 和 KV Access Operator，以及它们的接口边界、调用关系和后续扩展方向。

> **执行状态：`HARDWARE_EXECUTION_DISABLED`。** 下列 HCOMM MVP、AICPU owner、
> decode epoch 和 direct URMA H2D 文档中的动态流程均为历史设计或候选架构，
> 不是当前真机命令。当前可执行范围仅限 direct URMA 文档 6.7 节列出的两项
> 静态检查；检查通过也不表示 backend ready。

- [HISTORICAL：A5 HCOMM 控制面 + AIV/SIMT direct SQ/CQ H2D MVP](a5_aiv_simt_urma_h2d_mvp.md)
- [A5 不链接 HCOMM 的 direct URMA H2D：当前门禁与静态检查](a5_direct_urma_h2d.md)
- [HISTORICAL：A5 resource-free AICPU bounded owner probe 与分阶段 owner 设计](a5_aicpu_owner_probe.md)
- [DESIGN ONLY：A5 direct URMA Decode H2D epoch 设计](a5_decode_epoch_h2d.md)
- [A5 HCCP 控制面 + AIV/SIMT 直写数据面候选架构](a5_hccp_control_plane.md)
- [A5 HCCP/libra + 私有 HAL 只读能力审计（不代表 backend ready）](../scripts/preflight_a5_hccp_hal_capabilities.sh)
- [A5 官方 URMA 接口能力边界与分层门禁](a5_official_urma_capability_audit.md)
