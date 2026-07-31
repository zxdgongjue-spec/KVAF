# KV Access Operator

## 模块定位

KV Access Operator 是 KVAF 内部负责实际 KV 数据搬运的完整执行单元。它消费 Runtime 已准备好的访问描述和数据 Buffer，不维护 Key 到地址的映射。

## 主要职责

- 执行 BatchGet 和 BatchPut；
- 支持连续和离散 KV Block 的批量访问；
- 根据 AccessTable、Index、Mask 和输出 Buffer 组织 BatchGet；
- 根据远端访问描述组织 BatchPut；
- 在内部选择硬件和执行路径，并生成具体搬运计划。

Operator 内部按照 Selector、Executor、Template 组织（当前不需要考虑）

## POC 范围

- BatchGet：支持图模式下的 AIV 高性能执行路径；
- BatchPut：支持 AICPU 执行路径；
- 采用整批成功或整批失败语义；
- 仅需要提供上述两个能力，暂不考虑扩展性

## 当前 A5 最小实现

当前代码先实现 BatchGet 的一个受限子集，并保留两个可选资源控制后端：

- 设备侧固定 ABI：AccessTable、Index、Mask、连续 HBM 输出和 Completion；
- 32 个 SIMT lane 并行查表并生成 64B 定长 H2D 描述符；
- legacy HCOMM backend 通过 HCOMM 创建并交换
  Endpoint/MR/EID/TP/Jetty/SQ/CQ；
- direct URMA backend 通过 Host+AICPU provider 直接使用
  `liburma`/UDMA 创建这些资源，不链接 HCOMM，也不依赖 memcache 或
  memfabric；
- 单个 AIV 作为唯一 URMA SQ/CQ owner，直接填 READ WQE、敲 doorbell 并
  轮询带 phase/owner 的 CQE；
- Host DDR 显式执行 `halHostRegister(HOST_MEM_MAP_DEV)`，随后注册、导出并在
  Device endpoint 导入；
- Host/AICPU 完成 direct URMA 建链，AIV 独占数据面 SQ/CQ；并提供
  多描述符、乱序索引和 mask 的 H2D 硬件冒烟验证。

HCOMM 对照路径见
[A5 AIV+SIMT+URMA H2D MVP](../docs/a5_aiv_simt_urma_h2d_mvp.md)；
不链接 HCOMM 的路径和命令见
[A5 direct URMA H2D](../docs/a5_direct_urma_h2d.md)。
