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

