# KVAF 公共接口类型

本目录提供模块间共享的基础类型和请求描述，包括：

- `status.h`：统一状态返回类型；
- `types.h`：Event、AccessTableHandle 等公共句柄和基础类型；
- `requests.h`：BatchGet、BatchPut、Delete、Lookup 等请求类型占位。

这些类型只用于当前代码骨架固定接口位置和依赖关系。Key 的最终编码、AccessTable 的字段、Tensor/Buffer 描述以及异步 Event 语义仍需按开发文档继续细化。
