# KVAF Runtime

## 模块定位

KVAF Runtime 是 KVAF 对外提供统一 KV 访问能力的入口，承接来自 vLLM 侧 CacheOS-vLLM Adaptor 或 KVIP 的调用。

Runtime 不负责 Prefix Key、Forward 或 KV 业务流程本身，只负责组织请求、协调内部模块并返回执行结果。

## 主要职责

- 调用 Meta Manager 准备和查询 KV 访问描述；
- 调用 KV Access Operator 执行 BatchGet、BatchPut 等数据搬运；
- 管理 AccessTable 的准备、使用和释放；
- 对上层提供同步、异步以及后续图兼容的统一接口。

## POC 范围

POC 主要支持 Decode 阶段的 KV Offloading，以及 BatchGet 的完整图执行路径。Runtime 当前只组合本卡上的 Meta Manager 和 KV Access Operator，不负责跨卡或跨节点 Meta 协同。

具体接口以 KVAF 开发文档第三单元为准；当前头文件仅提供占位声明。
