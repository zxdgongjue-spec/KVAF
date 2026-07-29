# Meta Manager

## 模块定位

Meta Manager 是 KVAF 内部的元数据管理模块，负责维护 KV Block Key 与远端 DDR 地址之间的映射关系，并为 Runtime 提供地址准备、查询、提交和归还能力。

## 主要职责

- 注册和管理远端 DDR 区域；
- 为 BatchPut 预留地址，并在数据搬运成功后提交映射；
- 根据 KV Block Key 查询 UBA、Length 和 Valid 等访问描述；
- 在 AccessTable 使用期间保护已有地址映射的稳定性；
- 删除 Key 映射，并将地址区间归还内部空闲地址池供后续复用。

## 边界

Meta Manager 不负责 KV Payload 的实际搬运，也不负责 Prefix Key 到 KV Block Key 的业务展开。POC 阶段以单卡、单 KVAF 实例和远端 DDR Offloading 为管理边界。

远端 DDR 区域的生命周期与 KVAF 实例一致；单个 Key 被删除后只归还区域内的地址区间，不释放整个远端 DDR 区域。

具体接口以 KVAF 开发文档第四单元为准；当前头文件仅提供占位声明。
