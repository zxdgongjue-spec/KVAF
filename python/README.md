# Python 包目录

本目录存放 KVAF 的 Python 包入口，用于加载可选的 C++ 原生扩展，并向 Python 侧提供统一的 `kvaf` 命名空间。

当前 Python 包只承担调用入口和模块导出，不承担 KVAF Runtime、Meta Manager 或 KV Access Operator 的业务实现。
