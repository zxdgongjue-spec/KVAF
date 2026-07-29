# Python Bindings

本目录是 KVAF C++ 核心与 Python 之间的可选绑定层，使用 Pybind11 构建 `_kvaf` 原生扩展。

当前 Binding 只用于验证以下链路：

```text
C++ 编译 → Python 原生扩展 → Python 导入
```

它暂不实现 KVAF 业务逻辑，也不负责 vLLM 适配。后续只在 Runtime、Meta Manager 和 KV Access Operator 的 C++ 接口稳定后，将需要暴露的类和方法注册到 Python。

CacheOS-vLLM Adaptor 属于 vLLM 侧实现，不放在本目录。
