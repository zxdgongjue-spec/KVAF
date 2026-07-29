# `kvaf` Python 包

该目录提供 Python 侧的 `kvaf` 包入口。

构建 Pybind11 扩展后，目标使用形态为：

```python
import kvaf

kvaf.runtime
kvaf.meta_manager
kvaf.access_operator
```

当前三个子模块仅用于固定 Python 命名空间，具体类、函数和 Tensor 转换逻辑待 C++ 接口稳定后补充。vLLM 的 CacheOS-vLLM Adaptor 不在此处实现。
