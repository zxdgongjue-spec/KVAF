# KVAF 初期代码仓

当前目录包含 KVAF 同步 Runtime POC、MetaManager 地址管理实现，以及可选的
A5 AIV BatchGet demo/adapter。

## 模块

```text
kvaf/
├── include/kvaf/          # 模块间共享的占位类型和请求描述
├── runtime/               # KVAF Runtime
├── meta_manager/          # Meta Manager
├── access_operator/       # KV Access Operator
├── bindings/              # 可选的 Python/Pybind 注册入口
└── python/kvaf/           # Python 包入口
```

`CacheOS-vLLM Adaptor` 不属于当前代码仓；它后续在 vLLM 侧实现。HComm 是外部依赖，不在这里单独建立模块。SIMT 作为 AIV backend 的编程范式处理，也不单独建立模块。

## 构建关系

三个模块分别暴露独立的 CMake target：

```text
kvaf_meta_manager
kvaf_access_operator
kvaf_runtime
```

Runtime target 依赖另外两个 target，因此后续可以由 Runtime 统一组合调用；Meta Manager 和 KV Access Operator 仍可单独被构建和测试。

默认构建会生成 `kvaf_meta_manager` 和 `kvaf_runtime` 静态库。Runtime 支持
内部 DDR 初始化、同步 BatchPut、Prepare/Release AccessTable、同步 BatchGet
和 BatchFree；真实数据搬运由注入的 `AccessOperator` 实现提供。

## 构建

以下命令需要在本目录（`kvaf/`）下执行。

```bash
cmake -S . -B build
cmake --build build
```

Python 定义的功能测试可用以下命令运行：

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

详细 Runtime 生命周期、测试边界和 A5/CANN 集成命令见
[`runtime/README.md`](runtime/README.md)。

如需编译 Python 注册模块，需要环境中提供 `pybind11`，并显式开启：

```bash
cmake -S . -B build -DKVAF_BUILD_PYTHON=ON
cmake --build build
```

当前 Python 注册模块仍只提供命名空间；Python 测试通过独立的测试 driver
执行 C++ Runtime，因此不要求 pybind11。

因此，下面的命令当前仅作为目标使用方式示例，不保证在当前环境直接成功：

```bash
PYTHONPATH=build/python python3 -c 'import kvaf; print(kvaf.build_stage())'
```

Python 侧的目标形态为：

```python
import kvaf

kvaf.build_stage()
kvaf.runtime
kvaf.meta_manager
kvaf.access_operator
```

即使 Python 扩展成功加载，目前也只有 `build_stage()` 和三个空的命名空间；
正式产品 Python binding 留待接口进一步稳定后补充。
