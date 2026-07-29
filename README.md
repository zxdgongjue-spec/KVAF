# KVAF 初期代码仓

当前目录只提供 KVAF 初期开发阶段的代码骨架，用于先固定模块边界、公共头文件位置和构建依赖关系。

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

当前实现只有头文件占位，不包含业务逻辑，也不会产出有实际执行能力的库。开启 Python binding 时，唯一的 `.cpp` 文件仅用于注册 Python 模块和三个子模块名称，不承载 KVAF 业务实现。

## 构建

以下命令需要在本目录（`kvaf/`）下执行。

```bash
cmake -S . -B build
cmake --build build
```

上述默认构建当前可以执行，但只完成 CMake 配置和占位 target 的构建检查。由于三个模块目前均为头文件和 `INTERFACE` target，不会生成具有实际业务能力的 KVAF 库，也不能执行 BatchGet、BatchPut 或 Meta 操作。

如需编译 Python 注册模块，需要环境中提供 `pybind11`，并显式开启：

```bash
cmake -S . -B build -DKVAF_BUILD_PYTHON=ON
cmake --build build
```

当前 Python 注册模块可以编译，但 Python 运行时暂不作为当前环境下的可用能力保证。编译时使用的 Python 解释器、Python 版本、CPU 架构必须和运行时保持一致；当前仓库尚未提供统一的 Python 环境配置，也尚未完成这部分适配。

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

即使 Python 扩展成功加载，目前也只有 `build_stage()` 和三个空的命名空间可用，尚不能调用 KVAF 的实际接口。

具体可调用接口和实现内容后续分别在三个模块内补充。
