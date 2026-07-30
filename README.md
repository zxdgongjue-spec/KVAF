# KVAF

当前仓库在原有模块骨架上增加了第一条可执行数据路径：
**AIV + SIMT + HCOMM/URMA + Host DDR 到 Device HBM（H2D）**。

该路径是完整 KVAF 的最小子集，不包含分布式 Meta、BatchPut、vLLM
适配或图捕获。设计边界、真实完成状态以及 PC A / 服务器 B / 远端仓 C
的操作命令见
[A5 AIV+SIMT+URMA H2D MVP](docs/a5_aiv_simt_urma_h2d_mvp.md)。

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

`kvaf_access_operator` 当前包含固定容量的设备 ABI、CPU 参考规划器和单元测试。
A5 后端默认关闭；它依赖 Ascend 950、CANN、bisheng、HCOMM 和有效的
Host/Device EID 对。

## 构建

以下命令需要在本目录（`kvaf/`）下执行。

```bash
cmake -S . -B build
cmake --build build
```

默认构建不依赖 CANN，并会构建 CPU 参考规划器和测试：

```bash
ctest --test-dir build --output-on-failure
```

服务器 B 上构建 A5 数据路径：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/build_a5_aiv_urma.sh
```

构建成功后，正向硬件冒烟会从驱动自动解析当前物理设备的 Host/Device
EID pair，不依赖其他代码仓：

```bash
unset KVAF_HOST_EID KVAF_DEVICE_EID KVAF_PAIR_CHANNEL
export KVAF_LOGICAL_DEVICE_ID=0
bash scripts/run_a5_aiv_urma_h2d_smoke.sh
```

多个 active driver pair 会被拒绝；根据错误中列出的 channel 设置
`KVAF_PAIR_CHANNEL` 后再运行，不能默认选择 channel 0。

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
