# KVAF

> **当前执行状态：`HARDWARE_EXECUTION_DISABLED`。** 仓库中的 A5 H2D、
> zero-WQE、AICPU owner 和动态 URMA probe 都只是历史实现或架构候选，不能作为
> 当前真机操作指南。现阶段只允许 CPU/编译测试、已注册 AICPU package 的静态冲突
> 审计，以及 HCCP/HAL 声明面预检；静态结果也不会解锁硬件路径。

当前仓库在原有模块骨架上增加了 AIV/SIMT 直写 URMA SQ/CQ、Host DDR
到 Device HBM（H2D）的数据路径，并提供两个可选控制面：

- HCOMM backend，用于既有实现和对照；
- direct URMA backend，通过 Host+AICPU provider 直接创建
  MR/EID/Jetty/TP/SQ/CQ，不链接 HCOMM，也不依赖 memcache/memfabric。

该路径是完整 KVAF 的最小子集，不包含分布式 Meta、BatchPut、vLLM
适配或图捕获。旧 HCOMM MVP 只作为设计和失败经验记录保留，见
[A5 AIV+SIMT+URMA H2D 历史 MVP](docs/a5_aiv_simt_urma_h2d_mvp.md)。
不链接 HCOMM 的候选实现、能力缺口和当前允许的静态检查见
[A5 direct URMA H2D](docs/a5_direct_urma_h2d.md)。

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
A5 后端默认关闭。两个 backend 都依赖 Ascend 950、CANN、bisheng 和有效
的 Host/Device EID 对；HCOMM backend 额外依赖 HCOMM，direct backend
固定使用 openEuler UMDK
`12917421beb93c1351eb2f124bb2f34612d8bf2d` 及该版本锁定的 libummu
`70de020f776d8def0d8a8941cc7cbc5928d66c10`。UMDK common、完整 public
URMA core、UDMA 和官方 perftest TCP 控制面，以及配对的 UMMU 都从这两棵
源码树编进 Host/Device provider。构建还要求 UDMA、UMMU userspace UAPI 与
当前运行内核的对应 UAPI 逐字节相同，并在编译期禁用动态 provider 扫描。
direct backend 不链接或加载系统 `liburma.so`、`liburma_common.so`、
`liburma-udma.so`、`libummu.so` 或预编译 `liburma_perftest.so`，也不依赖
memcache、memfabric 或 HCOMM。

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

服务器 B 上的旧 A5 源码准备、provider 构建、AICPU package 部署和真机 smoke
流程已从本首页撤下。原因不只是早期的 worker-affinity 缺口：官方 URMA 的 AICPU
调用域、资源语义、AIV 可见地址域、私有队列 ABI 和可重试销毁仍缺少完整证据链。
direct runner 与 legacy HCOMM runner 现在都固定命中
`HARDWARE_EXECUTION_DISABLED`，不能把旧 HCOMM 路径作为替代入口。

当前服务器 B 只执行 [direct URMA 文档 6.7 节](docs/a5_direct_urma_h2d.md#67-docker当前唯一允许的下一步静态检查)
中的两项静态检查。固定 UMDK/libummu commit、runtime UAPI 配对和历史构建流程仍在
该文档中保留，供源码审计和未来恢复构建时参考；它们不是当前运行步骤。

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
