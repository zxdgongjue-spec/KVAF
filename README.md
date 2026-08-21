# KVAF 初期代码仓

当前目录包含 KVAF 同步 Runtime POC、MetaManager 地址管理实现，以及可选的
A5 AIV BatchGet/BatchPut demo/adapter。

## 模块

```text
kvaf/
├── common/include/        # 模块间共享的类型、请求与 Status
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

## 构建、部署与测试

以下命令均在本目录（`kvaf/`）执行。统一入口是 `build.sh`，它使用独立的
`build-cpu/` 与 `build-npu/`，不会混用两类 CMake 配置。

### CPU 服务机

CPU 服务机需要平台已经提供 UB 驱动、`/dev/uburma` 和可工作的 `liburma`。它不需要
CANN、HCCP、ACL 或 NPU 驱动。`build.sh cpu` 会自动拉取固定版本的 UMDK 源码作为头文件
依赖；若系统没有 `liburma.so`，会在本地编译该 UMDK 的 `urma` target，不执行全局安装。

```bash
./build.sh cpu

./build-cpu/runtime/kvaf_cpu_service \
  --listen 0.0.0.0 --port 28999
```

可通过 `KVAF_UMDK_ROOT=/path/to/umdk` 或
`KVAF_URMA_LIBRARY=/path/to/liburma.so ./build.sh cpu` 覆盖自动发现。服务启动后应先看到
`[KVAF_CPU] CANDIDATE_DISCOVERED` 和 `LISTEN`；不需要手工传入 EID。`--ub-device` 与
`--eid-index` 仅用于诊断固定候选。

### NPU 服务机

NPU 服务机需要已安装并已生效的 CANN、驱动和 `torch_npu` 环境。构建脚本会自动为所选
`PYTHON` 安装缺失的 `pybind11`，但不会安装厂商 CANN/驱动包。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh  # 按实际 CANN 路径调整
export ASCEND_CANN_PACKAGE_PATH=/path/to/cann

./build.sh npu
```

若 CANN 路径未通过环境变量提供，脚本会尝试常见的 Ascend 安装路径；找不到时会在配置前
退出并提示设置 `ASCEND_CANN_PACKAGE_PATH`。

### 机内或跨机端到端验证

先在 CPU 机启动服务，再在 NPU 机执行。NPU 只需要 CPU 服务的 `IP:Port`；EID、UB device
与 CLOS / pair-info fallback 都由 Runtime 自动发现和真实 RMA probe 选择。

```bash
export KVAF_CPU_SERVICE_ADDR=141.61.92.71:28999
export DEVICE_ID=0

tests/run_remote_cpu_e2e.sh "$PWD/build-npu"
```

测试通过时，NPU 侧会完成 Runtime 初始化、AIV BatchPut/BatchGet 正确性校验和 teardown；
CPU 侧会出现 `REGION_ALLOCATED`、`CHANNEL_CREATED` 与 `REGION_RELEASE`。数据面是 UB/RMA，
TCP 只用于 region、token 与 Jetty 控制信息交换。

Runtime API 性能测试使用相同的已启动 CPU 服务：

```bash
KVAF_CPU_SERVICE_ADDR=141.61.92.71:28999 DEVICE_ID=0 \
  tests/run_remote_cpu_runtime_perf.sh "$PWD/build-npu"
```

该脚本在开始采样前执行 BatchPut → BatchGet 正确性 preflight；只有该阶段通过后的
`PERF_RESULT` 才是有效性能结果。可用 `KVAF_REMOTE_CPU_PERF_TIMEOUT` 调整总超时。

### 本地无硬件检查

本地 Mock 不验证 UMDK 或 AIV 数据面，但可验证 Runtime/MetaManager/Python 基础接口：

```bash
cmake -S . -B build-local -DBUILD_TESTING=ON -DKVAF_BUILD_PYTHON=ON
cmake --build build-local --parallel
ctest --test-dir build-local --output-on-failure
```

详细 Runtime 生命周期、buffer 所有权与 AIV 性能口径见
[`runtime/README.md`](runtime/README.md)。
