# KVAF

当前仓库在原有模块骨架上增加了 AIV/SIMT 直写 URMA SQ/CQ、Host DDR
到 Device HBM（H2D）的数据路径，并提供两个可选控制面：

- HCOMM backend，用于既有实现和对照；
- direct URMA backend，通过 Host+AICPU provider 直接创建
  MR/EID/Jetty/TP/SQ/CQ，不链接 HCOMM，也不依赖 memcache/memfabric。

该路径是完整 KVAF 的最小子集，不包含分布式 Meta、BatchPut、vLLM
适配或图捕获。设计边界、真实完成状态以及 PC A / 服务器 B / 远端仓 C
的操作命令见
[A5 AIV+SIMT+URMA H2D MVP](docs/a5_aiv_simt_urma_h2d_mvp.md)。
不链接 HCOMM 的实现、依赖边界和服务器操作见
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

服务器 B 上构建 A5 数据路径：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/build_a5_aiv_urma.sh
```

构建不链接 HCOMM 的 direct URMA 路径。先在服务器 B 宿主机准备两棵固定、
无本地修改的源码树：

```bash
echo "EXECUTION_ENV=SERVER_B_HOST"

test -d /data/c00982601/umdk/.git || \
  git clone https://gitcode.com/openeuler/umdk.git \
    /data/c00982601/umdk
git -C /data/c00982601/umdk fetch origin
git -C /data/c00982601/umdk switch --detach \
  12917421beb93c1351eb2f124bb2f34612d8bf2d

test -d /data/c00982601/libummu/.git || \
  git clone https://gitcode.com/openeuler/libummu.git \
    /data/c00982601/libummu
git -C /data/c00982601/libummu fetch origin
git -C /data/c00982601/libummu switch --detach \
  70de020f776d8def0d8a8941cc7cbc5928d66c10

test -z "$(git -C /data/c00982601/umdk status --porcelain)"
test -z "$(git -C /data/c00982601/libummu status --porcelain)"
```

然后在 `vllm-ascend-cs-kvaf` Docker 中构建。两份 runtime header 必须来自
`uname -r` 对应的 kernel-devel/kernel source；若不在脚本的标准自动探测路径，
显式设置为 Docker 内可读的路径：

```bash
echo "EXECUTION_ENV=VLLM_ASCEND_CS_KVAF_DOCKER"
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export KVAF_UMDK_ROOT=/data/c00982601/umdk
export KVAF_LIBUMMU_ROOT=/data/c00982601/libummu
export KVAF_RUNTIME_UDMA_UAPI_HEADER=/data/c00982601/kernel-uapi/$(uname -r)/udma_abi.h
export KVAF_RUNTIME_UMMU_UAPI_HEADER=/data/c00982601/kernel-uapi/$(uname -r)/ummu_core.h

bash scripts/build_a5_direct_urma_provider.sh
bash scripts/package_a5_direct_urma_aicpu.sh
bash scripts/build_a5_direct_urma.sh
```

provider 构建产出 `kvaf_udma_uapi_probe`、
`kvaf_udma_private_abi_probe`、`kvaf_ummu_uapi_probe` 和 v3 build
manifest/success stamp。任一源码树不干净、commit 配对错误、runtime UAPI
缺失或不完全匹配，构建都会 fail closed。

构建成功后，正向硬件冒烟会从驱动自动解析当前物理设备的 Host/Device
EID pair，不依赖其他代码仓；随后按 EID 精确匹配 Host sysfs 与 Device
`hccn_tool` inventory。可以显式给出两端 UDMA device name/EID index，但 runner
仍会校验，不能从 NPU 号或 `pair_info` 猜测。当前 runner 还会在任何硬件访问前
因 AICPU 多次 KFC 调用缺少正式 worker-affinity 契约而 fail closed；下列命令目前
用于验证该门禁，不会创建 URMA/UMMU 资源：

```bash
unset KVAF_HOST_EID KVAF_DEVICE_EID KVAF_PAIR_CHANNEL
export KVAF_LOGICAL_DEVICE_ID=0
export KVAF_PHYSICAL_DEVICE_ID=0
bash scripts/run_a5_direct_urma_h2d_smoke.sh
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
