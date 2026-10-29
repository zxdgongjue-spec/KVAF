# AIV BatchGet 第一阶段实现与验证

## 范围

在 `codex/aiv-urma-scatter-gather` 实现无流水顺序路径，默认仍为 `direct_read`。
没有接入第二阶段的 chunk 消费、双缓冲或第三阶段的软件在途预算。
总体设计见 [三阶段设计](aiv_batch_get_scatter_gather_design.md)。

| 模式 | 顺序依赖 |
| --- | --- |
| `cpu_push_scatter` | AIV 生成源地址清单并发布 HBM notify → CPU READ 清单 → 四路 Jetty WRITE 整批 payload → done → AIV 单 UB 缓冲 scatter → 本地写出完成 |
| `gather_read` | AIV WRITE 清单 → WRITE command → WRITE notify → CPU 完成对应 lane 的 gather → 最终 ACK → AIV 多 SGE READ 到 output → CQ 完成 |

Scatter 只消费本轮整批 done，忽略参考分支 CPU transport 仍会发布的 chunk ready。
Gather 的 `pipeline_chunk_bytes=0`，不把 CPU 的 `~sequence` started ACK 当成完成。
多核 gather 每个核拥有独立 lane，各 lane 按上述依赖顺序执行。
AIV 不调用 UMDK；scatter 使用注册 HBM mailbox，gather 使用手写 USER_CTL WQE。

Gather 的控制与数据队列分开，每核固定两条 Jetty。可配置 1～6 个本地 SGE，
每条 READ 的总长度最多 64 KiB，按 64 B WQEBB 处理变长 WQE 和 SQ 回绕。
阶段内部批量提交，只在物理 SQ/WR 容量不足或阶段结束时 drain；
`batch_get_max_inflight_total` 仅对原 `direct_read` 生效。

## 代码入口

- `access_operator/demo/aiv_batch_get/kernel/execute_sequential_batch_get_aiv.asc`：
  两种模式、行校验、notify/ACK、CQ、精确字节搬运及状态回写。
- `access_operator/demo/aiv_batch_get/host/sequential_batch_get.cpp`：
  workspace、注册内存、CPU 会话、序号和异步状态回收。
- `access_operator/demo/aiv_batch_get/include/sequential_batch_get_abi.h`：
  独立的 V1 参数与状态 ABI；kernel symbol 为 `KvafSequentialBatchGetAivKernelV1`。
- `runtime/src/cpu_push_get_*`、`runtime/src/cpu_gather_executor.cc`：
  复用 `aicpu-urma` 的 CPU 数据面，未引入 AICPU kernel 或设备端 liburma。
- `runtime/src/peer_link.cc`、`runtime/src/cpu_runtime.cc`：
  增加 enable/open/start/stop；新增 `StopGatherLanes` 在释放 DDR 前 join worker。
- `runtime/src/npu_runtime.cc`、`bindings/src/npu_python_runtime.cc`：配置入口及
  gather 的 DDR 预留区，预留区不参与 KV slot 分配。

两端需一起重编译。初始化通过新增控制命令的响应确认能力；不支持的 CPU
服务会初始化失败。Host 和 AIV object 也必须来自同一版本。

构建产出两个独立二进制：`kvaf_execute_batch_get_kernels.o`（direct_read）和
`kvaf_sequential_batch_get_kernels.o`（scatter/gather），均复制到 `build-npu/bin/`。
AscendC 会在每个编译单元中生成 `g_tilingKey` 等全局符号，因此不合并两个
临时 object。顺序模式收到标准 `kvaf_execute_batch_get_kernels.o` 路径时，
自动加载同目录下的 `kvaf_sequential_batch_get_kernels.o`；也可直接传入顺序
二进制路径。部署时需要一起复制两个文件。

## 使用约束

```python
runtime = kvaf.NpuRuntime(
    get_kernel_path,
    batch_get_token_bytes=656,
    batch_get_tokens_per_block=128,
    batch_put_kernel_object_path=put_kernel_path,
    npu_core_count=1,                 # 首次上机先用单核，再验证 3/8 核
    aiv_batch_get_mode="gather_read", # 或 "cpu_push_scatter"
    aiv_gather_sgl_size=6,            # Gather: 1..6
)
runtime.link_to(cpu_service_address)
runtime.initialize(ddr_bytes, slot_size)
# 原有 DeviceAccessTable / addresses+valid 两种 batch_get 重载均可使用。
```

- 支持 1～8 核、单 EID、eager；首版拒绝 graph capture 和启动带宽调优。
- 请求容量为 1～65536，`miss_count=0` 是合法空批次。仅有效前缀参与搬运，
  支持重复源、随机且不重复的输出行。输出行重复不受支持。
- 固定 token 长度，当前初始化上限为 65535 B，保证 `65536 * token_bytes`
  可由共享协议的 uint32 表示。offset 为非负字节偏移，不能越过源 block。
- 每个 Runtime 复用一套 workspace，下一次 Get/Put 先 drain 上一批。
  `synchronize=False` 仅代表入队成功，设备错误在下一次 Get/Put 或 Shutdown
  回收状态时返回；调用方仍须保持 Tensor 和 stream 有效直到执行完成。
- 错误输出可能部分写入，不应消费。执行失败后拒绝继续提交，需 shutdown
  并重建 Runtime。队列或 CPU DMA 无法确认停止时保留注册资源，允许重试关闭。
- 轮询有次数上限，loader 设置 180 秒 kernel 超时；轮询次数不是毫秒超时，
  不构成性能或端到端超时承诺。
- Scatter HBM 约为 `65536 * token_bytes + 512 KiB + 控制区`；Gather 使用
  同量级 CPU DDR staging，另有每 lane 描述符、控制区及小型 HBM workspace。

## 构建与服务器回归

CPU 端使用匹配部署环境的 UMDK：

```bash
cmake -S . -B build-cpu -DKVAF_BUILD_REMOTE_CPU_SERVICE=ON -DKVAF_URMA_ROOT=/path/to/umdk
cmake --build build-cpu -j
./build-cpu/runtime/kvaf_cpu_service --listen 0.0.0.0 --port 28999 --gather-threads 8
```

NPU 端使用支持 A5 `dav-c310-vec` 的 CANN 环境，沿用项目原有环境初始化：

```bash
cmake -S . -B build-npu -DKVAF_ENABLE_AIV_BATCH_GET=ON -DKVAF_ENABLE_AIV_BATCH_PUT=ON \
  -DKVAF_BUILD_PYTHON=ON -DASCEND_CANN_PACKAGE_PATH=/path/to/cann
cmake --build build-npu -j
export PYTHONPATH="$PWD/build-npu/python"
export KVAF_RUN_NPU_TESTS=1
export KVAF_CPU_SERVICE_ADDR=CPU_IP:28999
export KVAF_BATCH_GET_KERNEL="$PWD/build-npu/access_operator/demo/aiv_batch_get/kernel/kvaf_execute_batch_get_kernels.o"
export KVAF_BATCH_PUT_KERNEL=/path/to/built/kvaf_execute_batch_put_kernels.o
KVAF_NPU_CORE_COUNT=1 python tests/python/test_aiv_sequential_batch_get.py
KVAF_NPU_CORE_COUNT=3 python tests/python/test_aiv_sequential_batch_get.py
KVAF_NPU_CORE_COUNT=8 python tests/python/test_aiv_sequential_batch_get.py
```

脚本默认分别测试两种模式，使用 8193 请求容量，覆盖空批次、尾批、重复调用、
随机输出、重复源、guard、A/B/A 输出、两种 Tensor 重载以及跨 stream 顺序复用。
可用 `KVAF_NPU_TOKEN_BYTES=13`、`KVAF_NPU_REQUESTS=65536`、
`KVAF_AIV_GATHER_SGL_SIZE=1` 增加边界组合。
CPU `--gather-threads` 至少等于 gather 的 AIV 核数。

## 当前验证状态

本地无 CANN/NPU，已完成：

- CMake 普通 C++ 构建和现有回归。
- 生产 `.asc` 的 CPU 模型执行：解码 SQ/CQ，调用真实 CPU gather executor，
  比较全部 output/guard。覆盖 1/3/8 核、1/13/656/4096/65535 B token、
  1～6 SGE、65536 请求、SQ/CQ 回绕、旧序号、非法行、超时及控制/数据 CQ 错误。
- 真实 Host owner 的延迟 ACL 模型：串行复用、异步错误上报、capture 拒绝、
  部分初始化和关闭失败时资源保留/重试。Host 集成源码使用 ACL 声明替身编译。
- 新增内核模型和 Host 生命周期测试通过 AddressSanitizer / UndefinedBehaviorSanitizer。
- CPU 新源码使用 UMDK 24.12 头文件通过语法检查；macOS 检查时仅补充 Linux
  的 `MADV_HUGEPAGE` 常量，未链接或执行 CPU 硬件服务。

这些检查不替代 CANN 编译和双机验收。尚待验证多 SGE 硬件布局、非对齐 MTE3
写出、CPU/NPU 缓存可见性、跨 Jetty done 保序及真实超时/断链回收。
硬件回归未启用时返回 skip，不能视为硬件验收通过。
