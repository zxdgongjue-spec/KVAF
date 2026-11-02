# AIV BatchGet 第一阶段实现与验证

## 数据通路

当前实现无流水的整批顺序路径，默认模式仍为 `direct_read`。总体设计见
[三阶段设计](aiv_batch_get_scatter_gather_design.md)。

| 模式 | 执行顺序 |
| --- | --- |
| `cpu_push_scatter` | AIV 生成源地址清单并发布 notify → CPU READ 清单 → 四路 Jetty WRITE payload → done → AIV scatter 写目标 HBM |
| `gather_read` | AIV WRITE 清单、command、notify → CPU gather → 最终 ACK → AIV 多 SGE READ 到 output → CQ 完成 |

Scatter 只消费本轮整批 done，不消费 chunk ready。Gather 的
`pipeline_chunk_bytes=0`，不把 `~sequence` started ACK 当成完成。
AIV 不调用 UMDK；两端通过注册内存和现有 CPU 控制协议通信。

Scatter 当前使用两个 32 KiB UB bank，顺序等待 CPU 整批 done 后执行 GM→UB→GM。
精确长度的 `DataCopyPad<uint8_t>` 不从 packed staging 多读 padding，复用 bank 前等待写出完成。
清单可运行时选择多核 SIMT 或 UB 构建，具体见 [性能测试说明](aiv_scatter_performance.md)。
包含 SIMT VF 的当前二进制由 loader 设置 `dynUBufSize=65536` 预留共享 UB；
旧纯 SIMD loader 的零动态 UB 配置不适用于这个混合二进制。
GM→UB 普通模式恢复、MTE2/MTE3 事件同步及退出屏障保留。

排障用的九种提前返回探针、逐字节 UB 比较、诊断版本、地址/指纹日志及采集脚本
已移除。正常 kernel 始终完成完整 scatter，正确性由端到端测试回读全部输出验收。
Host 保留每核 magic、sequence、result、phase 检查，以及 ACL 错误的阶段和序号。

## 独立二进制与类型声明

以下路径相对于 `access_operator/demo/aiv_batch_get/`：

| 层次 | Scatter | Gather |
| --- | --- | --- |
| Host owner | `host/scatter_batch_get.cpp` | `host/gather_batch_get.cpp` |
| Loader | `host/execute_scatter_batch_get_loader.cpp` | `host/execute_gather_batch_get_loader.cpp` |
| 参数 ABI | `include/scatter_batch_get_abi.h`，176 B | `include/gather_batch_get_abi.h`，216 B |
| AIV 源码 | `kernel/execute_scatter_batch_get_aiv.asc` | `kernel/execute_gather_batch_get_aiv.asc` |
| Symbol | `KvafScatterBatchGetAivKernelV1` | `KvafGatherBatchGetAivKernelV1` |
| 二进制 | `kvaf_scatter_batch_get_kernels.o` | `kvaf_gather_batch_get_kernels.o` |

Scatter 参数末尾的保留字段默认是 0，可通过版本化 tag 开启性能计时；旧探针 ID
仍会被 kernel 拒绝。176 B 参数和 64 B 业务状态布局不变，计时使用独立的每核
128 B 区域。部署时同步更新 Host/kernel，采集 CPU 通信计时还需更新 CPU 服务。
控制协议结构大小不变，计时报告按请求 opt in。

原 `direct_read` 继续使用 `kvaf_execute_batch_get_kernels.o`。三个模块独立编译，
避免重复链接 SDK 全局符号。顺序模式接收标准 direct/旧 sequential 路径时，
按模式选择同目录的新 `.o`；显式自定义路径按原样加载。缺失目标模块或 symbol
直接失败，不回退到其他模式。

Scatter 必须保留函数级 AIV 类型声明，方式见
[CANN 独立 AIV 下发示例](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/commlib/commopdev/docs/en/comm_op_dev_guide/aiv_comm_op_dev/op_dispatch.md)。
`.ascend.meta.KvafScatterBatchGetAivKernelV1` 中应包含
`tag=1, length=4, value=3 (K_TYPE_AIV)`，对应小端字节 `01000400 03000000`。
仅有 section 名不能证明核类型存在。检查实际产物：

```bash
python tests/python/aiv_kernel_metadata.py build-npu/bin/kvaf_scatter_batch_get_kernels.o
```

检查器无需 CANN，支持 NOTE/PROGBITS。scatter 硬件测试也会在导入 NPU 后端前
执行该检查，避免再次加载缺少类型声明的产物。

## 使用约束

```python
runtime = kvaf.NpuRuntime(
    get_kernel_path,
    batch_get_token_bytes=656,
    batch_get_tokens_per_block=4,
    batch_put_kernel_object_path=put_kernel_path,
    npu_core_count=1,
    aiv_batch_get_mode="cpu_push_scatter",
)
runtime.link_to(cpu_service_address)
runtime.initialize(ddr_bytes, slot_size)
```

- 支持 1～8 核、单 EID、eager；拒绝 graph capture。
- 请求容量 1～65536，固定 token 长度 1～65535 B；`miss_count=0` 是合法空批次。
- 仅搬运有效前缀，允许重复源和不连续输出行；输出行必须唯一，offset 不能越过源 block。
- Gather 每核两条 Jetty，SGE 数为 1～6；`--gather-threads` 至少覆盖其 AIV 核数。
- 同一 Runtime 复用 workspace，下一次 Get/Put 先 drain 上一批；
  `synchronize=False` 只表示入队成功，Tensor 和 stream 必须存活到执行完成。
- Host 在 Drain/Shutdown 检查设备状态。执行失败后拒绝继续提交；无法确认 DMA
  停止时保留注册资源，允许重试关闭。错误输出不能作为有效结果消费。
- done 轮询有次数上限，loader 的 kernel 超时为 180 秒；轮询次数不是毫秒。
- Scatter staging 按最大 65536 请求配置，约占 `65536 * token_bytes + 512 KiB` HBM，
  输出 Tensor 另行分配。大 token 的边界测试使用小请求容量以控制内存。

## 构建与完整回归

CPU 端首次配置匹配部署环境的 UMDK；已有正常运行的服务可以沿用：

```bash
cmake -S . -B build-cpu -DKVAF_BUILD_REMOTE_CPU_SERVICE=ON -DKVAF_URMA_ROOT=/path/to/umdk
cmake --build build-cpu -j
./build-cpu/runtime/kvaf_cpu_service --listen 0.0.0.0 --port 28990 --gather-threads 8
```

NPU 端沿用支持 A5 `dav-c310-vec` 的 CANN 环境，首次配置启用
`KVAF_ENABLE_AIV_BATCH_GET`、`KVAF_ENABLE_AIV_BATCH_PUT`、`KVAF_BUILD_PYTHON`。
在仓库根目录执行完整 scatter 回归：

```bash
set -euo pipefail
cmake --build build-npu -j
export PYTHONPATH="$PWD/build-npu/python${PYTHONPATH:+:$PYTHONPATH}"
export KVAF_RUN_NPU_TESTS=1
export KVAF_CPU_SERVICE_ADDR=127.0.0.1:28990
export KVAF_BATCH_GET_KERNEL="$PWD/build-npu/bin/kvaf_scatter_batch_get_kernels.o"
export KVAF_BATCH_PUT_KERNEL="$PWD/build-npu/bin/kvaf_execute_batch_put_kernels.o"
# 沿用已验证的 KVAF_NPU_EIDS；CPU 在远端时将地址改为实际 CPU_IP。
mkdir -p results
DEVICE_ID=7 KVAF_NPU_CORE_COUNT=1 KVAF_NPU_TOKEN_BYTES=656 KVAF_NPU_REQUESTS=8193 \
  python -u tests/python/test_aiv_scatter_batch_get.py \
  2>&1 | tee results/scatter-e2e-c1-t656-n8193.log
```

验收要求退出码为 0，最后打印：

```text
PASS: mode=cpu_push_scatter cores=1 token_bytes=656 counts=[0, 1, 7, 19, 8193, 8193, 19, 7, 1, 0]
```

测试包含真实 BatchPut、access table、BatchGet，回读比较全部有效输出及 guard，
并覆盖重复源、倒序输出行、A/B/A 缓冲、同步/异步两种调用、两个 caller stream
连续提交和最终 Shutdown。请使用普通 Python 执行，保留 assert 校验。

建议扩展如下组合，每次使用独立进程并保存日志：

| 核数 | token 字节 | 请求容量 | 目的 |
| --- | --- | --- | --- |
| 1/3/8 | 656 | 8193 | 多核、非均匀分工、大小批次及复用 |
| 8 | 1/13/32/4096/16384/16385/65535 | 19 | 尾部、16 KiB 分块边界及最大 token |
| 8 | 32 | 65536 | 最大请求容量 |

Gather 使用 `test_aiv_gather_batch_get.py` 和对应 gather `.o`。兼容入口
`test_aiv_sequential_batch_get.py` 根据 `KVAF_AIV_BATCH_GET_MODES` 为各模式启动
独立进程。未启用 `KVAF_RUN_NPU_TESTS=1` 返回 77，属于跳过。

## 验证记录与后续计时

此前 scatter 实际二进制缺少 `K_TYPE_AIV`，在 MTE2 路径出现 errcode 78。
补充类型声明后，用户已确认探针通过，并提供了 1 核、656 B、8193 请求容量的
完整回归 PASS。清理排障分支后仍需重编并按上述命令做目标机回归；多核和边界
组合以各自实际结果为准。

本地保留生产 kernel 的 CPU 搬运/协议模型、Host 异步生命周期、独立模块构建、
元数据和 Python 模式隔离回归。模型不代替 CANN 编译或设备测试。

已新增独立性能入口、包装脚本和可关闭的 CPU/AIV 分段计时，见
[scatter 性能测试与指标口径](aiv_scatter_performance.md)。测试分为关闭打点的
baseline 和带打点的 profiled 两轮，分别输出端到端、CPU→HBM 通信及本地 scatter
耗时/有效带宽，并保存 JSON/CSV 原始样本。

现有 `test_npu_batch_get_perf.py`/kernel profile 继续用于 `direct_read`。
scatter 使用 `tests/run_remote_cpu_scatter_perf.sh` 及 `start_scatter_profile` /
`scatter_profile` / `stop_scatter_profile` 接口，计时不复用业务完成状态字段。
