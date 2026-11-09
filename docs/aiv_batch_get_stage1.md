# AIV BatchGet 第一阶段实现与验证

## 数据通路

默认 BatchGet 模式为 `direct_read`。总体设计见
[三阶段设计](aiv_batch_get_scatter_gather_design.md)。

| 模式 | 执行顺序 |
| --- | --- |
| `cpu_push_scatter` | AIV 多核 SIMT 生成清单并 WRITE 到 CPU mailbox → 四路 Jetty WRITE payload → AIV 按 Ready scatter 到目标 HBM → 等待最终 done 和 DMA 排空 |
| `gather_read` | AIV WRITE 清单、command、notify → CPU gather → 最终 ACK → AIV 多 SGE READ 到 output → CQ 完成 |

Scatter 默认按 Jetty 分片的 Ready 提前消费。设置 `KVAF_SCATTER_PIPELINE=0` 后，CPU 省略
分片 Ready，AIV 等到整批 done 成功再开始搬运。两种模式下，可信请求超过 1024 个有效行时，
内核都会先发布首段清单，再合批发布剩余清单。
Gather 使用 `pipeline_chunk_bytes=0`，以最终 ACK 为完成条件。
AIV 不调用 UMDK；两端通过注册内存和现有 CPU 控制协议通信。

Scatter 使用 64 KiB UB stage ring 执行 GM→UB→GM，通过 MTE3→MTE2 事件安全复用 bank。
精确长度的 `DataCopyPad<uint8_t>` 用于读取 packed staging 和逐行写入目标。
包含 SIMT VF 的当前二进制由 loader 设置 `dynUBufSize=65536` 预留共享 UB；
旧纯 SIMD loader 的零动态 UB 配置不适用于这个混合二进制。
GM→UB 普通模式恢复、MTE2/MTE3 事件同步及退出屏障保留。

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

Scatter 参数末尾使用版本化选项字段。176 B 参数和 64 B 业务状态之外，计时使用独立的
每核 192 B 区域。Scatter mailbox 的打开、启动和停止命令使用 V8，Gather 和公共控制命令
使用 V6。部署 Scatter 时同步重编 CPU 服务、NPU Host 和 Scatter kernel。

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

下面的初始化示例使用性能测试的 block 布局，每个源 block 存 128 个 token，每个 token
为 656 B，slot 大小为 83,968 B。`batch_get_tokens_per_block` 是源数据布局参数，
Scatter 流水仍按每个 worker 的 1024 行 chunk 切分。

```python
token_bytes = 656
tokens_per_block = 128
slot_size = token_bytes * tokens_per_block

runtime = kvaf.NpuRuntime(
    get_kernel_path,
    batch_get_token_bytes=token_bytes,
    batch_get_tokens_per_block=tokens_per_block,
    batch_put_kernel_object_path=put_kernel_path,
    npu_core_count=4,
    aiv_batch_get_mode="cpu_push_scatter",
)
runtime.link_to(cpu_service_address)
runtime.initialize(ddr_bytes, slot_size)  # ddr_bytes 为 slot_size 的整数倍
```

- 支持 1～8 核、单 EID 和 eager 执行。可信 Scatter 入口由调用方保证 stream 未处于
  graph capture 中，带校验的入口会检查 capture 状态并拒绝 capture。
- 请求容量 1～65536，固定 token 长度 1～65535 B；`miss_count=0` 是合法空批次。
- 仅搬运有效前缀，允许重复源和不连续输出行；输出行必须唯一，offset 不能越过源 block。
- Gather 每核两条 Jetty，SGE 数为 1～6；`--gather-threads` 至少覆盖其 AIV 核数。
- 同一 Runtime 复用 workspace，下一次 Get/Put 先 drain 上一批；
  `synchronize=False` 只表示入队成功，Tensor 和 stream 必须存活到执行完成。
- Host 在 Drain/Shutdown 检查设备状态。执行失败后拒绝继续提交；无法确认 DMA
  停止时保留注册资源，允许重试关闭。错误输出不能作为有效结果消费。
- done 轮询有次数上限，loader 的 kernel 超时为 180 秒；轮询次数不是毫秒。
- Scatter HBM workspace 按最大 65536 请求配置，包括 `65536 * token_bytes` staging、
  512 KiB 地址清单和控制区。大 token 回归使用较小请求容量以缩小测试输出，workspace 仍按上限分配。

## 构建与完整回归

CPU 端使用与部署环境匹配的 UMDK 重编并启动服务：

```bash
cmake -S . -B build-cpu -DKVAF_BUILD_REMOTE_CPU_SERVICE=ON -DKVAF_URMA_ROOT=/path/to/umdk
cmake --build build-cpu -j
./build-cpu/runtime/kvaf_cpu_service --listen 0.0.0.0 --port 28990 --gather-threads 8
```

NPU 端沿用支持 A5 `dav-c310-vec` 的 CANN 环境，首次配置启用
`KVAF_ENABLE_AIV_BATCH_GET`、`KVAF_ENABLE_AIV_BATCH_PUT`、`KVAF_BUILD_PYTHON`。
在仓库根目录执行 Scatter 正确性回归。当前脚本固定使用 16 个源 block，每个 block
包含 4 个 token，并覆盖重复源、输出 guard 和 workspace 复用。上面的 128 token/block
对应性能测试使用的布局。

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

## Scatter 性能回归

miss-count 脚本固定使用 prepared 提交，提交后调用 `torch.npu.synchronize()`，
并将 profile 快照放在计时区间之外。以下示例使用每 block 128 个 token。

```bash
DEVICE_ID=7 KVAF_CPU_SERVICE_ADDR=127.0.0.1:28990 \
  bash tests/run_remote_cpu_scatter_miss_count_perf.sh "$PWD/build-npu" \
    --single --token-bytes 656 --tokens-per-block 128 --table-capacity 16384 \
    --request-length 32768 --miss-count 8192 --aiv-cores 4 --pipeline 1 \
    --warmup 3 --iterations 100 \
    --json results/scatter-aligned.json --csv results/scatter-aligned.csv
```

加 `--dry-run` 可查看配置。另一个入口 `tests/run_remote_cpu_scatter_perf.sh` 用于
请求和数据随样本变化的端到端性能回归。
