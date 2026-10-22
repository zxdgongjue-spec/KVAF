# Python BatchGet 使用 AICPU UMDK 流水

已有的 `NpuRuntime.batch_get` 可以选择 UMDK 后端。一次 launch 启动 N 个 AICPU 实例，
各自从本轮 NPU Tensor 的输入区间生成源清单，请求 CPU gather 到自己的 DDR staging 区间，
查询自己的就绪前缀、提交 READ，并等待自己的完成通知和 CPU ACK。
indices、offsets、mask、output_indices 和数据可以每轮变化，不缓存上一轮内容。

## 构建和启动

在仓库根目录执行：

```bash
./build.sh cpu
./build.sh aicpu-urma --install-kernel
export PYTHONPATH="$PWD/build-aicpu-urma/python:$PYTHONPATH"
```

CPU 服务使用新二进制重启：

```bash
./build-cpu/runtime/kvaf_cpu_service \
  --listen 0.0.0.0 --port 28990 \
  --gather-threads 8 --gather-mode active --gather-affinity on
```

NPU 进程先加载 CANN 环境，再使用现有 Tensor 接口：

```python
import os
import torch
import torch_npu
import kvaf

torch.npu.set_device(7)
runtime = kvaf.NpuRuntime(
    batch_get_kernel_object_path="",
    batch_get_token_bytes=656,
    batch_get_tokens_per_block=1,
    device_id=7,
    npu_core_count=1,
    aicpu_urma_kernel_json_path=os.path.join(
        os.environ["ASCEND_HOME_PATH"],
        "opp/vendors/cust/op_impl/aicpu/config/kvaf_aicpu_urma.json",
    ),
    urma_device="udmac0d1e5",
    urma_eid_index=0,
)
runtime.link_to("127.0.0.1:28990")
runtime.initialize(ddr_bytes=656 * 8192, slot_size=656)

keys = list(range(4096))
source = torch.randint(0, 256, (4096, 656), dtype=torch.uint8, device="npu:7")
runtime.batch_put(keys, source, [i * 656 for i in keys], [656] * 4096)
table = runtime.prepare_access_table(keys)
device_table = runtime.access_table_view(table, device="npu")
indices = torch.arange(4096, dtype=torch.int32, device="npu:7")
offsets = torch.zeros_like(indices)  # 表项内的字节偏移。
mask = torch.ones(4096, dtype=torch.uint8, device="npu:7")
output_indices = indices.clone()
output = torch.empty_like(source)
runtime.batch_get(device_table, indices, offsets, mask, output_indices, output,
                  synchronize=True)
assert torch.equal(output, source)
runtime.release_access_table(table)
runtime.shutdown()
```

设备、URMA device、EID 和服务地址按部署环境调整。原来的 addresses/valid Tensor 重载也可直接使用。
未设置 `aicpu_urma_kernel_json_path` 时仍走原 AIV 后端。

## 用 Python 检查正确性和比较 worker 数

CPU 服务启动后，在仓库根目录运行：

```bash
export PYTHONPATH="$PWD/build-aicpu-urma/python:$PYTHONPATH"
python3 access_operator/demo/aicpu_batch_get/python/test_batch_get.py \
  --device-id 7 --peer 127.0.0.1:28990 \
  --urma-device udmac0d1e5 --eid-index 0 \
  --workers 1 2 4 --blocks 4096 --token-bytes 656 \
  --tokens-per-block 2 --warmup 10 --iterations 1000
```

脚本从 CANN 环境变量寻找已安装的 kernel JSON，也可用 `--kernel-json` 显式指定。
每轮重新生成源数据和请求，稀疏模式下逐轮移动 mask；先完成 BatchPut，再执行 BatchGet，
逐字节检查整个输出，包括未写入的行和间隔；还在 NPU 上执行转换和加法，检查后续算子读取结果。
预检覆盖两种 Python 重载、稀疏 mask、全 mask，以及异步 BatchGet 后同 stream 的 NPU 消费者。
源和输出 Tensor 的注册地址持续有效到 Runtime 关闭；每轮释放 AccessTable 和 key 后重新写入，
不依赖上一轮数据或远端地址保持不变。设置 `--active-percent 75` 可测试稀疏请求。

每个 worker 配置输出一行 `AICPU_BATCH_GET_PASS`，包含 mean、p50、p95、p99（µs，小数点后 1 位）和有效载荷 GB/s。
计时覆盖 Python `batch_get(..., synchronize=True)` 调用至完成，包含 Host/AICPU/CPU 和同步开销；
输入准备、BatchPut、AccessTable 创建与上传、输出下载和正确性检查均在计时之外。
正常带宽统计不启用 profile。需要定位阶段耗时时，在同一命令末尾加 `--profile-rounds 100`：
脚本会在正常测量后另跑 100 轮采样，打印平均阶段耗时，不逐轮刷日志。

Python 也可直接获取单次采样：

```python
profile = runtime.batch_get(device_table, indices, offsets, mask, output_indices, output,
                            synchronize=True, profile=True)
print(profile["host"])
print(profile["workers"])
```

`profile` 默认关闭，保持原来的返回值 `None`；开启时仅支持同步 AICPU 调用，返回字典。
关闭时不读取新增的阶段时钟，也不分配或回传 profile 缓冲区。开启时额外采集时间和回传结果，
所以用正常测量轮次报告带宽，用独立的 profile 轮次定位耗时。

| 输出 | 含义 |
| --- | --- |
| Python `tensor_adapt_us` / `native_call_us` | Tensor 适配与参数检查 / 原生 Runtime 调用；合计为 `binding_total_us` |
| Python `python_other_us` | 外部 Python 计时减去 `binding_total_us`，含绑定分派与 profile 字典转换 |
| Host | 准备、清零入队、launch、状态及 profile D2H 入队、等待 stream、收尾；各项之和为 Host total |
| Device | 最早 worker 进入至最后 worker 主体退出，使用同一设备时钟；不含最后的 profile 导出 |
| Worker | 相对进入/退出、总时长、清单构建、源清单 WRITE、通知 CPU、流水 READ、最终 ACK |
| Within request | 输入缓存维护、清单缓存维护；已包含在清单构建中 |
| Within READ | 工作区准备、WR/SGL 构建、等待就绪前缀、provider 提交、最后 CQE 等待；已包含在 READ 中 |

各 worker 互相重叠，不能把它们的阶段最大值相加为端到端时间。
`ready_wait` 包含与 CPU gather 重叠的等待，并非 CPU gather 本身的计时。
provider 提交包含编码、写 SQ 和 doorbell，不能单独当作写 SQ 耗时。
Host wait 包含排队的设备执行和 D2H；Host 与设备的绝对时间戳不相减。

## 执行和资源约定

1..8 个 AICPU worker 共用一对 TP，每个 worker 有独立数据 Jetty/CQ 和控制 Jetty/CQ。
CPU 侧每条通道也各有独立 Jetty/RQ/CQ。CPU 与 AICPU 共用端点枚举、队列创建和 TP/Jetty 扩展代码；
AICPU 进程内的 liburma 动态加载与会话由 AICPU 自己维护。

每个 worker 负责输入请求下标的一个连续区间，在本区间内跳过 mask=0 的条目。
描述头、源清单、command、notify、ACK、ready_bytes 和 CPU gather 处理对象均按 worker 独立。
CPU 服务的 `--gather-threads` 是全部 worker 的总预算，须不小于 AICPU worker 数；
预算在 worker 之间均分，余数从 worker 0 开始分配。每个 worker 每完成 128 KiB 连续前缀即可开始 READ。

每次最多 65536 项；token 大小由构造参数指定，最大总载荷须能用 uint32 表示。
Runtime 在用户指定 DDR 数据容量之外按 worker 数预留 staging、源描述符和控制区，
每个 staging 区间可容纳 `ceil(65536 / worker_count)` 个 token，并单独对齐到 64 字节。
这些空间不进入 KV slot 分配。HBM 描述符和 scratch 使用独立内部缓冲，输出 Tensor 无需额外尾部空间。
READ 使用 SGL 6、每次提交 32 条 WR；每波最多 4096 条 WR，只请求最后一条的有序 CQE，完成后复用 WR/SGL 空间。
AICPU 只将输出地址交给 URMA，不直接读写输出 payload，因此 READ 前后不再逐段清理输出的 AICPU 缓存。
元数据、源清单和控制邮箱仍保留缓存维护；每个 worker 的流水查询只使用自己的 scratch 区间。

BatchPut 同步完成；BatchGet 默认只 enqueue，`synchronize=True` 等待完成并读取错误状态。
内部 workspace 在下一次操作或 Shutdown 前排空；跨 stream 依赖由调用方负责。
调用方须保留元数据 Tensor 至执行结束，已注册的源/输出 Tensor 至 Shutdown 前保持有效。
同一 AICPU 进程只支持一个活动 UMDK 会话。目前该后端使用 eager 调用，不支持 ACL Graph capture/replay。

本地验证使用 ACL/provider 替身；本分支的 NPU 兼容性与性能需要服务器验证。

## 实现位置

| 目录/文件 | 职责 |
| --- | --- |
| `access_operator/src/aicpu_access_operator.cc` | Runtime 适配、Get/Put 共享会话和跨操作同步 |
| `access_operator/demo/aicpu_batch_get/host/` | BatchGet workspace、参数和 kernel 提交 |
| `access_operator/demo/aicpu_batch_get/kernel/gather_read.cc` | 当前流水算法、READ WR/SGL 组装和调度 |
| `access_operator/demo/aicpu_batch_put/` | 原有同步 BatchPut 的 Host/设备实现 |
| `access_operator/src/umdk/host/` | 通用会话和 kernel 加载 |
| `access_operator/src/umdk/kernel/` | 通用队列、注册、提交、CQ 及统一设备包 |

当前 BatchGet 入口仅调用 `gather_read`；没有其他实现的占位代码或选择参数。
两个 demo 均按 `include/host/kernel` 分层，共用一个设备包和会话；Python 参数及构建命令保持不变。
