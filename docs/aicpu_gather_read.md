# Python BatchGet 使用 AICPU UMDK 流水

已有的 `NpuRuntime.batch_get` 可以选择 UMDK 后端。一次 launch 启动 N 个 AICPU 实例，
各自从本轮 NPU Tensor 的输入区间生成源清单，请求 CPU gather 到自己的 DDR staging 区间，
查询自己的就绪前缀、提交 READ，并等待自己的完成通知和 CPU ACK。
indices、offsets、miss_count、output_indices 和数据可以每轮变化，不缓存上一轮内容。
本页描述默认 `aicpu_batch_get_mode="gather_read"`。
Host direct HCCP 建链的 `cpu_push_scatter` 路径见 [CPU 回推说明](aicpu_cpu_push_scatter.md)。
与顶层 BatchGet 一致，`int32[1] miss_count` 指定输入数组的有效前缀长度；
只处理前 `miss_count` 项，取值范围为 0 到输入数组容量。
AICPU 读取这个计数，将有效前缀均分给各 worker；READ 直接使用调用方的 `output_indices` 组 SGL。
源地址直接取 `addresses[indices[i]] + offsets[i]`，调用方保证有效条目的索引、
块内偏移、地址有效性及输出行号正确，输入元数据在本轮完成前保持不变。

## 构建和启动

合入 main 后控制协议使用版本 6，CPU 服务和 NPU 客户端需同时重新构建并重启。
AICPU 使用独立的输入 ABI；重新安装设备包后再启动客户端。

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
  --gather-threads 8 --gather-mode active --gather-affinity off
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
miss_count = torch.tensor([4096], dtype=torch.int32, device="npu:7")
output_indices = indices.clone()
output = torch.empty_like(source)
runtime.batch_get(device_table, indices, offsets, miss_count, output_indices, output,
                  synchronize=True)
assert torch.equal(output, source)
runtime.release_access_table(table)
runtime.shutdown()
```

设备、URMA device、EID 和服务地址按部署环境调整。原来的 addresses/valid Tensor 重载也可直接使用。
`NpuRuntime::Impl::CreateCore()` 在创建 RuntimeCore 时选择 AccessOperator：
`aicpu_urma_kernel_json_path` 非空时创建 `UmdkAccessOperator`，为空时创建 AIV 的
`DefaultAccessOperator`。同一个 Runtime 的 Get/Put 使用选中的实现。
相应实现必须在构建时启用；`build.sh aicpu-urma` 只编译 AICPU 后端，因此必须提供该 JSON 路径。

如果调用方已经持有 NPU 地址表，直接调用现有 Tensor 重载，不必再创建 Host AccessTable 或重新上传地址表：

```python
runtime.batch_get(addresses, valid, indices, offsets, miss_count, output_indices, output,
                  stream=stream, synchronize=True)
```

调用方管理这些地址对应的远端数据生命周期。使用 `prepare_access_table` 的调用仍保留公共元数据接口，
Host 地址表通过整段 H2D 上传；每次 BatchGet 本身不遍历或复制 Host 地址表。

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
每轮重新生成源数据、输入索引、偏移和输出行号；先完成 BatchPut，再执行 BatchGet，
逐字节检查整个输出，包括未写入的行和间隔；还在 NPU 上执行转换和加法，检查后续算子读取结果。
预检覆盖两种 Python 重载、完整/部分/空前缀，以及异步 BatchGet 后同 stream 的 NPU 消费者。
源和输出 Tensor 的注册地址持续有效到 Runtime 关闭；每轮释放 AccessTable 和 key 后重新写入，
不依赖上一轮数据或远端地址保持不变。设置 `--active-percent 75` 可测试前 75% 的请求。

每个 worker 配置输出一行 `AICPU_BATCH_GET_PASS`，包含 mean、p50、p95、p99（µs，小数点后 1 位）和有效载荷 GB/s。
计时覆盖 Python `batch_get(..., synchronize=True)` 调用至完成，包含 Host/AICPU/CPU 和同步开销；
输入准备、BatchPut、AccessTable 创建与上传、输出下载和正确性检查均在计时之外。
正常带宽统计不启用 profile。需要定位阶段耗时时，在同一命令末尾加 `--profile-rounds 100`：
脚本会在正常测量后另跑 100 轮采样，打印平均阶段耗时，不逐轮刷日志。

Python 也可直接获取单次采样：

```python
profile = runtime.batch_get(device_table, indices, offsets, miss_count, output_indices, output,
                            synchronize=True, profile=True)
print(profile["host"])
print(profile["workers"])
```

`profile` 默认关闭，保持原来的返回值 `None`；开启时仅支持同步 AICPU 调用，返回字典。
关闭时不读取新增的阶段时钟，也不分配或回传 profile 缓冲区。开启时额外采集时间和回传结果，
所以用正常测量轮次报告带宽，用独立的 profile 轮次定位耗时。

| 输出 | 含义 |
| --- | --- |
| Python `binding_prepare_us` / `native_call_us` | 绑定层准备（Tensor 信息、参数检查、stream 解析和 profile 对象分配）/ 原生 Runtime 调用；合计为 `binding_total_us` |
| Python `python_other_us` | 外部 Python 计时减去 `binding_total_us`，含绑定分派与 profile 字典转换 |
| Host | 准备、launch、状态及 profile D2H 入队、等待 stream、收尾；各项之和为 Host total |
| Device | 最早 worker 进入至最后 worker 主体退出，使用同一设备时钟；不含最后的 profile 导出 |
| Worker | 相对进入/退出、总时长、清单构建、源清单 WRITE、通知 CPU、流水 READ、最终 ACK |
| Within request | 输入缓存维护、清单缓存维护；已包含在清单构建中 |
| Within READ | 工作区准备、WR/SGL 构建、等待就绪前缀、provider 提交、最后 CQE 等待；已包含在 READ 中 |

各 worker 互相重叠，不能把它们的阶段最大值相加为端到端时间。
`ready_wait` 包含与 CPU gather 重叠的等待，并非 CPU gather 本身的计时。
provider 提交包含编码、写 SQ 和 doorbell，不能单独当作写 SQ 耗时。
Host wait 包含排队的设备执行和 D2H；Host 与设备的绝对时间戳不相减。

## 全量性能扫描

在仓库根目录构建：

```bash
./build.sh cpu
./build.sh aicpu-urma --install-kernel
```

先在一个终端启动 CPU 服务，8 个 gather 线程支持扫描中的 1、2、4、8 worker：

```bash
./build-cpu/runtime/kvaf_cpu_service \
  --listen 0.0.0.0 --port 28990 \
  --gather-threads 8 --gather-mode active --gather-affinity off
```

另一个终端加载 CANN 环境，在仓库根目录运行：

```bash
python3 tests/run_aicpu_gather_read_sweep.py \
  --device-id 7 --peer 127.0.0.1:28990 \
  --urma-device udmac0d1e5 --eid-index 0
```

脚本自动使用 `build-aicpu-urma/python`，固定测试 `gather_read`，默认组合为：

| 参数 | 扫描值 |
| --- | --- |
| token_bytes | 128、256、512、656、1024、2048、4096 B |
| miss_count | 32、64、128、256、512、1024、2048、4096、8192 |
| workers | 1、2、4、8 |

共 252 组，每组独立进程，预热 10 轮、正常计时 100 轮、另采样 profile 100 轮。
使用 `python/benchmark_gather_read.py`，按 AIV 全量随机测试的方式建立源地址池：
默认 1,277,952 个源块，每块 128 个 token。数据和地址表初始化一次，测量期间复用。
每轮从整个源池有放回抽取块索引，随机选择块内 token，再生成不重复的随机输出行号。
输入数组和输出行数为 `max(32768, miss_count)`；只执行前 miss_count 项。
输出 HBM 连续分配，所有行都可以被选中，不再固定间隔一行保留保护区。
每轮在 NPU 上比较整个输出与预期值，同时检查随机未写入行；源数据包含 token 身份信息，以检查读错源块。
正常测量与 profile 轮次分开，输入准备、BatchPut、地址表准备及校验均在计时之外。

先跑小范围可使用：

```bash
python3 tests/run_aicpu_gather_read_sweep.py \
  --device-id 7 --peer 127.0.0.1:28990 \
  --urma-device udmac0d1e5 --eid-index 0 \
  --token-bytes 656 --miss-counts 32 4096 --workers 1 2 4 \
  --warmup 3 --iterations 20 --profile-rounds 20
```

- `--table-capacity` 默认 1,277,952，只控制源地址池的块数，与 miss_count 独立；`--tokens-per-block` 默认 128。
- 源 DDR 容量为 `table_capacity × tokens_per_block × token_bytes`。默认 656 B/token 对应约 99.94 GiB，4096 B/token 对应 624 GiB；每组日志和计划都会记录所需源池容量。
- 原 `python/test_batch_get.py` 仍用于逐轮改变源数据、检查保护行和异步调用的正确性测试，性能扫描不再使用它的数据布局。
- `--build-dir` 可指定构建目录；`--kernel-json` 可指定已安装的算子 JSON。
- `--output-dir` 指定一个新结果目录，默认 `reports/aicpu-gather-read/<时间戳>`。
- `--dry-run` 只生成和打印扫描计划，不加载 torch、不访问设备。
- `--timeout` 默认每组 600 秒，包含数据准备和校验。失败或超时后停止，已完成的 CSV 和日志保留。

当前 GatherRead 固定使用 SGL 6、每批 32 条 WR、128 KiB 就绪前缀；
Jetty 数由 worker 数确定，每个 worker 一条数据队列和一条控制队列，不扫描 AIV 核数或每核 Jetty 数。
CPU gather 线程预算和绑定策略由服务启动参数控制，同一次扫描中保持一致。

结果目录包含 `plan.json`、每组 `.log`/`.json` 和逐组更新的 `summary.csv`。
CSV 同时记录源池块数、每块 token 数、输入数组长度、源 DDR 容量及源/目标访问模式。
汇总字段沿用 AIV 扫描的命名，耗时单位为 µs（显示 1 位小数），带宽为十进制 GB/s：

| 字段 | 统计口径 |
| --- | --- |
| mean_us / p50_us / p90_us / p99_us | profile 轮次中，最早 AICPU worker 进入至最后 worker 退出的算子墙钟时间 |
| p50_GBps | 有效载荷 `token_bytes × miss_count` 除以算子 P50 时间 |
| post_* | 每轮各 worker 累计 provider 提交时间的最大值，再计算均值和分位数 |
| completion_wait_* | 每轮各 worker 累计 READ CQ 等待时间的最大值，再计算均值和分位数；不含 CPU ACK 等待 |
| end_to_end_* | 未启用 profile 的 Python BatchGet 调用至同步完成耗时，另列平均带宽与 P50 带宽 |
| read_mean_us / ready_wait_mean_us / wr_build_mean_us | 每轮各 worker 的流水 READ、就绪前缀等待、WR/SGL 构建时间取最大值，再求均值 |
| request_mean_us / source_write_mean_us / notify_mean_us / ack_mean_us | 每轮各 worker 的清单构建、源清单 WRITE、通知和最终 ACK 等待时间取最大值，再求均值 |

阶段最大值可能来自不同 worker，且存在嵌套和重叠，不可相加为总时间。
`ready_wait` 是 AICPU 等待 CPU 就绪前缀的时间，不是 CPU gather 的独立耗时。
AICPU profile 使用纳秒计时，无需 AIV 扫描的设备频率换算参数。

## 执行和资源约定

1..8 个 AICPU worker 共用一对 TP，每个 worker 有独立数据 Jetty/CQ 和控制 Jetty/CQ。
CPU 侧每条通道也各有独立 Jetty/RQ/CQ。CPU 与 AICPU 共用端点枚举、队列创建和 TP/Jetty 扩展代码；
AICPU 进程内的 liburma 动态加载与会话由 AICPU 自己维护。

每个 worker 负责有效前缀中的一个连续区间，直接按顺序构建源清单和 READ。
CPU 直接引用收到的源清单，使用到本轮拷贝结束、发布 ACK 为止，不再逐条复制清单或检查源地址范围。
BatchGet 在描述头中提供固定片段长度 `span_bytes`（例如 656 B）；CPU 用
`staging_offset / span_bytes` 和 `staging_offset % span_bytes` 定位源片段及片段内偏移，
不再扫描清单生成累计偏移表。相邻源地址在当前拷贝分块内合并成一次 memcpy；零散地址仍按片段拷贝。
旧的变长清单使用 `span_bytes=0`，仍按长度构建偏移表。清单头保持 48 B，进度字段位置保持不变。
启用固定长度描述头的版本需要同时更新 CPU 服务与 AICPU 包，并重启 CPU 服务。
单 worker 直接使用编号 0；多 worker 从 AICPU 进程内的循环编号器各领一个编号。
每次 launch 的 N 个实例恰好覆盖全部 N 个编号；上一轮结束后才允许复用资源。
Host 每轮直接提交算子，不再分配 HBM 编号缓冲或提交清零任务。
描述头、源清单、command、notify、ACK、ready_bytes 和 CPU gather 处理对象均按 worker 独立。
CPU 服务的 `--gather-threads` 是全部 worker 的总预算，须不小于 AICPU worker 数；
预算在 worker 之间均分，余数从 worker 0 开始分配。每个 worker 每完成 128 KiB 连续前缀即可开始 READ。

每次最多 65536 项；token 大小由构造参数指定，最大总载荷须能用 uint32 表示。
Runtime 在用户指定 DDR 数据容量之外按 worker 数预留 staging、源描述符和控制区，
每个 staging 区间可容纳 `ceil(65536 / worker_count)` 个 token，并单独对齐到 64 字节。
这些空间不进入 KV slot 分配。HBM 仅保留发给 CPU 的源描述符和控制 scratch，输出 Tensor 无需额外尾部空间。
READ 使用 SGL 6、每次提交 32 条 WR；每波最多 4096 条 WR，只请求最后一条的有序 CQE。
WR/SGL 工作区按本轮请求的上界增长，完成后复用容量；每轮仍重新填写全部有效请求。
会话用独立数组保存数据队列和控制队列，worker 0 与其他 worker 使用相同的存储和访问方式。
AICPU 只将输出地址交给 URMA，不直接读写输出 payload，因此 READ 前后不再逐段清理输出的 AICPU 缓存。
输入元数据读取前、源清单发送前、状态回传前仍做缓存维护。
每个 worker 的 scratch 独占一条 64 B 缓存行：通知发送前清理一次，此后 AICPU 只读取查询结果，
不再在每次 ready/ACK READ 前重复清理；收到 CQE 后仍清理再取值，避免读到旧缓存。
不清零 scratch 中的 ACK 槽，查询完成后直接使用远端覆盖的值。
单个不超过 64 KiB 的控制传输直接填写一个 WR/两端 SGE，保留在队列对象中至完成或队列销毁，
省去通用分段构建及 vector 操作；清单大于 64 KiB 和多段通知仍走通用路径。
连续性、设备、元素宽度、长度以及队列完成/错误检查仍在相应入口执行。

BatchPut 同步完成；BatchGet 默认只 enqueue，`synchronize=True` 等待完成并读取错误状态。
内部 workspace 在下一次操作或 Shutdown 前排空；跨 stream 依赖由调用方负责。
UMDK 控制/Put 的启动或完成处理失败后，会话拒绝后续 Get/Put；应关闭并重新初始化 Runtime。
Shutdown 会先等待尚未确认完成的控制/Put 操作，再销毁通信资源。若仍无法确认完成，保留缓冲区并返回错误，
调用方需继续保留已注册的源/输出 Tensor 和对应 stream。
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
| `access_operator/include/aicpu_urma/` | Host 会话、算子加载、Host/设备控制 ABI、设备通信接口和缓存工具 |
| `access_operator/src/aicpu_urma/host/` | Host 侧会话组织和 kernel 加载 |
| `access_operator/src/aicpu_urma/kernel/session.cc` | AICPU 加载 liburma、创建/连接队列、注册内存和资源释放 |
| `access_operator/src/aicpu_urma/kernel/transfer.cc` | READ/WRITE WR 构建、提交和 CQ 完成处理 |
| `access_operator/src/aicpu_urma/kernel/control_kernel.cc` | Host 控制命令的设备入口和分发 |
| `access_operator/src/aicpu_urma/package/` | 算子 JSON、UMDK 库打包和安装脚本 |

BatchGet 还提供 `cpu_push_scatter` 入口，通过 `aicpu_batch_get_mode` 在初始化时选择；
两条路径使用相同的输入前缀和输出地址语义。
两个 demo 均按 `include/host/kernel` 分层，共用一个设备包和会话；Python 参数及构建命令保持不变。
