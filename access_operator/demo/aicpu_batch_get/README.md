# AICPU BatchGet

显式设置 `aicpu_batch_get_mode="gather_read"`：一次启动 N 个 AICPU 实例，每个 worker 独立生成清单、
请求 CPU gather、查询就绪进度、READ 到输出 HBM，并确认自己的最终 ACK。

Runtime 的 mode 默认为空，表示使用 AIV；AICPU 包从 CANN 安装目录自动定位，无需传 JSON 路径。
Gather 和 Scatter 都支持通过 `npu_eids` 指定多个候选，逐个尝试并选用一个可用 EID。

设置 `aicpu_batch_get_mode="cpu_push_scatter"` 可使用 Host direct HCCP 建链、
CPU SGL WRITE 到 HBM staging、AICPU 本地 scatter 的路径。
构建时添加 `./build.sh aicpu-urma --cpu-push`。
其动态请求协议、构建和验证见 [CPU 回推说明](../../../docs/aicpu_cpu_push_scatter.md)。

- `host/batch_get_host.cc`：内部 workspace、当前请求参数、stream 和完成状态。
- `host/execute_batch_get_loader.cc`：一次提交 `KvafAicpuGatherRead`，执行实例数为 worker 数。
- `kernel/batch_get.cc`：导出 `gather_read` 和 `cpu_push_scatter` 的设备入口。
- `kernel/gather_read.cc`：领取 worker 编号，分配各自的输入和工作区，执行完整请求链。
- `include/batch_get_abi.h`：复用原 BatchGet 输入 ABI，定义流水参数和状态。

具体发送顺序为 `ReadBatches → BuildReadBatch → WaitBatchPrefix → PostBatch → WaitCompletion`。
最后两个通用通信操作由 `src/aicpu_urma/kernel/transfer.cc` 提供。

`gather_read` 与 Put 共用一个 UMDK 会话与设备 SO，资源管理由 `src/aicpu_urma` 负责。
每个 worker 有自己的控制和数据 Jetty/CQ、CPU gather 处理对象、描述头、邮箱和进度。
staging 来自同一块 DDR 分配，各 worker 使用不重叠的区间，所有通道共用一对 TP。
完整构建和 Python 调用见 [使用说明](../../../docs/aicpu_gather_read.md)。

`python/test_batch_get.py` 使用真实 `NpuRuntime.batch_get` 接口逐轮校验数据，
支持一次运行比较 1、2、4 等 worker 数，并输出 Python 端到端耗时和有效载荷带宽。

全量 GatherRead 性能扫描使用 `tests/run_aicpu_gather_read_sweep.sh`，调用 `python/benchmark_gather_read.py`。
性能测试从初始化好的大地址池中有放回随机读，向普通 HBM 的随机行写入，并逐轮校验整个输出。
默认扫描 token 大小、精确 miss_count 和 worker 数，共 252 组；
命令及 CSV 指标口径见 [GatherRead 性能扫描](../../../docs/aicpu_gather_read.md#全量性能扫描)。

CPU push/scatter 使用 `tests/run_aicpu_cpu_push_scatter_sweep.sh`，调用 `python/benchmark_cpu_push_scatter.py`，
复用同一随机负载与算子 P50 统计，输出 ready 等待、scatter 和 done 等待等阶段。
命令和源池预算见 [CPU push 性能扫描](../../../docs/aicpu_cpu_push_scatter.md#全量性能扫描)。
