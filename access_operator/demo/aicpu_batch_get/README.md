# AICPU BatchGet

当前只提供 `gather_read`：一次启动 N 个 AICPU 实例，每个 worker 独立生成清单、
请求 CPU gather、查询就绪进度、READ 到输出 HBM，并确认自己的最终 ACK。

- `host/batch_get_host.cc`：内部 workspace、当前请求参数、stream 和完成状态。
- `host/execute_batch_get_loader.cc`：一次提交 `KvafAicpuGatherRead`，执行实例数为 worker 数。
- `kernel/batch_get.cc`：统一设备入口，目前直接调用 `gather_read`。
- `kernel/gather_read.cc`：领取 worker 编号，分配各自的输入和工作区，执行完整请求链。
- `include/batch_get_abi.h`：复用原 BatchGet 输入 ABI，定义流水参数和状态。

具体发送顺序为 `ReadBatches → BuildReadBatch → WaitBatchPrefix → PostBatch → WaitCompletion`。
最后两个通用通信操作由 `src/aicpu_urma/kernel/transfer.cc` 提供。

Get/Put 共用一个 UMDK 会话与设备 SO，资源管理由 `src/aicpu_urma` 负责。
每个 worker 有自己的控制和数据 Jetty/CQ、CPU gather 处理对象、描述头、邮箱和进度。
staging 来自同一块 DDR 分配，各 worker 使用不重叠的区间，所有通道共用一对 TP。
完整构建和 Python 调用见 [使用说明](../../../docs/aicpu_gather_read.md)。

`python/test_batch_get.py` 使用真实 `NpuRuntime.batch_get` 接口逐轮校验数据，
支持一次运行比较 1、2、4 等 worker 数，并输出 Python 端到端耗时和有效载荷带宽。
