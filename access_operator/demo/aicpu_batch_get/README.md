# AICPU BatchGet

默认使用 `gather_read`：一次启动 N 个 AICPU 实例，每个 worker 独立生成清单、
请求 CPU gather、查询就绪进度、READ 到输出 HBM，并确认自己的最终 ACK。

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
