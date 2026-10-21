# AICPU BatchGet

当前只提供 `gather_read`：CPU gather 到 DDR staging，AICPU 按就绪前缀流水 READ 到输出 HBM。

- `host/batch_get_host.cc`：内部 workspace、当前请求参数、stream 和完成状态。
- `host/execute_batch_get_loader.cc`：提交 BatchGet 控制入口和多 worker READ 入口。
- `kernel/batch_get.cc`：统一设备入口，目前直接调用 `gather_read`。
- `kernel/gather_read.cc`：清单生成、CPU 通知、前缀等待、READ WR/SGL 组装及分波提交。
- `include/batch_get_abi.h`：复用原 BatchGet 输入 ABI，定义流水参数和状态。

具体发送顺序为 `ScatterQueue → AppendGatherBatch → WaitBatchPrefix → PostBatch → WaitCompletion`。
最后两个通用通信操作由 `src/umdk/kernel/transport.cc` 提供。

Get/Put 共用一个 UMDK 会话与设备 SO，资源管理由 `src/umdk` 负责。
完整构建和 Python 调用见 [使用说明](../../../docs/aicpu_gather_read.md)。
