# AICPU BatchPut

保留现有同步 WRITE 行为，与 AICPU BatchGet 共用同一个 UMDK 会话。

- `host/batch_put_host.cc`：生成源/目标传输清单，合并两端都连续的区间，上传描述符。
- `host/execute_batch_put_loader.cc`：提交 BatchPut 设备入口。
- `kernel/batch_put.cc`：通过共享 UMDK 传输层执行 WRITE；当前使用专用控制 Jetty，同步等完成。
- `include/batch_put_abi.h`：设备参数沿用共享 UMDK 控制记录。

Get/Put 的 kernel 与通用传输层统一打进一个设备 SO，避免重复的会话全局状态。
完整构建和 Python 调用见 [使用说明](../../../docs/aicpu_gather_read.md)。
