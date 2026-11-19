# URMA AICPU Gather Get

`host/batch_get_host.cc` 管理 Gather workspace、请求准备和完成状态；`kernel/gather_read.cc` 执行 CPU gather 通知、就绪等待和 URMA READ。

共同 ABI、Host loader、设备导出入口和 Python 测试保留在 [Get 公共目录](../../common/aicpu/README.md)。本目录的 CMake 将源文件加入现有 AICPU Get Host library，设备代码仍进入原 AICPU 安装包。

使用与性能测试见 [Gather 使用说明](../../../../../docs/aicpu_gather_read.md)。
