# URMA AICPU Put

本目录包含 AICPU Put 的 Host 请求准备、kernel loader、设备实现和私有 ABI。公共 UMDK 会话与设备包构建位于 `access_operator/transports/aicpu_urma`。

现有 `kvaf_aicpu_batch_put_host` 构建目标、设备入口与安装包不变。

完整构建和 Python 调用见 [使用说明](../../../../docs/aicpu_gather_read.md)。
