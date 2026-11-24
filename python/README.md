# Python 包目录

本目录存放 KVAF 的 Python 包入口，用于加载可选的 C++ 原生扩展，并向 Python 侧提供统一的 `kvaf` 命名空间。

当前 Python 包只承担调用入口和模块导出，不承担 KVAF Runtime、Meta Manager 或 KV Access Operator 的业务实现。

## 构建与导入

在仓库根目录加载 CANN 环境，使用安装了 `torch_npu` 的同一个 Python 构建和运行：

```bash
PYTHON="$(command -v python)" ./build.sh npu
export PYTHONPATH="$PWD/build-npu/python${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import torch; import torch_npu; import kvaf; print(kvaf.__file__)'
```

默认使用 `RelWithDebInfo` 并开启 Python 绑定及 CPU/NPU Pull；仅需 CPU Pull 时可加
`--cpu-pull-only`。自定义构建目录时，相应修改 `PYTHONPATH`。CPU 服务构建
`./build.sh cpu` 不生成 Python 包；直接运行 `cmake` 还需显式开启
`KVAF_BUILD_PYTHON` 和所需后端。完整构建选项见 [仓库构建说明](../README.md#构建)。

`kvaf.tcp_control` 额外提供独立的应用 TCP 通知通道，供 Connector 传递 receiver、NPU ReadyToken 和业务消息。`TcpControlListener` / `TcpControlChannel` 支持长度分帧、超时和双向通知；活动会话断线时不自动重放 READ 或释放源。纯 NPU benchmark 已使用该通道替代 Pipe，producer 监听应用 TCP，reader 主动连接并发送请求；独立 KVHR 硬件建链仍由 reader 监听、producer 注册接入。支持同机自动启动及两机独立启动，见 [TCP 通知使用说明](../docs/hbm_raw_addr_read.md#tcp-通知与两机独立启动)。

## 共享池 CPU Pull

启用 `KVAF_ENABLE_CPU_OFFLOAD` 后，`NpuRuntime(cpu_pull_options=CpuPullOptions(timeout_ms=30000))` 提供同步的 `batch_cpu_pull()`，以及 `prepare_put / submit_cpu_pull / query_cpu_pull / wait_cpu_pull / commit_put / abort_put` 显式事务接口，两者均支持本卡或跨卡源。
`CpuPullHbmSource.register(ddr_ticket, tensors)` 在生产卡完成注册和 CPU 建链，成功返回 None；不再单独交换 source_desc 或调用 attach_cpu_source。读取端直接使用 ReadyToken 提交。源端使用 `ready()`、`release_batch(ready)`、`cancel_batch(ready)` 管理批次。
DDR ticket 当前为 version=2，仅含 `version/address/region_id/epoch/token/eid`；读取端 DDR 布局仍从 `runtime.remote_region` 获取。ReadyToken 不再携带完整 MR 列表：CPU 源描述为 `source_id/region_id/region_epoch/mr_length`，NPU 源描述为 `source_id/source_epoch`，两种 Ready 均含 `sequence/mr_id`。对象仍可 pickle，旧序列化格式会被拒绝。当前 CPU Pull 为 KVRP v8，成功批次只有一次 SubmitAndWait，读取端本地提交元数据后由应用通知生产端解冻，CPU service 与 NPU 扩展需配套重编译、重启。显式 Submit/Wait 接口保留。NPU Pull 为 KVHR v9，NPU 两端需一起更新。两端 EID 均可省略，注册时自动选择同机 fullmesh 或跨机 CLOS。

NPU 直读由读取端 `runtime.listen_hbm()` 监听，生产端 `NpuPullHbmSource.register(tensors, receiver=receiver)` 一次完成注册建链。`batch_get_raw_addr()` 接受三个连续 NPU int64 Tensor，与普通 `batch_get()` 一样默认 `synchronize=False`，只入队并返回 `state="submitted"`；调用方负责同 stream 消费或外层同步后使用结果、通知源端复用。显式 `synchronize=True` 才内部等待并返回 done。Host 不生成启动编号、不回读 status；应用负责输入合法性和输出校验，每批没有 AcquireRaw、Complete 或 Cancel RPC。
正常异步调用由应用保留 metadata Tensor、storage、stream 和源直到完成；KVAF 将 output
Tensor 及其实际 storage 保留到 Runtime 成功关闭。图模式先对同一 MR/output 预热并外部同步，
再用框架捕获普通 raw 调用，无须额外的 KVAF Begin/End 或 graph lease。图回放不执行 Host
ReadyToken 检查，应用匹配每批的源/MR，保持元数据、stream、源/输出和通信资源存活；停止
replay、同步并销毁图后，才能注销源/关闭 Runtime。Runtime 关闭不隐式完成上述步骤。
成功读取不消费 ReadyToken，同一冻结源可重复读取。cancel_raw_addr 只拦截后续 Host 提交，
不能撤回已入队任务或捕获图；pending_raw_addr 仅表示提交/显式等待失败后的隔离状态，
不是异步完成查询。只有本次调用完成状态未知时，KVAF 才保留其 metadata/storage/stream
至安全关闭。source.release_batch 仍由应用在确认消费者完成后调用。
旧类名、旧重载及独立 OffloadRuntime 已删除。完整接口见 [Connector 设计](../docs/connector_pull_api_design.md)。

基础初始化与 DDR 分开：无布局的 `NpuRuntime(device_id=..., npu_pull_options=NpuPullOptions(...))`
加 `initialize()` 即可 `listen_hbm()`；CPU Pull 需要 `link_to()` 和 `initialize_ddr()`。
原来带布局的构造、普通 Put/Get、`initialize(ddr_bytes, slot_size)` 和 Get graph 接口继续可用。
普通执行配置及其 Drain/graph lease、独立的 Pull 选项与生命周期说明见 [Bindings](../bindings/README.md)。
