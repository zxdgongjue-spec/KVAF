# CPU SGL Write：连续输出的 BatchGet

`GetExecutionConfig("urma", "aicpu", "cpu_sgl_write")` 复用 CPU Push Scatter
的请求构建、四路 Jetty、预构建 WRITE 链和 done 协议。CPU 将离散 DDR 源直接
SGL WRITE 到调用方注册的 HBM output，AICPU 不执行 scatter，也不分配 payload staging。
原 `scatter` 模式保持原有输出映射能力。

## 接口和限制

```python
get = kvaf.GetExecutionConfig("urma", "aicpu", "cpu_sgl_write")
# 将 get 放入 default_get/get_resources，或传给 batch_get(execution=get)。
output_indices = torch.arange(request_length, dtype=torch.int32, device="npu")
```

两种 AccessTable 重载、普通 stream 调用、同步/异步完成以及 Get 图租约共用现有接口。
有效前缀中的 `output_indices[i]` 必须等于 `i`，每项写入
`output + i * token_bytes`；源索引和偏移仍可随机。有效输出前缀必须容纳
`miss_count * token_bytes` 字节，后缀保持不变。`miss_count=0` 不通知 CPU、不修改 output。
请求容量上限仍为 65536，token 大小固定为 Runtime 配置值。

AICPU 在发布 notify 前由各 worker 检查本分片的输出映射。任一分片不合法时不会
通知 CPU，避免部分写入；错误随现有同步、drain 或 status 检查报告。
设备执行错误与原 Scatter 一样使该执行资源失效，须按原 Get 恢复/关闭流程重建。

## 内存和完成

Host 用 direct HCCP 注册 output，控制 workspace 额外携带一份导出的 segment 描述。
CPU 每轮读取并导入该 output，复用 WRITE 计划时同时更新目标地址和 segment。
同一 WR 仍为多个源 SGE、一个连续目标 SGE，源上限为
`min(16, max_jfs_sge)`，单 WR 总字节最多为 `min(65536, max_msg_size)`。

本版保留原 chunk/Jetty 划分和 ready WRITE，AICPU 不轮询 ready。
CPU 收齐全部数据链尾 CQE，释放本轮导入的 output 后，再发布 done；AICPU 等待
done 并执行可见性屏障后结束。异步调用仍须等待完成才可读取、释放或复用输出。
发生传输故障时保留注册；Shutdown 先 STOP CPU DMA 队列，再注销 HBM。

普通调用在同步完成或 drain 后注销 output。`begin_get_output`/`end_get_output`
支持在性能循环中复用注册。图准备阶段注册固定 output，图退役后释放；多个图节点
共享同一 output 的注册按引用数保留。捕获过程中不新增注册，回放无需 Host 代理。

由于本版增加了 output 描述 READ 和每轮 CPU segment 导入/释放，端到端耗时包含
这些成本，与 Scatter 的差值不应直接视为 scatter 复制耗时。
设备 profile 的 `scatter_us`、`scatter_copies` 和 ready 查询数为零，最终等待计入 `ack_us`。
CPU profile 的 `address_read_us` 包含本轮 output 描述读取/导入，`done_us` 包含释放。

## 构建和真机验证

CPU 服务、NPU Host 库和设备包须一起重建并重启。CPU Push setup 增加 `output_mode`，
请求结构从 424 B 变为 428 B；旧 CPU 服务与新 Host 不兼容，不会自动回退。
Scatter 的 V3 设备入口和 launch 参数布局保留；新设备入口为
`KvafAicpuCpuSglWriteExecutionV1`，启动时检查，缺失则报错。

```bash
# CPU 端
./build.sh cpu

# NPU 端：先加载部署环境实际使用的 CANN set_env.sh
./build.sh npu
./build.sh npu --check-kernel

KVAF_CPU_SERVICE_ADDR=127.0.0.1:21111 DEVICE_ID=2 \
bash tests/run_remote_cpu_batch_get_perf.sh build-npu \
  --backend cpu_sgl_write --aicpu-workers 4 --execution both \
  --request-length 4096 --table-capacity 4096 --miss-count 4096 \
  --token-bytes 656 --tokens-per-block 1 \
  --source-access-pattern random --output-access-pattern contiguous \
  --no-sort-input-indices --warmup 10 --iterations 100 --kernel-profile
```

源地址排序会改变原 output 映射，当前性能入口拒绝此模式的 `--sort-input-indices`
和随机输出。默认复用 output 注册，也可通过 `KVAF_PERF_REUSE_GET_OUTPUT=0` 测瞬时注册。
用相同参数运行 `--backend cpu_push_scatter` 作为对照。
另以 `--miss-count 7`、`--miss-count 0` 验证部分/空前缀，并用
`--multi-graph-check` 验证多节点、多图和普通调用的串行复用。

无 SDK 的内核模型回归及其边界见 [诊断说明](../tests/diagnostics/README.md#cpu-sgl-write-kernel-model-host-only)。
