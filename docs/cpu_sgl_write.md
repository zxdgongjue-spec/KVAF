# CPU SGL Write：按 WR 指定输出位置的 BatchGet

`GetExecutionConfig("urma", "aicpu", "cpu_sgl_write")` 复用 CPU Push Scatter
的请求构建、四路 Jetty、预构建 WRITE 链和 done 协议。CPU 将离散 DDR 源直接
SGL WRITE 到调用方注册的 HBM output。每个数据 WR 写入一个连续目标区间，各 WR 可以写到
同一 output 中不同的位置。AICPU 负责构建请求，不执行 scatter，也不分配 payload staging。

## 接口和限制

```python
get = kvaf.GetExecutionConfig("urma", "aicpu", "cpu_sgl_write")
# 将 get 放入 default_get/get_resources，或传给 batch_get(execution=get)。
resources = kvaf.ExecutionResources(cores=4, sge_per_wr=4)
# get_resources=[(get, resources)]；sge_per_wr=8 可测 8 SGE。
output_indices = torch.arange(request_length, dtype=torch.int32, device="npu")
```

两种 AccessTable 重载、普通 stream 调用、同步/异步完成以及 Get 图租约共用现有接口。
`output_indices` 在 `miss_count` 指定的有效前缀内给出每个 token 的目标行。若一条 WR 的
首个请求项为 `first`、实际源 SGE 数为 `k`，则该 WR 的 `output_indices[first + j]`
必须等于 `output_indices[first] + j`，其中 `0 <= j < k`。不同 WR 的目标区间可以留空隙或
换顺序，必须互不重叠且落在 output 内。源索引和偏移可随机。`miss_count=0` 不通知 CPU、
不修改 output。请求容量上限为 65536，token 大小固定为 Runtime 配置值。

AICPU 在发布 notify 前检查 WR 内连续性和目标范围，CPU 在提交数据 WRITE 前检查全部 WR
的目标越界与重叠。错误随现有同步、drain 或 status 检查报告。
设备执行错误与原 Scatter 一样使该执行资源失效，须按原 Get 恢复/关闭流程重建。

`ExecutionResources.sge_per_wr` 设置 CPU SGL Write Get 的源 SGE 上限。显式值允许 1..16，
超过设备的 SGE 上限时初始化报错。默认 `0` 选择 `min(16, max_jfs_sge)`。
实际分组大小 `rows_per_wr` 为 `min(SGE 上限, 消息字节上限 / token_bytes)`，除法向下取整。
消息字节上限为 `min(65536, max_msg_size)`。每个源 SGE 搬一个完整 token，一个 token
放不下时初始化报错。先按 worker、chunk、Jetty 分片，再按 `rows_per_wr` 切分，分片尾部
按剩余 token 数生成 WR。配置在执行资源准备时固定，切换 4/8 使用新的 Runtime。
WR 编号依次按 worker、chunk、Jetty 和分片内次序分配。ready、done 和描述读取使用单 SGE。

例如 `cores=1`、`sge_per_wr=4`、`miss_count=32`、`token_bytes=656`，设备消息上限足够时，
4 个 Jetty 各有 8 行，每个 Jetty 有 2 条数据 WR。`output_indices = i + (i // 4) * 4`
会让 8 条 WR 分别写入第 0–3、8–11、…、56–59 行，output 需要至少 60 行。

## 内存和完成

Host 用 direct HCCP 注册整块 output。AICPU 发布的请求头带布局版本和数据 WR 数，
描述区依次存放 `OutputTarget`、`sources[]` 和按缓存行对齐的 `dst_row[]`。CPU 读取头部后，
批量读取三者的有效描述区，校验后导入 output。每个 post 前只刷新该 post 的源 SGE 和
目标 SGE。目标地址为 `output_base + dst_row[wr_id] * token_bytes`，长度由预构建 WR
的 `num_sge` 决定。相同请求数可以复用 WRITE 计划，并在每轮指定不同目的行。

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
CPU profile 的 `descriptor_read_us` 统计 output 描述和两份有效数组的批量读取，
`descriptor_validation_us` 统计布局、源地址、目标范围和重叠校验。两项都包含在
`address_read_us` 内，后者还包含 output segment 导入。`done_us` 包含释放。
带宽分子只计 `miss_count * token_bytes` 的有效 payload。

## 构建和真机验证

CPU 服务、NPU Host/Python 库和 AICPU 设备包须使用匹配版本重建，服务和进程须重启。
CPU Push setup 回包从 300 B 变为 308 B，携带 SGE 上限和消息字节上限。
Scatter 的 V3 设备入口和 launch 参数布局保留，直接输出使用
`KvafAicpuCpuSglWriteExecutionV2`，启动时检查该入口。

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

## 对比 4 SGE 和 8 SGE

```bash
for sge in 4 8; do
  KVAF_CPU_SERVICE_ADDR=127.0.0.1:21111 DEVICE_ID=2 \
  bash tests/run_remote_cpu_batch_get_perf.sh build-npu \
    --backend cpu_sgl_write --sge-per-wr "$sge" --aicpu-workers 4 \
    --execution eager --request-length 4096 --table-capacity 4096 --miss-count 4096 \
    --token-bytes 656 --tokens-per-block 1 --seed 20260903 \
    --source-access-pattern random --output-access-pattern contiguous \
    --no-sort-input-indices --warmup 50 --iterations 1000 \
    --output-dir "reports/cpu-sgl-${sge}sge"
done
```

同一 CPU 服务即可接受两次测试的配置。也可用环境变量
`KVAF_BATCH_GET_PERF_SGE_PER_WR` 设置默认值；显式命令行覆盖环境变量。
单卡 `summary.csv`、配置日志及多卡子进程命令都记录/传递 `sge_per_wr`；0 表示请求自动值，
CPU 日志 `CPU_PUSH_CAP` 中 `max_sge` 是生效的 SGE 上限，`rows_per_wr` 是实际分组大小。
例如配置 8 SGE，每个 token 为 10,000 B，消息上限为 31,000 B 时，日志显示
`max_sge=8` 和 `rows_per_wr=3`。调用方按 `rows_per_wr` 和分片边界安排 `output_indices`。
加 `--kernel-profile` 并用较少迭代可检查 CPU 的 `data_wrs`。
4096×656 B、4 worker、当前 1024 行 chunk/四路 Jetty 划分下，
4 SGE 应为 1024 个数据 WR，8 SGE 应为 512 个；均另有 16 个 ready WR。
