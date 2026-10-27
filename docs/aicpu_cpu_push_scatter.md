# AICPU BatchGet 的 CPU 回推路径

需要由 CPU 将离散 DDR 数据直接送入连续 HBM、再由 AICPU 写入调用方输出时，
在现有 `NpuRuntime` 构造参数中设置：

```python
runtime = kvaf.NpuRuntime(
    batch_get_kernel_object_path="",
    batch_get_token_bytes=656,
    batch_get_tokens_per_block=2,
    device_id=7,
    npu_core_count=4,
    aicpu_urma_kernel_json_path=kernel_json,
    aicpu_batch_get_mode="cpu_push_scatter",
    urma_device="udmac0d1e5",  # BatchPut 的 UMDK 端点，按部署环境调整。
    urma_eid_index=0,
)
runtime.link_to("127.0.0.1:28990")
runtime.initialize(ddr_bytes=656 * 2 * 8192, slot_size=656 * 2)
```

其余 `batch_put`、AccessTable 和 `batch_get` 调用保持现有方式。
`aicpu_batch_get_mode` 默认为 `gather_read`，两个值均可使用原有两种 Tensor 重载。
CPU push 自动发现单个 NPU EID，也可通过 `npu_eids=[eid]` 指定一个 EID。
自动发现的候选按现有 Runtime 初始化流程尝试；多 EID 条带和 AIV 自适应调优不用于此路径。
Get 使用 Host direct HCCP/RA；Put 继续使用既有 AICPU UMDK session。
CPU 使用四路 Jetty 和最多四个 OpenMP 线程；`npu_core_count` / `--workers` 控制 AICPU worker 数。

## 构建并验证

两端代码、设备 SO 和 kernel JSON 一起更新，重启 CPU 服务及 NPU 进程：

```bash
./build.sh cpu
./build.sh aicpu-urma --cpu-push --install-kernel
export PYTHONPATH="$PWD/build-aicpu-urma/python:$PYTHONPATH"
```

设备 `memcpy_sp` 需要 securec 头文件和 AArch64 动态库，构建会查找 CANN 内的依赖并打包。
CPU 构建链接 OpenMP，并启用 `-O2`、保留调试信息。
`--cpu-push` 对应 CMake 的 `KVAF_ENABLE_AICPU_CPU_PUSH=ON`，默认关闭。
原有 gather-only 构建不查找或链接 securec/direct HCCP。
布局不同时设置 `KVAF_AICPU_SECUREC_INCLUDE_DIR` 和 `KVAF_AICPU_SECUREC_LIBRARY`。
CPU 服务启动方法见 [gather_read 使用说明](aicpu_gather_read.md)。

```bash
python3 access_operator/demo/aicpu_batch_get/python/test_batch_get.py \
  --device-id 7 --peer 127.0.0.1:28990 \
  --urma-device udmac0d1e5 --eid-index 0 \
  --modes gather_read cpu_push_scatter --workers 1 2 4 8 \
  --blocks 4096 --token-bytes 656 --tokens-per-block 2 \
  --output-row-step 1 --warmup 10 --iterations 100
```

`--output-row-step 1` 检查随机排列的相邻输出行；改为 2 检查输出间隔。
脚本每轮重新写入源数据和输入元数据，比较整个输出，并用后继 NPU 算子消费输出。
预检包括部分/空 `miss_count`、两种 Tensor 重载和异步调用。
需要阶段耗时时增加 `--profile-rounds 100`；采样轮次不计入正常带宽。

## 全量性能扫描

CPU 服务启动后，在加载 CANN 环境的终端运行：

```bash
./tests/run_aicpu_cpu_push_scatter_sweep.sh build-aicpu-urma \
  --device-id 7 --peer 127.0.0.1:28990 \
  --urma-device udmac0d1e5 --eid-index 0
```

默认与 GatherRead sweep 一样扫描 252 组：token 为 128、256、512、656、1024、2048、4096 B，
miss_count 为 32、64、128、256、512、1024、2048、4096、8192，worker 为 1、2、4、5。
每组预热 10 轮、测量 100 轮，均开启 profile。

源数据池按 100 GiB DDR 预算初始化一次，每块默认 128 个 token；CPU push 的 staging 和地址清单在 HBM，
不从 DDR 预算扣除 GatherRead 工作区。每轮有放回随机选择源 token，随机选择互不重复的输出行，
输入和输出行数为 `max(32768, miss_count)`，逐轮校验整个输出及未写入行。
`--table-capacity` 可缩小源池；与 GatherRead 对比时可指定相同容量。

主指标为 `device_p50_us` 和 `p50_GBps`，带宽按有效字节数除以算子 P50 计算。
各阶段每轮取 worker 最大值，再对测量轮次求均值：`lookup_table_us`、`notify_cpu_us`、
`push_total_us`、`ready_wait_us`、`scatter_us`、`done_wait_us`。
ready 等待和 scatter 包含在 push 总时间内，各阶段最大值不能相加。
CPU 自身的 READ/WRITE 分段耗时仍在服务端 `CPU_PUSH_PROFILE` 日志中。

缩小扫描范围示例：

```bash
KVAF_AICPU_TOKEN_BYTES="656" \
KVAF_AICPU_MISS_COUNTS="32 4096" \
KVAF_AICPU_WORKERS="1 2 4 5" \
./tests/run_aicpu_cpu_push_scatter_sweep.sh build-aicpu-urma \
  --device-id 7 --peer 127.0.0.1:28990 \
  --urma-device udmac0d1e5 --eid-index 0 \
  --table-capacity 4096 --warmup 3 --iterations 20
```

每组 `.log` 和 `summary.csv` 默认写入 `reports/aicpu-cpu-push-scatter/<时间戳>`。
`KVAF_AICPU_SWEEP_LOG_DIR` 可指定一个新目录；任一组失败后停止并保留已完成结果。
单组入口为 `python/benchmark_cpu_push_scatter.py`，参数与 GatherRead 相同，另支持 `--npu-eid`。

## 请求与数据流

一次调用处理 `int32 miss_count` 指定的有效前缀，最多 65536 项。
每项读取 `addresses[indices[i]] + offsets[i]` 的 `token_bytes` 字节，
目标为 `output + output_indices[i] * token_bytes`。
调用方继续负责有效 key、token 布局、输出映射及这些资源在操作完成前的生命周期。

初始化时，Host 通过 RA 注册固定 HBM workspace，创建四个 TargetChannel，
将 segment、Jetty key 和 TP 信息发给源 DDR 所属 CPU Runtime。
CPU 建立专用回写队列，返回对端信息；Host 完成导入后才激活驻留 CPU worker。
激活时先同步 READ 请求控制字并 WRITE 初始完成字，确认实际双向 RMA 路径可用。
这套控制协商通过现有 PeerLink 完成，无需新增 NPU listener。

每轮过程如下：

1. Host 在调用方 stream 清零本次 AICPU coordination，再启动配置数量的实例。
2. worker 0 将有效源地址写入连续的 `uint64` 数组，在 HBM 发布请求头和 request sequence。
3. CPU 通过 URMA READ 整块读取地址数组，OpenMP 按 Jetty 并行更新预构建 SGL 的源地址。
4. CPU 直接从已注册 DDR 源 WRITE 到 packed HBM staging，不打包 payload 到 CPU 临时区。
5. 按 worker 的 staging 区间划分 1024 行 chunk，各 Jetty 在其数据之后强序 WRITE ready。
6. AICPU 取四个 Jetty 的共同就绪前缀，将新就绪范围 scatter 到 `output_indices`。
7. CPU 收齐数据链尾 CQE 后发布 done；全部 scatter 与 done 均完成后 kernel 结束。

ready 用高 32 位 sequence、低 32 位 chunk 数编码；done 在同一个 64 位字内编码 sequence 和结果。
请求序号从 Host 递增，达到 uint32 上限后要求重建 session。
空前缀不通知 CPU、不修改输出；CPU 允许此类调用造成的序号间隔。
当前实现针对普通 stream enqueue，Host 固定序号不能用于 graph replay。

CPU 数据面沿用 `hbm-sgl-probe` V24 的 prepared WRITE、indexed CQ 和轮转 CQ 扫描。
每个 chunk 按连续行分到四个 Jetty；请求数量不变时复用数据/ready 链，每轮只更新源地址和序号。
OpenMP 并行更新与提交四路队列，协调线程收齐数据链尾 CQE 后提交预构建的 done 链。
每个 Jetty 的 SQ 深度为 8192，doorbell 为 32 WR，CQ batch 为 8，每 32 次扫描检查超时。
超出单 wave 容量的请求分 wave 执行，完成前保留全部 WR/SGL；每个 ready 使用独立的注册源字。
Runtime 按每次 BatchGet 启动 kernel，接收本轮动态源地址和 `output_indices`。

## 内存、缓存和完成

HBM workspace 包含独立缓存行上的请求头、notify、done、各 worker 的四路 ready，
随后是源清单和连续 staging。staging 容量为 `65536 * token_bytes`，
另有 512 KiB 的源地址数组；总 payload 必须能用 uint32 表示。
此路径不预留远端 DDR gather staging，也不注册调用方 output 为 URMA 目标。

此路径使用部署环境提供的 DMA 可见性，不执行手动缓存行刷新。
算子只遍历输入生成源清单和执行 scatter，不逐项校验 Tensor 内容或校验 payload。
调用方须提供有效的索引、偏移、源地址和输出行；保留请求容量和有效前缀长度检查。

一次执行完成前不复用 staging。异步返回表示已经入队；下次 Get/Put 会先 Drain，
检查前一轮执行状态。`synchronize=True` 在本次调用内完成等待和状态检查。
Shutdown 先停止 CPU worker 并销毁其 DMA 队列，再等待设备执行、释放 HBM。
传输或 stream 失败会阻止执行资源复用；关闭失败时保留资源以便重试。

## Profile

原有 profile 字段保留，增加 `scatter_us` 和 `output_cache_us`。
push 路径的 `read_us` 是消费 ready 并执行 scatter 的区间，`ack_us` 是最终 done 等待；
`source_write_us` 为零，因为本轮源清单由 CPU READ 获取。
缓存维护相关字段为零。设备 worker 的区间重叠，不能相加作为端到端时长。

`--profile-rounds` 同时开启 CPU 分段计时。CPU 服务在每个采样请求完成后输出一行
`CPU_PUSH_PROFILE`，包含 profile 预热轮；普通带宽轮次不采集或打印这些计时。
`header_read_us`、`address_read_us` 分别是 READ 请求头和整块源地址数组的耗时；
`write_us` 包含 `build_us`、`post_us`
和 `cq_wait_us`，分别表示计划构建/源地址更新及 OpenMP 调度、并行提交、提交后的 CQ 等待。
`submit_to_cqe_us` 从首次提交阶段到最后一个数据链尾 CQE，包含提交期间与 DMA 的重叠；
`done_us` 是最终 done WRITE 及其 CQ 等待。计时从 CPU 看到新请求后开始，不含空闲轮询。
`data_wrs`、`ready_wrs`、`post_calls`、`waves` 给出数据 WR、ready WR、provider 提交及 wave 数；
`plan_reused=1` 表示复用了预构建链；`cpu_threads` 是实际 OpenMP 线程数。
`tail_cqes` 统计数据链尾 CQE，不含 done。4096 × 656 B、4 worker、max_sge=13 时，
预期为 320 个数据 WR、16 个 ready WR、4 个数据尾 CQE，随后一个 done 尾 CQE。
`optimized=1` 表示 CPU transport 编译时已启用优化。

主要文件对应：`cpu_push_host.cc` 管 Host 资源，`cpu_push_get_worker.cc` 管 CPU 请求循环，
`cpu_push_get_transport.cc` 管 URMA 数据链，`cpu_push_scatter.cc` 管设备消费与本地复制。
