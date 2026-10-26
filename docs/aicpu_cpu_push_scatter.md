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

## 构建并验证

两端代码、设备 SO 和 kernel JSON 一起更新，重启 CPU 服务及 NPU 进程：

```bash
./build.sh cpu
./build.sh aicpu-urma --cpu-push --install-kernel
export PYTHONPATH="$PWD/build-aicpu-urma/python:$PYTHONPATH"
```

设备 `memcpy_sp` 需要 securec 头文件和 AArch64 动态库，构建会查找 CANN 内的依赖并打包。
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
3. CPU 通过 URMA READ 整块读取地址数组，直接将其中的地址填入本轮 WRITE SGL。
4. CPU 直接从已注册 DDR 源 WRITE 到 packed HBM staging，不打包 payload 到 CPU 临时区。
5. 按 worker 的 staging 区间划分 1024 行 chunk，各 Jetty 在其数据之后强序 WRITE ready。
6. AICPU 取四个 Jetty 的共同就绪前缀，将新就绪范围 scatter 到 `output_indices`。
7. CPU 收齐数据链尾 CQE 后发布 done；全部 scatter 与 done 均完成后 kernel 结束。

ready 用高 32 位 sequence、低 32 位 chunk 数编码；done 在同一个 64 位字内编码 sequence 和结果。
请求序号从 Host 递增，达到 uint32 上限后要求重建 session。
空前缀不通知 CPU、不修改输出；CPU 允许此类调用造成的序号间隔。
当前实现针对普通 stream enqueue，Host 固定序号不能用于 graph replay。

CPU 每个提交 wave 至多保留 512 个 WR 槽位，并按设备 SGE/message 上限拆分请求，
只在有序尾 CQE 后复用 WR/SGL。多个 chunk 合并提交，仅在 WR 槽位用满或整批结束时等待 CQE。
各 chunk 的 ready 使用独立、已注册的源字，提交后保持不变；WRITE 与 AICPU scatter 可重叠。

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

主要文件对应：`cpu_push_host.cc` 管 Host 资源，`cpu_push_get_worker.cc` 管 CPU 请求循环，
`cpu_push_get_transport.cc` 管 URMA 数据链，`cpu_push_scatter.cc` 管设备消费与本地复制。
