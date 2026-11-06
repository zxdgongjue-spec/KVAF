# Gather 分片进度流水与调用快路径

当前 gather 默认启用独立 shard 进度消费。改动仅使用 gather 的 CPU memcpy、
READ SQ/CQ、sequence 和最终 ACK 生命周期；未引入 scatter 的 CPU-push 协议、
payload WRITE 链、done 状态或 workspace 布局。

## 调用快路径

Python `NpuRuntime(..., aiv_batch_get_mode="gather_read")` 默认
`trusted_gather_inputs=True`。每次读取七个当前 data_ptr、indices/address 容量和
output 首维，按配置推导其余大小；使用缓存的方法名和 CPython method vectorcall。
缓存 current_stream callable，但每次调用都重新解析当前 stream/handle。
其他模式维持 checked 通路。需要完整 Python/Runtime/Host 布局检查时，设置
`trusted_gather_inputs=False`。

trusted 调用方保证连续 NPU Tensor、int32 请求数组等长、int32[1] miss_count、
64 位地址表及等长 uint8 valid、output 行字节数匹配 token_bytes，且容量和生命周期
合法。这条路径仅用于 eager，略过 capture 查询；它不支持 graph capture/replay。
C++ 默认仍 checked，显式 `BatchGetExecutionOptions::trusted_npu_request=true`
可使用同一合同。当前指针、容量和 stream 不跨调用缓存。

同流异步提交、切流时的 fence、状态检查开关、MR/entity 缓存和失败后资源保留，
都沿用 gather 原有行为。没有重新加入逐批 memset、普通异步 D2H 或逐次 Host Drain。

## GDR3 分片进度

每 lane 的逻辑 READ 数为 `ceil(requests / SGL)`，按完整 WR 均衡分成最多 64 个
连续 shard。CPU 各线程静态轮转处理 shard，无需全局领取计数器或 coordinator
连续前缀汇总。每个 shard 的一个线程在复制完成并执行系统 barrier 后，直接发布
本 shard 的 generation-tagged 完成 WR 数。每次发布覆盖该 shard 的连续局部前缀，
不会等待更前面的其他 shard。

chunk 参数控制 shard 内部的发布间隔，按完整 WR 向下取整且至少一个 WR；到达
shard 尾部立即发布。小于 chunk 的 shard 不必等待凑齐 chunk。
例如 4096×656 B、SGL=6：683 个 WR 分成 64 片，每片 10～11 WR，约 38～42 KiB，
不再由一个线程先复制 128 KiB 才产生第一个可用前缀。

wire 使用 gather 专有 `GDR3` magic：48 B header 与 16 B span 的偏移保持不变，
header 原 `ready_bytes` 字在 GDR3 下为不可变的 READ 分组大小 SGL。
`align_up(48 + rows*16, 64)` 处增加 512 B trailer，含 64 个 8 B progress word：
高 32 位 sequence，低 32 位该 shard 已完成的 WR 数。每个 word 只有一个 CPU
写者；AIV 不要求不同 word 来自同一时刻。

AIV 用一个控制 wave 批量读取最终 ACK 和进度向量，正常 SQ 下两条 READ、
一次 doorbell、一个有序尾 CQ。小 SQ 仍遵守其物理容量限制。
检查当前 generation 的进度单调性和范围，忽略旧 generation、拒绝未来 generation，
并独立检查 CPU 错误 ACK。

调度器优先选择任何已有未消费 ready WR 的 shard，记录每片消费位置，保证每个
源区间只 READ 一次。没有可用进度时，先将候选 WQE 构造到 UB，重叠控制查询；
若查询返回其他 shard ready，可以替换候选。未确认 ready 的 WQE 不写 SQ，
不敲 doorbell，也不在已发布 PI 中留下空洞。

后续 bank 的 VF 与上一 bank 的 MTE3 重叠；已知 ready 数据可继续提交，控制查询
留在途中。相邻且连续已就绪的 shard 合并成最多 128 WR 的 post，避免强制每片
单独启动 VF/doorbell。跨 shard 保留 bank 及数据 READ 在途状态。首批立即发布；只有物理
WR/WQEBB 容量 wave 尾和整批尾等待数据 CQ。
整批最后发布后，在 READ 仍在途时确认最终 ACK，再确认数据 CQ；两者同时成功
才允许复用 descriptor/staging。任何错误继续 poison 两队列并由 Shutdown quiesce。

## ABI、构建与对照

kernel 符号升级为 `KvafGatherBatchGetAivKernelV5`；args 仍为 240 B，末尾原
reserved 改为 `pipeline_shards`。profile 仍为 192 B。loader 拒绝 V1～V4。
远端每 lane descriptor 预留增加 512 B，HBM 本地 scratch 为每 lane 576 B。
必须配套重建 NPU Host/Runtime/Python/kernel 和 CPU 服务，并重启 CPU 服务。

```bash
cmake --build build-cpu -j
# 使用原参数重启新 CPU 服务。
cmake --build build-npu -j
export PYTHONPATH="$PWD/build-npu/python"
export KVAF_BATCH_GET_KERNEL="$PWD/build-npu/bin/kvaf_gather_batch_get_kernels.o"
KVAF_AIV_GATHER_SHARD_PIPELINE=1 python -u tests/python/test_aiv_gather_batch_get.py
```

同一份 V5 二进制可用以下配置对照，环境变量在初始化时捕获：

| 设置 | 实际消费方式 |
| --- | --- |
| `KVAF_AIV_GATHER_CHUNK_BYTES=0` | 原整批 ACK 后 READ，GDR2 |
| chunk 非零，`KVAF_AIV_GATHER_SHARD_PIPELINE=0` | 原连续 ready 前缀流水，GDR2 |
| chunk 非零，`KVAF_AIV_GATHER_SHARD_PIPELINE=1`（默认） | 独立 shard ready 优先消费，GDR3 |

诊断 probe 和 `token_bytes*SGL > 65536` 的 Scalar 回退自动关闭两种流水。
SGL=9/13 的实际硬件支持范围沿用现有约束，本改动不扩展 READ 的硬件能力。

```bash
# 保持相同 token、请求数、核数、CPU 线程、NUMA、SGL 和输入/输出布局。
KVAF_AIV_GATHER_SHARD_PIPELINE=0 python -u tests/python/test_aiv_gather_batch_get_perf.py
KVAF_AIV_GATHER_SHARD_PIPELINE=1 python -u tests/python/test_aiv_gather_batch_get_perf.py
# 分别在相同流水模式下比较 checked / trusted 调用。
KVAF_GATHER_TRUSTED_INPUTS=0 python -u tests/python/test_aiv_gather_batch_get_perf.py
KVAF_GATHER_TRUSTED_INPUTS=1 python -u tests/python/test_aiv_gather_batch_get_perf.py
```

## 计时与验证

CSV schema 为 5，记录 configured_shard_pipeline、trusted_gather_inputs 和每 lane
实际 pipeline_shards。GDR3 的 `first_read_ready_bytes` 是首次提交所依据的快照中
各 shard 已就绪有效字节的总和，**不是全局连续前缀**。其余 boundary timestamp
和原重叠计时规则保持不变，不能将 wait_cpu 与 read 相加。

开启现有 gather profile 时，Host 另输出 `call_*_ns`（CSV 为 `call_*_us`），覆盖
binding、Tensor metadata、stream、Runtime validation、operator、锁等待、capture
查询、prior Drain、Input、output acquire、launch、状态/profile D2H 提交、同步与
校验。父子范围相互包含，不能全部相加；`call_stream_sync` 包含设备执行。
未开启 gather profile 时这些计时器不读取时钟。

本地检查包含：实际 CPU executor 的并发消费者、暂停首 shard 时后片可完成、
旧 generation 和 changing payload、生产 kernel 的逆序 shard producer、每条 READ
范围就绪及每行仅消费一次、跨片 MTE3/READ 在途、进度 CQ 延迟至 VF 后、小 SQ、
尾项/空 lane/重复调用、coalesced SGE、ACK/CQ 错误/超时及 poison 后拒绝复用。
Python native helper 测试检查每轮变化的指针/容量/stream 和 checked 错误类型；
Host/Runtime 测试检查 trusted 合同、配置快照及原生命周期。

这些模型不模拟 CANN 代码生成、DMA 缓存一致性和 CPU/NIC 争用。实际性能以目标
双机完整 output/guard 正确后、profile 关闭的 E2E p50/p95/p99 为准。

2026-11-06 本地验证：CMake 编译（含 fake ACL Host 编译检查）通过；CTest 的
28 个非硬件测试通过，2 个 NPU 硬件入口按环境跳过。CPU executor 与生产 kernel
模型分别通过 AddressSanitizer/UBSan；Python binding 全文件通过语法编译，三个
原生 metadata/stream helper 用例通过。未进行目标 CANN 编译或双机性能采样。
