# CPU gather 与 payload READ 分块流水

当前 gather_read 默认使用每 lane 128 KiB CPU chunk，将 CPU gather 与 NIC
payload READ 重叠。保留原有 AIV WQE/SQ 双 bank 流水、整批 staging、每 lane
一个控制 Jetty 和一个数据 Jetty；清单仍完整生成和发布。不加入跨 batch
流水、额外 payload 缓冲或软件在途预算。

## 执行和修改点

1. `host/gather_batch_get.cpp` 在 Runtime 初始化时读取
   `KVAF_AIV_GATHER_CHUNK_BYTES`。默认/空值为 131072；0 恢复整批 ACK 后
   READ；其他值必须是 uint32 范围内的非零 64 B 倍数。诊断 probe 和
   `token_bytes * SGL > 65536` 的 Scalar 回退自动使用 0。
2. `BuildGatherDescriptor` 将 chunk 大小写入已有的 48 B wire header。
   CPU 继续使用已有 `CpuGatherExecutor::ProcessSource`：多线程动态领取 chunk，
   memcpy 和系统 barrier 完成后标记 chunk done，只发布无空洞的连续前缀。
3. `ReadyPrefix::WaitFor` 经控制队列 READ 描述符的 `[8,48)`，取回本轮
   sequence 和 ready_bytes。目的地址使用现有每 lane 64 B scratch 的
   `[16,56)`，初次通知 WRITE 完成后才复用。校验序号、错误位、范围、
   单调性和 chunk 边界；最终不足一个 chunk 的尾部允许 ready=total。
4. `ReadGatherPipeline` 按 lane 原有 SGL 分组推进。第 i 条 WR 的源区间
   结束位置为 `min((i+1)*SGL, lane_rows)*token_bytes`；整个区间 ready 后
   才可提交，不拆改 SGL，也不读取半条 token。已 ready 的 WR 可组成不足
   128 条的 post，当前 UB bank 全部发布后即可继续查下一段 ready。
5. 跨 ready 前缀保留在途 READ。只在物理 WQEBB/WR 容量形成的 wave 末尾
   和整批末尾等待数据 CQ；不逐 chunk 等数据 CQ。尾 WR 的 completion 位
   在发布前设置，已发布 SQ 内容不回写修改。
6. 全部数据 CQ 完成后，仍查询本轮最终 ACK。CPU 的 started ACK、
   ready=total、数据 READ 完成均不能单独允许复用描述符和 staging。
   最终 ACK 确认 CPU helper 已汇合，成功退出后才允许下一次调用复用。

```text
CPU: gather chunk 0 -> ready 0 -> gather chunk 1 -> ready 1 -> ... -> final ACK
NIC:                            READ chunk 0 -> READ chunk 1 -> ... -> data CQ
AIV: notify -> query ready / post eligible READs -> data CQ -> confirm final ACK
```

遇到进度错误、CPU 错误、CQ 错误或超时，停止补发，控制/数据队列均标记
不可复用。部分 wave 可能尚未提交 signaled 尾 WR；不修改已发布 WQE 来补
completion。沿用 Shutdown 的 transport quiesce 和 StopGatherLanes 回收，
重建 Runtime 后再继续。

## ABI 与部署

Gather kernel 符号升级为 `KvafGatherBatchGetAivKernelV4`，args 从 232 B
增至 240 B；loader 拒绝 V1/V2/V3。profile 利用原 padding 增加六个字段，
仍为 192 B。同步重编 NPU Host、Runtime、Python binding 和 gather kernel：

```bash
cmake --build build-npu -j
export PYTHONPATH="$PWD/build-npu/python"
```

CPU 服务代码和 wire ABI 未改；现有包含 chunk executor 的 CPU 服务可以
直接使用。实际加载日志应显示 `symbol=KvafGatherBatchGetAivKernelV4 args_bytes=240`。
修改环境变量后需要新建 Runtime 实例。

## 验证和 A/B 对比

沿用已配置的 CPU 地址、设备和 kernel 环境，先运行完整 payload/guard 回归：

```bash
KVAF_RUN_NPU_TESTS=1 KVAF_AIV_GATHER_CHUNK_BYTES=131072 \
  python -u tests/python/test_aiv_gather_batch_get.py
```

性能脚本顶部保持同一 token、请求数、SGL、核数、输入 seed、输出布局；
固定 CPU 线程、NUMA/绑核和页策略。两轮分别执行：

```bash
KVAF_RUN_NPU_TESTS=1 KVAF_AIV_GATHER_CHUNK_BYTES=0 \
  python -u tests/python/test_aiv_gather_batch_get_perf.py
KVAF_RUN_NPU_TESTS=1 KVAF_AIV_GATHER_CHUNK_BYTES=131072 \
  python -u tests/python/test_aiv_gather_batch_get_perf.py
```

两轮都使用 V4；0 保留原有 AIV 内部流水，仅关闭 CPU chunk 消费。CSV 保存
configured_chunk_bytes，逐 lane 保存实际 pipeline_chunk_bytes。Scalar 回退
和空 lane 的实际值为 0。先比较 profile=off 的端到端 p50/p95/p99 和有效带宽。

## Profile 口径

CSV schema_version 升级为 4。旧的七个 boundary timestamp 保留，但 chunk
模式中 ack_done_cycle 是数据完成后观察到最终 ACK 的时间，因此通常晚于
read_done_cycle。它不是 CPU 实际结束的远端时间戳。

| 字段 | 定义 |
| --- | --- |
| first_ready_cycle | 首次观察到非零连续 ready 前缀的 AIV cycle |
| first_read_submit_cycle | 首个数据 doorbell 发布前的 AIV cycle |
| first_read_ready_bytes | 首次数据提交所依据的连续 ready 前缀快照 |
| ready_queries | 进度查询 READ 的次数，与最终 ACK 查询次数分开 |
| ready_wait_cycles | 等待可提交 WR 的进度查询循环累计耗时，含控制 CQ 等待 |
| pipeline_chunk_bytes | 本 lane 实际采用的 chunk 大小，0 为整批模式 |

整批模式六阶段继续沿用原口径。chunk 模式中：

- wait_cpu = notify_done 到最终 ACK 观察，包含与数据 READ 重叠的时间；
- read = first_read_submit 到 read_done，含后续 ready 等待及 WQE 提交；
- finalize = max(ack_done, read_done) 到 kernel_finish；
- read_build_cycles 累计每段已 ready WR 的构造/发布区间，不包含段间
  ready_wait_cycles；data_cq_wait_cycles 仍仅为物理 wave 尾部剩余等待。

wait_cpu 与 read 不能相加。first_read_ready_bytes 小于该 lane 的总字节数，
表明调度器在仅观察到部分数据 ready 时已经提交；CPU 可能在查询返回到
doorbell 之间继续推进，因此该快照不能单独证明真实硬件上的精确重叠时长。
CPU/NPU 时钟不可直接相减，硬件性能和 DMA 可见性仍需目标双机验证。

## 本地覆盖

- 直接执行生产 kernel 源码的模型按查询逐块复制数据，断言每条 READ 的
  完整源区间已 ready，并验证首次 READ 时后续 chunk 尚未复制。
- 覆盖 64 B/4 KiB/128 KiB chunk、656 B 非对齐 token、SGL 1/6/13、
  coalesced SGE、1/3/8 核、空 lane、重复调用、尾 WR 和 SQ/CQ 环回。
- 大队列验证多个 ready 前缀只消耗物理 wave 尾 CQ；小队列验证容量 drain。
- 覆盖旧/未来序号、进度回退/越界/错位、CPU 错误、进度 CQ 错误、
  数据 CQ 错误/超时、最终 ACK 错误/超时以及部分提交后的复用拒绝。
- CPU executor 的并发消费者检查已发布前缀内容；Host/loader 验证配置捕获、
  profile 校验和旧 ABI 拒绝。无 profile 的模型执行检查不读取计时器。

本地模型不模拟 CANN 编译、设备缓存一致性或 CPU/NIC DDR 带宽争用。
