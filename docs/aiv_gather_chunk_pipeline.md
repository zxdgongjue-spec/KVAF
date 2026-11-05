# CPU gather 与 payload READ 分块流水

当前 gather_read 默认使用每 lane 128 KiB ready 发布粒度，将 CPU gather 与 NIC
payload READ 重叠。保留原有 AIV WQE/SQ 双 bank 流水、整批 staging、每 lane
一个控制 Jetty 和一个数据 Jetty；清单仍完整生成和发布。不加入跨 batch
流水、额外 payload 缓冲或软件在途预算。

## 执行和修改点

1. `host/gather_batch_get.cpp` 在 Runtime 初始化时读取
   `KVAF_AIV_GATHER_CHUNK_BYTES`。默认/空值为 131072；0 恢复整批 ACK 后
   READ；其他值必须是 uint32 范围内的非零 64 B 倍数。诊断 probe 和
   `token_bytes * SGL > 65536` 的 Scalar 回退自动使用 0。
2. `BuildGatherDescriptor` 将 chunk 大小写入已有的 48 B wire header。
   CPU 拷贝任务与 ready 发布粒度分开：任务按线程数均衡分轮，单任务不超过
   256 KiB，线程按编号分担各轮任务，不再竞争全局领取计数器。当前
   4096 x 656 B / 16 线程得到 16 个 164 KiB 任务，与 ready chunk 大小无关。
   每个写线程在跨过全局 ready chunk 边界或到达任务末尾时，执行系统 barrier，
   再更新各自 64 B 隔离的已完成字节数。任务分工不变，但不必等完整任务结束
   才报告进度。例如首线程先复制 `[0,128 KiB)`、barrier、发布 ready=128 KiB，
   再复制该任务剩余的 36 KiB。边界可以落在 token 内，CopyRange 按字节处理。
   coordinator 是唯一 ready 发布者，汇总连续已完成任务及首个未完成任务的
   部分进度后按 chunk 向下取整；不能越过前面的数据空洞。
   全部字节完成时发布实际尾部长度。移除 helper 的公共 ready CAS 和 chunk
   模式重复的线程末尾 barrier。coordinator 等待 helper 时继续推进前缀。
   128 KiB 只是默认值；设置 `KVAF_AIV_GATHER_CHUNK_BYTES=65536` 可使用
   64 KiB 发布粒度。任务大小不变，首前缀更早发布，但 barrier/进度更新更频繁。
3. `ReadyPrefix::BeginQuery/FinishQuery/TryFinishQuery` 经控制队列 READ 描述符的 `[8,48)`，取回本轮
   sequence 和 ready_bytes。目的地址使用现有每 lane 64 B scratch 的
   `[16,56)`，初次通知 WRITE 完成后才复用。校验序号、错误位、范围、
   单调性和 chunk 边界；最终不足一个 chunk 的尾部允许 ready=total。
   两个字段只做一次 scratch 缓存行维护。已知尾项的控制 READ 和 notify
   WRITE 在初次构造时设置 completion/order，不再写 SQ 后修改尾头并重复 Flush。
4. `ReadGatherChunks` 按 lane 原有 SGL 分组推进。第 i 条 WR 的源区间
   结束位置为 `min((i+1)*SGL, lane_rows)*token_bytes`；整个区间 ready 后
   才可提交，不拆改 SGL，也不读取半条 token。已 ready 的 WR 可组成不足
   128 条的 post。首个 post 在一次 VF/MTE3 后立即发布，后续保留两 bank
   的三阶段调度。准备已 ready WQE 时，可提前发起一条后续进度查询；查询
   完成后直接延续 bank 状态，不按每个 ready 快照重启流水。只有确实要
   等待新进度时才先发布本地已准备的 post，保证 NIC 不等本地 bank 收尾。
5. 跨 ready 前缀保留在途 READ。只在物理 WQEBB/WR 容量形成的 wave 末尾
   和整批末尾等待数据 CQ；不逐 chunk 等数据 CQ。尾 WR 的 completion 位
   在发布前设置，已发布 SQ 内容不回写修改。
6. 最后一波数据 post 全部发布后，即查询最终 ACK，与 NIC 在途 READ 重叠；
   随后确认数据 CQ。CPU 的 started ACK、
   ready=total、数据 READ 完成均不能单独允许复用描述符和 staging。
   最终 ACK 确认 CPU helper 已汇合，成功退出后才允许下一次调用复用。

```text
CPU: gather chunk 0 -> ready 0 -> gather chunk 1 -> ready 1 -> ... -> final ACK
NIC:                            READ chunk 0 -> READ chunk 1 -> ... -> data CQ
AIV: notify -> query/lookahead ready + post READs -> confirm ACK while READs run -> data CQ
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

CPU 调度/发布优化修改了 CPU executor，必须重编并重启 CPU 服务：

```bash
cmake --build build-cpu -j
```

本次任务内部的部分进度发布只修改 CPU executor；如果 N 侧已部署上述 V4
流水实现，无需再次重编 NPU。wire ABI、V4 args 和 profile 布局不变；
已有 V4 Host/Python 接口兼容。
实际加载日志应显示 `symbol=KvafGatherBatchGetAivKernelV4 args_bytes=240`。
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

CSV schema_version 仍为 4，保留七个 boundary timestamp。当前优化版先在
READ 在途期间确认 ACK，再确认最后的数据 CQ，因此成功的非空 chunk lane
通常表现为 ack_done_cycle 早于 read_done_cycle；旧版相反。Host/脚本允许
两种顺序，退出前必须同时满足两个完成条件。两者都是 AIV 观察时间，不是
CPU 实际结束或 NIC 实际完成的远端时间戳。

| 字段 | 定义 |
| --- | --- |
| first_ready_cycle | 首次观察到非零连续 ready 前缀的 AIV cycle |
| first_read_submit_cycle | 首个数据 doorbell 发布前的 AIV cycle |
| first_read_ready_bytes | 首次数据提交所依据的连续 ready 前缀快照 |
| ready_queries | 进度查询 READ 的次数，与最终 ACK 查询次数分开 |
| ready_wait_cycles | 进度查询提交、完成检查和前台控制 CQ 等待的累计耗时，不含与 WQE 工作重叠的查询在途间隔 |
| pipeline_chunk_bytes | 本 lane 实际采用的 chunk 大小，0 为整批模式 |

整批模式六阶段继续沿用原口径。chunk 模式中：

- wait_cpu = notify_done 到最终 ACK 观察，与部分数据 READ 重叠；不能当成 CPU memcpy 时间；
- read = first_read_submit 到 read_done，含后续 ready 等待及 WQE 提交；
- finalize = max(ack_done, read_done) 到 kernel_finish；
- read_build_cycles 累计已 ready WR 的构造/发布及本地 bank 收尾区间，不包含
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
- 人为延迟进度 CQ 到后继 VF 之后，验证查询可与构造重叠；人为延迟数据
  CQ 到最终 ACK 确认后，验证 ACK 不再依赖先消费数据 CQ。检查首 doorbell
  只需一次 VF，控制 SQ 每条已知尾 WR 只清一次，ready 每次只刷新一条 scratch 行。
- 覆盖旧/未来序号、进度回退/越界/错位、CPU 错误、进度 CQ 错误、
  数据 CQ 错误/超时、最终 ACK 错误/超时以及部分提交后的复用拒绝。
- CPU executor 的并发消费者检查已发布前缀内容；Host/loader 验证配置捕获、
  profile 校验和旧 ABI 拒绝。无 profile 的模型执行检查不读取计时器。
- 16 线程、656 B/4096 B token 的均衡任务检查覆盖多个 ready 粒度，包括
  ready chunk 大于整批数据的情况，验证调大发布粒度不会丢失拷贝并行度。
- 在 4096 x 656 B / 16 线程场景，测试暂停首任务在首个 64 KiB/128 KiB
  前缀之后的复制，验证 ready 已发布、前缀内容正确且任务尾部尚未修改；
  覆盖开启/关闭 profile。

本地模型不模拟 CANN 编译、设备缓存一致性或 CPU/NIC DDR 带宽争用。
