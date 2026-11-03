# 共享 DDR 与 CPU Pull

同一个 `NpuRuntime(enable_cpu_offload=True)` 管理普通 BatchPut/BatchGet、统一 `batch_cpu_pull()` 和显式 CPU Pull 事务。它们共享 RuntimeCore、MetaManager、DDR allocator 和 DDR region；数据面连接各自独立。

源由 `CpuPullHbmSource` 注册，可位于 Runtime 本卡或另一张卡。源端不分配第二个 DDR 池，只绑定读取端提供的 DDR ticket。完整应用职责及接口见 [Connector 设计](connector_pull_api_design.md)。本文沿用验收脚本的 A=读取端、B=生产端。

`runtime.offload_region()` 返回 version=2 的生产端 DDR ticket，仅含 `version/address/region_id/epoch/token/eid`。DDR 布局仍保存在读取端，应用从 `runtime.remote_region` 读取目的基址等信息。ReadyToken 只带源和池身份、sequence、mr_id 及本批 mr_length，不再携带完整 MR 清单。

`initialized` 仅表示本卡基础初始化，CPU Pull 还要求 `ddr_initialized=True`。
旧 `link_to(...) → initialize(ddr_bytes, slot_size)` 保留为组合入口。
`shutdown_ddr()` 在任务和 CPU 源租约退役后释放 DDR 及其通信资源，保留独立 NPU Pull 链路和 CPU 目标；失败可原地重试，不重发 READ。
随后可重新 initialize_ddr；旧 PutHandle 不适用于新的 DDR core。纯 NPU Pull 读取端只需无参数 initialize 和 listen_hbm，生产端 register(receiver=...) 自动建链。

## 跨卡调用

```python
# A 初始化并将 ticket 通过应用 Connector 发送给 B。
runtime.initialize()
runtime.link_to(cpu_address)
runtime.initialize_ddr(ddr_bytes=8 * 131072, slot_size=131072)
ticket = runtime.offload_region()

# B 注册并完成与 CPU 的建链；成功返回 None，无需向 A 单独发送 desc。
source = kvaf.CpuPullHbmSource(device_id=device_b)
source.register(ticket, hbm_b, stream=producer_stream)

# B 每批生产完成后发布身份，应用将 ready 与业务元数据发送给 A。
producer_stream.synchronize()
ready = source.ready()

# A 计算偏移并提交；此处一个源段对应一个目标槽位。
lengths = torch.tensor([656, 100001], dtype=torch.int64, device="cpu")
src = torch.tensor([7, 100003], dtype=torch.int64, device="cpu")
put = runtime.prepare_put([1, 3], lengths)
ddr_base = runtime.remote_region["base"]
dst_offsets = torch.tensor([e.address - ddr_base for e in put.entries], dtype=torch.int64, device="cpu")
runtime.submit_cpu_pull(ready, put, src, dst_offsets, lengths)
runtime.wait_cpu_pull(put)
runtime.commit_put(put)

# A 应用发送完成消息；B 确认后可覆写下一批。
source.release_batch(ready)
```

CPU 源偏移相对于注册 HBM 的逻辑起点；目的在提取 PreparePut 地址时计算 `entry.address - runtime.remote_region["base"] + 槽内偏移`。Submit 直接接收相对偏移，CPU 加回同一 DDR 池基地址执行 READ；便利入口内部完成偏移提取。提交范围必须精确覆盖本次 PreparePut 的有效数据范围，支持拆分/重排，禁止空洞、重叠或跨槽。与普通 BatchPut 使用同一个 MetaManager，无需 reserve_raw_ddr。CPU 服务保存源连接和 MR；Runtime 使用 ready.source 校验 DDR 归属并提交，由 CPU 核对选中 MR 的真实注册信息，不需要应用保存源句柄或预先调用 attach。

## 统一便利入口

初始化注册和 Ready 流程与上面相同。本卡与非本卡都可将 Prepare/Submit/Wait/Commit 简化为：

```python
runtime.batch_cpu_pull(keys, ready, src_offsets, lengths, timeout=30000)
```

每个 key 对应一个源段，目标在内部提取 PreparePut 返回的槽位起点时减去 DDR base，以相对偏移提交。
默认成功返回 None 表示数据和元数据提交均完成；内部 Put 不暴露给应用。
可选 `profile=True` 时成功返回含 Host 阶段时间的 profile 字典。bench 使用此统一入口，
默认保留阶段统计，`--no-profile` 关闭内部计时；数据校验仍位于调用总计时之外。
生产同步、Ready、Release/Cancel 和源注销仍由生产端管理；支持 Ready 选择预注册的多个 MR。
同参数再次调用是新操作，未决恢复使用 `runtime.resolve_cpu_pull(timeout=30000)`。

独立 OffloadRuntime、旧类别名、带范围 Ready、offload/offload_from、旧恢复入口以及
直接传本卡 tensor/stream 的 batch_cpu_pull 重载已删除。当前保留统一便利入口及显式事务接口。

## 建链与协议

Runtime 先分配共享 DDR，固定 CPU candidate。源注册在同一池上探测 CPU↔源卡 EID，使用 scratch READ，不覆盖缓存槽位。NPU↔NPU raw READ 的 fullmesh 链路独立配置。

CPU Pull 控制协议 KVRP v6，支持一个源连接下多个 HBM MR；NPU Pull 控制协议为 KVHR v7。使用 CPU Pull 的服务和应用需同时重编译重启；旧协议拒绝连接。KVCP 的 DDR 分配协议保持不变，独立 KVOF 服务已删除。多 MR 接口和验收命令见 [CPU Pull 使用说明](cpu_pull_offload_usage.md)。

同一共享池一次只允许一个未完成或未确认的 Pull；每批最多 4096 spans，CPU 内部四 Jetty 并行。服务保留 source 的关闭记录用于确认重试，源注册次数（包括失败候选）上限为 4096，推荐长驻注册复用。

## CPU Pull 性能验证

CPU 数据面参考 `aicpu-urma@8bc5a2a` 的批量 WRITE 提交方式：每次最多链接 32 个 WQE，
先向所有活跃 Jetty 提交，再轮询它们的 CQ。每个 Jetty 每轮最多 1024 个 READ（同时受返回的
SQ depth 限制），只让本轮尾部 WQE 请求 CQE；comp_order 和尾部 READ fence 保证前面的
READ 完成后才能正常确认该尾部。所有活跃 Jetty 尾部均成功才允许 CommitPut 和 B 复用源。
错误 CQE（包括非尾部 WQE 的错误）、部分提交失败、异常 CQE、超时均隔离整个传输，保留
所有队列、MR、SGE/WR 存储和 DDR lease，不重发 READ。

例如 4096 个不超过单条 READ 上限的离散区间：仍需 4096 个 READ WQE，通常为 128 次
post 调用、4 个成功尾部 CQE。更大 span 会拆成更多 WQE，超过本轮容量后分轮执行。
这里不使用 WRITE 的本地多 SGE gather：远端离散地址无法由一个源 SGE 描述。

CPU 服务使用 Release 构建，数据传输固定使用四 Jetty 批量模式。PROFILE 环境变量在 Open
时读取，用于共享池 CPU Pull。串行对照模式已移除；启动连通性探测仍保留
queue 0 上的单条 READ，以确认失败候选能安全释放后再回退。

```bash
cmake -S . -B build-cpu-shared-ddr -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu-shared-ddr -j

# 四个 Jetty，多 WQE 在途、每个 Jetty 每轮一个尾部 CQE。
KVAF_CPU_PULL_PROFILE=1 \
  ./build-cpu-shared-ddr/runtime/kvaf_cpu_service --listen 0.0.0.0 --port 28999
```

上面的 CMake 命令用于已有构建目录，保留其 URMA 等配置。NPU 侧使用当前 Python 扩展：

```bash
python tests/python/bench_remote_cpu_offload.py \
  --cpu 127.0.0.1:28999 --device-a 6 --device-b 7 --cpu-candidate 0 \
  --blocks 4096 --block-bytes 656 --warmup 10 --iterations 100
```

也可用 `--block-bytes 4096`、`65536`、`100001` 比较大小块和拆分后的表现。
B 默认自动选择经过 scratch READ 校验的 EID，也可以用 `--npu-eid-b` 固定已验证的 CPU↔B EID；
此测试不使用 A↔B fullmesh 或 AIV 算子。HBM 长驻注册，每轮改变内容并校验完整有效载荷的
checksum；B 收到提交确认后才写下一轮。

`CPU_PULL_PROFILE` 显示 build/post/cq_wait/total_us、READ WQE 数、post 次数、尾部 CQE 数、
waves、活跃 Jetty 数和 plan_reused。该 total_us 包含描述符准备、提交和等待，**不包含**
服务端 checksum、控制 RPC、线程创建或 Meta 提交。`CPU_PULL_BENCH_PASS` 的 Python 时间
计量 `prepare_put()` 至提交完成（包括目的数组组织），包含控制 RPC、线程创建和 Meta 提交；排除数据校验、注册、
生产者写入/Ready 和 Free，输出 `cpu_checksum_included=0`。每轮计时结束后显式调用
`last_offload_checksum()`，服务端此时才计算完整 payload 的校验和；与预期值一致后才 Free
并通知 B 复用。`verification_mean_us` 单独统计校验 RPC 和比较的耗时，不计入带宽分母。

`last_offload_checksum()` 读取当前 DDR，不返回历史缓存。调用方必须保留本批目标槽位且
禁止覆写，直到校验返回；应在 B 的下一次 Ready 或 source detach 之前调用。校验 RPC 丢失
可以重试，但不会重发 READ，也不会撤销已经完成的 Meta 提交。运行中/未决/已取消/已关闭
批次拒绝校验；同一池另有运行中或未决的 CPU Pull 时也拒绝。启动 scratch 探测的校验保留。

两种计时指标应分别比较；不要将 Python effective_GBps 当作纯 READ 带宽。每轮校验仍会
占用实际运行时间，只是与传输计时分开。诊断后将 PROFILE 关闭再测整体性能，可排除日志影响。

### 2026-10-31 历史记录：1024 个 82 KiB 块

此记录使用当时的旧便利接口；原始日志和指标保留。当前脚本使用统一 batch_cpu_pull 封装，指标名为 `python_batch_cpu_pull_to_commit`，需重新测量，不能将历史数字标为新接口成绩。

用户硬件实测通过。删除串行对照、拆分校验计时后的版本为
`codex/cpu-pull-hbm-reference` / `7fc988571e5ddf99f7ba4b9fecd38673b6fab900`。
下列日期为记录日期；测试输出未打印运行端 Git SHA 或构建参数。
原始依据保存在 [客户端命令与输出](benchmarks/2026-10-31-cpu-pull-82mib/client.log)
和 [CPU 服务端日志节选](benchmarks/2026-10-31-cpu-pull-82mib/cpu-profile.log)。

```bash
python tests/python/bench_remote_cpu_offload.py \
  --cpu 127.0.0.1:28999 --device-a 6 --device-b 7 \
  --blocks 1024 --block-bytes 83968 --warmup 10 --iterations 100
```

- A 为 device 6，B 为 device 7；CPU 自动选择 candidate 0，EID 为
  `00000000003f020000100000df080b00`。
- B 的 CLOS EID `00000000003f030000100000df00eb00` 探测失败后，回退至
  `000000000000020000100000df00e100`，scratch READ 校验通过。该过程不计入批次传输时间。
- 每轮 1024 个离散源块，每块 83,968 字节（82 KiB），有效载荷合计
  85,983,232 字节（82 MiB）。包含源块间间隔的 HBM 缓冲为 86,051,840 字节。
- 预热 10 轮、计时 100 轮；每轮改变输入，完整校验本批有效数据后才释放 DDR 槽位并通知 B 复用。
  本测试不调用 AIV BatchGet、BatchPut 或 A 直接读 B HBM 的路径。

| 指标 | 结果 | 统计范围 |
|---|---:|---|
| `offload_from()` 至提交完成平均耗时 | 2,607.1 µs | 100 轮，不含校验 |
| P50 / P95 / P99 | 2,580.7 / 2,627.6 / 3,662.3 µs | 100 轮 |
| Python `effective_GBps` | 32.981 GB/s | 有效载荷除以平均调用耗时 |
| `verification_mean_us` | 150,477.4 µs | 100 轮，校验 RPC 与结果比较单独计时 |
| CPU 数据面 `total_us` 平均值 | 1,782.8 µs | 日志节选中的 19 轮，sequence 92–110 |
| CPU 数据面中位数 / 最小值 / 最大值 | 1,726.5 / 1,721.9 / 2,793.7 µs | 同上，包含抖动样本 |
| CPU 数据面有效带宽 | 48.23 GB/s | 85,983,232 / 1,782.8158 / 1000 |
| `build_us` / `post_us` / `cq_wait_us` 平均值 | 4.9 / 77.1 / 1,697.4 µs | 同上 19 轮 |

GB/s 使用十进制单位。CPU 数据面时间包含描述符准备、提交和 CQ 等待；Python 时间另外包含
控制 RPC、工作线程创建和 Meta 提交。CPU 节选不是全部 100 轮，不能将两组均值之差作为精确的
控制流程耗时分解。

19 条 CPU PROFILE 的计数均一致：`status=0`、`read_wrs=2048`、`post_calls=64`、
`tail_cqes=4`、`waves=1`、`active_jettys=4`、`plan_reused=1`、`cpu_threads=1`。
每块拆成 64 KiB 和 18 KiB 两条 READ，每个 Jetty 512 个 WQE；每次 post 最多链接 32 个，
每个 Jetty 仅等待本轮尾部的成功 CQE。

`sequence=101` 的 `total_us=2793.7`、`cq_wait_us=2708.2`，该轮增加的耗时主要出现在 CQ 等待。
其余 18 条 `total_us` 平均 1,726.7 µs；上表均值保留了该抖动样本，日志不足以确定抖动原因。

结论：四 Jetty 批量传输和逐轮完整校验均通过，`cpu_checksum_included=0` 确认校验已分离。
传输调用与校验均值相加为 153.0845 ms，尚不含生产数据、Ready、Free 等阶段；32.981 GB/s
不代表包含逐轮校验的持续吞吐。当前完成查询在未完成时休眠 1 ms，可作为后续控制流程优化点，
但尚未通过独立计时验证其具体占比。

## 未决操作

显式事务由应用持有一个 put：Submit 成功返回 None，查询、等待、提交和撤销均使用原 put。参数校验失败且 `put.submitted=False` 时可以修正参数后提交；`put.submitted=True` 表示请求已记录，即使 Submit 响应丢失也用原 put 恢复，不可重复 PreparePut 替代未决任务。它不表示 CPU 已接受或 DMA 已完成。生产端通过 `release_batch(ready)` 或尚未获取批次时的 `cancel_batch(ready)` 核验身份及状态。

统一 `batch_cpu_pull()` 自动处理可确认失败的 Abort。未决 DMA 保留内部事务并阻止新调用，应用使用 `resolve_cpu_pull()` 查询、等待并完成原 Commit/Abort，不重发 READ。仍未决时保持源和 DDR 有效；Commit 已开始后不能逆转决定。生产端收到应用通知后 Release/Cancel，关闭源后才能释放消费端 DDR。shutdown 只尝试一次完成查询，不等待在途 DMA，也不代替生产端注销源。

## 构建与双卡验证

CPU 服务按 [原构建说明](cpu_pull_offload_usage.md) 重编并重启。NPU 的纯 Pull 验收仍可使用
`KVAF_ENABLE_CPU_OFFLOAD=ON`、`KVAF_BUILD_PYTHON=ON`、AIV 两项 OFF；现在该构建也导出
`NpuRuntime` 和 `CpuPullHbmSource`。

```bash
python tests/python/test_remote_cpu_offload.py \
  --cpu 127.0.0.1:28999 --device-a 0 --device-b 7 --cpu-candidate 0
```

如需要固定此前跑通的 B 路由，增加
`--npu-eid-b 000000000000020000100000df00e100`。device/candidate 以实际拓扑为准。

验证 A 的 BatchPut、CPU Pull 和 A 的 BatchGet 共用同一池时，同时启用三个 CMake 开关：

```bash
cmake -S . -B build-shared-ddr \
  -DKVAF_ENABLE_CPU_OFFLOAD=ON -DKVAF_ENABLE_AIV_BATCH_GET=ON \
  -DKVAF_ENABLE_AIV_BATCH_PUT=ON -DKVAF_BUILD_PYTHON=ON -DBUILD_TESTING=OFF
cmake --build build-shared-ddr -j8
export PYTHONPATH="$PWD/build-shared-ddr/python${PYTHONPATH:+:$PYTHONPATH}"
python tests/python/test_remote_cpu_offload.py \
  --cpu 127.0.0.1:28999 --device-a 0 --device-b 7 --cpu-candidate 0 \
  --batch-get-kernel /实际路径/kvaf_execute_batch_get_kernels.o \
  --batch-put-kernel /实际路径/kvaf_execute_batch_put_kernels.o
```

脚本在两个独立 NPU 进程间只传描述符和确认，连续三轮改变 B 输入、验证 CPU DDR 校验和、
重复通知幂等性、地址所属池和 B 复用确认。带 AIV kernel 时，A 在 B 注册前先写 key 0，
随后通过 BatchGet 比较 key 0 和 B 写入 keys 的完整输出，检查探测没有覆盖已有缓存。
成功打印 `REMOTE_OFFLOAD_PASS`。无法确认资源已释放时脚本会停留并保留进程；应先检查 CPU
日志、停止未决传输，再结束相关进程。

本地已覆盖真实 Core/协议/Exporter 编排的模拟测试，以及借用 DDR 的 URMA provider 模型测试。
这些测试不代替硬件验收；尤其是同一 allocation 的两份 MR 注册、A/B 各自路由和实际 NPU
cache 可见性，需要在上述双卡环境中验证。

## A 直接读取 B 的第二块 HBM

独立的裸地址读取路径及接口见 [hbm_raw_addr_read.md](hbm_raw_addr_read.md)。
A 沿用同一个 NpuRuntime；该路径不使用 DDR MetaManager 或 AccessTable。
