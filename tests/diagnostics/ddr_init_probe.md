# DDR initialization probe

Run these commands from the KVAF root on the NPU host, using matching CPU/NPU
builds. The probe loads no model and does not fill the DDR pool with benchmark
data. It retains KVAF's initialization-time connectivity probes.

Start one local CPU service in a separate terminal:

```bash
./build-cpu/runtime/kvaf_cpu_service \
  --listen 127.0.0.1 --port 34555 --memory-page-mode thp \
  > cpu-34555.log 2>&1
```

Use the actual service binary path if your build directory differs. Wait for
`LISTEN` in its log before running clients. `127.0.0.1` assumes the clients and
service share the host network namespace. Fix the service's NUMA placement
across experiments when placement matters; check node capacity as well as
total free memory.

Measure one device at a time, freeing each region before the next sample:

```bash
for gib in 1 8 16 32; do
  python3 -u tests/diagnostics/ddr_init_probe.py \
    --build build-npu --address 127.0.0.1:34555 \
    --device 0 --ddr-gib "$gib" \
    > "ddr-device0-${gib}g.log" 2>&1 || break
done
```

`--ddr-gib` controls the first argument to `initialize(ddr_bytes, slot_size)`.
It is rounded up to a whole number of slots. The default 86,016-byte slot
matches the local vLLM layout: 656 bytes/token x 128 tokens, aligned to 4,096.
The initial `config` record prints requested and actual bytes; `ready` prints
the returned remote region. Slot size must stay fixed for capacity comparisons.
Set `--slot-size` to match the deployed layout if it differs. The kernel paths
default to `build-npu/bin/kvaf_execute_batch_{get,put}_kernels.o`; override them
with `--batch-get-kernel` and `--batch-put-kernel` if needed.

Read the JSON log line with `event=done`, `operation=initialize`, `elapsed_s`:

```text
DDR_INIT_PROBE {"device": 0, "elapsed_s": 12.345, "event": "done", "operation": "initialize", ...}
```

This is elapsed wall time around the synchronous Python/native initialize
call. It includes CPU service queuing, allocation, pre-touch (in thp/no-thp modes),
registration, metadata/channel setup and connectivity probes. It excludes
constructor, link_to, holding and shutdown, which have separate records. The
number is NOT the isolated malloc time or steady-state BatchGet time. No
external NPU synchronization is added to this scope.

For sequential multi-device allocation with earlier regions retained, start
one foreground terminal per device and use `--hold`:

```bash
python3 -u tests/diagnostics/ddr_init_probe.py \
  --build build-npu --device 0 --ddr-gib 32 --hold
```

Wait for `ready`/`hold`, then start device 1 in another terminal with
`--device 1`, and so on. Keep every earlier terminal at its hold prompt. Press
Enter only after all measurements are complete. Eight live 32-GiB requests
require about 256 GiB for the regions plus runtime/system memory. Device IDs
are logical IDs under the current visibility configuration.

For overlapping launches, `--hold-seconds 3600` retains completed regions
for an hour without requiring terminal input. Background processes launched
in a shell loop are overlapping, not a strict simultaneous barrier. On this
version, one CPU service processes requests serially regardless of client count.
Compare each client's initialize time with server stage times before concluding
there is parallel allocation contention.

Optional `--peer-ub-device` and `--npu-eids` allow a known-working path to be
held fixed. Keep these and the AIV configuration fixed across runs.

## 单机初始化串行对照

重建 KVAF NPU Runtime/binding 后，可用 `KVAF_SERIALIZE_INITIALIZE=1` 将同一
CPU endpoint 的多进程初始化串行化；不改 vLLM 或诊断脚本，CPU service 无需
仅为此更新。默认关闭，完整变量及文件锁限制见
[Runtime 初始化串行开关](../../runtime/README.md#单机多进程初始化串行开关)。

先用小容量验证开关和锁日志，例如在 CPU service 已 `LISTEN` 后运行：

```bash
export KVAF_SERIALIZE_INITIALIZE=1
export KVAF_INITIALIZE_LOCK_TIMEOUT_SECONDS=1800
for device in 4 5 6 7; do
  python3 -u tests/diagnostics/ddr_init_probe.py \
    --build build-npu --address 127.0.0.1:34555 \
    --device "$device" --ddr-gib 1 --hold-seconds 30 \
    > "npu-gate1-device${device}-1g.log" 2>&1 &
done
wait
```

`--hold-seconds` 在初始化完成后保留 region，文件锁则在初始化返回时已经释放。
上述后台启动只是同时拉起进程，`set_device` 的差异仍可能让各卡进入初始化的
时间错开；确认日志里实际出现等锁后，才能判断这一轮是否覆盖了重叠初始化。
锁不保证按 4→5→6→7 的顺序获取。

`[KVAF_INIT_GATE]` 独立记录 `wait/acquired/released/timeout`，不需要打开
profiling。完整 `DDR_INIT_PROBE` 的 `operation=initialize` 耗时包含等锁时间；
`KVAF_INIT_PROFILE` 的 `candidate_initialize` 和内部阶段在拿锁后才开始，
不包含等锁。不要把排队更久误判为 DDR 分配更慢。

正式比较时，将容量改为 32/64 GiB，确认有足够内存；先释放上一组所有 region，
再使用 `KVAF_SERIALIZE_INITIALIZE=0` 和新的日志文件名运行基线。两组保持相同
CPU service 页模式、容量、候选和起始驻留压力，保留全部样本；比较每次持锁
初始化耗时及所有卡的就绪耗时。串行开关是缓解并发启动压力的措施，不证明 THP
或多卡竞争已经是唯一根因。做后面的 THP 因果对照时，两种页模式应使用相同
开关设置，原有并发基线则保持 `KVAF_SERIALIZE_INITIALIZE=0`。

## THP 长耗时诊断：匹配预触页的对照

`origin` 与 `thp` 的实测差异可用于定位代码路径，但不能单独证明内核 THP 有问题。
`origin` 不预触页，实际物理驻留量、缺页发生阶段与 `thp` 可能不同。严格对照使用
新模式 `no-thp`：

| CPU service 模式 | 起始地址对齐 | 内存建议 | 注册前全量清零 |
| --- | --- | --- | --- |
| `thp` | 2 MiB | `MADV_HUGEPAGE` | 相同的 `memset` |
| `no-thp` | 2 MiB | `MADV_NOHUGEPAGE` | 相同的 `memset` |
| `origin` | 4 KiB | 不主动设置 | 不预触页，保留原行为 |

`thp` 与 `no-thp` 使用相同容量和注册流程，只切换页建议。`thp` 是请求大页，
不保证实际全部使用大页；Linux 页粒度也不保证与驱动、UDMA/IOMMU 粒度相同。
默认 slot 仍为 86,016 B：它是池内数据布局的 4 KiB 对齐，与整块池的 2 MiB
起始地址对齐是两个概念。两组均按 slot 向上取整，比较 `config` 中相同的
`allocated_bytes`，不要把请求的 32/64 GiB 当成精确分配字节数。

### 构建与采集

在相同代码版本上重建 CPU service 和 NPU Python/native binding；仅复制 Python
脚本不会给旧二进制增加打点选项。下面命令从 KVAF 根目录执行，环境须已配置 CANN：

```bash
./build.sh cpu
./build.sh npu
```

每轮保存独立日志。先启动 CPU service：

```bash
./build-cpu/runtime/kvaf_cpu_service \
  --listen 127.0.0.1 --port 34555 \
  --memory-page-mode thp --profile-initialize \
  > cpu-thp-r01.log 2>&1
```

等 `LISTEN` 后，在另一个终端启动客户端，例如固定正常的 7 号卡、64 GiB：

```bash
python3 -u tests/diagnostics/ddr_init_probe.py \
  --build build-npu --address 127.0.0.1:34555 \
  --device 7 --ddr-gib 64 --profile-initialize --hold \
  > npu-thp-r01-device7-64g.log 2>&1
```

`--hold` 需要前台交互终端；上例把输出重定向到文件，在日志中查看 `ready`/`hold`。
此时内存保持存活，观察完成后在运行客户端的终端按 Enter，等待 `shutdown done`。
也可用 `--hold-seconds 30` 替代交互等待；等待时间
不计入 `initialize`。采集 `no-thp` 组前，释放所有 region、正常停止旧服务，
将服务参数改为 `--memory-page-mode no-thp`，并为两端日志使用新的文件名。
两端都显式启用 `--profile-initialize`；默认关闭，不影响日常日志量。

### 实验顺序与控制项

先固定一张正常卡和一个容量，如 64 GiB，重复对照；再测 32 GiB，最后增加 hold。
让 A=`thp`、B=`no-thp` 交替运行（如 A/B/B/A），统一为每轮重启 CPU service，
记录卡号、顺序和重启边界。重启服务不重置内核碎片或页状态，所以仍需比较每次的
起始内存快照。不要只留下最快一次，也不要用重跑后的结果覆盖慢样本；保留全部原始
日志，先比较样本数、中位数、最小值和最大值，少量样本不适合解释 P99。

固定 CPU service 的 CPU/NUMA 绑定、UB device/EID、NPU EID、AIV core/Jetty 数，
容量与 slot_size；记录 THP `enabled`/`defrag`、内核版本和其他大内存任务。
NUMA 绑定应根据机器拓扑选择，并确认目标节点容量充足：绑核和绑内存都作用于
实际分配 DDR 的 CPU service。不要在 A/B 中途改变系统 THP 全局配置；
`no-thp` 已对该 region 显式请求 `MADV_NOHUGEPAGE`。

每组留存机器配置，例如：

```bash
uname -r
cat /sys/kernel/mm/transparent_hugepage/enabled
cat /sys/kernel/mm/transparent_hugepage/defrag
numactl --hardware
```

做 hold 对照时，每一组都使用相同卡序、每卡容量和前置 region 数；等上一张
`ready` 再启动下一张。比如比较“已有三张卡各持有 32 GiB 时，第四张再申请
32 GiB”，两组都必须保留相同前置资源。当前同一 CPU service 串行处理请求，
hold 测累计占用，不等于并行申请。记录前置 region 的实际驻留量，避免只凭
请求容量认定两组物理内存压力相同。

### 打点口径与分析

原有 `DDR_INIT_PROBE` 仍以 `event=done, operation=initialize` 的 `elapsed_s`
表示客户端完整初始化。新增 `KVAF_INIT_PROFILE` JSONL 记录分段 begin/end 和
内存快照，可与普通 KVAF 日志混合保存：

| 角色 | 主要阶段 | 含义 |
| --- | --- | --- |
| CPU | `total_allocate` | 一次服务端 region 分配的总耗时 |
| CPU | `candidate_resources` | 获取/创建本次 UB 候选资源 |
| CPU | `posix_memalign` | 申请对齐地址空间 |
| CPU | `madvise` | 提交本 region 的页建议 |
| CPU | `memset` | 全量触页、清零，可能在此发生缺页、回收和整理 |
| CPU | `alloc_token` / `register_seg` | token 分配与内存注册 |
| NPU | `allocate_rpc` | 一次分配 RPC，包含网络、服务端排队和处理 |
| NPU | `candidate_initialize` | 一次候选初始化总耗时 |
| NPU | `transport_initialize` / `metadata_initialize` / `startup_probe` | 通路、元数据和实际启动写验证 |

用一次运行的文件与 PID、NPU attempt、CPU session 关联快慢样本；服务重启后
session/attempt 可能重用，不能跨日志仅按数字合并。一次 `initialize` 中可能有
多次候选尝试、整块 DDR 重新申请，分析前先核对成功/失败及 region 生命周期。
`REGION_ALLOCATED` 仍是完成标志，不代表单独的 malloc 耗时。

阶段耗时使用单调时钟。`timestamp_unix_ns` 是墙上时钟时间戳，仅辅助对齐日志；
它可能受时钟调整或跨机器时差影响，不用于跨主机计算精确耗时或排队时间。`allocate_rpc`
比对应 CPU `total_allocate` 多出的时间，还包括控制协议和网络开销，不能全算
作 DDR 分配。总计时嵌套包含分段和采集开销；不要把 `total_allocate`、
`candidate_initialize` 与内部阶段再相加。内层阶段计时排除该阶段自身采样开销，
两组应使用相同 profiling 开关。

每个内存快照另有 `collection_elapsed_ms`，记录采集本身的耗时。特别是
`smaps_rollup` 需要检查页映射，普通页与大页的采样成本可能不同；同样开启
profiling 并不保证两组采集开销相同。诊断时优先比较 `memset`、`register_seg`
等最内层阶段，单独报告采集开销；完整 `initialize` / `total_allocate` 包含它，
不能把这部分差异误归因于内存分配。

`end` 中的 `outcome` 为 `unknown`、`success` 或 `failed`；异常退出可能只有
`begin`，不能当成零耗时。支持的平台还记录 `user_cpu_ms`、`system_cpu_ms`、
`minor_faults`、`major_faults` 和上下文切换的增量。`usage_scope` 表示
`thread`、`process` 或 `unavailable`，不同口径的结果不要直接混比。

CPU 在 `before_allocate`、`before_memset`、`after_memset`、`before_register`、
`after_register`、`after_release` 保存快照，包括全局 vmstat、
`global_meminfo_kib` 中的内存余量等瞬时值、各 NUMA node 的空闲内存，以及进程
RSS/匿名大页用量。`before_memset → after_memset` 与
`before_register → after_register` 可分别观察触页、注册附近的全局计数增量；
分析器也提供整个分配窗口的增量。窗口仍包含相邻快照的部分采集过程，不能作为
这些计数由该阶段独占产生的证明。

快照是整台机器或整个 CPU service 的状态，不是当前 region 的独立统计：
全局 `compact_*`、`thp_fault_*`、直接回收和换页增量也可能来自其他任务；
进程 `process_anon_huge_pages_kib` 来自 `smaps_rollup`，不能当成当前 region 的
精确大页比例。`AnonHugePages` 主要反映传统 PMD 级 THP 映射，不覆盖所有 mTHP，
所以该数值为零也不足以证明所有分配均为基础页。字段语义见
[内核 THP 监控说明](https://www.kernel.org/doc/html/latest/admin-guide/mm/transhuge.html#monitoring-usage)
和 [proc 文档](https://www.kernel.org/doc/html/latest/filesystems/proc.html)。
采样失败或内核未提供的字段应视为缺失，而非零。

可生成同一轮的汇总：

```bash
python3 tests/diagnostics/analyze_ddr_init.py \
  --cpu-log cpu-thp-r01.log \
  --npu-log npu-thp-r01-device7-64g.log \
  --output summary-thp-r01.json
```

优先比较同容量的一快一慢样本：

- `memset` 长，且同时出现明显 `compact_stall` / 整理扫描增量：支持触页期间
  同步整理的解释；结合 THP 实际用量和其他负载继续确认。
- `memset` 长，并伴随直接回收、major faults 或换页：检查可用内存、NUMA 节点
  压力；minor faults 本身是首次触页的正常工作，不能单凭数量判为异常。
- `register_seg` 长而 `memset` 稳定：将重点转向内存注册；Linux 大页是否改变
  注册代价需要依据实测，不能假定驱动使用相同页粒度。
- CPU 分段稳定而 NPU 端慢：检查服务排队、候选回退、建链和 `startup_probe`。
- `madvise` 很快但 `memset` 很慢并不矛盾：提交建议与后续实际触页分属不同阶段。

THP 的同步回收/整理行为还取决于 `defrag` 策略；具体模式含义见
[内核 THP 全局配置说明](https://www.kernel.org/doc/html/latest/admin-guide/mm/transhuge.html#global-thp-controls)。

当 `thp` 与 `no-thp` 在相同驻留压力下，仅前者稳定出现同一阶段的长尾，才进一步
针对该阶段提出机制假设；这组对照本身不能直接定性为内核 bug。

Local validation (no NPU required):

```bash
python3 tests/diagnostics/ddr_init_probe.py \
  --build build-npu --ddr-gib 32 --dry-run
```

Actual initialization, NUMA placement and service timings require an NPU host.
