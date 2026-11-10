# Gather 性能采集与排障

当前实现、各阶段优化和配置合同见 [Gather 实现与配置](aiv_gather_prepared_prefix.md)。
以下统一使用 V9 / args 248 B / PRF5 256 B / CSV schema 9，历史 ABI 命令已移除。

## 构建与入口

K 侧重编 CPU 服务，沿用已配置的 listen、线程、绑核、NUMA 和页策略：

```bash
cmake --build build-cpu -j
./build-cpu/runtime/kvaf_cpu_service --listen 0.0.0.0 --port 34555 \
  --gather-threads 32 --gather-mode adaptive --gather-affinity on --memory-page-mode thp
```

N 侧同步重编 Host/Runtime/Python/kernel，沿用设备和 EID 环境：

```bash
cmake --build build-npu -j
export PYTHONPATH="$PWD/build-npu/python"
export KVAF_RUN_NPU_TESTS=1
export KVAF_CPU_SERVICE_ADDR=CPU_IP:34555
export KVAF_BATCH_GET_KERNEL="$PWD/build-npu/bin/kvaf_gather_batch_get_kernels.o"
export KVAF_BATCH_PUT_KERNEL="$PWD/build-npu/bin/kvaf_execute_batch_put_kernels.o"
python -u tests/python/test_aiv_gather_batch_get.py
```

启动日志须显示 `symbol=KvafGatherBatchGetAivKernelV9 args_bytes=248`。
功能脚本校验 payload/guard、空批及重复调用；probe 结果不能当完整回归 PASS。
环境变量在 Runtime 初始化时捕获，换组须新建 Runtime。

## 性能对照

修改 `tests/python/test_aiv_gather_batch_get_perf.py` 顶部配置：

```python
TOKEN_BYTES = 656          # 另测 4096
REQUEST_COUNT = 8192       # 可比较 1024 / 8192 / 8193 / 65536
GATHER_SGL_SIZE = 6        # 软件实验范围 1..13，设备限制见排障
WARMUP = 20
ITERATIONS = 100           # 无 profile E2E；0 跳过
PROFILE_SAMPLES = 10       # 独立同步诊断组；0 跳过
PROFILE_CLOCK_MHZ = 1000   # 已确认的 950DT 环境；未知设备设 None
CSV_OUTPUT_DIR = "gather-perf-results"
RUN_LABEL = "gather_prepared_prefix"
```

源表规模、tokens/block、随机 seed 也在该配置区。核数用 KVAF_NPU_CORE_COUNT，
设备用 DEVICE_ID，布局用 KVAF_GATHER_OUTPUT_LAYOUT=random/segmented/contiguous，
合并用 KVAF_AIV_GATHER_COALESCE_SGES=0/1。调用快路径可用
KVAF_GATHER_TRUSTED_INPUTS=0/1 对照，默认 1。

先保持核数、SGL、源/目标布局、seed、CPU 线程/绑核/NUMA/页策略一致，再独立比较：

```bash
unset KVAF_AIV_GATHER_SHARD_PIPELINE KVAF_AIV_GATHER_PACKED_POSTS KVAF_AIV_GATHER_PROGRESS_MODE
unset KVAF_AIV_GATHER_FIRST_CHUNK_DIVISOR KVAF_AIV_GATHER_CHECK_STATUS KVAF_AIV_GATHER_PROBE

# 整批 ACK + AIV 内部流水
KVAF_AIV_GATHER_CHUNK_BYTES=0 KVAF_AIV_GATHER_PREPARE_AHEAD=0 \
  KVAF_AIV_GATHER_COOPERATIVE_CHUNKS=0 python -u tests/python/test_aiv_gather_batch_get_perf.py
# 旧 128 KiB byte-prefix
KVAF_AIV_GATHER_CHUNK_BYTES=131072 KVAF_AIV_GATHER_PREPARE_AHEAD=0 \
  KVAF_AIV_GATHER_COOPERATIVE_CHUNKS=0 python -u tests/python/test_aiv_gather_batch_get_perf.py
# 整批 ACK + WQE 提前准备
KVAF_AIV_GATHER_CHUNK_BYTES=0 KVAF_AIV_GATHER_PREPARE_AHEAD=1 \
  KVAF_AIV_GATHER_COOPERATIVE_CHUNKS=0 python -u tests/python/test_aiv_gather_batch_get_perf.py
# 提前准备 + 均分协作块；分别比较 2/4/8/16/32
KVAF_AIV_GATHER_CHUNK_BYTES=131072 KVAF_AIV_GATHER_PREPARE_AHEAD=1 \
  KVAF_AIV_GATHER_COOPERATIVE_CHUNKS=4 python -u tests/python/test_aiv_gather_batch_get_perf.py
# 小首块：1/16 + 6 个尾块；分别调整 D/N
KVAF_AIV_GATHER_CHUNK_BYTES=131072 KVAF_AIV_GATHER_PREPARE_AHEAD=1 \
  KVAF_AIV_GATHER_FIRST_CHUNK_DIVISOR=16 KVAF_AIV_GATHER_COOPERATIVE_CHUNKS=6 \
  python -u tests/python/test_aiv_gather_batch_get_perf.py
```

CPU profile 由远端服务的 KVAF_GATHER_PROFILE 单独控制，正式 E2E 关闭它。
脚本的请求 Tensor 只准备一次并复用，初始化、BatchPut、预检、数据校验和 CSV I/O
均在计时循环外。合并 SGE/小首块会在计时前采 profile 检查配置实际生效。

无 profile 组调用 `synchronize=False`，再由 Python 同步至本批完成，不做状态 D2H；
计时合同为 `async_call_to_stream_completion`。profile 组用同步调用，含 status/profile
D2H 和校验，合同为 `synchronous_call_with_status_check`，快照复制在外层计时之外。
二者差值包含诊断回收成本，不能全归为设备打点开销。

比较 direct 时对齐数据 Jetty 数：gather 每核仅一条数据 Jetty，控制 Jetty 不搬 payload。
还须对齐软件在途预算、有效字节、统计量和计时范围；kernel-only 带宽不能直接比 E2E。

## 时钟与阶段含义

使用 GetSystemCycle 的系统计数器频率，不能使用 AI Core 主频或根据 Host 等待时间倒推。
当前脚本默认已确认的 950DT 1000 MHz；A2/A3 对应 50 MHz，换设备需查对应
[CANN GetSystemCycle 说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/tool_interface/system_resources_and_variables/GetSystemCycle_ISASI.md)。
可用 `npu-smi info -m` 和 `npu-smi info -t board -i <物理设备ID>` 核对芯片。
未知频率设 None，只输出 cycle；已确认时 `us=cycles/clock_mhz`，CSV 保留原始 cycle。

| AIV 阶段 | 时间边界 |
| --- | --- |
| input | kernel_start → input_done，含请求数检查及分区。 |
| descriptor | input_done → descriptor_done，含队列准备、SIMT 清单和 MTE3 完成。 |
| publish | descriptor_done → notify_done，即有序控制 WRITE 完成。 |
| wait_cpu | notify_done → ack_done，表示 AIV 观察区间，不是 CPU memcpy 时间。 |
| read | 整批模式 ack_done → read_done；chunk 模式 first_read_submit → read_done。 |
| finalize | 整批模式 read_done → kernel_finish；chunk 模式 max(ack_done,read_done) → kernel_finish。 |
| total | kernel_start → kernel_finish；空 lane 仍保留入口/出口开销。 |

chunk 的 wait_cpu/read 重叠；整批提前准备的 wait_cpu 与 WQE 构造重叠。
ACK 和数据 CQ 的观察顺序可不同，成功必须同时满足；不可将重叠阶段或各核时间相加。

| AIV 字段 | 含义 |
| --- | --- |
| requests / data_wrs / data_wqebbs / data_doorbells | 请求数、READ WR、实际 WQEBB 和发布次数。 |
| ack_queries / ready_queries | 最终 ACK 查询与进度查询次数。 |
| first_ready_cycle / first_read_submit_cycle | 首次非零 ready 快照、首数据 doorbell 发布前的 AIV 时间。 |
| first_read_ready_bytes | 首提交依据的连续 byte-prefix 快照；不等于 doorbell 时 CPU 的精确进度。 |
| pipeline_chunk_bytes | 每 lane 实际块大小；混合模式为尾块大小；整批/空 lane 为 0。 |
| read_build_cycles / read_vf_calls | 构造/发布及本地 bank 收尾累计时间、VF 次数；提前准备可发生在 ACK 前。 |
| ready_wait_cycles | 查询提交、完成检查、前台控制 CQ 等待；不含与 WQE 工作重叠的查询在途间隔。 |
| data_cq_wait_cycles | 物理 wave 尾 CQ 的剩余等待。 |
| prepare_ahead / cooperative_chunks | 实际启用标记及配置块数；混合模式为尾块数 N，空 lane 为 0。 |
| prepare_ahead_wrs / prepare_ahead_cycles | 构造时未被 AIV 确认 ready 的 WR 数和相应 Scalar 服务周期，不代表实际隐藏时长。 |

Host prepare_ns 含 owner 参数适配/output 注册，launch_ns 只含入队；status_enqueue_ns
是诊断 D2H 入队，wait_ns 是剩余执行/复制等待，finish_ns 是校验/回收。
total_ns 从上批 Drain 后的本批 owner 准备开始，不包含外层 Python/Runtime/capture。
call_* 字段补充 binding、metadata、stream、native、Runtime validation、operator、mutex、
capture、prior drain、input、launch、status copy、sync、validation、output acquire 的归因；
它们是嵌套包含关系，不能相加。CPU/Host ns 与 AIV cycle 属于不同钟域，不直接相减。

## Profile API 与 CSV

```python
runtime.start_gather_batch_get_profile()
runtime.batch_get(..., synchronize=True)
profile = runtime.gather_batch_get_profile()  # sequence, host, cores
runtime.stop_gather_batch_get_profile()
```

只返回最新成功同步批的完整快照；异步采集拒绝，失败/未采样不会返回旧快照。
停止采集不隐式读取快照，也不要求先有成功样本。

每轮新建包含北京时间、核数、token 字节数和请求数的目录，同秒运行追加序号，不覆盖旧结果。
四份 CSV 共享 run_id/config/revision/dirty/schema/timing_contract：

| 文件 | 内容 |
| --- | --- |
| summary.csv | E2E/profile E2E 的 samples、mean/p50/p95/p99/useful_GiBps，Host/core 均值和 sampling_status。 |
| e2e.csv | profile=off 的逐样本延迟和带宽。 |
| profile_host.csv | profile=on 的 sequence、外层调用时间、Host 及 call_* 归因。 |
| profile_core.csv | 逐样本逐核原始 cycle、阶段耗时和计数。 |

CSV 时间用 us，另保留 AIV cycle；Host/core 分行，不复制 Host 总时间到每核求和。
终端/summary 带宽统一按 `payload_bytes/(p50_us/1e6)/1024**3`；逐样本带宽用自身延迟。
列按字段名定位。失败也导出已完成样本并标记异常，未采到的阶段只有表头，不伪造零值。
CSV 导出及 shutdown 异常不覆盖先前的设备异常。

## CPU 性能采集与分析

K 侧启动服务前设置 KVAF_GATHER_PROFILE=1，其他参数保持一致。N 侧采相同请求，
按 session/lane/sequence 对照 CPU 日志和 profile_core.csv；lane 对应 AIV core。
N 侧 CSV 没有 CPU session，须确认同轮并按 sequence 筛掉 preflight/warmup/E2E。

每 lane 预分配 512 条记录，计时循环不分配/打印；正常 Runtime shutdown、StopGatherLanes
或 region 释放时输出 CPU_GATHER_PROFILE_BUFFER（含 samples/capacity/dropped）、
CPU_GATHER_PROFILE 和 CPU_GATHER_THREAD_PROFILE。强制结束服务可能丢失缓冲。

| CPU 请求字段 | 含义 |
| --- | --- |
| observed_ns / done_ns | 观察 notify / 最终 ACK barrier 完成的 CPU 时间戳。 |
| start_ack_ns / prepare_ns / copy_ns / ack_ns / total_ns | started ACK、校验/发布/唤醒、全部线程拷贝与汇合、最终 ACK、整批耗时。 |
| dispatch_to_copy_ns / join_wait_ns | generation 发布至 coordinator 开始复制、coordinator 完成本地任务后的等待，均与上项重叠。 |
| notify_poll_window_ns / notify_poll_window_valid | 最近一次带时间戳的空轮询到观察 notify 的宽松上界，每 256 次采一次；valid=0 不可解释为零延迟。 |

| CPU 线程字段 | 含义 |
| --- | --- |
| ready_delay_ns / start_delay_ns | generation 至取得任务、coordinator 开始至本线程开始工作的延迟。 |
| copy_ns / barrier_ns / work_ns | ProcessSource 墙钟、末尾同步、两者合计；chunk 的份额内 barrier 已计入 copy_ns。 |
| thread_cpu_ns / cpu_time_valid | 工作区间附近的线程 CPU 时间及有效性。 |
| bytes / tid / cpu / wait_path | 复制量、Linux TID、拷贝开始所在 CPU、是否经过条件变量分支；不保证实际睡眠或始终未迁移。 |
| progress_publications | 协作模式中该线程完成份额的块数，不等于实际 ready 更新次数。 |

优先找每 sequence 最晚完成的线程：start_delay 高看唤醒/调度，copy 高且 CPU 时间接近
墙钟看执行/访存；work 明显大于 CPU 时间可能有未运行区间。小差值不足以证明抢占。
join_wait 高结合最晚线程定位；不能用低 barrier_ns 推断 chunk 内 barrier 没有成本。
CPU total 与 AIV wait_cpu 起止边界不同，差值不是严格的“通知耗时”。

保留同轮日志并检查 `lscpu -e=CPU,NODE,CORE`、进程/线程 Cpus_allowed_list 和 numa_maps；
可用 `numastat -p PID` 辅助判断 NUMA。汇总脚本按 bytes/rows 识别 token，SGL 从文件名
（如 gather-cpu-656-sgl6.log）读取，无文件名标记时传 --sgl。

```bash
python3 tests/python/analyze_gather_cpu_profile.py 'cpu-logs/*.log' -o cpu_summary.csv \
  --fields total copy prepare join_wait --first-sequence 123 --last-sequence 132
```

省略筛选参数会统计全部 sequence/耗时字段；失败请求及 valid=0 字段排除，绝对时间戳
不当作耗时。按文件/token/SGL/lane/rows/active_threads 分组，保留慢样本；输出 samples、
mean、线性插值 p50/p95/p99、min/max、样本标准差/方差/CV。单位 us/us²/%，单样本的
标准差/方差/CV 留空。诊断计时有成本，正式吞吐关闭 CPU profile。

## 排障

先运行 `KVAF_RUN_NPU_TESTS=1 python -u tests/python/test_aiv_gather_pipeline_debug.py`，
按 ack → wqe → sq → first_read → full 独立子进程验证，首次失败停止；使用性能脚本配置形状。
probe 只代表对应前缀完成，不是完整 BatchGet 成功，也不能用于性能采样。

| 检查点/错误 | 定位范围 |
| --- | --- |
| load sequential kernel | 检查异常中的 read_file/create_binary/binary_load/get_function、实际路径、symbol/args_bytes 和加载的 Python 模块；仅扫描字符串不能证明 ACL 二进制可用。 |
| ack | 清单、command、notify 和最终 CPU ACK；output 应仍为 guard。 |
| wqe | ack 之后完成首批 VF 编码，不配置/搬运数据 SQ，不发数据 READ。 |
| sq | 首批 MTE3 写 SQ 完成，但不敲数据 doorbell。 |
| first_read | 首批数据 READ CQ 完成，只允许对应输出行改变。 |
| invalid gather profile | 普通状态与同步已通过；检查 reason、magic/sequence/result/mask、七个 cycle 和 requests/data_wrs 状态对照，不推测单一根因。 |
| valid_mask=0x41 | 曾观察到内层阶段戳丢失；保留当前每核显式 GM volatile 记录写入，不能改回跨函数可变局部 profile。仍要求 mask=0x7f。 |

profile 快速复测可设 WARMUP=0、ITERATIONS=0、PROFILE_SAMPLES=1，仍执行 payload/guard
预检，保留完整异常及当前部署版本。进度/CPU/CQ 错误或超时后重建 Runtime，不复用 poison 队列。

软件允许 SGL=1..13，默认 6，但 `max_jfs_sge=13` 不能证明 READ 可用 13 个目的 SGE。
已有公开 UDMA [READ 提交检查](https://github.com/openeuler-mirror/umdk/blob/master/src/urma/hw/udma/udma_u_jfs.c#L364)
使用独立的 UDMA_JFS_MAX_SGE_READ=6，须结合目标 HCCP/SDK/真机判断。
当前快速路径实际编码所选 9/13 个 SGE，不静默降为 6；模型仅验证编码布局。

SGL=9 的 first_read 若报 phase=5 result=2 expected_phase=23，说明 ACK 后等首批数据 CQ
超时，不能仅据此认定格式错误。排障时同时记录完整异常、SGL/token/request 配置及部署版本。
为隔离实际 SGE 数，固定 1 核、656 B、SGL=9，在功能脚本依次请求 6/7/9 行：

```bash
KVAF_NPU_CORE_COUNT=1 KVAF_NPU_TOKEN_BYTES=656 KVAF_AIV_GATHER_SGL_SIZE=9 \
  KVAF_NPU_REQUESTS=7 KVAF_AIV_GATHER_PROBE=first_read \
  python -u tests/python/test_aiv_gather_batch_get.py
```

三组均一条 READ、bank 0、192 B WQE，实际 SGE 数分别 6/7/9；不要用覆盖请求数的 debug 总控。
6 通过而 7/9 失败支持操作专用数量限制的判断；单条 9 通过而大批失败则继续查多 WR/队列。
性能和设备 DMA/CANN 兼容性最终以目标双机回归为准。
