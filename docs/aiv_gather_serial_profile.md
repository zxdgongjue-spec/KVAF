# Gather 串行阶段计时

此入口只测试 `gather_read`，保持第一阶段顺序依赖，不加入 chunk 流水或软件在途预算。
功能回归仍使用 `tests/python/test_aiv_gather_batch_get.py`；性能采集使用
`tests/python/test_aiv_gather_batch_get_perf.py`。

## 参数在哪里修改

直接编辑性能脚本顶部的配置区：

```python
TOKEN_BYTES = 656       # 每条 token 的字节数，不是请求数
REQUEST_COUNT = 8193   # 每批有效请求数：可改为 1024 / 8193 / 65536
WARMUP = 20            # 不计入结果的预热次数
ITERATIONS = 100       # 关闭 profile 的端到端性能采样次数，0 可跳过
PROFILE_SAMPLES = 10   # 开启 profile 的阶段诊断采样次数
CSV_OUTPUT_DIR = "gather-perf-results"  # 相对执行命令时的工作目录
RUN_LABEL = "serial"   # 手工标注串行 / chunk 流水等对照版本
```

请求数、token 长度、预热和采样次数不需要新增启动参数。核数仍通过
`KVAF_NPU_CORE_COUNT=1/3/8` 设置。源码区还可配置源表规模、每个 block 的
token 数、SGL 和随机种子；比较不同实现时保持这些值一致。

`PROFILE_CLOCK_MHZ` 默认不填，只输出 AIV 原始 cycle。确认部署平台的
`GetSystemCycle()` 计数器频率后才填写 MHz，脚本再额外换算微秒；不能将
AI Core 主频或某个固定默认值直接作为该频率。

## CSV 输出与后续对比

每次运行结束自动创建一个测试目录，将四份 CSV 放在一起，并打印
`GATHER_CSV_FILE path=...` 的绝对路径。目录按北京时间（Asia/Shanghai、
UTC+08:00）命名，包含年月日、时分秒、核数、token bytes、request-count：

```text
gather-perf-results/
  20261031_093235-cores1-token-bytes656-request-count8193/
    summary.csv
    e2e.csv
    profile_host.csv
    profile_core.csv
```

同一秒、同一配置重复运行时，目录自动追加 `_2`、`_3`，不会覆盖或混入旧
测试。CSV 使用 UTF-8 BOM，可直接在 Excel 中导入。默认结果目录不进入 Git。

| 文件 | 内容 |
| --- | --- |
| `summary.csv` | 本轮汇总，通常先看这一份 |
| `e2e.csv` | 关闭 AIV/Host 打点的逐次端到端样本 |
| `profile_host.csv` | 开启打点后的逐次 Host 时间、外层调用时间和 sequence |
| `profile_core.csv` | 每次采样、每个核的原始时间戳、六段耗时和计数 |

四份文件均包含配置和 run_id：label、代码 revision/dirty、设备/核数、请求数、
token 字节数、SGL、表规模、seed、预热/采样次数、计数器频率、kernel 路径
和 CPU 服务地址。远端 CPU 日志开关仍记作 external，不假定它已经关闭。

summary 按 `row_kind` 区分：`e2e_summary` 是无打点性能基线，
`profile_e2e_summary` 是有打点、同口径的外层调用统计，另有
`profile_host_mean` 和各核的 `profile_core_mean`。通过 run_label、cores、
request_count、token_bytes 等列筛选对照，避免把不同核或不同采集口径相加。

列顺序按字段名固定：`clock_mhz`、`clock_source`、`cpu_profile`、`kernel_path`、
`put_kernel_path`、`request_plan`、`timing_contract` 放在 `mode` 前；summary 的
`error_type`、`error_message` 放在 `total_us` 后。移动后部分 Excel 列字母会
变化，整理数据时建议按表头名称定位。

summary 的 Host 和 AIV 行共享表头，但各自只填自己的指标：

| row_kind | 读取哪些时间 | READ 时间在哪里 |
| --- | --- | --- |
| e2e_summary / profile_e2e_summary | mean_us、p50_us 等整批统计 | 已包含在整批时间中，不额外拆阶段 |
| profile_host_mean | prepare_us、launch_us、status_enqueue_us、wait_us、finish_us、total_us | Host 只观察同步等待，没有独立 READ 计时 |
| profile_core_mean | input / descriptor / publish / wait_cpu / read / finalize / total 的 cycles 和 us | read_cycles，频率核实后有 read_us |

`profile_host_mean` 行的阶段 cycles 列应为空。`wait_us` 是 Host 的剩余同步
等待，包括设备执行和状态/profile D2H；AIV 的 `wait_cpu_cycles` 只覆盖通知
完成到最终 ACK，`read_cycles` 从最终 ACK 到数据 CQ 完成。READ 内又有
`read_build_cycles`（构造和 SQ 写入）与 `data_cq_wait_cycles`（数据 CQ 等待）
等子计数。Host 等待、AIV 阶段、READ 子计数存在包含关系，不能重复相加。
当 `clock_source=unverified` 时，AIV 的 read_us 等时间列留空，read_cycles
仍在 profile_core_mean 行输出。这表示尚缺换算频率，不表示漏采 READ。

CSV schema_version 为 2。所有耗时列以 `_us` 命名，不再导出 `_ns` 列；
Host 的内部 ns 在导出时除以 1000。AIV 保留原始 cycle 供诊断，六阶段、
构造及 CQ 等待均有对应 `_us` 列；频率未核实时这些列留空，不猜测频率。
测试开始时间仍以带时区的日期文本保存，便于对应目录。
CSV I/O 在测量循环之外进行，不计入样本时间。
开启打点时的外层调用计时包含本次同步和状态/profile D2H，快照的 Python
复制在调用计时之外，便于与无打点组比较。

如果采样中途报错，已完成的样本仍会输出，summary 的 `sampling_status`
行为 failed 并保留异常；没有采到的阶段文件只有表头，不伪造零耗时样本。
sampling_status 只描述采样阶段，随后 Runtime shutdown 的异常仍单独报告。

## 构建与运行

沿用第一阶段的 CANN/Python/CPU 地址环境，在两端仓库根目录执行。
本次 gather 参数增加 profile 指针，Host 和 kernel 必须一起更新：新 loader
要求 `KvafGatherBatchGetAivKernelV2`，旧 kernel 会加载失败，不会静默漏采。
scatter 和 direct 的 kernel ABI 不变。

```bash
# NPU：更新 Host、Python binding 和 gather kernel
cmake --build build-npu -j
export PYTHONPATH="$PWD/build-npu/python"
export KVAF_RUN_NPU_TESTS=1
export KVAF_CPU_SERVICE_ADDR=CPU_IP:28999
export KVAF_BATCH_GET_KERNEL="$PWD/build-npu/bin/kvaf_gather_batch_get_kernels.o"
export KVAF_BATCH_PUT_KERNEL="$PWD/build-npu/bin/kvaf_execute_batch_put_kernels.o"
DEVICE_ID=7 KVAF_NPU_CORE_COUNT=1 python tests/python/test_aiv_gather_batch_get_perf.py
```

CPU 线程总预算至少等于 AIV 核数，继续使用同一个 CPU 服务。需要 CPU 阶段
诊断时，重编并在启动 CPU 服务前设置：

```bash
cmake --build build-cpu -j
KVAF_GATHER_PROFILE=1 ./build-cpu/runtime/kvaf_cpu_service \
  --listen 0.0.0.0 --port 28999 --gather-threads 8
```

CPU 日志开关在 worker 创建时读取。日志在最终 ACK 后打印，包含
`session/lane/sequence`、rows、bytes、实际线程数、`prepare_ns`、`copy_ns`、
`ack_ns`、`total_ns` 和结果。`copy_ns` 包括本 lane 所有 helper 完成等待；
`total_ns` 包含准备、拷贝及控制开销，不与子项重复相加。

CPU 日志用于诊断，会干扰性能。正式端到端基线关闭此环境变量；需要解释
CPU 成本时另跑一轮诊断，不将开启 CPU 日志的结果作为无打点吞吐。

## 计时范围

每次调用显式 `synchronize=True`，当前批的状态回收也包含在本批完成之内。
初始化、BatchPut、请求 Tensor 构造和结果校验不在测量循环内。预热后先采
关闭 profile 的端到端延迟及有效 GiB/s，再单独开启 profile 采阶段数据。

每个 AIV 核记录六段：

| 阶段 | 边界 |
| --- | --- |
| 输入准备 | kernel 入口 → 输入缓存处理与本 lane 校验完成 |
| 清单构造 | 输入准备完成 → 清单构造、flush 完成 |
| 请求发布 | 清单完成 → descriptor、command、notify 的 WRITE 完成 |
| 等待 CPU | notify WRITE 完成 → 观察到最终 ACK |
| 数据 READ | 最终 ACK → 全部数据 READ 的 CQ 完成，包含 WQE 构造和容量回收 |
| 收尾 | 数据完成 → 状态发布及 profile 结束时间点 |

空批或无任务 lane 不发通知或数据 READ，其阶段时间按跳过处理。原始 profile
同时保留请求数、数据 WR/WQEBB、doorbell、ACK 查询及队列等待等计数。
这些子计数用于解释数据阶段，不与其总时间重复相加。

Host 的 `prepare_ns` 覆盖 owner 内的参数检查及 output 注册，`launch_ns`
覆盖状态清零及 launch 入队，`status_enqueue_ns` 覆盖状态/profile D2H 入队，
`wait_ns` 是设备执行及 D2H 的剩余完成等待，不是完整 kernel 耗时；
`finish_ns` 覆盖结果校验与资源回收。
`total_ns` 从上批 Drain 后的本批 owner 准备开始到回收结束，不包含 Python、
Runtime 参数适配和 capture 查询；因此它
不必等于 Python 外层端到端时间。profile 自身的写回/复制也有成本。
输出中的 `profile=off/on` 仅描述 AIV/Host，`cpu_profile=external` 表示远端
CPU 开关由服务进程单独控制，脚本无法替它关闭。

CPU/Host 的 ns 与 AIV 的 cycle 属于不同的时钟域，只计算各自差值，以
`sequence + lane/core` 关联。AIV 等待 CPU 已覆盖部分 CPU 工作及轮询延迟，
不能再把 CPU gather 时间加进去。各核并行时间也不能相加。

## Python 接口

```python
runtime.start_gather_batch_get_profile()
runtime.batch_get(..., synchronize=True)
profile = runtime.gather_batch_get_profile()
runtime.stop_gather_batch_get_profile()
```

快照包含 `sequence`、Host 时间字典 `host` 和每核时间线 `cores`。只接受
成功的同步 gather；开启 profile 时异步调用会明确拒绝。开启后尚未采样、
执行失败或随后执行未采样批次时，不会返回旧快照。关闭采集无需已有快照。

## 后续与二阶段比较

保留串行版本，依次比较“串行 / chunk 流水 / chunk 加内部流水”。固定输入、
有效字节、核数、CPU 线程预算、SGL 和队列容量，保持软件在途预算关闭。
比较 p50/p95/p99、有效 GiB/s 和串行耗时除以流水耗时的加速比。流水阶段
另加 chunk ready、提交和完成事件验证重叠，不把重叠阶段耗时相加。

本地 CPU 模型和 Host 替身只验证计时记录、生命周期及数据语义；CANN 编译、
计数器频率、远端 DMA 可见性和真实性能仍需目标双机验证。

## 已有端到端结果与 profile 校验失败

`GATHER_PERF_RESULT` 是关闭 AIV/Host profile 的同步调用端到端带宽。旧脚本的
`operation=batch_get_kernel_total` 则只使用 kernel 时间，比较前须统一计时范围。
例如 8192 × 656 B 约 5.37 MB，35 GB/s 对应约 154 us；不能直接拿它与含
Python/Host、CPU gather、控制握手和状态回收的整批延迟比较。8192 与 8193
的请求数差异仅约 0.012%，无法解释数量级差距。

资源也要对齐：原 direct 的 2 核 × 每核 2 条 Jetty 是四条数据队列；gather
每核固定一条数据队列和一条控制队列。`KVAF_NPU_CORE_COUNT=2` 可以对齐
核数，但 gather 仍只有两条数据队列。当前 gather 还逐项构造清单、标量写
SQ 及 flush；在全部 READ WQE 能放进物理队列时，数据阶段末尾才敲 doorbell。
原 direct 则已有 SIMT 校验、分批 WQE 构造、UB/MTE 提交和内部流水。
这些是已确认的实现差别，实际瓶颈仍以阶段数据为准。

`invalid gather profile` 在同步及普通执行状态检查通过后产生。新异常包含
`reason`、记录 magic/sequence/result/valid_mask、七个原始 cycle、请求数/WR
数及其状态对照，且在资源清理前保存到异常文本。校验门槛保持不变。

| reason | 可以确定的失败项 |
| --- | --- |
| magic / sequence / result / valid_mask | 记录头或完成标记不符合预期 |
| timestamp_* | 指定的相邻原始时间戳倒序；尚未进行频率换算 |
| requests / data_wrs | profile 与普通执行状态的计数不一致 |

没有实际字段时，不能将该错误认定为时钟、缓存或 ABI 的某一种故障。
新的日志用于区分这些方向；本地模型不复现远端设备上的失败。

快速复测时，只改性能脚本顶部：

```python
WARMUP = 0
ITERATIONS = 0
PROFILE_SAMPLES = 1
```

脚本仍执行 payload/guard 预检，随后跳过端到端性能循环，打印
`GATHER_PROFILE_PHASE_BEGIN` 并开始阶段采集。失败时保留完整的
`invalid gather profile ... reason=...` 行。`6d2ecaf` 的诊断增强只修改 Host 和
Python；下文针对 `0x41` 的修改位于 kernel，必须重编对应 kernel。

## valid_mask=0x41、早期时间戳为零的修正

硬件反馈：magic、sequence、result、requests、data_wrs 均匹配，但 mask 为
`0x41`，start/input/descriptor/notify 为零，ack/read/finish 已有非零 cycle。
`0x41` 仅包含外层入口、出口两位，与函数内部阶段和 mask 的更新未一致保留
相符。它不表示数据 READ 失败，也不是由 MHz 换算产生。没有匹配的设备
指令或编译器中间表示，尚不能断定具体的编译器/局部内存机制。

修改将计时状态从跨 Gather/Queue 函数传递的可变局部结构体移到每核独占的
HBM profile 记录，使用显式 `__gm__ volatile` 指针。入口显式初始化，各阶段
直接写对应字段并累积 mask，结尾设置 result/magic 并 flush；不再把局部
结构体整体或逐字段回拷。关闭 profile 时不读时钟、不访问 profile 内存。
每条记录仍只有一个 AIV 核写入，Host 仍在 stream 完成后读取。

ABI 保持 V2 / args 224 B / profile 192 B，Host 仍要求 `valid_mask=0x7f`，
继续拒绝硬件日志中的不完整形状。此次需在 NPU 目标机运行：

```bash
git pull --ff-only
cmake --build build-npu --target kvaf_gather_batch_get_kernel -j
DEVICE_ID=7 KVAF_NPU_CORE_COUNT=1 python tests/python/test_aiv_gather_batch_get_perf.py
```

沿用已有 Host、环境和 kernel 路径；若尚未部署 `6d2ecaf` 的 Host 诊断，先
执行一次完整 `cmake --build build-npu -j`。计时入口前的 profile 初始化和
最后记录导出的开销不属于六段 cycle，外层同步调用仍包含这些成本。

模型新增在 descriptor/ACK/data doorbell 时检查生产 GM 记录的阶段进度，确保
早期时间戳在内层函数仍执行时已保留；同时覆盖空 lane、重复批次与错误。
这些模型检查不模拟 CANN 的局部对象编译行为，是否消除本次硬件失败仍需
上述复测。当前日志的后半段仅作为性能线索，不作为通过验收的六阶段结果。
