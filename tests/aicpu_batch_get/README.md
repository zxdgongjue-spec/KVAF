# AICPU 发起、CPU URMA 回推：scatter / pipeline / direct output 带宽测试

这是独立 C++ 测试，不调用 Runtime BatchPut，不改 Python 或 MetaManager。

当前普通传输入口为 **V14**（从 V11 基线增加传输打点，未引入 chunk/prefetch 优化）。
CPU 增加 `--profile-transfer`，可选 `--profile-transfer-csv FILE` 输出逐轮记录；
只接受 `--transfer-mode scatter`。两端程序、device .so、JSON 必须一起更新，安装后重启 NPU 进程。
完整实际流程、各计时边界、运行命令和结果解读见
[AICPU + SGL 分阶段计时](../../docs/aicpu_sgl_transfer_profile.md)。

scatter 在 656 B 行上使用 AICPU NEON 向量复制，其他行长使用 `memcpy`，
支持 `--scatter-threads 1..32`（默认 1）。
CPU 将 SGL 描述符按 output 物理 slot 排列，仍 WRITE 到连续 HBM staging；
AICPU 将同一个 slot 顺序复制到 output。
调用线程也参与复制，其余 pthread 在进入测试循环前创建，跨预热和计时轮复用。
CSV 标注 `aicpu_memcpy_sgl_reordered` /
`aicpu_memcpy_sgl_reordered_mt`，并输出请求与参与分片的线程数。
`active_threads` 不是实际占用的核数；实际核采样需启用下面的诊断开关。
线程创建失败报告 `pthread_error`，不会静默回退。设备 pthread 支持及带宽需在目标机器验证。

### 可切换的批内流水线（V11）

CPU 端选择 `--transfer-mode scatter_pipeline`；默认仍为 `scatter`，
`direct_output` 继续保留。流水线沿用现有 `ScatterRange`：4096×656 B、4 个 worker
时，每核仍负责 1024 行，使用现有 NEON copy 和 payload clean。
CPU 依次对每个完整分片调用同步 `WriteBatch`，等待该分片所有 Jetty 完成后，
向独立的 64 B ready 控制记录 WRITE 当前 sequence，然后立即发送下一分片。
设备在 request 发布后释放全部 block，每核只等待自己的 ready，随后
invalidate 自己的 staging、copy、clean。block 0 同时负责分片 0 的 scatter。
无需第五个核、双缓冲或生产 transport 的异步 API；整批 staging 的各分片不重叠，
下一轮仅在全部分片完成、全局 producer done 及计时外校验完成后请求。

控制区的 per-range ready 从 offset 512 开始，每核独占一个 cache line，与 request、
global done、staging 分离。旧 sequence 继续等待，未来 sequence、缺失 ready、
超时或其他 block 的错误会使本轮失败。沿用现有 CPU drain/设备资源保留握手。
ready 的正确性仍依赖 provider WRITE completion 到远端 HBM 可见性的契约；
单个 Jetty 的 tail 不用于推断其他 Jetty 完成。

流水线要求 NPU `--aicpu-blocks` 等于实际分片数；也支持一个 worker 的流水线对照。
不支持多 worker 配置下的单 block pthread 回退；configure 阶段即拒绝该组合。
V11 增加 `verify_payload` 字段，更新 magic、Configure 命令和 kernel 入口，
必须重新构建 **CPU 程序、NPU 程序、device .so 和 JSON**，安装后重启 NPU 测试进程。

```bash
bash build.sh aicpu-cpu
bash build.sh aicpu-npu
bash tests/run_aicpu_get_npu.sh --install-kernel --device-id 7 \
  --aicpu-blocks 4 --profile-scatter
```

CPU 端（跨机器时将 peer 改为 NPU 主机地址）：

```bash
bash tests/run_aicpu_get_cpu.sh --peer 127.0.0.1:29991 --device-id 7 \
  --transfer-mode scatter_pipeline --rows 4096 --row-bytes 656 --scatter-threads 4 \
  --remote-sgl-size 16 --jettys 4 --prepare-threads 4 \
  --warmup 5 --iterations 1000 --output aicpu-pipeline.csv
```

重启 NPU 进程，用相同参数改成 `--transfer-mode scatter` 做基线。
流水线 engine 为 `aicpu_memcpy_sgl_reordered_pipeline`。CSV 将两列改为
`first_ready_ns`（request 到最早观察到任意分片 ready）和 `pipeline_ns`
（首片 ready 到全部 scatter 和最终 producer done），二者之和仍为 `mean_ns`。
`pipeline_ns` 包含后续分片等待和重叠执行，**不是纯 scatter**。
额外输出 `cpu_notify_ns`（每轮分片 ready 加最终 done 的提交/等待耗时）、
`mean_data_batches` 和 `mean_notify_wrs`。`cpu_write_ns`/prepare/post/poll/mean_wrs
累加本轮所有分片的 **数据** WriteBatch，通知 WR 单独计量。
这些 CPU 时间不能与设备流水线时间直接相加，跨机器的时钟也不能直接相减。

启用 profile 时，每核 `avg_start_us`/`avg_finish_us` 均相对 request 起点；
invalidate/memcpy/clean 和 thread CPU time 仅覆盖实际 scatter。
SUMMARY 的 pool 包含 ready 等待，envelope 表示最早 scatter 开始到最后 scatter 结束，
cpu_parallelism 表示该 pool 区间内的有效 scatter CPU 时间比。
输出冷缓存准备、逐轮校验和打印仍在端到端计时外。

正确性检查可加 `--verify-payload --warmup 1 --iterations 10`：CPU 每轮改变所有
payload 字节，设备在计时外逐字节检查并验证 padding，最终 host readback 也使用对应
sequence 的期望值。它能发现仅行头更新而其余字节过旧的结果。
该开关对三种模式都可用；源数据生成在 request 等待内，**仅用于验证，性能对照应关闭**。
分片发送会增加控制及提交开销，同时通信与 scatter 可能竞争访存资源，收益以
`mean_ns`/p99 和各阶段实测为准。本版是每核一个完整分片，尚未实现分片内的 256/512 行细分。

无 NPU 的协议测试：

```bash
cmake --build build-aicpu-cpu --target kvaf_aicpu_pipeline_test kvaf_aicpu_get_protocol_test
ctest --test-dir build-aicpu-cpu -R 'kvaf_aicpu_(pipeline|get_protocol)' --output-on-failure
```

### 可切换的直接输出（V10）

CPU 端新增 `--transfer-mode scatter|direct_output`，默认 `scatter`，沿用 V8 的复制和
缓存准备路径。选择 `direct_output` 时，URMA WRITE 的落点就是最终 output；
AICPU 只发布请求、等待完成、执行完成屏障，不启动 scatter 线程池或多 block 团队。
这仍是独立带宽测试，不接入生产 Runtime，也不接收调用方 Tensor。

两种模式使用相同的 CPU 源行 stride（656 B 时为 768 B），相同的反序逻辑映射和
按 output slot 排列的 SGL，以及相同的 CPU `WriteBatch`、全部 Jetty CQ 等待、done 发布
和退出握手。直接模式将测试 target 的已注册 HBM payload 用作最终 output，复用同一
RA context、导出 UB segment 和远端 WRITE 通道；不另分配 staging 或复制到其他输出。
远端传输继续使用 `--peer NPU_HOST:29991`；远端 CPU/NPU 的 URMA 数据连接需可达。

默认直接输出 stride 为 `row_bytes`，656 B 行紧密排列，便于合并相邻目标。
可加 `--direct-output-stride 768` 保留行间空洞；只 WRITE 有效字节，空洞保持 `0xA5`，
该配置下每行成为一个数据 WR。此开关用于观察目标空洞的代价，不等同于实现动态 mask。
scatter 模式不接受非零 `--direct-output-stride`。源布局、有效字节数及源顺序均不随该开关改变。

直接 output 前后各有 64 B 边界 guard；最终 readback 验证所有 payload 字节、行间空洞及
边界 guard。首轮/预热/正式轮都保留计时外的 `PrepareRound`，发布 request 后直到完成前
不访问或写脏 payload；逐轮行头校验仍在计时外。CPU 仅在数据 `WriteBatch` 全部完成后
写入 done。该测试的完成屏障不能代替 provider 的远端可见性保证，实际机器仍需验证。
测试通过也不代表已验证后续 AIV 算子的消费可见性或生产 Tensor 的注册生命周期。

V10 扩展 Config 和测试 Configure 帧，更新 magic、命令及 kernel 入口，避免加载旧版本。
当前版本需重新构建两端并安装 V14 kernel；安装后重启 NPU 测试进程。V9 tile 融合未包含在此版本中。

```bash
bash build.sh aicpu-cpu
bash build.sh aicpu-npu
bash tests/run_aicpu_get_npu.sh --install-kernel --device-id 7 \
  --aicpu-blocks 4 --profile-scatter
```

另一个终端运行直接输出：

```bash
bash tests/run_aicpu_get_cpu.sh --peer 127.0.0.1:29991 --device-id 7 \
  --transfer-mode direct_output --rows 4096 --row-bytes 656 --scatter-threads 4 \
  --remote-sgl-size 16 --jettys 4 --prepare-threads 4 \
  --warmup 5 --iterations 1000 --output aicpu-direct.csv
```

重启 NPU 脚本（无需再次安装），用相同参数把 `--transfer-mode` 改为 `scatter` 做对照；
再重启可测试 `direct_output --direct-output-stride 768`。直接模式自动将实际 blockDim
设为 1，日志保留 `requested_block_dim`；`--scatter-threads 4` 仍记录为请求值，
`active_threads=1` 表示唯一协调核。`--profile-scatter` 明确输出 SKIPPED，不生成虚假的 profile。

CSV 列不变，直接模式 engine 为 `aicpu_urma_direct_output`，`scatter_ns=0`，
`wait_ns=mean_ns`，完成屏障和控制开销都计入 wait。缓存准备及校验仍不计入这些数字。
对比 `mean_ns`、`cpu_prepare_ns`、`cpu_post_ns`、`cpu_poll_ns` 和 `mean_wrs`；
紧密目标的 WR 数受 SGE/消息长度限制，是否与原 staging 相近以实测为准。

### 4096 × 656 B、多线程测试

更新后需重新构建两端、安装 V14 kernel，并重启测试进程：

```bash
bash build.sh aicpu-cpu
bash build.sh aicpu-npu
bash tests/run_aicpu_get_npu.sh --install-kernel --device-id 4
```

另一个终端：

```bash
bash tests/run_aicpu_get_cpu.sh --peer 127.0.0.1:29991 --device-id 4 \
  --rows 4096 --row-bytes 656 --scatter-threads 4 \
  --warmup 5 --iterations 100
```

每轮有效载荷 2,686,976 B（2.5625 MiB），带宽不计 padding/guard。
分别测试线程数 1、2、4、8；每次先重启 NPU 脚本，后续无需重复安装。
V6 实现 B 版优化：输出只 clean 每行有效 payload 覆盖的 cache line，
包含非对齐尾部所在的整条 line，跳过完全未写入的 guard line。
4096×656 B、stride=768 B 时，输出 `dc cvac` 从 49,152 次降为 45,056 次（减少 8.3%）；
每个 worker 仍只在全部输出 clean 指令之后执行一次 `dsb sy`。
V7 在 B 版基础上实现 C 版优化：656 B 行通过十组完全展开的 64 B NEON load/store
加一个 16 B 尾部复制，恰好读写 656 B；行长判断在每个 worker 的行循环外完成。
其他行长保留通用 `memcpy`，输入 invalidate、B 版输出 clean、调度和每轮校验保持原样。
pthread 和 multi-block 共用此路径；未加入 tile 融合或预取。
启动日志显示 `scatter_clean=payload_cache_lines`，656 B 时显示 `scatter_copy=neon_656`，
其他行长显示 `scatter_copy=libc`；`KERNEL_STATUS` 应显示 `revision=11`。
单线程基线也应使用本版本重新测量，CSV 列和 engine 名称保持不变。
线程调度、完成等待、staging invalidate 和输出 clean 计入 scatter；
创建/销毁线程、数据校验和下面的输出缓存准备不计入。

### 每轮输出缓存起点（V8）

旧版 scatter 结束后，block 0 会 invalidate 整个 output 并读取所有行头校验序号。
这虽然在计时区间外，却会将行头重新加载到校验核的缓存，改变下一轮写入的缓存状态。
V8 在每轮计时开始及发布 request 之前，对整个 output 执行 `dc civac` 和末尾的
`dsb sy`，消除初始化、上一轮 scatter 和校验留下的输出缓存驻留。
首轮、预热轮和正式轮都执行；准备后直到 scatter 前不再访问 output。
单 block 的调用线程和 multi-block 的 block 0 都使用同一个 `PrepareRound`。
其他 worker 等待派发，不在准备过程中访问 output；输入 staging 的 invalidate 仍在
计时区间内，逐轮 tag 校验和最终逐字节 payload/guard 校验继续保留。

启动日志增加 `output_cache=cold_before_request cache_prepare=outside_timing`。
这定义了固定的输出冷缓存测试口径；缓存准备和校验的时间均不计入 `mean_ns`、
`wait_ns`、`scatter_ns` 或 profile 的三段时间，因此不是包含准备/校验的整轮吞吐。
重新测量相同 V8 口径下的单核/多核数据；V7 及更早数据的输出缓存状态不同，
不宜将两者的 scatter 差异直接归因于复制优化。

### 定位四核复制降速：独立 scatter probe

新增独立入口 `KvafAicpuScatterProbeV1`。它使用与正常 scatter 相同的
缓存行操作/DSB 序列、`CopyScatterRows` 和 payload clean，直接分配本地设备源/目标内存，
不建立 URMA、不启动 CPU peer、不执行逐轮 payload 校验。probe 使用独立 ABI，
正常传输入口当前为 V14。源由 Host 初始化，目标按冷缓存口径准备；
源的 invalidate 在每轮被测阶段内，保证复制开始前输入失效。

每个 block 固定负责 `rows_per_worker` 行。套件内的源/目标地址与分配大小保持不变，
packed 只改变目标行距。每轮先等待所有 block 完成计时外的输出缓存准备，
然后执行测试。除了 pipeline，操作前后有阶段屏障；serial 系列再按 worker
顺序逐个执行操作。阶段等待不计入 `avg_op_us`，原始时间戳保留等待形成的间隔。

先构建并安装更新后的 `.so` 和 JSON，独立 probe 不需要重建 CPU 程序：

```bash
bash build.sh aicpu-npu
bash tests/run_aicpu_get_npu.sh --install-kernel --device-id 7 \
  --probe-scatter-blocks 4 --probe-rows-per-worker 1024 \
  --probe-row-bytes 656 --probe-iterations 200 --probe-warmup 5 \
  --probe-scatter-case all --probe-trace-rounds 3 2>&1 | tee scatter-probe-4.log
```

保持每核 1024 行，改变并发数：

```bash
for n in 1 2 4; do
  bash tests/run_aicpu_get_npu.sh --device-id 7 \
    --probe-scatter-blocks "$n" --probe-rows-per-worker 1024 \
    --probe-row-bytes 656 --probe-iterations 200 --probe-scatter-case all \
    --probe-trace-rounds 3 2>&1 | tee "scatter-probe-${n}.log"
done
```

| case | 操作与用途 |
| --- | --- |
| `pipeline` | 各核独立 invalidate→copy→clean，复现当前三段交错 |
| `phased` | 全部 invalidate 完成→并行 copy→全部 copy 完成→并行 clean，排除阶段交错 |
| `serial_copy` | 同一批 block、相同分片逐个 copy，其余核等待；每个 turn 追加计时外 DSB，再释放下一核 |
| `copy_fenced` | phased copy 后逐核追加一次 `dsb sy`，单独记录 store drain，检查复制返回与提交完成的区别 |
| `packed` / `packed_serial` | 仍使用同一逐行复制实现，目标行距由 768 改为 656；比较连续输出的扩展性 |
| `read` / `read_serial` | 只读 packed 输入，计算 16 个字节 lane 的 XOR 校验；比较读路径并发与串行 |
| `write` / `write_serial` | 只写 output 的有效字节，padding 不变；比较写路径并发与串行 |
| `compute` / `compute_serial` | 相同次数的寄存器整数运算；检查是否连计算控制也出现并发降速 |

`avg_op_us` 对 copy case 相当于原来的 memcpy 阶段。`avg_op_cpu_us` 是包围操作的
线程 CPU 时间，包含边界时钟调用的少量开销，不包含 drain 和阶段屏障。
serial 系列与 copy_fenced 的 `avg_drain_us` 单独计量 DSB；DSB 不能代替 output clean，
也不表示此时所有脏缓存行已写回 HBM。
线程 CPU 时间接近墙钟时间只能说明线程持续占用 CPU，不能排除硬件访存停顿。
CPU 时钟或核号查询失败明确报告 errors/unavailable；`frequency` 是计时器频率，
不能当作 AICPU 实际执行频率。

invalidate/clean 额外拆成 `avg_*_issue_us`（缓存行指令循环）和 `avg_*_dsb_us`
（末尾 DSB），用于区分循环执行成本与完成等待。issue 段本身也可能因缓存维护
队列反压而停顿，不能理解为纯指令发射时间；额外时间戳会给诊断带来少量开销。

SUMMARY 的 `avg_op_envelope_us` 每轮取操作最早开始到最晚结束再平均，
`avg_sum_worker_op_us` 则是各 worker 操作耗时之和；两者不等价。
`avg_max_overlap`/`max_overlap` 由逐轮真实区间计算。串行系列若出现操作重叠直接失败。
`avg_round_envelope_us` 包含阶段间等待和 clean，未包含最前面的缓存准备和最终统计。
`payload_GBps` 只按有效字节/操作区间估算，不包含缓存分配额外流量；在写入缓冲尚未
排空的 case 中也不能解释为完整 HBM 写回带宽。compute case 不报告有意义的带宽。

`TRACE` 默认输出前 3 个正式轮次，各 worker 的 invalidate/op/drain/clean 起止时间
都相对该轮最早 invalidate 起点。日志、输出逐字节/guard 校验、read/compute 的
逐轮 checksum 校验和 device readback 均在 kernel 同步完成后进行。
核号在每轮操作前后采样，affinity 在循环外查询；边界采样不能检测迁出再迁回。

判断顺序：

1. 查看核采样和 overlap，确认并行 case 实际同时运行在不同 CPU；比较 serial 与
   parallel 时核分配应相同或可对应，不只看请求 block 数。
2. `serial_copy` 每核明显快于 `phased`：支持并发访存导致减速；仅凭这一项不能区分
   共享缓存、互联、内存端口、访存延迟或功耗/频率影响。
3. `phased` 明显快于 `pipeline`：invalidate/clean 与其他核 copy 交错是重要因素。
4. read 或 write 的并行/串行比更差：分别指向读路径或写入/分配路径；这些控制的
   指令混合与 copy 不同，因此用于定位扩展性，不能直接相减得到 copy 各部分耗时。
5. packed 明显改善并发/串行比：目标布局和缓存行访问是重要因素，但不能据此认定
   写分配是唯一原因；compute 也同比降速则需继续检查执行频率/核资源。

再用 16、64、256、1024 行/核重复 `all`（656 B 时必须是 4 行的整数倍），观察小
工作集与大工作集的扩展性；如果仅大工作集降速，进一步支持缓存之外的访存限制。
可用 `--probe-timeout-ms 1000` 调整屏障超时，上限 10000；不能同时调度全部 block
时会报错，不把串行调度结果当作并行。iterations+warmup 最多 2000，默认 205 轮。

先跑未启用 PMU 的套件，再在同样命令中添加 `--probe-pmu`。probe 尝试通过设备
Linux `perf_event_open` 为当前线程建立 cycles、instructions、cache_references、
cache_misses、backend_stalls 的硬件事件组。打开、启停和读取均在操作墙钟计时之外；
事件区间包围操作及少量用户态边界采样。PMU 不是默认性能口径，小工作集尤其容易
受到这些额外系统调用及核间启动错峰影响。

每个事件输出有效轮数、原始均值、按 enabled/running 比例估算的 scaled 均值和
`avg_running_fraction`。比例明显低于 1 时，复用缩放不能作为精确硬件结论；为 0
表示事件组没有实际运行，也会显示 unavailable。不支持的事件或权限不足报告
`open_errno`，启停/读取错误报告 `read_errno`，不会将缺失计数冒充 0。缺少 perf
头文件或 syscall 能力也保留完整普通 probe，PMU 显示 unavailable。

事件组必须整体能被 PMU 调度；若 running_fraction 为 0，可用 `--probe-pmu-mask 3`
只采集 cycles/instructions，再用 11 添加 cache_misses 或用 19 添加 backend_stalls。
mask 的 bit 0..4 对应上述五个事件；`--probe-pmu` 默认选择全部事件（31）。
enabled/running 按逐轮差值计算，因为 RESET 不重置这些累计时间。
计数格式和组语义参考 [Linux perf_event_open](https://man7.org/linux/man-pages/man2/perf_event_open.2.html)。

相同复制代码下 instructions/row 应接近；如果四核 cycles/row、cycles/instruction
和 backend_stalls/row 同时明显升高，而 compute 控制稳定，更支持访存停顿。
若 instructions/row 本身大幅改变，先检查迁移、插桩或执行路径。泛化的 cache
事件含义依赖平台 PMU，不能把 cache_misses 直接换算成 HBM 读流量。确定具体
共享端口仍需芯片对应的互联/内存控制器计数器。

这个 probe 隔离了本地复制，不复现 DMA 新写入后的完整缓存/一致性状态。如果
pipeline 与真实 URMA 测试明显不同，需返回真实传输链路继续排查。确定具体硬件
端口或 cache 层级还需要目标平台可用的 PMU/互联计数器；这些控制测试不会把推断
冒充硬件计数器结论。

无 NPU 的检查目标：

```bash
cmake --build build-aicpu-cpu --target kvaf_aicpu_scatter_probe_test
ctest --test-dir build-aicpu-cpu -R '^kvaf_aicpu_scatter_probe$' --output-on-failure
```

### 实际运行核与 scatter 三段计时

在 **NPU 端**增加 `--profile-scatter`，默认关闭。升级需重新编译并安装 V14 kernel，
独立的 V14 入口可避免旧 device `.so` 静默使用旧测试口径。
两端需一起升级测试控制协议；原有 CSV 列保持不变。先启动 NPU 端：

```bash
bash build.sh aicpu-npu
bash tests/run_aicpu_get_npu.sh --install-kernel --device-id 7 \
  --profile-scatter 2>&1 | tee aicpu-profile-t4.log
```

另一个终端启动 CPU 端：

```bash
bash build.sh aicpu-cpu
bash tests/run_aicpu_get_cpu.sh --peer 127.0.0.1:29991 --device-id 7 \
  --rows 4096 --row-bytes 656 --scatter-threads 4 \
  --remote-sgl-size 16 --jettys 4 --prepare-threads 4 \
  --warmup 5 --iterations 1000 --output aicpu-t4.csv
```

分别测 1、2、4、8 线程时，每次重启 NPU 脚本并修改日志文件名，后续不必重复安装。
保持 CPU 端 OpenMP/绑核环境一致。测试结束后，NPU 日志输出 `[AICPU_PROFILE] SUMMARY`
及每个 worker 的 `[AICPU_PROFILE] WORKER`，所有均值只覆盖正式测量轮次，不含预热。
worker 0 是 kernel 调用线程，其余是持久 pthread。

| 字段 | 含义 |
| --- | --- |
| `allowed_cpus` | 测试循环前在该 worker 上读取的 affinity；这是允许集合，不是实际运行核 |
| `cpu_hits=CPU:次数,...` | 每个正式轮次的工作前后各采样一次 `sched_getcpu()`；不是各核驻留时间/利用率 |
| `first_cpu` / `last_cpu` | 首个/最后一个有效核采样；`-1` 表示无有效采样 |
| `boundary_cpu_changes` | 同一轮开始与结束采样核不同的次数；不能捕获两次采样间迁出又迁回 |
| `avg_invalidate_us` | 该 worker 输入区的 `dc civac` 循环和末尾 `dsb sy` |
| `avg_memcpy_us` | 该 worker 逐行复制循环；656 B 为 NEON，其他行长为 memcpy |
| `avg_clean_us` | 该 worker 输出区的 `dc cvac` 循环和末尾 `dsb sy` |
| `avg_start_us` / `avg_finish_us` | 相对本轮 pool dispatch 的三段工作起止时间均值，可看线程启动延迟/错峰 |
| `avg_thread_cpu_us` | 工作区间的线程 CPU 时间，有效轮次均值；包括包围三段工作的少量计时开销 |
| `avg_pool_us` | 从调用线程派发到全部 worker 返回的墙钟时间 |
| `avg_dispatch_us` | 派发至最早 worker 开始三段工作 |
| `avg_envelope_us` | 最早 worker 开始至最后 worker 完成三段工作的时间，包含区间内调度等待 |
| `avg_join_us` | 最后 worker 完成三段工作至 pool 返回，含收尾统计/同步 |
| `cpu_parallelism` | 所有 worker 线程 CPU 时间之和 / pool 墙钟时间，近似平均同时运行线程数；不是配置线程数 |

`avg_pool_us = avg_dispatch_us + avg_envelope_us + avg_join_us`。
多线程阶段可能重叠，不能把各 worker 三段耗时相加后与 scatter 墙钟直接比较。
三段是墙钟耗时，线程被抢占也会计入；通过线程 CPU 时间辅助区分执行与调度等待。
`cpu_parallelism` 接近 1 只说明平均并行度低，需要结合允许核集合判断是否限制在单核。

affinity、核采样、线程 CPU 时钟分别报告 `affinity_errno`、`cpu_sample_errors/errno`、
`cpu_time_rounds/errors/errno`。查询失败显示 `unavailable`，不冒充 CPU 0 或零耗时。
核分布支持 CPU ID 0..1023，超出范围计入 `cpu_out_of_range`；affinity 缓冲不足等查询错误显式报告。
若任一 worker 缺少正式轮次的线程 CPU 时间，汇总 `cpu_parallelism=unavailable`。

所有打印和 affinity 查询都在正式计时循环之外；轮内核采样、CPU 时钟和统计本身仍有开销，
`avg_dispatch_us/avg_join_us` 不能视为纯锁开销，CSV 的 scatter 也会包含插桩开销。
诊断后去掉 `--profile-scatter` 重跑性能基线。默认模式不执行这些额外采样或阶段计时。

### blockDim 核探测与 multi-block scatter

`--scatter-threads` 只是在**一个** AICPU block 内创建 pthread，不保证 pthread 会被
AICPU 调度器放到不同物理核。先用独立 probe 验证 Runtime 对 `blockDim` 的实际调度；
probe 不建立 CPU/NPU 通道，也不会触发远端 WRITE：

```bash
bash build.sh aicpu-npu
bash tests/run_aicpu_get_npu.sh --install-kernel --device-id 7 \
  --probe-aicpu-blocks 4 --probe-hold-ms 20
```

只有日志中的 `[AICPU_CORE_PROBE] SUMMARY` 同时满足
`claimed=4 completed=4 distinct_cpus=4 max_overlap=4 multicore_ready=1`，才说明这台
设备的 4 个 block 确实并发占用了 4 个不同的 AICPU CPU。每个 `BLOCK` 行还会给出
开始/结束 tick、两个 `sched_getcpu()` 样本和允许的 affinity 集合。若
`multicore_ready=0`，不要开启 multi-block；应先检查 AICPU 资源配额、CPU affinity
和同设备上的其他 AICPU/HCOMM 负载。

probe 通过后，将 NPU 服务切到 multi-block。`--aicpu-blocks N` 必须等于本轮
`active_threads`，也就是 `min(--scatter-threads, 可用 64 B 行组数)`；这样每个 block
恰好拥有一个不重叠的 scatter 区间。4096×656 B、4 路的例子：

```bash
# NPU 端：每一组测试重新启动一次
bash tests/run_aicpu_get_npu.sh --device-id 7 --aicpu-blocks 4 \
  --profile-scatter 2>&1 | tee aicpu-block4.log

# CPU 端
bash tests/run_aicpu_get_cpu.sh --peer 127.0.0.1:29991 --device-id 7 \
  --rows 4096 --row-bytes 656 --scatter-threads 4 \
  --remote-sgl-size 16 --jettys 4 --prepare-threads 4 \
  --warmup 5 --iterations 1000 --output aicpu-block4.csv
```

multi-block 下 block 0 仍独占 request/done、tag 校验和每轮计时；其余 block 只执行
自己的 invalidate、memcpy、clean。所有 block clean 完输出后 block 0 才发布下一轮，
并在发布前完成计时区间外的输出缓存准备，
因此 staging 和输出正确性检查与单 block 模式相同。启动栅栏只有 1 秒：若 Runtime
没有同时调度请求的 block，测试会返回 timeout，而不会在 CPU 通道上等待完整的每轮超时。
`--aicpu-blocks 1` 保持原有 pthread 模式，适合做基线对比。

在 multi-block 的 `--profile-scatter` 输出中，`WORKER id` 指 AICPU block/range，
不再指 pthread。`cpu_parallelism`、三段计时和 `cpu_hits` 的定义保持不变；成功的
multi-block 运行应与 probe 的核分布相互印证。

### SGL 按 output slot 重排

逻辑行 `r` 的最终位置仍是 `output slot = rows - 1 - r`。CPU 不再将
`host row r` 写到 `staging slot r`，而是构造 `SGL[slot]` 指向
`host row (rows - 1 - slot)`，同时保持远端偏移连续：
`staging + slot * row_bytes`。AICPU 随后执行 `staging slot → output slot`
的正向 copy。这个变换不增加 WR、不会写入 host stride 中的 guard 字节，最终 tag、
payload 与 guard 语义不变。

运行日志会显示 `SGL_LAYOUT=source_by_output_slot` 和
`scatter_layout=forward`，可与旧布局的基线区分。

统计逻辑、输出 clean 覆盖范围和复制正确性的无设备测试：

```bash
cmake --build build-aicpu-cpu --target kvaf_aicpu_get_protocol_test kvaf_aicpu_scatter_profile_test kvaf_aicpu_core_probe_test kvaf_aicpu_scatter_cache_test kvaf_aicpu_scatter_copy_test
ctest --test-dir build-aicpu-cpu -R '^kvaf_aicpu_(get_protocol|scatter_profile|core_probe|scatter_cache|scatter_copy)$' --output-on-failure
```

复制测试覆盖源/目的地址全部 64 种 line 内偏移、16 B 尾部、输出 guard、重复更新、
其他行长回退和空分片；Linux/macOS 上另用保护页检测最后一行越界读写。
AArch64 主机运行真实 NEON；其他架构以同一展开复制逻辑模拟 16 B load/store，
不能代替目标 AICPU 的指令执行验证。protocol 测试也使用实际的分片复制函数。
protocol 测试另外检查首轮/预热/正式轮均先准备缓存再发布 request、先校验再准备
下一轮，并通过增加准备/校验模拟耗时，验证它们不进入计时样本和轮内等待超时。
直接模式测试检查不调用 scatter、只报告一个协调核、完成屏障计入 wait、无 scatter 计时，
并覆盖超时/未来序号/陈旧数据、紧密和带空洞目标及实际 WRITE planner 的合并和越界检查。

## 执行流程

1. CPU 与现有 CpuRuntime 共用 `CpuUrmaDiscovery` 和 `SelectCpuUrmaDeviceLink`，
   通过 `PeerLink::GetHbmEidCandidates(device_id)` 查询 NPU 候选，逐对尝试建链。
   NPU 测试进程使用现有 Runtime 共用的 CLOS/pair_info 发现函数，按选定候选
   分配/注册 HBM，向 CPU 导出 region 并处理 CreateChannel。
2. CPU 主线程初始化已有 `CpuUrmaTransport`，预注册离散源和控制缓冲区。
3. NPU 启动一次 AICPU 算子，在算子内部跑 warmup + iterations 轮。
4. 每轮 AICPU 先在计时区间外清空 output 缓存驻留，再开始计时并发布
   `request_seq`，CPU 用 `ReadSglBatch` 轮询它。
5. CPU 更新每行 8 字节序号，调用 `WriteBatch` 写入 staging 或直接 output，等待返回，再用
   `WriteSglBatch` 写 `done_seq`。
6. AICPU 等待序号。scatter 模式 invalidate staging、memcpy scatter、clean 输出；
   direct_output 模式仅执行完成屏障。检查各行本轮序号后才请求下一轮，不存在两轮覆盖 payload。
7. CPU 停止/销毁 transport 并确认 drained；NPU 检查输出后才释放 target。

CPU source 的行步长始终为 round_up(row_bytes, 64) + 64。scatter 输出采用相同步长，
NPU staging 保持连续；direct_output 默认紧密排列，可指定目标 stride。
逻辑输出仍按反向行序排列，但 CPU 已将 SGL 重排为 output 物理 slot 顺序，因此
AICPU 的 staging 与 output 访问均为正向。行长至少 8 B（序号头），支持 656 B；
此时行步长为 768 B。
线程按 64 / gcd(row_bytes, 64) 行分组，656 B 时每组 4 行，避免线程间共享输入缓存行。
实际线程数不超过分组数。源/目标描述符固定，不包含动态 AccessTable 解析。

## 快速构建和运行（推荐）

升级共享发现逻辑版本后，必须重新编译并重启 CPU/NPU 两端程序；旧程序不支持新的
候选查询/配置流程。
**当前缓存指令修复版本需要重新安装 kernel**：若旧版本出现 `stream=507000`，并伴随
CPU 等待请求超时，应更新、重新编译两端，再用 `--install-kernel` 启动 NPU 端。
旧 kernel 的 `dc ivac` 在 EL0 不可执行，已改为参考 AICPU 算子使用的 `dc civac`。
仅更新 host 程序不会替换已安装的 device `.so`。

在两端的仓库根目录执行。`build.sh` 自动查找 CANN/UMDK；UMDK 未安装时
沿用已有脚本逻辑获取固定版本。已有 UMDK 可以通过 `KVAF_UMDK_ROOT` 指定，
已有 liburma 可以通过 `KVAF_URMA_LIBRARY` 指定。两个新模式不构建 Python 或 AIV。

CPU 端构建：

```bash
bash build.sh aicpu-cpu
```

NPU 端加载本机 CANN 环境后构建，首次运行或重新构建 kernel 后显式安装：

```bash
bash build.sh aicpu-npu
bash tests/run_aicpu_get_npu.sh --install-kernel --device-id 4
```

安装需要 CANN 目录写权限；不加 `--install-kernel` 不修改安装目录。
启动脚本自动选择构建目录内的 JSON，默认端口 29991。
CPU 端先验证小规模正确性（替换机器地址，device-id 为 NPU 逻辑设备号）：

```bash
bash tests/run_aicpu_get_cpu.sh --smoke \
  --peer NPU_HOST:29991 --device-id 4
```

小规模测试结束后，重新启动 NPU 脚本，再在 CPU 端测量默认 32 MiB/batch：

```bash
# NPU 端
bash tests/run_aicpu_get_npu.sh --device-id 4

# CPU 端
bash tests/run_aicpu_get_cpu.sh \
  --peer NPU_HOST:29991 --device-id 4 \
  --output results/aicpu_get.csv
```

两端的构建/启动脚本均支持 `--build-dir DIR`。CPU 启动脚本的 `--smoke` 设置
16 行×64 字节、预热 1 轮、测量 3 轮；显式 `--rows` 等参数可覆盖这些值。
其他 benchmark 参数原样传递。`--output` 同时显示和保存 CSV，保留测试失败退出码。
环境变量 `KVAF_AICPU_CPU_BUILD_DIR` / `KVAF_AICPU_NPU_BUILD_DIR` 可设置启动脚本
默认构建目录，默认分别为仓库下 `build-aicpu-cpu` / `build-aicpu-npu`。
自定义交叉编译器使用 `KVAF_AICPU_CXX`；安装根目录可用启动脚本的 `--cann-root` 指定。

## 手动 CMake 构建

CPU 端需要 liburma、UMDK headers 和 OpenMP：

```bash
cmake -S . -B build-aicpu-cpu \
  -DBUILD_TESTING=ON -DKVAF_BUILD_REMOTE_CPU_SERVICE=ON \
  -DKVAF_URMA_ROOT=/path/to/umdk
cmake --build build-aicpu-cpu --target kvaf_aicpu_get_cpu -j
```

NPU 端需要 CANN（包含 `aclrtBinaryLoadFromFile`、
`aclrtLaunchKernelWithHostArgs`）、HCCP/RA、UMDK headers，以及 CANN AICPU 交叉编译器：

```bash
cmake -S . -B build-aicpu-npu \
  -DBUILD_TESTING=ON -DKVAF_BUILD_AICPU_GET_TEST=ON \
  -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" \
  -DKVAF_URMA_ROOT=/path/to/umdk
cmake --build build-aicpu-npu --target kvaf_aicpu_get_npu -j
```

编译器默认在 CANN 的 `toolkit/toolchain/hcc/bin/`；若安装布局不同，设置
`-DKVAF_AICPU_CXX=/absolute/path/aarch64-target-linux-gnu-g++`。
该开关不要求启用 AIV kernel 或 NPU Runtime service。

显式安装 device kernel（需要 CANN 目录写权限，构建不会自动安装）：

```bash
bash tests/aicpu_batch_get/install_kernel.sh \
  build-aicpu-npu/tests/aicpu_batch_get "$ASCEND_HOME_PATH"
```

脚本安装测试专属 tar/JSON 并添加 `ascend_package_load.ini` 条目。已有配置首次
修改前备份为 `.kvaf-test.bak`。设备的自定义算子加载/签名策略需符合本机 CANN
要求；脚本不改变验签配置。卸载时删除测试 tar/JSON 及该 name 对应的配置段，
不要用旧备份覆盖后来其他安装程序增加的配置。

## 运行

先启动 NPU 端：

```bash
build-aicpu-npu/tests/aicpu_batch_get/kvaf_aicpu_get_npu \
  --port 29991 --device-id 4 \
  --kernel-json build-aicpu-npu/tests/aicpu_batch_get/libkvaf_aicpu_test.json
```

按选定设备读取 CLOS/pair_info 的 NPU EID，与现有 NPU Runtime 共用发现函数；
也可以传 `--eid` 加 32 位十六进制 EID，将测试 NPU 候选固定为一个。
然后在 CPU 数据所在机器运行：

```bash
build-aicpu-cpu/tests/aicpu_batch_get/kvaf_aicpu_get_cpu \
  --peer NPU_HOST:29991 --device-id 4 \
  --rows 8192 --row-bytes 4096 --warmup 5 --iterations 100 \
  --remote-sgl-size 16 --jettys 4 --prepare-threads 4 \
  --wqes-per-doorbell 0 --timeout-ms 30000
```

先用 `--rows 16 --row-bytes 64 --warmup 1 --iterations 3` 验证，再测大 batch。
CPU 默认复用 Runtime 的 CPU UB/EID 枚举和 device-id 候选查询流程，
按 CPU 候选优先、NPU 候选其次的顺序尝试建链，不需要手填设备名或 EID index。
`--device-id 4` 指 NPU 逻辑设备号，绝不会直接用作 CPU EID index。
NPU 端未指定 `--device-id` 时按 CPU 请求选择；指定时会拒绝不一致的请求。
`--ub-device` / `--eid-index` 保留为可选诊断过滤器。建链失败时清理后尝试下一对
候选；成功建链后的数据探测若失败则终止，避免回收可能仍有 DMA 的目标。
每个 NPU 测试进程仅接受一次配置/测量，下一组参数重新启动。

## 计时与结果

成功且最终数据验证通过才输出一行 CSV：

- `e2e_GBps`：有效 payload 字节数 / 平均端到端 ns，十进制 GB/s。
- `mean_ns/p50_ns/p99_ns`：AICPU 发布请求前到模式对应的传输完成边界。
- `wait_ns`：请求发布、CPU 控制轮询、行序号更新、WRITE 和 done 通知。
- `scatter_ns`：scatter 模式的 staging invalidate、AICPU memcpy 和输出 cache clean；
  direct_output 为 0，完成屏障和控制开销计入 wait，`wait_ns=mean_ns`。
- `cpu_write_ns/cpu_prepare_ns/cpu_post_ns/cpu_poll_ns`：现有 WriteBatch 统计。
- `mean_wrs`：仅数据 WriteBatch 的平均 WR 数；`control_reads` 包含预热的控制轮询次数。

初始化、注册、launch、warmup、每轮输出缓存准备、tag 校验和最终逐字节 readback 不在报告的
端到端样本内。每轮 O(rows) 的 CPU tag 更新在端到端内，不在 cpu_write_ns 内。
因此该结果是完整串行握手模式的有效带宽，不是只测 post 的带宽，也不是
多批次重叠时的稳态吞吐。纯 CPU gather WRITE 基线继续使用
`kvaf_cpu_urma_eid_perf`。

## 正确性和平台限制

- 控制记录分离 256 字节，设备按 64 字节 cache line 显式 clean/invalidate；
  cntvct/cntfrq 提供设备侧计时。需要目标 AICPU 支持这些指令及其内存映射语义。
- 每轮更新/验证所有行的序号，最后验证完整 payload 和所有 guard。
- 依赖 provider 的成功 WRITE completion 保证数据对远端可见；全部数据 WR
  完成后才提交 done WRITE。测试结果通过并不能替代对 provider 内存顺序契约
  的确认。如果硬件要求额外 fence，应在得到已验证接口后加入，不能凭空假设。
- 失败不报告有效带宽。CPU 出错时停止 transport，NPU 算子超时退出。
  传输/stream 失败或 NPU 无法收到 CPU drained 确认时保留进程和 target，打印诊断；确认 CPU 已退出后
  再结束 NPU 测试进程，防止在途 WRITE 访问已释放内存。
- 异常终止/断连不是自动恢复场景；修复环境后重启两端。未实现 SDMA、双缓冲、
  graph capture、动态地址列表或 Runtime 接入。

### AICPU local-copy / SDMA capability probe

scatter 模式的远端路径为 CPU URMA SGL gather WRITE 到本地 HBM staging。SDMA 若可用，
只能替换 staging 到 output 的**本地 HBM→HBM** scatter，不能替代 CPU DDR 经 RoCE 的
URMA WRITE。

CANN 9.1.0 已发现 PTO `TPUT_ASYNC`/`TGET_ASYNC` 的 SDMA 头文件，但它不能直接作为
AICPU backend：接口使用 `__gm__` 指针和 AI Core SDMA 指令；现有 kernel 是由
`aarch64-target-linux-gnu-g++` 构建的 AICPU Linux ELF。并且 PTO async SDMA 路径拒绝
多维或非连续 tensor，而 scatter 模式的 `656 B` staging 连续、output 行距 `768 B`，其 scatter
不是单个连续 DMA。拆为 4096 个 SDMA 提交通常会比当前四核 `memcpy` 更差。

下面脚本除扫描安装目录外，还用**同一个 AICPU 交叉编译器**验证 PTO 头能否被 AICPU
内核编译，并打印 PTO 的架构选择宏和连续布局限制：

```bash
bash tests/aicpu_batch_get/probe_aicpu_sdma_api.sh
```

若输出 `aicpu_toolchain_defines___CCE_AICORE__=0` 或
`direct_pto_a2a3_header_compile=fail`，PTO SDMA 只能用于独立的 AI Core kernel，不能
链接或内联到本测试的 AICPU `.so`。不要添加 host `aclrtMemcpyAsync` 作为替代：它不能
从 AICPU 的完成路径提交，也不能保证和 AICPU 请求序号同步。

只有 AICPU 可调用的 submit、completion 和 error-query API 均通过同一 toolchain 验证，
并且存在支持本地 HBM 非连续目的地址的批量/2D SDMA 语义后，才能接入实际 backend。随后
必须验证：URMA WRITE completion 后 staging 对 SDMA 可见；SDMA 支持本地 HBM 地址；完成
事件可由 AICPU 等待；以及异常时 slot 不会被提前复用。

## 本机测试（无需 CANN/URMA）

```bash
cmake -S . -B build-host-aicpu -DBUILD_TESTING=ON
cmake --build build-host-aicpu --target kvaf_aicpu_get_protocol_test -j
ctest --test-dir build-host-aicpu -R kvaf_aicpu_get_protocol --output-on-failure
```

模拟器执行生产 kernel 共用的设备状态循环，覆盖预热/轮次、延迟完成、超时、
过期数据、错误序号、反向 scatter、guard、配置溢出和带宽统计。
它不验证 AICPU 指令权限、CANN 加载或实际 UB/HBM 缓存一致性。

## 故障诊断

`LINK_PROBE ... ok=1` 表示 CPU→HBM 的数据探测已通过，不代表 AICPU kernel 已运行。
失败时保留两端 stderr：

- CPU 超时打印 `expected`（等待轮次）、`observed`（读到的请求序号）、`last_done`。
  例如 expected=2 / observed=1 / last_done=1 表示第一轮 CPU WRITE 已完成，
  AICPU 没有继续发布第二轮请求；expected=1 / observed=0 则尚未观察到首次发布。
- NPU 在 stream 失败时立即打印 `STREAM_FAILED` 和 `aclGetRecentErrMsg()`。
- 尝试读取 `KERNEL_STATUS`：0=pending、1=成功、2=参数错误、3=等待超时、
  4=序号错误、5=数据不匹配。还会打印已完成轮数、失败轮次和计时器频率。
- 如果设备异常导致状态无法回读，打印 `KERNEL_STATUS_UNAVAILABLE`；这不等于状态为 0。

507000 本身不能唯一定位根因。修复后若仍失败，需要上述新日志以及 CANN device/plog
异常记录判断加载失败、指令异常、参数/地址访问或算子主动返回错误。

共享链路选择有独立的 `kvaf_cpu_urma_link_selection` 单测；找到 UMDK headers 时
还会构建 `kvaf_cpu_urma_discovery`，使用真实 API 声明及模拟 provider 验证枚举、
排序、过滤和 URMA 初始化所有权，不依赖 liburma 或硬件。
