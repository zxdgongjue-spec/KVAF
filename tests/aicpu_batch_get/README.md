# AICPU 发起、CPU URMA 回推、AICPU scatter 带宽测试

这是独立 C++ 测试，不调用 Runtime BatchPut，不改 Python 或 MetaManager。

scatter 在 656 B 行上使用 AICPU NEON 向量复制，其他行长使用 `memcpy`，
支持 `--scatter-threads 1..32`（默认 1）。
CPU 将 SGL 描述符按 output 物理 slot 排列，仍 WRITE 到连续 HBM staging；
AICPU 将同一个 slot 顺序复制到 output。
调用线程也参与复制，其余 pthread 在进入测试循环前创建，跨预热和计时轮复用。
CSV 标注 `aicpu_memcpy_sgl_reordered` /
`aicpu_memcpy_sgl_reordered_mt`，并输出请求与参与分片的线程数。
`active_threads` 不是实际占用的核数；实际核采样需启用下面的诊断开关。
线程创建失败报告 `pthread_error`，不会静默回退。设备 pthread 支持及带宽需在目标机器验证。

### 4096 × 656 B、多线程测试

更新后需重新构建两端、安装 V8 kernel，并重启测试进程：

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
其他行长显示 `scatter_copy=libc`；`KERNEL_STATUS` 应显示 `revision=8`。
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

### 实际运行核与 scatter 三段计时

在 **NPU 端**增加 `--profile-scatter`，默认关闭。升级需重新编译并安装 V8 kernel，
独立的 V8 入口可避免旧 device `.so` 静默使用旧测试口径。
CPU 控制协议和原有 CSV 列保持不变。先启动 NPU 端：

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

## 执行流程

1. CPU 与现有 CpuRuntime 共用 `CpuUrmaDiscovery` 和 `SelectCpuUrmaDeviceLink`，
   通过 `PeerLink::GetHbmEidCandidates(device_id)` 查询 NPU 候选，逐对尝试建链。
   NPU 测试进程使用现有 Runtime 共用的 CLOS/pair_info 发现函数，按选定候选
   分配/注册 HBM，向 CPU 导出 region 并处理 CreateChannel。
2. CPU 主线程初始化已有 `CpuUrmaTransport`，预注册离散源和控制缓冲区。
3. NPU 启动一次 AICPU 算子，在算子内部跑 warmup + iterations 轮。
4. 每轮 AICPU 先在计时区间外清空 output 缓存驻留，再开始计时并发布
   `request_seq`，CPU 用 `ReadSglBatch` 轮询它。
5. CPU 更新每行 8 字节序号，调用 `WriteBatch`，等待返回，再用
   `WriteSglBatch` 写 `done_seq`。
6. AICPU 等待序号，invalidate staging，memcpy scatter，clean 输出。
   检查各行本轮序号后才请求下一轮，不存在两轮同时使用 staging。
7. CPU 停止/销毁 transport 并确认 drained；NPU 检查输出后才释放 target。

CPU source 和输出的行步长为 round_up(row_bytes, 64) + 64，NPU staging 保持连续。
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
- `mean_ns/p50_ns/p99_ns`：AICPU 发布请求前到 scatter 完成并 clean 输出。
- `wait_ns`：请求发布、CPU 控制轮询、行序号更新、WRITE 和 done 通知。
- `scatter_ns`：staging invalidate、AICPU memcpy 和输出 cache clean。
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

当前测试的远端路径固定为 CPU URMA SGL gather WRITE 到本地 HBM staging。SDMA 若可用，
只能替换 staging 到 output 的**本地 HBM→HBM** scatter，不能替代 CPU DDR 经 RoCE 的
URMA WRITE。

CANN 9.1.0 已发现 PTO `TPUT_ASYNC`/`TGET_ASYNC` 的 SDMA 头文件，但它不能直接作为
AICPU backend：接口使用 `__gm__` 指针和 AI Core SDMA 指令；现有 kernel 是由
`aarch64-target-linux-gnu-g++` 构建的 AICPU Linux ELF。并且 PTO async SDMA 路径拒绝
多维或非连续 tensor，而当前 `656 B` staging 连续、output 行距 `768 B`，其 scatter
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
