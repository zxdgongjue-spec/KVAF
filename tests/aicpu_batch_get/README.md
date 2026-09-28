# AICPU 发起、CPU URMA 回推、AICPU scatter 带宽测试

这是独立 C++ 测试，不调用 Runtime BatchPut，不改 Python 或 MetaManager。

scatter 使用 AICPU `memcpy`，支持 `--scatter-threads 1..32`（默认 1）。
调用线程也参与复制，其余 pthread 在进入测试循环前创建，跨预热和计时轮复用。
CSV 标注 `aicpu_memcpy` / `aicpu_memcpy_mt`，并输出请求与参与分片的线程数。
`active_threads` 不是实际占用的核数；实际核采样需启用下面的诊断开关。
线程创建失败报告 `pthread_error`，不会静默回退。设备 pthread 支持及带宽需在目标机器验证。

### 4096 × 656 B、多线程测试

更新后需重新构建两端、安装 V3 kernel，并重启测试进程：

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
本版本按线程批量 clean 输出，单线程基线也应使用本版本重新测量。
线程调度、完成等待及缓存操作计入 scatter；创建/销毁线程、数据校验不计入。

### 实际运行核与 scatter 三段计时

在 **NPU 端**增加 `--profile-scatter`，默认关闭。升级需重新编译并安装 V3 kernel，
本次新增了 kernel 参数，独立的 V3 入口可避免旧 device `.so` 静默忽略诊断参数。
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
| `avg_memcpy_us` | 该 worker 逐行 memcpy 循环 |
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

统计逻辑的无设备测试：

```bash
cmake --build build-aicpu-cpu --target kvaf_aicpu_scatter_profile_test
ctest --test-dir build-aicpu-cpu -R '^kvaf_aicpu_scatter_profile$' --output-on-failure
```

## 执行流程

1. CPU 与现有 CpuRuntime 共用 `CpuUrmaDiscovery` 和 `SelectCpuUrmaDeviceLink`，
   通过 `PeerLink::GetHbmEidCandidates(device_id)` 查询 NPU 候选，逐对尝试建链。
   NPU 测试进程使用现有 Runtime 共用的 CLOS/pair_info 发现函数，按选定候选
   分配/注册 HBM，向 CPU 导出 region 并处理 CreateChannel。
2. CPU 主线程初始化已有 `CpuUrmaTransport`，预注册离散源和控制缓冲区。
3. NPU 启动一次 AICPU 算子，在算子内部跑 warmup + iterations 轮。
4. 每轮 AICPU 发布 `request_seq`，CPU 用 `ReadSglBatch` 轮询它。
5. CPU 更新每行 8 字节序号，调用 `WriteBatch`，等待返回，再用
   `WriteSglBatch` 写 `done_seq`。
6. AICPU 等待序号，invalidate staging，memcpy scatter，clean 输出。
   检查各行本轮序号后才请求下一轮，不存在两轮同时使用 staging。
7. CPU 停止/销毁 transport 并确认 drained；NPU 检查输出后才释放 target。

CPU source 和输出的行步长为 round_up(row_bytes, 64) + 64，NPU staging 保持连续，
输出按反向行序 scatter。行长至少 8 B（序号头），支持 656 B；此时行步长为 768 B。
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

初始化、注册、launch、warmup、每轮 tag 校验和最终逐字节 readback 不在报告的
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
