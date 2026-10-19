# AICPU 发起、CPU URMA 回推、AICPU scatter 带宽测试

这是独立 C++ 测试，不调用 Runtime BatchPut，不改 Python 或 MetaManager。

CPU 预构建 SGL/WR，有序 WRITE 发布各 Jetty 的 chunk 进度，AICPU 按范围并行
调用 securec `memcpy_sp`，通信与 scatter 重叠。支持 `--scatter-threads 1..32`。
传输路径固定为离散 CPU 源地址 → 连续 HBM staging → 离散 output slot。
每行保留一个本地 SGE，仅按目标连续性和设备上限聚合为 WR；不做源段合并或 CPU 打包。
默认 4096×656 B、4 核/4 Jetty、chunk 1024、doorbell 32、CQ batch 8/deadline 32，
payload 缓存维护关闭、仅末轮回读校验；CPU caller 局部 -O2，URMA 初始化编译方式不变。
CSV engine 为 `aicpu_sgl_ordered_write`。旧 `--profile-scatter` 及实验模式参数已移除。

### 4096 × 656 B、多线程测试

更新后需重新构建两端、安装 V21 kernel，并重启测试进程：

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
CPU、NPU executable 与 device kernel 必须一起更新；blockDim 自动匹配实际核数。
缓存策略对照使用 `--payload-cache-maintenance on`；逐轮正确性验证使用 `--verify-payload`，
它逐轮修改全部 payload 并回读检查，不用于带宽比较。

## 执行流程

1. CPU 与现有 CpuRuntime 共用 `CpuUrmaDiscovery` 和 `SelectCpuUrmaDeviceLink`，
   通过 `PeerLink::GetHbmEidCandidates(device_id)` 查询 NPU 候选，逐对尝试建链。
   NPU 测试进程使用现有 Runtime 共用的 CLOS/pair_info 发现函数，按选定候选
   分配/注册 HBM，向 CPU 导出 region 并处理 CreateChannel。
2. CPU 主线程初始化已有 `CpuUrmaTransport`，预注册离散源和控制缓冲区。
3. NPU 启动一次 AICPU 算子，在算子内部跑 warmup + iterations 轮。
4. 每轮 AICPU 发布 `request_seq`，CPU 用 `ReadSglBatch` 轮询它。
5. CPU 更新每行序号，复用 prepared WRITE 链：每个 Jetty 的 data 后接强序 ready WRITE。
   按 Jetty 依次提交整条链，沿用 doorbell 分批；只等待各队列最终尾 CQE，随后发送 done。
6. AICPU 每个 worker 等待所有 Jetty 的本轮最小 ready 进度，合并已就绪的连续 chunk
   向前复制；各核完成且 global done 到达后才允许下一轮复用 staging。
7. 默认末轮 CPU 通过 URMA READ 检查 output/guard 并 ACK，AICPU 在 ACK 前不触碰 output；
   CPU 停止 transport 并确认 drained 后，NPU 才释放 target。

CPU source 和输出的行步长为 round_up(row_bytes, 64) + 64，NPU staging 保持连续，
CPU 按 output slot 重排源 SGL，staging 按逻辑反向行序连续落地，scatter 向前访问。行长至少 8 B（序号头），支持 656 B；此时行步长为 768 B。
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
启动脚本自动选择构建目录内的 JSON，默认端口 29991，可用 `--npu-candidate N` 固定远端 EID 候选。需要 NUMA 绑定时，
在启动 CPU 脚本前使用 `numactl --cpunodebind=N --membind=N bash ...`；程序沿用当前进程策略。
CPU 端先验证小规模正确性（替换机器地址，device-id 为 NPU 逻辑设备号）：

```bash
bash tests/run_aicpu_get_cpu.sh --smoke \
  --peer NPU_HOST:29991 --device-id 4
```

小规模测试结束后，重新启动 NPU 脚本，再在 CPU 端测量默认 2.5625 MiB/batch：

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
`aclrtLaunchKernelWithHostArgs`）、HCCP/RA、UMDK headers，以及 CANN AICPU 交叉编译器和 device securec 库：

```bash
cmake -S . -B build-aicpu-npu \
  -DBUILD_TESTING=ON -DKVAF_BUILD_AICPU_GET_TEST=ON \
  -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" \
  -DKVAF_URMA_ROOT=/path/to/umdk
cmake --build build-aicpu-npu --target kvaf_aicpu_get_npu -j
```

编译器默认在 CANN 的 `toolkit/toolchain/hcc/bin/`；若安装布局不同，设置
`-DKVAF_AICPU_CXX=/absolute/path/aarch64-target-linux-gnu-g++`。
该开关不要求启用 AIV kernel 或 NPU Runtime service。securec 自动从 CANN 查找；
布局不同时指定 `KVAF_AICPU_SECUREC_INCLUDE_DIR` 与 `KVAF_AICPU_SECUREC_LIBRARY`，后者须为 AArch64 ELF。

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

- `e2e_GBps` / `mean_ns`：有效 payload / 请求开始到全部 scatter 和 done 完成。
- `first_ready_ns` / `pipeline_ns`：请求到首 ready / 首 ready 到全部完成；两者之和为 mean_ns。
- `cpu_data_GBps` / `cpu_data_span_ns`：首次 data Submit 到观察到全部链尾 CQE，包含标记。
- `cpu_cycle_GBps` / `cpu_round_ns`：CPU 完整轮次，包含等待、源更新及安排的回读校验。
- `cpu_producer_ns`：发送完整数据链和最终 done 的总墙钟时间；不再逐次给 Poll 打点。
- `mean_wrs/mean_marker_wrs/mean_tail_cqes`：数据 WR、标记 WR、实际尾 CQE；max_sge=13 时默认 320/16/5。
- `npu_prepare_ns/npu_verify_ns`：计时外的冷 output 准备/外部校验握手均摊；控制字缓存维护始终保留。

单位为 ns，payload 字节数除以 ns 即十进制 GB/s。通信与 scatter 重叠，不能串行相加。
`cpu_data_GBps` 是数据交付带宽，不是物理链路速率。建议预热 50、测量 1000 轮并独立重复 5 次。
默认 `output_verify=final`，末轮校验按 iterations 均摊；`--output-verify urma` 改为逐轮回读。

## 正确性和平台限制

- 控制记录分离 256 字节，设备按 64 字节 cache line 显式 clean/invalidate；
  cntvct/cntfrq 提供设备侧计时。需要目标 AICPU 支持这些指令及其内存映射语义。
- 每轮更新行序号，默认末轮检查完整 payload 和 guard；`--verify-payload` 单独验证逐轮新数据。
- payload maintenance off 是当前实测平台的快速策略；其他部署可使用 `--payload-cache-maintenance on`。
  计时外的 cold-output 准备仍保留，不能将其误认为 scatter 内的 payload invalidate。
- 有序标记依赖同一 Jetty/目标上的强序 WRITE 语义；每个 worker 等全部队列进度，
  不假设跨 Jetty 全局顺序。全部数据链尾完成后才提交 done WRITE。测试结果通过并不能替代对 provider 内存顺序契约
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

单测覆盖 kernel 共用的 chunk 消费/ready 协议、缓存行范围、旧序号与超时、
CQ credit/乱序完成、源段合并、复制 guard、外部校验握手和带宽统计。
它不验证 AICPU 指令权限、CANN 加载或实际 UB/HBM 缓存一致性。

## 故障诊断

`LINK_PROBE ... ok=1` 表示 CPU→HBM 的数据探测已通过，不代表 AICPU kernel 已运行。
失败时保留两端 stderr：

- CPU 超时打印 `expected`（等待轮次）、`observed`（读到的请求序号）、`last_done`。
  例如 expected=2 / observed=1 / last_done=1 表示第一轮 CPU WRITE 已完成，
  AICPU 没有继续发布第二轮请求；expected=1 / observed=0 则尚未观察到首次发布。
- NPU 在 stream 失败时立即打印 `STREAM_FAILED` 和 `aclGetRecentErrMsg()`。
- 尝试读取 `KERNEL_STATUS`：0=pending、1=成功、2=参数错误、3=等待超时、
  4=序号错误、5=数据不匹配、6=复制错误。还会打印已完成轮数、失败轮次和计时器频率。
- 如果设备异常导致状态无法回读，打印 `KERNEL_STATUS_UNAVAILABLE`；这不等于状态为 0。

507000 本身不能唯一定位根因。修复后若仍失败，需要上述新日志以及 CANN device/plog
异常记录判断加载失败、指令异常、参数/地址访问或算子主动返回错误。

共享链路选择有独立的 `kvaf_cpu_urma_link_selection` 单测；找到 UMDK headers 时
还会构建 `kvaf_cpu_urma_discovery`，使用真实 API 声明及模拟 provider 验证枚举、
排序、过滤和 URMA 初始化所有权，不依赖 liburma 或硬件。
