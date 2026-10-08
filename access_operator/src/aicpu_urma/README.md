# AICPU URMA BatchPut / BatchGet / GatherRead

这个可选后端不创建第二条链路：沿用 KVAF `NpuRuntime::LinkTo` 的候选 EID/PeerLink
选择，以及现有 Direct-HCCP Host 侧的 HBM/Host DDR 注册、远端导入、USER_CTL
Jetty、SQ/CQ 和 token。AICPU 只从现有 transport 的 entity/queue 表生成并提交
WRITE（BatchPut）或 READ（BatchGet）WQE，等待 CQ，写回结果。

BatchPut/BatchGet 目前采用**单 AICPU worker、单 Jetty**，每 32 个请求提交一次带 completion 的
批次；AIV 模式保持默认不变。BatchPut 同步等待并验证状态；BatchGet 保留调用方
`synchronize` 语义，开启同步时检查 AICPU 状态。不支持与同一 Jetty 上 AIV
kernel 并发使用；异步 BatchGet 需要调用方在释放输出/transport 前同步 stream。

## 构建与部署

在有 CANN toolchain 的 NPU 主机上：

```bash
./build.sh aicpu-urma --install-kernel
```

`build.sh` 自动检测 CANN 根目录；`--install-kernel` 显式安装到同一目录，
需要写入权限。安装后会检查压缩包内容、JSON、加载配置中唯一且完整的条目，
并对比安装文件与构建产物；检查失败则命令以非零状态退出。不传该选项时只构建，
不修改 CANN。检查与卸载不重新编译：

```bash
./build.sh aicpu-urma --check-kernel
./build.sh aicpu-urma --uninstall-kernel
```

这两个命令只作用于**当前主机**的 CANN 根目录。构建主机与运行主机不同时，
必须在运行主机安装相同的包；`ascend_package_load.ini.kvaf-aicpu-urma.bak`
只是安装前备份，不代表当前加载配置仍含 KVAF 条目。卸载仅移除 KVAF 的条目和
两个安装文件，不用备份覆盖其他包的配置。若 `ASCEND_CANN_PACKAGE_PATH`
和 `ASCEND_HOME_PATH` 指向不同 CANN 根目录，构建/检查会直接报错。

当前 `NpuRuntime` 仍复用 AIV BatchGet/BatchPut 的 Direct-HCCP Host 目标，
所以构建 AICPU 客户端会连带编译其 AIV kernel 依赖；客户端本身只执行 AICPU 路径。

原生客户端默认使用构建时的
CANN 路径查找 AICPU JSON，并使用单 worker、单 Jetty；无需配置 `CANN_ROOT`
或提供 AIV kernel object 路径。若 `ASCEND_HOME_PATH` 未设置，客户端在 ACL 初始化前
将其设为构建时的 CANN 根目录，以便设备包加载器找到安装包；显式设置的值不覆盖。
默认服务端是 KVAF 原有 CPU Runtime 服务；跨 NPU 模式使用 `kvaf_npu_service`。

Host 单测（含部署脚本的本地模拟测试）：
`ctest --test-dir build-aicpu-urma -R 'kvaf_aicpu_urma_|kvaf_aicpu_gather_read_|kvaf_cpu_gather' --output-on-failure`。
无需 CANN 的机器也可编译并运行 WQE、gather 和内核握手单测。

真实 CPU/NPU 回读测试：CPU 端先运行 `./build.sh cpu` 和
`build-cpu/runtime/kvaf_cpu_service --listen 0.0.0.0 --port 28999`；
NPU 端安装内核并重启测试进程后执行：

```bash
build-aicpu-urma/tests/kvaf_aicpu_urma_client \
  --peer CPU_HOST:28999 \
  --device-id 0 --blocks 16 --block-bytes 4096 \
  --warmup 1 --iterations 5
```

同机 CPU service 使用默认监听地址和端口时，客户端不带参数也可运行：
`build-aicpu-urma/tests/kvaf_aicpu_urma_client`。其他 peer 用 `--peer` 覆盖；
内核安装在其他目录时可用 `--kernel-json` 覆盖。默认 JSON 缺失或不可读时，
Runtime 在枚举 EID 前直接报错。

`--blocks` 为每批 KV block 数；`--block-bytes` 为每个 block 的写入字节数。
`--token-bytes` 默认等于最终的 `--block-bytes`；只有需要将 block 拆为多个
BatchGet token 时才指定，且 `--block-bytes` 必须能被它整除。客户端会对每个
block 的所有 token 执行 BatchGet，反向排列输出并逐字节比对，因此覆盖完整
BatchPut payload 和 BatchGet READ。
CPU DDR 分配大小固定为 `blocks * block-bytes`。
每轮先写入新 payload，再读取并验证；warmup 不计入统计，也不输出每轮成功日志。
完成后汇总 Put/Get 的平均、最小、p50、p95、p99、最大耗时和平均
`put_GBps/get_GBps`。Put 计时包含 MetaManager 的预留和提交；Get 计时包含
Runtime 提交与 stream 同步，不包含 AccessTable 准备、HBM 上传/回读和逐字节校验，
因此不是纯 SQ 带宽。底层 Direct-HCCP 的诊断日志仍可能出现。
失败时退出码非零且不会输出 `AICPU_URMA_PASS`；`--help` 查看完整参数。

## NPU → NPU HBM BatchPut / BatchGet

原生客户端支持 A 卡的 AICPU 通过 URMA 读写 B 卡的 HBM。B 端复用现有
`kvaf_npu_service`：Host 负责分配、注册 HBM 和交换连接描述符；数据由 UB
传输硬件在两张卡的 HBM 之间搬运。此模式使用普通 BatchPut/BatchGet。

在 B 所在主机启动 HBM 服务（已有服务产物可直接使用；构建命令为
`./build.sh npu --build-dir build-npu`）：

```bash
build-npu/runtime/kvaf_npu_service \
  --listen 0.0.0.0 --port 29888 --device-id 1
```

在 A 所在主机构建、安装内核并运行：

```bash
./build.sh aicpu-urma --install-kernel
build-aicpu-urma/tests/kvaf_aicpu_urma_client \
  --peer NPU_B_HOST:29888 \
  --device-id 0 --peer-device npu --peer-device-id 1 \
  --blocks 16 --block-bytes 4096 --token-bytes 256 \
  --warmup 1 --iterations 5
```

同机双卡时把 `NPU_B_HOST` 换为 `127.0.0.1`，两个进程分别选择卡 0 和卡 1。
跨主机时两端逻辑卡号可以相同；卡号按各进程的设备可见性解释。
`--device-id` 始终选择发起端 A，`--peer-device-id` 选择 B 服务进程中的目标卡。
`--peer-device` 默认 `cpu`；选择 `npu` 后远端卡号默认 0。

Runtime 查询 B 的 HBM EID 候选，与 A 的本地 EID 组合逐对尝试。每次尝试申请
B 的 HBM region，复用 Direct-HCCP 远端 segment 导入和双方 Jetty/CTP 握手，
再以 AICPU WRITE/CQ 探测链路。失败会先销毁传输资源并释放 region，再尝试下一对。
连接成功后，客户端每轮更换源数据，BatchPut 写 B 的 HBM，再用 BatchGet 将全部
数据读回 A 的不同输出 buffer，反序排列 token 并逐字节校验。
成功行包含 `AICPU_URMA_PASS ... peer_device=npu peer_device_id=1`。
这验证传输往返，不验证 B 侧计算核并发消费；后者还需要双方通知和 stream 同步协议。

控制协议 v6 将 HBM segment 的内存 EID 与 B 端 context 的路由 EID 分开传递。
CTP 必须使用双方选定的 context EID；WQE 和 segment 导入继续使用 provider
导出的内存描述符。两个 EID 可能不同，不能相互覆盖。升级时需要同时重建并重启
客户端和 HBM 服务；CPU 客户端也需要与服务端使用同一协议版本。

服务端 `HBM_EXPORTED` 打印 `memory_eid` 和 `route_eid`；客户端
`DIRECT_CTP_READY` 的 `peer_route_eid` 应与服务端的 `route_eid` 相同。
客户端 WQE 日志仍应显示 memory EID 的反序编码。
若首条 WRITE 返回 `cq_status=2 cq_sub_status=2`，它表示错误 CQE，不是超时；
该子状态的硬件含义尚未确认，路由修正后仍需回读验证。若日志停在资源销毁阶段，
`DIRECT_JETTY_DESTROY_BEGIN/END` 可用于判断哪条 Jetty 的 provider 销毁调用未返回。

若路由 EID 已对齐，第一条 WRITE 仍返回 `2/2`，继续检查内存注册和 WQE。
HBM 服务现在与独立 HBM 测试端一致，导出 `RaCtxLmemRegister` 返回的 token ID，
而不是注册前 `RaCtxTokenIdAlloc` 的返回值。`HBM_TOKEN` 同时打印
`allocated_raw`、`registered_raw`、`segment_raw`、`exported_raw` 和右移 8 位后的
`wqe_token`。三个 raw token 是否一致需要实机日志确认；若一致，单独修正导出来源
不会改变本次 WQE 的 token 值，也不能据此宣称已解决 CQ 错误。

设置 `KVAF_AIV_VERBOSE_DIRECT_HCCP=1` 后，发起端的 `DIRECT_MEMORY_TOKEN` 打印本地
源 HBM 的分配、注册及实际使用的 token。AICPU WRITE 收到错误 CQE 时还会保存
并输出 `PUT_ERROR_FIRST_WQE` / `PUT_ERROR_LAST_WQE` 的 16 个原始 32 位字，
以及 `PUT_ERROR_CQE` 的前 8 个字。它们是实际 SQ/CQ 内容，不是 Host 重建值。
多 WQE 批次的首尾记录不一定就是具体出错的 WQE；初始化探针只提交一条 WQE，
此时两份记录应完全相同。诊断使用状态结构的保留空间，不改变其大小。
更新该诊断必须执行 `./build.sh aicpu-urma --install-kernel` 并重启客户端。

客户端日志若暂时停在 Jetty 销毁，而同一次运行的服务端随后出现旧 session 释放和
下一 session 建链，说明候选扫描已有后续进展，应收集到客户端最终退出的完整日志。

C++ 使用同一 Runtime API，在 `LinkTo` 前配置：

```cpp
kvaf::NpuRuntimeOptions options("", 256, 16);
options.aicpu_urma_kernel_json_path = kernel_json;
options.device_id = 0;
options.peer_device = kvaf::DeviceKind::kNpu;
options.peer_device_id = 1;
options.npu_core_count = 1;
options.jettys_per_core = 1;
kvaf::NpuRuntime runtime(std::move(options));
// 检查每个返回的 Status。
runtime.LinkTo({"NPU_B_HOST:29888"});
runtime.Initialize(16 * 4096, 4096);  // B 端 HBM 大小和 slot 大小
// runtime.core().BatchPut(...) / BatchGet(...)
runtime.Shutdown();
```

当前模式访问由 B 服务新分配的 HBM；不导出已有 Tensor，也不要求 B 执行 AICPU
数据处理内核。READ/WRITE 内核及 WQE 格式沿用原实现。调用方需在源数据生产完成后
提交操作，并在异步 BatchGet 完成后才读取输出或销毁资源。双卡 EID 的 UB/CTP
可达性及 HBM 可见性需要目标硬件验证；Host 单测不能替代双卡回读测试。

无需 CANN 的控制面回归测试：

```bash
cmake -S . -B build-host -DBUILD_TESTING=ON
cmake --build build-host --target kvaf_peer_memory_allocator_test \
  kvaf_peer_hbm_eid_test kvaf_aicpu_urma_client_options_test
ctest --test-dir build-host \
  -R '^(kvaf_peer_memory_allocator|kvaf_peer_hbm_eid|kvaf_aicpu_urma_client_options)$' \
  --output-on-failure
```

## CPU Gather + AICPU 多 SGL READ

此路径使用同一个 `NpuRuntime::LinkTo` / Direct-HCCP Jetty，不改变 AIV 算子。
客户端预留独立的 DDR descriptor mailbox；CPU 服务端只在首次 GatherRead 时
通过 RPC 启用直连 worker。每轮源地址清单和 staging 地址先上传到输出 HBM 的附加空间，
AICPU 用 opcode 3 WRITE 将其写入 DDR mailbox，等 WRITE CQE 后再发 opcode 5
WRITE_WITH_NOTIFY（两个 64 字节 SQ WQEBB），将 mailbox 地址写入控制区并通知 CPU。
CPU worker 检测通知后先写入临时 ACK 标记 `~sequence`，再校验清单、按清单将
DRAM 数据复制到连续 staging，最后写回完成序号。AICPU 在同一条 ACK READ 轮询中
记录首次看到临时标记的时间，用于区分通知响应延迟与后续 gather/ACK 可见性延迟。
AICPU 用 READ 轮询完成序号，匹配后将连续 staging 读入连续
HBM 的离散目标 span。每条 READ WQE 最多带 6 个本地 SGE，占 1 到 3 个 64 字节
WQEBB；队列深度允许时每 64 条 WQE 在末条请求 CQE。单条 WQE 的总数据量不超过 256 MiB，
更长的 span 会拆分。所有目标 span 当前必须落在同一块已注册的 HBM 分配内；
不支持跨独立 HBM 分配的 SGL。未指定 `source_descriptor_address` 的旧调用仍使用
每轮配置 RPC；一个 DDR region 不得混用两种模式。

GatherRead 的 READ WQE 默认先在 AICPU 本地以最多 24 个 WQEBB 为一组编码，
再按 SQ 环绕位置批量写入；原有的每个提交批次仍只敲一次 doorbell、等待一次 CQE。
`--sq-batch-wqes 8/16/32/64` 控制一次 SQ 连续写入最多合并的 READ WQE 数，
默认 8；6-SGE workload 对应最多 24/48/96/192 个 WQEBB。
这是连续 SQ 写入的合并粒度，不改变每个至多 64-WQE 提交批次的 CQE 和 doorbell 边界，
也不改变 SGL 上限或建链。遇到 SQ 环尾仍拆成两段写入，不填充空洞。
每个 READ 任务只准备一次固定 WQE 头，复用 EID、TP 和 token 字段。
编码时覆盖有效字段，仅清零尾部 padding，不再先把整个 WQE 清零。
SGL 规划直接覆盖有效条目，顺便累计 WQEBB 数量；提交时不再重复扫描 SGL。
SQ slot 和 owner 在批次内递增，到环尾时翻转 owner，避免逐 WQE 的除法。
AArch64 SQ 写入按 64 B 使用通用寄存器 LDP/STP，不调用通用 memcpy；
本地编码缓冲区按 64 B 对齐。每个 WQEBB 仍完整写入，包括保留字段和 padding，
仍在门铃之前执行屏障。仅 CQ 保持缓存行维护。
固定头准备计入任务初始化，整体效果以 READ 墙钟耗时评估。
本地 WQE 编码空间最大 12 KiB，另有原来的 SGL 规划数组，不使用设备堆分配。
较大粒度可能减少连续写入段数，实际收益须在设备验证；默认不自动切换。
GatherRead 默认使用一个 worker；`--aicpu-cores` 接受 1 到 8 的任意整数，
可使用 4、5、6、8 对比 READ 并行度。此参数设置 kernel block 数，
不代表固定的物理核心数，也不保证所有 worker 同时开始执行。
建链时为每个 worker 分配独立 Jetty、SQ/CQ，后续轮次复用，不共享 producer index。
Host 在同一 ACL stream 上顺序排入两个 kernel：单 block 的 GatherRead 负责
WRITE 源清单、通知一次 CPU、轮询 gather ACK；随后 N block 的 GatherScatter
执行 READ。各 block 用原子计数获取唯一 worker ID，独立组 WQE、写 SQ、
敲 doorbell、等待自己的 READ CQE。没有跨 block 自旋屏障，也没有设备端
OpenMP 或新建 std::thread；并行 block 由 ACL/AICPU Runtime 调度。
原生客户端默认 `--read-schedule dynamic --read-task-spans 192`：
将目标清单切成每批最多 192 个 span，worker 用原子计数领取一批，READ 完成后
继续领取，直到任务耗尽。4096 个 span 会形成 22 批；SGL=6 且没有超大 span 时，
一个完整批次包含 32 条 READ WQE。可用 `--read-task-spans 96/192/384` 对比粒度，
此参数接受 1..65536。小批次增加领取和 CQ 等待次数，实际收益需设备对比。
控制核在已有的布局校验遍历中生成每批 staging 起点；起点表和独占缓存行的
任务计数器位于输出描述符分配的尾部，容量随描述符缓冲区复用。
每轮重置计数并重建起点，支持变长 span、地址乱序和布局变化。
未领到任务的 worker 直接结束，不等待其他 worker，也不共用 SQ/CQ。
`--read-schedule static` 使用原固定分工：每个 worker 一个连续索引区间，
起点保存在各自 queue state 中。Runtime API 的 `read_task_spans=0` 表示固定分工，
正数表示动态批次最多包含的 span 数。
CPU gather 算法和线程数不受此参数影响；用于准备源数据的 BatchPut/Get 仍为单核。
单核保留原先一次 kernel launch，多核多一次 launch，因此小 batch 可能更慢。
动态调度已覆盖 Host 内核模拟中的变长 span、尾批、SQ 回绕、空任务 worker、
晚启动 worker 和 CQ 错误；真实设备上的性能需实测。

更新动态调度时，需要在 NPU 侧执行 `./build.sh aicpu-urma --install-kernel`，
同时更新 Host 和 kernel；CPU 服务无需因这项调度修改而重编译。
沿用已启动的 CPU 服务，在 NPU 终端对比：

```bash
./build.sh aicpu-urma --install-kernel
for schedule in static dynamic; do
  build-aicpu-urma/tests/kvaf_aicpu_gather_read_client \
    --peer 127.0.0.1:28990 --device-id 6 \
    --blocks 4096 --block-bytes 656 --output-stride 1312 --sgl-size 6 \
    --aicpu-cores 5 --sq-batch-wqes 8 \
    --read-schedule "$schedule" --read-task-spans 192 \
    --warmup 10 --iterations 1000 --profile-detail 1
done
```

比较 READ 整体墙钟耗时、各 worker 的 `read_tasks` / `read_wqes` 和结束时刻。
`claim tasks` 是每轮各 worker 领取任务耗时的最大值再求平均，包含在 READ 总耗时中；
task initialization 是各 worker 累计的批次初始化耗时，再按每轮最大值汇总。

GatherRead 仅在每次轮询 AICPU CQE 前清理并失效对应缓存行；CPU 通知、ack、
descriptor、源 DDR、staging 和 AICPU 的描述符、SQ、HBM 输出、状态缓冲
均不执行显式缓存行维护。仍保留内存屏障以保证提交与完成的顺序。其余缓冲区
依赖目标平台的 CPU DDR、HBM 与 URMA DMA 一致性，不能从注册参数推断该条件
已经满足。Host 模拟无法验证这一点，更新后必须在真实设备上逐轮改变数据并
完整回读校验；若通知、SQ 或状态不可见，需恢复相应维护。
这里没有使用 OpenMP：
CPU worker 使用常驻 `std::thread`。默认 `--gather-mode adaptive`：协调线程
持续轮询远端控制行；辅助线程在完成一轮后先自旋约 100 us，再等待协调线程
下一次准备通知。远端 WRITE_WITH_NOTIFY 不会唤醒本地的 C++ 条件变量，因此
协调线程不能靠定时休眠保证低延迟。`--gather-mode active` 会让协调线程和
辅助线程都无限自旋，`--gather-mode passive` 则让协调线程定时休眠、辅助线程
等待通知，仅用于对照。Linux 上
默认将 coordinator 和 helper 绑到进程允许的不同逻辑 CPU；可用
`--gather-affinity off` 关闭以作对比。`GATHER_THREAD` 日志记录 TID、CPU 和绑核结果。
默认按 CPU 核数自动选择上限 32，并按每线程至少 256 KiB 的
payload 限制实际并行度；服务端可用 `--gather-threads N` 指定上限（1 到 64，
显式指定时阈值降为每线程 64 KiB，设为 1 可作为单线程基线）。
`[KVAF_CPU] GATHER_THREADS` 日志显示实际线程数。
URMA 清单准备将源地址范围、缓冲区重叠和总长度检查与 staging 偏移表构建
合并为一次遍历；不再调用 RPC 路径的两遍校验，也不再单独重建偏移表。
不检查未使用的 span `reserved` 字段。清单仍先复制到 CPU 持有的缓冲区，
每轮重新生成偏移，支持源地址、顺序、数量和长度变化。缓冲区容量与 worker 复用。
这段纯 CPU 准备代码独立编译为 `kvaf_cpu_gather_plan`，使用 `-O2`；
URMA 初始化仍保留原编译方式。更新后执行 `./build.sh cpu` 并重启 CPU 服务，
NPU 客户端和 AICPU kernel 无需重装。对比时间线中的
`CPU read and prepare source list` 与整个通知到 ACK 完成区间。
CPU profiling 的拷贝耗时包含线程分发与同步，需在目标机器上用同一
batch 比较 `--gather-threads 1/4/8/16` 后再选线程数。可用 `taskset`
限制服务进程的候选 CPU 集合，绑核只在该集合内选核。

两端分别运行：

```bash
# CPU 终端
./build.sh cpu
build-cpu/runtime/kvaf_cpu_service --listen 0.0.0.0 --port 28999 \
  --gather-threads 8 --gather-mode adaptive --gather-affinity off

# NPU 终端
./build.sh aicpu-urma --install-kernel
build-aicpu-urma/tests/kvaf_aicpu_gather_read_client \
  --peer CPU_HOST:28999 --device-id 0 \
  --blocks 16 --block-bytes 4096 --output-stride 8192 --sgl-size 6 \
  --aicpu-cores 4 --warmup 1 --iterations 5
```

相同输入对比单核和多核（同机使用以下 peer；跨机器替换 CPU 地址）：

```bash
for cores in 1 2 4 5 6 8; do
  build-aicpu-urma/tests/kvaf_aicpu_gather_read_client \
    --peer 127.0.0.1:28999 --device-id 6 \
    --blocks 4096 --block-bytes 656 --output-stride 1312 --sgl-size 6 \
    --aicpu-cores "$cores" --warmup 10 --iterations 1000 --profile-detail 1
done
```

默认 `--descriptor-transport urma`。同一参数另起一次客户端进程，指定
`--descriptor-transport rpc` 可测旧版每轮 TCP 配置路径；两种模式不能在同一
DDR region 内切换。对比时固定 CPU 线程数、源块数、输出布局、warmup 和轮数。

客户端每轮 BatchPut 不同 payload 到 DDR，按 block 逆序生成 gather 清单，再用
多 SGL READ 将数据写入原 block 顺序的稀疏 HBM 槽位；逐字节校验槽位及空隙，
仅汇总 GatherRead 计时。客户端的输入和输出 HBM 使用
`ACL_MEM_MALLOC_HUGE_FIRST` 优先分配大页；是否实际获得大页由 CANN 决定。
`--sgl-size` 接受 1 到 2^32-1；大于 6 时，
内核自动将请求拆成每条最多 6 个目标 SGE 的 READ WQE，不会发出未经验证的
8-SGE WQE；因此设置大于 6 不会减少 WQE 数量。`--output-stride` 不得小于
`--block-bytes`，默认是两倍 block 大小。客户端从保留 key 的 AccessTable 中选择连续
staging 和独立 mailbox，避免 LinkTo 探针轮转 slot 后与源块重叠；另保留控制区并占住其 slot，
避免 MetaManager 后续复用。调用 `NpuRuntime::GatherRead` 时，源地址、staging 和
`source_descriptor_address` 必须在注册 DDR 范围内且互不重叠，mailbox 需 64 字节对齐，
容量至少为 `32 + 16 * blocks` 字节。末尾 64 字节控制区不可用于 KV slot。
输出 HBM 的最后一个 span 之后需要额外的 64 字节对齐 scratch 空间，再加相同大小的
源地址清单空间；span 不得重叠。一个 region 同时
只允许一个未完成的 gather。GatherRead 计时包含输出布局校验、SGL 描述符上传、
CPU gather 握手和 CQ 等待，不含输入上传、输出回读及逐字节校验。多 SGL READ 的
真实设备兼容性和性能仍需目标机器验证。

客户端先打印端到端统计，再按执行顺序打印 `GATHER_READ_TIMELINE`：Host 上传清单与目标、
等待与收尾；等待期间的 AICPU WRITE 清单、通知、CPU gather、AICPU READ 到 HBM。各值是测量轮次
平均微秒，保留三位小数即纳秒精度。Host 等待包含 CPU/AICPU 工作，AICPU 等 CPU ack
与 CPU gather 重叠，不能相加；AICPU READ 阶段包含 WQE 规划与提交、
完成等待，不等于纯传输时间。profiling 增加时钟采样，并将 ack READ 从 8 字节扩展到
56 字节，因此小 batch 的总耗时不应直接与旧版无 profiling 结果比较。
需要原始纳秒计数器与逐核时序时使用 `--profile-detail 1`；默认只显示分层时间线。更新后需
同时重新部署 CPU service、NPU Runtime 和 AICPU kernel。直连模式 profiling 中的
`idle_since_previous_ack` 从上一轮 ack 到本轮收到通知，包含两轮之间的空闲，
不是通知传递耗时。`CPU start observed after ACK polling began` 是 AICPU 首次读到
临时 ACK 标记的时间；若 CPU 在第一次轮询前就完成，标记可能不会被观察到。
`direct_setup` 包含每轮 mailbox 布局检查，只有首次调用才会启用 CPU worker。
CQ 缓存行维护计入 `ack_cq_ns` / `scatter_cq_ns` 等完成等待时间，不计入
`ack_cache_ns`；其余缓存维护禁用后的正确性和性能尚需目标机器验证，
不能从 Host 模拟推断。

`AICPU READ staging -> HBM` 取所有 worker 最早开始到最后完成的墙钟区间，
包含各 worker 的领号、初始化、WQE 规划、编码提交和 CQ 等待，不是各核耗时之和。
多核日志中的 plan/post/cq 分别取每轮各核最大值；它们可能来自不同核，
互相重叠，不能相加还原 READ 总时间。`read_wqes` 是所有核之和；分片边界
可能产生额外的尾部 WQE。额外 kernel 调度开销仍包含在 Host stream 等待和
GatherRead 总耗时中，不包含在首个 READ worker 开始之前的设备 READ 区间。

`AICPU validate HBM targets and prepare worker starts` 是控制核在通知 CPU 前
执行的一次布局校验与起点计算，包含在 Host 等待中，不在 READ worker 区间内。
`Worker initialization` 覆盖 READ block 进入后的原子领号、状态/队列准备和起点读取。
`Build local SGL` 是按该核目标 span 生成本地 SGE 列表；`Within submit` 的
编码（含批次参数检查）、SQ 批量拷贝、doorbell 与内存屏障均包含在 `Submit WQEs` 中。
`--profile-detail 0` 现在真正关闭每次 SQ 拷贝组的两次额外时钟采样，
编码/SQ/doorbell 子计数器保持 0，日志明确标记禁用，不再打印伪零耗时。
仍保留每核进入/结束、READ 批次 plan/post/CQ、握手和 Host 阶段计时，
因此不是完全无 profiling。`--profile-detail 1` 启用组内采样；采样本身有开销。
各阶段多核最大值可能来自不同核，不能相加。
`Worker start spread` 是最晚与最早 READ block 进入时间之差；`Longest worker duration`
是单核进入到完成的最长区间。两者用于识别调度错开与单核耗时，均在 READ 墙钟内，
不是额外阶段，也不能相加。READ 分组数组仅覆盖实际使用的 SGE，不清零未使用条目。

`GATHER_READ_WORKERS` 每行对应一套 SQ/CQ worker，不代表固定物理 AICPU ID。
start/end 是每轮相对最早 READ worker 进入时刻的纳秒差，输出为微秒；
每行显示平均 start/end/duration、start p50/p95，以及该 worker 的 WQE、SQ 连续写入
段数和 WQEBB 数。各核区间重叠，不能相加；晚进入 worker 的起点集中在早进入
worker 的完成附近，只能作为分波执行的证据，不能单凭均值推断可用物理核数。
`SQ writes` 统计全部 worker 的连续写入段数和实际写入的 64-byte WQEBB，不计控制握手。
逐 worker 的 `sq_write_ranges` 同样表示连续写入段数；一个段可能包含多个 WQEBB。

沿用现有 CPU 服务，对比采样开关和 SQ 拷贝粒度：

```bash
./build.sh aicpu-urma --install-kernel
for detail in 0 1; do
  for batch in 8 16 32 64; do
    build-aicpu-urma/tests/kvaf_aicpu_gather_read_client \
      --peer 127.0.0.1:28990 --device-id 6 \
      --blocks 4096 --block-bytes 656 --output-stride 1312 --sgl-size 6 \
      --aicpu-cores 8 --sq-batch-wqes "$batch" --profile-detail "$detail" \
      --warmup 10 --iterations 1000
  done
done
```

保持输入、CPU gather 配置及 AICPU 核数一致，先用 detail=0 比较总耗时和 READ
墙钟，再用 detail=1 定位写 SQ/编码及逐核启动分布，避免把采样开销当作传输退化。

服务端加 `--gather-notify-profile` 后，直连 CPU worker 退出时输出一行
`GATHER_NOTIFY_POLL`。此采样有开销，正式测带宽时不要启用。
`detect_gap_*`
是读到新通知前最后一个采样窗口的墙钟时间，`detect_offcpu_*` 是该窗口墙钟
减去线程 CPU 时间，`sampled_offcpu_max_ns` 覆盖整个轮询期间的采样窗口。
active/adaptive 每 4096 次轮询采样，passive 每次休眠后采样。active/adaptive
模式的较大 off-CPU 间隔表明协调线程未运行；passive 模式的 off-CPU 时间还包含
主动休眠。off-CPU 字段只在 Linux 上有效。这些值不是 NPU 写入通知到 CPU
看到通知的端到端耗时。

升序或降序且不重叠的 HBM 输出 span 使用线性布局校验；其他顺序仍通过排序
检查重叠。GatherRead 不再刷新输出 span 或 SQ WQE 的 cache line；此变更不改变
WQE、CQ 或链路协议，但要求设备对这些缓冲区的 DMA 访问可见。

**硬件验证状态**：本分支的 Host WQE ABI/模拟队列测试和内核语法检查不等价于 NPU
收发验证。部署后应先用 Runtime 自带的 LinkTo WRITE 探针确认链路，再在真实
CPU DDR/HBM 之间做小 batch 的 put/get 数据回读验证，最后再进行性能测试。

## 排查 AICPU 探针失败

如果日志已出现 `PEER_LINK_JETTY_IMPORT_READY`，但 WRITE 探针在
`aclrtSynchronizeStream` 返回 `507018`，这表示 AICPU 执行异常，不能仅凭
`PEER_LINK_EID_PAIR_FAILED` 判断 EID 不匹配。此时没有从设备拷回状态；
`BATCH_PUT_DEVICE_STATUS_UNAVAILABLE` 表示 Host 侧没有有效的内核状态。
先保留完整的 Host plog 和 AICPU device 日志，查看 `PrintAicpuErrorInfo`、
`HandleTaskTimeout`、算子加载失败或内存访问异常，再区分加载、访问和 CQ 等待问题：

```bash
find "$HOME/ascend/log" -type f -mmin -30 -print 2>/dev/null
grep -R -nE '507018|PrintAicpuErrorInfo|HandleTaskTimeout|KvafAicpuUrma' \
  "$HOME/ascend/log" 2>/dev/null | tail -100
```

使用同一 CPU peer 的纯 AIV WRITE 探针可作为对照，但不能跳过 AICPU 探针直接
选定 EID：这会把 AICPU 执行失败延后到正式 BatchPut。

如果设置 CANN 根目录后变成 `execute kernel param invalid`，说明 `.so` 已加载，
但旧版 AICPU 内核把所有失败都返回为任务错误。新版同步 BatchPut 在状态缓冲区
可用时通过 `BATCH_PUT_DEVICE_ERROR` 输出具体 `result`：1=入参形状，
2=entity，3=队列，4=源地址，5=目标地址，6=SQ 已满，7=CQ 超时，
8=CQ 错误，9=长度；`failed_request` 仅在逐请求参数校验失败时给出索引。

`result=8` 表示 USER_CTL CQ 已返回错误完成，并不等于 EID 建链失败。
AICPU BatchPut 会额外打印 `cq_status`、`cq_sub_status` 和原始
`cqe_word`（保存在状态 ABI 的 `reserved[0]`）。只有收到错误 CQE 时这些字段
才有效；`result=7` 是 CQ 超时，没有错误 CQE。当前日志中的 `result=8` 但
`cq_status=0 cq_sub_status=0` 来自旧版 AICPU 内核未写回 CQE 字段，无法据此
判定具体硬件错误。重新构建、安装内核并重新运行客户端，用单 block 探针采集新值：

```bash
./build.sh aicpu-urma --install-kernel
build-aicpu-urma/tests/kvaf_aicpu_urma_client \
  --peer CPU_HOST:28999 --blocks 1 --warmup 0 --iterations 1
```

对照纯 AIV 使用同一 CPU peer 的探针结果和新 CQE 状态，再区分 token、地址、
WQE 协议或互联故障；不能仅凭当前 `result=8` 归因到其中一种。

最新单 block 日志的 `cqe_word=0x02020006`，即 `cq_status=2`、
`cq_sub_status=2`。这表明硬件返回了错误完成，不是 CQ 超时；两个 NPU EID
候选在同一个 CPU 候选上返回相同的 CQE，但测试在继续扫描 CPU 候选时被中断，
不能据此断言所有 26 个候选都失败。当前没有经过确认的该设备子状态码映射，
不要把 `2/2` 直接解释为某一种 token、地址或 EID 错误。

需要区分共享的 Host-HCCP/WQE 问题与 AICPU 提交路径问题时，单独构建原生
AIV 客户端作对照，无需 Python 或修改 AICPU 客户端：

```bash
./build.sh aiv-native
build-aiv-native/tests/kvaf_aiv_batch_roundtrip \
  --peer 127.0.0.1:28991 --device-id 0 --blocks 1 --iterations 1
```

AIV 原生客户端不调用 `NpuRuntime::Initialize` 的 WRITE 预探针；比较结果前应从
日志确认双方实际选择的 CPU/NPU EID 候选一致。

若 AIV 在同一候选的 WRITE 探针成功而 AICPU 返回 `2/2`，重点检查 AICPU 写 SQ、
缓存可见性和门铃顺序；若两者在同一候选均返回 `2/2`，先检查共享的 WQE 字段、
token、地址与 PeerLink 资源。两种后端的批量 WQE 目前都是单 SGE/WQE，
`--blocks` 不会将一个 WQE 变为多 SGL。

无设备环境下评估 CPU gather 后使用可配置 SGL 发送的映射和理论时间，可运行
[`aicpu_sgl_simulator.md`](../../../tests/aicpu_sgl_simulator.md) 中的 CPU-only
仿真。它假设 AICPU 发送带宽为 25 GB/s，不改变现有 kernel 的单 SGE/WQE 实现。

后续 AIV 对照在 `peer_candidate=9` 的两个 NPU EID 上也得到 `status=2`、
`sub_status=2`。AIV 的 `processed=1` 只表示请求已发出，不能视为 WRITE 成功；
旧版 AIV 的 `cqe_word=0` 是因为未保存原始 CQE。由此可以排除 AICPU 独有的提交路径
是唯一原因，但仍不能从该片段确定 26 个 CPU 候选的最终结果。

共享 WQE 控制字与本地 HCOMM AICPU 实现对照后，曾尝试把 AICPU 和 AIV 的
首轮 owner 改为 0、末条 fence 改为 1；真机 AIV 探针仍返回 `2/2`。
现已将 AIV BatchPut/BatchGet kernel 和 Put Host 注册行为回退到本分支起点的
实现；AICPU WQE 编码仍保留该假设，尚需单独验证。对照时要注意两者目前
不再使用相同的 WQE 控制字。分别运行两个独立客户端的单 block 命令：

```bash
./build.sh aiv-native
build-aiv-native/tests/kvaf_aiv_batch_roundtrip \
  --peer 127.0.0.1:28991 --device-id 0 --blocks 1 --iterations 1
./build.sh aicpu-urma --install-kernel
build-aicpu-urma/tests/kvaf_aicpu_urma_client \
  --peer 127.0.0.1:28991 --device-id 0 --blocks 1 --warmup 0 --iterations 1
```

上述实验中首个 CPU 候选的两个 NPU EID 仍返回 `2/2`，因此 owner/fence 不是该错误的
充分解释。本地 HCOMM A5 状态表将 `status=2, sub_status=2` 标为
`Local Operation Error / Local Access Error`；该表不是当前驱动的实时诊断，
不能仅凭它认定具体的 HBM 地址或 token 有误。AIV 状态不保存原始 CQE，
因此其 `cqe_word=0` 不代表硬件返回 0；AICPU 状态仍保存原始 CQE。
CPU 服务 `REGION_ALLOCATED` 会打印 `token_id` / `seg_token_id` 供对照。
不要把候选扫描中途的片段当作最终结果。
