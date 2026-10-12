# AICPU SQ 拷贝独立基准

本基准从 worker 栈上与 GatherRead 同样大小的 WQE 编码缓冲区，使用同一个 `CopySqWqebbs` 函数，比较写普通 HBM 和写 HCCP/驱动 SQ 的耗时。设备端统一调用 securec 的 `memcpy_sp`，手写 ARM 拷贝路径已删除。它创建独占的 USER_CTL SQ/CQ，使用与 GatherRead 相同的深度和创建方式，不连接远端、不写 doorbell、不产生网络 READ 或 CQE。

同一次运行也测量 AIV 写入。两个引擎轮流使用同一组已创建的 SQ 和同一组 HBM 目标，地址、字节数、批次大小和目标偏移完全相同。AIV 的本地源在核内 UB 中；对应 AICPU 的本地源在 worker 栈上。AIV 的 UB → SQ/HBM 使用与现有 BatchGet AIV 相同的 `copy_ubuf_to_gm_align_v2`，结尾等待 MTE3 → Scalar 完成事件后结束计时。HBM 源路径包含 MTE2 读入 UB、MTE3 写入目标及必要的 UB 复用等待，使用一个 UB 缓冲，不做双缓冲优化。原有 BatchGet/BatchPut AIV kernel 没有修改。

同一次运行还测试 **HBM → HBM** 和 **HBM → SQ**：HBM 源和 HBM 目标来自两个独立的 `aclrtMalloc` 分配，均使用 `ACL_MEM_MALLOC_HUGE_FIRST`；SQ 仍由驱动分配。AICPU 从源 HBM 顺序读取，每条路径内每个源字节只拷贝一次，使用相同拷贝函数、总字节数和批次大小。Host 每轮提前上传源数据，不计入拷贝耗时；AICPU 不额外初始化或预热源 HBM。两条 HBM 源路径共用源缓冲，前一次读取可能影响后一次读取的缓存状态，因此结果不是严格冷缓存测量。这是 AICPU 执行内存拷贝，不是 `aclrtMemcpy` 或 SDMA 执行的拷贝。

HBM 目标使用 `ACL_MEM_MALLOC_HUGE_FIRST`。每轮总拷贝量固定为 128 KiB，在 workers 间按 64 B 分配；1 worker 和 5 workers 的总字节数相同。默认每次拷贝 1536 B，相当于 8 个 6-SGE WQE，尾块按实际字节数拷贝。目标地址逐轮推进，在队列容量内循环，避免一直写同一个小槽位。

每轮轮换十个测试的执行顺序，减少顺序偏差：原有八条分批路径，加上两条 AICPU HBM 源整段拷贝路径。AICPU 栈源缓冲区在每次 kernel 内生成；HBM 源数据由 Host 提前上传，AIV 的 UB 源在计时前由 MTE2 从这份 HBM 源加载一个批次并等待完成，随后重复写出该 UB 批次。数据随轮次和 worker 改变。AICPU 显式 `dsb sy` 已删除，不增加缓存行刷新，其计时截至拷贝函数返回；AIV 计时包含 MTE3 完成等待。所有 worker 拷贝结束后，再由同一 stream 上的独立 AICPU 校验 kernel 逐 32-bit 字检查目标全部数据，避免校验读取干扰仍在计时的 worker。失败时打印首个不匹配位置，包含 warmup 在内全部正确才打印 PASS。验证 SQ 内存不代表验证网络传输，因为本基准不提交网络操作。

warmup 前额外执行一轮检查，不计入性能统计。每条路径打印 `SQ_COPY_PREFLIGHT`，分别同步拷贝 kernel 和校验 kernel，确认二者都成功后打印该路径的 PASS。正式 warmup 和测量轮次保持一次同步，不增加计时区间内的等待。校验 kernel 将数据不匹配和状态错误写入 `result=2/3` 后正常返回，由 Host 报失败；不再因这类可报告的错误提前中断 stream 并遮住不匹配信息。

若同步失败，`SQ_COPY_STREAM_FAIL` 打印引擎、源、目标、阶段、轮次、ACL 错误码和立即获取的 `aclGetRecentErrMsg()`，随后尝试读回每个 worker 的状态。`507018` 对应 AICPU 执行异常，单独这个错误码不能证明 AIV 搬运失败，也不能区分校验失败、共享库加载失败或处理器异常。特别是 AIV 写入后仍由 AICPU 校验，两者的缓存可见性需要设备结果确认；本测试不通过新增缓存刷新掩盖这个问题。

设备日志已确认旧版 AIV 的首条 UB → HBM 拷贝报 `507015`，故障内核为 `KvafAivSqCopyBenchmark`，错误为标量指令异常；之前的四条 AICPU 路径和 AIV 时钟校准均通过。故障时还没有运行 AIV 写 SQ，不能归因于 SQ。新版删除测试专用 SIMT 模式生成代码，改为纯 MTE2/MTE3 搬运；长度和计数在已验证的 2 MiB 上限内使用 32-bit 运算，避免计时后为统计次数执行动态 64-bit 除法，并补齐与现有 BatchGet 相同的 AIV 内核类型元数据。这些改动需要真实 CANN 构建和设备重跑确认，日志本身尚不能锁定是哪条指令触发异常。

## 执行

程序自动追加 `copy_layout=whole_per_worker` 的 AICPU HBM → HBM 和 HBM → SQ，
不新增命令行参数。每个 worker 只调用一次 `memcpy_sp`，直接使用 Host 提前分配并
上传的 HBM 源，不申请 128 KiB 本地临时缓冲，也不在计时内生成源数据。
日志 `copy_calls=1.00` 用于确认整段拷贝；`chunk_bytes` 是最大单 worker 拷贝长度。

`--bytes 131072 --aicpu-cores 1` 才是一次完整 128 KiB 拷贝。
`--aicpu-cores 5` 仍分摊总计 128 KiB，各 worker 一次拷贝约 25.6 KiB，
不代表各自都拷贝 128 KiB。原有 `copy_layout=batched` 默认每次 1536 B，
可在同一次运行中比较分批和整段写入的耗时。两条整段路径仍完整校验数据，
不敲 doorbell、不测实际网络 WQE 提交；这次不改变正式 GatherRead 的提交实现。

不需要 CPU service。在 NPU 机器的仓库根目录执行：

```bash
./build.sh aicpu-urma --install-kernel

for cores in 1 5; do
  ./build-aicpu-urma/tests/kvaf_aicpu_sq_copy_benchmark \
    --device-id 7 \
    --aicpu-cores "$cores" \
    --bytes 131072 \
    --sq-batch-wqes 8 \
    --sq-copy memcpy_sp \
    --warmup 10 \
    --iterations 1000
done
```

程序通过 RA 枚举当前 NPU 的 EID，尝试创建本地 HDC context；不需要选择跨机链路。必要时用 `--eid-list-index N` 指定枚举列表中的索引（不是物理设备号或 `eid_index`）。新的 benchmark kernel 必须重新编译并安装；现有 GatherRead 参数和 ABI 没有变化。

## 计时含义

每种路径分别输出一行 `SQ_COPY_STATS`，分位数和平均值均以 **us** 表示。`engine=aicpu` 的 `copy_layout=batched` 有四条路径：`source=stack target=hbm`、`source=stack target=sq`、`source=hbm target=hbm` 和 `source=hbm target=sq`；另有两条 `source=hbm copy_layout=whole_per_worker` 路径。`engine=aiv` 仍为原有四条分批路径，对应使用 `source=ub` 或 `source=hbm`。比较相同 worker 数和总字节数下 AICPU 分批和整段写入同一目标的 `mean_us`，以及同样分批大小下两个引擎的耗时。

AIV 记录 `GetSystemCycle()` 原始 tick。程序先用单 AIV worker 执行 200 万和 2000 万 tick 的空等待，各测三次取 Host 最短时间，再用长短时间差除以 tick 差估计 ns/tick，以消除近似固定的启动开销。转换频率和两种等待的 Host 时间打印在 `AIV_COPY_CLOCK` 中。AIV 的 us 和 GB/s 是基于这次校准的估计值，Host 延迟变化仍可能影响精度；不硬编码计数器频率。计时字段在 AIV 返回状态中暂存 tick，由 Host 按差值转换后统计。首次校准不计入 warmup 或测量轮次。

| 数据 | 区间 |
| --- | --- |
| Copy wall / mean_us | 首个 worker 开始拷贝至最后一个 worker 返回 memcpy_sp（AICPU）或确认 MTE3 完成（AIV），包含启动时间差；晚开始 worker 的准备时间可能体现在启动差中。不含第二个 kernel 的数据校验或 Host 操作 |
| worker copy_us | 单 worker 全部 memcpy_sp 调用（AICPU）或含 MTE3 等待的搬运流程（AIV）的平均耗时，不含自身本地源初始化和数据校验 |
| worker start_us | 相对本轮最早开始拷贝的 worker 的平均时间差 |
| thread_cpu_us | 包围拷贝区间的线程 CPU 时间；内存访问停顿仍计入 CPU 时间 |
| offcpu_estimate_us | `max(0, copy wall - thread CPU)`，只用于估计调度暂停，不能理解为内存访问延迟 |
| Host launch + sync | Host 启动到同步返回，包含未计入 Copy wall 的源初始化和数据校验，不包含状态回读 |

若设备不支持线程 CPU 时钟，对应字段显示 `unavailable`，不以 0 冒充有效测量。worker 编号标识独立 SQ 槽位，不是物理 CPU 编号。不同 worker 的区间相互重叠，不能相加。

AIV 的 worker 对应 block ID，不输出 ARM 线程 CPU 时间。测试中的 AICPU 拷贝计时区间没有源码显式 DSB/DMB，只有编译器屏障；正式 GatherRead 中的 acquire/release 原子同步、CPU gather 的屏障和 CQ 缓存维护不参与这个独立测试。securec 库内部的指令仍需检查设备上的实际库。本地 AIV 测试通过异步搬运模拟器验证数据、UB 复用、计时结束前的 MTE3 等待和错误处理，不能证明真实设备性能或替代 CANN 编译。

结果判断：普通 HBM 快、SQ 慢，说明 SQ 访问路径值得继续排查；两者都慢，则检查共同的源缓冲区和 memcpy 实现；worker 拷贝本身快但启动时间差大，则检查 AICPU 调度。比较 1/5 workers 时总字节数固定，不能把吞吐缩放直接等同于纯内存带宽缩放。

## 查看设备实际指令

使用构建 kernel 的同一套 CANN 交叉工具链，反汇编生成的设备 `.so`：

```bash
CANN_CXX="${KVAF_AICPU_CXX:-${ASCEND_CANN_PACKAGE_PATH:-${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++}"
CANN_OBJDUMP="${CANN_CXX%g++}objdump"
"$CANN_OBJDUMP" -d --disassemble=KvafAicpuUrmaSqCopyBenchmark \
  build-aicpu-urma/access_operator/src/aicpu_urma/aicpu_kernels_device/libkvaf_aicpu_urma.so \
  > /tmp/kvaf_sq_copy_benchmark.asm
```

`memcpy_sp` 是 securec 宏，实际展开可能调用 securec 优化函数或其他拷贝实现，需要结合设备头文件和链接的 securec 库确认，不能假定它等同于标准 `memcpy`。不能仅凭 kernel `.so` 的反汇编断言使用了某种 SIMD 指令，也不能仅凭 SQ 的虚拟地址判断物理内存位置或缓存属性。旧参数 `--sq-copy memcpy` 仍可用，日志统一显示 `sq_copy=memcpy_sp`。
