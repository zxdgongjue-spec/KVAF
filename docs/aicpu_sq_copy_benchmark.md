# AICPU SQ 拷贝独立基准

本基准从 worker 栈上与 GatherRead 同样大小的 WQE 编码缓冲区，使用同一个 `CopySqWqebbs` 函数，比较写普通 HBM 和写 HCCP/驱动 SQ 的耗时。它创建独占的 USER_CTL SQ/CQ，使用与 GatherRead 相同的深度和创建方式，不连接远端、不写 doorbell、不产生网络 READ 或 CQE。

同一次运行还测试 **HBM → HBM** 和 **HBM → SQ**：HBM 源和 HBM 目标来自两个独立的 `aclrtMalloc` 分配，均使用 `ACL_MEM_MALLOC_HUGE_FIRST`；SQ 仍由驱动分配。AICPU 从源 HBM 顺序读取，每条路径内每个源字节只拷贝一次，使用相同拷贝函数、总字节数和批次大小。Host 每轮提前上传源数据，不计入拷贝耗时；AICPU 不额外初始化或预热源 HBM。两条 HBM 源路径共用源缓冲，前一次读取可能影响后一次读取的缓存状态，因此结果不是严格冷缓存测量。这是 AICPU 执行内存拷贝，不是 `aclrtMemcpy` 或 SDMA 执行的拷贝。

HBM 目标使用 `ACL_MEM_MALLOC_HUGE_FIRST`。每轮总拷贝量固定为 128 KiB，在 workers 间按 64 B 分配；1 worker 和 5 workers 的总字节数相同。默认每次拷贝 1536 B，相当于 8 个 6-SGE WQE，尾块按实际字节数拷贝。目标地址逐轮推进，在队列容量内循环，避免一直写同一个小槽位。

每轮轮换四个测试的执行顺序，减少顺序偏差。栈源缓冲区在每次 kernel 内生成，HBM 源数据由 Host 提前上传，数据随轮次和 worker 改变。拷贝结束使用 `dsb sy`；不增加缓存行刷新。所有 worker 拷贝结束后，再由同一 stream 上的独立校验 kernel 逐 32-bit 字检查目标全部数据，避免校验读取干扰仍在计时的 worker。失败时打印首个不匹配位置，包含 warmup 在内全部正确才打印 PASS。验证 SQ 内存不代表验证网络传输，因为本基准不提交网络操作。

## 执行

不需要 CPU service。在 NPU 机器的仓库根目录执行：

```bash
./build.sh aicpu-urma --install-kernel

for cores in 1 5; do
  ./build-aicpu-urma/tests/kvaf_aicpu_sq_copy_benchmark \
    --device-id 6 \
    --aicpu-cores "$cores" \
    --bytes 131072 \
    --sq-batch-wqes 8 \
    --sq-copy memcpy \
    --warmup 10 \
    --iterations 1000
done
```

程序通过 RA 枚举当前 NPU 的 EID，尝试创建本地 HDC context；不需要选择跨机链路。必要时用 `--eid-list-index N` 指定枚举列表中的索引（不是物理设备号或 `eid_index`）。新的 benchmark kernel 必须重新编译并安装；现有 GatherRead 参数和 ABI 没有变化。

## 计时含义

每种路径分别输出一行 `SQ_COPY_STATS`，分位数和平均值均以 **us** 表示：`source=stack target=hbm`、`source=stack target=sq`、`source=hbm target=hbm` 和 `source=hbm target=sq`。比较四行的 `mean_us`；只比较相同 worker 数、总字节数、批次大小和拷贝方式的运行。

| 数据 | 区间 |
| --- | --- |
| Copy wall / mean_us | 首个 worker 开始拷贝至最后一个 worker 结束拷贝，包含启动时间差和结尾 DSB；晚开始 worker 的准备时间可能体现在启动差中。不含第二个 kernel 的数据校验或 Host 操作 |
| worker copy_us | 单 worker 全部 memcpy 加结尾 DSB 的平均耗时，不含自身源初始化和数据校验 |
| worker start_us | 相对本轮最早开始拷贝的 worker 的平均时间差 |
| thread_cpu_us | 包围拷贝区间的线程 CPU 时间；内存访问停顿仍计入 CPU 时间 |
| offcpu_estimate_us | `max(0, copy wall - thread CPU)`，只用于估计调度暂停，不能理解为内存访问延迟 |
| Host launch + sync | Host 启动到同步返回，包含未计入 Copy wall 的源初始化和数据校验，不包含状态回读 |

若设备不支持线程 CPU 时钟，对应字段显示 `unavailable`，不以 0 冒充有效测量。worker 编号标识独立 SQ 槽位，不是物理 CPU 编号。不同 worker 的区间相互重叠，不能相加。

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

默认 memcpy 分支可能调用外部 libc 的 memcpy；此时 `.so` 中只能看到调用指令，还需要确认运行时 libc 的实际实现。不能仅凭这个 `.so` 的反汇编断言 memcpy 使用了某种 SIMD 指令，也不能仅凭 SQ 的虚拟地址判断物理内存位置或缓存属性。
