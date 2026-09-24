# CPU gather + AICPU SGL 仿真

这是不依赖 CANN、URMA 或 NPU 的原生 C++ 测试。它把物理上逆序、相互间有间隔的
CPU 数据行复制到连续 staging buffer，再沿用 KVAF 的
`BuildWriteGatherShard` 分组规则，把每行作为一个本地 SGE，按
`--sgl-size` 指定的**每条 WRITE 最多本地 SGE 数**组成模拟 WR。每条 WR
写入连续的模拟远端地址；程序逐行核对数据，并检查远端目标前后的保护字节。

```bash
./build.sh aicpu-sgl-sim
build-aicpu-sgl-sim/tests/kvaf_aicpu_sgl_simulator \
  --rows 1024 --row-bytes 4096 --source-gap-bytes 64 \
  --sgl-size 8 --gather-threads 8 --remote-offset 4096 --iterations 3
ctest --test-dir build-aicpu-sgl-sim -R '^kvaf_aicpu_sgl_sim_' --output-on-failure
```

`--sgl-size 1` 是单 SGE 基线；例如 1024 行使用 SGL=8 时，每轮由
1024 条模拟 WR 降为 128 条。`--help` 列出全部参数。

`--gather-threads N` 控制 CPU gather 线程数，默认取本机逻辑核数，上限 64；
每个线程复制互不重叠的行分片，worker 在所有迭代间复用。数据不足时按行数和
每线程至少 256 KiB 自动减少线程数，输出同时报告请求值与实际值。
`--gather-threads 1` 是串行基线。`--gather-mode active` 让 worker 在迭代间
保持活跃，通过原子计数同步；默认的 `--gather-mode passive` 使用条件变量唤醒。
`--gather-mode omp` 在编译器支持 OpenMP 时使用与 CPU transport 相同的
`parallel num_threads` 静态分片方式；`OMP_WAIT_POLICY=ACTIVE` 可请求运行时
积极等待，但 OpenMP 不保证线程永久驻留。active 会持续占用 CPU，可能拖慢其它
工作，不保证比 passive 快；三种模式的计时均包含每轮同步，不包含初次创建线程。

同一负载下比较等待策略与线程数：

```bash
for mode in passive active; do
  for threads in 1 2 4 8 16; do
    build-aicpu-sgl-sim/tests/kvaf_aicpu_sgl_simulator \
      --rows 1024 --row-bytes 4096 --sgl-size 8 --iterations 100 \
      --gather-threads "$threads" --gather-mode "$mode"
  done
done
OMP_WAIT_POLICY=ACTIVE build-aicpu-sgl-sim/tests/kvaf_aicpu_sgl_simulator \
  --rows 1024 --row-bytes 4096 --sgl-size 8 --iterations 100 \
  --gather-threads 8 --gather-mode omp
```

最后一条仅在构建检测到 OpenMP 时可用。比较 `gather_avg_ns`，并重复
整组测试；active 的空转会影响其余阶段，不应只凭模拟 `effective_GBps` 选策略。

默认日志保留累计耗时，并增加 `CPU_PREP_PER_ITERATION` 的平均、最小、最大耗时
（单位 ns）、`AICPU_SEND_PER_ITERATION` 的模型耗时及每轮的流水线平均估算。
需要逐轮采样时加 `--log-iterations 1`；`ITERATION_NS` 在计时结束后统一打印，
避免日志输出影响 gather 和分组计时。`send_model_ns` 和 `total_model_ns`
仍是模型估算，不是设备实测。

发送阶段不调用硬件，而是按十进制 25 GB/s 计算
`wire_ns = payload_bytes / 25`；默认每 WR 附加开销为 0，因此仅增大
SGL 不会凭空提高模型发送带宽。可用 `--wr-overhead-ns N` 显式加入
**假设的**每 WR 开销，模型发送时间再加 `WR_count * N`。
`CPU_PREP_MEASURED` 是运行测试的本机实际 gather 和分组耗时；
`PIPELINE_ESTIMATE` 是该耗时加上发送模型，不是部署环境的实测性能。

当前 KVAF AICPU kernel 仍是**单 SGE/WQE**。本测试只验证多 SGL 的数据映射、
WR 分组和理论时间，不证明设备支持该格式，也不计入 CPU staging 到设备可访问
内存的搬运、内存注册、建链、门铃或 CQE 开销。落地前需确认 AICPU 可访问
staging buffer 的方式和真实的多 SGL WQE ABI。
