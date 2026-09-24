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
  --sgl-size 8 --remote-offset 4096 --iterations 3
ctest --test-dir build-aicpu-sgl-sim -R '^kvaf_aicpu_sgl_sim_' --output-on-failure
```

`--sgl-size 1` 是单 SGE 基线；例如 1024 行使用 SGL=8 时，每轮由
1024 条模拟 WR 降为 128 条。`--help` 列出全部参数。

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
