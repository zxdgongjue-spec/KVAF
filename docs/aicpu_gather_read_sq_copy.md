# GatherRead SQ 写入对照

客户端增加 `--sq-copy memcpy|paired`，默认 `memcpy`。该参数仅控制 AICPU READ WQE 从 worker 编码缓冲区写入 SQ 的方式。

- `memcpy`：每个连续 SQ 区间调用一次 `std::memcpy`，例如 8 条完整的 6-SGE WQE，一次复制 1536 B。
- `paired`：保留原有手写 ARM `LDP/STP`，每次循环复制 64 B；非 AArch64 本地仿真使用原来的逐 64 B `memcpy`。

跨 SQ 环尾仍分成两个连续区间。两种方式沿用相同的模板、任务分配、WQE 编码、producer/owner 维护、doorbell、屏障和 CQ 轮询；不会清理 SQ 缓存行。模式选择发生在连续区间复制处，不在逐 WQEBB 循环内。

HCOMM 普通 AICPU 路径调用 `memcpy_sp` 写入 SQ，本次对照使用标准 `memcpy`，未引入 HCOMM/securec 链接依赖；不能假设这两个复制函数底层实现完全相同。

`GatherReadArgs` 扩展为 144 B。测试前需要同时重编客户端并安装新版 AICPU kernel，使用 `./build.sh aicpu-urma --install-kernel` 完成。

CPU service 保持原配置运行于 `127.0.0.1:28990` 时，在 KVAF 目录执行：

```bash
./build.sh aicpu-urma --install-kernel

for mode in paired memcpy; do
  ./build-aicpu-urma/tests/kvaf_aicpu_gather_read_client \
    --peer 127.0.0.1:28990 --device-id 6 \
    --blocks 4096 --block-bytes 656 --output-stride 1312 \
    --sgl-size 6 --aicpu-cores 5 --sq-batch-wqes 8 \
    --read-schedule dynamic --read-task-spans 192 \
    --descriptor-transport urma --profile-detail 1 \
    --sq-copy "$mode" --warmup 10 --iterations 1000
done
```

末尾 `AICPU_GATHER_READ_PASS` 打印 `sq_copy` 标识结果。对比 `Write SQ`、完整 `AICPU READ staging -> HBM` 和总调用耗时；`Write SQ` 的计时边界沿用原来区间，包含全部连续区间复制，不包含门铃和 CQ 等待。两种方式都必须通过数据正确性验证。默认选择只是对照实验入口，实际性能仍需设备实测后确定。
