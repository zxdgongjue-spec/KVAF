# GatherRead SQ 写入

设备端 SQ 写入统一调用 CANN securec 的 `memcpy_sp`，不再使用手写 ARM `LDP/STP` 拷贝。默认参数为 `--sq-copy memcpy_sp`。旧的 `--sq-copy memcpy` 和值为 0 的 API 字段继续兼容，实际调用 `memcpy_sp`；`paired` 已删除。

每个连续 SQ 区间调用一次 `memcpy_sp`，例如 8 条完整的 6-SGE WQE，一次复制 1536 B。拷贝失败会返回错误，不继续敲门铃。

跨 SQ 环尾仍分成两个连续区间。模板、任务分配、WQE 编码、producer/owner 维护、doorbell 和 CQ 轮询沿用原实现。设备 kernel 中的显式 `dsb sy` 已移除，只保留不生成硬件屏障指令的编译器屏障；CQ 缓存行刷新仍保留。

设备构建使用 CANN 的 `securec.h` 和 AArch64 securec 共享库。构建系统检查共享库为 AArch64 ELF，链接时检查未解析符号，并把依赖库随 kernel 一起打包，通过 `$ORIGIN` 查找。若 CANN 的目录布局不同，可用 CMake 的 `KVAF_AICPU_SECUREC_INCLUDE_DIR` 和 `KVAF_AICPU_SECUREC_LIBRARY` 指定设备头文件和共享库位置。本地仿真使用标准内存拷贝，只检查功能，不测量设备 securec 性能。

需要同时重编客户端并安装新版 AICPU kernel，使用 `./build.sh aicpu-urma --install-kernel` 完成。

CPU service 保持原配置运行于 `127.0.0.1:28990` 时，在 KVAF 目录执行：

```bash
./build.sh aicpu-urma --install-kernel

./build-aicpu-urma/tests/kvaf_aicpu_gather_read_client \
    --peer 127.0.0.1:28990 --device-id 6 \
    --blocks 4096 --block-bytes 656 --output-stride 1312 \
    --sgl-size 6 --aicpu-cores 5 --sq-batch-wqes 8 \
    --read-schedule dynamic --read-task-spans 192 \
    --descriptor-transport urma --profile-detail 1 \
    --sq-copy memcpy_sp --pipeline-chunk-bytes 131072 \
    --warmup 10 --iterations 1000
```

末尾 `AICPU_GATHER_READ_PASS` 打印 `sq_copy=memcpy_sp`。比较 `Write SQ`、完整 `AICPU READ staging -> HBM` 和总调用耗时。`Write SQ` 包含全部连续区间复制，不包含门铃和 CQ 等待。设备性能和去掉屏障后的数据正确性需在 NPU 上实测确认。
