# 对齐 main 与 unified 的异步 Get 计时

在子仓根目录运行以下命令，比较远端 main 与当前工作区的四种 Get。
先加载 CANN 环境，并停止占用 21111 端口的旧测试服务。

```bash
DEVICE_ID=2 bash tests/run_batch_get_main_comparison.sh
```

脚本从子仓 `origin/main` 导出一份源代码快照，execution 直接使用当前工作区，
包括尚未提交的修改。它不切换、重置或提交当前分支，允许在没有新 commit 的情况下验证改动。
两边分别构建 CPU、NPU，安装并检查对应 AICPU 包，再后台启动该版本的
CPU service；分支内全部用例共用这个服务，切换分支前停止服务。
成功或失败都清理本次启动的服务，保留源码快照、构建产物和日志供定位。

默认使用 device 1，`DEVICE_ID=2` 选择 device 2，无需编辑脚本。
端口固定为 21111，使用 8 个 CPU gather 线程和 active 模式。
四种后端分别使用 AIV 2 核、UBMEM 16 核、Gather 5 worker、Scatter 5 worker，
覆盖 2048、4096、8192 个有效请求，每组预热 10 次、正式采样 100 次。
每组均测 eager 和 graph，并保留设备 kernel 时间与对应带宽。

## Eager 只计提交和设备完成等待

main 使用原异步入口；execution 使用新增的 `--get-timing async-wait`。
两边都从 `batch_get()` 提交前开始计时，到 `torch.npu.synchronize()` 返回后停止。
execution 随后执行 `drain_get()`，完成等待和请求租约清理，再验证输出、
读取 profile 并进入下一轮。AIV/UBMEM eager 完成路径不再回读或检查状态报告；
AICPU 的完成检查仍由相应 provider 执行。
所有调用者持有的请求 Tensor 均保留到 drain 完成。

修改 provider 完成路径后需重新构建 NPU 客户端；比较脚本为两边构建匹配产物，
避免混用 AICPU 包和 CPU 服务。
默认 `async-drain` 继续测提交加完整 drain；`sync-call` 继续测一次同步 API 调用。
Graph 保持 replay 加 stream 同步，图资源准备和释放在计时外。

## 从日志区分两种口径

结果保存在 `reports/main-vs-unified-async-wait-d设备号-p21111-时间/`：

- `main/summary.txt`、`execution/summary.txt`：完整结果及提交、完成分段。
- 各分支的 `commit.txt`：实际测试提交。
- `execution/working-tree.patch` 和 `working-tree.txt`：工作区相对提交的改动及状态。
- `main-source/`：被测 main 源码快照及独立构建目录。
- 各分支的 `cpu-service.log`、构建日志和独立用例日志：运行记录。

execution 的 `BATCH_GET_RESULT` 新增 `get_timing` 和
`completion_cleanup_excluded`。eager `async-wait` 标记清理在计时外；
`async-drain` 与 `sync-call` 标记清理在计时内。
Graph 标记 `get_timing=graph-replay-wait`，其清理同样在计时外。
kernel 指标均不计入 Host 收尾处理，原 `profile_readback_excluded=1` 保留。

`PERF_RESULT operation=batch_get_post_wait_drain` 单独报告计时外 drain，单位为毫秒。
它包含 drain 内的完成等待和请求清理，AICPU 还包含状态回读与检查，是外部同步后的收尾耗时。
它不能当作纯 D2H 拷贝时间，也不等同于默认 `async-drain` 的收尾增量；
各段 P50 不能直接相加得到完整调用的 P50。
