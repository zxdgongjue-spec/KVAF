# 对齐 main 与 unified 的异步 Get 计时

在子仓根目录运行以下命令，比较 main 和 execution 分支的四种 Get。
先加载 CANN 环境，并停止占用 21111 端口的旧测试服务。

```bash
git fetch origin
git switch codex/cache-transport-execution
git pull --ff-only origin codex/cache-transport-execution
bash tests/run_batch_get_main_comparison.sh
```

脚本只使用子仓的 `origin`，分别拉取 main 和 execution 最新提交。
每个分支分别构建 CPU、NPU，安装并检查对应 AICPU 包，再后台启动该分支的
CPU service；分支内全部用例共用这个服务，切换分支前停止服务。
成功结束后恢复原分支；失败时清理本次启动的服务，并保留日志供定位。

固定参数为 device 1、端口 21111、8 个 CPU gather 线程和 active 模式。
四种后端分别使用 AIV 2 核、UBMEM 16 核、Gather 5 worker、Scatter 5 worker，
覆盖 2048、4096、8192 个有效请求，每组预热 10 次、正式采样 100 次。
每组均测 eager 和 graph，并保留设备 kernel 时间与对应带宽。

## Eager 只计提交和设备完成等待

main 使用原异步入口；execution 使用新增的 `--get-timing async-wait`。
两边都从 `batch_get()` 提交前开始计时，到 `torch.npu.synchronize()` 返回后停止。
execution 随后仍执行 `drain_get()`，完成状态回读、错误检查和请求租约清理，
再验证输出、读取 profile 并进入下一轮。所有调用者持有的请求 Tensor 均保留到 drain 完成。

这次修改只增加 Python 性能测试的计时选项，沿用已有构建也可单独运行；
比较脚本会为两个分支构建匹配产物，避免混用 AICPU 包和 CPU 服务。
默认 `async-drain` 继续测提交加完整 drain；`sync-call` 继续测一次同步 API 调用。
Graph 保持 replay 加 stream 同步，图资源准备和释放在计时外。

## 从日志区分两种口径

结果保存在 `reports/main-vs-unified-async-wait-d1-p21111-时间/`：

- `main/summary.txt`、`execution/summary.txt`：完整结果及提交、完成分段。
- 各分支的 `commit.txt`：实际测试提交。
- 各分支的 `cpu-service.log`、构建日志和独立用例日志：运行记录。

execution 的 `BATCH_GET_RESULT` 新增 `get_timing` 和
`completion_cleanup_excluded`。eager `async-wait` 标记清理在计时外；
`async-drain` 与 `sync-call` 标记清理在计时内。
Graph 标记 `get_timing=graph-replay-wait`，其清理同样在计时外。
kernel 指标均不计入 Host 收尾处理，原 `profile_readback_excluded=1` 保留。

`PERF_RESULT operation=batch_get_post_wait_drain` 单独报告计时外 drain，单位为毫秒。
它包含 drain 内的完成等待、状态回读与检查、请求清理，是外部同步后的收尾耗时。
它不能当作纯 D2H 拷贝时间，也不等同于默认 `async-drain` 的收尾增量；
各段 P50 不能直接相加得到完整调用的 P50。
