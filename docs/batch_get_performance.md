# 四种 Get 共用一个性能入口

构建并安装 NPU 产物、启动匹配的 CPU service 后，在仓库根目录运行：

```bash
KVAF_CPU_SERVICE_ADDR=127.0.0.1:21111 DEVICE_ID=2 \
bash tests/run_remote_cpu_batch_get_perf.sh build-npu \
  --backend aiv --aiv-cores 2 --execution both \
  --request-length 32768 --table-capacity 4096 --miss-count 4096 \
  --token-bytes 656 --tokens-per-block 128 --jettys-per-core 2 \
  --seed 20260903 --no-sort-input-indices \
  --source-access-pattern random --output-access-pattern contiguous \
  --warmup 10 --iterations 100 --kernel-profile
```

选择 backend 及其并行度参数，其他负载参数保持一致：

| `--backend` | 执行路径 | 常用并行度参数 |
|---|---|---:|
| `aiv` | URMA AIV Direct | `--aiv-cores 2` |
| `ubmem` | UBMEM AIV Direct | `--aiv-cores 16` |
| `gather_read` | URMA AICPU Gather | `--aicpu-workers 5` |
| `cpu_push_scatter` | URMA AICPU Scatter | `--aicpu-workers 5` |

四种后端共用请求生成、建表、预检、预热、输出校验、eager/graph 测量和结果格式。
`--aiv-cores` 表示 AIV 核数，`--aicpu-workers` 表示 AICPU worker 数（1..8）；
旧 `--workers` 写法仍接受以兼容已有命令。
UBMEM 不使用 Jetty 参数。`--table-capacity` 是源块数量，
`--miss-count` 是本轮有效请求数，二者独立。

## 计时范围

默认 `--get-timing async-drain` 测异步提交到 `drain_get()` 返回，
包括设备完成等待和请求租约清理。
AIV、UBMEM 的 eager 完成路径只等待并释放资源，不清零、回读或检查 Get 状态报告；
同步 `--get-timing sync-call` 使用相同完成规则。ACL 提交和同步错误仍会返回失败。
Get 输出正确性由计时后的 Tensor 校验验证；显式 profile 快照仍读取并验证设备报告。
UBMEM Put 事务确认和 graph 释放时的状态检查有各自用途。

与 main 的异步计时边界对齐时，execution 可使用 `--get-timing async-wait`：
提交后外部同步设备并停止计时，再通过 `PERF_RESULT operation=batch_get_post_wait_drain`
单独报告 drain 耗时（毫秒）。这段包含请求清理，AICPU 还可能包含状态回读和检查。
Graph 固定测 replay 加 stream 等待，资源准备、输出校验和 profile 读取均在计时外。
可在 `--execution graph` 或 `both` 下加 `--multi-graph-check`，在单节点图计时前
验证两张图交替回放、多个 Get 的数据正确性，以及释放其中一张后另一张继续回放。

默认关闭算子 profiling，只输出 `BATCH_GET_RESULT scope=end_to_end` 的 Host 延迟和带宽。
加上 `--kernel-profile` 才开启设备打点、每轮 profile 回读和详细分段输出，
并增加 `scope=kernel` 的设备时间和带宽。eager 和 graph 使用同一开关。
UBMEM 同时开启地址准备、MTE 提交和最终 drain 的细分采样。
仅显式传入 `--kernel-profile` 时启用采集。
关闭 profiling 时没有设备时间戳，因此不输出 kernel 指标。

时间单位为微秒，带宽使用有效载荷一次除以对应耗时；`p50_GBps` 对应 P50 延迟。
上述示例显式启用了 profiling；测无 profiling 的端到端性能时，去掉 `--kernel-profile`。
开启后算子内打点的开销计入测得时间，profile 回读和结果打印在计时外。
阶段可嵌套或重叠，各段 P50 不能相加得到总 P50。

## 扫描负载

扫描使用同一入口的普通循环，例如将单点命令中的 `--miss-count` 替换成
`"$misses"`，外层使用 `for misses in 2048 4096 8192; do ...; done`。
调整 token 字节数或 worker 数时采用相同方式，不再维护独立后端 benchmark、
Host 扫描器、周期扫描器及其专用结果解析格式。

Get 性能使用本文入口，数据预检和输出校验保留在计时之外。
多卡并发复用相同负载和图执行组件，测试独立/并发的持续吞吐；
具体命令和与逐次采样的区别见[四种 Get 的多卡带宽测试](batch_get_multicard_performance.md)。

## Python 入口与公共模块

`tests/python/` 只保留性能入口和 trace 汇总工具：

- `test_npu_batch_get_perf.py`：四种 Get 的统一性能测试。
- `test_npu_batch_put.py`：三种 Put 的同步性能测试。
- `test_npu_runtime_perf.py`：Runtime 各 API 的性能扫描。
- `batch_get_multicard_perf.py`：多卡并发性能组织。
- `summarize_get_dispatch_trace.py`、`summarize_put_dispatch_trace.py`：调度分段汇总。

公共模块统一放入 `utils/`，通过 `from utils.xxx import ...` 使用：
backend 发现、DSA key、NPU EID、请求数据模式、共享参数和校验、执行资源与绑定、
图输入更新、捕获与释放、统计和多卡 worker。
独立功能、配置和单元测试以及配套启动脚本已移除。
