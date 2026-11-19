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
未显式设置并行度时，AIV Direct 使用 2 核、UBMEM 使用 16 核、Gather/Scatter 使用
5 个 worker，与 Runtime 的算法默认值一致；命令行参数或环境变量可覆盖。
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

Graph 的 E2E 使用同一 stream 上 replay 前后两个 NPU Event 的间隔，
资源准备、输入更新及同步、输出校验和 profile 读取均在计时外。
可在 `--execution graph` 或 `both` 下加 `--multi-graph-check`，在单节点图计时前
验证两张图交替回放、多个 Get 的数据正确性，以及释放其中一张后另一张继续回放。
Gather/Scatter 额外检查未回放节点不能误报完成；配合 `--kernel-profile` 还验证
未回放节点没有可读 profile、各节点 profile 独立，以及其他图回放不改写其报告。
回归命令见[执行框架验证说明](cache_transport_execution_validation.md)。

默认关闭算子 profiling，`BATCH_GET_RESULT scope=end_to_end` 在 eager 下报告 Host 延迟和带宽，
在 graph 下报告 Event 间隔和对应带宽。
加上 `--kernel-profile` 才开启设备打点、每轮 profile 回读和详细分段输出，
并增加 `scope=kernel` 的设备时间和带宽。eager 和 graph 使用同一开关。
UBMEM 同时开启地址准备、MTE 提交和最终 drain 的细分采样。
仅显式传入 `--kernel-profile` 时启用采集。
关闭 profiling 时没有核内时间戳，因此不输出 kernel 指标；Graph Event 计时仍启用。

时间单位为微秒，带宽使用有效载荷一次除以对应耗时；`p50_GBps` 对应 P50 延迟。
上述示例显式启用了 profiling；测无 profiling 的端到端性能时，去掉 `--kernel-profile`。
开启后算子内打点的开销计入测得时间，profile 回读和结果打印在计时外。
阶段可嵌套或重叠，各段 P50 不能相加得到总 P50。

## 导出 CSV

每次单卡测试成功后，脚本自动将 `summary.csv` 写到当前目录下的
`reports/batch-get/<时间戳>-<backend>-d<设备号>/`。开启 `--kernel-profile` 时，
同一目录还会生成 `profile.csv`。要指定目录，只需传 `--output-dir`：

```bash
DEVICE_ID=2 bash tests/run_remote_cpu_batch_get_perf.sh build-npu \
  --peer 127.0.0.1:21111 --backend ubmem --aiv-cores 16 --execution both \
  --request-length 32768 --table-capacity 4096 --miss-count 4096 \
  --warmup 10 --iterations 100 --kernel-profile \
  --output-dir reports/get-ubmem-4096
```

总结表每个执行模式一行，列名固定，可以合并不同后端的结果。`e2e_p50_GBps`
是端到端有效带宽，`kernel_p50_GBps` 是算子有效带宽。未开启 profiling 时，
`kernel_*` 指标留空；`aiv_cores`、`aicpu_workers` 和 Jetty 等参数只填适用项。

`profile.csv` 每个指标一行，
用 `kind`、`name`、`metric` 区分阶段、AIV core 或 AICPU worker 的统计值。
时间、周期和带宽单位在指标名中标明。扫描多个后端或规模时，省略 `--output-dir`
即可为每次运行分别保存结果。手动指定目录时请按用例分开：同名文件会被覆盖，
关闭 profiling 不会删除目录中已有的明细文件。

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
