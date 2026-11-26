# 同卡 AIV direct 与 AICPU scatter 并发带宽

用于验证 CPU SGL WRITE 与 AIV READ 同时运行时，完整 Get 路径的有效带宽是否下降。
数据方向均为 CPU DDR → NPU HBM：AIV 发 READ 直接写调用方输出；Scatter 由 CPU 发 SGL
WRITE 到 HBM staging，再由 AICPU 复制到输出。READ/WRITE 的发起方向不同，不代表载荷走相反方向。

## 构建与运行

CPU 端使用同次源码构建并启动服务；NPU 端加载 CANN 环境后构建所有算子，包含 CPU Push：

```bash
# CPU 主机
./build.sh cpu
./build-cpu/runtime/kvaf_cpu_service --listen 0.0.0.0 --port 21111

# NPU 主机
./build.sh npu
./build.sh npu --check-kernel
bash tests/run_remote_cpu_batch_get_aiv_scatter_perf.sh build-npu \
  --peer 127.0.0.1:21111 --device-id 2 \
  --aiv-cores 2 --jettys-per-core 2 --max-inflight-total 2048 \
  --aicpu-workers 5 \
  --request-length 8192 --miss-count 8192 \
  --table-capacity 4096 --tokens-per-block 128 --token-bytes 656 \
  --source-access-pattern random --output-access-pattern contiguous \
  --no-sort-input-indices --graph-calls 32 --warmup 3 \
  --seconds 3 --repeats 3 --output reports/aiv-scatter/run-01
```

跨主机时替换 `--peer`；输出目录必须尚不存在。脚本不启动、重启或关闭 CPU service。
默认超时 900 秒，可通过 `KVAF_REMOTE_CPU_PERF_TIMEOUT` 调整。
`--help` 无需硬件或构建产物；执行测试需要 Ascend 和 `torch_npu`，无硬件时报错退出。
两路各分配一份 DDR 源池，上例每路 328 MiB、合计 656 MiB；另需算子 workspace、输出及图节点资源。

## 并发方式

同一进程创建两个独立 Runtime 和两个独立 stream，各自保持 DDR、AccessTable、请求数组、
输出、通信队列和图租约的生命周期。普通 `batch_get` 在同一 Runtime 上会先 drain 前一个 Get，
不能直接用同一个 Runtime 连续两次 eager 调用来测试并发。

先依次完成两路建表、真实预热与 ACL 图捕获；每张图包含 `--graph-calls` 个串行 Get。
测量时每路一个主机线程，起点使用共同 barrier，循环向自己的 stream 提交图并仅等待自己的 stream。
回放使用 `AclGraph.replay_timed` 的原生 ACL Event 等待，C++ 等待期间明确释放 GIL，
使另一主机线程能够继续下发。每图两个 Event 的记录/等待开销包含在实际测量时长内；
原始 JSON 的 `device_event_ms` 记录各图 Event 时间之和，不用于替代主机窗口计算总吞吐。
一条路径完成后可继续提交，不需要等待另一条路径。所有阶段均保持两个 Runtime 及两份源池驻留。
每轮依次测试 AIV 单独、Scatter 单独、同时运行；下一轮反转阶段顺序，减轻顺序漂移影响。

两路使用相同 seed 生成相同请求轨迹，但访问各自独立源池和输出。源数据按 token ID 编码。
捕图后先校验完整输出；每个测量阶段前重置输出为哨兵值，阶段后检查有效载荷、未写区域，
并检查所有 Scatter 节点的设备结果。初始化、Put、预热、输出校验与文件写入均排除在计时外。
计时循环不启用算子内部 profiling、不回读各节点设备状态报告。

## 结果与解释

- `config.json`：源码 revision、设备、CPU service、工作负载、请求与实际核数/Jetty 预算。
- `repeat_<轮次>_<阶段>.json`：每路实际起止时间、图数、有效字节数和正确性结果。
- `summary.csv`：每轮每路的单独/并发 GB/s、保留比例和下降比例。
- `aggregate.csv`：并发总吞吐、单独带宽之和、主机窗口重合比例及起点偏差。
- `report.md`：各轮中位数和计时说明。

每路带宽 = 完成图数 × graph_calls × miss_count × token_bytes ÷ 本路实际耗时。
总吞吐 = 两路有效字节之和 ÷（两路最晚结束 − 两路最早开始）。
GB/s 使用十进制；Scatter 的 staging 写入与内部复制不重复计入有效载荷。
每次图回放均完整完成后计数，因此实际耗时包含截止点之后最后一张图的尾部。
每路并发/单独带宽的比值为保留比例；1 − 保留比例为下降比例，可为负值。

`host_window_overlap_percent` 是主机测量窗口的交集/并集，不能证明设备 kernel 真正重叠。
必要时另行采集设备时间线，确认两个 stream 上的 AIV/AICPU 在设备上同时执行；
采集 profiling 的运行应与带宽主测分开，以免其开销混入带宽结果。

若并发带宽显著下降，可说明完整路径有资源干扰；不能单凭下降断定 SGL WRITE 与 READ
在特定链路上冲突。它还包含 CPU DDR、互联/端口、HBM、Scatter 复制及设备调度的影响。
建议在同一硬件配置下做这些对照：

1. 先用 `--npu-eids <一个32位十六进制EID>` 固定同一 NPU 路径，并确认日志中的实际 CPU FE/路由。
   默认自动发现可能选不同路径；多个 EID 时 AIV 可以条带化，Scatter 使用首项。
2. 在两个独立输出目录比较随机/连续源、不同 `--token-bytes` 和 `--miss-count`。
3. 扫描 AIV 核数/Jetty/在途数及 AICPU worker 数，区分供给不足与共享资源饱和。
4. 增大 `--graph-calls`（如 32 → 128）和 `--seconds`，确认吞吐稳定、主机提交开销与图尾部影响足够小。
   增加 graph_calls 会增加图节点及报告资源，不能无限增大。

本地无硬件验证：

```bash
python3 -m unittest discover -s tests/python -p test_batch_get_aiv_scatter.py -v
```

这些检查验证请求映射、双线程下发/独立等待、失败传播及带宽统计；不代替设备并发实测。
