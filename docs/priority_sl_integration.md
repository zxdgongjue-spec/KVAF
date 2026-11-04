# NPU peer、Priority/SL 与并发压测

本分支整合 `tests/priority_sl`，保留 CPU DDR / NPU HBM peer 和
`lookup`、`raw`、`no-pipeline` 三种 BatchGet 模式。

## 构建与接口变化

- NPU 侧重新执行 `./build.sh npu`，同步更新 host、Python binding 和两个 kernel。
  合并后的 ABI 为 V5：参数 128 字节、profile 5824 字节、每核 status 5888 字节。
  kernel loader 拒绝旧版本；CPU/NPU 服务控制协议仍为 v7。
- DSA 固定映射 `slot_id = key`，Key 范围为 `[0, region.length / slot_size)`。
  原有使用任意大整数 Key 的调用方需要改用有效 slot ID。
- BatchPut 支持整数序列或连续一维 CPU `torch.int64` 元数据，采用 pinned
  缓冲区和异步拷贝后统一同步；对外仍然同步完成。
- 默认每核两个 Jetty，BatchGet 性能测试默认 table capacity 为 16384。
  sweep 默认测试 656B、miss count 2048/4096/8192、两个 core、每核两个 Jetty。
  四个 `KVAF_BATCH_GET_SWEEP_*` 环境变量仍可覆盖整个矩阵。

## 单路优先级测试

```bash
KVAF_PEER_DEVICE=npu KVAF_PEER_SERVICE_ADDR=NPU1_HOST:28999 \
DEVICE_ID=0 KVAF_BATCH_GET_JETTY_PRIORITIES=3,13 \
bash tests/run_remote_cpu_batch_get_perf.sh "$PWD/build-npu" \
  --batch-get-mode raw --request-length 8192 --miss-count 8192 \
  --table-capacity 16384 --aiv-cores 2 --jettys-per-core 2 \
  --kernel-profile --jetty-csv results/npu-raw-jettys.csv
```

CPU 场景改为 `KVAF_PEER_DEVICE=cpu` 并使用 CPU 服务地址。
priority 范围为 0–15，按 core-major 顺序循环分配；只作用于本地 BatchGet
Jetty。有效 SL 由设备映射决定，不直接由这个变量设置。

逐 Jetty CSV 包含 peer 类型、算子模式、priority、EID 路径、请求量、提交和
观察完成时间及有效带宽。Profiling 使用轮询 CQ 的软件观察时间；并非硬件
报文时间戳。no-pipeline 同时保留 WQE fill、MTE copy、doorbell 分阶段统计。

## 两路同步并发

同一张 NPU 同时读取本机 NPU HBM 和远端 CPU DDR，可直接使用
[`run_mixed_peer_batch_get.py` 专用入口](mixed_peer_batch_get.md)，支持固定
CPU UB 设备，以及可选的单路基线与并发结果对比。

在同一台客户端主机启动两个独立进程。每路可以使用相同或不同 NPU，也可以
分别指定 CPU/NPU peer、服务地址、EID、算子模式、shape 和优先级。

```bash
python3 tests/python/run_concurrent_batch_get.py "$PWD/build-npu" \
  --flow1-device 0 --flow2-device 1 \
  --flow1-priorities 3 --flow2-priorities 13 \
  --flow1-args="--peer-device npu --peer-service-addr NPU2_HOST:28999 --batch-get-mode raw --request-length 8192 --miss-count 8192 --table-capacity 16384 --aiv-cores 2 --jettys-per-core 2" \
  --flow2-args="--peer-device npu --peer-service-addr NPU2_HOST:28999 --batch-get-mode raw --request-length 8192 --miss-count 8192 --table-capacity 16384 --aiv-cores 2 --jettys-per-core 2" \
  --output-dir results/two-flows
```

每路独立初始化、校验和预热，随后约定共同的主机单调时钟开始时刻，在首次
正式测量前放行；这不是硬件同步或每轮 barrier。输出保存在 `flow1/flow2`，
目录已存在时拒绝覆盖。`KVAF_REMOTE_CPU_PERF_TIMEOUT` 同时限制准备和测量。
并发实验应固定 CPU UB device，不使用自适应校准；NPU peer 不支持 CPU UB
device 选择参数。

## 诊断与配套实验

| 入口 | 用途 |
| --- | --- |
| `tests/diagnostics/query_npu_priority_sl.py` | 查询指定 NPU/EID 的 priority→SL 映射 |
| `tests/diagnostics/collect_read_qos_info.py` | CPU 驱动、设备和 QoS 配置快照 |
| `tests/diagnostics/collect_read_qos.py` | CPU 多 URMA 设备 port/VL 采样，同物理端口组去重 |
| `tests/diagnostics/npu_port_stat.py` | NPU UB 端口带宽、可选 CTP/flit 速率 |
| `stat.py` | `ubctl` chip/port 包率与带宽采样 |
| `tests/diagnostics/bandwidth_scopes.py` | 配对 Host/Event/kernel/传输窗口计时；接受 benchmark 模式及 peer 参数 |
| `tests/diagnostics/bandwidth_matrix.py` | Jetty 数、请求大小和局部性对照矩阵；peer 可通过环境变量设置 |
| `tests/python/test_npu_batch_put.py` | DSA 单层 BatchPut 带宽测试，支持 CPU/NPU peer 环境变量 |

CPU 的 `--adaptive-ub-device` sweep 先执行完整性能路径校准，再复用所选设备。
多 EID 的各路径允许不同 local token ID；`shuffled-contiguous` 只重排有效
miss 前缀，保证与 contiguous 的有效读取集合一致。

详细参数见 [BatchGet](../access_operator/demo/aiv_batch_get/README.md)、
[诊断工具](../tests/diagnostics/README.md) 和 [NPU peer](npu_peer.md)。

## 验证

无 NPU 的 Linux 环境可运行：

```bash
cmake -S . -B build-check -DKVAF_BUILD_LEGACY_CPP_TESTS=ON
cmake --build build-check -j4
ctest --test-dir build-check --output-on-failure
```

CPU kernel 模型覆盖 lookup/raw 的流水和无流水提交、逐 Jetty profiling、
多 EID token、尾组、队列环回和异常。服务测试使用 ACL/RA 替身；Python 测试
覆盖并发启动、超时清理、三模式 sweep、CSV、校准和诊断解析。
真实 CANN 编译、两种 peer 的端到端读写和 QoS/带宽效果仍需在目标 A5 环境验证。

2026-11-04 本地 WSL/GCC 验证：15 项 CTest 全部通过，Python 语法与 Git
差异检查通过。当前环境未安装 pybind11，未执行原生 Python binding 构建。
