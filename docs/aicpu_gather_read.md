# AICPU GatherRead

CPU 按源地址清单 gather 到 DDR staging，AICPU 再 READ 到 `output_indices` 指定的输出行，
支持离散输出。算法选择：`GetExecutionConfig("urma", "aicpu", "gather")`。

## 构建与启动

CPU 端：

```bash
./build.sh cpu
./build-cpu/runtime/kvaf_cpu_service --listen 0.0.0.0 --port 21111
```

NPU 端先加载 CANN `set_env.sh`，再构建并安装设备包：

```bash
./build.sh npu
./build.sh npu --check-kernel
export PYTHONPATH="$PWD/build-npu/python${PYTHONPATH:+:$PYTHONPATH}"
```

两端使用配套代码，更新后重启进程。仅构建 AICPU Gather/Put 可用 `./build.sh aicpu-urma`。

## 默认配置

| 配置 | 默认值 | 覆盖方式 |
| --- | --- | --- |
| CPU Gather 线程总预算 | **12** | 服务参数 `--gather-threads N` |
| AICPU Get worker 数 | **6** | `ExecutionResources(cores=N)`；压测参数 `--aicpu-workers N` |
| 每 worker 数据 Jetty 数 | 1，另有 1 条控制链 | `ExecutionResources(jettys=N)`；压测参数 `--jettys-per-worker N` |
| 每次发送的地址块数 | **3** | `ExecutionResources(gather_address_groups_per_post=N)` 或 `KVAF_GATHER_ADDRESS_GROUPS_PER_POST=N` |

地址块数接受任意正整数，优先级：显式资源配置 > 环境变量 > 默认值 3。
初始化日志 `GATHER_READ_CONFIG` 显示生效值；不同 Runtime 可独立配置。

CPU 线程预算在 worker 间均分，须不小于 worker 数；默认等待模式为 `active`。
需要 NUMA 绑定时，在 CPU 服务命令前加 `numactl --cpunodebind=N --membind=N`。
NPU EID 自动探测；CPU 多 FE 与 AIV 共用 `peer_ub_device="A,B"` 或压测参数 `--peer-ub-device A,B`。
建链时按 `worker_id % FE数` 固定分配 CPU 设备，每个 worker 的控制链和数据链绑定同一 FE。
例如两个 FE 时，worker 0/2/4 固定绑定 A，1/3/5 固定绑定 B，运行中不切换 FE。
多 FE 用于跨机、单个 NPU EID，设备名须来自服务公布的列表；
自动选择共用主仓的 `KVAF_DEFAULT_PEER_FE_COUNT`，默认 2 个 FE。同机使用配对拓扑选路。

## 地址流水

前两批地址各自发送，后续按配置合并发送；配置超过剩余块数时，一次发完剩余地址。
例如单个 worker 读 1,366 项、每批 192 项时：

| 操作 | 分块 |
| --- | --- |
| 地址发送，默认值 3 | `192 + 192 + 576 + 406` |
| CPU gather / 数据 READ | 七批 192 项，末批 22 项 |

AICPU 发出异步 ready 查询，在查询期间构建 WQE 和后续地址；数据就绪后提交 READ。
地址和通知走控制 Jetty，数据走数据 Jetty；数据链只在每波结束时收取各链尾批 CQE。
CPU 发布最终 ready 前已写入 ACK，AICPU 不再单独查询 ACK。
WQE 优化自动生效，目标地址仍逐项读取本轮 `output_indices`。

大请求沿用主仓的内部分窗处理，`miss_count` 为 int32；可用 `gather_ddr_bytes` 设置工作区预算。
READ 每批最多 32 条 WR，每条最多 6 个 SGE，每个 SGE 对应一个完整 token。
不再限制 `6 × token_bytes <= 65536`，单条 READ 可以超过 64 KiB。

## 压测与计时

沿用统一压测入口，省略 worker 参数即使用默认 6 个。将 CPU 地址和设备号替换为实际值：

```bash
KVAF_CPU_SERVICE_ADDR=127.0.0.1:21111 DEVICE_ID=2 \
bash tests/run_remote_cpu_batch_get_perf.sh build-npu \
  --backend gather_read --execution both \
  --request-length 32768 --table-capacity 4096 --miss-count 4096 \
  --token-bytes 656 --tokens-per-block 128 \
  --seed 20260903 --no-sort-input-indices \
  --source-access-pattern random --output-access-pattern contiguous \
  --warmup 10 --iterations 100 --kernel-profile
```

`scope=kernel` 是最早 worker 进入到最晚 worker 退出，包含 CPU 和网络等待。
`scope=end_to_end` 在 eager 下测 Host 调用到完成，graph 下测同一 stream 的 NPU Event 间隔。
Gather/Scatter 图模式默认使用 ACL 原生单 Get 图计时；输入更新、校验和 profile 回读不计入测量。

分段看 `BATCH_GET_BREAKDOWN`：每轮取最后完成的 worker，再对配对阶段取平均，可以相加：

```text
kernel = start_offset + request + publish + read + ack + outer_other
read   = read_setup + wr_build + ready_wait + post + cq_wait + read_other
```

`request` 含查地址及缓存维护，`publish` 含地址 WRITE 提交及实际完成等待。
`ready_wait` 是 ready 查询的提交和收取时间，不等于 CPU 的纯拷贝时间；
期间构建 WQE、查地址和发布地址各记入自己的阶段，不重复计时。
`cq_wait` 是数据 READ 完成等待，当前流水的 `ack` 为零。
`BATCH_GET_PROFILE` 的阶段分别取 worker 最大值，不能直接相加；原始配对数据见 `kernel_breakdown.csv`。

完整 Tensor 接口、Graph、输出绑定和生命周期见 [实现与使用](cache_transport_execution_implementation.md)；
更多压测参数见 [统一 Get 性能测试](batch_get_performance.md)。
