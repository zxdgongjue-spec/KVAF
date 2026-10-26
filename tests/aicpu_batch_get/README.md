# AICPU SGL → HBM 传输测试

独立 C++ 测试，复用 `CpuUrmaTransport` 和 NPU RA/TargetChannel，不接入 Runtime BatchPut。
数据路径固定为离散 CPU 源地址 → 连续 HBM staging → AICPU scatter → output slot。

默认使用 4 个 worker；NPU 的 `--aicpu-blocks N` 可设置 1～32，CPU 自动同步分片计划。
固定使用 4 个 Jetty、1024 行 chunk、doorbell 32、CQ batch 8/deadline 32，
本地 SGE 请求上限 16 并按设备能力裁剪；小数据量自动减少有效 worker。
每行保留一个本地 SGE，按目标连续性聚合 WR，不做 CPU payload 打包。
scatter 使用 securec `memcpy_sp`，只在末轮通过 URMA 回读校验完整输出和 guard。

## 构建与运行

V25 新增 `--scatter-start ready|done` 诊断，共享配置布局有变化。
CPU、NPU executable 和 device kernel 一起重建，
重新安装 kernel 后重启 NPU 测试进程。

```bash
bash build.sh aicpu-cpu
bash build.sh aicpu-npu
```

先启动 NPU 端；首次使用或 kernel 更新后加 `--install-kernel`：

```bash
bash tests/run_aicpu_get_npu.sh --install-kernel --device-id 7 --aicpu-blocks 4
```

另一个终端启动 CPU 端：

```bash
bash tests/run_aicpu_get_cpu.sh \
  --peer 127.0.0.1:29991 --device-id 7 --npu-candidate 1 \
  --rows 4096 --row-bytes 656 --warmup 50 --iterations 1000 \
  --output results/aicpu.csv
```

`--npu-candidate` 只在需要固定远端 EID 时填写；默认自动选择。
`--ub-device` / `--eid-index` 可筛选本地通路。`--device-id` 始终指 NPU 逻辑设备号。
每次 CPU 测试前重新启动 NPU 脚本；后续启动无需重复安装。
小规模验证使用 CPU 脚本的 `--smoke`：16 行 × 64 B、预热 1 轮、测量 3 轮。

CPU 保留连接参数、`--rows`、`--row-bytes`、`--warmup`、`--iterations`、`--timeout-ms`
以及诊断选项 `--scatter-start ready|done`（默认 `ready`）。
脚本的 `--build-dir`、`--output` 用于选择构建产物和保存结果。完整帮助见两个脚本的 `--help`。
CANN/UMDK 路径通过构建脚本环境或 `KVAF_UMDK_ROOT`、`KVAF_URMA_LIBRARY` 指定。
交叉工具链与 securec 布局不同时，指定 `KVAF_AICPU_CXX`、
`KVAF_AICPU_SECUREC_INCLUDE_DIR` 和 `KVAF_AICPU_SECUREC_LIBRARY`；securec 必须为 AArch64 库。

## 执行流程

1. 发现 CPU UB/EID 与 NPU candidate，建立 Jetty 通道，注册源内存及目标 HBM。
2. 按 output slot 顺序重排源 SGL，预构建数据/ready WRITE 链及全局 done 计划。
3. NPU 一次启动多个 AICPU block，在 kernel 内执行全部轮次。
4. 每轮 AICPU 直接发布 request；CPU 轮询请求并更新各源行的 8 B 序号。
5. CPU 提交预构建链。每个 Jetty 在对应数据后发布强序 ready WRITE，仅队列尾产生 CQE。
6. 各 AICPU worker 收齐全部 Jetty 的进度后，向前复制已就绪范围；连续 chunk 可合并消费。
   `done` 诊断模式仍记录首次 ready，但首次复制前等待 CPU global done。
7. CPU 收齐数据尾 CQE 后发送 done。全部 scatter 完成且 done 到达，才允许下一轮复用 staging。
8. 末轮发布 output-ready，CPU URMA READ 回读并校验后 ACK；CPU transport 停止后再释放 HBM。

默认 4096 × 656 B 的有效 payload 为 2,686,976 B；源/output stride 为 768 B。
max_sge=13 时，每轮为 320 个数据 WR、16 个 ready WR、1 个 done WR，共 5 个尾 CQE。
worker 边界按完整 staging 缓存行对齐。kernel 不再执行显式 invalidate/clean，
覆盖数据、控制词、结果和计时缓冲区；保留 DSB、原子同步、超时和序号检查。
output 跨轮复用，不再人为准备冷缓存。

## 指标

| CSV 字段 | 范围 |
|---|---|
| `mean_ns` / `e2e_GBps` | request 开始到全部 scatter 和 done 完成，排除末轮校验 |
| `first_ready_ns` / `pipeline_ns` | 请求到最早 ready / 最早 ready 到本轮结束 |
| `cpu_data_span_ns` / `cpu_data_GBps` | 数据 Submit 开始到观察到全部数据链尾 CQE，包含 ready 标记 |
| `cpu_data_submit_ns` | 数据 Submit 调用及其包装检查的耗时 |
| `cpu_data_wait_ns` | Submit 返回到观察到整条数据链完成，包含设备等待及 CQ 轮询 |
| `cpu_producer_ns` | 数据链及最终 done 的完整提交、等待时间 |
| `cpu_round_ns` / `cpu_cycle_GBps` | CPU 完整轮次，包含请求等待、源更新及末轮校验 |
| `npu_prepare_ns` | 保留 CSV 字段以便前后对比；已删除准备流程，恒为 0 |
| `npu_verify_ns` | 末轮校验握手的每轮均摊值 |

CSV 末尾追加建链与并行配置，自动选路时也记录成功建链的实际值：

| CSV 字段 | 含义 |
|---|---|
| `ub_device` / `eid_index` / `eid` | CPU UB 设备名、本地 EID 索引及完整 EID（32 位十六进制） |
| `npu_device_id` / `npu_candidate` / `npu_eid` | NPU 逻辑设备号、选中的 candidate 索引及通路 EID |
| `aicpu_blocks` / `active_threads` | NPU 配置的 block 数 / 实际有效 worker 数 |
| `jetty_count` | CPU 传输使用的 Jetty 总数，当前固定为 4；不按 AICPU block 绑定 |
| `scatter_start` / `logical_chunks` | 复制释放条件 / 全部 worker 的 chunk 总数 |

单位为 ns，有效 payload 字节数除以对应平均 ns 即十进制 GB/s。
`cpu_data_GBps` 是数据交付带宽；通信与 scatter 重叠，相关耗时不能直接相加。
`cpu_data_span_ns = cpu_data_submit_ns + cpu_data_wait_ns`（格式化舍入可能有微小差异）。
`cpu_data_wait_ns` 不是 CPU 纯轮询开销。
仅成功完成并通过末轮校验才输出 CSV。建议独立重复 5 次，比较中位数及尾延迟。

## 保持 WRITE 链相同的 scatter A/B

chunk 上限固定为 1024 行，先划分 worker 的行范围，再对每个范围分 chunk。
对于 4096 B 行：

| rows | AICPU blocks | 每个 worker 的行数 | 每个 worker 的 chunk 数 | 总 chunk 数 |
|---|---|---|---|---|
| 1024 | 1 | 1024 | 1 | 1 |
| 2048 | 1 | 2048 | 2 | 2 |
| 4096 | 1 | 4096 | 4 | 4 |
| 4096 | 4 | 1024 | 1 | 4 |

两组都使用 V25，保持 rows、行宽、block 数和通路一致，只改变 `--scatter-start`。
`ready` 立即复制已就绪范围；`done` 等 CPU 观察到全部数据尾 CQE、记录通信结束并发送
global done 后才复制。两种模式使用相同的源/目标布局、chunk、数据 WR、标记和 CQE。
`first_ready_ns` 都记录首次观察到分片 ready；`done` 的 `pipeline_ns` 还包含等待 global done。

重建两端后，NPU 终端先运行（首次加 `--install-kernel`）：

```bash
bash tests/run_aicpu_get_npu.sh --install-kernel --device-id 7 --aicpu-blocks 1
```

CPU 终端运行基线：

```bash
bash tests/run_aicpu_get_cpu.sh \
  --peer 127.0.0.1:29991 --device-id 7 --npu-candidate 1 \
  --rows 4096 --row-bytes 4096 --warmup 50 --iterations 1000 \
  --scatter-start ready --output results/aicpu-ready.csv
```

NPU 终端重新启动相同命令（无需重复安装），CPU 再运行：

```bash
bash tests/run_aicpu_get_cpu.sh \
  --peer 127.0.0.1:29991 --device-id 7 --npu-candidate 1 \
  --rows 4096 --row-bytes 4096 --warmup 50 --iterations 1000 \
  --scatter-start done --output results/aicpu-done.csv
```

先比较 `mean_wrs`、`mean_marker_wrs`、`mean_tail_cqes`、`logical_chunks` 和通路字段是否一致。
上述配置应分别为 320、16、5、4；1024/2048 行应为 80/160、4/8、5、1/2。
再比较 `cpu_data_span_ns`、`cpu_data_submit_ns`、`cpu_data_wait_ns` 和 `cpu_data_GBps`：
若 `done` 模式明显恢复带宽，说明批内重叠影响显著；若仍低且 Submit 很短，优先查源访存、
映射及设备执行/完成延迟；若 Submit 本身明显增长，先查提交后端。端到端耗时可能因移除重叠而增大。
可继续对 1024、2048、4096 行，以及四核做同样对比，每次重启 NPU 并确认校验通过。

## 无缓存维护版本的验证

先用 `--smoke` 确认握手和末轮回读成功，再按上述 4096 × 656 B 命令运行完整测试。
NPU LAUNCH 日志应包含 `revision=25`、所选 `scatter_start` 和 `kernel_cache_maintenance=none`，结束时显示
`status=1 completed=1050 failed_sequence=0`（预热 50、测量 1000），CPU 成功输出 CSV。
比较 `cpu_data_GBps`、`e2e_GBps`、`cpu_cycle_GBps` 和 p99，并确认 `npu_prepare_ns=0`。
当前版本与 V23 的区别包含移除冷 output 准备，带宽变化不能全部归因于通信加速。
末轮回读或序号检查失败、任一端超时，都表示此次删除尚未通过验证。

协议单测覆盖范围划分、ready 进度、过期序号、超时、CQ credit/乱序完成、复制和 guard、
校验握手及计时统计，并检查 done 前不复制、首次 ready 计时及两种模式的 chunk 一致性；
不依赖 CANN/URMA。传输或 stream 失败时保留 stderr 和设备日志，
并确认 CPU 已停止后再结束保留 HBM 的 NPU 进程。
