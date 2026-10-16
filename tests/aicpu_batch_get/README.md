# AICPU SGL → staging → output：有序 WRITE 流水线

这是独立 C++ 传输基准，不改变 Runtime / AIV GatherRead API。
CPU 使用离散源 SGL，URMA 写入连续 HBM staging；AICPU 按范围并行复制到带间隔的 output slot。
目标分支原有的 GatherRead、HBM2HBM / HBM2SQ 工具继续独立使用。

## 保留的优化

- 初始化时预构建 SGE/WR 和 registered segment 关联；相邻且同 segment 的源段可无复制合并。
- 每个 Jetty 形成 `data → 强序 ready WRITE → data → ready` 链，先 post 所有队列的首范围，再 post 剩余链。
- AICPU 每个范围等待全部参与 Jetty 的本轮进度，然后开始 scatter；后续通信与各核复制重叠。
- 源按 output slot 排列，staging 与 output 均向前访问；复制调用 CANN 的 securec `memcpy_sp`。
- indexed CQ context、活动队列轮询、每队列仅最终尾 CQE、一次持有提交/CQ session。
- CPU caller 使用局部 `-O2`；保留 transport 初始化的原编译方式，避免已知部署 ABI 问题。

已移除旧的 CPU-ready 调度、整批 scatter 对照、WRITE_NOTIFY、direct-output、staging-only 多槽实验、
复制后端切换、热 Poll / backoff 参数、独立 probe、详细线程 profile 和历史诊断矩阵。

## 默认配置与运行

默认：4096 行 × 656 B、4 核 / 4 Jetty、chunk 1024、doorbell 32、请求紧轮询、
CQ batch 8 / deadline 32、NUMA auto、payload maintenance off、末轮 output 回读、warmup 50 / iterations 1000。
不硬编码设备编号、EID 或 NUMA node，仍可通过 CLI 选择。

V21 入口 `KvafAicpuBatchGetV21`，Configure `0x7015`，Config 96 B。
CPU、NPU executable 与 device kernel 必须同时更新，安装后重启 NPU 进程。
旧的 `--profile-scatter` 已移除，启动 NPU 时不要再传；当前耗时指标直接由 CPU CSV 输出。

```bash
bash build.sh aicpu-cpu
bash build.sh aicpu-npu
```

终端 A，每次 CPU 测试前重新启动：

```bash
# 首次更新安装；后续运行可去掉 --install-kernel
bash tests/run_aicpu_get_npu.sh --device-id 7 --install-kernel
```

终端 B，默认最优配置：

```bash
bash tests/run_aicpu_get_cpu.sh \
  --peer 127.0.0.1:29991 --device-id 7 --npu-candidate 1 \
  --output results/aicpu-ordered-write.csv
```

短的逐轮正确性验证，另启一次 NPU 后运行：

```bash
bash tests/run_aicpu_get_cpu.sh \
  --peer 127.0.0.1:29991 --device-id 7 --npu-candidate 1 \
  --verify-payload --warmup 2 --iterations 20 \
  --output results/aicpu-validation.csv
```

## 计时与带宽

每批有效负载 `P = rows * row_bytes`。单位为 B/ns 时，数值即十进制 GB/s。

| CSV 字段 | 定义 |
|---|---|
| e2e_GBps / mean_ns | 请求开始到全部 scatter 和 global done 完成；排除 output 准备与校验 |
| cpu_data_GBps / cpu_data_span_ns | 首次 data Submit 到 CPU 观察到全部链尾 CQE；含提交、传输、标记和轮询等待 |
| cpu_cycle_GBps / cpu_round_ns | CPU 完整轮次；含等待、源更新和安排的回读校验 |
| first_ready_ns / pipeline_ns | 请求到首范围 ready / 首 ready 到全部完成；两者相加为本轮计时 |
| npu_prepare_ns / npu_verify_ns | 计时外的冷 output 准备 / 外部校验握手耗时均值 |
| cpu_data_submit_ns / cpu_cq_poll_work_ns | CPU 在提交 / 轮询调用内累计消耗的墙钟时间 |

三种带宽自动输出，均为有效负载带宽；`cpu_data_GBps` 不是物理链路速率。
数据搬运与 scatter 重叠，不能将两个阶段简单串行相加。
默认配置在 max_sge=13 的平台预期 320 条数据 WR、16 条标记 WRITE、5 个尾 CQE（含 done）。
源 tag 每轮更新，`--verify-payload` 还会更新每个字节；验证模式的吞吐不用于性能比较。

## 缓存、顺序和生命周期

`--payload-cache-maintenance off` 保留已经实测的快速路径；仅代表当前已验证平台支持该访问方式，
不能据此认定缓存维护在所有设备上都无效。其他部署需要时使用 `--payload-cache-maintenance on`，
统一执行 staging invalidate 和 output clean。已删除两项独立开关及四组合实验。

控制字 invalidate/clean、完成屏障、每轮计时外的 cold-output 准备始终保留。
外部 URMA READ 在 AICPU 触碰/维护 output 之前完成校验并 ACK，避免本地校验制造可见性假象。
普通性能模式只检查末轮；逐轮 payload 验证单独运行。

有序链要求 RM 的保序配置、同一 Jetty/目标关联中的数据先于强序标记；不提供跨 Jetty 全局顺序。
每个 worker 等各队列进度的最小值；8 B 对齐控制字携带 sequence + chunk progress，旧序号不能释放新数据。
源数据、token 和 WR 描述符保持到全部 DMA 完成；staging 等 scatter 完成后才能复用。
部分 post/CQ 错误会 poison transport，在本端队列 teardown 完成前保留注册和描述符。
每队列源行数加标记预算上限 8192；更大的批次需拆批，不会静默越过 SQ 容量。

## 无硬件回归

```bash
# 本地可先做脚本语法检查，不调用编译器
bash -n tests/run_aicpu_get_cpu.sh tests/run_aicpu_get_npu.sh

# 可在服务器构建后运行协议、队列 credit、SGL 合并、复制 guard / 错误传播测试
cmake --build build-aicpu-cpu --target kvaf_aicpu_get_protocol_test kvaf_aicpu_write_chain_test \
  kvaf_aicpu_prepared_transport_test kvaf_aicpu_output_verify_test kvaf_aicpu_securec_scatter_copy_test
ctest --test-dir build-aicpu-cpu -R '^kvaf_aicpu_' --output-on-failure
```

整理前 V20 的用户实测参考：有序链约 7.47 GB/s 端到端、38.63 GB/s 数据交付。
这些数字不代表 V21 已完成硬件验证；本次收敛后需在同一服务器复测。
