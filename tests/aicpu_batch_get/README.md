# AICPU SGL → HBM 传输测试

独立 C++ 测试，复用 `CpuUrmaTransport` 和 NPU RA/TargetChannel，不接入 Runtime BatchPut。
数据路径固定为离散 CPU 源地址 → 连续 HBM staging → AICPU scatter → output slot。

默认使用 4 个 worker；NPU 的 `--aicpu-blocks N` 可设置 1～32，CPU 自动同步分片计划。
固定使用 4 个 Jetty、1024 行 chunk、doorbell 32、CQ batch 8/deadline 32，
本地 SGE 请求上限 16 并按设备能力裁剪；小数据量自动减少有效 worker。
每行保留一个本地 SGE，按目标连续性聚合 WR，不做 CPU payload 打包。
scatter 使用 securec `memcpy_sp`，只在末轮通过 URMA 回读校验完整输出和 guard。

## 构建与运行

V23 恢复了可配置的 AICPU 并行度。CPU、NPU executable 和 device kernel 必须一起重建，
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

CPU 保留连接参数、`--rows`、`--row-bytes`、`--warmup`、`--iterations` 和 `--timeout-ms`。
脚本的 `--build-dir`、`--output` 用于选择构建产物和保存结果。完整帮助见两个脚本的 `--help`。
CANN/UMDK 路径通过构建脚本环境或 `KVAF_UMDK_ROOT`、`KVAF_URMA_LIBRARY` 指定。
交叉工具链与 securec 布局不同时，指定 `KVAF_AICPU_CXX`、
`KVAF_AICPU_SECUREC_INCLUDE_DIR` 和 `KVAF_AICPU_SECUREC_LIBRARY`；securec 必须为 AArch64 库。

## 执行流程

1. 发现 CPU UB/EID 与 NPU candidate，建立 Jetty 通道，注册源内存及目标 HBM。
2. 按 output slot 顺序重排源 SGL，预构建数据/ready WRITE 链及全局 done 计划。
3. NPU 一次启动多个 AICPU block，在 kernel 内执行全部轮次。
4. 每轮 AICPU 准备冷 output 后发布 request；CPU 轮询请求并更新各源行的 8 B 序号。
5. CPU 提交预构建链。每个 Jetty 在对应数据后发布强序 ready WRITE，仅队列尾产生 CQE。
6. 各 AICPU worker 收齐全部 Jetty 的进度后，向前复制已就绪范围；连续 chunk 可合并消费。
7. CPU 收齐数据尾 CQE 后发送 done。全部 scatter 完成且 done 到达，才允许下一轮复用 staging。
8. 末轮发布 output-ready，CPU URMA READ 回读并校验后 ACK；CPU transport 停止后再释放 HBM。

默认 4096 × 656 B 的有效 payload 为 2,686,976 B；源/output stride 为 768 B。
max_sge=13 时，每轮为 320 个数据 WR、16 个 ready WR、1 个 done WR，共 5 个尾 CQE。
worker 边界按完整 staging 缓存行对齐。scatter 内不执行 payload invalidate/clean；
控制词缓存维护、DSB 和计时外的冷 output 准备保留。

## 指标

| CSV 字段 | 范围 |
|---|---|
| `mean_ns` / `e2e_GBps` | request 开始到全部 scatter 和 done 完成，排除准备和校验 |
| `first_ready_ns` / `pipeline_ns` | 请求到最早 ready / 最早 ready 到本轮结束 |
| `cpu_data_span_ns` / `cpu_data_GBps` | 数据 Submit 开始到观察到全部数据链尾 CQE，包含 ready 标记 |
| `cpu_producer_ns` | 数据链及最终 done 的完整提交、等待时间 |
| `cpu_round_ns` / `cpu_cycle_GBps` | CPU 完整轮次，包含请求等待、源更新及末轮校验 |
| `npu_prepare_ns` / `npu_verify_ns` | 计时外冷 output 准备 / 末轮校验握手的每轮均摊值 |

单位为 ns，有效 payload 字节数除以对应平均 ns 即十进制 GB/s。
`cpu_data_GBps` 是数据交付带宽；通信与 scatter 重叠，相关耗时不能直接相加。
仅成功完成并通过末轮校验才输出 CSV。建议独立重复 5 次，比较中位数及尾延迟。

协议单测覆盖范围划分、ready 进度、过期序号、超时、CQ credit/乱序完成、复制和 guard、
校验握手及计时统计；不依赖 CANN/URMA。传输或 stream 失败时保留 stderr 和设备日志，
并确认 CPU 已停止后再结束保留 HBM 的 NPU 进程。
