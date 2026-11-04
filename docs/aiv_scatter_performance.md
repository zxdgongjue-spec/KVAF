# AIV CPU-push scatter 性能测试

CPU 通过四条 Jetty 发起 URMA SGL WRITE，将数据写入 NPU HBM staging；AIV 将 staging
中的 token scatter 到输出行。默认固定采用 SIMT 多核融合清单构建和 UB 双缓冲。
唯一算法开关是流水掩盖：`--pipeline 0|1`，对应环境变量 `KVAF_SCATTER_PIPELINE`，
未设置时默认开启。初始化后不再读取开关，不允许改变在途请求的算法。

`--fused-build`、`--double-buffer`、`--simt-build` 及其同名环境变量已移除。
串行两遍构建、UB/DMA 构建对照和单缓冲算法已删除；JSON 用
`source_list_builder=parallel_simt`、`scatter_copy=ub_double_buffer` 描述固定实现。
计时开关与请求规模、队列、源分布等测试参数继续保留。

## 构建与运行

Host 和 AIV kernel 必须一起重编；混合 SIMD/SIMT kernel 的 loader 始终申请 65536 B 动态 UB，
kernel 类型必须为 AIV。接口参数仍为 176 B，生产输入沿用 trusted-input 契约。
CPU 源内存默认保留 THP 分配；NPU workspace 保留 `ACL_MEM_MALLOC_HUGE_FIRST`，
大页不足时可回退普通页。显式 HugeTLB 模式、CPU tail-only completion-order 试验、
额外启动 READ/WRITE 探测、ACK 调参和专用诊断 RPC 已删除。

```bash
cmake --build build-cpu -j --target kvaf_cpu_service
cmake --build build-npu -j
```

重新启动 CPU 服务和 NPU 进程后测试。指定 C1 端口可在 CPU 服务上使用
`--ub-device udmac1d1e2`，测试侧使用 `--peer-ub-device udmac1d1e2`。
CPU 核和内存 NUMA 位置由启动服务时的 `numactl` 策略控制，peer 候选编号不代表物理 CPU。

## 两套端到端场景

保留 changing-data 回归入口：

```bash
DEVICE_ID=7 KVAF_CPU_SERVICE_ADDR=127.0.0.1:28990 \
bash tests/run_remote_cpu_scatter_perf.sh "$PWD/build-npu" \
  --cores 4 --blocks 1024 --tokens-per-block 4 --token-bytes 656 \
  --pipeline 1 --iterations 100 --profile-rounds 100 \
  --json results/scatter.json --csv results/scatter.csv
```

一个样本是一次同步 BatchGet。默认 full-blocks 为 1024 block × 4 token/block × 656 B，
有效 payload 2686976 B。`--request-length` 是输入容量，`--miss-count` 是本次有效 token 数；
`--workload one-token-per-block` 复现每 block 一个 token 的请求语义。
changing 模式在计时外更新源数据、请求和表，计时后验证完整输出与 guard。
`--input-mode steady` 复用表；`--profile-iterations 0` 关闭可选打点。
`--source-pattern random|sequential`、`--output-order random|sequential` 和
`--output-row-step 1|2` 控制访问分布；随机大源表按块生成，避免 Python randbytes 溢出。

参考 direct-READ miss-count 场景的稳态入口：

```bash
DEVICE_ID=7 KVAF_CPU_SERVICE_ADDR=127.0.0.1:28990 \
bash tests/run_remote_cpu_scatter_miss_count_perf.sh "$PWD/build-npu" \
  --single --token-bytes 656 --miss-count 8192 --request-length 32768 \
  --tokens-per-block 128 --table-capacity 1277952 --aiv-cores 4 \
  --pipeline 1 --warmup 30 --iterations 100 \
  --json results/scatter-aligned.json --csv results/scatter-aligned.csv
```

该入口共享 `test_npu_batch_get_perf.py` 的参数和请求生成器，对齐
`tests/run_remote_cpu_batch_get_miss_count_perf.sh`。
默认从全表随机取 block、再随机取 block 内 token；每轮重新生成请求。
源表一次性填满，正式计时不含初始化、BatchPut、表准备、请求生成及预同步。
默认 table-capacity=16384×78、tokens-per-block=128；
656 B 源区为 107307073536 B（约 99.94 GiB），每次 miss-count=8192 仅传 5373952 B。
缩减 table-capacity 同时改变 CPU 源访问范围，对照实现应保持相同配置。

常用对齐参数：

| 参数 | 默认/限制 |
|---|---|
| `--request-length` | 32768，输入 Tensor 容量 |
| `--miss-count` | 有效 token 数；不超过 request-length，最大65536 |
| `--table-capacity` | 1277952 个 block |
| `--tokens-per-block` | 128 |
| `--source-access-pattern` | random、contiguous、shuffled-contiguous |
| `--output-access-pattern` | 默认 contiguous，也支持 random |
| `--sort-input-indices` | 按解析后的源地址排序；排序在 BatchGet 计时外 |
| `--aiv-cores` | 1..8；CPU payload 始终4条 Jetty |
| `--jettys-per-core` | 固定2，仅用于 Host setup/BatchPut |
| `--pipeline` | 默认1；设0比较全批 done 后 scatter |
| `--kernel-profile/--no-kernel-profile` | 对齐入口默认开启 AIV/CPU 打点 |
| `--host-profile/--no-host-profile` | 默认关闭 Host 细分打点 |
| `--completion-mode` | reference 或 validated |
| `--case-timeout` | 整个 case，包含初始化、填表、采样和拆除 |

不带 `--single` 时扫描 token-sizes、miss-counts、core-counts，在 output-dir 下输出分组 CSV/JSON
与 summary.csv；`--dry-run` 只打印计划，不访问硬件。
只支持单 NPU EID 和固定 peer-ub-device；不做 adaptive UB 设备校准。
固定4条 CPU Jetty不能等价解释为 aiv-cores×jettys-per-core 条 payload 队列。

## 流水和完成语义

pipeline=0：所有 payload 尾 CQE完成，CPU 发布 done，随后 AIV scatter。
pipeline=1：CPU 按每核1024行 chunk 组织四条 Jetty，分别发布各 Jetty shard 的 Ready；
AIV 优先消费已就绪 shard，UB 双缓冲跨 Ready 边界保持 MTE3 搬运在途。
两种模式都必须等待最终匹配序号的 done，并完成全部输出 DMA 后才成功返回。
Ready 表示某个 shard 已可读，不代表整次请求成功。

CPU payload 恢复全 WR `comp_order=1`；每条链保留强序尾 WR 和真实尾 CQE，
ready/done 的必要强序保证继续保留。不得删除最终完成等待或提前发布成功。
CPU 控制 READ 合并 header/descriptor，WR/SGL 计划复用，每次刷新真实源地址，
四队列分批交错提交与逐队列刷新/提交重叠继续保留。

## 计时和带宽口径

CSV 为32列；删除恒定的 fused_build、double_buffer 两列，耗时统一 µs，带宽为十进制 GB/s。
JSON保留原始纳秒、worker cycle、CPU post/CQ等待和配置。
有效带宽始终为 `payload_bytes / elapsed_ns`，没有额外乘8，也不将本地读+写流量翻倍。

| 字段组 | 含义 |
|---|---|
| phase、iteration、sequence、cores、pipeline | 样本标识和执行配置 |
| request_length、miss_count、token_bytes、payload_bytes | 请求容量、有效token数、token字节、实际有效载荷 |
| end_to_end_us / end_to_end_gbps | 一次 BatchGet 的完整调用/同步窗口 |
| cpu_comm_us / cpu_comm_gbps | 首个 payload WRITE 提交到最后尾 CQE 被观测 |
| cpu_write_us / cpu_write_gbps | CPU WriteBatch 整体，包含计划复用处理和地址刷新 |
| aiv_build_notify_us | 输入就绪到核0发布通知，包含多核清单构建和汇合 |
| aiv_notify_to_done_us | 通知到核0观测最终 done，包含CPU控制、数据传输与完成发布 |
| aiv_scatter_us / aiv_scatter_gbps | 最大单核scatter活跃耗时，流水模式下排除等待Ready的区间 |
| aiv_body_us / aiv_body_gbps | 最大单核主体耗时 |
| aiv_wall_us / aiv_wall_gbps | 跨核最早start到最晚finish的设备跨度 |
| aiv_input_wall_us | 最早start到最晚inputs-ready的跨度 |
| aiv_scatter_tail_us | 全局done之后仍未完成的scatter尾部跨度 |
| aiv_ready_wait_us | 最大单核等待Ready/done耗时，与通信等设备阶段存在重叠 |
| call_outside_aiv_us | end-to-end减AIV wall，包括提交、包装和设备窗口之外的同步等 |
| host_tensor_metadata_us | Tensor元数据解析/必要参数绑定 |
| host_launch_us | kernel提交 |
| host_status_copy_enqueue_us | 状态回拷提交 |
| host_stream_sync_us | Host同步调用；包含尚未完成的设备执行 |
| host_python_dispatch_us | 绑定计时外的包装、派发与返回差值 |

`cpu_comm` 还包含首个提交后后续队列的地址刷新/提交及完成观测延迟，不能直接等同物理线速。
`cpu_header_ns`、`cpu_descriptor_ns` 是控制读取，不可作为payload通信分母。
Host和设备时钟不能跨域相减；各核计时不能求和。
pipeline=1时CPU通信、Ready等待、scatter活跃区间重叠，不能将这些列直接相加。
做关闭流水的近似层级拆解可用：

```text
端到端
├─ 调用外围
└─ AIV wall
   ├─ 输入
   ├─ 构建清单并通知
   ├─ 通知→done（内部包含CPU通信）
   └─ done之后的scatter尾部
```

对齐入口 reference：异步 BatchGet后全设备同步，完成后在采样外drain/验证status。
validated：同步API，状态回拷/检查计入端到端，不额外全设备同步。
profile snapshot及最终payload验证均在采样外；打点本身的报告写回开销仍在设备流程中。
Host profile用于解释外围；其stream-sync耗时不能再与AIV时长叠加。
最终收益应同时比较无打点端到端与相同打点配置下的阶段变化。

## 校验

保留preflight有效前缀/guard验证、最后一轮随机payload验证与正常CQ/超时/失败处理。
changing-data入口继续覆盖逐轮改变源数据的完整回归。

```bash
python3 -m unittest discover -s tests/python -p 'test_scatter*py'
python3 -m unittest discover -s tests/python -p test_cpu_write_bandwidth_config.py
ctest --test-dir build-npu -R 'kvaf_sequential_' --output-on-failure
ctest --test-dir build-cpu -R kvaf_cpu_push_transport_test --output-on-failure
```

独立纯CPU WRITE带宽入口保留，参见 [CPU WRITE 带宽测试](cpu_urma_write_bandwidth.md)。
旧的优化探索和计时审计文档属于历史记录，其已移除开关不再适用于当前运行命令。
