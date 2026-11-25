# CPU SGL Write：连续输出的 BatchGet

`GetExecutionConfig("urma", "aicpu", "cpu_sgl_write")` 复用 CPU Push Scatter
的请求构建、四路 Jetty、预构建 WRITE 链和 done 协议。CPU 将离散 DDR 源直接
SGL WRITE 到调用方注册的 HBM output，AICPU 不执行 scatter，也不分配 payload staging。
原 `scatter` 模式保持原有输出映射能力。

## 接口和限制

```python
get = kvaf.GetExecutionConfig("urma", "aicpu", "cpu_sgl_write")
# 将 get 放入 default_get/get_resources，或传给 batch_get(execution=get)。
resources = kvaf.ExecutionResources(cores=4, sge_per_wr=4)
# get_resources=[(get, resources)]；sge_per_wr=8 可测 8 SGE。
output_indices = torch.arange(request_length, dtype=torch.int32, device="npu")
```

两种 AccessTable 重载、普通 stream 调用、同步/异步完成以及 Get 图租约共用现有接口。
有效前缀中的 `output_indices[i]` 必须等于 `i`，每项写入
`output + i * token_bytes`；源索引和偏移仍可随机。有效输出前缀必须容纳
`miss_count * token_bytes` 字节，后缀保持不变。`miss_count=0` 不通知 CPU、不修改 output。
请求容量上限仍为 65536，token 大小固定为 Runtime 配置值。

AICPU 在发布 notify 前由各 worker 检查本分片的输出映射。任一分片不合法时不会
通知 CPU，避免部分写入；错误随现有同步、drain 或 status 检查报告。
设备执行错误与原 Scatter 一样使该执行资源失效，须按原 Get 恢复/关闭流程重建。

`ExecutionResources.sge_per_wr` 只用于 CPU SGL Write Get，配置单个数据 WRITE WR 的
源 SGE 上限。默认 `0` 自动选 `min(16, max_jfs_sge)`；显式值允许 1..16，超过设备
上限时初始化报错，不静默缩小。此配置在执行资源准备时固定，切换 4/8 使用新的 Runtime。
目标 SGE 始终为 1。分片末尾或 64 KiB 消息上限可使实际源 SGE 数少于配置值；
ready、done 和请求描述传输仍使用单 SGE。

## 内存和完成

Host 用 direct HCCP 注册 output，控制 workspace 额外携带一份导出的 segment 描述。
CPU 每轮读取并导入该 output，复用 WRITE 计划时同时更新目标地址和 segment。
同一 WR 仍为多个源 SGE、一个连续目标 SGE，源上限为
`sge_per_wr` 的配置值（0 时自动选择），单 WR 总字节最多为 `min(65536, max_msg_size)`。

本版保留原 chunk/Jetty 划分和 ready WRITE，AICPU 不轮询 ready。
CPU 收齐全部数据链尾 CQE，释放本轮导入的 output 后，再发布 done；AICPU 等待
done 并执行可见性屏障后结束。异步调用仍须等待完成才可读取、释放或复用输出。
发生传输故障时保留注册；Shutdown 先 STOP CPU DMA 队列，再注销 HBM。

普通调用在同步完成或 drain 后注销 output。`begin_get_output`/`end_get_output`
支持在性能循环中复用注册。图准备阶段注册固定 output，图退役后释放；多个图节点
共享同一 output 的注册按引用数保留。捕获过程中不新增注册，回放无需 Host 代理。

由于本版增加了 output 描述 READ 和每轮 CPU segment 导入/释放，端到端耗时包含
这些成本，与 Scatter 的差值不应直接视为 scatter 复制耗时。
设备 profile 的 `scatter_us`、`scatter_copies` 和 ready 查询数为零，最终等待计入 `ack_us`。
CPU profile 的 `address_read_us` 包含本轮 output 描述读取/导入，`done_us` 包含释放。

## 构建和真机验证

CPU 服务、NPU Host/Python 库须使用匹配版本重建并重启。首次引入直接输出时也须更新设备包。
CPU Push setup 增加 `output_mode` 和 `sge_per_wr`，请求结构从 424 B 变为 432 B
（先前无 SGE 配置的直接输出版本为 428 B）；旧 CPU 服务与新 Host 不兼容，不会自动回退。
Scatter 的 V3 设备入口和 launch 参数布局保留；新设备入口为
`KvafAicpuCpuSglWriteExecutionV1`，启动时检查，缺失则报错。

```bash
# CPU 端
./build.sh cpu

# NPU 端：先加载部署环境实际使用的 CANN set_env.sh
./build.sh npu
./build.sh npu --check-kernel

KVAF_CPU_SERVICE_ADDR=127.0.0.1:21111 DEVICE_ID=2 \
bash tests/run_remote_cpu_batch_get_perf.sh build-npu \
  --backend cpu_sgl_write --aicpu-workers 4 --execution both \
  --request-length 4096 --table-capacity 4096 --miss-count 4096 \
  --token-bytes 656 --tokens-per-block 1 \
  --source-access-pattern random --output-access-pattern contiguous \
  --no-sort-input-indices --warmup 10 --iterations 100 --kernel-profile
```

源地址排序会改变原 output 映射，当前性能入口拒绝此模式的 `--sort-input-indices`
和随机输出。默认复用 output 注册，也可通过 `KVAF_PERF_REUSE_GET_OUTPUT=0` 测瞬时注册。
用相同参数运行 `--backend cpu_push_scatter` 作为对照。
另以 `--miss-count 7`、`--miss-count 0` 验证部分/空前缀，并用
`--multi-graph-check` 验证多节点、多图和普通调用的串行复用。

无 SDK 的内核模型回归及其边界见 [诊断说明](../tests/diagnostics/README.md#cpu-sgl-write-kernel-model-host-only)。

## 对比 4 SGE 和 8 SGE

```bash
for sge in 4 8; do
  KVAF_CPU_SERVICE_ADDR=127.0.0.1:21111 DEVICE_ID=2 \
  bash tests/run_remote_cpu_batch_get_perf.sh build-npu \
    --backend cpu_sgl_write --sge-per-wr "$sge" --aicpu-workers 4 \
    --execution eager --request-length 4096 --table-capacity 4096 --miss-count 4096 \
    --token-bytes 656 --tokens-per-block 1 --seed 20260903 \
    --source-access-pattern random --output-access-pattern contiguous \
    --no-sort-input-indices --warmup 50 --iterations 1000 \
    --output-dir "reports/cpu-sgl-${sge}sge"
done
```

同一 CPU 服务即可接受两次测试的配置。也可用环境变量
`KVAF_BATCH_GET_PERF_SGE_PER_WR` 设置默认值；显式命令行覆盖环境变量。
单卡 `summary.csv`、配置日志及多卡子进程命令都记录/传递 `sge_per_wr`；0 表示请求自动值，
实际值看 CPU 的 `CPU_PUSH_CAP max_sge=... device_max_sge=... requested_sge=...`。
加 `--kernel-profile` 并用较少迭代可检查 CPU 的 `data_wrs`。
4096×656 B、4 worker、当前 1024 行 chunk/四路 Jetty 划分下，
4 SGE 应为 1024 个数据 WR，8 SGE 应为 512 个；均另有 16 个 ready WR。

## 测 CPU SGL 写入 HBM 的有效载荷带宽

在单卡 benchmark 加 `--sgl-payload-only`。这个开关对应 Get 资源的
`sgl_payload_only=True`。CPU 先读描述符、导入本轮 output、填完所有源/目标 SGE，
再用原有 Jetty 分片和 post 大小提交数据链。诊断链省去 chunk 的 ready WRITE，
每条非空 Jetty 的最后一个数据 WR 使用强序并产生一个 CQE。

CPU 单调时钟从第一次调用 `urma_post_jetty_send_wr` 前计到最后一条数据链尾 CQE
验证成功。窗口包括 CPU 提交、DMA 及轮询观察完成的耗时。报告和 done 在窗口结束后传输。
只为非空、单 wave 请求报告带宽。请求需要多个 wave 时，benchmark 会提示缩小
`--miss-count`。每轮通过输出校验才纳入统计，预热轮次排除。

CPU 端在仓库根目录重建，并用新二进制启动服务：

```bash
./build.sh cpu
./build-cpu/runtime/kvaf_cpu_service --listen 0.0.0.0 --port 21111
```

NPU 端加载部署所用 CANN 环境后，在同一版本的仓库根目录执行。跨主机时将
`127.0.0.1` 换成可达的 CPU 服务地址，按实际卡号设置 `DEVICE_ID`。

```bash
./build.sh npu
./build.sh npu --check-kernel
export KVAF_CPU_SERVICE_ADDR=127.0.0.1:21111 DEVICE_ID=2
for sge in 4 8; do
  bash tests/run_remote_cpu_batch_get_perf.sh build-npu \
    --backend cpu_sgl_write --sgl-payload-only --sge-per-wr "$sge" --aicpu-workers 4 \
    --execution eager --request-length 4096 --table-capacity 4096 --miss-count 4096 \
    --token-bytes 656 --tokens-per-block 1 --seed 20260903 \
    --source-access-pattern random --output-access-pattern contiguous --no-sort-input-indices \
    --warmup 50 --iterations 1000 --output-dir "reports/sgl-payload-${sge}sge"
done
```

看终端的 `scope=sgl_payload` 或各目录的 `summary.csv`。
`sgl_payload_mean_GBps` 是有效字节总数除以 CPU 窗口总纳秒数（十进制 GB/s），
`sgl_payload_p50_us` 和 `sgl_payload_p95_us` 给出单轮耗时分位数。
上述配置每轮 2,686,976 B，4/8 SGE 应分别对应 1024/512 个数据 WR，
两组均应为 0 ready WR、1 wave、4 个数据尾 CQE。CPU 服务端的
`CPU_SGL_WRITE_SAMPLE` 按 sequence 给出原始纳秒及计数，包含启用 profile 的预热轮次。
诊断结果由 CPU 服务、NPU Host/Python 和 AICPU 设备包共同传回，三者均须更新。

### CPU 分段耗时

启用 `--sgl-payload-only` 和 kernel profiling 时，CPU 服务端在原有
`CPU_SGL_WRITE_SAMPLE` 后额外输出同一 sequence 的 `CPU_SGL_WRITE_STAGES`。
统一单卡 benchmark 已自动开启 profiling；`--kernel-profile` 还会打印 NPU worker 明细。
新增分段仅保存在 CPU 侧，不修改协议或设备 profile 布局；已有 payload 版本只需
重建 CPU 服务并重启，测试命令保持不变。

| 字段（单位 µs） | 计时范围 |
| --- | --- |
| `notify_read_us` | 读到本轮新 sequence 的那一次 notify READ，包含等待 CQE |
| `header_read_us` | 请求头 READ，包含等待 CQE |
| `address_list_read_us` | 源地址清单 READ，包含等待 CQE |
| `address_validate_us` | 检查源地址范围 |
| `output_desc_read_us` | output 描述 READ，包含等待 CQE |
| `output_import_us` | 本轮 `urma_import_seg` |
| `plan_us` | 构建或复用 WRITE 计划，扣除目标更新耗时 |
| `target_update_us` | 更新目标地址及 segment |
| `source_sge_update_us` | 更新所有源 SGE，并统计数据 WR 数 |
| `data_begin_us` | 设置本 wave 的完成跟踪及尾 WR context |
| `payload_post_us` | 第一次数据 post 前到全部数据 post 返回 |
| `payload_cq_wait_us` | 全部数据 post 返回到所有数据尾 CQE 校验完成 |
| `output_unimport_us` | 本轮 `urma_unimport_seg` |
| `done_post_us` | 提交统计报告和 done 控制链 |
| `done_cq_wait_us` | 等待并校验 done 控制链尾 CQE |
| `cpu_total_us` | 本轮 notify READ 开始到 done 控制链 CQE 校验完成 |

`plan_reused=1` 表示复用数据链。单 wave 时，`payload_post_us + payload_cq_wait_us`
等于 `CPU_SGL_WRITE_SAMPLE.elapsed_ns / 1000`（允许打印舍入误差）。两段是 CPU 时间窗口，
传输可在提交期间推进；`payload_cq_wait_us` 只表示提交完之后剩余的完成等待。
多 wave 不生成有效 payload 耗时，这两个字段保持零，其他 CPU 分段按本轮累计。

`notify_read_us` 不包含此前的空闲轮询，也无法测出 NPU 发布 notify 到 CPU 发现它的
完整延迟。`cpu_total_us` 包含各分段之间的控制和打点开销，不能再与各分段相加。
其起点可能早于 kernel 进入，done CQE 的 CPU 观察也可能晚于 kernel 退出，因此不能
直接从 kernel 时间减去 `cpu_total_us`。AICPU 请求构建、worker 同步和 done 可见性等待
仍需结合 NPU `request_us`、`ack_us` 和设备起止时间分析；`ack_us` 包含 payload 等待。
打印在本轮控制链完成之后，日志包含预热；不要将 CPU/NPU 两种时钟的绝对时间相减。

```bash
rg 'CPU_SGL_WRITE_(SAMPLE|STAGES)' cpu-service.log
```

## AIV CPU SGL Write

AIV 版本使用 `GetExecutionConfig("urma", "aiv", "cpu_sgl_write")`：

```python
get = kvaf.GetExecutionConfig("urma", "aiv", "cpu_sgl_write")
resources = kvaf.ExecutionResources(cores=4, sge_per_wr=8)
# 将 get/resources 放入 default_get/get_resources。
```

源地址构建参考 `origin/aiv-scatter` 的 `BuildSourcesSimt`：SIMT 绕过私有
DCache 读取输入，经 UB → GM 写入地址描述符。所有 AIV 核验证输出恒等映射并完成
描述符写入后，核 0 发布 notify。CPU 复用上文的四路 Jetty、SGL WRITE、output
segment 导入和 done 协议，直接写调用方连续 output。AIV 不分配 payload staging、
不轮询 ready，也不执行 payload scatter。`sge_per_wr`、`sgl_payload_only`、有效前缀、
输出注册复用和故障时先 STOP 再注销 HBM 的规则与 AICPU SGL 相同。

`cores` 支持 1..8，CPU 全局固定四路 Jetty；此算法不接受 `jettys` 或
`max_inflight_total` 资源配置。每个核独占 epoch/builder cache line，在同一资源的
多个图节点、图回放和普通调用之间推进 sequence，不依赖 Host 回放代理或 AIV
跨核标量原子操作。任一核发现非法输入时不通知 CPU；后续排队回放也保留失败。
控制轮询使用有界 poll budget，不将 AIV cycle 当作 CPU 单调时间。

同步、异步、两种 AccessTable 重载、`begin_get_output` 和 Get 图租约沿用现有接口。
`batch_get(profile=True)` 的纳秒明细接口不适用于 AIV SGL；使用
`start_batch_get_kernel_profile` / `batch_get_kernel_profile`，时间戳和 request/notify/ack
分段为 AIV cycles。CPU payload sample 仍为 CPU 纳秒，两种时钟不能直接相减。
性能入口的 `--kernel-profile-clock-mhz` 负责换算 AIV cycles，请填写目标设备的实际频率。

AIV 设备入口 `KvafCpuSglWriteAivKernelV1` 编入
`bin/kvaf_execute_batch_get_kernels.o`。重建 NPU Host/Python 库和该 kernel object；
CPU 服务使用支持本文现有 SGL 协议的版本即可。`./build.sh npu --without-aicpu`
也包含此实现，无需安装 AICPU 设备包。

```bash
./build.sh npu --without-aicpu
KVAF_CPU_SERVICE_ADDR=127.0.0.1:21111 DEVICE_ID=2 \
bash tests/run_remote_cpu_batch_get_perf.sh build-npu \
  --backend aiv_cpu_sgl_write --aiv-cores 4 --sge-per-wr 8 --execution both \
  --request-length 4096 --table-capacity 4096 --miss-count 4096 \
  --token-bytes 656 --tokens-per-block 1 \
  --source-access-pattern random --output-access-pattern contiguous --no-sort-input-indices \
  --warmup 10 --iterations 100 --kernel-profile --multi-graph-check
```

以实际频率设置 `--kernel-profile-clock-mhz`。另测 `--miss-count 7` 和
`--miss-count 0`（空请求测试去掉 `--multi-graph-check`）；测 CPU 有效载荷窗口时增加 `--sgl-payload-only`，可对比
`--backend cpu_sgl_write --aicpu-workers 4` 的 CPU 侧统计。

不依赖 SDK 的检查入口为 `bash tests/diagnostics/run_cpu_sgl_host_checks.sh`。
包含真实 AIV `.asc` 控制流的多线程模型、1/4/8 核、超过单个 UB tile 的请求、
多轮 sequence、输入变化、部分/空前缀、非法映射和源输入、等待 done、远端失败与超时，
并检查 MTE 只复制地址描述符。ASan/UBSan 和 Host 声明检查不替代 CANN 编译、
真实 DMA 可见性、注册退役和图回放的目标设备验证。
