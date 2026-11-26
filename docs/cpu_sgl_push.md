# CPU SGL Push：设备推送地址清单、CPU 直接写输出

新增独立算法 `cpu_sgl_push`，支持 AIV 和 AICPU。设备通过专用控制 Jetty 将请求头、
输出注册信息和源地址清单 WRITE 到 CPU mailbox；CPU 从本地 mailbox 消费请求，
用四条数据 Jetty 将离散 DDR 源 SGL WRITE 到调用方的连续 HBM output。
两种设备实现共用 CPU service 的 mailbox 消费、SGL 计划和完成协议，不执行 payload scatter。

| 算法/API | benchmark 后端 | 地址清单传输 | 流水 |
| --- | --- | --- | --- |
| AICPU `cpu_sgl_push` | `cpu_sgl_push` | 设备 WRITE 到 CPU | 内置 Head/Suffix |
| AIV `cpu_sgl_push` | `aiv_cpu_sgl_push` | 设备 WRITE 到 CPU | 内置 Head/Suffix |
| AICPU `cpu_sgl_write` | `cpu_sgl_write` | CPU READ 设备 | 原整批或可选逐 chunk |
| AIV `cpu_sgl_write` | `aiv_cpu_sgl_write` | CPU READ 设备 | 原整批或可选逐 chunk |

现有 `cpu_sgl_write`、其 `--sgl-pipeline` 逐 chunk 路径和 `scatter` 保留。
新算法通过独立 setup 命令 20 协商 mailbox，旧 setup 和 launch ABI 不变。
`aiv-scatter` 分支的 USER_CTL WRITE 编码、延迟回收 head CQE 和 CPU credits 调度
作为本实现的参考；没有引入 payload staging、ready WRITE 或设备端 payload 拷贝。

## 接口与限制

```python
get = kvaf.GetExecutionConfig("urma", "aiv", "cpu_sgl_push")
# AICPU：kvaf.GetExecutionConfig("urma", "aicpu", "cpu_sgl_push")
resources = kvaf.ExecutionResources(cores=4, sge_per_wr=4)
# default_get=get, get_resources=[(get, resources)]
```

两种实现均支持 1..8 个设备 worker；CPU 固定四条数据 Jetty，另有一条描述符控制连接。
资源不接受 `jettys` 和 `max_inflight_total` 调整。`sge_per_wr` 与原 SGL 规则相同：
0 自动选择 `min(16, max_jfs_sge)`，显式值 1..16 必须不超过硬件上限。

Head/Suffix 始终启用，资源查询和性能脚本记录 `sgl_pipeline=True`，
日志/CSV 标识 `sgl_pipeline_mode=device_push_head_suffix`。这里不选择原逐 chunk READ 路径。
新算法不支持 `sgl_payload_only`；性能比较使用完整 kernel 和端到端计时。

有效前缀的 `output_indices[i]` 必须等于 `i`，输出至少有
`miss_count * token_bytes` 字节；后缀保持不变。源索引/偏移可以随机。
容量上限 65536，token 大小按 Runtime 配置。空请求不通知 CPU、不修改 output。
性能入口要求连续输出和 `--no-sort-input-indices`。

`cpu_sgl_push` 的 CPU service 信任设备生成的源地址清单，关闭逐项源地址范围扫描，
同时适用于 AIV 和 AICPU。请求头、sequence、数量和 output 容量/segment 检查保留；
设备端现有检查保持不变。调用方须保证源索引、偏移和地址表有效，且源注册在请求完成前存活。
原 `cpu_sgl_write`、逐 chunk READ 流水和 scatter 的 CPU 源地址校验继续保留。

同步、异步、两种 AccessTable 重载、`begin_get_output` 注册复用和 Get 图租约
沿用现有接口。每次回放由设备 epoch 推进 sequence，控制队列与 mailbox 常驻，
不需要 Host 逐轮重置或代理发送。异步输出仍须等完成后才能消费或复用。

## 流水与完成

1. 核 0 构建并校验前 `min(count, 1024)` 项，将请求头、output 描述和 head 地址
   按序 WRITE 到 CPU，最后 WRITE notify 作为 commit。
2. head 控制链最后一个 CQE 延后回收。核 0 继续构建自己的后缀，其他核并行构建
   其后缀；CPU 本地 acquire 观察 notify 后导入 output、填充 head SGE、开始数据 WRITE。
3. 设备汇总后缀构建结果并回收 head 控制 CQE。成功时 WRITE 后缀地址，最后
   WRITE `Progress(sequence, result)`；失败时只 WRITE 错误 token。
   CPU 本地 acquire 等待后缀 commit，期间可回收已完成的数据计划。
4. CPU 观察到后缀成功 commit 后填充 SGE 并提交后缀 WRITE，不再单独扫描源地址范围。
   每个计划独占 WR/SGE 存储到 CQE，
   最多四个计划在途，按 SQ credits 回收；credit 足够时不用先等 head 全部完成。
5. 全部数据链尾 CQE 成功后，CPU 释放本轮 output 导入，再强序 WRITE done。
   设备等待本轮 done、执行可见性同步后退出。

不足或等于 1024 项只发送 head；控制 WRITE 按消息上限拆分。AIV 使用 USER_CTL SQ、
WQE 模板和有界 CQ 轮询；AICPU 使用原生 UMDK 的独立 post/wait 控制接口，
让 head DMA 与后缀构建重叠。CPU 请求头、output 描述、地址清单和进度均无 RDMA READ。
CPU 仍逐轮导入/释放输出 segment，这部分成本计入整体延迟。

流水发布后发生后缀错误时，CPU 不提交错误后缀，排空已提交的合法数据后返回失败 done。
失败请求可能已写入 head，output 不可消费；按现有执行资源失效和重建流程处理。
Shutdown 先确认设备发送 kernel 已停止，再销毁描述符发送队列；CPU STOP 完成后才释放
HBM 注册。无法确认 kernel 结束时保留两端目标内存，报告 `CompletionUnknown`。

## 构建与比较

CPU service、NPU Host/Python 和设备 kernel 均须使用此版本重建，重启两端进程。
仅更新 Host 无法启用新协议；不自动回退到旧 READ 模式。

```bash
# CPU 端
./build.sh cpu
./build-cpu/runtime/kvaf_cpu_service --listen 0.0.0.0 --port 21111

# NPU 端：加载实际部署的 CANN set_env.sh 后构建、检查设备包
./build.sh npu
./build.sh npu --check-kernel
```

AICPU 新入口为 `KvafAicpuCpuSglPushExecutionV1`，与旧入口一起装入 AICPU 包，
启动时检查入口。AIV 新入口 `KvafCpuSglPushAivKernelV1` 与旧 AIV SGL 一起编入
`kvaf_execute_batch_get_kernels.o`；只测 AIV 可用 `./build.sh npu --without-aicpu`。

以下命令在 NPU 端执行，替换服务地址和卡号。先以相同 4 worker、4 SGE 比较
设备推送和旧 READ 模式，统一开启输出校验与两种执行方式：

```bash
export KVAF_CPU_SERVICE_ADDR=127.0.0.1:21111 DEVICE_ID=2
for backend in cpu_sgl_push aiv_cpu_sgl_push cpu_sgl_write aiv_cpu_sgl_write; do
  case "$backend" in
    aiv_*) workers=(--aiv-cores 4) ;;
    *) workers=(--aicpu-workers 4) ;;
  esac
  bash tests/run_remote_cpu_batch_get_perf.sh build-npu \
    --backend "$backend" "${workers[@]}" --sge-per-wr 4 --execution both \
    --request-length 8192 --table-capacity 8192 --miss-count 8192 \
    --token-bytes 656 --tokens-per-block 1 --seed 20260903 \
    --source-access-pattern random --output-access-pattern contiguous --no-sort-input-indices \
    --warmup 10 --iterations 100 --kernel-profile \
    --output-dir "reports/${backend}"
done
```

AIV 的 `--kernel-profile-clock-mhz` 应按目标设备频率设置。另给旧后端加
`--sgl-pipeline`、单独指定报告目录，可与旧逐 chunk 流水比较。
将 `--sge-per-wr 4` 改为 8 可比较 SGE 数；不要在新后端加 `--sgl-payload-only`。
真机正确性还应覆盖 `miss-count=0/7/1024/1025/65536`（相应调整容量）、1/4/8 worker，
并用 `--multi-graph-check` 验证非空请求的多图/多节点复用。

CPU profiling 输出 `CPU_SGL_PUSH_PROFILE`：`descriptor_reads=0`、`ready_wrs=0`，
并以 `source_validation=disabled address_validate_us=0.000` 标记 CPU 源地址扫描已关闭。
包括 head 首次提交、head 提交、后缀等待/提交、SGE 更新、output 导入、
整轮耗时，以及 `data_wrs/tail_cqes/max_inflight/plan_reused/result`。
这些阶段可嵌套并与 DMA 重叠，不能相加作为 kernel 时间。
设备 profile 的 scatter 和 ready 查询均为零，ack 包含 CPU 数据传输和完成等待。
关闭 CPU 源地址扫描只需更新、重建 CPU 服务并重启，NPU Host/kernel 和测试命令无需修改。

若在 `CPU_PUSH_INIT stage=register_control` 后报注册失败，先确认 CPU 服务已包含
mailbox 权限修复：新 mailbox 注册使用 `READ | WRITE`（access=6）。URMA 的
`LOCAL_ONLY`（bit 0）不能与远端权限组合，WRITE 注册还要求 READ；access=5 会被拒绝。
开放 READ 权限是注册约束，设备推送路径仍不发起描述符 READ。
此修复只改 CPU 服务，更新后重建 CPU 并重启服务即可，测试命令保持不变。

## 本地验证边界

```bash
KVAF_TEST_URMA_INCLUDE_DIR=/path/to/umdk/include \
  bash tests/diagnostics/run_cpu_sgl_host_checks.sh /tmp/kvaf-sgl-host-checks
```

该入口运行旧/新 AIV 和 AICPU 的真实控制流模型、CPU fake-provider 传输模型、
ASan/UBSan、GM 独立地址空间编译、Host 声明检查及 Python 配置回归。
新模型检查实际 AIV WQE 编码/拆分/SQ-CQ 回绕、多轮 sequence、部分/空请求、
head/suffix 错误、等待 done、零描述符 READ、head CQE 未收齐时提交 suffix，以及
在途 WR/SGE 不被复用，并模拟 URMA 注册权限的互斥和依赖校验。
没有 UMDK 头文件时跳过 CPU 传输模型和原生 UMDK 语法检查。

这些检查不替代 CANN/ccec 编译、真实 DMA 可见性、注册退役及 NPU 图回放验证。
