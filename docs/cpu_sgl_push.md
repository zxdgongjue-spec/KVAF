# CPU SGL Push：设备推送地址清单、CPU 直接写输出

新增独立算法 `cpu_sgl_push`，支持 AIV 和 AICPU。设备通过专用控制 Jetty 将请求头、
输出注册信息和源地址清单 WRITE 到 CPU mailbox；CPU 从本地 mailbox 消费请求，
用四条数据 Jetty 将离散 DDR 源 SGL WRITE 到调用方的连续 HBM output。
两种设备实现共用 CPU service 的 mailbox 消费、SGL 计划和完成协议，不执行 payload scatter。

| 算法/API | benchmark 后端 | 地址清单传输 | 流水 |
| --- | --- | --- | --- |
| AICPU `cpu_sgl_push` | `cpu_sgl_push` | 设备 WRITE 到 CPU | 默认 Head/Suffix，可选多段 |
| AIV `cpu_sgl_push` | `aiv_cpu_sgl_push` | 设备 WRITE 到 CPU | 默认 Head/Suffix，可选多段 |
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
# 多段设备推送：增加 sgl_push_chunked=True，原 Head/Suffix 默认不变。
# default_get=get, get_resources=[(get, resources)]
```

两种实现均支持 1..8 个设备 worker；CPU 固定四条数据 Jetty，另有一条描述符控制连接。
资源不接受 `jettys` 和 `max_inflight_total` 调整。`sge_per_wr` 与原 SGL 规则相同：
0 自动选择 `min(16, max_jfs_sge)`，显式值 1..16 必须不超过硬件上限。

设备推送的资源查询和性能脚本记录 `sgl_pipeline=True`；默认采用 Head/Suffix，
日志/CSV 标识 `sgl_pipeline_mode=device_push_head_suffix`。
新增资源开关 `sgl_push_chunked=True`，对应 CLI `--sgl-push-chunked`，选择 1024 行多段设备推送，
日志/CSV 标识 `sgl_pipeline_mode=device_push_chunked`，并记录 `sgl_push_chunked=1`。
该开关仅适用于两个 `cpu_sgl_push` 后端，与原 `--sgl-pipeline` 的逐 chunk READ 模式分别保留。
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

### 可选多段设备推送

启用 `--sgl-push-chunked` 后，AIV 和 AICPU 共用以下调度及 CPU service：

1. 将有效前缀划分为 1024 行一段，段号 `chunk` 由 `chunk % workers` 对应的设备 worker
   构建。每个 worker 逐段写 HBM 地址清单，再 release 发布本地完成段数。
2. 核/worker 0 统一操作控制 SQ；每轮从各 worker 取一段已就绪清单，WRITE 到 CPU 对应切片，
   最后 WRITE 该 worker 的 `Progress(sequence, completed)`。各 worker 间无需按段号全局排序。
   第一段额外携带请求头、output 描述和 notify；仍需先完成第一段的构建与提交。
3. 核 0 在自己的两次构建之间及等待其他 builder 时持续转发；某个 builder 未就绪时，
   可转发其他已就绪段。控制侧最多四条链在途，满窗才回收最早的 CQE。
   每段使用独立、已注册的 HBM commit 源槽位，保留到控制 CQE 完成；不直接 DMA
   正在变化的 builder 进度行。AICPU 的每条在途链也独占 WR/SGE 数组。
4. CPU acquire 读取本地 mailbox 的进度，从任意已就绪段更新 SGE 并提交数据 WRITE，
   不等待整批后缀。仍为一个 CPU service 线程、四条数据 Jetty、最多四个在途数据计划，
   并按 SQ 的 WR/SGE credits 排空；每个计划独占 WR/SGE 存储到数据 CQE。
5. CPU 数据全部完成后发布 done；设备也须回收全部控制 CQE 才能退出。
   后续图回放按 sequence 区分进度，无需 Host 清零 mailbox。

地址清单、请求头和进度全程无描述符 READ；CPU 源地址范围扫描保持关闭，无 payload scatter。
失败只发布一次带本轮 sequence 的错误进度，CPU 停止调度后续段并排空已接受的数据链，
随后返回失败 done；STOP 同样排空已接受的数据。失败 output 不可消费。
这是可重叠的“构建 → 推送地址 → CPU 更新/提交 → 数据 DMA”多段调度，
并不保证所有阶段在每个时刻都重叠；核 0 自己构建时不同时操作控制 SQ。

流水发布后发生后缀错误时，CPU 不提交错误后缀，排空已提交的合法数据后返回失败 done。
失败请求可能已写入 head，output 不可消费；按现有执行资源失效和重建流程处理。
Shutdown 先确认设备发送 kernel 已停止，再销毁描述符发送队列；CPU STOP 完成后才释放
HBM 注册。无法确认 kernel 结束时保留两端目标内存，报告 `CompletionUnknown`。

## 构建与比较

CPU service、NPU Host/Python 和设备 kernel 均须使用此版本重建，重启两端进程。
仅更新 Host 无法启用新协议；不自动回退到旧 READ 模式。
多段设备推送使用 setup output mode 5；旧 CPU service 会拒绝该模式，须同步更新两端。

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

对比两段与多段，只切换 `--sgl-push-chunked`，其余参数相同。以下使用 32768 容量、
8192 miss、656 B token 和每 block 128 token；CPU 服务地址和卡号沿用环境变量：

```bash
for mode in head_suffix chunked; do
  extra=()
  if [[ "$mode" == chunked ]]; then extra=(--sgl-push-chunked); fi
  bash tests/run_remote_cpu_batch_get_perf.sh build-npu \
    --backend aiv_cpu_sgl_push --aiv-cores 4 --sge-per-wr 4 "${extra[@]}" \
    --request-length 32768 --table-capacity 1277952 --miss-count 8192 \
    --token-bytes 656 --tokens-per-block 128 \
    --source-access-pattern random --output-access-pattern contiguous --no-sort-input-indices \
    --execution graph --warmup 10 --iterations 100 --kernel-profile \
    --output-dir "reports/aiv_push_${mode}"
done
```

AICPU 使用 `--backend cpu_sgl_push --aicpu-workers 4`，同样增加 `--sgl-push-chunked`。
复测原来的 5 worker 配置则改为 `--aicpu-workers 5`。多段 CPU 日志前缀为
`CPU_SGL_PUSH_CHUNK_PROFILE`，包含 `chunk_rows=1024 chunks=... descriptor_reads=0 ready_wrs=0`、
`first_post_us`、`chunk_ready_us`（本地进度扫描）、`source_sge_update_us`、
`payload_post_us`、`chunk_submit_us`、`chunk_cq_wait_us` 和 `max_inflight`。
`local_progress_polls` 是本地扫描轮数。`payload_post_us` 只计 provider 提交调用；
`chunk_submit_us` 也包含该段 SGE 更新等工作，两个值不能相加。

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

### 设备 Head 独立打点

新 NPU Host/Python 和 kernel 增加两个独立指标，原测试命令加上已有的
`--kernel-profile` 即会输出，并写入 `profile.csv`：

```text
BATCH_GET_PROFILE ... stage=head_build mean_us=... p50_us=... p99_us=...
BATCH_GET_PROFILE ... stage=head_doorbell mean_us=... p50_us=... p99_us=...
```

- `head_build`：核 0 构建前 `min(count, 1024)` 个源地址清单的函数耗时。
  AIV 包含 SIMT 查表/检查、UB 到 HBM 拷贝及其同步；AICPU 包含对应的地址填充循环和输出映射检查。
  请求头、输出描述准备和后缀构建不计入该值。
- `head_doorbell`：从本轮最早设备 worker 进入，到核 0 提交含 Head notify 的最后一批控制 WRITE。
  AIV 在实际 `st_dev` doorbell 后记录；AICPU 的 provider 内部操作 doorbell，故以
  `urma_post_jetty_send_wr` 成功返回作为可观察边界，不等待控制 CQE。
  日志的 `head_doorbell_boundary=doorbell/native_post_return` 标明两者区别。

原始记录仅核 0 保存 Head 耗时，其他核和空请求为 0，每轮回放重新初始化。
Python snapshot 的 AIV 原始字段为 `sgl_head_build_cycles`、`sgl_head_doorbell_cycles`，
AICPU 为 worker stages 中的 `head_build_us`、`head_doorbell_us`。
snapshot 的 doorbell 值相对核 0 自身入口；benchmark 根据各核入口时间换算为相对最早 worker，
避免漏掉发布核较晚开始执行的间隔。AIV 换算仍使用 `--kernel-profile-clock-mhz`。

这两个指标相互嵌套，并与原 `request/notify` 重叠，不能相加。
它们不包含 CPU 观察 notify 的延迟及首次数据 WRITE 的准备时间，也不测出地址清单的纯网络耗时。
默认仍为 Head/Suffix；启用多段时，Head 指第一段，以上两处打点保留。
多段的 `request` 记录初始化与各次地址构建，`notify` 记录控制提交/CQE 等待，
不将等待其他 builder 的空转计入 `request`。

Head 打点使用带版本标记的现有 profile 槽位，记录大小仍为 192 B，旧算法的字段含义不变。
测这两个新指标须同步重建 NPU Host/Python 和所选 kernel；CPU 服务无需为此重建。
Host/脚本遇到缺失或越过 kernel 时间窗口的 Head 记录会报错，避免将旧版本记录当作 0。

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
多段模型还检查 1/4/5/8 worker、乱序段就绪、暂停一个 builder 时其他段仍可转发、
跨轮次进度、错误/STOP 排空和零描述符 READ。原生 AICPU post/wait 检查四条在途链、
FIFO 回收/槽位复用及部分提交失败后 WR/SGE 存储保留；资源解析和 Python/子命令/CSV
回归检查新增开关确实传递到 Get 配置且默认关闭。
没有 UMDK 头文件时跳过 CPU 传输模型和原生 UMDK 语法检查。

这些检查不替代 CANN/ccec 编译、真实 DMA 可见性、注册退役及 NPU 图回放验证。
