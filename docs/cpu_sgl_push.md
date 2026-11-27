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
新增资源 `sgl_push_head_rows`，默认 1024；非默认值要求同时启用 `sgl_push_chunked=True`。
CLI 对应 `--sgl-push-chunked --sgl-push-head-rows 256`，AIV/AICPU 均支持 1..1024。
例如 `kvaf.ExecutionResources(cores=4, sge_per_wr=4, sgl_push_chunked=True, sgl_push_head_rows=256)`。
Head 行数按执行资源固定，图回放可改变有效请求长度；配置与 CSV 记录 `sgl_push_head_rows`。
可选 `sgl_push_cooperative=True` / `--sgl-push-cooperative`，让所有设备核共同构建每段，
要求同时启用多段，且 H 为 8..1024 中的 8 的倍数。默认关闭，保留原按段分核方案。
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
其中 AIV 的 chunk 0 构建完成后直接推送，跳过通用转发状态机对本核进度和管理状态的回读。
首段 doorbell 之后才初始化本轮 `sent/pending/notified/total`，并发布本核的本地进度；
随后立即转发其他已就绪段，再继续本核下一段构建。管理状态使用逐字段 GM 写入，按 cache line
集中刷回，并在下一次 SIMT 前持久化；每段 commit 源仍保留到控制 CQE 完成。
空请求和首段构建失败在退出前初始化为未发布状态；部分提交失败保留已尝试通知的标记，
继续按原失败完成/超时规则处理。此优化只需重建 NPU 的 AIV kernel，测试命令和多段开关不变。

失败只发布一次带本轮 sequence 的错误进度，CPU 停止调度后续段并排空已接受的数据链，
随后返回失败 done；STOP 同样排空已接受的数据。失败 output 不可消费。
这是可重叠的“构建 → 推送地址 → CPU 更新/提交 → 数据 DMA”多段调度，
并不保证所有阶段在每个时刻都重叠；核 0 自己构建时不同时操作控制 SQ。

### 小 Head 与后续 1024 行分段

设配置 Head 行数为 H，第一段为 `[0, min(count, H))`；第 c 段（c≥1）为
`[H+(c-1)*1024, min(count, H+c*1024))`。默认全局段号仍由 `c % workers` 的 worker
构建，CPU、设备发布和 SGE/WR 计划共用同一套边界计算。空请求仍不通知 CPU。
H=256、count=8192 时共有 9 段：256 行 Head、7 段 1024 行、最后 768 行。
在四条数据 Jetty、SGE4 下仍是 2048 个数据 WR，按段提交为 36 次 post 和 36 个尾 CQE；
默认 H=1024 为 8 段、32 次 post/尾 CQE。缩短 Head 不保证同比缩短 kernel，须同时看后续供给和提交开销。

默认 H=1024 保留 setup mode 5、原工作区大小和 launch ABI。小 Head 使用独立 setup 标记：
`output_mode` 低 8 位为 6，高位保存 H；请求头 `reserved_words[0]=1024`、`[1]=H`。
CPU 先验证协商值，再接受请求头；旧 CPU 服务会拒绝该 setup，旧 kernel 发出的固定 1024 头也会被拒绝。
65536 行最多产生 65 段，新增第 65 个成功 commit 源放在原 `ChunkPushState` 尾后的一条 Word；
原 64 个成功槽位、错误槽位和管理状态的偏移均保留，每条源数据继续保持到其控制 CQE 完成。

### 可选各核协同构建

开启 `--sgl-push-cooperative` 后，AIV 和 AICPU 均按以下方式构建；CPU 的数据计划、
四条数据 Jetty、单线程提交和在途窗口保持相同：

1. 每个 worker 都依次参与第 0、1、2…段，在每段内按 8 个地址一组分配连续子区间。
   每个 64 B 地址清单缓存行只有一个写入核；H 要求是 8 的倍数，末段可以不足 8 行。
   某些子区间为空时该 worker 仍发布完成进度。
2. 本地 Ready 表示该 worker 已完成的全局段数。核 0 取所有 worker 进度的最小值，
   只有全段各部分均就绪才推送该段。Head 同样由所有核构建，核 0 汇齐后直接提交。
3. 无需所有核每段停在同一个 barrier。非发布核完成自己的子区间后可以继续构建后段；
   最多提前构建四段，再等待核 0 对更早段写入本轮不可变 commit 源槽位。
   该标记表示发布已经准备，可能早于 doorbell，不代表控制 DMA/CQE 已完成。
4. 核 0 每段只构建自己的一部分，在段间及等待窗口时转发；控制侧仍最多四条链在途。
   对 CPU 继续发布原 `chunk % workers` 虚拟 owner 的进度，CPU 无需改为全核聚合。
   各段按段号发布，单个慢核会阻挡相应段；该代价需要和并行构建收益一起实测。

例如 H=512、4 核时，Head 子区间分别为 `[0,128)`、`[128,256)`、
`[256,384)`、`[384,512)`；下一段为 `[512,1536)`，各核各构建 256 项。
核 0 构建和发布仍然串行执行，但其单次构建量缩小为约 1/workers。
`head_build` 在此模式下从核 0 开始构建自己的 Head 部分，到汇齐所有 Head 部分为止，
包含汇齐等待；`head_doorbell` 口径不变。其他核的 Head 打点仍为零。

该模式使用 setup mode 7（高位保存 H）和请求头 `reserved_words[2]=1` 双重校验，
避免旧 kernel 忽略协同标记后混用进度含义。须同步重建 CPU 服务、NPU Host/Python 和 kernel。
默认按段分核仍使用原 mode 5/6；两种策略可在同一版本内切换。
配置/CSV 记录 `sgl_push_cooperative`，设备配置输出 `sgl_push_build_mode`，
CPU chunk 日志输出 `build_mode=cooperative/round_robin`。

### AICPU 专职发布核与 AIV 仅 Head 协同

两个实验模式分别保留开关，默认关闭；本轮不修改 CPU 的数据 WR 计划、提交粒度、
128 WR post 上限、四条数据 Jetty、四计划在途窗口及 CQ 调度，不做 CPU 后段合批。

| 模式 | 配置 | 设备行为 |
| --- | --- | --- |
| AICPU 专职发布 | `sgl_push_chunked=True, sgl_push_cooperative=True, sgl_push_publisher_only=True` | Head 全部 worker 协同；之后 0 只发布，其余 worker 协同构建后段 |
| AIV 仅 Head 协同 | `sgl_push_chunked=True, sgl_push_head_cooperative=True` | Head 全部核协同；后续段按原 `chunk % workers` 分核，0 继续构建和统一发布 |

两种模式均要求 Head 行数为 8..1024 中的 8 的倍数。AICPU 专职发布仅支持 AICPU、
至少 2 worker，要求同时打开全段协同；5 worker 时 Head 分给 5 个 worker，后段分给 1..4。
发布 Head 前依然汇齐所有 worker，发布后只聚合 1..N-1 的后段前缀，0 的进度停留在 1。
后段仍保留四段提前构建窗口。该模式不会额外创建 worker，核数仍为原配置总数。

AIV 仅 Head 协同仅支持 AIV，与全段协同及专职发布互斥。Head 分片构建后，各核可直接
继续自己的整段后缀，无需全局复位进度。每个本地 Ready 使用独立的 Head 完成 bit 与
原循环 owner 完成段数；该 bit 只在设备内部使用，CPU 收到的仍是原进度格式。
因此“worker 1 的 Head 部分已完成”不会被当作“第 1 个后续段已完成”。
Head=1024、4 核、8192 行时，共有 4 次 Head 分片构建和 7 次后段构建；
0 核仍负责后段 4，其他核负责后段 1/5、2/6、3/7。

两者的 `head_build` 都包含 0 核自己的 Head 构建及汇齐等待；其他核和空请求仍为零。
`head_doorbell`、kernel 和端到端的计时口径保持不变。

协议分别使用 setup mode 8/9（高位为 Head 行数），请求头 `reserved_words[2]=2/3`。
CPU 只增加模式协商、请求头核对和 `build_mode=publisher_only/head_cooperative` 标识，
数据调度仍按原虚拟 owner 消费。须同步重建 CPU 服务、NPU Host/Python 和 kernel，重启两端；
旧 CPU 会拒绝新 setup，旧 kernel 的请求头标记也会在数据 WRITE 之前被拒绝。

以下在 NPU 仓库根目录执行，两组均固定 Head=1024、SGE4、ACL 图模式：

```bash
# AICPU：5 worker，0 核在 Head 后专职发布
KVAF_CPU_SERVICE_ADDR=141.62.38.71:21112 DEVICE_ID=7 \
bash tests/run_remote_cpu_batch_get_perf.sh build-npu \
  --backend cpu_sgl_push --aicpu-workers 5 --sge-per-wr 4 \
  --sgl-push-chunked --sgl-push-cooperative --sgl-push-publisher-only \
  --sgl-push-head-rows 1024 --execution graph --graph-api acl \
  --request-length 32768 --table-capacity 1277952 --miss-count 8192 \
  --token-bytes 656 --tokens-per-block 128 \
  --source-access-pattern random --output-access-pattern contiguous --no-sort-input-indices \
  --warmup 10 --iterations 100 --kernel-profile \
  --output-dir reports/aicpu-publisher-only/head1024-sge4

# AIV：4 核仅 Head 协同，后段恢复按段分核
KVAF_CPU_SERVICE_ADDR=141.62.38.71:21112 DEVICE_ID=7 \
bash tests/run_remote_cpu_batch_get_perf.sh build-npu \
  --backend aiv_cpu_sgl_push --aiv-cores 4 --sge-per-wr 4 \
  --sgl-push-chunked --sgl-push-head-cooperative \
  --sgl-push-head-rows 1024 --execution graph --graph-api acl \
  --request-length 32768 --table-capacity 1277952 --miss-count 8192 \
  --token-bytes 656 --tokens-per-block 128 \
  --source-access-pattern random --output-access-pattern contiguous --no-sort-input-indices \
  --warmup 10 --iterations 100 --kernel-profile \
  --output-dir reports/aiv-head-cooperative/head1024-sge4
```

AICPU 去掉 `--sgl-push-publisher-only` 即为原全段协同对照；AIV 去掉
`--sgl-push-head-cooperative` 即为原按段分核对照。各自使用不同报告目录。
配置输出和 CPU profile 应分别显示 `publisher_only` / `head_cooperative`。
资源查询、Python 构造参数、子命令和 summary.csv 同时记录两个新布尔开关。

流水发布后发生后缀错误时，CPU 不提交错误后缀，排空已提交的合法数据后返回失败 done。
失败请求可能已写入 head，output 不可消费；按现有执行资源失效和重建流程处理。
Shutdown 先确认设备发送 kernel 已停止，再销毁描述符发送队列；CPU STOP 完成后才释放
HBM 注册。无法确认 kernel 结束时保留两端目标内存，报告 `CompletionUnknown`。

## 构建与比较

CPU service、NPU Host/Python 和设备 kernel 均须使用此版本重建，重启两端进程。
仅更新 Host 无法启用新协议；不自动回退到旧 READ 模式。
多段设备推送使用 setup output mode 5；旧 CPU service 会拒绝该模式，须同步更新两端。
小 Head 还需要两端包含 mode 6 的协商及分段支持；启用前重建 CPU 服务、NPU Host/Python 和 kernel，重启两端。

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

先固定一种图 API，再比较 Head 行数，例如在 NPU 端沿用服务地址和卡号：

```bash
for head in 1024 512 256 128; do
  bash tests/run_remote_cpu_batch_get_perf.sh build-npu \
    --backend aiv_cpu_sgl_push --aiv-cores 4 --sge-per-wr 4 \
    --sgl-push-chunked --sgl-push-head-rows "$head" \
    --execution graph --graph-api acl \
    --request-length 32768 --table-capacity 1277952 --miss-count 8192 \
    --token-bytes 656 --tokens-per-block 128 \
    --source-access-pattern random --output-access-pattern contiguous --no-sort-input-indices \
    --warmup 10 --iterations 100 --kernel-profile \
    --output-dir "reports/aiv-acl-head-${head}"
done
```

AICPU 替换为 `--backend cpu_sgl_push --aicpu-workers 5`，使用另一组报告目录。
比较构建策略时固定 Head、核数和图 API，只切换协同开关，例如：

```bash
for strategy in round_robin cooperative; do
  extra=()
  if [[ "$strategy" == cooperative ]]; then extra=(--sgl-push-cooperative); fi
  KVAF_CPU_SERVICE_ADDR=141.62.38.71:21112 DEVICE_ID=7 \
    bash tests/run_remote_cpu_batch_get_perf.sh build-npu \
    --backend aiv_cpu_sgl_push --aiv-cores 4 --sge-per-wr 4 \
    --sgl-push-chunked --sgl-push-head-rows 512 "${extra[@]}" \
    --execution graph --graph-api acl \
    --request-length 32768 --table-capacity 1277952 --miss-count 8192 \
    --token-bytes 656 --tokens-per-block 128 \
    --source-access-pattern random --output-access-pattern contiguous --no-sort-input-indices \
    --warmup 10 --iterations 100 --kernel-profile \
    --output-dir "reports/aiv-acl-head512-${strategy}"
done
```

AICPU 同样使用此开关，改为 `--backend cpu_sgl_push --aicpu-workers 5` 并更换报告目录。
先比较同一 Head 的策略差异，再独立比较 Head=1024/512/256/128。
按段分核时 `head_build` 测的是前 `min(count, H)` 行的实际构建，协同时包含汇齐等待；
`head_doorbell` 的计时起点和提交边界保持不变。
CPU 日志增加 `head_rows=H`，并新增完整的无就绪段时间：

- `no_ready_us`：首次扫描确认没有可提交段，到后续发现就绪段的区间累计，包含区间内的扫描、循环和 CQ 回收。
- `no_ready_inflight_us`：上述区间中，CPU 仍跟踪着未回收数据计划的部分。
- `no_ready_idle_us`：上述区间中，CPU 没有在途数据计划的部分。

后两者之和等于 `no_ready_us`；已知有就绪段但因 SQ credits/在途窗口等待的时间不计入。
这些都是 CPU 观察到的 wall-clock 区间，包含 CPU 调度影响；“在途”不证明 DMA 始终忙碌。
它们与 `chunk_ready_us`、部分 `chunk_cq_wait_us` 重叠，不能相加。以 kernel/端到端耗时判断净收益。

### CPU 整轮计时、掩盖与相加关系

两个设备推送后端共用同一 CPU service。`cpu_total_us` 从 CPU acquire 观察本轮 Head
notify 后开始，到收回最终 Done WRITE 的 CQE 结束；不包含此前设备的 Head 构建及通知传输。
profile 打印在该窗口外。CPU 一个线程依次处理请求、提交/回收四条 Jetty 的数据计划，
多队列和最多四个在途计划提供异步传输并发，并非四个 CPU 提交线程。

```text
CPU：观察 notify → 检查/导入/计划准备 → 扫 Ready → 更新 SGE/post → … → 排空数据 → unimport → Done post/CQE
DMA：                                          └── 数据传输 ───────────┘
设备：Head 构建/通知 ─────────────── 后段构建及地址推送 ────────────── 等待 Done
```

| 日志字段 | 计时边界 | 相加关系 |
| --- | --- | --- |
| `first_post_us` | CPU 起点到第一次数据 provider post 调用的入口 | 前缀；包含检查、导入、计划准备及首批 SGE 更新，不是额外阶段 |
| `chunk_ready_us` | 累计 Ready 扫描调用的时间 | 单线程内与提交/CQ 调用互斥；不代表完整地址等待 |
| `chunk_submit_us` | 累计 `PostPrepared` 时间 | 包含 SGE 更新、数据 provider post、循环/统计开销 |
| `source_sge_update_us` | 逐批把源地址填入缓存 SGE | 是 submit 子项，可与 provider post 相加解释 submit |
| `payload_post_us` | 数据 `urma_post_jetty_send_wr` 调用累计时间 | 是 submit 子项，不测异步数据传输总时间 |
| `chunk_cq_wait_us` | 数据 CQ 轮询累计时间，包括非阻塞回收和阻塞等待 | 与 submit/Ready 扫描互斥；含轮询/调度开销，不等于纯 DMA 时长 |
| `done_post_us`、`done_cq_wait_us` | 数据排空、output 解除导入之后的 Done 提交和 CQ 等待 | 收尾串行阶段，已在 cpu_total 内 |
| `no_ready_us` | 首次确认无就绪段到发现就绪段的完整区间累计 | 与 Ready 扫描及部分 CQ 回收重叠，不能再加进前三项 |

成功请求的 CPU 路径依次为：检查本地请求头及 output 描述、导入输出 segment、
准备或复用 WR/SGE 计划并更新输出目标、调度各段、排空数据、解除 output 导入、发送及等待 Done。
当前 chunk 日志没有打印全部导入/解除导入、计划准备和外层循环成本，不能仅用已打印字段
精确还原 cpu_total。`plan_reused=1` 表示计划存储复用，仍需更新输出目标及本轮源 SGE。
源地址范围扫描在 device push 下已关闭；头部、进度与输出边界检查继续执行。

可以使用以下恒等关系，输出舍入可能带来微小误差：

```text
no_ready_us = no_ready_inflight_us + no_ready_idle_us
chunk_submit_us = source_sge_update_us + payload_post_us + 其余提交开销
cpu_total_us = chunk_ready_us + chunk_submit_us + chunk_cq_wait_us
               + done_post_us + done_cq_wait_us + 其余 CPU 窗口时间
```

最后一项包含检查/导入/计划/解除导入、扫描外的等待循环、调度和计时边界开销，
不能全部归为等待地址；也不能再把 first_post 或 no_ready 加到右侧。
首批 post 后，前批 DMA 可掩盖后续 SGE 更新、post 和设备后段构建，但实际掩盖量
无法由这些累计计时直接推出。`no_ready_inflight` 只有“可能被在途传输掩盖”的含义，
CQE 可能已经到达而尚未回收；`no_ready_idle` 没有既有数据计划可掩盖，是更明确的供给断档。
数据全部提交后的末尾排空、解除导入及 Done 是收尾依赖；准备首批数据之前也没有本轮数据 DMA 可掩盖。

设备 `head_build` 嵌套于 `head_doorbell`，不能相加。后者与 cpu_total 使用不同侧的边界：
中间遗漏通知可见性间隔，末尾设备看到 Done 与 CPU 收到 Done CQE 也不同；
因此 `head_doorbell + cpu_total` 不能精确拼成 kernel，`kernel - cpu_total` 也不是纯设备计算时间。
总收益使用相同图 API、相同工作量及采样条件下的 kernel/端到端数据判断。

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

- `head_build`：默认按段分核时，核 0 构建前 `min(count, H)` 个源地址清单的函数耗时，H 默认 1024。
  AIV 包含 SIMT 查表/检查、UB 到 HBM 拷贝及其同步；AICPU 包含对应的地址填充循环和输出映射检查。
  请求头、输出描述准备和后缀构建不计入该值。
  协同构建时改为核 0 自己的 Head 构建及汇齐所有 Head 部分的时间，见上节。
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
多段的 `request` 记录启动检查与各次地址构建，`notify` 记录控制提交/CQE 等待，
以及 AIV 首段 doorbell 后的流水管理初始化，
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
小 Head 覆盖 128/256/512、非对齐 1/7/1023、Head/后续段边界、65 段上限、
旧式请求头/错误协商拒绝、失败/STOP、不同输出目标的计划复用；延迟进度发布的模型验证
`no_ready_inflight_us` 覆盖完整等待区间而不只是本地扫描耗时。
协同模型覆盖所有核的实际构建范围与缓存行唯一所有权、短 Head/空子区间、慢核导致 Head 延迟、
其他核可提前构建、Head 发布前/后失败、65 段、多轮进度和 mode 7/请求头标记不匹配拒绝。
新增模式检查 AICPU 0 核的构建范围仅为其 Head 分片、剩余 worker 覆盖所有后段、
0 的前缀停在 1 时发布仍能完成；AIV 检查 Head 分片加循环后段的实际复制范围、
暂停后段时不会把 Head 完成误当作后段就绪，并覆盖空请求、空分片、失败、65 段及跨轮次进度。
CPU fake-provider 对 mode 8/9 验证成功、错误/STOP 排空及四种请求头标记之间的错误混用拒绝。
没有 UMDK 头文件时跳过 CPU 传输模型和原生 UMDK 语法检查。

这些检查不替代 CANN/ccec 编译、真实 DMA 可见性、注册退役及 NPU 图回放验证。
