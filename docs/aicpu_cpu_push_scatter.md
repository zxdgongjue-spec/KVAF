# AICPU CPU Push Scatter：构建、调用与验证

需要 CPU SGL 直接写调用方连续 output、跳过 AICPU scatter 时，见
[CPU SGL Write](cpu_sgl_write.md)。

本页描述 Scatter 数据流和兼容接口示例。按调用选择算法、驻留资源和 Get 图模式见
[实现与使用](cache_transport_execution_implementation.md)；统一功能与性能比较见
[验证步骤](cache_transport_execution_validation.md)。

需要由 CPU 将离散 DDR 数据直接送入连续 HBM、再由 AICPU 写入调用方输出时，
使用独立的 Put/Get 执行配置和资源预算：

```python
import torch
import torch_npu
import kvaf

torch.npu.set_device(7)
put = kvaf.PutExecutionConfig("urma", "aicpu")
get = kvaf.GetExecutionConfig("urma", "aicpu", "scatter")
runtime = kvaf.NpuRuntime(
    batch_get_token_bytes=656,
    batch_get_tokens_per_block=2,
    device_id=7,
    transport_mode="urma", default_put=put, default_get=get,
    put_resources=[(put, kvaf.ExecutionResources(cores=4))],
    get_resources=[(get, kvaf.ExecutionResources(cores=4))],
)
runtime.link_to("127.0.0.1:28990")
runtime.initialize(ddr_bytes=656 * 2 * 8192, slot_size=656 * 2)
# 调用 batch_put / batch_get 等接口，结束后执行 runtime.shutdown()。
```

其余 `batch_put`、AccessTable 和 `batch_get` 调用保持现有方式。
Runtime 自动定位 CANN 安装目录中的 AICPU kernel JSON，无需公开路径参数；安装路径规则见 [Gather 使用说明](aicpu_gather_read.md)。
`GetExecutionConfig("urma", "aicpu", "scatter")` 选择 Scatter，`gather` 选择 Gather；
两者均支持原有两种 Table 重载。Put 由独立的 `default_put` 或本次 `execution` 选择。
AICPU 仅支持 URMA；Runtime 的 transport 模式须允许 URMA。AICPU 资源项不传 kernel
路径，设备包在首次准备时解析；`gather_read_direct` 不对外开放。
CPU push 使用首个自动发现候选，也可通过严格的 `npu_eids=[eid1, eid2]` 指定列表并使用首项；
不执行启动探测、候选自动切换、多 EID 条带或自适应调优。
Get 使用 Host direct HCCP/RA；Put 使用 AICPU UMDK session，两者使用同一个 Host 选定 EID。
UMDK 根据选定 EID 匹配 device/index，匹配或准备失败会报错，不再提供独立过滤参数。
控制参数有更新，Host 库和设备包必须一起重建；仍保留单一 UMDK 控制入口。
CPU push 的 V3 数据/请求协议及 TP 管理保持本分支原有实现。
CPU 使用四路 Jetty，由所属服务线程按批轮转提交预构建链；
资源项 `cores` / `--aicpu-workers` 控制 AICPU worker 数。

## 构建并验证

两端代码、设备 SO 和 kernel JSON 一起更新，重启 CPU 服务及 NPU 进程。
当前 push 协议为 V3，逐行复制源于 `codex/aicpu-ordered-write` 的
`a12bb16c120e885eb18ed61581739e1c212c32e8`（V24），Ready 由 AICPU 按 Jetty 分片消费。
当前设备入口为 `KvafAicpuCpuPushScatterExecutionV3`。主机库和设备 SO 必须一起更新，
包括移除废弃排序字段后的 profile 布局。以下命令从各自仓库根目录执行。
CPU 端构建服务：

```bash
./build.sh cpu
```

NPU 端先加载实际安装的 CANN `set_env.sh`，再统一构建：

```bash
./build.sh npu
./build.sh npu --check-kernel
export PYTHONPATH="$PWD/build-npu/python${PYTHONPATH:+:$PYTHONPATH}"
```

默认包含 AIV、UBmem、AICPU Put/Gather/Scatter，输出一份 Python 包，并在构建成功后
自动安装 AICPU 包，无需 `--install-kernel`。安装需要对所选 CANN 包目录及加载配置有写权限。
`ASCEND_HOME_PATH` 和 `ASCEND_CANN_PACKAGE_PATH` 同时设置时须指向同一真实目录；
构建和运行必须选择同一套 CANN。

仅构建 AICPU 时，可用 `./build.sh aicpu-urma --cpu-push`，同样自动安装；将本文的
`build-npu` 换为 `build-aicpu-urma`，安装检查改用
`./build.sh aicpu-urma --cpu-push --check-kernel`。
不带 `--cpu-push` 的 AICPU-only 构建只提供 Gather/Put；`npu --without-aicpu`
仅提供 AIV/UBmem，两者都不能运行 Scatter。

设备 `memcpy_sp` 需要 securec 头文件和 AArch64 动态库，构建会查找 CANN 内的依赖并打包。
设备 scatter 单独按参考提交使用 `-O3`；其余设备 UMDK/session 源码继续使用 `-O2`。
CPU 构建链接 OpenMP，并启用 `-O2`、保留调试信息，push 提交由一个服务线程执行。
`--cpu-push` 对应 CMake 的 `KVAF_ENABLE_AICPU_CPU_PUSH=ON`；`npu` 默认开启，
`aicpu-urma` 入口需显式传入该选项。
原有 gather-only 构建不查找或链接 securec/direct HCCP。
依赖安装路径不同时，可设置 `KVAF_AICPU_SECUREC_INCLUDE_DIR` 和 `KVAF_AICPU_SECUREC_LIBRARY`。
初始化会检查 UMDK Control、BatchPut 和 Scatter 入口；缺少 Scatter 时在远端 DDR
分配前报错，不自动回退到 Gather 或 AIV。

CPU 端启动同一个服务：

```bash
./build-cpu/runtime/kvaf_cpu_service \
  --listen 0.0.0.0 --port 21111 \
  --gather-threads 8 --gather-mode active
```

两个 `--gather-*` 参数只影响 Gather，供下面的双模式对照使用；单独运行 Scatter 时
可省略，它们不控制 CPU Push 的线程数。CPU Push 由所属服务线程提交，AICPU worker
数由客户端 资源项 `cores` / `--aicpu-workers` 控制。
示例设备号和本机服务地址按部署调整；跨主机时替换为 CPU 服务的可达地址。
首次准备 Scatter 时的日志应包含：

```text
CPU_PUSH_REFERENCE commit=a12bb16c120e885eb18ed61581739e1c212c32e8 protocol=3 copy=per_row_memcpy_sp barrier=dsb_sy device_optimization=O3 workspace_policy=huge_first
```

性能测试使用下一节的统一入口；通过 `--backend` 选择 Gather 或 Scatter，
通过 `--aicpu-workers` 设置 worker 数。实测 CPU Push Scatter 在 8 个 AICPU worker 时存在问题，
示例使用 5 个 worker。

## 使用统一性能入口

CPU 服务启动并加载 CANN 环境后，运行四种 Get 共用的入口：

```bash
KVAF_CPU_SERVICE_ADDR=127.0.0.1:21111 DEVICE_ID=2 \
bash tests/run_remote_cpu_batch_get_perf.sh build-npu \
  --backend cpu_push_scatter --aicpu-workers 5 --execution both \
  --request-length 32768 --table-capacity 4096 --miss-count 4096 \
  --token-bytes 656 --tokens-per-block 128 \
  --seed 20260903 --no-sort-input-indices \
  --source-access-pattern random --output-access-pattern contiguous \
  --warmup 10 --iterations 100 --kernel-profile
```

`BATCH_GET_RESULT scope=end_to_end` 报告 Host 调用到完成的延迟和带宽；
`scope=kernel` 报告最早 worker 进入到最晚 worker 退出的设备时间和带宽。
设备时间包含算子中的 CPU/网络等待及 profile 打点。
`BATCH_GET_PROFILE` 保留各 worker 阶段的统计，阶段可嵌套和重叠，不能相加为设备总时间。
数据初始化、请求生成、预热、输出校验和 profile 快照在正式性能采样之外。
去掉 `--kernel-profile` 仍报告端到端和算子时间、带宽，不输出 worker 分段明细。

源池大小由 `--table-capacity` 和块布局显式决定，不再由独立后端脚本分配 100 GiB。
扫描不同 miss 数、token 大小或 worker 数时，循环调用同一个入口。
详细口径及其他后端参数见 [统一 Get 性能测试](batch_get_performance.md)。

## 请求与数据流

一次调用处理 `int32 miss_count` 指定的有效前缀，最多 65536 项。
每项读取 `addresses[indices[i]] + offsets[i]` 的 `token_bytes` 字节，
目标为 `output + output_indices[i] * token_bytes`。
调用方继续负责有效 key、token 布局、输出映射及这些资源在操作完成前的生命周期。

初始化时，Host 用 `ACL_MEM_MALLOC_HUGE_FIRST` 申请 HBM workspace，再通过 RA 注册并创建四个 TargetChannel，
将 segment、Jetty key 和 TP 信息发给源 DDR 所属 CPU Runtime。
`workspace_policy=huge_first` 记录申请策略；大页不足时可回退普通页。
CPU 建立专用回写队列，返回对端信息；Host 完成导入后才激活驻留 CPU worker。
激活时先同步 READ 请求控制字并 WRITE 初始完成字，确认实际双向 RMA 路径可用。
这套控制协商通过现有 PeerLink 完成，无需新增 NPU listener。

每轮过程如下：

1. Host 在调用方 stream 启动配置数量的实例；设备通过常驻 coordination 的 ticket
   取得本轮 worker 编号和 sequence，不在每次提交前由 Host 清零 coordination。
2. 各 worker 并行查表，将本核分片的源地址写入原请求位置；worker 0 填写请求头。
3. 所有非空分片可见后，worker 0 发布一次 notify，携带 sequence 和可选 profile 位。
   CPU READ 请求头，验证后 READ 有效源地址前缀，
   由所属线程更新预构建 SGL 的源地址。
4. CPU 直接从已注册 DDR 源 WRITE 到 packed HBM staging，不打包 payload 到 CPU 临时区。
5. 按 worker 的 staging 区间划分 1024 行 chunk，各 Jetty 在其数据之后强序 WRITE ready。
6. AICPU 读取四个 Ready 字，独立推进各 Jetty 的消费进度。将本次新就绪的连续 staging
   行合并为一个复制区间，再按原始 `output_indices[i]` 逐行执行 `memcpy_sp`。
   Ready 读取前、复制前及 worker 复制完成后执行 `dsb sy`。
7. CPU 收齐数据链尾 CQE 后发布 done；全部 scatter 与 done 均完成后 kernel 结束。

ready 用高 32 位 sequence、低 32 位 chunk 数编码；done 在同一个 64 位字内编码 sequence 和结果。
请求序号由设备 ticket 推进，达到 uint32 上限后要求重建相应执行资源。
空前缀不通知 CPU、不修改输出；CPU 允许此类调用造成的序号间隔。
普通 stream enqueue 和 graph replay 使用相同的设备序号机制；图租约固定
算法、stream、输出和容量；每个 Get 节点使用独立租约和报告。可用 `graph_resources()`
自动收集节点；不同图可使用不同 stream，与普通 Get 的串行顺序由调用方保证。
捕获与结束流程见实现文档。

CPU 数据面沿用该提交 V24 的 prepared WRITE、indexed CQ 和轮转 CQ 扫描，
CPU 与设备共享 worker 分区和 chunk 枚举逻辑。
每个 chunk 按连续行分到四个 Jetty；请求数量不变时复用数据/ready 链，每轮只更新源地址和序号。
所属线程按 post 更新 SGL 并轮转提交四路队列，收齐数据链尾 CQE 后提交预构建的 done 链，
消除每轮 OpenMP team 唤醒和多次 barrier；多 wave 请求仍逐 wave 排空再复用 SQ。
每个 Jetty 的 SQ 深度为 8192，每次 post 最多 128 个 payload WR，尾随的 Ready WR
留在同一次 post 内。CQ batch 为 8，每 32 次扫描检查超时。
超出单 wave 容量的请求分 wave 执行，完成前保留全部 WR/SGL；每个 ready 使用独立的注册源字。
Runtime 按每次 BatchGet 启动 kernel，接收本轮动态源地址和 `output_indices`。
ready 热循环与 V24 一样只读 worker 协调状态；Runtime 的远端错误完成字在阻塞时每
32 次轮询检查一次，在最终 done 等待中逐次检查。

## 与指定提交的对应关系

生产代码沿用参考提交的缓存行分区、1024 行 chunk 和逐行复制算法。
`cpu_push_get_layout.h` 提供 CPU/设备共享的分区，`cpu_push_pipeline.h` 实现消费与复制。
不再携带独立测试的 ABI、结果统计、校验协议或结果转换适配层。

以下接口和工作量差异保留，用于支持 Runtime 的动态请求，不能由参考代码直接覆盖：

| 项目 | a12bb16 V24 独立测试 | 当前 Runtime |
| --- | --- | --- |
| 执行 | 一次 launch，设备内循环所有轮次 | 每次 BatchGet 独立 launch；统一性能入口支持异步入队和同步调用 |
| 源 SGL | CPU 预先构建，逐轮修改 ready 序号和源数据标签 | 每轮从动态 Tensor 解析源地址，CPU READ 地址清单并更新 SGL |
| 输出 | 预先安排顺序输出，656 B 行使用 768 B stride | 按动态 `output_indices` 写入 656 B slot，默认测试随机排列 |
| 校验 | 源标签每轮变化，末轮完整读回输出 | 请求生成与输出校验在性能计时之外 |
| 带宽计时 | 驻留 kernel 内 request 到 scatter/done | Host 端到端时间；启用 profile 后另报设备时间 |

`--output-access-pattern contiguous` 可单独测量目标访问顺序的影响；它不改变 slot 大小或带宽计时范围，
也不能把当前结果等同于旧常驻测试的结果。

## 内存、缓存和完成

HBM workspace 包含独立缓存行上的请求头、notify、done、各 worker 的四路 ready，
随后是源清单和连续 staging。
staging 容量为 `65536 * token_bytes`，另有 512 KiB 的源地址数组；
workspace 在初始化时分配，每轮复用，总 payload 必须能用 uint32 表示。
此路径不预留远端 DDR gather staging，也不注册调用方 output 为 URMA 目标。

此路径使用部署环境提供的 DMA 可见性，不执行手动缓存行刷新。
算子只遍历输入生成源清单和执行 scatter，不逐项校验 Tensor 内容或校验 payload。
调用方须提供有效的索引、偏移、源地址和输出行；保留请求容量和有效前缀长度检查。

一次执行完成前不复用 staging。异步返回表示已经入队；同一 Get 方向的下次普通调用会先 Drain，
检查前一轮执行状态。`synchronize=True` 在本次调用内完成等待和状态检查。
Shutdown 先停止 CPU worker 并销毁其 DMA 队列，再等待设备执行、释放 HBM。
传输或 stream 失败会阻止执行资源复用；关闭失败时保留资源以便重试。

## Profile

`request_us` 记录每个 worker 的源地址分片构建，含发布屏障；worker 0 还计入请求头填写。
分片汇合等待不计入 `request_us`，`scatter_us` 记录本地复制。
`scatter_copies` 为实际逐行 memcpy 次数。4096 × 656 B、完整前缀、4 worker 时，
每个 worker 为 1024 次，总计 4096 次。
push 路径的 `read_us` 是消费 ready 并执行 scatter 的区间，`ack_us` 是最终 done 等待；
`source_write_us` 为零，因为本轮源清单由 CPU READ 获取。
缓存维护相关字段为零。设备 worker 的区间重叠，不能相加作为端到端时长。

启用 CPU Push profiling 时，CPU 服务在采样请求完成后输出
`CPU_PUSH_PROFILE`。统一性能入口的 `--kernel-profile` 控制测试端的分段明细输出。
`header_read_us` 是请求头 READ 耗时，`address_read_us` 是有效源地址数组 READ 耗时；
`write_us` 包含 `build_us`、`post_us`
和 `cq_wait_us`，分别表示计划构建/源地址更新、URMA post 调用、提交后的 CQ 等待。
`submit_to_cqe_us` 从首次 payload post 到最后一个数据链尾 CQE，包含提交与 DMA 的重叠；
`done_us` 是最终 done WRITE 及其 CQ 等待。计时从 CPU 看到新请求后开始，不含空闲轮询。
`data_wrs`、`ready_wrs`、`post_calls`、`waves` 给出数据 WR、ready WR、provider 提交及 wave 数；
`plan_reused=1` 表示复用了预构建链；`cpu_threads=1` 表示由所属服务线程提交。
`tail_cqes` 统计数据链尾 CQE，不含 done。4096 × 656 B、4 worker、max_sge=13 时，
预期为 320 个数据 WR、16 个 ready WR、4 个数据尾 CQE，随后一个 done 尾 CQE。
`optimized=1` 表示 CPU transport 编译时已启用优化。

## 4096 × 656 B 性能复测

完成双端重建、设备包安装和 CPU 服务重启后，在 NPU 终端运行：

```bash
bash <<'BASH'
set -euo pipefail
mkdir -p results
KVAF_CPU_SERVICE_ADDR=127.0.0.1:21111 DEVICE_ID=2 \
bash tests/run_remote_cpu_batch_get_perf.sh build-npu \
  --backend cpu_push_scatter --aicpu-workers 4 --execution both \
  --request-length 4096 --table-capacity 4096 --miss-count 4096 \
  --token-bytes 656 --tokens-per-block 1 \
  --seed 20260903 --no-sort-input-indices \
  --source-access-pattern random --output-access-pattern random \
  --warmup 50 --iterations 1000 --kernel-profile \
  2>&1 | tee results/cpu-push-4096x656.log
BASH
```

固定通路、NUMA 策略和测试参数，分别比较 `scope=end_to_end` 与 `scope=kernel` 的
`p50_us`、`p99_us` 和 `p50_GBps`。4 worker 下每 worker 应有 1024 次复制。
另一次运行改为 `--output-access-pattern contiguous`，其余参数不变，可观察输出顺序的影响。
部分前缀使用较小的 `--miss-count`。

端到端计时包含每次 Python 调用、kernel launch 和所选完成等待；
旧常驻 kernel 的设备内循环带宽具有不同计时范围。

主要文件对应：`cpu_push_host.cc` 管 Host 资源，`cpu_push_get_worker.cc` 管 CPU 请求循环，
`cpu_push_get_transport.cc` 管 URMA 数据链，`cpu_push_scatter.cc` 管设备消费与本地复制。
