# AICPU BatchGet 的 CPU 回推路径

需要由 CPU 将离散 DDR 数据直接送入连续 HBM、再由 AICPU 写入调用方输出时，
在现有 `NpuRuntime` 构造参数中设置：

```python
runtime = kvaf.NpuRuntime(
    batch_get_kernel_object_path="",
    batch_get_token_bytes=656,
    batch_get_tokens_per_block=2,
    device_id=7,
    npu_core_count=4,
    aicpu_urma_kernel_json_path=kernel_json,
    aicpu_batch_get_mode="cpu_push_scatter",
    urma_device="udmac0d1e5",  # BatchPut 的 UMDK 端点，按部署环境调整。
    urma_eid_index=0,
)
runtime.link_to("127.0.0.1:28990")
runtime.initialize(ddr_bytes=656 * 2 * 8192, slot_size=656 * 2)
```

其余 `batch_put`、AccessTable 和 `batch_get` 调用保持现有方式。
`aicpu_batch_get_mode` 默认为 `gather_read`，两个值均可使用原有两种 Tensor 重载。
CPU push 自动发现单个 NPU EID，也可通过 `npu_eids=[eid]` 指定一个 EID。
自动发现的候选按现有 Runtime 初始化流程尝试；多 EID 条带和 AIV 自适应调优不用于此路径。
Get 使用 Host direct HCCP/RA；Put 继续使用既有 AICPU UMDK session。
CPU 使用四路 Jetty，由所属服务线程更新并依次提交预构建链；
`npu_core_count` / `--workers` 控制 AICPU worker 数。

## 构建并验证

两端代码、设备 SO 和 kernel JSON 一起更新，重启 CPU 服务及 NPU 进程。
当前 push 协议为 V3，复制和 ready 流水线基于
`codex/aicpu-ordered-write` 的 `a12bb16c120e885eb18ed61581739e1c212c32e8`（V24）。
设备入口为 `KvafAicpuCpuPushScatterA12V24`。主机库和设备 SO 必须一起更新，
包括移除废弃排序字段后的 profile 布局：

```bash
./build.sh cpu
./build.sh aicpu-urma --cpu-push --install-kernel
export PYTHONPATH="$PWD/build-aicpu-urma/python:$PYTHONPATH"
```

设备 `memcpy_sp` 需要 securec 头文件和 AArch64 动态库，构建会查找 CANN 内的依赖并打包。
设备 scatter 单独按参考提交使用 `-O3`；其余设备 UMDK/session 源码继续使用 `-O2`。
CPU 构建链接 OpenMP，并启用 `-O2`、保留调试信息，push 提交由一个服务线程执行。
`--cpu-push` 对应 CMake 的 `KVAF_ENABLE_AICPU_CPU_PUSH=ON`，默认关闭。
原有 gather-only 构建不查找或链接 securec/direct HCCP。
布局不同时设置 `KVAF_AICPU_SECUREC_INCLUDE_DIR` 和 `KVAF_AICPU_SECUREC_LIBRARY`。
CPU 服务启动方法见 [gather_read 使用说明](aicpu_gather_read.md)。
初始化日志必须包含：

```text
CPU_PUSH_REFERENCE commit=a12bb16c120e885eb18ed61581739e1c212c32e8 protocol=3 copy=per_row_memcpy_sp barrier=dsb_sy device_optimization=O3
```

```bash
python3 access_operator/demo/aicpu_batch_get/python/test_batch_get.py \
  --device-id 7 --peer 127.0.0.1:28990 \
  --urma-device udmac0d1e5 --eid-index 0 \
  --modes gather_read cpu_push_scatter --workers 1 2 4 8 \
  --blocks 4096 --token-bytes 656 --tokens-per-block 2 \
  --output-row-step 1 --warmup 10 --iterations 100
```

`--output-row-step 1` 检查相邻输出行；改为 2 检查输出间隔。
`--output-order random`（默认）随机排列目标行；`sequential` 按目标行顺序请求。
这个选项只改变测试输入，算子按调用方给定的顺序执行，不排序 Tensor。
脚本每轮重新写入源数据和输入元数据，比较整个输出，并用后继 NPU 算子消费输出。
预检包括部分/空 `miss_count`、两种 Tensor 重载和异步调用。
需要阶段耗时时增加 `--profile-rounds 100`；采样轮次不计入正常带宽。

## 请求与数据流

一次调用处理 `int32 miss_count` 指定的有效前缀，最多 65536 项。
每项读取 `addresses[indices[i]] + offsets[i]` 的 `token_bytes` 字节，
目标为 `output + output_indices[i] * token_bytes`。
调用方继续负责有效 key、token 布局、输出映射及这些资源在操作完成前的生命周期。

初始化时，Host 通过 RA 注册固定 HBM workspace，创建四个 TargetChannel，
将 segment、Jetty key 和 TP 信息发给源 DDR 所属 CPU Runtime。
CPU 建立专用回写队列，返回对端信息；Host 完成导入后才激活驻留 CPU worker。
激活时先同步 READ 请求控制字并 WRITE 初始完成字，确认实际双向 RMA 路径可用。
这套控制协商通过现有 PeerLink 完成，无需新增 NPU listener。

每轮过程如下：

1. Host 在调用方 stream 清零本次 AICPU coordination，再启动配置数量的实例。
2. worker 0 按原始请求顺序生成源地址清单，在 HBM 发布请求头和 request sequence。
   不生成设备排序数组，不改写调用方 Tensor。
3. notify 发布 sequence 和可选 profile 位；CPU READ 请求头，验证后 READ 有效源地址前缀，
   由所属线程更新预构建 SGL 的源地址。
4. CPU 直接从已注册 DDR 源 WRITE 到 packed HBM staging，不打包 payload 到 CPU 临时区。
5. 按 worker 的 staging 区间划分 1024 行 chunk，各 Jetty 在其数据之后强序 WRITE ready。
6. AICPU 取四个 Jetty 的共同就绪前缀，合并连续就绪的 chunk，逐行执行 `memcpy_sp`，
   目标为原始 `output_indices[i]`。ready 读取前、复制前及 worker 复制完成后执行 `dsb sy`。
7. CPU 收齐数据链尾 CQE 后发布 done；全部 scatter 与 done 均完成后 kernel 结束。

ready 用高 32 位 sequence、低 32 位 chunk 数编码；done 在同一个 64 位字内编码 sequence 和结果。
请求序号从 Host 递增，达到 uint32 上限后要求重建 session。
空前缀不通知 CPU、不修改输出；CPU 允许此类调用造成的序号间隔。
当前实现针对普通 stream enqueue，Host 固定序号不能用于 graph replay。

CPU 数据面沿用该提交 V24 的 prepared WRITE、indexed CQ 和轮转 CQ 扫描，
CPU 与设备共享 worker 分区和 chunk 枚举逻辑。
每个 chunk 按连续行分到四个 Jetty；请求数量不变时复用数据/ready 链，每轮只更新源地址和序号。
所属线程更新并依次提交四路队列，收齐数据链尾 CQE 后提交预构建的 done 链，
消除每轮 OpenMP team 唤醒和多次 barrier；多 wave 请求仍逐 wave 排空再复用 SQ。
每个 Jetty 的 SQ 深度为 8192，doorbell 为 32 WR，CQ batch 为 8，每 32 次扫描检查超时。
超出单 wave 容量的请求分 wave 执行，完成前保留全部 WR/SGL；每个 ready 使用独立的注册源字。
Runtime 按每次 BatchGet 启动 kernel，接收本轮动态源地址和 `output_indices`。
ready 热循环与 V24 一样只读 worker 协调状态；Runtime 的远端错误完成字在阻塞时每
32 次轮询检查一次，在最终 done 等待中逐次检查。

## 与指定提交的对应关系

生产代码仅保留参考提交的缓存行分区、1024 行 chunk、共同 ready 前缀和逐行复制算法。
`cpu_push_get_layout.h` 提供 CPU/设备共享的分区，`cpu_push_pipeline.h` 实现消费与复制。
不再携带独立测试的 ABI、结果统计、校验协议或结果转换适配层。

以下接口和工作量差异保留，用于支持 Runtime 的动态请求，不能由参考代码直接覆盖：

| 项目 | a12bb16 V24 独立测试 | 当前 Runtime |
| --- | --- | --- |
| 执行 | 一次 launch，设备内循环所有轮次 | 每次 Python BatchGet 独立 launch、同步并检查状态 |
| 源 SGL | CPU 预先构建，逐轮修改 ready 序号和源数据标签 | 每轮从动态 Tensor 解析源地址，CPU READ 地址清单并更新 SGL |
| 输出 | 预先安排顺序输出，656 B 行使用 768 B stride | 按动态 `output_indices` 写入 656 B slot，默认测试随机排列 |
| 校验 | 源标签每轮变化，末轮完整读回输出 | 每轮变化输入、校验整个输出并由后继 NPU 算子消费 |
| 带宽计时 | 驻留 kernel 内 request 到 scatter/done | Python 调用开始到本次 stream 完成 |

`--output-order sequential` 可单独测量目标访问顺序的影响；它不改变 slot 大小或带宽计时范围，
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

一次执行完成前不复用 staging。异步返回表示已经入队；下次 Get/Put 会先 Drain，
检查前一轮执行状态。`synchronize=True` 在本次调用内完成等待和状态检查。
Shutdown 先停止 CPU worker 并销毁其 DMA 队列，再等待设备执行、释放 HBM。
传输或 stream 失败会阻止执行资源复用；关闭失败时保留资源以便重试。

## Profile

`request_us` 包含动态源地址清单构建，`scatter_us` 记录本地复制。
`scatter_copies` 为实际逐行 memcpy 次数。4096 × 656 B、完整前缀、4 worker 时，
每个 worker 为 1024 次，总计 4096 次。
push 路径的 `read_us` 是消费 ready 并执行 scatter 的区间，`ack_us` 是最终 done 等待；
`source_write_us` 为零，因为本轮源清单由 CPU READ 获取。
缓存维护相关字段为零。设备 worker 的区间重叠，不能相加作为端到端时长。

`--profile-rounds` 同时开启 CPU 分段计时。CPU 服务在每个采样请求完成后输出一行
`CPU_PUSH_PROFILE`，包含 profile 预热轮；普通带宽轮次不采集或打印这些计时。
`header_read_us` 是请求头 READ 耗时，`address_read_us` 是有效源地址数组 READ 耗时；
`write_us` 包含 `build_us`、`post_us`
和 `cq_wait_us`，分别表示计划构建/源地址更新、四路顺序提交、提交后的 CQ 等待。
`submit_to_cqe_us` 从首次提交阶段到最后一个数据链尾 CQE，包含提交期间与 DMA 的重叠；
`done_us` 是最终 done WRITE 及其 CQ 等待。计时从 CPU 看到新请求后开始，不含空闲轮询。
`data_wrs`、`ready_wrs`、`post_calls`、`waves` 给出数据 WR、ready WR、provider 提交及 wave 数；
`plan_reused=1` 表示复用了预构建链；`cpu_threads=1` 表示由所属服务线程提交。
`tail_cqes` 统计数据链尾 CQE，不含 done。4096 × 656 B、4 worker、max_sge=13 时，
预期为 320 个数据 WR、16 个 ready WR、4 个数据尾 CQE，随后一个 done 尾 CQE。
`optimized=1` 表示 CPU transport 编译时已启用优化。

## 4096 × 656 B 性能复测

完成上述双端重建、设备包安装和 CPU 服务重启后，在 NPU 终端运行：

```bash
set -o pipefail
mkdir -p results
python3 -u access_operator/demo/aicpu_batch_get/python/test_batch_get.py \
  --device-id 7 --peer 127.0.0.1:28990 \
  --modes cpu_push_scatter --workers 4 \
  --blocks 4096 --token-bytes 656 --tokens-per-block 1 \
  --active-percent 100 --output-row-step 1 --output-order random \
  --warmup 50 --iterations 1000 --profile-rounds 100 \
  2>&1 | tee results/cpu-push-a12bb16-4096x656.log
```

保留相同通路、NUMA 策略和测试参数，比较正常轮次的 `GBps`、`mean_us`、`p99_us`。
profile 中同时检查 `request_us`、`ready_wait_us`、`scatter_us` 和 `scatter_copies`：
该配置每 worker 为 1024 次复制；CPU 应为 `cpu_threads=1`，
预热后 `plan_reused=1`。保留 NPU 日志及 CPU 端的 `CPU_PUSH_PROFILE`。
在单独一次运行中改用 `--output-order sequential`，其余参数不变，可观察输出顺序的影响。
将 `--output-row-step` 改为 2 或 `--active-percent` 改为 75，继续验证间隔和部分前缀。

端到端仍然计入每次 Python 调用、kernel launch 和状态同步。
旧常驻 kernel 的设备内循环带宽和当前 Python 端到端带宽具有不同计时范围。

不依赖 NPU 的回归测试可单独构建运行：

```bash
cmake -S . -B build-scatter-tests -DBUILD_TESTING=ON
cmake --build build-scatter-tests --target kvaf_cpu_push_scatter_test
ctest --test-dir build-scatter-tests -R '^kvaf_cpu_push_scatter_test$' --output-on-failure
```

回归测试覆盖 ready 延迟、跳跃、回退、
未来序号、超时和复制失败；并执行生产 kernel 的本地线程模型，校验动态源地址、
输出索引、空/部分前缀、非整除 worker 分区、远端错误及完整输出 guard。

主要文件对应：`cpu_push_host.cc` 管 Host 资源，`cpu_push_get_worker.cc` 管 CPU 请求循环，
`cpu_push_get_transport.cc` 管 URMA 数据链，`cpu_push_scatter.cc` 管设备消费与本地复制。
