# AIV BatchGet 第一阶段实现与验证

## load 故障的独立诊断入口

`tests/python/diagnose_aiv_scatter.py` 默认实际执行 load，不会替换为 done。它为每个
检查点启动独立 Python 进程，合并保存 stdout/stderr，在首个非零退出码处停止，
不会在设备异常后继续启动下一个检查点。默认不执行 MTE3 的 copy 或完整回归。

先同时重编 Host 与 kernel（此次所有 scatter 探针增加设备返回的诊断版本 1，
旧 kernel 不会被新 Host 误认为通过）：

```bash
cmake --build build-npu -j
export PYTHONPATH="$PWD/build-npu/python"
# 沿用 KVAF_CPU_SERVICE_ADDR、KVAF_BATCH_PUT_KERNEL 和有效的 NPU EID。
KVAF_RUN_NPU_TESTS=1 DEVICE_ID=7 KVAF_NPU_CORE_COUNT=1 KVAF_NPU_TOKEN_BYTES=656 \
  KVAF_BATCH_GET_KERNEL="$PWD/build-npu/bin/kvaf_scatter_batch_get_kernels.o" \
  python -u tests/python/diagnose_aiv_scatter.py

# 显式选择一项对照检查点；仍会使用独立进程。
KVAF_RUN_NPU_TESTS=1 DEVICE_ID=7 \
  python -u tests/python/diagnose_aiv_scatter.py --stages dma
```

默认只运行 load。需要缩小范围时，显式传入 `--stages setup gm body dma ub load`。
各项含义如下，所有检查点都保留真实 CPU descriptor/payload/done 协议，
只有正常同步并核对每核状态之后才打印完成标记：

| 检查点 | 新执行范围 | 成功证据 |
| --- | --- | --- |
| done | AIV 读到本轮 done | sequence/result/phase 匹配 |
| flush | 输出行寻址、静态 UB 构造与 staging flush | output 保持 guard |
| setup | MTE2 配置与 S→MTE2 同步 | configured_bytes |
| gm | 标量逐字节读取 staging，不执行 payload DMA/UB 读取 | scalar_bytes、每核 checksum |
| body | 仅整块 DataCopy 及每次 MTE2→S 同步，不执行尾部 DMA/UB 读取 | body_bytes |
| dma | 主体和尾部 CopyIn 及 MTE2→S 同步，不读取 UB | dma_bytes |
| ub | 完整 CopyIn 后逐字节读取 UB，不做 GM 参考比较 | scalar_bytes、每核 checksum |
| load | 完整 CopyIn 后比较 UB/staging 的有效字节 | verification=gm_ub_byte_compare、verified_bytes |

656 B 对应 640 B 主体和 16 B 尾部。body 成功、dma 失败时，优先核对尾部
DataCopyPad、同步与该路径收尾；dma 成功、ub 失败时，优先核对 UB 标量读取；
gm/ub 都成功但 load 失败时，核对组合比较路径和实际异常指令。检查点会改变
执行范围及编译结果，成败不能直接证明某一条指令是根因。所有路径仍执行状态
写回、flush 和退出屏障。checksum 只用于保留读取、观察内容，不代表业务数据验收。
小于 32 B 的 token 无主体，显式选择 body 会被拒绝，应直接用 dma 检查尾部。

默认日志目录为 `scatter-diagnostics-<时间>/`，可用 `--log-dir` 指定新目录。
`manifest.txt` 保存实际脚本、Get/Put kernel 的 SHA-256 和 checkout commit，
每个 stage 的日志保留子进程 PID。`SCATTER_KERNEL_READY` 另打印实际加载字节
的 FNV-1a 指纹和 Host 诊断版本；`SCATTER_LAUNCH` 打印 HBM workspace、
descriptor、staging、done、output 和 status 地址。Host 的 BEGIN/LAUNCH
日志只证明准备/提交，不能证明设备到达相应位置。

正常同步后 `SCATTER_STATUS` 包含设备诊断版本、每核 result/phase/requests
以及 detail0/detail1。gm/ub 中它们分别是 checksum 和读取字节数；内容比较
失败时错误会打印 `local_request`、token 内的 `byte_offset`、`gm_byte`、
`ub_byte`。旧 Host 缺少完成版本时，新诊断脚本拒绝通过；旧 kernel 返回版本
不匹配时，新 Host 拒绝通过。参数仍为 176 B，状态仍为 64 B，CPU 协议不变。

若仍有 507015，不在失败 stream 上尝试额外回读 checkpoint。保留首个失败的
stage 日志、manifest、对应 `.o`，以及相同 PID/Device/时刻的 CPU 和 Ascend
设备异常摘要（fault kernel、errcode、fixedPCOffset、MTE_ERROR、SU_ERROR）。
确认故障进程退出和设备可用后再复测；旧 PC 偏移不能映射到重编后的产物。

## 独立 scatter 测试入口

`test_aiv_scatter_batch_get.py` 已删除临时强制 done 的覆盖，保留调用方指定的
`KVAF_AIV_SCATTER_PROBE`。设置 load 会执行真正的 GM→UB、同步和内容比较；
未设置 probe 时执行完整 scatter 回归。只有显式选择 done 才会提前返回。

```bash
git pull --ff-only
KVAF_AIV_SCATTER_PROBE=load DEVICE_ID=7 KVAF_NPU_CORE_COUNT=1 \
  KVAF_BATCH_GET_KERNEL="$PWD/build-npu/bin/kvaf_scatter_batch_get_kernels.o" \
  python tests/python/test_aiv_scatter_batch_get.py
```

沿用已有 CPU 服务、Python、BatchPut kernel 和 `KVAF_RUN_NPU_TESTS=1` 环境。
本次诊断版本更新需要同时重编 Host/kernel。
默认 1 核、656 B 时连续运行 `[1, 3, 1]` 三批，检查序号/工作区复用和
output 全部保持 `0xAD`；只有正常 shutdown 后才打印：

```text
PROBE_COMPLETE: stage=load ... output=unchanged mte2=byte_verified_against_staging (not a full regression PASS)
```

load 验收 staging→UB 内容一致性，不代表完整 scatter 输出正确。
诊断状态不会被普通 BatchGet 调用当成成功，gather 独立入口不受影响。

## 范围

在 `codex/aiv-urma-scatter-gather` 实现无流水顺序路径，默认仍为 `direct_read`。
没有接入第二阶段的 chunk 消费、双缓冲或第三阶段的软件在途预算。
总体设计见 [三阶段设计](aiv_batch_get_scatter_gather_design.md)。

| 模式 | 顺序依赖 |
| --- | --- |
| `cpu_push_scatter` | AIV 生成源地址清单并发布 HBM notify → CPU READ 清单 → 四路 Jetty WRITE 整批 payload → done → AIV 单 UB 缓冲 scatter → 本地写出完成 |
| `gather_read` | AIV WRITE 清单 → WRITE command → WRITE notify → CPU 完成对应 lane 的 gather → 最终 ACK → AIV 多 SGE READ 到 output → CQ 完成 |

Scatter 只消费本轮整批 done，忽略参考分支 CPU transport 仍会发布的 chunk ready。
Gather 的 `pipeline_chunk_bytes=0`，不把 CPU 的 `~sequence` started ACK 当成完成。
多核 gather 每个核拥有独立 lane，各 lane 按上述依赖顺序执行。
AIV 不调用 UMDK；scatter 使用注册 HBM mailbox，gather 使用手写 USER_CTL WQE。

Scatter 的本地 payload 搬运通过一个 16 KiB UB 缓冲顺序执行 MTE2（GM→UB）
和 MTE3（UB→GM）。MTE2 参考 `ops-transformer v9.1.0`：32 B 整数倍主体
使用 `DataCopy<uint8_t>`，不足 32 B 的尾部使用 `DataCopyPad<uint8_t>`；
两次搬运各自等待 MTE2→S 完成。MTE3 仍使用原有精确长度的 DataCopyPad，
完成后才复用缓冲并报告完成，UB padding 不写到 output。
UB 通过公开的 `LocalTensor<uint8_t>(TPosition::VECIN, 0, 16384)` 构造，
每次读入前单独调用 `ResetLoopModePara(DataCopyMVType::OUT_TO_UB)`，
将 GM→UB 恢复为普通搬运模式。这项配置不调用 `InitSocState()` 或 TPipe。
Scatter 的 Host loader 按纯 SIMD 启动，`dynUBufSize=0`；16 KiB 静态 UB
缓冲不通过 SIMT 的动态 UB 启动参数分配。
源 GlobalTensor 禁用 L2 cache hint，且不再经 SIMT 读取被 CPU WRITE 更新的
staging。原实现的 MainScalar cache 刷新不能代替 SIMT cache 刷新。
Scatter kernel 已按硬件故障定位需要撤掉入口的 `InitSocState()`，保留退出前的
`PipeBarrier<PIPE_ALL>()` 与原有搬运事件同步；此次只回退入口初始化，
尚未确认原有 `load` 阶段的 MTE 异常是否消除。
接口约束见 [GMToUB DataCopyPad](https://asc.gitcode.com/api/SIMD-API/basic_api/memory_vector_compute/data_move/DataCopyPad_GMToUB.html)、
[UBToGM DataCopyPad](https://asc.gitcode.com/api/SIMD-API/basic_api/memory_vector_compute/data_move/DataCopyPad_UBToGM.html) 和
[SIMD/SIMT 内存层次](https://asc.gitcode.com/guide/programming_guide/advanced_programming/advanced_ai_core_programming_model/simd_simt_hybrid_programming/memory_hierarchy.html)。

Gather 的控制与数据队列分开，每核固定两条 Jetty。可配置 1～6 个本地 SGE，
每条 READ 的总长度最多 64 KiB，按 64 B WQEBB 处理变长 WQE 和 SQ 回绕。
Gather 只使用标量 GM 访问、缓存操作与 USER_CTL 队列，不使用 LocalTensor
或 MTE payload 拷贝；其入口不调用 `InitSocState()`，退出不附加
`PipeBarrier<PIPE_ALL>()`，恢复此前通过硬件测试时的执行方式。
阶段内部批量提交，只在物理 SQ/WR 容量不足或阶段结束时 drain；
`batch_get_max_inflight_total` 仅对原 `direct_read` 生效。

## 代码入口与模式隔离

Python 继续使用统一的 `kvaf.NpuRuntime(..., aiv_batch_get_mode=...)` 和
`batch_get()`；每个 Runtime 在初始化时选择一个 backend。CPU 服务与 wire
协议保持不变，仍分别使用 `cpu_push::Worker` 和 `CpuGatherWorker`。

| 层次 | Scatter | Gather |
| --- | --- | --- |
| Host owner | `host/scatter_batch_get.cpp` | `host/gather_batch_get.cpp` |
| Loader | `host/execute_scatter_batch_get_loader.cpp` | `host/execute_gather_batch_get_loader.cpp` |
| 参数 ABI | `include/scatter_batch_get_abi.h`，176 B | `include/gather_batch_get_abi.h`，216 B |
| AIV 源码 | `kernel/execute_scatter_batch_get_aiv.asc` | `kernel/execute_gather_batch_get_aiv.asc` |
| Kernel symbol | `KvafScatterBatchGetAivKernelV1` | `KvafGatherBatchGetAivKernelV1` |
| 二进制 | `kvaf_scatter_batch_get_kernels.o` | `kvaf_gather_batch_get_kernels.o` |
| 独立测试 | `tests/python/test_aiv_scatter_batch_get.py` | `tests/python/test_aiv_gather_batch_get.py` |

表中 Host/include/kernel 路径均相对于 `access_operator/demo/aiv_batch_get/`。
`host/sequential_batch_get.cpp` 只按模式创建 backend，公共 interface 只约定
Initialize/Execute/Drain/Shutdown。两种 owner 分别管理自己的 workspace、
序号、资源回收和错误状态；两个 loader 分别保存自己的 binary/function handle。
Host 代码仍链接到原有 `libkvaf_execute_batch_get_host.so`，上层链接配置不变。

只复用稳定的基础部分：`sequential_batch_get_common.h` 的输入检查，
`batch_get_kernel_common.h` 的地址/行校验与缓存工具，及通用请求/状态结构。
两套 launch 参数均不含 mode 分支；gather 参数不含 scatter probe，scatter
参数不含 gather 的 lane/远端 staging/SGL 配置。gather 忽略 scatter 探针环境变量，
scatter 不校验 gather 专用 SGL 参数。

结构拆分后，gather 入口另行移除了从 scatter 带入的静态 Tensor 初始化和
退出屏障（见下文空批次回归记录）。scatter 507015 尚未解决，
也没有加入流水或软件在途控制。共享基础头文件的修改仍需两种
模式回归；修改某一种模式的独立 `.asc` 不会导致另一种 kernel 重编译。

原 `direct_read` 仍使用 `kvaf_execute_batch_get_kernels.o`。三个模块各自编译和
链接，均安装到 `build-npu/bin/`，不合并包含 SDK 全局符号的临时 object。
默认构建包含三个模块，也可只构建单个 kernel：

```bash
cmake --build build-npu --target kvaf_gather_batch_get_kernel -j
cmake --build build-npu --target kvaf_scatter_batch_get_kernel -j
```

传入标准 `kvaf_execute_batch_get_kernels.o` 或旧
`kvaf_sequential_batch_get_kernels.o` 路径时，Host 按模式选择同目录的新模块，
不加载旧的合并二进制；这两个路径在顺序模式下只是选择器，文件本身无需存在。
也可直接传入对应模式的新 `.o` 或包含对应新 symbol 的自定义文件。缺失目标
模块或 symbol 会报错，不回退到另一模式。旧合并模块不再构建。

升级时必须同时更新 NPU Host 库和对应的新 kernel，避免混用参数 ABI。
已经部署第一阶段协议的 CPU 服务不需要因这次拆分更新。首次部署第一阶段时，
两端仍须具备对应的 enable/open/start/stop 控制命令。

## 使用约束

```python
runtime = kvaf.NpuRuntime(
    get_kernel_path,
    batch_get_token_bytes=656,
    batch_get_tokens_per_block=128,
    batch_put_kernel_object_path=put_kernel_path,
    npu_core_count=1,                 # 首次上机先用单核，再验证 3/8 核
    aiv_batch_get_mode="gather_read", # 或 "cpu_push_scatter"
    aiv_gather_sgl_size=6,            # Gather: 1..6
)
runtime.link_to(cpu_service_address)
runtime.initialize(ddr_bytes, slot_size)
# 原有 DeviceAccessTable / addresses+valid 两种 batch_get 重载均可使用。
```

- 支持 1～8 核、单 EID、eager；首版拒绝 graph capture 和启动带宽调优。
- 请求容量为 1～65536，`miss_count=0` 是合法空批次。仅有效前缀参与搬运，
  支持重复源、随机且不重复的输出行。输出行重复不受支持。
- 固定 token 长度，当前初始化上限为 65535 B，保证 `65536 * token_bytes`
  可由共享协议的 uint32 表示。offset 为非负字节偏移，不能越过源 block。
- 每个 Runtime 复用一套 workspace，下一次 Get/Put 先 drain 上一批。
  `synchronize=False` 仅代表入队成功，设备错误在下一次 Get/Put 或 Shutdown
  回收状态时返回；调用方仍须保持 Tensor 和 stream 有效直到执行完成。
- 错误输出可能部分写入，不应消费。执行失败后拒绝继续提交，需 shutdown
  并重建 Runtime。队列或 CPU DMA 无法确认停止时保留注册资源，允许重试关闭。
- 轮询有次数上限，loader 设置 180 秒 kernel 超时；轮询次数不是毫秒超时，
  不构成性能或端到端超时承诺。
- Scatter HBM 约为 `65536 * token_bytes + 512 KiB + 控制区`；Gather 使用
  同量级 CPU DDR staging，另有每 lane 描述符、控制区及小型 HBM workspace。

## 构建与服务器回归

CPU 端使用匹配部署环境的 UMDK：

```bash
cmake -S . -B build-cpu -DKVAF_BUILD_REMOTE_CPU_SERVICE=ON -DKVAF_URMA_ROOT=/path/to/umdk
cmake --build build-cpu -j
./build-cpu/runtime/kvaf_cpu_service --listen 0.0.0.0 --port 28999 --gather-threads 8
```

NPU 端使用支持 A5 `dav-c310-vec` 的 CANN 环境，沿用项目原有环境初始化：

```bash
cmake -S . -B build-npu -DKVAF_ENABLE_AIV_BATCH_GET=ON -DKVAF_ENABLE_AIV_BATCH_PUT=ON \
  -DKVAF_BUILD_PYTHON=ON -DASCEND_CANN_PACKAGE_PATH=/path/to/cann
cmake --build build-npu -j
export PYTHONPATH="$PWD/build-npu/python"
export KVAF_RUN_NPU_TESTS=1
export KVAF_CPU_SERVICE_ADDR=CPU_IP:28999
export KVAF_BATCH_GET_KERNEL="$PWD/build-npu/bin/kvaf_execute_batch_get_kernels.o"
export KVAF_BATCH_PUT_KERNEL="$PWD/build-npu/bin/kvaf_execute_batch_put_kernels.o"
ls -l "$KVAF_BATCH_GET_KERNEL" "$KVAF_BATCH_PUT_KERNEL" \
  "$PWD/build-npu/bin/kvaf_scatter_batch_get_kernels.o" \
  "$PWD/build-npu/bin/kvaf_gather_batch_get_kernels.o"
# 两个入口固定测试自己的模式，不受 KVAF_AIV_BATCH_GET_MODES 影响。
DEVICE_ID=7 KVAF_NPU_CORE_COUNT=1 python tests/python/test_aiv_gather_batch_get.py
DEVICE_ID=7 KVAF_NPU_CORE_COUNT=3 python tests/python/test_aiv_gather_batch_get.py
DEVICE_ID=7 KVAF_NPU_CORE_COUNT=8 python tests/python/test_aiv_gather_batch_get.py
DEVICE_ID=7 KVAF_NPU_CORE_COUNT=1 python tests/python/test_aiv_scatter_batch_get.py
```

两个入口复用 `aiv_batch_get_test_common.py` 中的数据准备和校验，使用 8193
请求容量，覆盖空批次、尾批、重复调用、
随机输出、重复源、guard、A/B/A 输出、两种 Tensor 重载以及跨 stream 顺序复用。
可用 `KVAF_NPU_TOKEN_BYTES=13`、`KVAF_NPU_REQUESTS=65536`、
`KVAF_AIV_GATHER_SGL_SIZE=1` 增加边界组合。
CPU `--gather-threads` 至少等于 gather 的 AIV 核数。
上述命令在仓库根目录执行。单模式测试只检查对应 Get 二进制和 BatchPut，
不要求另一模式的二进制存在；路径检查失败会在初始化 NPU 前报具体路径。
兼容入口 `test_aiv_sequential_batch_get.py` 仍支持 `KVAF_AIV_BATCH_GET_MODES`，
但现在为每种模式启动独立 Python 子进程，即使第一种失败也执行后一种，最后
汇总非零退出码。这隔离了 Python/ACL 进程状态，不承诺设备故障后自动恢复硬件。
CTest 也分别注册两个硬件用例。

## 初始化失败的定位

`runtime.initialize()` 会先建立传输，再运行原有 BatchPut kernel 做连通性探针。
本测试默认 16 个 block，探针为 16 行、每行 64 B（合计 1024 B），
此时还没有运行 scatter/gather BatchGet kernel。

Scatter 的四次 `JETTY_IMPORT_READY` 只表示接口导入成功；随后
`CPU_PUSH_START` 还会 READ HBM notify、WRITE 初始完成状态，检验实际 RMA 路径。
协议 `result=3` 是 `kResourceError`，需要结合命令名和 CPU 日志确定原因。
CPU 端现在记录 `CPU_PUSH_START_FAILED session=... reason=...`，区分
`startup notify READ` 与 `startup completion WRITE`；停止失败则记录
`CPU_PUSH_STOP_FAILED`。出现这种错误时需要同时保留两端日志。

NPU 端可单独运行 scatter 并开启阶段日志：

```bash
DEVICE_ID=5 KVAF_NPU_CORE_COUNT=1 KVAF_AIV_BATCH_GET_MODES=cpu_push_scatter \
  KVAF_AIV_VERBOSE_DIRECT_HCCP=1 \
  python -u tests/python/test_aiv_scatter_batch_get.py > npu-stage1.log 2>&1
```

观察 `STARTUP_CANDIDATE_BEGIN`、`STARTUP_PROBE_BEGIN/READY`、
`DIRECT_BATCH_PUT_STAGE`、`DIRECT_BATCH_PUT_RUNTIME_ERROR` 和
`STARTUP_CANDIDATE_FAILED`。`phase=synchronize` 后返回 ACL 错误说明等待设备
执行失败，仍需对应进程的 Ascend 日志定位具体 kernel/指令；不能仅凭
`Vector core execution timed out` 判定为 notify 或 CQ 的问题。
BatchPut 路径上的本地 ACL 失败会停止候选 EID 重试；正常返回的 CQ 错误仍允许
连接回退。清理同时失败时，异常保留原始原因并附加清理错误。

若 `STARTUP_PROBE_READY` 和实际 `batch_put` 均已完成，而首次非空 scatter
失败，应查看 CPU 的 `CPU_PUSH_WORKER_FAILED`。其中 `phase` 区分 notify READ、
descriptor READ、payload WRITE 和 completion WRITE；`last_sequence=0` 表示
worker 尚未接受非空请求。控制 CQ 错误包含数值 `status`、`expected_ctx`、
`actual_ctx`、队列/Jetty/CQ/TPN、WR 操作和访问地址，不能只凭旧版
`CQ status/context mismatch` 判断是链路失败还是完成记录错配。
NPU drain 异常包含 `sequence`、`phase=copy_status/synchronize` 和原始 ACL
错误码。测试脚本会另外打印清理失败，但保留原始执行异常作为最终异常。

`phase=read_notify last_sequence=N`（N 非零）意味着 worker 已完成第 N 批
payload 和 done 的 CQ 回收，回到下一轮 notify READ；不要把这个后续 READ
错误直接解释为第 N 批没有收到请求或没有写完 payload。`actual_ctx` 与
`expected_ctx` 相同但 `status` 非零时，是 CQ 状态失败而非 context 错配。
当前 UMDK 头文件中 status 9 为 `URMA_CR_ACK_TIMEOUT_ERR`，部署端仍应以
匹配的头文件/库为准。`acl=507015` 表示 AI Core 执行异常，具体指令和先后
因果需要结合对应进程的 Ascend 设备异常日志确认。

卡 7、PID 2365 的异常日志进一步报告 `KvafSequentialBatchGetAivKernelV1`、
`errcode=259`（标量指令异常）、`SU_ERROR_T0_0=0x8`，修正后的偏移为
`KvafSequentialBatchGetAivKernelV1+0xa18`；该偏移只适用于此次故障的旧二进制。
需要在重编译前保留旧 `.o` 并用匹配 CANN 的反汇编工具查看该偏移，才能判断
具体哪条指令违反约束。DataCopyPad 改动修正了 payload 缓存可见性并替换裸搬运
路径，但在硬件复测/指令核对前，不将它视为已证实的 errcode 259 根因修复。

## 独立 Gather 首个空批次的回归

拆分后提交 `32a52c4` 的硬件反馈：Device 7、PID 12157，BatchPut startup
probe、写入和 access table 准备已完成，但 `gather_read` 的第一次
`epoch=0 active=0` 在同步时返回 507015。该空批次不进入 `Gather()`，
不会提交 descriptor、command、notify 或数据 READ；应先核对入口、输入缓存
处理、状态写回和退出路径，不能据此归因于 CPU gather 或网络传输。

之前通过 1、3、8 核测试的 gather 路径没有 `InitSocState()` 和
`PipeBarrier<PIPE_ALL>()`。它们在 `bc5aad5` 为 scatter 的静态 LocalTensor
搬运加入共用入口，随后被保留在拆分后的两个 kernel 中。`d949503` 只从 gather
移除这两项，保留空批次真实 launch、输入/状态缓存处理和原有队列逻辑。
该提交未改变 scatter 的代码与诊断路径。

模型增加 1、3、8 核及 SGL=1/6 的首批为空、随后非空序列，检查空批次状态
完成、output guard 和 SQ/CQ 不推进，后续请求仍能得到正确数据。
模型不能复现 CANN 生成的设备指令；当前日志也没有对应的 errcode、
fixedPCOffset 或 fault kernel name。用户随后确认上述修改后的独立 gather
测试成功（所给命令为 1 核）；入口回退已有硬件反馈，但不能单独认定是哪项
操作或哪条指令导致了异常。

更新后只需重编译 gather kernel，再运行独立测试（沿用现有 peer 配置）：

```bash
git pull --ff-only
cmake --build build-npu --target kvaf_gather_batch_get_kernel -j
KVAF_RUN_NPU_TESTS=1 DEVICE_ID=7 KVAF_NPU_CORE_COUNT=1 \
  KVAF_BATCH_GET_KERNEL="$PWD/build-npu/bin/kvaf_gather_batch_get_kernels.o" \
  python tests/python/test_aiv_gather_batch_get.py
```

## Scatter 入口初始化回退

在 gather 回退通过后，按用户要求单独撤掉 scatter 入口的 `InitSocState()`。
保留 `PipeBarrier<PIPE_ALL>()`、S→MTE2、MTE2→MTE3、MTE2→S 和 MTE3→S
事件同步，DataCopyPad 参数、UB 配置、notify/done 协议和四个探针均保持不变。
此次没有改用 TPipe，也没有增加开关或流水。

此前模型用 `dma_state_dirty` 布尔量强制要求先调用初始化再搬运，它并不模拟
实际寄存器或设备指令。此次同时撤掉这项人为门禁，模型继续检查拷贝参数、
UB 范围、异步搬运依赖、缓冲复用和输出内容；模型通过不代表硬件初始化状态
已验证。原来的 MTE 异常是否仍然存在，需要用新二进制复测。

回退初始化时只需重编译 scatter kernel。当前 `load` 已增加 MTE2 内容校验，
需要同时重编译 Host 和 kernel，沿用现有 CPU 服务和 peer 配置：

```bash
git pull --ff-only
cmake --build build-npu -j
KVAF_RUN_NPU_TESTS=1 DEVICE_ID=7 KVAF_NPU_CORE_COUNT=1 \
  KVAF_AIV_SCATTER_PROBE=load \
  KVAF_BATCH_GET_KERNEL="$PWD/build-npu/bin/kvaf_scatter_batch_get_kernels.o" \
  python -u tests/python/test_aiv_scatter_batch_get.py
```

当前先验收 `load`；MTE3 的 `copy` 和完整 scatter 回归留待后续。

## MTE2 独立内容校验

本次先落实 GM→UB，`load` 不发出 MTE3 payload 拷贝，也不执行 MTE2→MTE3
事件握手。其路径为：CPU 发布整批 done → 刷新 staging → 配置 MTE2 普通模式
→ S→MTE2 → `DataCopyPad` → MTE2→S → 标量逐字节比较 UB 与 staging。
每一块最多 16 KiB，只比较本次有效的 n 字节，比较完才复用 UB；output 始终
保留 guard。保留 kernel 退出屏障，不引入流水或在途控制。

参考源码固定到官方 `ops-transformer v9.1.0`：

- [arch35 LocalWindowCopy](https://gitcode.com/cann/ops-transformer/blob/v9.1.0/mc2/moe_distribute_dispatch_teardown/op_kernel/arch35/moe_distribute_dispatch_teardown_arch35.h)：
  使用字节长度的 DataCopyPad 参数、关闭源 L2 cache hint，并在标量读取 UB 前
  同步 MTE2→S。此处只参考读入与同步，不复制其量化和写出逻辑。
- [GM→UB→GM helper](https://gitcode.com/cann/ops-transformer/blob/v9.1.0/mc2/common/op_kernel/gm_ub_gm_copy.h)：
  参考缓冲生命周期；不照搬其向上取整的 GM 尾块读入长度。

同时核对了官方 `asc-devkit v9.1.0`（`e5451c7`）的实际实现：

- [LocalTensor 构造和 GetValue](https://gitcode.com/cann/asc-devkit/blob/v9.1.0/impl/basic_api/kernel_tensor_impl.h)：
  构造函数设置 UB 偏移、位置及长度，不调用全局初始化；`GetValue` 读取 UB。
- [3510 TPipe 构造](https://gitcode.com/cann/asc-devkit/blob/v9.1.0/impl/basic_api/dav_3510/kernel_tpipe_impl_c310.h)：
  会调用 `InitSocState()`，因此此次不引入 TPipe/TQue。
- [3510 搬运实现](https://gitcode.com/cann/asc-devkit/blob/v9.1.0/impl/basic_api/dav_3510/kernel_operator_data_copy_impl.h)：
  `ResetOutLoopModeParaImpl` 仅将 `LOOP_SIZE_OUTTOUB` 写为普通模式值
  `(1UL << 21) | 1UL`。通过公开的 `ResetLoopModePara(OUT_TO_UB)` 调用，
  不恢复 CTRL、Mask、atomic 或 MTE3 配置。模型不模拟这个寄存器的硬件效果。

协议布局不变，但 `load` 完成 phase 改为 11，表示字节校验完成；旧 phase=9
仅代表执行到达，新的 Host 会拒绝。出现内容不一致时返回 result=7，错误含
`detail=MTE2_GM_UB_byte_mismatch`，不报告 probe 完成。全部通过时输出：

```text
[KVAF_AIV_SCATTER_PROBE_COMPLETE] stage=load ... verification=gm_ub_byte_compare verified_bytes=...
```

Python 校验标记和 `verified_bytes == active * token_bytes`，同时检查 output
未变。默认在一个 Runtime 中执行 `[1, min(2 * cores + 1, capacity), 1]` 三个
批次，改变源索引，覆盖多行/多批次 UB 复用与无任务核；不仅测试一次到达。
CPU 模型还注入单字节 UB 损坏，确保同步完成但内容错误时测试会失败。

先按上文命令测试 1 核、656 B；再通过 `KVAF_NPU_TOKEN_BYTES` 分别测试
31、32、33、16383、16384、16385，覆盖非对齐长度和 16 KiB 边界，
再用 `KVAF_NPU_CORE_COUNT=3/8` 测试分核。大 token 测试可设置
`KVAF_NPU_REQUESTS=19` 降低输出张量内存占用。当前只要求验收 `load`，
不运行 `copy`。本地没有 CANN 编译器和 NPU，硬件正确性需以上测试确认。

这个校验证明的是当前 staging→UB 的内容一致性；它不能单独证明 CPU 取到了
正确的业务源数据，也不能代替后续的完整输出校验。

### 修正 scatter 的 SIMD 启动参数

2026-10-29 19:48，PID 18131、Device 7 的新版 `load` 仍失败，用户确认
`flush` 通过。设备日志为 `errcode=78`、`MTE_ERROR_T0_0=0x4000`、
`SU_ERROR_T0_0=0`，匹配的故障符号是 `kvaf_scatter::Scatter+0x5b4`，
`fixedPCOffset=0x768`。参数末字 `0x305f5e100` 对应 `load`，没有执行
payload MTE3。这次日志报告 MTE 异常，不能沿用旧二进制的标量异常结论；
该偏移也不能脱离匹配二进制直接映射到某条源码语句。

核对 Host 启动配置时发现，scatter loader 仍继承 direct SIMT 路径的
`dynUBufSize=160 * 1024`。当前 scatter 是纯 SIMD kernel，使用静态
LocalTensor。根据 [CANN 9.1 aclrtLaunchKernelAttrValue](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_1814.html)，
`dynUBufSize` 用于 SIMT 动态 UB，纯 SIMD 算子只能设为 0。
此次将 scatter 的该参数设为 0，保持 `.asc`、ABI 和拷贝同步流程不变。
这个接口约束错误已修正，但用户确认新 Host 生效后仍报 507015，MTE2 尚未
通过。后续按下文的 `setup`/`dma` 检查点继续分段定位。

本次修改位于 Host loader，**只重编译 kernel `.o` 不会生效**。在目标机器
运行 `cmake --build build-npu -j`，确保 Python 加载新生成的
`libkvaf_execute_batch_get_host.so`，然后使用上文的 `load` 命令。新 Host
加载 scatter binary 时会打印一次：

```text
[KVAF_AIV] SCATTER_KERNEL_READY path=... dyn_ubuf_bytes=0
```

若没有此行，先核对实际加载的 Host 库路径。验收仍要求三个批次的
`verification=gm_ub_byte_compare` 标记与正确的 `verified_bytes`。
本地 loader 回归直接检查传给 ACL 的 scatter 动态 UB 为 0，同时检查
direct SIMT 的 160 KiB 配置；修复前新增断言失败，修复后通过。

### setup / dma 独立检查点

在回退元数据及 ELF 检查器提交后，增加 `setup` 和 `dma`。两者沿用真实的
CPU payload/done 协议，在同一份 `CopyPayload` 中按每个最多 16 KiB 的块
依次执行到指定位置：

- `setup`：完成 staging flush、tensor/拷贝参数准备、`ResetLoopModePara(OUT_TO_UB)`
  和 S→MTE2 的 set/wait，然后跳到下一块；不调用 payload DMA。
- `dma`：在上述基础上执行 GM→UB CopyIn，完成每次搬运的 MTE2→S set/wait
  后才处理下一块；不调用 `scratch.GetValue`，也不比较 staging 内容。
- `load`：在 `dma` 的执行范围之后逐字节比较 UB 和 staging。

三个检查点均不发出 MTE3 payload 拷贝，output 必须保持 guard。现有状态写回、
flush 和最终 `PipeBarrier<PIPE_ALL>()` 仍执行，所以异常定位范围也包括对应
路径的收尾；不能把检查点的成败直接等同于某一条具体指令。

新增 probe 编号 `setup=5`、`dma=6`，完成 phase 分别为 12、13；保留原编号
`done=1/flush=2/load=3/copy=4` 和 176 B 参数 ABI。Host 校验每个核的
magic/sequence/result/phase，并拒绝总处理请求数为零的 MTE2 检查点。
完成标记分别带 `configured_bytes`、`dma_bytes`；Python 要求它们等于该批
`active * token_bytes`，不把配置完成或 DMA 完成当成内容校验。

```text
[KVAF_AIV_SCATTER_PROBE_COMPLETE] stage=setup ... configured_bytes=656
[KVAF_AIV_SCATTER_PROBE_COMPLETE] stage=dma ... dma_bytes=656
[KVAF_AIV_SCATTER_PROBE_COMPLETE] stage=load ... verification=gm_ub_byte_compare verified_bytes=656
```

`done/setup/dma/load` 各在一个 Runtime 中执行 `[1, min(2 * cores + 1, capacity), 1]`
三个批次。默认 1 核时是 `[1, 3, 1]`，相应字节数为 656、1968、656。
需要同时重编 Host 与 kernel；旧 Host 会拒绝新开关，旧 kernel 会拒绝
新增 probe 编号，不能混用。CPU 服务协议没有改变。

### 按 ops-transformer 重写 MTE2

硬件反馈为 `setup` 通过、`dma` 仍报 507015；当前暂停继续增加诊断点，
按用户指定的算子库实现 GM→UB 搬运。参考版本固定为
`ops-transformer v9.1.0`（`c77c0829f1f0f0b8f882db1d9b051a758b605472`）：

- [GmUbGmCopy::GmUbGmDataBlockCopy](https://gitcode.com/cann/ops-transformer/blob/v9.1.0/mc2/common/op_kernel/gm_ub_gm_copy.h)
  的普通块使用 `DataCopy`，搬运元素数为 32 B 的整数倍。
- [MoeDistributeDispatchTeardown::LocalWindowCopy](https://gitcode.com/cann/ops-transformer/blob/v9.1.0/mc2/moe_distribute_dispatch_teardown/op_kernel/arch35/moe_distribute_dispatch_teardown_arch35.h)
  使用 `DataCopyPad`、按字节配置长度、禁用源 L2 hint，并在读取 UB 前同步 MTE2→S。

新的 `CopyIn` 先搬运 `floor(n / 32) * 32` 字节主体，再把剩余字节精确搬到
`scratch[body]`。例如 656 B 为 640 B DataCopy + 16 B DataCopyPad；13 B 只有
DataCopyPad；16384 B 只有 DataCopy。UB 尾部地址始终 32 B 对齐。
GM 使用 uint8_t 类型，按照 [DataCopy 连续搬运约束](https://asc.gitcode.com/api/SIMD-API/basic_api/memory_vector_compute/data_move/DataCopy_GMAndUB_continuous.html)
只需满足 1 B 对齐，因此相邻 packed token 的起点不要求 32 B 对齐。

这里是针对本项目的适配，未整份移植参考类：原通用类会把尾部读长度向上取整，
而本项目的 CPU staging 按 token 紧凑排列，所以尾部必须精确读取。参考类的
TPipe/TQue 管理也未引入：核对 `asc-devkit v9.1.0` 的
`impl/basic_api/dav_3510/kernel_tpipe_impl_c310.h` 后确认，TPipe 构造会隐式调用
`InitSocState()`，与已确认的初始化故障回退冲突。继续使用公开的静态 LocalTensor
和显式事件同步，每次搬运完成后才发下一次；没有双缓冲或流水。

本次仅替换 scatter 的 MTE2 实现，MTE3 搬运、gather、Host/参数 ABI、Python
调用及 CPU wire 协议保持原有行为。沿用现有 `dma` 和 `load` 检查点：

```bash
cmake --build build-npu --target kvaf_scatter_batch_get_kernel -j
DEVICE_ID=7 KVAF_NPU_CORE_COUNT=1 KVAF_NPU_TOKEN_BYTES=656 \
  KVAF_AIV_SCATTER_PROBE=load \
  KVAF_BATCH_GET_KERNEL="$PWD/build-npu/bin/kvaf_scatter_batch_get_kernels.o" \
  python tests/python/test_aiv_scatter_batch_get.py
```

沿用此前的 Python、CPU 地址、BatchPut kernel 和测试开关环境。已有最新 Host
时仅需重编 scatter kernel；`load` 要求三个批次的 GM/UB 字节比较全部通过，
而且 output 保持 guard。先验收 MTE2，MTE3 仍留到后续。
本地模型覆盖主体/尾部、全部 64 种源地址余数、1～65535 B 长度、跨 UB 分块、
1/3/8 核及重复批次；模型对每次 GM 读取施加 token 边界，防止越界读取被
分配余量掩盖。本地没有 CANN/NPU，模型通过不代表 507015 已经修复。

## Scatter 507015 的分段定位

拆分前，用户已反馈 `gather_read` 在 1、3、8 核通过。当时 Gather 和 scatter
共用 `KvafSequentialBatchGetAivKernelV1`，通过 mode 走不同分支；gather 不经过 scatter 的
HBM done 轮询和本地 `CopyPayload`，所以这些部分仍需单独定位。

新增 scatter Host 模型还发现：packed 的 open 请求中，`npu_tp` 字段的地址
不满足 `uint64_t*` 对齐要求。现改为先查询到对齐的局部变量，再赋值到协议
字段，UBSan 可复现并验证此修正。这是 Host 初始化问题，尚无证据将它与
AIV kernel 的 507015/errcode 259 联系起来。

`KVAF_AIV_SCATTER_PROBE` 提供同一 kernel 的九个检查点。flush/copy 各运行
一个非空批次，done/setup/gm/body/dma/ub/load 各运行三个批次。保留真实的 descriptor/notify、
CPU payload WRITE 和 done 协议；不能
用 CPU 侧 WRITE CQ 成功代替 AIV 读到本轮 done 的证明。

| 值 | AIV 执行到哪里 | Python 检查 |
| --- | --- | --- |
| `done` | 确认本轮 done 的 sequence/status，随后返回 | output 全部保持 guard |
| `flush` | 再执行输出行寻址、UB tensor 初始化和 staging 的 scalar cache 处理 | output 全部保持 guard，没有 payload DMA |
| `setup` | 再完成 MTE2 普通模式配置和 S→MTE2 同步 | configured_bytes 匹配，output 保持 guard，没有 payload DMA |
| `dma` | 再执行 GM→UB 并完成 MTE2→S 同步 | dma_bytes 匹配，output 保持 guard，不读取 UB 校验，没有 MTE3 写出 |
| `load` | GM→UB 后通过 MTE2→S 等待读入，再逐字节比较 UB/staging | 必须有字节校验完成标记且字节数匹配，output 保持 guard，没有 MTE3 写出 |
| `copy` | 完整执行 GM→UB→GM 并等待 MTE3 完成 | 精确比较 payload 和所有 guard |

`load` 现在验证全部有效 UB 数据与 staging 一致；`copy` 进一步验证最终输出。
各探针保留普通回归的默认 8193 请求容量、656 B token 和输出行位置。
flush/copy 的 active=min(cores, capacity)；done/setup/dma/load 的批次数量见上节。

先在 NPU 端更新并重编译 Host 和 kernel，保留前文的 Python/kernel/CPU 地址
环境变量。本次诊断开关不修改 CPU 协议，CPU 端可继续使用已匹配分支的服务。

```bash
git pull --ff-only
cmake --build build-npu -j
export PYTHONPATH="$PWD/build-npu/python"

# Bash：每个检查点使用独立进程，首次失败就停下，日志同时留存。
set -o pipefail
for stage in done flush setup dma load; do
  KVAF_RUN_NPU_TESTS=1 DEVICE_ID=7 KVAF_NPU_CORE_COUNT=1 KVAF_NPU_TOKEN_BYTES=656 \
    KVAF_AIV_BATCH_GET_MODES=cpu_push_scatter KVAF_AIV_SCATTER_PROBE="$stage" \
    python -u tests/python/test_aiv_scatter_batch_get.py 2>&1 | tee "scatter-${stage}.log"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    break
  fi
done
```

当前先定位 MTE2；`copy` 的 MTE3 检查和完整 scatter 回归留待 MTE2 通过后。

完成检查点会打印 `[KVAF_AIV_SCATTER_PROBE_COMPLETE] stage=...`，验证 output
和正常 shutdown 后打印 `PROBE_COMPLETE: ...`，不打印普通回归的 `PASS`。
探针最初使用旧 232 B ABI 的 reserved word；拆分后使用 scatter 独立的
176 B ABI 中的 `scatter_probe` 字段。Host 必须核对每核的
magic、sequence、专用 result 和 phase，旧 kernel 忽略该字段时会明确失败。
探针要求 `synchronize=True`，不允许异步普通成功；普通调用者会收到明确的
诊断异常，只有诊断脚本识别这个完成标记。空批次不能证明检查点已执行，因而
在探针模式下被拒绝。

按第一个失败的检查点缩小范围：

- `done` 就失败：尚不能证明 AIV 已读到 done，继续检查参数/描述符、notify、
  done 轮询和状态写回，结合该次 CPU 日志与 Ascend 异常。
- `done` 完成、`flush` 失败：优先检查输出行寻址、UB 初始化及 staging cache 操作。
- `flush` 完成、`setup` 失败：优先检查 MTE2 普通模式配置、S→MTE2 同步及收尾。
- `setup` 完成、`dma` 失败：范围缩小到 GM→UB CopyIn、MTE2→S 同步及收尾。
- `dma` 完成、`load` 失败：重点检查 UB 标量读取和内容比较；若返回
  `MTE2_GM_UB_byte_mismatch` 则是校验不一致，需与设备异常 507015 区分。
- `load` 完成、`copy` 失败：优先检查 UB→GM、输出地址和 MTE3 完成同步；若是
  output 比较失败而非 507015，则转查内容、长度和可见性。

这是执行范围的定位，不能直接断言某条指令就是根因。507015 后不尝试从已
abort 的 context 强行读取状态；保留本轮 `PROBE_BEGIN`、CPU 日志和设备异常。
先确保故障进程结束、设备恢复可用，再运行下一次试验。新二进制的故障 PC
需要与新 `.o` 对应，不能沿用旧 `+0xa18`。

后续 `copy` 检查点也完成后，清除开关运行普通回归，覆盖空批次、序号复用、A/B/A
输出、大小批次和跨 stream；检查点测试不能替代这些完整输出场景。

```bash
KVAF_AIV_SCATTER_PROBE= DEVICE_ID=7 KVAF_NPU_CORE_COUNT=1 \
  KVAF_AIV_BATCH_GET_MODES=cpu_push_scatter \
  python -u tests/python/test_aiv_scatter_batch_get.py
```

### flush 完成、load 报错后的更新

PID 7820、卡 7 的新日志将失败进一步限定在 MTE：`errcode=78`、
`MTE_ERROR_T0_0=0x4000`、`SU_ERROR_T0_0=0`、`fixedPCOffset=0xd28`。
这与旧日志的标量异常 259 不同，不能仅凭两次 ACL 都是 507015 认定根因相同。
此次 kernel 参数末字的高 32 位为 3，对应 `load` 检查点；该检查点不执行
payload 的 UB→GM。首个 staging 源地址按参数计算为
`0x120000200000 + 0x80100 = 0x120000280100`，token 长度为 656 B，
代码配置的 UB 偏移为 0、缓冲大小为 16 KiB；显式长度/对齐未见违反已知约束。
这不等于已经验证了编译器实际生成的指令参数。

代码核对发现顺序 kernel 使用静态 LocalTensor，却遗漏了 `InitSocState()`。
[官方静态 Tensor 指南](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/programug/Ascendcopdevg/docs/en/guide/programming_guide/programming_model/ai_core_simd_programming/cpp_tensor_programming/static_tensor_programming.md)
要求入口初始化、结束前执行全流水屏障；
[InitSocState 说明](https://asc.gitcode.com/api/SIMD-API/basic_api/tool_interface/system_init/InitSocState.html)
列出 950 AIV 的初始化内容包含 loop-mode、原子操作和 CTRL 等状态。
当时据此补齐了入口初始化及退出屏障，原有 DataCopyPad 参数、同步和四个
检查点保持不变。该入口初始化现已撤回，见上文回退记录。

当时根据静态 Tensor 的接口约束补充初始化，但尚未证明它就是此次 errcode 78
的根因。模型曾在每次 launch/core 前设置一个模拟 DMA 状态标记；这一抽象
不能验证真实 MTE 寄存器或硬件错误码 78，现已随入口初始化一起撤回。

重编译后依次复测 `KVAF_AIV_SCATTER_PROBE=load`、`copy`，均完成后清空开关
跑完整 scatter 回归。若 `load` 仍失败，应保留新 `.o` 和此次异常摘要，并核对
部署端 CANN 版本、实际头文件和新二进制的故障指令；不能继续沿用旧 `+0xd28`。

### 初始化修正后，异常前移到标量指令

PID 8769、卡 7 的后续日志仍是 `load`（参数末字高 32 位为 3），但异常变为
`errcode=259`、`SU_ERROR_T0_0=0x8`、`MTE_ERROR_T0_0=0`，修正偏移为
`+0x148`，kernel hash 为 `2185961939610012134`。这表明初始化修正没有完成
硬件验收；不能将保留的 `MTE_ERR_INFO=0x34a` 当成当前仍有 MTE 异常的证据。
偏移靠近入口，优先核对新增 `InitSocState()` 及其编译结果，但在看到匹配的
汇编前不能把 `+0x148` 直接对应到某个 C++ 调用，更不能判断原 MTE 问题已解决。

这一轮先保留 kernel 逻辑，采集部署端实际编译环境。不要为采集先重编译或
覆盖故障 `.o`；更新脚本后，在原有 Unix Makefiles 构建目录运行：

```bash
git pull --ff-only
python tests/python/collect_aiv_scatter_debug.py build-npu
```

脚本打印新建的 `build-npu/scatter-debug-*/report.txt` 路径。它不导入
torch/kvaf，也不运行 NPU kernel；读取 `build.make` 中现有的真实编译命令，
保留原来的架构、优化和 include 参数，只将编译动作改为 `-E/-dM`，输出
到独立目录。遇到多个候选命令或不支持的 recipe 会明确拒绝，不猜参数。

采集内容：

- 原部署、链接和临时 `.o` 的副本、SHA-256、时间戳，以及本地源码快照。
  若设置了 `KVAF_BATCH_GET_KERNEL`，额外保存其实际 sequential 路径；也可用
  `--kernel /path/to/kernel.o` 明确指定。
- 实际调用的编译器及其 symlink 解析位置、版本、CMake 选择的头文件路径，
  可发现的 CANN 版本文件和 `__CCE_AICORE__` / `__NPU_ARCH__` 等宏。
- 编译器预处理后的完整 `.ii`、实际包含的 common/data-copy 实现头文件、
  从 `.ii` 提取的 `InitSocState` 与 `InitSocStateImpl` 完整函数体及来源行号。
  初始化实现所在头文件也会保存，不再受固定头文件名列表限制。
- 每一步的命令、输出和退出码。某一步失败会保留其他结果并返回
  `COLLECTION_PARTIAL`，不会误报采集完成或硬件通过。

先提供 `report.txt`。预处理源码只能检查实际启用的 SDK 实现，不能映射故障
PC；精确 `+0x148` 映射仍须对应原二进制。

### CANN 9.1 采集报告：改为分析已保存的初始化源码

用户提供的 `scatter-debug-ig6ic27m/report.txt` 显示：编译器解析到
`cann-9.1.0/tools/bisheng_compiler/bin/bisheng`，实际包含的 SDK 头文件也来自
`cann-9.1.0/aarch64-linux/asc`；当前没有版本混用证据。实际编译选项为
`--cce-aicore-arch=dav-c310-vec`，预处理宏为 `__CCE_AICORE__=310`、
`__NPU_ARCH__=3510`。不能仅凭两个宏的数值不同断言架构不匹配。
driver 报告版本为 `25.6.rc1.b175`；现有材料不足以判断其兼容性。
已保存的 installed/linked/configured 三份 `.o` 的 SHA-256 一致，但还没有
取得原二进制 `+0x148` 对应的具体指令。

旧采集脚本的设备侧 `-S` 被编译器明确拒绝，错误为
`unsupported option '-S' on device side`。这是采集方法不适用，不能据此判断
NPU 故障原因。旧报告只摘录了调用 `AscendCUtils::InitSocStateImpl()` 的包装
函数，遗漏了真正实现；完整 `.ii` 已经成功保存，无需再次编译或运行 NPU。

更新脚本后，在仓库根目录分析这次保存的目录：

```bash
git pull --ff-only
python tests/python/collect_aiv_scatter_debug.py --analyze build-npu/scatter-debug-ig6ic27m
```

脚本打印新的 `init-report-*.txt` 路径，提供该文件内容即可。此模式只读取原
manifest、宏和 `.ii`，按预处理后的有效定义提取初始化函数，保留源文件行号；
不会调用编译器、读取后来更改的 SDK 头文件、覆盖旧报告或加载 kernel。
若未找到 `InitSocStateImpl` 函数体会明确返回不完整，包装层不能算采集成功。
后续新采集也已去掉不支持的 `-S`，不再输出大量无关的头文件保护宏。

## 当前验证状态

本地无 CANN/NPU，已完成：

- CMake 普通 C++ 构建和现有回归。
- 拆分隔离回归：独立 loader 的路径/symbol/ABI 和交错调用、缺失另一模式二进制，
  scatter 增量重编译不改变 gather/direct 产物、独立 Python 入口和组合子进程失败汇总。
- Loader 启动参数回归：scatter 的纯 SIMD 动态 UB 参数为 0，direct SIMT
  路径仍为 160 KiB。
- 候选选择回归覆盖连接回退、本地设备错误停止重试、普通/自适应选择及清理错误
  不覆盖原始失败。
- 生产 `.asc` 的 CPU 模型执行：解码 SQ/CQ，调用真实 CPU gather executor，
  比较全部 output/guard。覆盖 1/3/8 核、1/13/656/4096/65535 B token、
  1～6 SGE、65536 请求、SQ/CQ 回绕、旧序号、非法行、超时及控制/数据 CQ 错误。
- payload 搬运模型延迟执行 MTE2/MTE3，验证同步后可见及单 UB 复用；覆盖
  GM 地址的 64 种对齐余数、1～65535 B 尾部和 16 KiB 分块边界、重复源更新、
  output 与 UB 边界 guard。模型不替代真实 CANN 编译和设备异常检查。
- 真实 Host owner 的延迟 ACL 模型：串行复用、异步错误上报、capture 拒绝、
  部分初始化和关闭失败时资源保留/重试。Host 集成源码使用 ACL 声明替身编译。
- Scatter 九个检查点的模型覆盖：真实 notify/done 流程、1/3/8 核、656/16385 B
  token、重复序号、预期 DMA 次数、output guard、空请求/非法模式/超时拒绝；
  分别检查普通模式配置、S→MTE2/MTE2→S 等待次数，以及 UB 标量读取次数；
  注入损坏 UB 时，dma 仅报告搬运完成，load 必须报告内容不匹配。
  Host 覆盖旧 kernel、错误 phase、设备错误保留、同步要求和诊断后的资源回收。
  Python 检查只接受对应 checkpoint 标记及匹配字节数，不吞掉 507015 或其他异常。
- 新增内核模型和 Host 生命周期测试通过 AddressSanitizer / UndefinedBehaviorSanitizer。
- CPU 新源码使用 UMDK 24.12 头文件通过语法检查；macOS 检查时仅补充 Linux
  的 `MADV_HUGEPAGE` 常量，未链接或执行 CPU 硬件服务。

用户此前已在硬件反馈 gather 的 1/3/8 核回归通过；拆分后撤回入口操作的独立
gather 也已反馈成功（所给命令为 1 核）。scatter 探针此前反馈 flush 完成、
load 失败，补初始化后仍有 507015。撤掉整体初始化、按官方源码调整 MTE2
配置、UB 构造及内容验证后，新日志仍报告 MTE 异常 78。修正 scatter Host
的纯 SIMD 启动参数后，用户确认新 Host 生效但仍报 507015。元数据及 ELF
检查器提交已按用户要求回退；当前增加 setup/dma 分段检查点，尚待上机验证。
尚待验证 scatter
的非对齐 MTE3 写出、CPU/NPU 缓存可见性、跨 Jetty
done 保序及真实超时/断链回收。这些本地检查不替代 CANN 编译和双机验收。
硬件回归未启用时返回 skip，不能视为硬件验收通过。
