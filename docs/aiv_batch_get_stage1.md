# AIV BatchGet 第一阶段实现与验证

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

Scatter 的本地 payload 搬运使用 `DataCopyPad<uint8_t>`，通过一个 16 KiB UB
缓冲顺序执行 MTE2（GM→UB）和 MTE3（UB→GM）。MTE2 完成后才写出，MTE3
完成后才复用缓冲并报告完成；长度按字节给出，UB padding 不写到 output。
源 GlobalTensor 禁用 L2 cache hint，且不再经 SIMT 读取被 CPU WRITE 更新的
staging。原实现的 MainScalar cache 刷新不能代替 SIMT cache 刷新。
顺序 kernel 在入口调用 `InitSocState()`，在退出前调用
`PipeBarrier<PIPE_ALL>()`；没有 TPipe 的静态 Tensor 写法必须自行初始化和收尾，
尤其不能假定前一个算子留下的搬运 loop-mode、原子操作等全局状态为默认值。
接口约束见 [GMToUB DataCopyPad](https://asc.gitcode.com/api/SIMD-API/basic_api/memory_vector_compute/data_move/DataCopyPad_GMToUB.html)、
[UBToGM DataCopyPad](https://asc.gitcode.com/api/SIMD-API/basic_api/memory_vector_compute/data_move/DataCopyPad_UBToGM.html) 和
[SIMD/SIMT 内存层次](https://asc.gitcode.com/guide/programming_guide/advanced_programming/advanced_ai_core_programming_model/simd_simt_hybrid_programming/memory_hierarchy.html)。

Gather 的控制与数据队列分开，每核固定两条 Jetty。可配置 1～6 个本地 SGE，
每条 READ 的总长度最多 64 KiB，按 64 B WQEBB 处理变长 WQE 和 SQ 回绕。
阶段内部批量提交，只在物理 SQ/WR 容量不足或阶段结束时 drain；
`batch_get_max_inflight_total` 仅对原 `direct_read` 生效。

## 代码入口

- `access_operator/demo/aiv_batch_get/kernel/execute_sequential_batch_get_aiv.asc`：
  两种模式、行校验、notify/ACK、CQ、精确字节搬运及状态回写。
- `access_operator/demo/aiv_batch_get/host/sequential_batch_get.cpp`：
  workspace、注册内存、CPU 会话、序号和异步状态回收。
- `access_operator/demo/aiv_batch_get/include/sequential_batch_get_abi.h`：
  独立的 V1 参数与状态 ABI；kernel symbol 为 `KvafSequentialBatchGetAivKernelV1`。
- `runtime/src/cpu_push_get_*`、`runtime/src/cpu_gather_executor.cc`：
  复用 `aicpu-urma` 的 CPU 数据面，未引入 AICPU kernel 或设备端 liburma。
- `runtime/src/peer_link.cc`、`runtime/src/cpu_runtime.cc`：
  增加 enable/open/start/stop；新增 `StopGatherLanes` 在释放 DDR 前 join worker。
- `runtime/src/npu_runtime.cc`、`bindings/src/npu_python_runtime.cc`：配置入口及
  gather 的 DDR 预留区，预留区不参与 KV slot 分配。

两端需一起重编译。初始化通过新增控制命令的响应确认能力；不支持的 CPU
服务会初始化失败。Host 和 AIV object 也必须来自同一版本。

构建产出两个独立二进制：`kvaf_execute_batch_get_kernels.o`（direct_read）和
`kvaf_sequential_batch_get_kernels.o`（scatter/gather），均复制到 `build-npu/bin/`。
AscendC 会在每个编译单元中生成 `g_tilingKey` 等全局符号，因此不合并两个
临时 object。顺序模式收到标准 `kvaf_execute_batch_get_kernels.o` 路径时，
自动加载同目录下的 `kvaf_sequential_batch_get_kernels.o`；也可直接传入顺序
二进制路径。部署时需要一起复制两个文件。

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
  "$PWD/build-npu/bin/kvaf_sequential_batch_get_kernels.o"
KVAF_NPU_CORE_COUNT=1 python tests/python/test_aiv_sequential_batch_get.py
KVAF_NPU_CORE_COUNT=3 python tests/python/test_aiv_sequential_batch_get.py
KVAF_NPU_CORE_COUNT=8 python tests/python/test_aiv_sequential_batch_get.py
```

脚本默认分别测试两种模式，使用 8193 请求容量，覆盖空批次、尾批、重复调用、
随机输出、重复源、guard、A/B/A 输出、两种 Tensor 重载以及跨 stream 顺序复用。
可用 `KVAF_NPU_TOKEN_BYTES=13`、`KVAF_NPU_REQUESTS=65536`、
`KVAF_AIV_GATHER_SGL_SIZE=1` 增加边界组合。
CPU `--gather-threads` 至少等于 gather 的 AIV 核数。
上述命令在仓库根目录执行。Get、Put 及同目录下的 sequential 三个 `.o` 文件
都应存在；路径检查失败时脚本会打印具体变量及绝对路径，并在初始化 NPU 前退出。

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
  python -u tests/python/test_aiv_sequential_batch_get.py > npu-stage1.log 2>&1
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

## Scatter 507015 的分段定位

用户已反馈 `gather_read` 在 1、3、8 核通过。Gather 和 scatter 使用同一个
`KvafSequentialBatchGetAivKernelV1`，但走不同分支；gather 不经过 scatter 的
HBM done 轮询和本地 `CopyPayload`，所以这些部分仍需单独定位。

新增 scatter Host 模型还发现：packed 的 open 请求中，`npu_tp` 字段的地址
不满足 `uint64_t*` 对齐要求。现改为先查询到对齐的局部变量，再赋值到协议
字段，UBSan 可复现并验证此修正。这是 Host 初始化问题，尚无证据将它与
AIV kernel 的 507015/errcode 259 联系起来。

`KVAF_AIV_SCATTER_PROBE` 提供同一 kernel 的四个检查点。每次只运行一个非空
批次，保留真实的 descriptor/notify、CPU payload WRITE 和 done 协议；不能
用 CPU 侧 WRITE CQ 成功代替 AIV 读到本轮 done 的证明。

| 值 | AIV 执行到哪里 | Python 检查 |
| --- | --- | --- |
| `done` | 确认本轮 done 的 sequence/status，随后返回 | output 全部保持 guard |
| `flush` | 再执行输出行寻址、UB tensor 初始化和 staging 的 scalar cache 处理 | output 全部保持 guard，没有 payload DMA |
| `load` | 再执行参数设置、事件同步和 GM→UB，等待 MTE2 完成 | output 全部保持 guard，没有 MTE3 写出 |
| `copy` | 完整执行 GM→UB→GM 并等待 MTE3 完成 | 精确比较 payload 和所有 guard |

`load` 为安全返回额外使用 MTE2→S 同步；它验证读取阶段及其诊断收尾，
不提供 UB 数据正确性的证明。数据正确性由 `copy` 的逐字节比较验证。
各探针使用相同的请求形状；单核默认 active=1、token=656 B，保留普通回归的
8193 请求容量和输出行位置。多核时 active=min(cores, capacity)。

先在 NPU 端更新并重编译 Host 和 kernel，保留前文的 Python/kernel/CPU 地址
环境变量。本次诊断开关不修改 CPU 协议，CPU 端可继续使用已匹配分支的服务。

```bash
git pull --ff-only
cmake --build build-npu -j
export PYTHONPATH="$PWD/build-npu/python"

# Bash：每个检查点使用独立进程，首次失败就停下，日志同时留存。
set -o pipefail
for stage in done flush load copy; do
  KVAF_RUN_NPU_TESTS=1 DEVICE_ID=7 KVAF_NPU_CORE_COUNT=1 KVAF_NPU_TOKEN_BYTES=656 \
    KVAF_AIV_BATCH_GET_MODES=cpu_push_scatter KVAF_AIV_SCATTER_PROBE="$stage" \
    python -u tests/python/test_aiv_sequential_batch_get.py 2>&1 | tee "scatter-${stage}.log"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    break
  fi
done
```

完成检查点会打印 `[KVAF_AIV_SCATTER_PROBE_COMPLETE] stage=...`，验证 output
和正常 shutdown 后打印 `PROBE_COMPLETE: ...`，不打印普通回归的 `PASS`。
实现使用原 ABI 的 reserved word，参数总长仍是 232 B。Host 必须核对每核的
magic、sequence、专用 result 和 phase，旧 kernel 忽略该字段时会明确失败。
探针要求 `synchronize=True`，不允许异步普通成功；普通调用者会收到明确的
诊断异常，只有诊断脚本识别这个完成标记。空批次不能证明检查点已执行，因而
在探针模式下被拒绝。

按第一个失败的检查点缩小范围：

- `done` 就失败：尚不能证明 AIV 已读到 done，继续检查参数/描述符、notify、
  done 轮询和状态写回，结合该次 CPU 日志与 Ascend 异常。
- `done` 完成、`flush` 失败：优先检查输出行寻址、UB 初始化及 staging cache 操作。
- `flush` 完成、`load` 失败：优先检查 GM→UB 参数、cache hint、事件同步与诊断收尾。
- `load` 完成、`copy` 失败：优先检查 UB→GM、输出地址和 MTE3 完成同步；若是
  output 比较失败而非 507015，则转查内容、长度和可见性。

这是执行范围的定位，不能直接断言某条指令就是根因。507015 后不尝试从已
abort 的 context 强行读取状态；保留本轮 `PROBE_BEGIN`、CPU 日志和设备异常。
先确保故障进程结束、设备恢复可用，再运行下一次试验。新二进制的故障 PC
需要与新 `.o` 对应，不能沿用旧 `+0xa18`。

四个检查点均完成后，清除开关运行普通回归，覆盖空批次、序号复用、A/B/A
输出、大小批次和跨 stream；探针只测一个批次，不能替代这些场景。

```bash
KVAF_AIV_SCATTER_PROBE= DEVICE_ID=7 KVAF_NPU_CORE_COUNT=1 \
  KVAF_AIV_BATCH_GET_MODES=cpu_push_scatter \
  python -u tests/python/test_aiv_sequential_batch_get.py
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
现已补齐入口初始化及退出屏障，原有 DataCopyPad 参数、同步和四个检查点保持不变，
先用同一 `load` 用例验证此修正，不再增加同步对照开关。

遗漏初始化是已确认的代码缺陷，但尚未证明它就是此次 errcode 78 的根因。
本地模型在每次 launch/core 前模拟被其他算子改变的 DMA 状态，验证 kernel
会先初始化再搬运；模型不解码真实 MTE 寄存器，也不模拟硬件错误码 78。

重编译后依次复测 `KVAF_AIV_SCATTER_PROBE=load`、`copy`，均完成后清空开关
跑完整 scatter 回归。若 `load` 仍失败，应保留新 `.o` 和此次异常摘要，并核对
部署端 CANN 版本、实际头文件和新二进制的故障指令；不能继续沿用旧 `+0xd28`。

## 当前验证状态

本地无 CANN/NPU，已完成：

- CMake 普通 C++ 构建和现有回归。
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
- Scatter 四个检查点的模型覆盖：真实 notify/done 流程、1/3/8 核、656/16385 B
  token、重复序号、预期 DMA 次数、output guard、空请求/非法模式/超时拒绝；
  Host 覆盖旧 kernel、错误 phase、设备错误保留、同步要求和诊断后的资源回收。
  Python 检查只接受对应 checkpoint 标记，不吞掉 507015 或其他异常。
- 新增内核模型和 Host 生命周期测试通过 AddressSanitizer / UndefinedBehaviorSanitizer。
- CPU 新源码使用 UMDK 24.12 头文件通过语法检查；macOS 检查时仅补充 Linux
  的 `MADV_HUGEPAGE` 常量，未链接或执行 CPU 硬件服务。

用户已在硬件反馈 gather 的 1/3/8 核回归通过；scatter 仍有 507015，新增探针
待上机。尚待验证 scatter 的非对齐 MTE3 写出、CPU/NPU 缓存可见性、跨 Jetty
done 保序及真实超时/断链回收。这些本地检查不替代 CANN 编译和双机验收。
硬件回归未启用时返回 skip，不能视为硬件验收通过。
