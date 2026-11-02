# AIV scatter BatchGet 优化实现方案

日期：2026-10-31。代码基线：`aiv-scatter@7a499b4`。
本文包含本轮实现和后续候选；性能收益待目标 NPU 的端到端测试验证。
已实现单遍校验/清单构建，以及每 bank 32 KiB、最多 32 行的双 bank 搬运（共 64 KiB）。
只在 bank 复用或退出前确认其写回完成；656 B 行使用 672 B UB pitch，
超过单 bank 大小的行保留精确分段路径。两项改动均可独立关闭做端到端对照。
按当前验收要求，直接使用现有完整 BatchGet 性能测试，不增加独立 staging→output 性能测试。
最新 100 条 4 MiB 样本的分析和更新后的优先级见
[UB 批量版本性能分析](aiv_scatter_ub_batch_results.md)。

## 1. 建议与依据

当前落地 **单遍清单构建、UB 多行搬入与双缓冲**，
按现有完整 BatchGet 测试分别判断两项收益，**SIMT 直接 GM→GM** 保留为后续对照候选。
规则输出增加多块写出快路径；随后再做 CPU ready 驱动的通信与 scatter 重叠。
前两条路线共享正确性、计时和基准，分别归因后再考虑 UB+SIMT 混合。

当前数据和计时边界见 [计时审计](aiv_scatter_timing_audit.md)。两组同形状数据为
1024 个源 block、每 block 4 个 token、有效请求 4096、4 个 AIV 核：

| 均值 | 656 B/token | 4096 B/token |
| --- | ---: | ---: |
| 有效 payload | 2,686,976 B | 16,777,216 B |
| 同步 BatchGet 端到端 | 1884.210 μs | 2925.455 μs |
| CPU 通信窗口 | 78.038 μs | 1116.938 μs |
| core 0 构建和 notify | 361.501 μs | 422.438 μs |
| max(各核 scatter 时长) | 1272.579 μs | 1156.049 μs |
| 本地 scatter 有效带宽 | 2.111 GB/s | 14.513 GB/s |

656 B 的 scatter 约占端到端 68%；数据量扩大约 6.24 倍，scatter 时长反而略降。
结合代码，优先怀疑逐行指令、缓存维护和同步开销；当前打点不能证明这些子项各占多少。
4 核时每核处理 1024 行，平均约 1.24 μs/行（656 B）和 1.13 μs/行（4096 B）。
不能把 4096 个有效 token 与请求 Tensor 容量 8192 混为一谈。

## 2. 基线实现的具体限制

入口是 `access_operator/demo/aiv_batch_get/kernel/execute_scatter_batch_get_aiv.asc`。
CPU 已将本轮 payload 按请求顺序紧凑写入 HBM staging；AIV 完成的工作为：

```text
等待整批 done
  → 每核遍历本核 token
    → staging[i] 刷新缓存、设置 Tensor、重置搬运模式
    → 640 B GM→UB，等待 MTE2→S
    → 16 B GM→UB，等待 MTE2→S
    → 等待 MTE2→MTE3
    → 656 B UB→output[output_indices[i]]
    → 等待 MTE3→S，再开始下一行
```

此外还有每行 S→MTE2 事件。656 B 每行总计约五组 set/wait，
4096 行共 8192 次输入搬运调用、4096 次输出搬运调用。
当前 `kCopyBytes=16384` 只是单行搬运的 scratch 上限，没有聚合多个 token。
`Flush()` 还逐 64 B cache line 调用 DCCI，且每行重复执行。

因此应减少每行固定工作、让多次小搬运在流水线上连续执行，最后按 tile 回收资源。
单纯扩大 scratch，或仅把两份 UB 数组声明出来，都不会自动形成流水。

## 3. 主线：UB tile + 批量完成 + 双缓冲

### 3.1 先用一个精确长度 DataCopyPad 验证输入简化

保留逐行处理和完成等待，只将 `640+16` 两次输入替换成一次精确 656 B
`DataCopyPad`。这是验证目标 SDK 的非对齐 GM 读和 UB padding 的独立对照，
也能区分“减少输入命令”与后续“减少同步”的收益。

### 3.2 多行输入，UB 内按 672 B 排列

不再在 CPU 收集一次：HBM staging 本身已经连续。每核一次取 B 行，拟用 Normal 模式：

```text
DataCopyPad GM→UB:
  blockCount = B
  blockLen   = 656 bytes
  srcStride  = 0       // GM 中两行之间无额外间隔
  dstStride  = 0       // UB 中无额外 dataBlock 间隔
  left/right padding = 0

GM staging: [656][656][656]...
UB bank:    [656+16 dummy][656+16 dummy][656+16 dummy]...
UB 行起点:  bank + j*672
写回:       从该行起点仅写 656 B
```

Normal 模式每块补齐到 32 B；UB 行起点需满足 32 B 对齐。
若直接把 `B*656` 字节紧凑放进 UB，再用 `UB+j*656` 分行 MTE3 写出，
奇数行会出现 16 B 对齐偏移，不能照搬。
672 B 只用于 UB 排列，workspace、CPU 协议、有效字节数和 output stride 仍为 656 B。
写回 672 B 会破坏相邻输出，禁止使用这种方式消除尾部处理。

先测 B=16/32/64。B=32 时一 bank 为 21,504 B，两 bank 为 43,008 B；
B=64 时两 bank 为 86,016 B，另加索引和必要的资源预留。
按可用 UB 自适应，而不是所有 token 长度都固定相同行数：

```text
row_pitch = align_up(token_bytes, 32)
tile_rows <= floor((available_ub - metadata_and_reserve) / banks / row_pitch)
```

例如 4096 B 行使用 32 行双缓冲就需要 256 KiB，不能复用 656 B 的默认配置。
先以 656 B/B=32、4096 B/B=8 为试验起点，再扫描参数；预算不足或单行过大时
采用分段搬运/原路径。实际预算以目标设备、编译资源和 Host launch 配置为准。

这一优化减少软件发出的搬运命令，不意味着多块指令内部没有逐块成本。
B=64、4096 行时，理想输入调用数从 8192 降为 64，随机输出调用仍为 4096。
不能按调用数比例推断带宽提升。

### 3.3 每个 tile 完成一次，避免每个 token 排空

一个 tile 的 MTE2 完成后建立 MTE2→MTE3 依赖，在 MTE3 上连续发出该 tile 的
有效输出写入。只在 bank 复用或退出前确认这批 MTE3 全部完成，
不在每行后执行 MTE3→S 完成等待。

稳定的 Tensor 设置、`ResetLoopModePara` 和索引准备尽量移出逐行路径。
逐项检查目标 SDK 的指令队列、事件和循环模式约束；必要的硬件背压不能省略。
第一版单 bank 实现批量完成，单独测量后再引入双 bank。

### 3.4 已实现：双 bank 重叠 MTE2 和 MTE3，性能待目标机验证

```text
MTE2:  load tile0/bank0 | load tile1/bank1 | load tile2/bank0 | ...
MTE3:                   store tile0      | store tile1      | store tile2
```

每 bank 有独立事件，状态为 free→loading→ready→storing→free。
加载前等该 bank 上一批 MTE3 完成；写出前等该 bank MTE2 完成。
处理首批预取、最后不足 B 行、空分区和最终 drain，禁止读未就绪数据或提前复用 UB。
这只是本地缓冲复用，不引入 CPU URMA 每固定组排空的在途限流。

## 4. 对照路线：SIMT 并行 scatter

### 4.1 直接 GM→GM

利用当前已用于 BatchPut 的 `__simt_vf__` / `AscendC::Simt::VF_CALL` 工具链，
按较大 tile 或每核区间调用 VF，通过线程并行执行 staging→output。
不能每个 token 调一次 VF，否则又引入细粒度调度开销。

初始映射为一个 32-thread Warp 协作一个 token，Warp 内 lane 搬同一行的相邻元素，
多个 Warp 处理不同 token。656 B = 164 个 uint32，概念映射为：

```text
row = core_begin + warp_id + k*warps_in_vf
for word = lane; word < 164; word += 32:
    output[output_indices[row]][word] = staging[row][word]
```

实际先从 4 B 访存起步，再比较 8/16 B 的目标 SDK 内置短向量访问，
选择线程数、Warp 数和每个 VF 的行数。所有强类型访存需校验源/目的对齐，
不足向量宽度的通用长度精确处理尾部。656 B 可被 16 整除，但其他长度不一定。
不要按“一线程复制一整行”分配不同随机目标给相邻 lane，这会破坏 Warp 内访存合并。

SIMT 省去显式逐行 MTE 搬运和事件，但 GM 访存、寄存器压力、cache 与 VF 开销
可能抵消收益。必须与 UB tile 在相同形状和映射上比较，不能预定它一定更快。

### 4.2 UB+SIMT 混合，作为条件性的第三条路线

如果 UB tile 已改善输入，但标量发出大量 MTE3 行命令仍是瓶颈，再试：
MTE2 批量读入 UB→SIMT 从 UB 并行写入随机 GM 行。
复用 672 B pitch 和 tile 元数据，明确 MTE2→V、V 完成→bank 回收的依赖，
检查 UB bank 冲突。它比直接 GM→GM 多一层管理，不应一开始与其他改动捆绑。

### 4.3 缓存和 launch 是正确性的一部分

CPU URMA WRITE 对 payload 的可见性必须适用于所选加载路径。
SIMT 普通 GM load 与当前 MTE2 的 cache 路径不同；done/ready、`volatile` 或线程栅栏
本身不能替代所需的缓存一致性维护。按目标 SDK 验证 cache hint、DCCI 与发布/消费协议。
不得通过仅读固定输入来证明协议正确。

当前 scatter Host 为纯 SIMD 设置 `dynUBufSize=0`。纯 SIMD 扩大静态 UB 仍按该模式管理；
引入 SIMT 后，要同步确认函数资源信息、动态 UB、DCache 和静态 UB 的预算，
不能直接套用另一个 kernel 的 160 KiB 配置。
所有变体保留 AIV 函数类型元数据；此前缺失 `K_TYPE_AIV` 的问题不能回归。

## 5. 其他有价值的方向

| 方向 | 具体做法 | 条件与优先级 |
| --- | --- | --- |
| 规则输出多块写出 | 顺序连续目标或固定行间隔使用一次多块 DataCopyPad，UB srcStride=0，GM dstStride 为行间额外字节；只输出有效长度 | 高；随机映射仍走离散写出 |
| 连续 run 合并 | 按源顺序发现连续递增的输出行，用大块/多块搬运；仅在开销有收益时启用扫描 | 中；dense 随机排列不等于源顺序连续 |
| metadata 批量加载 | 按 tile 获取 output_indices、计算目的地址并复用，减轻逐行标量读取和重复设置 | 与 UB/SIMT 一起评估，不能假设 cache miss 数等于 token 数 |
| 输出分区/逆映射 | 对唯一且稠密的目标，构建 output→staging 逆索引，按输出区间并行复制，将随机写转换为按行顺序写 | 条件性实验；额外索引构建与随机 staging 读必须计入总耗时，不能免费预排序 |
| 缓存维护优化 | 统计并评估逐行 Flush；验证后尝试按 tile 范围维护，或采用有依据的 payload cache 策略 | 高，但须证明外部 DMA 可见性；不能直接删除 |
| 多核/任务划分 | 扫描 1/2/4/8 核，避免小请求使用过多核；tile 尽量按完整 staging cache group 划分 | 中；不能假定线性扩展 |
| CPU ready 驱动 | 哪个 worker/chunk 完成就先 scatter，重叠 CPU→HBM 和本地复制 | 独立后续阶段，详见下一节 |
| CPU WRITE 到最终 output | 传递最终目标映射并注册/import output，减少 staging 后的整次 HBM 读写 | 架构备选；随机目标可能增加 WR，需重做生命周期和传输计划 |
| 下游融合 | 下游算子直接消费 packed staging+映射，消除独立 scatter | 业务接口允许时收益潜力大，涉及布局与生命周期 |

本地复制压缩后，core 0 的构建和 notify 约 0.36 ms 会成为更显著的固定成本。
下一步可分核执行 ValidateRows/Source 和写清单，核间汇总结果并确认清单可见，
全部校验成功后才由 core 0 发布 notify；任何核失败都不能让 CPU 解引用未校验清单。
这属于控制阶段优化，应单独记录构建耗时与核间协调成本，不能归入 scatter 的提速。

656 B 行逐行 Flush 约触及 11 个 64 B 行。仅把同一 packed 区间的重复边界行合并，
4096 token 的行操作约从 45,056 降至 41,984，节省约 7%，不能据此解释数量级提升。
真正是否能减少更多维护，应通过 cache 路径证据和实测决定。
随机输出可能让不同核触及同一 cache line 的不同字节，精确长度写回和 guard 校验必须保留。
重复源可以合法复用；输出行沿用 [Stage 1 接口约束](aiv_batch_get_stage1.md) 中的唯一性要求，
批内离散写出的有效目的区间因此不重叠。不扩展源输出重叠语义，也不擅自排序请求。

## 6. CPU 通信与 scatter 重叠的落地条件

CPU 已有每 worker、每 Jetty 的 1024 行 chunk ready；AIV 当前只等待整批 done。
后续让各核使用与 CPU 一致的 `WorkerRange`，消费四路 Jetty 共同完成的连续前缀。
序号、范围、进度回退、错误和超时仍需检查；kernel 返回前仍须满足最终 done 和本地写出完成。
设备端应共享等价的分区定义，不能直接包含依赖 Host 标准库的 layout 实现。

**当前 4096 token/4 核恰好每 worker 1024 行，每 worker 只有一个 chunk。**
复用现有 ready 只能提供 worker 间较粗的先后重叠，不能形成每 worker 的多 chunk 流水。
UB tile 32/64 行也不会让 CPU 自动每 32/64 行发布 ready。
先验证原协议消费路径，再将 128/256/512 行 chunk 作为独立协议实验，
同步修改 CPU marker 存储、计划/范围、两端常量和兼容检查；不允许两端默认为不同 chunk 大小。
更细的 per-Jetty shard 消费可另行设计，不在首版引入。

656 B 数据的 CPU 通信只有约 78 μs，本地 scatter 却约 1273 μs，
因此优先压缩本地工作。4096 B 数据的通信和 scatter 都约 1.1 ms，细粒度重叠更有价值。
重叠会竞争 HBM 资源，须同时观察 CPU 通信指标；优化本地 scatter 不等于提升 URMA 链路带宽。
开启流水后，notify→done 与 scatter 会重叠，旧的线性四段时间拆解不能继续直接相加。

## 7. 工程实现拆分

| 阶段 | 交付 | 合入依据 |
| --- | --- | --- |
| P0 | 保存基线二进制及现有完整 BatchGet 的原始结果 | 每个样本仍是一次同步 BatchGet；使用相同形状、映射和测试配置 |
| P1 | 已落地 UB tile 双 bank 和单遍校验/清单构建，提供两个独立运行时开关 | 模型覆盖四种开关组合；目标机完整 changing-input 正确性和端到端性能仍需复测 |
| P2 | SIMT GM→GM；必要时 UB+SIMT | 相同请求、核数、映射与缓存条件下 A/B，按端到端和资源代价决定默认实现 |
| P3 | 规则输出多块写出、cache 维护和核数调优 | 快路径命中条件可验证，随机输出和通用长度保留可靠路径 |
| P4 | worker/chunk ready 消费，再评估更细 chunk | 协议测试及全流程正确，端到端收益超过控制面与 HBM 竞争代价 |

主要改动位置：

- `access_operator/demo/aiv_batch_get/kernel/execute_scatter_batch_get_aiv.asc`：tile/SIMT、分区、完成语义。
- copy helper 保留在 scatter kernel 内，由现有完整 BatchGet 测试覆盖。
- `host/execute_scatter_batch_get_loader.cpp`：变体所需资源配置与 kernel 类型检查。
- `tests/python/test_aiv_scatter_perf.py`、`scatter_perf_common.py` 和包装脚本：参数、原始样本、汇总。
- `common/include/scatter_batch_get_profile.h` 及对应 Host/绑定：确需增加字段时显式版本化。
- P4 才涉及 `cpu_push_get_abi.h`、`cpu_push_get_layout.h`、`runtime/src/cpu_push_get_transport.cc`。

当前通过 `KVAF_SCATTER_FUSED_BUILD`、`KVAF_SCATTER_DOUBLE_BUFFER` 环境变量在初始化时选择
`reference/fused/double/both` 四种对照；测试入口也提供 `--fused-build 0/1`、
`--double-buffer 0/1`。同一二进制使用固定的最大 64 KiB 静态 UB 地址范围；
SIMT 列为后续独立候选。
`fused_build=1` 已进一步实现多核分摊和 UB 元数据批处理：8 项对齐分区、
256 项构建 tile、4096 项以内 address/valid 表在 UB 缓存，较大表走标量查找 fallback。
每核私有完成行汇合后 core 0 发布 notify，不改变 CPU 服务协议。
省略全容量输入缓存维护、二次读取和逐条 GM descriptor 写；保留内容边界校验。
构建使用前 41 KiB UB，payload 阶段复用这些地址。Host/kernel 需配套重编译。
核参数维持现有 176 B 合约，reserved 使用新版 options 标签，profile 是独立 bit；
保留旧 zero/profile-only 标签的两遍清单、单 bank 语义，未知版本/bit 明确拒绝。
SIMT 变体同步提供 Host 资源配置，不仅替换一个 object 文件。
每种变体单独进程运行：当前 loader 首次 load 后缓存函数句柄，同进程换路径不会重新加载。
记录 kernel 指纹、运行时开关、设备/CANN、算法与 tile/bank/vector 配置，保证对比可复现。

## 8. 测试与计时验收

沿用现有真实 CPU 服务→HBM staging→scatter→下游 NPU 消费→完整 output/guard 的测试，
所有优化均由完整 BatchGet 的端到端结果决定收益。
已有 kernel 模型测试扩展多块 DMA 和延迟写出，验证 UB padding、尾批、
大行 fallback、bank 复用和最后写回完成；该测试仅作正确性回归，不作为性能基准。

首轮性能扫描采用 4096 有效 token、656/4096 B、随机输出、1/2/4/8 核。
先在 4 核扫描 tile=16/32/64 的可用配置和 SIMT 参数，再扩展核数，避免全参数盲扫。
对比均使用相同 source/output 映射和随机 seed；源随机性影响 CPU 通信，
而 staging 已连续，不能把源随机性误认为本地 scatter 的随机输入读。

正确性还须覆盖：

- 长度 1/16/31/32/33/655/656/657/4096，以及超过单 bank 容量的分段 fallback。
- 空、单条、小于核数、非整除分区，tile−1/tile/tile+1、chunk 边界和最大有效前缀。
- 顺序/随机、row-step=1/2、输出空洞、重复源和非法输入；不扩展未定义的重复输出语义。
- 每轮改变 payload、源 offset 和输出映射；旧 sequence、CPU 错误、超时及尾部 drain。
- 同 stream 后继 NPU 算子直接消费输出；逐字节结果和所有未写区域 guard 一致。
- 每种二进制检查 `K_TYPE_AIV`、资源配置和内存边界；不能通过强制 done 或跳过阶段通过。

计时分两类：关闭 profile 的真实同步调用时延是主指标；开启 profile 的样本用于归因。
先预热，采样轮次至少沿用当前 100 条归因/1000 条基线，并交替重跑候选与 reference，
报告 p50/p95 及总有效字节/总耗时，不能只挑最快轮次。

建议新增每核/每批汇总的 tile 数、输入/输出搬运命令数、事件完成组数、
cache 维护行数、有效字节和算法参数；优先离线推导计数，避免逐 token 时间戳干扰性能。
必要时使用单独诊断变体或目标机 profiler 判断 MTE、标量、同步与缓存瓶颈。
流水版本另记首个 ready、首个 copy、最后 copy、最终 done/返回的时间线，
copy 活跃时间、等待 ready 时间、CPU 通信窗口分别定义，不将重叠耗时相加。

```text
payload_bytes = active_tokens * token_bytes
end_to_end_gbps = payload_bytes / end_to_end_ns
local_scatter_effective_gbps = payload_bytes / local_scatter_wall_ns
cpu_comm_effective_gbps = payload_bytes / cpu_comm_ns
```

多核取共同时间轴的完成跨度，不累加各核时长；如另展示 `2*payload/time`，
必须标为本地读+写逻辑流量，不能称为实测物理 HBM 带宽。
SIMT/cache 命中和实际总线事务会影响物理流量。
流水后 `local_scatter_wall_ns` 需区分是否包含等待 ready，不能沿用旧列名隐式改变含义。

第一阶段只接受完整正确的候选；以 656 B scatter 减半为优化目标，
最终以关闭 profile 的端到端 p50/p95 改善为合入判断。
4096 B 和小请求若明显退化，则按形状分派或保留 reference，不能只优化一个点后全局替换。
本机无目标 NPU/CANN 编译器，本方案中的 API、事件、预算和收益均须在目标机验收。

## 9. API 依据

- [DataCopyPad：Normal padding、UB 32 B 对齐、多块与 stride 单位](https://www.hiascend.com/doc_center/source/zh/canncommercial/900/API/ascendcopapi/atlasascendc_api_07_0265.html)。
- [SIMT 访存合并：相邻 lane 访问同一行连续元素](https://asc.gitcode.com/guide/operator_practice/simt_operator_optimization/memory_access/memory_access_merge.html)。
- [SIMD/SIMT 混合使用 UB 的优化思路](https://asc.gitcode.com/guide/operator_practice/simd_simt_hybrid_optimization/memory_access/ub_memory_access.html)。
- [launch 动态 UB 参数及可用资源查询](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_1813.html)。

Ascend C 的 GitCode 文档为官方开发分支预览，函数命名、支持度和约束以目标机安装的
CANN/asc-devkit 版本为准，沿用仓库现有 VF_CALL 工具链做最小验证。
