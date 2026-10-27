# 单 DDR Region、多逻辑 Pool：文件修改清单

> 状态：实现前清单，尚未修改业务代码。
>
> 目标：一个 RuntimeCore、一块已注册 DDR，由 MetaManager 管理不同 slot 大小的逻辑 Pool 和 layer 绑定；所有 Pool 复用根 region 的数据面资源。
>
> 本清单以当前工作区的 DSA 固定寻址、扁平 BatchPut、`miss_count` BatchGet 接口为依据。

## 1. 本次方案边界

```text
一个 NpuRuntime / RuntimeCore
├─ MetaManager
│  ├─ Pool A：offset、slot_size=128×1024、slot_count
│  ├─ Pool B：offset、slot_size=128×656、slot_count
│  ├─ layer → Pool / 该层 DSA 范围
│  └─ key 状态、事务、pin、地址计算
└─ AccessOperator
   └─ 对整块 DDR 初始化传输资源，按请求地址和布局执行读写
```

不同 slot 大小不新增独立 region allocation，也不按 Pool 创建新的 transport。现有 Put/Get、多 EID 和多 Jetty 资源仍按后端配置存在；“一次建联”表示不随 Pool 数量重复创建同类传输资源。

首版保留按层预留的固定 DSA 范围，不引入动态 key 到 slot 的分配映射。一个请求只访问一个 layer/布局，跨 Pool 混合请求另行拆分。

Root DDR 的数据区、Pool 子区间和系统 workspace 必须分别定义。MetaManager 只能分配数据区，不能覆盖 Gather staging、descriptor 或 control 区域。

## 2. 先固定接口契约

建议新增的类型集中放入 `common/include/pool_types.h`，避免配置和句柄散落：

```text
PoolSpec：pool_id、offset、slot_size、slot_count、token_bytes、tokens_per_block
LayerSpec：layer_id、pool_id、slot_offset、key_capacity
LayerHandle：Runtime 身份、layer 身份、generation
LayerLayout：解析后的 base、slot_size、key_capacity、token/block 布局
```

多模型需要 namespace；首版可将 namespace 作为 LayerHandle 的身份字段或在 Runtime 内配置默认命名空间。句柄应跨 Runtime、销毁重建后可校验失效，不能仅使用一个裸下标。

建议的业务接口改动：

| 接口 | 契约 |
| --- | --- |
| Initialize | 配置总数据容量、Pool 和 layer 布局；保留旧单 Pool 适配入口 |
| 获取 layer | 从 layer_id 解析并返回轻量 LayerHandle，可缓存 |
| BatchPut | 增加 LayerHandle，keys 为该层范围内的局部 DSA key |
| PrepareAccessTable | 增加 LayerHandle，table 内保存所属 layer 和布局 |
| BatchGet，受管理 table | 从 table 取得 layer/布局，无须重复提供 layer_id |
| BatchGet，裸 Tensor | 必须提供 LayerHandle 或同等受管理上下文 |
| BatchFree | 增加 LayerHandle，回收正确的层内记录 |
| ReleaseAccessTable | 仍只需 table handle，保留完成后才能解 pin 的约束 |
| AccessOperator Initialize | 仅绑定根 DDR region，与 layer/Pool 无关 |

`LayerLayout.base` 在配置时预计算：

```text
layer.base = root.base + pool.offset + layer.slot_offset × pool.slot_size
address = layer.base + local_key × layer.slot_size
```

必须检查容量、整数溢出、对齐、子区间重叠、layer 范围重叠、有效长度和输出行宽。基地址注册对齐与 Pool 的 slot 步长是不同约束。

当前 [DSA key helper](../tests/python/dsa_keys.py) 已把 pool row、layer、block_q 编入 key。新增局部 key 接口时必须提供明确的旧 key 解码/适配规则，不能把已编码全局 key 当成局部 key再加一次 layer 偏移，也不能用改变 KeyId 高位而不改变 DSA 元数据索引范围的方式兼容。

## 3. 公共类型、MetaManager 与 Runtime：必须修改

| 文件 | 当前耦合点 | 需要修改的内容 |
| --- | --- | --- |
| 新增 `common/include/pool_types.h` | 无独立 Pool/layer 类型 | 定义 PoolSpec、LayerSpec、LayerHandle、LayerLayout 及必要的初始化配置 |
| [common/include/types.h](../common/include/types.h) | RemoteRegionDesc 同时描述 DDR 和单一 slot_size | 分离根 region 注册范围与逻辑 Pool 规格；slot_size 不再表示整个 DDR 的唯一切分方式，旧字段如保留须明确兼容语义 |
| [common/include/requests.h](../common/include/requests.h) | RuntimeInitRequest、Put/Prepare/Free 无 layer；Get 布局由上层填写 | 初始化增加布局配置；Put/Prepare/Free 增加层上下文；为受管理 table 的布局查询提供接口；保持扁平 input/offsets/lengths 和 miss_count 契约 |
| [meta_manager/include/meta_manager.h](../meta_manager/include/meta_manager.h) | 注册一块按单 slot_size 铺开的 DSA 空间 | 定义注册根数据区及配置 Pools/layers 的接口；Put/Lock/Lookup/Free 接收已解析层上下文；保持 reserve/commit/abort、稳定读和 pin 语义 |
| [meta_manager/src/in_memory_meta_manager.cc](../meta_manager/src/in_memory_meta_manager.cc) | `region_`、`records_[key]`、`base + key × slot_size` | 管理 Pool/layer 配置，预计算基地址；记录按层分段或按层索引；改写 key 边界与地址计算；事务及读锁上下文携带 layer 身份；跨层相同 key 不冲突 |
| [runtime/include/runtime.h](../runtime/include/runtime.h) | AccessTableState 只存地址、长度和 valid | 保留唯一根 region 和唯一 Core；增加配置/解析入口；table 保存 LayerHandle、generation、token/block 布局；增加查询布局的窄接口供 Binding 使用 |
| [runtime/src/runtime.cc](../runtime/src/runtime.cc) | Host allocator 和初始化检查总容量必须整除单 slot_size；调用 Meta 时无 layer | 根内存分配校验与 Pool 校验分开；初始化 Meta 布局；按层编排 Put/Get/Free；从 table 得到 Get 布局并校验归属；失败初始化完整回滚，数据面只初始化一次 |
| [runtime/include/npu_runtime.h](../runtime/include/npu_runtime.h) | token/block 是唯一全局规格，Initialize(ddr_bytes, slot_size) | 增加多 Pool/layer 初始化入口和 layer 查询；全局设备/后端配置与 Pool 布局分开；保留旧单 Pool 重载及查询的明确语义 |
| [runtime/src/npu_runtime.cc](../runtime/src/npu_runtime.cc) | PeerDdrAllocator、workspace、探针、启动调优依赖全局 slot/token | 按根数据区与最大资源需求分配一次注册内存；配置 Meta Pools/layers；重写依赖单 slot_size 的容量推导；保留候选验证、清理及正常 shutdown 的顺序 |

首版可以直接在现有 MetaManager 内保存 Pool/layer 结构，不必再新增独立 PoolManager、第二个 Core 或空闲队列。若布局校验过长，可后续拆出 `meta_manager/src/pool_layout.h/.cc`；这属于代码组织选择。

元数据仍保留 DSA 状态记录，不因为增加 Pool 就回到无界 key 哈希分配模型。层解析每批一次，地址计算使用缓存的 LayerLayout，不逐 key 重复查 layer 配置。

## 4. 远端分配接口：需要联合调整

CPU 服务只负责根 DDR 的分配/注册，不管理模型层或 Pool：

| 文件 | 需要修改或确认的内容 |
| --- | --- |
| [runtime/include/peer_link.h](../runtime/include/peer_link.h) | AllocateRegion 的参数从逻辑 slot_size 解耦；只表达根注册范围及真实分配要求 |
| [runtime/src/peer_link.cc](../runtime/src/peer_link.cc) | 修改客户端校验和请求构造，不能继续要求根数据区按某个业务 Pool 的 slot_size 整除 |
| [runtime/include/peer_protocol.h](../runtime/include/peer_protocol.h) | AllocateRegionRequest 当前含 slot_size；确定删除或保留兼容字段的方案；字节布局或协议语义不兼容时更新版本和静态尺寸断言 |
| [runtime/src/cpu_runtime.cc](../runtime/src/cpu_runtime.cc) | 修改 Region Allocate 与请求分发中的单 slot_size 校验；保留根 region 的内存注册、session/token、通道创建和释放逻辑 |

协议变更需要两端配套。若为了兼容保留旧 wire 字段，应明确旧消息和新消息的识别及校验规则，不能静默忽略协议差异。Pool 规格和 layer_id 不应被加入 CreateChannel 请求。

根注册长度可能大于用户数据长度：

```text
registered_bytes = data_bytes + aligned_auxiliary_bytes
MetaManager 可见范围 = data_bytes
AccessOperator/allocator 保留完整注册资源的生命周期
```

不能为了消除 slot_size 耦合，误把辅助区计入可寻址 DSA key 容量。

## 5. Python Binding 与默认 CPU Mock：必须同步

| 文件 | 需要修改的内容 |
| --- | --- |
| [bindings/src/npu_python_runtime.cc](../bindings/src/npu_python_runtime.cc) | 新配置、LayerHandle、Put/Prepare/Free 参数；DeviceAccessTable 保存并查询 layer/layout；Get 输出行宽校验改用 table/LayerLayout；裸 Tensor 重载补上下文；remote_region 不再返回一个误导性的全局 slot_size |
| [bindings/src/python_module.cpp](../bindings/src/python_module.cpp) | CPU Mock 同步支持多 Pool/layer，使配置、DSA、事务、pin、句柄校验可在无 NPU 环境验证；保留默认单 Pool API |
| [python/kvaf/__init__.py](../python/kvaf/__init__.py) | 如对外导出 PoolSpec、LayerHandle 等类型，更新导出列表；使用 dict 配置时也要统一入口与文档 |
| [bindings/verify_python_binding.py](../bindings/verify_python_binding.py) | 检查新增接口/类型以及旧兼容接口，避免编译后遗漏绑定 |
| [bindings/include/access_table_python.h](../bindings/include/access_table_python.h) | 若 Host view 对外暴露 pool/layer/layout 信息，补充字段；如这些信息完全通过独立查询提供，此文件可不改 |

`bindings/include/npu_python_runtime.h` 当前只声明注册函数。新增类型可以在源文件注册，不应仅为凑齐文件清单而改这个头文件。

## 6. AIV 数据面：以复用和验证为主

| 文件 | 处理建议 |
| --- | --- |
| [access_operator/include/access_operator.h](../access_operator/include/access_operator.h) | 确认根 region 描述和请求类型变更可兼容；不增加按 Pool 初始化 transport 的接口 |
| [access_operator/src/access_operator.cc](../access_operator/src/access_operator.cc) | 核对整个数据区范围检查和每请求 token/block 参数；维持根 region 的 transport 与注册缓存；如当前逻辑已满足，不必修改算法 |
| [access_operator/src/batch_get_args.h](../access_operator/src/batch_get_args.h) | 核对 AICPU 共用参数构造保持使用本次请求的 output_block_bytes，而不是新的全局默认值 |
| [AIV BatchGet ABI](../access_operator/demo/aiv_batch_get/include/batch_get_abi.h) | 已有 token_bytes / tokens_per_block，首版优先保持 ABI |
| [AIV BatchGet transport](../access_operator/demo/aiv_batch_get/host/batch_get_host.cpp) | 已按根地址、长度、token 绑定，Pool 不新增 transport；通常保持实现 |
| [AIV BatchPut transport](../access_operator/demo/aiv_batch_put/host/batch_put_host.cpp) | 已按目标地址和实际长度执行，通常保持实现 |

AIV kernel、通道创建、EID/Jetty 发现和本地 HBM 注册逻辑不因 layer_id 本身增加新的设备参数。Host table 校验必须确保访问属于正确的 Pool 子区间，不能把整个根注册范围的保护误认为已实现 Pool 隔离。

多个 Pool 共用同一 transport 的执行状态时，首版按同一 stream/串行 ExecutionContext 使用。并发 stream 需要独立工作区/队列状态或显式依赖，不能根据 Pool 不同就直接并发复用。

## 7. 启动探针和自适应调优：容易漏改

| 文件 | 需要修改或核对的内容 |
| --- | --- |
| [runtime/src/startup_bandwidth_probe.h](../runtime/src/startup_bandwidth_probe.h) | Put、PrepareAccessTable 和 Get 改为显式内部 layer 上下文，或受控的底层探针路径；不能占用用户 layer/key；清理沿用正确 handle |
| [runtime/include/startup_selection.h](../runtime/include/startup_selection.h) | 现有 StartupProbeConfig 本身已有独立 token/block，可继续作为显式探针布局；核对 BuildStartupProbeLayout 的容量/对齐是否仍适用，不必强制改造成业务 Pool 配置 |
| [runtime/src/npu_runtime.cc](../runtime/src/npu_runtime.cc) | 64 B BatchPut 探针、probe rows 和自适应候选初始化不能继续用全局 slot_size 推算用户层布局；探针配置与业务配置分离 |

如果业务 Pool 要求完整 block 写入，64 B 连通性探针需要明确的内部例外或独立布局，不能直接经过完整 block 校验后失败。

## 8. AICPU 后端：启用时必须适配

这些不是额外创建 Pool transport，而是让现有共享后端支持多种请求长度。

### 8.1 GatherRead

当前 `BatchGetTransport` 在构造时固定 token_bytes，并在 Execute 中要求请求 token_bytes 与之相等；因此不能只改 Binding。

| 文件 | 需要修改的内容 |
| --- | --- |
| [access_operator/include/aicpu_access_operator.h](../access_operator/include/aicpu_access_operator.h) | token_bytes 从唯一业务规格调整为资源容量上限等明确语义，实际布局来自请求 |
| [access_operator/src/aicpu_access_operator.cc](../access_operator/src/aicpu_access_operator.cc) | 按容量配置创建共享 transport，执行时允许多种合法 token 大小 |
| [access_operator/demo/aicpu_batch_get/include/kvaf_aicpu_batch_get.h](../access_operator/demo/aicpu_batch_get/include/kvaf_aicpu_batch_get.h) | 区分容量参数与每次实际 token_bytes |
| [access_operator/demo/aicpu_batch_get/host/batch_get_host.cc](../access_operator/demo/aicpu_batch_get/host/batch_get_host.cc) | 修改固定长度相等校验；按最大容量保留 workspace，按请求填写实际字节数；descriptor/control 的地址在池切换和图回放时保持稳定 |
| [common/include/gather_workspace.h](../common/include/gather_workspace.h) | 明确最大容量与实际搬运量；必要时补资源布局函数，避免每次 token 改变导致控制区偏移改变 |
| [runtime/src/npu_runtime.cc](../runtime/src/npu_runtime.cc) | root DDR 额外 workspace 按已配置的最大容量预留，并使用独立对齐，不能按单一业务 slot 对齐 |

核对 [GatherRead kernel](../access_operator/demo/aicpu_batch_get/kernel/gather_read.cc)、[CPU Gather executor](../runtime/src/cpu_gather_executor.cc) 及控制协议的字节边界；已有请求字段可满足时不修改 kernel ABI。不能仅删除 token 相等校验而不验证 staging 和各 lane 容量。

### 8.2 CPU Push / Scatter

当前 Host 与 CPU 侧都固定 token_bytes，并存在预构造写计划缓存，需一起调整：

| 文件 | 需要修改的内容 |
| --- | --- |
| [access_operator/demo/aicpu_batch_get/include/cpu_push_transport.h](../access_operator/demo/aicpu_batch_get/include/cpu_push_transport.h) | Config 区分最大 token 容量与实际请求规格 |
| [access_operator/demo/aicpu_batch_get/host/cpu_push_host.cc](../access_operator/demo/aicpu_batch_get/host/cpu_push_host.cc) | Initialize 保留足够的稳定 HBM workspace；Execute 接受不同合法 token 大小，不按 Pool 重新打开 session |
| [runtime/src/cpu_push_get_transport.h](../runtime/src/cpu_push_get_transport.h)、[runtime/src/cpu_push_get_transport.cc](../runtime/src/cpu_push_get_transport.cc) | CPU 侧移除唯一长度假设；按请求长度构建 SGE、目标步长和总字节数；预构造计划缓存键包含 token/layout，不能只看 count |
| [common/include/cpu_push_get_layout.h](../common/include/cpu_push_get_layout.h) | 区分稳定容量布局和请求使用长度，Host/CPU/kernel 对同一布局有一致解释 |
| [common/include/cpu_push_get_abi.h](../common/include/cpu_push_get_abi.h) | Request 已包含 token_bytes，优先复用；如更改容量描述或共享结构语义，配套更新版本/断言和两端 |
| [access_operator/demo/aicpu_batch_get/kernel/cpu_push_scatter.cc](../access_operator/demo/aicpu_batch_get/kernel/cpu_push_scatter.cc) | 核对按请求构造 layout、scatter 步长和容量上限；若引入独立容量参数需同步 |
| [runtime/include/peer_protocol.h](../runtime/include/peer_protocol.h)、[runtime/src/cpu_runtime.cc](../runtime/src/cpu_runtime.cc) | 核对 OpenCpuPushGet 的 token 字段从固定长度变为容量时的消息语义与校验 |

`runtime/src/cpu_push_get_worker.cc` 主要驱动 Transport，先核对调用边界；长度计算集中在 Transport 时无需改 worker 算法。reference pipeline 不应仅因 Pool 管理而重写。

如果首批只交付 AIV，应明确 AICPU 多布局暂不开放，并在选择后端时拒绝不支持的配置。不能默认为已经覆盖两个后端。

## 9. 测试与构建清单

| 文件 | 验证内容 |
| --- | --- |
| 新增 `tests/python/test_runtime_pools.py` | CPU Mock 两种 slot/layer；相同局部 key 的隔离；地址步长；配置越界/重叠/溢出；generation；pin/free；旧单 Pool 兼容 |
| 新增 `tests/python/test_npu_pools.py` | 1024/656 B 交替真实读写；整个根 region 只注册一份对应资源；Pool 切换无额外 transport 创建；正确 stream/图生命周期 |
| [tests/python/test_runtime_mock.py](../tests/python/test_runtime_mock.py) | 保留旧 API 回归；必要时适配类型和错误信息 |
| [tests/python/dsa_keys.py](../tests/python/dsa_keys.py)、[tests/python/test_dsa_keys.py](../tests/python/test_dsa_keys.py) | 现有全局编码与新增局部 key 的显式适配，防止 layer 偏移计算两次 |
| [meta_manager/tests/in_memory_meta_manager_test.cc](../meta_manager/tests/in_memory_meta_manager_test.cc) | Meta 接口变更后维护 C++ 用例；该 target 当前默认关闭，启用 legacy 测试时也应可编译 |
| [tests/python/test_npu_batch_put.py](../tests/python/test_npu_batch_put.py) | 当前读取 remote_region.slot_size 的调用需改为指定层布局，或保留明确的旧单 Pool 查询 |
| [tests/python/test_npu_batch_get.py](../tests/python/test_npu_batch_get.py)、[tests/python/test_npu_runtime_perf.py](../tests/python/test_npu_runtime_perf.py)、[tests/python/test_npu_batch_get_perf.py](../tests/python/test_npu_batch_get_perf.py) | 保留单规格基线；多规格用相同请求规模比较，分开统计 Host 管理时间和设备传输时间 |
| [access_operator/demo/aicpu_batch_get/python/test_batch_get.py](../access_operator/demo/aicpu_batch_get/python/test_batch_get.py) | AICPU GatherRead/CPU Push 连续切换 token 大小，尤其 count 不变的 1024→656→1024，检测缓存/步长错误 |
| [tests/CMakeLists.txt](../tests/CMakeLists.txt) | 注册新增 Mock 测试和条件硬件集成测试 |
| [meta_manager/CMakeLists.txt](../meta_manager/CMakeLists.txt) | 仅在拆出实现文件或新增 Meta target 时改；保留 legacy 测试开关 |

`common` 当前为 INTERFACE target，仅新增公共头文件通常不需要修改 common/CMakeLists.txt。Runtime/Binding CMakeLists 只在新增编译单元时修改。

最小正确性验收必须覆盖：两种布局在同一 transport 交替运行、不同 layer 的相同 key、table 归属错误、辅助区越界、异步读取期间禁止回收。管理开销验证使用同样的请求/字节规模，不把不同 token 大小的吞吐差异归因于 Pool 查找。

## 10. 文档和实施顺序

同步更新：

- [meta_manager/README.md](../meta_manager/README.md) 与 [MetaManager 地址设计](meta_manager_address_management_design.md)：多逻辑 Pool、DSA 分层公式、key 兼容规则。
- [runtime/README.md](../runtime/README.md)：一份根注册、布局初始化、LayerHandle、旧入口兼容。
- [bindings/README.md](../bindings/README.md) 与 [python/kvaf/README.md](../python/kvaf/README.md)：新 API、table 自带布局、裸 Tensor 上下文与异步生命周期。
- [KVAF 总体架构](kvaf_architecture_design.md)：区分注册 region 与逻辑 Pool。
- 对应后端 README：分别说明 AIV、GatherRead、CPU Push 的多布局支持范围。

建议按以下顺序实施：

1. 定义配置、LayerHandle、局部 DSA key 与旧 key 的兼容契约。
2. 完成 MetaManager、多层请求和 CPU Mock 的最小闭环。
3. 联合调整根 DDR 分配协议/校验，隔离 data 与 auxiliary 范围。
4. 适配 NpuRuntime、Binding、启动探针，验证 AIV 共享传输资源。
5. 若交付范围包含 AICPU，完成容量与实际请求布局解耦，再分别验证两种 AICPU 路径。
6. 对照旧单规格基线验证管理开销，并补文档与兼容回归。

本方案无需按 Pool 复制 AccessOperator、PeerLink 或 RuntimeCore；主要工作集中在 MetaManager 的层级布局、请求/句柄身份，以及现有全局 token/slot 假设的拆分。
