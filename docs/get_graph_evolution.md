# Get 执行与图模式演进

本文记录请求级执行分支合入 main 后的实现与验证边界。范围是四种 Get：
URMA AIV Direct、UBMEM Direct、AICPU Gather Read、AICPU CPU Push Scatter。
Put 保持同步事务语义。多图和多 Get 节点使用同一 stream 串行执行。

## 1. 合并 main

合入 `de4fd37e02fa86357f71fe4fcbbbecc3e75d1d58`：Scatter 并行描述符构建、
各 Jetty 独立 Ready 消费，以及 AICPU 设备侧 ticket、图捕获约束、状态检查和
kernel profiling。保留本分支的资源 owner / typed view / Invocation、多 session、
canonical 地址转换，以及按输入/输出绑定复用注册的接口。

设备 ABI 不混用：

- `KvafAicpuUmdkControlSessionsV5`（第 5 步增加 QueueProfile；V4 只保留兼容入口）
- `KvafAicpuGatherReadExecutionV3`
- `KvafAicpuCpuPushScatterExecutionV3`

CPU 服务、NPU host 与 AICPU 包一起重编译、安装并启动新进程。CPU 服务使用
main 的线程调度行为，`--gather-affinity` 已移除；性能对比显式固定线程数和模式。

性能入口统一为 `tests/run_remote_cpu_batch_get_perf.sh`：`--backend` 选择四种 Get，
`--execution eager|graph|both` 选择执行方式。当前 Runtime 只有 unified 路径。

独立 C++/Python 功能测试及配套 stub/model 已移除。统一性能入口保留数据预检、
动态输入和图回放校验；它们随目标设备上的性能测试执行。

## 2. Get 驻留实例

每个 Runtime / DDR 代次内，每种已启用 Get 配置首次使用时建立一个实例。
`GetResourceReference` 保留 owner 和绑定的执行状态；`GetResourceLease` 表示本次
调用占用的资源。图节点复用驻留 owner，单独保留节点状态。普通同步调用或成功 Drain 后释放执行占用、清空借用的请求与
表信息，实例继续存在；不重复构造和绑定执行状态。

同一 stream、容量未增长的热调用跳过 Describe 和 Warmup。容量增长或换 stream
重新检查并准备；每次依然验证请求/绑定边界，并在必要时等待上一次完成。
`acquire` 计时在热调用中只包含驻留引用上的轻量执行占用。

输出绑定由具体算法实例保存。URMA AIV Direct 与 AICPU Gather 可各持有一个绑定，
并交替调用；UBMEM / Scatter 仍不要求输出注册。释放空闲算法的绑定不释放另一个
在途算法持有的 Tensor 或 AccessTable。图租约存活期间，Get 方向专用于图；
已有输出绑定只阻止相同实例进入图模式。

Recover 要求目标实例无输出/图绑定，先完成在途操作，再关闭并重建资源。关闭失败
保留实例、状态及资源引用，允许重试；成功 Shutdown 才统一释放它们。

## 3. 四种 Get 的图模式

四个后端均通过 `begin_get_graph(config, output, capacity, stream)` 提前准备资源；
捕获时 `batch_get(..., execution=config, graph_lease=lease, synchronize=False)`
只提交已准备的设备任务。固定配置、stream、输出地址和容量后，可在相同输入存储中
更新 indices、offsets、miss_count、output_indices，重放全量、部分和零请求。
图租约存活时禁止同一 Runtime 的 eager Get、Drain、Warmup、Recover 和 Shutdown。
一个租约仅允许一个 Get 节点，一张图可使用多个租约。Put 不进入图。

AIV Direct 和 UBMEM eager Get 均不在完成时回读或检查设备状态，
UBMEM 同步 Get 也不再清零状态报告。完成路径只等待并释放请求资源，ACL 提交和同步错误仍返回失败。
显式 profile 快照在执行完成后读取并校验报告；UBMEM Put 的状态初始化、事务确认和
Get graph 释放时的状态检查保留各自语义。AICPU Gather/Scatter
使用设备侧 ticket 推进跨 replay 的通信序号；
捕获阶段不注册、不分配、不同步，显式状态检查须在 replay 完成后执行。
未重放的 AICPU 图正常结束会报未完成，可 `discard_result=True` 取消。
同步/注销失败保留资源和租约以供重试。共享 AICPU 会话失败后，先停止回放并销毁
引用该算法实例的所有图，再对其租约调用 `end_get_graph(..., discard_result=True)`。
尚有同实例租约未进入丢弃流程时，返回 Busy 并保留资源。最后一个租约关闭会话后，
重试此前收到 Busy 的租约，再调用 `recover(config)`。

Profiling 和状态查询通过算法配置路由到驻留实例，默认取 Runtime 的默认 Get。
图前启用 profile、每轮完成后读取，销毁图并结束租约后停用。保留最后成功 profile
供结束后读取。AICPU 的 `check_batch_get_status(stream, execution=config)` 检查
实际对应算法；AIV 无需新增这一检查。

更新 CPU、NPU 和 AICPU 包后，通过统一性能入口逐个测量 eager / graph，
使用[统一 Get 性能测试](batch_get_performance.md)中的各后端核数。
`--graph-replays` 控制正式计时前的动态输入、部分和零请求验证；
性能样本数由 `--iterations` 决定。
Host 延迟对照需使用相同的 `--get-timing` 和 profile 设置。

## 4. 多图与多个 Get 节点

在图外为每个节点准备一个租约。下例假设三个节点的输入和输出 Tensor、访问表
及 Runtime 已准备好；`capacity` 是每个节点的请求数组容量。

```python
leases = [runtime.begin_get_graph(cfg, output, capacity, stream)
          for output in (out_a, out_b, out_c)]
graph1, graph2 = kvaf.AclGraph(), kvaf.AclGraph()

def get_node(plan, output, lease):
    indices, offsets, miss_count, output_indices = plan
    runtime.batch_get(table, indices, offsets, miss_count, output_indices, output,
                      execution=cfg, stream=stream, graph_lease=lease)

graph1.capture_begin(stream)
get_node(plan_a, out_a, leases[0])
get_node(plan_b, out_b, leases[1])
graph1.capture_end()
graph2.capture_begin(stream)
get_node(plan_c, out_c, leases[2])
graph2.capture_end()

graph1.replay(stream)
graph2.replay(stream)
graph1.replay(stream)
graph1.synchronize(stream)
# 校验 out_a、out_b、out_c；启用 profiling 时可按节点读取：
# runtime.batch_get_kernel_profile(graph_lease=leases[0])

graph1.reset()
for lease in leases[:2]:
    runtime.end_get_graph(lease, stream)
graph2.replay(stream)
graph2.synchronize(stream)
graph2.reset()
runtime.end_get_graph(leases[2], stream)
```

各节点独立保留输出注册引用、Tensor、访问表和完成报告；同算法节点共用通信会话、
队列、工作区和连续的设备 ticket。回放不增加 Host 调度或节点间同步。
Torch Graph 同样必须在捕获 stream 上回放；Runtime 无法拦截外部图的 replay。
一个算法实例有多个节点时，状态和 profile 查询用 `graph_lease=lease` 指定节点；
报告保存的是该节点最近一次完成的结果。没有跨 stream 并发回放。

## 5. 共享资源和生产目录

本轮把可共用且生命周期明确的依赖从执行会话中提取出来：

| 对象 | 共享身份 / 生命周期 | 保持独立的对象 |
|---|---|---|
| HCCP transport API 表 | 进程首次完整加载成功的 provider；库引用保留至退出 | RaCtx、Jetty/CQ、entity、注册与执行状态 |
| AICPU `ApiEnvironment` | 实际加载的 liburma 对象；多 session 共享初始化与函数表 | 各 session 的 context、MR、远端导入和队列 |
| CPU `CpuUrmaMemoryPool` | 一次 allocation 的代次；按 candidate/EID/context 创建兼容 DDR 注册 | 各执行 session 的通道、Gather/Scatter worker 和控制状态 |
| UBMEM HAL / mapping | 沿用进程级 HAL、带 allocation 代次的设备映射引用 | Put/Get 的 metadata、完成状态和槽位 |

AICPU 的执行入口只解析一次 session；函数表在准备时复制到 session，WR/CQ 热路径
不增加环境查找。最后使用者关闭环境前先释放所有队列、注册和 context；失败保留句柄
以便重试。HCCP 的成功加载引用进程驻留，所有 transport 复用这一份表，避免每次
创建 transport 都增加一份永不释放的 dlopen 引用。首次成功后锁定当前进程 provider；
更换 SDK/库应使用新进程。

CPU 的共享 MR 不再随每个执行 session 注销。同一 allocation、context、candidate/EID、
地址范围、权限和凭据完全一致才复用；context/allocation 代次参与身份，同址新分配不会
命中旧注册。关闭一个 session 只关闭自己的工作与队列并归还引用；Region 关闭时先注销
MR/token，再关 context，最后释放 DDR。任何一步失败保留下游内存与所有权，允许重试。

本轮 CPU wire 协议不变，execution session 身份仍独立。CPU 注册共享不意味着 NPU/AICPU
的远端导入句柄共享：不同本地 context 继续独立导入，也不把 session ID 当作内存绑定 ID。
更大范围的 context、HBM 注册和远端导入共享仍需明确存储代次及协议绑定身份。

AICPU 队列用显式 `UmdkQueueProfile` 配置：Put 为 **1 control / 0 data**，Gather 为
**N control / N data**，CPU 侧通道数与之对应。Put 的提交、完成与事务处理语义不变。
Control 使用 V5 能力检查；在原 4 字节 padding 放 profile，不改变原结构大小或其他字段
偏移。旧 Control/V3/V4 入口保留原 Gather 队列布局，新 Host 不静默退回旧包。

生产实现已进一步按算法迁移到（共享 ABI 和组合编译入口保留在 common）：

```text
access_operator/algorithms/
  get/direct/{urma_aiv,ubmem_aiv,common}/
  get/gather/urma_aicpu/
  get/scatter/urma_aicpu/
  get/common/aicpu/
  put/{urma_aiv,urma_aicpu,ubmem_aiv,common}/
```

跨方向的适配器和资源 owner 位于 `access_operator/providers`，公共 view 位于
`access_operator/include`，传输和映射位于 `access_operator/transports`。Runtime 调度、
CPU 内存及服务分别位于 `runtime/execution`、`runtime/memory`、`runtime/cpu_service`。
CMake、测试、脚本和公开文档路径一起迁移，库名、kernel 符号和 `build-npu/bin` 产物路径
保持不变。AICPU 构建包改在 `build-npu/access_operator/transports/aicpu_urma`，安装脚本
同步使用新路径。此前要求保留的本地设计草稿未修改、未提交。

需重新配置并构建 CPU/NPU，重新安装 AICPU 包，并启动新的客户端和 CPU 服务进程。
统一性能入口在正式计时之外验证数据与 graph 回放。
统一性能入口仍输出单节点图的性能数据；传入 `--multi-graph-check` 可在计时前
验收多图。测试在同一个 stream 上交替回放两张图，其中一张含两个 Get，验证完整、
部分及零请求，并在释放双节点图后再次回放剩余图。每个成功后端打印
`BATCH_GET_MULTI_GRAPH_PASS`。图内混合算法和 Put 图模式不在本轮验收范围。
