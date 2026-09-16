# BatchGet 建链读回与 CPU EID 资源排查

本说明用于排查已复现的现象：4 条活跃 BatchGet Jetty 访问同一个 CPU EID 时吞吐下降，拆到两个 CPU EID 后缓解。沿用现有测试配置，不增加跑分矩阵，也不修改发包、保序或队列参数。

## 1. 开启建链诊断

CPU 和 NPU 两端都需要重新编译对应的 host/runtime 代码：

```bash
# CPU 机器，使用原来的 UMDK/库配置
./build.sh cpu

# NPU 机器，使用原来的 CANN 配置
./build.sh npu
```

在两端启动进程的终端中设置：

```bash
export KVAF_LINK_DIAGNOSTICS=1
```

随后用原来的命令启动 CPU 服务和 NPU benchmark，分别保存完整 stdout/stderr。若有两个 CPU 服务，两者都设置这个环境变量并分别保存日志。NPU 的子进程会继承它。

默认关闭；只有值为 `1` 时开启。查询发生在初始化、创建或导入通信资源时，不进入 BatchGet 的测量循环。不增加累计 CQ 错误留存。

连续吞吐脚本在该开关开启时，额外输出 `BATCH_GET_DFX_PHASE`：每个阶段提交前、全部 worker 完成后的墙钟/monotonic 对照点，并将它们保存到结果的 `diagnostic_clock_anchors`。它们不改变原来的带宽分母，也不在测量循环中查询设备。

CPU 使用编译所选的 `urma_api.h` 类型调用查询接口。请保持头文件、liburma 和 provider 版本配套；构建默认使用的 UMDK 头不一定与机器上额外安装的 CANN/UMDK 库相同。诊断日志包含符号所在 DSO 路径，快照中的进程 maps 可进一步核对实际加载的 provider。

可选 API 缺失、驱动不支持、查询失败会记录为 unavailable/unsupported/failed，不中断正常建链。CPU 构建时检查 TP 属性 API 的声明；旧头没有声明时跳过该项，不自行猜测扩展结构。改变 UMDK 头目录后，应重新配置 CMake；必要时清除 `KVAF_HAVE_URMA_TP_ATTR_API` 缓存项。

旧版头文件没有 JFS `order_type` 时输出 `order_type=unavailable`，同时保留原始 `jfs_flags`。不会把旧 ABI 的保留位解释成新字段，也不会把缺失字段报告为数值 0。

## 2. 日志如何关联

新增日志使用 `LINK_DIAG` 标识。保留旧的 `CHANNEL_CREATED`、`BATCH_GET_TRANSPORT_WORKER`、`PEER_LINK_JETTY_IMPORT_READY` 和 `BATCH_GET_THROUGHPUT_READY`，它们提供 session、stream/core/slot 和测试 rank 的对应关系。

首先把每条 BatchGet 连接整理成一行：

| NPU 侧 | CPU 侧 | 两侧连接信息 |
|---|---|---|
| PID、物理 device、EID、context、role、完整 Jetty ID、CQ、SQ/doorbell | PID、session、UB device、eid_index、EID、context、完整 Jetty ID、JFR、JFC | 各自 import 返回的 TPN、local/peer TP handle、tx/rx PSN |

关联规则：

1. 使用 **EID + UASID + Jetty ID** 识别端点。不要只比较 Jetty 的数字 ID，也不要跨进程比较 context/虚拟地址来判断资源是否相同。
2. NPU 的 local TP handle 对应 CPU 的 peer TP handle，反向同理；NPU 的 tx PSN 对应 CPU 的 rx PSN，反向同理。
3. 对端 TPN 如果未通过控制协议返回，则从对端日志关联获取。**64 位 TP handle 不能当作硬件 TPN 使用或直接传给 `urma_admin ... tp`。**
4. BatchPut 的初始化也会创建资源；按 role 和 `BATCH_GET_TRANSPORT_WORKER` 区分 BatchGet 活跃资源，不能把全部已创建 Jetty 当成测量阶段的活跃 Jetty。
5. 不同 TPN 只证明软件使用了不同的传输对象；是否落到同一个硬件事务资源池、SL 队列或物理端口，需要设备映射/DFX 信息。

查询结果按来源区分：请求配置、创建/导入返回值、query 读回值。特别关注：

- 实际 Jetty 状态、实际队列深度和 priority。NPU 的 Jetty 查询接口仅暴露部分属性；未提供的属性不会被请求值冒充。
- 设备能力里的 `priority_info`：本实现请求 `priority=2`，核对其实际 SL 和支持的 TP 类型。不同设备的映射不必相同。
- RM/CTP、保序配置、TP 属性的有效 bitmap。没有返回的字段不是数值 0。
- EID 到 UB device、GUID、die/function、端口/NUMA 的对应关系；接口没有提供的物理归属仍需驱动工具。

创建成功和属性读回只能定位连接配置，不能直接报告 READ credit 的实时占用。

## 3. 先采集可通用读取的 DFX 基础信息

提供只读快照工具：`tests/collect_urma_dfx.py`。它保存工具版本/帮助、设备信息、支持的通信资源查询、指定进程的库映射和有限的 sysfs 属性。不会清计数器、修改 priority/SL、重建链路或开启设备调试模式。

在 CPU 机器执行一次环境快照：

```bash
python3 tests/collect_urma_dfx.py --output-dir dfx-cpu-environment
```

从 CPU `CANDIDATE_READY` / `LINK_DIAG` 日志找到**实际 UB device 名**，不要用 TCP 网卡名或 EID 字符串替代。进程建立连接后，在资源仍存活时抓取动态资源：

```bash
# 将两个值替换为当前 CPU 服务实际使用的设备名、PID。
CPU_UB_DEVICE='实际设备名'
CPU_SERVICE_PID='实际PID'
python3 tests/collect_urma_dfx.py \
  --output-dir dfx-cpu-live \
  --device "$CPU_UB_DEVICE" --pid "$CPU_SERVICE_PID"
```

`--device`、`--pid` 可以重复，适用于两个 CPU 服务。输出目录必须是新目录，避免把不同时间的资源信息混在一起。NPU host 可用同一工具采集其可见的 URMA 设备和 worker PID，但 NPU Device/HDC 内部资源不保证在 host 的 `urma_admin` 中可见；这部分以 RA 诊断日志和对应设备工具为准。

工具内的资源查询采用当前官方 `urma_admin show dev ...` 语法。旧版本不支持时保留退出码/错误和 `show_res`、`show_stats` 帮助，不猜测旧版 `resource_type` 数字。缺失、不支持或权限不足不代表资源数/错误数为零。默认单命令超时 5 秒，可用 `--command-timeout` 调整。

快照退出码：`0` 为全部采集项成功，`1` 为存在缺失/失败但结果已保存，`2` 为输出目录等致命错误。以 `manifest.json` 逐项状态为准；部分旧版本不支持某些查询时仍可利用其余结果。

**资源快照要在 NPU runtime 关闭前采集。** benchmark 退出后 CPU 服务进程虽然还在，session 的 Jetty/TP 也可能已经释放。手动来不及抓的 Jetty 身份和创建后读回值已保存在应用日志；不要拿下一次运行重新分配的 ID 解释上一次结果。

## 4. 真正的运行期硬件 DFX 怎么采

前面的快照工具不冒充完整的硬件性能采集器。READ credit、事务上下文占用、SL 背压、UMMU 翻译停顿等指标需要目标 CPU UB 设备的驱动/固件工具，公开 URMA 接口没有跨设备统一的读取命令。

按以下顺序操作：

1. **确定对象。** 用建链日志定位 CPU EID 对应的设备，以及本次连接的 Jetty/TPN；结合设备工具定位 UE/function、SL、端口。不要从 EID 的某几位自行推导端口或 die。
2. **确认本机支持的计数器。** 先看快照里的 `urma_admin` 版本和帮助；支持 `show_stats` 时，按本机帮助选择设备及资源类型。若只提供包/字节/错误总数，继续使用该设备配套 DFX 工具获取资源停顿计数。向设备支持方明确索取下表指标及查询命令，并说明运行的是 RM + CTP 单边 READ。
3. **覆盖现有测试阶段。** 启动有时间戳的设备计数采集，再运行原来的 `solo_0`、`solo_1`、`concurrent` benchmark，结束后停止采集。设备若只支持累积计数快照，必须在对应阶段前后读取并做差；整个程序前后一次做差会混入填表、预热、校验和三个阶段，不能归因到 concurrent。
4. **对齐时间。** 设备采样使用的时钟必须明确。benchmark 的 `start_ns/end_ns` 是 NPU host 的 monotonic 时钟，不能直接与 CPU 主机的 UTC 或另一台主机的 monotonic 相减。采用两端校时后的墙钟标记，或让采集工具与阶段边界显式同步。粗粒度采样必须说明阶段边界误差；现有窗口只有数秒，不能把跨阶段样本当成纯 concurrent。
5. **保留原始输出。** 记录起止时间、计数器单位、资源作用域、计数是否读后清零/回绕，以及采集开销。软件查询尽量放在测量区间外；需要运行中采样的硬件统计采用工具推荐的低开销方式。

本版脚本提供的 `BATCH_GET_DFX_PHASE` 可用于时间对齐。取一个对照点：

```text
anchor_mono = (monotonic_before_ns + monotonic_after_ns) / 2
某个 worker 边界的估计 Unix 时间 = unix_ns + (worker.start_ns 或 end_ns - anchor_mono)
```

对照点的采样误差至少要考虑两个 monotonic 读数的间隔；跨机器还要考虑时钟同步误差。用阶段前后的两个对照点核对墙钟是否发生跳变。DFX_PHASE 的打印时间本身不是 kernel 起止时间，也不要替换原始 `start_ns/end_ns` 来计算吞吐。

| 优先获取的指标 | 要核对的问题 |
|---|---|
| READ 事务资源占用、水位、credit/resource stall | 是否在该 UE/function 或更小资源域达到上限；4 Jetty 时等待是否增加 |
| 请求接收数、响应发送数、待处理请求/响应水位 | 积压发生在 READ 接收处理、内存读取还是响应返回阶段 |
| 实际 SL/发送队列利用率、credit 等待、背压和端口流量 | 多条不同 TP 是否仍汇聚到同一个受限调度域 |
| TP/Jetty 重试、超时、恢复、异常计数 | 本次运行是否存在硬件层面的异常处理；不需要为此增加 kernel 累计错误代码 |
| UMMU/TLB/page-walk 停顿、内存访问延迟 | 是否卡在 CPU EID 对应的地址转换或内存访问通路 |

`max_jetty`、SQ/JFC 深度是对象/队列容量，不能当成 READ 硬件在途窗口。单纯看到某个累计计数变大也不足以归因；应比较同一作用域下的增量、时间或请求数归一化值。

**不要把 `urma_admin perf start/stop/show` 当作上述数据面计数采集。** 当前官方帮助将它定义为控制面 DFX 时延跟踪，用于创建/建链等控制操作耗时，不能替代 READ 执行期间的 credit/stall 数据。

## 5. 需要保留的排查材料

- CPU 服务启动到本次 session 释放的日志；NPU 完整建链日志及原 benchmark 结果。
- 两端快照目录，尤其 `urma_admin` 版本/帮助、设备属性、实际 DSO/provider 路径。
- 能获取时：对应 CPU UB 资源的带时间戳硬件计数原始输出、字段定义和采集命令。

只拿到前两项时可以先检查配置不一致、误绑定、priority/SL 选择和资源归属；如果接口不公开事务资源水位，仍需要该驱动/固件的 DFX 才能说出具体是哪项硬件限制，不能从“4 Jetty”倒推一个不存在于日志中的配额数字。

参考：[URMA User Guide，工具与 DFX](https://docs.openeuler.org/zh/docs/24.03_LTS_SP3/unifiedbus/unifiedbus/urma/URMA%20User%20Guide.ch.html)、[URMA API Guide，设备与 TP 属性](https://docs.openeuler.org/zh/docs/24.03_LTS_SP3/unifiedbus/unifiedbus/urma/URMA%20API%20Guide.ch.html)。工具语法、字段和能力以目标机器安装版本为准。
