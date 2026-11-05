# Gather 阶段计时与 AIV 内部流水

此入口只测试 `gather_read`。CPU 仍整批 gather 后 ACK；当前版本在 ACK 后加入
AIV 内部的 WQE/SQ 流水，不加入 CPU chunk 流水或软件在途预算。
功能回归仍使用 `tests/python/test_aiv_gather_batch_get.py`；性能采集使用
`tests/python/test_aiv_gather_batch_get_perf.py`。

当前 gather 与 direct 一样，关闭 AIV 的逐请求 `ValidateRows` 校验；输入由
调用方保证合法。SIMT 输入不再执行全量 Scalar Flush；只保留 Scalar 请求数、
队列/ACK 的必要缓存维护，大 token 回退仅 Flush 本 lane 的输出行映射。基础参数和请求数检查仍保留，六阶段打点
边界不变，因此 `input_us` 不会归零。当前 SGL 上限扩展到 13，需要同步重编
NPU Runtime/Host/Python 与 gather kernel；CPU 服务和 wire 协议保持不变。

## Gather HBM 页策略与 payload 路径

Host 初始化时，`workspace` 和 `queues` 使用 `ACL_MEM_MALLOC_HUGE_FIRST`，
优先申请大页，大页不足时允许 Runtime 回退到普通页；不使用强制大页策略，
也不为了触发大页而扩大逻辑申请尺寸。这是 CANN 的 Device 内存分配策略，
不是 Linux CPU DDR 的 THP 合页机制。

按 [CANN 页策略说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_1349.html)，
`HUGE_FIRST` 在申请量不超过 1 MiB 时仍分配普通页，超过时优先使用 2 MiB
大页。因此本次修改表示大页优先，不能据此宣称实际分配一定为大页。
`workspace` 为最大 65536 请求预留约 1 MiB 的清单和 scratch；`queues`
每核只有两个 64 B 软件队列状态，1～8 核为 128～1024 B，仍可能使用普通页。
本次修改需重编 NPU Host/Runtime 并重建 Runtime 实例，CPU 服务和 kernel ABI
不变；本地生命周期测试不验证真机物理页大小或带宽收益。

当前 `gather_read` 中，各缓冲区职责如下：

| 缓冲区 | 内容与用途 |
| --- | --- |
| HBM `workspace` | 每 lane 64 B scratch（sequence/ACK/command）和源地址清单；不存 token payload |
| HBM `queues` | 每 lane 控制/数据队列的 PI、CI、初始化和错误状态；不是实际 SQ/CQ 缓冲区 |
| AIV UB | 先生成描述清单，再复用为 READ WQE 双 bank；不缓存 payload |
| HCCP 分配的 SQ/CQ | 实际 WQE 和完成记录；本次不改变其分配策略 |
| CPU DDR staging | CPU 按请求顺序将源 token 复制到连续区间 |
| 调用方 output HBM | 数据 READ 的最终目的地址，由调用方/torch_npu 分配并由 Host 注册 |

执行时，AIV 从地址表、indices 和 offsets 计算每条源地址，在 UB 生成清单，
经 MTE3 写入 workspace，再用控制 WRITE 发送到 CPU DDR。CPU 根据清单将
真正的 payload memcpy 到 DDR staging，完成后发布 ACK。AIV 观察到最终 ACK
后构造 READ WQE：远端源为 `staging + i * token_bytes`，本地目的 SGE 为
`output + output_indices[i] * token_bytes`。WQE 从 UB 经 MTE3 写入实际 SQ，
NIC 按 WQE 将 DDR staging 的 payload 直接 READ 到 output HBM。

因此 payload 路径是“CPU 源 DDR → CPU staging DDR → output HBM”，不经过
HBM workspace、软件 queues 或 UB。以 8192 × 4096 B 为例，payload 为 32 MiB；
单 lane 的有效描述清单只有 `48 + 8192 * 16 = 131120 B`。SGL=6 时，完整
READ WR 从 staging 连续读取 24 KiB，直接写入 output 的六个指定 4096 B 行，
没有额外的 AIV payload scatter 拷贝。workspace 的大页策略只直接影响清单/
控制区访问，不能替代 output HBM 页策略的验证。

## 控制通知合批

`publish` 将源清单的分段 WRITE、command 和 notify 两条 8 B WRITE 合批，
在 SQ 容量允许时只敲一次 doorbell、等待一次有序尾 CQ。notify 尾 WQE 使用
strong place order 和 completion，保留清单/command 先于 notify 的可见性约束。
正常队列下，publish 从两次提交/等待减少为一次；小 SQ 仍按 WQEBB/WR 实际
容量提前 drain。CPU gather、最终 ACK 和 payload READ 的协议不变。

每 lane 的 64 B scratch 在通知前一次写好 sequence（+0）和 descriptor 地址
（+16），对所在缓存行执行一次 clean/invalidate；两个 WRITE 不再复用同一源槽。
ACK 仍使用 +8，查询 CQ 完成后的缓存失效继续保留，避免读取旧值。
`notify_done_cycle` 仍记录在通知尾 CQ 完成之后，六阶段打点边界不变。
本次只需重编部署 gather kernel，Host/CPU 协议与 ABI 不变；收益和 DMA 可见性
需在目标 NPU 上复测。

## 地址清单的 SIMT 构造与批量搬运

`input` 不再 Flush `valid_table`、地址表、indices、offsets 或全量 output_indices。
这些数组在快速路径中由 SIMT 读取，Scalar DCCI 不能代替 SIMT 的缓存一致性；
访问方式与 direct 保持一致。Scalar 读取的请求数仍失效一条缓存行；超过 64 KiB
WR 的 Scalar 回退仅在进入回退后 Flush 当前 lane 的有效 output_indices 范围。

`descriptor` 不再由 Scalar 逐条写 HBM 后遍历整份清单 Flush，改为：

1. 128 个 SIMT 线程分步并行读取 indices/offsets/address_table，在 UB 中生成
   源地址、length 和 reserved；48 B 头部也在 UB 内生成，协议布局不变。
2. 两个 24 KiB UB bank 交替使用，SIMT 构造下一块时，MTE3 将上一块批量写入
   本 lane 的 HBM 清单。首块最多 1533 条 span，后续整块最多 1536 条。
3. 最后一块向上补齐到 64 B，仅写本 lane 已分配的对齐空间；发送 CPU 的字节数
   仍是 `48 + 16 * 请求数`，不包含补齐字节。
4. 等待最后一次 MTE3 完成后才结束 `descriptor` 阶段、进入控制 WRITE 发布。
   HBM 清单全程由 MTE3 写入，没有 Scalar 缓存脏行，不再执行整份清单 Flush。
   全部 bank/event 在返回前回收，后续 READ 阶段复用同一组 UB。

队列实体检查、每条 span 的 length/reserved、CPU gather 和 ACK 协议均保持不变。
`descriptor_us` 仍包含队列准备、清单生成和可供网卡读取前的完成等待，六阶段
口径不变。8193 请求、1 核时，清单分为 6 次 VF 构造和 6 次 MTE3 搬运。
本地模型验证内容、边界和同步；实际收益和设备缓存可见性需在 950DT 上复测。

只更新清单构造时可以单独重编 gather kernel；本轮 SGL 上限扩展请使用
“构建与运行”中的完整 NPU 构建命令，同步更新 Runtime/Host/Python。

## 本地 SGE 上限扩展到 13

根据本次 NPU 侧查询的 `max_jfs_sge=13`，软件支持范围扩展为 1～13。
API 和脚本默认仍为 6；真机测试时修改脚本顶部 `GATHER_SGL_SIZE = 13`，
也可依次测试 5、9、13。每条完整 WQE 分别占 128、192、256 B。

READ 每个 UB bank 按最大 13 SGE 预留 32 KiB（128 WR × 256 B），两 bank
共 64 KiB。Scalar 控制/大 token 回退编码区相应扩到 64 个 uint32，所有
Runtime/Host/kernel/Python 上限同步调整。实际 WQEBB 搬运量仍按所选 SGL
计算。清单继续使用 24 KiB 分块，两阶段先后复用 UB，避免改变 descriptor
构造粒度。单 WR 的 64 KiB 限制保持不变；SGL=13 时，token 不超过 5041 B
使用 SIMT 快速路径，更大 token 走 Scalar 拆分路径。

1 核、8192 或 8193 请求、656 B/token、SGL=13 时，预期 631 条 READ WR、
2522 个数据 WQEBB、5 次数据 doorbell；标准队列容量下等待一次数据尾 CQ。
本地模型覆盖全部 1～13 SGE、所有尾项长度、256 B WQE 环回、小 SQ 的容量
drain、1/3/8 核和 64 KiB 边界两侧的快速/Scalar 路径，实际编码兼容性和性能
仍需在目标 NPU 上复测。

## 参数在哪里修改

直接编辑性能脚本顶部的配置区：

```python
TOKEN_BYTES = 656       # 每条 token 的字节数，不是请求数
REQUEST_COUNT = 8192   # 每批有效请求数：可改为 1024 / 8192 / 8193 / 65536
WARMUP = 20            # 不计入结果的预热次数
ITERATIONS = 100       # 关闭 profile 的端到端性能采样次数，0 可跳过
PROFILE_SAMPLES = 10   # 开启 profile 的阶段诊断采样次数
CSV_OUTPUT_DIR = "gather-perf-results"  # 相对执行命令时的工作目录
RUN_LABEL = "aiv_pipeline"  # 串行基线用 serial，当前版本用 aiv_pipeline
```

请求数、token 长度、预热和采样次数不需要新增启动参数。核数仍通过
`KVAF_NPU_CORE_COUNT=1/3/8` 设置。源码区还可配置源表规模、每个 block 的
token 数、SGL 和随机种子；比较不同实现时保持这些值一致。

NPU 输出布局可通过环境变量 `KVAF_GATHER_OUTPUT_LAYOUT` 切换，默认仍为 `random`：

| 值 | NPU 输出布局 |
| --- | --- |
| `random` | 原有随机排列，每个 token 的落点离散 |
| `segmented` | 每条 WQE 内的本地地址依次加 token_bytes，完整段之间随机排列；每核不足 SGL 的尾段保留原位 |
| `contiguous` | 全批输出索引递增，跨 WQE 也连续 |

```bash
DEVICE_ID=4 KVAF_NPU_CORE_COUNT=2 KVAF_GATHER_OUTPUT_LAYOUT=segmented \
  python tests/python/test_aiv_gather_batch_get_perf.py
DEVICE_ID=4 KVAF_NPU_CORE_COUNT=2 KVAF_GATHER_OUTPUT_LAYOUT=contiguous \
  python tests/python/test_aiv_gather_batch_get_perf.py
```

三种布局保留相同的随机 CPU 源请求、数据量、保护行和通用 SGL 编码。
完整段可跨 lane 打乱，但不会重叠或越界；分段布局要求 token_bytes × SGL ≤ 65536，
与当前 SIMT READ 分组一致。启动日志和所有 CSV 均记录 `output_layout`。
8192 请求、2 核、SGL=6 时，按每核 4096 个请求分组，各自最后一条有 4 个 SGE。
只需更新 Python 脚本，不需要重编 Host/kernel 或更改 CPU 服务。

### 合并连续目的 SGE 的对照实验

`KVAF_AIV_GATHER_COALESCE_SGES=1` 开启实验，默认关闭。
仍保留 `GATHER_SGL_SIZE=6` 作为每条逻辑 READ 的 token 数，
将连续的 6 个 4096 B 目的 SGE 编码为一个 24576 B SGE。
READ 数、数据量、每批发布的 WR 数和尾 CQ 等待方式不变，WQE 从 192 B 减至 64 B。
尾条长度按实际 token 数计算；2 核、8192 请求时，尾条是一个 16384 B SGE。

本次需要同步重编并部署 N 侧 Host/Python 和 gather kernel，CPU 服务无需修改：

```bash
cmake --build build-npu -j
```

保持脚本 `TOKEN_BYTES=4096`、`REQUEST_COUNT=8192`、`GATHER_SGL_SIZE=6`，
在同一个新构建上比较关闭与开启；也可将布局换成 `segmented`：

```bash
DEVICE_ID=4 KVAF_NPU_CORE_COUNT=2 KVAF_GATHER_OUTPUT_LAYOUT=contiguous \
  KVAF_AIV_GATHER_COALESCE_SGES=0 python tests/python/test_aiv_gather_batch_get_perf.py
DEVICE_ID=4 KVAF_NPU_CORE_COUNT=2 KVAF_GATHER_OUTPUT_LAYOUT=contiguous \
  KVAF_AIV_GATHER_COALESCE_SGES=1 python tests/python/test_aiv_gather_batch_get_perf.py
```

每核 `data_wrs` 应均为 683，`data_doorbells` 均为 6，
`data_wqebbs` 从 2048 变为 683。启动日志和所有 CSV 记录 `coalesce_sges`。
开启后，脚本在计时外采集一次 profile 核对 READ/WQEBB 数并校验全部 payload；
旧 Host 忽略开关会被计数检查识别，不会作为成功的合并实验继续采样。

合并只支持 token_bytes × SGL ≤ 65536，调用方须保证每条逻辑 READ 内的
`output_indices` 严格依次加 1；内核不额外扫描校验这一前置条件。
性能脚本仅允许在 `segmented/contiguous` 布局下开启。
使用离散保护行的功能/前缀探针脚本会拒绝该开关，正确性由性能脚本的预检验证。
参数结构仍为 V3 / 232 B，复用原来的末尾 reserved 字段为 `coalesce_sges`；
旧 V3 kernel 会拒绝非零的该字段。普通关闭路径保持原有编码。

当前测试设备已确认为 Ascend 950DT，`PROFILE_CLOCK_MHZ` 默认设为 `1000`，
脚本和 CSV 自动将 AIV cycle 换算成微秒，并保留原始 cycle。更换设备时，
须按对应平台的 `GetSystemCycle()` 计数器频率调整，不能使用 AI Core 主频。
若尚未确认新设备的频率，可设为 `None`，此时只输出原始 cycle。

### 如何确认 clock_mhz

这里需要的是 GetSystemCycle 系统计数器的频率。先在目标机确认芯片型号：

```bash
npu-smi info -m
npu-smi info -t board -i 7
```

`-i 7` 替换为查询结果中的实际 NPU ID，不能只凭 Python 的可见设备序号
猜测物理 ID；部分平台的 board 查询还需按工具帮助指定 chip ID。
然后按 [CANN 9.1 GetSystemCycle 官方说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/tool_interface/system_resources_and_variables/GetSystemCycle_ISASI.md)
确认该型号的换算基准：

| 型号 | PROFILE_CLOCK_MHZ | 换算 |
| --- | --- | --- |
| Ascend 950PR / Ascend 950DT | 1000 | us = cycles / 1000 |
| Atlas A2 / Atlas A3 训练、推理系列 | 50 | us = cycles / 50 |

本项目当前使用 A5/3510 构建路径，已确认的 950DT 设备对应脚本顶部默认值
`PROFILE_CLOCK_MHZ = 1000`，重新执行 Python 脚本即可输出 read_us，
不需为换算重编 kernel。不要把 npu-smi 的 AI Core 运行频率当成此参数。
例如已采集的 `read_cycles=3031076.4`，在 950PR/950DT 上为
`read_us=3031.0764`，约 3.031 ms。计数器频率由平台/API 定义确认，不通过
Host 的同步等待时间倒推。

## CSV 输出与后续对比

每次运行结束自动创建一个测试目录，将四份 CSV 放在一起，并打印
`GATHER_CSV_FILE path=...` 的绝对路径。目录按北京时间（Asia/Shanghai、
UTC+08:00）命名，包含年月日、时分秒、核数、token bytes、request-count：

```text
gather-perf-results/
  20261031_093235-cores1-token-bytes656-request-count8192/
    summary.csv
    e2e.csv
    profile_host.csv
    profile_core.csv
```

同一秒、同一配置重复运行时，目录自动追加 `_2`、`_3`，不会覆盖或混入旧
测试。CSV 使用 UTF-8 BOM，可直接在 Excel 中导入。默认结果目录不进入 Git。

| 文件 | 内容 |
| --- | --- |
| `summary.csv` | 本轮汇总，通常先看这一份 |
| `e2e.csv` | 关闭 AIV/Host 打点的逐次端到端样本 |
| `profile_host.csv` | 开启打点后的逐次 Host 时间、外层调用时间和 sequence |
| `profile_core.csv` | 每次采样、每个核的原始时间戳、六段耗时和计数 |

四份文件均包含配置和 run_id：label、代码 revision/dirty、设备/核数、请求数、
token 字节数、SGL、表规模、seed、预热/采样次数、计数器频率、kernel 路径
和 CPU 服务地址。远端 CPU 日志开关仍记作 external，不假定它已经关闭。

summary 按 `row_kind` 区分：`e2e_summary` 是无打点性能基线，
`profile_e2e_summary` 是有打点且含状态/profile D2H 的同步调用统计，另有
`profile_host_mean` 和各核的 `profile_core_mean`。通过 run_label、cores、
request_count、token_bytes、timing_contract 等列筛选对照，避免把不同核或不同采集口径相加。
无打点组使用 `async_call_to_stream_completion`；打点组使用
`synchronous_call_with_status_check`。二者均等待本批完成，但后者额外回收诊断
数据，差值包含 D2H 和校验成本，不能全部归为设备计时指令的开销。

终端 `GATHER_PERF_RESULT` 和 summary 中 `e2e_summary`、`profile_e2e_summary`
的 `useful_GiBps` 均按各自样本的 `p50_us` 计算：
`request_count * token_bytes / (p50_us / 1e6) / 1024**3`。
终端同时输出 `bandwidth_basis=p50_us`；mean/p95/p99 继续保留。
`e2e.csv` 的逐样本带宽仍使用该样本自己的 `e2e_us`；Host/core 阶段汇总
仍为均值。CSV 列名、列顺序和时间单位不变，旧结果的带宽不会自动重算。

列顺序按字段名固定：`clock_mhz`、`clock_source`、`cpu_profile`、`kernel_path`、
`put_kernel_path`、`request_plan`、`timing_contract` 放在 `mode` 前；summary 的
`total_us` 后紧跟 input/descriptor/publish/wait_cpu/read/finalize 六列 us，
随后是 input/descriptor/publish/wait_cpu/read/finalize/total 七列 cycles，
再放 `sampling_status`、`core`，然后 `error_type`、`error_message`。
移动后部分 Excel 列字母会变化，整理数据时建议按表头名称定位。

summary 的 Host 和 AIV 行共享表头，但各自只填自己的指标：

| row_kind | 读取哪些时间 | READ 时间在哪里 |
| --- | --- | --- |
| e2e_summary / profile_e2e_summary | mean_us、p50_us 等整批统计 | 已包含在整批时间中，不额外拆阶段 |
| profile_host_mean | prepare_us、launch_us、status_enqueue_us、wait_us、finish_us、total_us | Host 只观察同步等待，没有独立 READ 计时 |
| profile_core_mean | input / descriptor / publish / wait_cpu / read / finalize / total 的 cycles 和 us | read_cycles，频率核实后有 read_us |

例如 core total 约 190 us、Host total 约 234 us、Python E2E 约 287 us 时，
Host 与 core 间的约 44 us 包含 Host 参数准备及输出 buffer acquire、launch、
设备调度、kernel 打点区间外的 profile 初始化/
导出与退出、状态/profile D2H、同步返回及 Host 收尾。Host 提交与设备执行可
重叠，不能将各项调用耗时简单相加。多核应按每个样本的最早 kernel_start
到最晚 kernel_finish 计算设备区间，不能用各核 total 的平均值替代整批时间。
Host total 与 Python E2E 的约 53 us 则还覆盖绑定层、Tensor 元数据提取、
stream 解析及进入 Host profile 前的检查等工作。以上是计时范围解释，
不是对各项耗时的实测分摊；应使用同一 profile 样本比较。

`profile_host_mean` 行的阶段 cycles 列应为空。`wait_us` 是 Host 的剩余同步
等待，包括设备执行和状态/profile D2H；AIV 的 `wait_cpu_cycles` 只覆盖通知
完成到最终 ACK，`read_cycles` 从最终 ACK 到数据 CQ 完成。READ 内又有
`read_build_cycles`（WQE 发布路径）与 `data_cq_wait_cycles`（数据 CQ 等待）
等子计数。串行基线中的 build 是逐条编码、写 SQ 和 flush 的累计耗时；
当前流水版的 build 是各物理 wave 从启动首批 VF 到发布最后一批 doorbell 的
墙钟时间，包括 VF/MTE3 的必要等待，不包含 wave 尾部的 CQ 等待。
build 期间 NIC 已在执行前面发布的 READ；CQ wait 只反映发布结束后剩余的
完成等待，不能当成网络传输总时间。Host 等待、AIV 阶段、READ 子计数存在
包含关系，不能重复相加，也不能把重叠的 VF/MTE3/NIC 工作分别相加。
当 `clock_source=unverified` 时，AIV 的 read_us 等时间列留空，read_cycles
仍在 profile_core_mean 行输出。这表示尚缺换算频率，不表示漏采 READ。

CSV schema_version 为 3，区分异步基线和同步诊断的 timing_contract。
所有耗时列以 `_us` 命名，不再导出 `_ns` 列；
Host 的内部 ns 在导出时除以 1000。AIV 保留原始 cycle 供诊断，六阶段、
构造及 CQ 等待均有对应 `_us` 列；频率未核实时这些列留空，不猜测频率。
测试开始时间仍以带时区的日期文本保存，便于对应目录。
CSV I/O 在测量循环之外进行，不计入样本时间。
开启打点时的外层调用计时包含本次同步和状态/profile D2H，快照的 Python
复制在调用计时之外；与无打点组比较时需考虑诊断 D2H 和状态校验的差异。

如果采样中途报错，已完成的样本仍会输出，summary 的 `sampling_status`
行为 failed 并保留异常；没有采到的阶段文件只有表头，不伪造零耗时样本。
sampling_status 只描述采样阶段，随后 Runtime shutdown 的异常仍单独报告。

## 构建与运行

沿用第一阶段的 CANN/Python/CPU 地址环境，在仓库根目录执行。
当前 gather 使用含可选诊断字段的 V3 ABI，loader 要求
`KvafGatherBatchGetAivKernelV3`，旧 V1/V2 kernel 会加载失败。
本次必须同步更新 Host/Python 和 gather kernel，不能只重编 kernel。
scatter 和 direct 的 kernel ABI 不变。

```bash
# NPU：更新 Host、Python binding 和 gather kernel
cmake --build build-npu -j
export PYTHONPATH="$PWD/build-npu/python"
export KVAF_RUN_NPU_TESTS=1
export KVAF_CPU_SERVICE_ADDR=CPU_IP:28999
export KVAF_BATCH_GET_KERNEL="$PWD/build-npu/bin/kvaf_gather_batch_get_kernels.o"
export KVAF_BATCH_PUT_KERNEL="$PWD/build-npu/bin/kvaf_execute_batch_put_kernels.o"
DEVICE_ID=7 KVAF_NPU_CORE_COUNT=1 python tests/python/test_aiv_gather_batch_get_perf.py
```

CPU 线程总预算至少等于 AIV 核数，继续使用同一个 CPU 服务。需要 CPU 阶段
诊断时，重编并在启动 CPU 服务前设置：

```bash
cmake --build build-cpu -j
KVAF_GATHER_PROFILE=1 ./build-cpu/runtime/kvaf_cpu_service \
  --listen 0.0.0.0 --port 28999 --gather-threads 8
```

CPU 日志开关在 worker 创建时读取。记录在最终 ACK 后写入预分配缓冲，
正常停止/释放 lane 时统一打印，每 lane 最多保存 512 条请求并报告 dropped。包含
`session/lane/sequence`、rows、bytes、实际线程数、`prepare_ns`、`copy_ns`、
`ack_ns`、`total_ns` 和结果。`copy_ns` 包括本 lane 所有 helper 完成等待；
`total_ns` 包含准备、拷贝及控制开销，不与子项重复相加。
新增每线程启动延迟、拷贝/barrier、线程 CPU 时间和 coordinator join 等待，
采集命令、通知窗口的限制及字段说明见
[Gather CPU 性能采集](aiv_gather_cpu_profile.md)。

CPU 日志用于诊断，会干扰性能。正式端到端基线关闭此环境变量；需要解释
CPU 成本时另跑一轮诊断，不将开启 CPU 日志的结果作为无打点吞吐。

## AIV 内部流水版

这次将三项优化合在同一版本中，CPU 协议、每核一个控制 Jetty 和一个数据
Jetty 和六阶段计时边界保持原样。数据仍由 NIC 从 CPU staging
直接 READ 到离散输出，UB 只存 WQE，不存 payload。

1. 每次最多发布 128 条 READ WQE。发布后继续准备后续 WQE，直到物理 SQ
   WQEBB 容量或传输 WR 上限，再等待该 wave 的有序尾 CQ；整批末尾也等尾 CQ。
   128 是发布粒度，不是在途预算，绝不每 128 条强制等一次 CQ。
2. 128 个 SIMT 线程分别构造一条含 1～13 个 SGE 的 WQE，连续写入 UB。
   MTE3 批量搬到 SQ，环形队列跨尾时分两次搬运。PI、owner 和复制长度均按
   实际 64 B WQEBB 计算；尾部不足 SGL 的 WQE 也按实际长度紧密排列。
3. READ 的两个 UB bank 各 32 KiB：VF 构造下一批时，MTE3 搬上一批；搬运完成后
   Scalar 按 direct 的三阶段调度发布前两批的 doorbell：MTE(n-1)、VF(n)、
   commit(n-2)，尾部再排空剩余提交。UB 等 MTE3 完成即可复用，SQ 空间等尾 CQ
   才能复用。尾 WQE 的 completion/order 位在发布前设定，不修改已发布 WQE。

快速路径条件为 `TOKEN_BYTES * GATHER_SGL_SIZE <= 65536`，覆盖当前
656 B、SGL=6/13。超过该范围时保留原有 Scalar 拆 token 路径，支持一条 token
跨多个 WR；这部分暂未做 SIMT 优化。

例如 1 核、8193×656 B、SGL=6、标准 SQ 深度时，仍是 1366 条 READ WR、
4097 个 WQEBB，发布 11 次 doorbell，只消费 1 个数据尾 CQ。较大批次按
物理容量分 wave，不引入额外的软件窗口。

重新编译并部署 `kvaf_gather_batch_get_kernels.o` 后，仍用上面的启动命令和
CSV 目录格式。当前 Host/kernel 参数 ABI 为 V3，232 B args / 192 B profile，
无需更新 CPU 服务。不要只更新 Python label 而继续加载旧 kernel。

```bash
cmake --build build-npu -j
```

与串行版对比时，不需要分三步单独跑：保存旧版本的 CSV，固定 token、
request-count、SGL、seed、CPU 线程数和设备，分别对 1/3/8 核运行新版本。
先对比 `profile=off` 的 mean/p50/p95/p99、useful_GiBps，再对比各核
`read_us`、`total_us`、`read_build_us` 和 `data_cq_wait_us`。
多核以最慢 lane 的 read/total 解释整批延迟，不把各核时间相加。
`data_wrs`、`data_wqebbs` 应与基线一致，`data_doorbells` 增多是提前发布的
预期行为。实际收益需在 950DT 上测量，本地 CPU 模型只能验证协议与同步。

## 流水版首次调用报 507015 的复测

目前反馈为 1 核、sequence=1 在 Host synchronize 返回 507015，尚无设备侧
PC/错误位。这个日志不能区分异常来自标量、SIMT 还是 MTE，也不能把
shutdown 同步失败当成第二个独立原因。

编码器已移除运行期索引的线程私有 `words[48]`，改为各线程直接写自己的
UB WQE 区间，并展开最多 6 个 SGE 的写入，避免额外的私有临时数组。
这是针对新代码差异的候选修正，尚未证明私有栈就是此次异常的根因。
分批发布、双缓冲、队列容量限制和 CPU 协议保持原样。

为在只有 Host 错误码时仍能推进定位，V3 增加四个可选前缀检查点。普通
运行默认关闭；Host 同步成功并核实本轮 sequence/result/phase 后才报告完成。
这些检查点不会被当成普通 BatchGet 成功或性能样本。

| stage | 实际完成的范围 | 数据输出 |
| --- | --- | --- |
| ack | 输入缓存处理、清单、控制 WRITE、CPU gather 和最终 ACK | 全部保持 guard |
| wqe | ack + 首批 SIMT WQE 构造，并等待 VF→Scalar 完成；跳过数据 SQ 的 MTE3 配置 | 全部保持 guard；未写 SQ |
| sq | ack + 首批 VF 与 UB→SQ MTE3 完成 | 全部保持 guard；未敲数据 doorbell |
| first_read | ack + 每个非空 lane 首批最多 128 READ WR，等待有序尾 CQ | 核对已读前缀及其余 guard |
| full | 关闭所有探针，运行原有完整输出回归 | 核对全部数据、guard、批次/stream 复用 |

wqe/sq 检查点验证相应执行单元已完成，不单独声称 WQE 内容已逐字校验；
first_read/full 进一步验证实际 payload。前缀不留下已发布但未完成的 READ。
输入为空不会被 Host 当成有效诊断；探针禁止异步调用和性能 profile。
当前 ack 以前已包含清单的 SIMT/MTE3，因此 wqe 仅跳过后续数据 SQ 的 MTE3；
若 ack 尚未通过，不能根据 wqe 失败将故障直接定位到 READ WQE 构造。

沿用原来的 kernel/Python/CPU 服务环境，更新并完整重编后执行一个命令：

```bash
git pull --ff-only
cmake --build build-npu -j
DEVICE_ID=7 KVAF_NPU_CORE_COUNT=1 python -u tests/python/test_aiv_gather_pipeline_debug.py
```

debug 脚本读取性能脚本顶部的 TOKEN_BYTES、REQUEST_COUNT、GATHER_SGL_SIZE，
按 ack → wqe → sq → first_read → full 启动独立子进程，只做功能检查，
不采每一步的性能收益。首个失败后立即停止，打印
`GATHER_PIPELINE_DEBUG_STOP stage=...`；提供该行及其前面的 traceback 即可
缩小实际执行范围。全部完成才打印 `GATHER_PIPELINE_DEBUG_PASS`。
加载日志应包含 `symbol=KvafGatherBatchGetAivKernelV3 args_bytes=232`。
诊断脚本的源数据使用功能回归的确定性字节模式，不作为吞吐对比工作负载。

### initialize 报 load sequential kernel

如果日志停在 `phase=initialize`，异常为 `load sequential kernel`，说明 gather
尚未启动。此时 `GATHER_PIPELINE_DEBUG_STOP stage=ack` 只表示第一个子进程
失败，不能理解为已经执行到 ACK 轮询。多个 peer/EID 重试也不会改变同一个
本地 kernel 文件的加载结果。

探针入口会打印实际 kernel 绝对路径、文件长度、扫描到的 V1/V2/V3 名字
标记，以及 Python package / native extension 的实际导入路径。名字标记仅
辅助识别旧产物，不代替 ACL 的 binary/function 校验。
loader 失败时打印 `GATHER_KERNEL_LOAD_ERROR`，包含 path、目标 symbol、
phase 和可取得的 ACL 返回码，重复 EID 尝试只打印一次相同错误：

| phase | 失败位置 |
| --- | --- |
| read_file | 文件读取 |
| create_binary | aclrtCreateBinary 返回空指针，无直接 ACL 状态码 |
| binary_load | aclrtBinaryLoad |
| get_function | 查找 KvafGatherBatchGetAivKernelV3 |

先让 Python 和 kernel 明确指向同一个 build-npu：

```bash
cmake --build build-npu --target kvaf_gather_batch_get_kernel _kvaf -j
export PYTHONPATH="$PWD/build-npu/python"
export KVAF_BATCH_GET_KERNEL="$PWD/build-npu/bin/kvaf_gather_batch_get_kernels.o"
export KVAF_BATCH_PUT_KERNEL="$PWD/build-npu/bin/kvaf_execute_batch_put_kernels.o"
DEVICE_ID=7 KVAF_NPU_CORE_COUNT=1 python -u tests/python/test_aiv_gather_pipeline_debug.py
```

若仍失败，提供 `GATHER_KERNEL_FILE`、`GATHER_PYTHON_IMPORT` 和首条
`GATHER_KERNEL_LOAD_ERROR`。这样可以区分旧文件/路径与 ACL 的实际加载失败，
不再根据通用初始化异常猜测 WQE 或网络故障。

如需单独运行某个检查点，可设置 `KVAF_AIV_GATHER_PROBE=ack/wqe/sq/first_read`
并执行 `test_aiv_gather_batch_get.py`。正常性能脚本会拒绝非空探针设置，
避免输出部分数据却被误记成性能结果。

### V3 已加载、ACK 完成后 WQE 检查点异常

新反馈已出现 `GATHER_KERNEL_READY ... V3 args_bytes=232`、
`GATHER_PROBE_PASS stage=ack`，随后 `probe=wqe` 同步返回 507015。
因此加载问题已解决，8193 请求的控制链路已到达最终 ACK，尚未通过首批
WQE 构造检查；没有运行到 SQ payload 搬运、数据 doorbell 或数据 CQ。
该版本 wqe 检查点仍包含 MTE3 寄存器配置，所以此结果尚不能单独证明
异常指令就在 SIMT 函数体内。移除私有数组后仍复现，根因仍需验证。

本轮按同一日志中已成功执行的 BatchPut 路径做三项调整：

1. MainScalar 生成 `__ubuf__ uint32_t*` bank 指针并传入 VF；VF 只在该 UB
   指针上做偏移，不再在 VF 内调用 get_imm 构造地址。
2. 为 gather 导出显式 `FunLevelKType / K_TYPE_AIV` 元数据，与 BatchPut 和
   direct_read 一致。独立二进制和 Host launch 配置见
   [CANN 9.1 官方 AIV 调度示例](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/commlib/commopdev/docs/en/comm_op_dev_guide/aiv_comm_op_dev/op_dispatch.md)。
3. wqe 检查点跳过数据 SQ 的 MTE3 loop/atomic 配置；sq、first_read 和正常
   流水继续执行原有配置。在 ack 已通过的前提下，wqe 新增范围只包含
   READ WQE 的 VF 准备、构造及完成等待。

ABI 仍为 V3 / args 232 B / profile 192 B，已有 V3 Host 可以只重编 gather
kernel。它们是针对已定位范围的修正，不等于已证明 507015 的具体根因。

```bash
git pull --ff-only
cmake --build build-npu --target kvaf_gather_batch_get_kernel -j
DEVICE_ID=7 KVAF_NPU_CORE_COUNT=1 python -u tests/python/test_aiv_gather_pipeline_debug.py
```

debug 的子进程默认设置 `ASCEND_SLOG_PRINT_TO_STDOUT=1`、
`ASCEND_GLOBAL_LOG_LEVEL=3`，将 CANN ERROR 日志打屏；已有显式设置时保持
调用者的值，BEGIN 行会打印实际值。设置仅作用于诊断子进程，不修改 shell
和性能脚本环境。日志开关含义见
[日志打屏](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/maintenref/envvar/envref_07_0121.html) 和
[全局日志等级](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/maintenref/envvar/envref_07_0122.html)。
如果仍失败，保留首个失败检查点之前的完整 CANN ERROR 块；若包含 PC、
SU/MTE/VF 错误位，可继续对照本次二进制定位，不能沿用旧二进制的偏移。

## 计时范围

无打点基线使用 `synchronize=False`，随后由 Python 调用方同步到本批完成；
正常异步路径不自动 D2H 回收 status，同流连续提交也不再隐式同步上批。
预检与 profile 使用 `synchronize=True`，当前批的状态回收包含在本批完成之内。
初始化、BatchPut、请求 Tensor 构造和结果校验不在测量循环内。预热后先采
关闭 profile 的端到端延迟及有效 GiB/s，再单独开启 profile 采阶段数据。

每个 AIV 核记录六段：

| 阶段 | 边界 |
| --- | --- |
| 输入准备 | kernel 入口 → 输入缓存处理、请求数检查和本 lane 分区完成；逐请求校验已关闭 |
| 清单构造 | 输入准备完成 → 队列准备、清单 SIMT 构造和 UB→HBM MTE3 完成 |
| 请求发布 | 清单完成 → descriptor、command、notify 的 WRITE 完成 |
| 等待 CPU | notify WRITE 完成 → 观察到最终 ACK |
| 数据 READ | 最终 ACK → 全部数据 READ 的 CQ 完成，包含 WQE 构造和容量回收 |
| 收尾 | 数据完成 → 状态发布及 profile 结束时间点 |

空批或无任务 lane 不发通知或数据 READ，其阶段时间按跳过处理。原始 profile
同时保留请求数、数据 WR/WQEBB、doorbell、ACK 查询及队列等待等计数。
这些子计数用于解释数据阶段，不与其总时间重复相加。

Host 的 `prepare_ns` 覆盖 owner 内的参数检查及 output 注册，`launch_ns`
只覆盖 launch 入队；status/profile 在设备入口完整初始化，不再逐批由 Host 清零。
`status_enqueue_ns` 覆盖状态/profile D2H 入队，
`wait_ns` 是设备执行及 D2H 的剩余完成等待，不是完整 kernel 耗时；
`finish_ns` 覆盖结果校验与资源回收。
`total_ns` 从上批 Drain 后的本批 owner 准备开始到回收结束，不包含 Python、
Runtime 参数适配和 capture 查询；因此它
不必等于 Python 外层端到端时间。profile 自身的写回/复制也有成本。
输出中的 `profile=off/on` 仅描述 AIV/Host，`cpu_profile=external` 表示远端
CPU 开关由服务进程单独控制，脚本无法替它关闭。

CPU/Host 的 ns 与 AIV 的 cycle 属于不同的时钟域，只计算各自差值，以
`sequence + lane/core` 关联。AIV 等待 CPU 已覆盖部分 CPU 工作及轮询延迟，
不能再把 CPU gather 时间加进去。各核并行时间也不能相加。

## Python 接口

```python
runtime.start_gather_batch_get_profile()
runtime.batch_get(..., synchronize=True)
profile = runtime.gather_batch_get_profile()
runtime.stop_gather_batch_get_profile()
```

快照包含 `sequence`、Host 时间字典 `host` 和每核时间线 `cores`。只接受
成功的同步 gather；开启 profile 时异步调用会明确拒绝。开启后尚未采样、
执行失败或随后执行未采样批次时，不会返回旧快照。关闭采集无需已有快照。

## 后续与二阶段比较

保留串行基线，先比较“串行 / AIV 内部流水”，后续再加入 CPU chunk 流水。
固定输入、有效字节、核数、CPU 线程预算、SGL 和队列容量，保持软件在途预算关闭。
比较 p50/p95/p99、有效 GiB/s 和串行耗时除以流水耗时的加速比。流水阶段
另加 chunk ready、提交和完成事件验证重叠，不把重叠阶段耗时相加。

本地 CPU 模型和 Host 替身只验证计时记录、生命周期及数据语义；CANN 编译、
计数器频率、远端 DMA 可见性和真实性能仍需目标双机验证。

## 已有端到端结果与 profile 校验失败

`GATHER_PERF_RESULT` 是关闭 AIV/Host profile 的同步调用端到端带宽。旧脚本的
`operation=batch_get_kernel_total` 则只使用 kernel 时间，比较前须统一计时范围。
例如 8192 × 656 B 约 5.37 MB，35 GB/s 对应约 154 us；不能直接拿它与含
Python/Host、CPU gather、控制握手和状态回收的整批延迟比较。8192 与 8193
的请求数差异仅约 0.012%，无法解释数量级差距。

资源也要对齐：原 direct 的 2 核 × 每核 2 条 Jetty 是四条数据队列；gather
每核固定一条数据队列和一条控制队列。`KVAF_NPU_CORE_COUNT=2` 可以对齐
核数，但 gather 仍只有两条数据队列。串行基线中的 gather 逐项构造清单、标量写
SQ 及 flush；在全部 READ WQE 能放进物理队列时，数据阶段末尾才敲 doorbell。
原 direct 则已有 SIMT 分批 WQE 构造、UB/MTE 提交和内部流水。本次已将
这些 WQE 优化引入 gather 的快速路径。当前 direct 和 gather 的逐请求校验
调用均已注释；gather 仍保留输入缓存维护和基础检查，总耗时仍计入全部六阶段。
实际瓶颈仍以阶段数据为准。

`invalid gather profile` 在同步及普通执行状态检查通过后产生。新异常包含
`reason`、记录 magic/sequence/result/valid_mask、七个原始 cycle、请求数/WR
数及其状态对照，且在资源清理前保存到异常文本。校验门槛保持不变。

| reason | 可以确定的失败项 |
| --- | --- |
| magic / sequence / result / valid_mask | 记录头或完成标记不符合预期 |
| timestamp_* | 指定的相邻原始时间戳倒序；尚未进行频率换算 |
| requests / data_wrs | profile 与普通执行状态的计数不一致 |

没有实际字段时，不能将该错误认定为时钟、缓存或 ABI 的某一种故障。
新的日志用于区分这些方向；本地模型不复现远端设备上的失败。

快速复测时，只改性能脚本顶部：

```python
WARMUP = 0
ITERATIONS = 0
PROFILE_SAMPLES = 1
```

脚本仍执行 payload/guard 预检，随后跳过端到端性能循环，打印
`GATHER_PROFILE_PHASE_BEGIN` 并开始阶段采集。失败时保留完整的
`invalid gather profile ... reason=...` 行。`6d2ecaf` 的诊断增强只修改 Host 和
Python；下文针对 `0x41` 的修改位于 kernel，必须重编对应 kernel。

## valid_mask=0x41、早期时间戳为零的修正

硬件反馈：magic、sequence、result、requests、data_wrs 均匹配，但 mask 为
`0x41`，start/input/descriptor/notify 为零，ack/read/finish 已有非零 cycle。
`0x41` 仅包含外层入口、出口两位，与函数内部阶段和 mask 的更新未一致保留
相符。它不表示数据 READ 失败，也不是由 MHz 换算产生。没有匹配的设备
指令或编译器中间表示，尚不能断定具体的编译器/局部内存机制。

修改将计时状态从跨 Gather/Queue 函数传递的可变局部结构体移到每核独占的
HBM profile 记录，使用显式 `__gm__ volatile` 指针。入口显式初始化，各阶段
直接写对应字段并累积 mask，结尾设置 result/magic 并 flush；不再把局部
结构体整体或逐字段回拷。关闭 profile 时不读时钟、不访问 profile 内存。
每条记录仍只有一个 AIV 核写入，Host 仍在 stream 完成后读取。

该次 GM 时间线修复保持 V2 / args 224 B / profile 192 B；后续诊断升级为 V3。
Host 仍要求 `valid_mask=0x7f`，
继续拒绝硬件日志中的不完整形状。此次需在 NPU 目标机运行：

```bash
git pull --ff-only
cmake --build build-npu --target kvaf_gather_batch_get_kernel -j
DEVICE_ID=7 KVAF_NPU_CORE_COUNT=1 python tests/python/test_aiv_gather_batch_get_perf.py
```

沿用已有 Host、环境和 kernel 路径；若尚未部署 `6d2ecaf` 的 Host 诊断，先
执行一次完整 `cmake --build build-npu -j`。计时入口前的 profile 初始化和
最后记录导出的开销不属于六段 cycle，外层同步调用仍包含这些成本。

模型新增在 descriptor/ACK/data doorbell 时检查生产 GM 记录的阶段进度，确保
早期时间戳在内层函数仍执行时已保留；同时覆盖空 lane、重复批次与错误。
这些模型检查不模拟 CANN 的局部对象编译行为，是否消除本次硬件失败仍需
上述复测。当前日志的后半段仅作为性能线索，不作为通过验收的六阶段结果。
