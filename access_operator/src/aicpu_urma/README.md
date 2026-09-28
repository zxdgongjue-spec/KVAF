# AICPU URMA BatchPut / BatchGet / GatherRead

这个可选后端不创建第二条链路：沿用 KVAF `NpuRuntime::LinkTo` 的候选 EID/PeerLink
选择，以及现有 Direct-HCCP Host 侧的 HBM/Host DDR 注册、远端导入、USER_CTL
Jetty、SQ/CQ 和 token。AICPU 只从现有 transport 的 entity/queue 表生成并提交
WRITE（BatchPut）或 READ（BatchGet）WQE，等待 CQ，写回结果。

目前采用**单 AICPU worker、单 Jetty**，每 32 个请求提交一次带 completion 的
批次；AIV 模式保持默认不变。BatchPut 同步等待并验证状态；BatchGet 保留调用方
`synchronize` 语义，开启同步时检查 AICPU 状态。不支持与同一 Jetty 上 AIV
kernel 并发使用；异步 BatchGet 需要调用方在释放输出/transport 前同步 stream。

## 构建与部署

在有 CANN toolchain 的 NPU 主机上：

```bash
./build.sh aicpu-urma --install-kernel
```

`build.sh` 自动检测 CANN 根目录；`--install-kernel` 显式安装到同一目录，
需要写入权限。安装后会检查压缩包内容、JSON、加载配置中唯一且完整的条目，
并对比安装文件与构建产物；检查失败则命令以非零状态退出。不传该选项时只构建，
不修改 CANN。检查与卸载不重新编译：

```bash
./build.sh aicpu-urma --check-kernel
./build.sh aicpu-urma --uninstall-kernel
```

这两个命令只作用于**当前主机**的 CANN 根目录。构建主机与运行主机不同时，
必须在运行主机安装相同的包；`ascend_package_load.ini.kvaf-aicpu-urma.bak`
只是安装前备份，不代表当前加载配置仍含 KVAF 条目。卸载仅移除 KVAF 的条目和
两个安装文件，不用备份覆盖其他包的配置。若 `ASCEND_CANN_PACKAGE_PATH`
和 `ASCEND_HOME_PATH` 指向不同 CANN 根目录，构建/检查会直接报错。

当前 `NpuRuntime` 仍复用 AIV BatchGet/BatchPut 的 Direct-HCCP Host 目标，
所以构建 AICPU 客户端会连带编译其 AIV kernel 依赖；客户端本身只执行 AICPU 路径。

原生客户端默认使用构建时的
CANN 路径查找 AICPU JSON，并使用单 worker、单 Jetty；无需配置 `CANN_ROOT`
或提供 AIV kernel object 路径。若 `ASCEND_HOME_PATH` 未设置，客户端在 ACL 初始化前
将其设为构建时的 CANN 根目录，以便设备包加载器找到安装包；显式设置的值不覆盖。
服务端仍是 KVAF 原有 CPU Runtime 服务。

Host 单测（含部署脚本的本地模拟测试）：
`ctest --test-dir build-aicpu-urma -R 'kvaf_aicpu_urma_|kvaf_aicpu_gather_read_|kvaf_cpu_gather' --output-on-failure`。
无需 CANN 的机器也可编译并运行 WQE、gather 和内核握手单测。

真实 CPU/NPU 回读测试：CPU 端先运行 `./build.sh cpu` 和
`build-cpu/runtime/kvaf_cpu_service --listen 0.0.0.0 --port 28999`；
NPU 端安装内核并重启测试进程后执行：

```bash
build-aicpu-urma/tests/kvaf_aicpu_urma_client \
  --peer CPU_HOST:28999 \
  --device-id 0 --blocks 16 --block-bytes 4096 \
  --warmup 1 --iterations 5
```

同机 CPU service 使用默认监听地址和端口时，客户端不带参数也可运行：
`build-aicpu-urma/tests/kvaf_aicpu_urma_client`。其他 peer 用 `--peer` 覆盖；
内核安装在其他目录时可用 `--kernel-json` 覆盖。默认 JSON 缺失或不可读时，
Runtime 在枚举 EID 前直接报错。

`--blocks` 为每批 KV block 数；`--block-bytes` 为每个 block 的写入字节数。
`--token-bytes` 默认等于最终的 `--block-bytes`；只有需要将 block 拆为多个
BatchGet token 时才指定，且 `--block-bytes` 必须能被它整除。客户端会对每个
block 的所有 token 执行 BatchGet，反向排列输出并逐字节比对，因此覆盖完整
BatchPut payload 和 BatchGet READ。
CPU DDR 分配大小固定为 `blocks * block-bytes`。
每轮先写入新 payload，再读取并验证；warmup 不计入统计，也不输出每轮成功日志。
完成后汇总 Put/Get 的平均、最小、p50、p95、p99、最大耗时和平均
`put_GBps/get_GBps`。Put 计时包含 MetaManager 的预留和提交；Get 计时包含
Runtime 提交与 stream 同步，不包含 AccessTable 准备、HBM 上传/回读和逐字节校验，
因此不是纯 SQ 带宽。底层 Direct-HCCP 的诊断日志仍可能出现。
失败时退出码非零且不会输出 `AICPU_URMA_PASS`；`--help` 查看完整参数。

## CPU Gather + AICPU 连续 READ

此路径使用同一个 `NpuRuntime::LinkTo` / Direct-HCCP Jetty，不改变 AIV 算子。
CPU 服务端先收到离散源地址清单及 staging 地址；AICPU 发一条 opcode 5 的
WRITE_WITH_NOTIFY（96 字节，占两个 64 字节 SQ WQEBB），将序号写入 DDR 控制区并
通知 CPU worker。worker 按清单将 DRAM 数据复制到连续 staging，刷出 CPU 缓存并
写回完成序号。AICPU 用 READ 轮询完成序号，匹配后将连续 staging 读入连续
HBM 输出；超过 256 MiB 时按 hcomm 的单 WQE 上限分段提交。当前不执行 HBM
离散地址 scatter，也不是多 SGL WQE。

两端分别运行：

```bash
# CPU 终端
./build.sh cpu
build-cpu/runtime/kvaf_cpu_service --listen 0.0.0.0 --port 28999

# NPU 终端
./build.sh aicpu-urma --install-kernel
build-aicpu-urma/tests/kvaf_aicpu_gather_read_client \
  --peer CPU_HOST:28999 --device-id 0 \
  --blocks 16 --block-bytes 4096 --warmup 1 --iterations 5
```

客户端每轮 BatchPut 不同 payload 到 DDR，按 block 逆序生成 gather 清单，校验连续
HBM 输出；仅汇总 GatherRead 计时。客户端从保留 key 的 AccessTable 中选择连续
staging，避免 LinkTo 探针轮转 slot 后与源块重叠；另保留控制区并占住其 slot，避免 MetaManager
后续复用。调用 `NpuRuntime::GatherRead` 时，源地址与 staging 必须在注册 DDR
范围内且不重叠，末尾 64 字节控制区不可用于 KV slot。输出 HBM 需要额外的
64 字节对齐 scratch 空间；数据本身从输出基址开始。一个 region 同时只允许
一个未完成的 gather。真实设备的 WRITE_WITH_NOTIFY 支持和性能仍需在目标机器验证。

**硬件验证状态**：本分支的 Host WQE ABI/模拟队列测试和内核语法检查不等价于 NPU
收发验证。部署后应先用 Runtime 自带的 LinkTo WRITE 探针确认链路，再在真实
CPU DDR/HBM 之间做小 batch 的 put/get 数据回读验证，最后再进行性能测试。

## 排查 AICPU 探针失败

如果日志已出现 `PEER_LINK_JETTY_IMPORT_READY`，但 WRITE 探针在
`aclrtSynchronizeStream` 返回 `507018`，这表示 AICPU 执行异常，不能仅凭
`PEER_LINK_EID_PAIR_FAILED` 判断 EID 不匹配。此时没有从设备拷回状态；
`BATCH_PUT_DEVICE_STATUS_UNAVAILABLE` 表示 Host 侧没有有效的内核状态。
先保留完整的 Host plog 和 AICPU device 日志，查看 `PrintAicpuErrorInfo`、
`HandleTaskTimeout`、算子加载失败或内存访问异常，再区分加载、访问和 CQ 等待问题：

```bash
find "$HOME/ascend/log" -type f -mmin -30 -print 2>/dev/null
grep -R -nE '507018|PrintAicpuErrorInfo|HandleTaskTimeout|KvafAicpuUrma' \
  "$HOME/ascend/log" 2>/dev/null | tail -100
```

使用同一 CPU peer 的纯 AIV WRITE 探针可作为对照，但不能跳过 AICPU 探针直接
选定 EID：这会把 AICPU 执行失败延后到正式 BatchPut。

如果设置 CANN 根目录后变成 `execute kernel param invalid`，说明 `.so` 已加载，
但旧版 AICPU 内核把所有失败都返回为任务错误。新版同步 BatchPut 在状态缓冲区
可用时通过 `BATCH_PUT_DEVICE_ERROR` 输出具体 `result`：1=入参形状，
2=entity，3=队列，4=源地址，5=目标地址，6=SQ 已满，7=CQ 超时，
8=CQ 错误，9=长度；`failed_request` 仅在逐请求参数校验失败时给出索引。

`result=8` 表示 USER_CTL CQ 已返回错误完成，并不等于 EID 建链失败。
AICPU BatchPut 会额外打印 `cq_status`、`cq_sub_status` 和原始
`cqe_word`（保存在状态 ABI 的 `reserved[0]`）。只有收到错误 CQE 时这些字段
才有效；`result=7` 是 CQ 超时，没有错误 CQE。当前日志中的 `result=8` 但
`cq_status=0 cq_sub_status=0` 来自旧版 AICPU 内核未写回 CQE 字段，无法据此
判定具体硬件错误。重新构建、安装内核并重新运行客户端，用单 block 探针采集新值：

```bash
./build.sh aicpu-urma --install-kernel
build-aicpu-urma/tests/kvaf_aicpu_urma_client \
  --peer CPU_HOST:28999 --blocks 1 --warmup 0 --iterations 1
```

对照纯 AIV 使用同一 CPU peer 的探针结果和新 CQE 状态，再区分 token、地址、
WQE 协议或互联故障；不能仅凭当前 `result=8` 归因到其中一种。

最新单 block 日志的 `cqe_word=0x02020006`，即 `cq_status=2`、
`cq_sub_status=2`。这表明硬件返回了错误完成，不是 CQ 超时；两个 NPU EID
候选在同一个 CPU 候选上返回相同的 CQE，但测试在继续扫描 CPU 候选时被中断，
不能据此断言所有 26 个候选都失败。当前没有经过确认的该设备子状态码映射，
不要把 `2/2` 直接解释为某一种 token、地址或 EID 错误。

需要区分共享的 Host-HCCP/WQE 问题与 AICPU 提交路径问题时，单独构建原生
AIV 客户端作对照，无需 Python 或修改 AICPU 客户端：

```bash
./build.sh aiv-native
build-aiv-native/tests/kvaf_aiv_batch_roundtrip \
  --peer 127.0.0.1:28991 --device-id 0 --blocks 1 --iterations 1
```

AIV 原生客户端不调用 `NpuRuntime::Initialize` 的 WRITE 预探针；比较结果前应从
日志确认双方实际选择的 CPU/NPU EID 候选一致。

若 AIV 在同一候选的 WRITE 探针成功而 AICPU 返回 `2/2`，重点检查 AICPU 写 SQ、
缓存可见性和门铃顺序；若两者在同一候选均返回 `2/2`，先检查共享的 WQE 字段、
token、地址与 PeerLink 资源。两种后端的批量 WQE 目前都是单 SGE/WQE，
`--blocks` 不会将一个 WQE 变为多 SGL。

无设备环境下评估 CPU gather 后使用可配置 SGL 发送的映射和理论时间，可运行
[`aicpu_sgl_simulator.md`](../../../tests/aicpu_sgl_simulator.md) 中的 CPU-only
仿真。它假设 AICPU 发送带宽为 25 GB/s，不改变现有 kernel 的单 SGE/WQE 实现。

后续 AIV 对照在 `peer_candidate=9` 的两个 NPU EID 上也得到 `status=2`、
`sub_status=2`。AIV 的 `processed=1` 只表示请求已发出，不能视为 WRITE 成功；
旧版 AIV 的 `cqe_word=0` 是因为未保存原始 CQE。由此可以排除 AICPU 独有的提交路径
是唯一原因，但仍不能从该片段确定 26 个 CPU 候选的最终结果。

共享 WQE 控制字与本地 HCOMM AICPU 实现对照后，曾尝试把 AICPU 和 AIV 的
首轮 owner 改为 0、末条 fence 改为 1；真机 AIV 探针仍返回 `2/2`。
现已将 AIV BatchPut/BatchGet kernel 和 Put Host 注册行为回退到本分支起点的
实现；AICPU WQE 编码仍保留该假设，尚需单独验证。对照时要注意两者目前
不再使用相同的 WQE 控制字。分别运行两个独立客户端的单 block 命令：

```bash
./build.sh aiv-native
build-aiv-native/tests/kvaf_aiv_batch_roundtrip \
  --peer 127.0.0.1:28991 --device-id 0 --blocks 1 --iterations 1
./build.sh aicpu-urma --install-kernel
build-aicpu-urma/tests/kvaf_aicpu_urma_client \
  --peer 127.0.0.1:28991 --device-id 0 --blocks 1 --warmup 0 --iterations 1
```

上述实验中首个 CPU 候选的两个 NPU EID 仍返回 `2/2`，因此 owner/fence 不是该错误的
充分解释。本地 HCOMM A5 状态表将 `status=2, sub_status=2` 标为
`Local Operation Error / Local Access Error`；该表不是当前驱动的实时诊断，
不能仅凭它认定具体的 HBM 地址或 token 有误。AIV 状态不保存原始 CQE，
因此其 `cqe_word=0` 不代表硬件返回 0；AICPU 状态仍保存原始 CQE。
CPU 服务 `REGION_ALLOCATED` 会打印 `token_id` / `seg_token_id` 供对照。
不要把候选扫描中途的片段当作最终结果。
