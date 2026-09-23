# AICPU URMA BatchPut / BatchGet

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
需要写入权限。不传该选项时只构建，不修改 CANN。原生客户端默认使用构建时的
CANN 路径查找 AICPU JSON，并使用单 worker、单 Jetty；无需配置 `CANN_ROOT`
或提供 AIV kernel object 路径。若 `ASCEND_HOME_PATH` 未设置，客户端在 ACL 初始化前
将其设为构建时的 CANN 根目录，以便设备包加载器找到安装包；显式设置的值不覆盖。
服务端仍是 KVAF 原有 CPU Runtime 服务。

Host 单测：`ctest --test-dir build-aicpu-urma -R kvaf_aicpu_urma_ --output-on-failure`。
无需 CANN 的机器也可运行：`cmake -S . -B build-host -DBUILD_TESTING=ON && cmake --build build-host --target kvaf_aicpu_urma_wqe_test kvaf_aicpu_urma_kernel_test kvaf_aicpu_urma_client_options_test && ctest --test-dir build-host -R kvaf_aicpu_urma_ --output-on-failure`。

真实 CPU/NPU 回读测试：CPU 端先运行 `./build.sh cpu` 和
`build-cpu/runtime/kvaf_cpu_service --listen 0.0.0.0 --port 28999`；
NPU 端安装内核并重启测试进程后执行：

```bash
build-aicpu-urma/tests/kvaf_aicpu_urma_client \
  --peer CPU_HOST:28999 \
  --device-id 0 --blocks 16 --block-bytes 4096 --token-bytes 256 \
  --warmup 1 --iterations 5 --seed 123
```

同机 CPU service 使用默认监听地址和端口时，客户端不带参数也可运行：
`build-aicpu-urma/tests/kvaf_aicpu_urma_client`。其他 peer 用 `--peer` 覆盖；
内核安装在其他目录时可用 `--kernel-json` 覆盖。默认 JSON 缺失或不可读时，
Runtime 在枚举 EID 前直接报错。

`--blocks` 为每批 KV block 数；`--block-bytes` 为每个 block 的写入字节数，
必须能被 `--token-bytes` 整除。客户端会对每个 block 的所有 token 执行 BatchGet，
反向排列输出并逐字节比对，因此覆盖完整 BatchPut payload 和 BatchGet READ。
`--ddr-bytes` 可选，至少为 `blocks * block-bytes` 且是 block 的整数倍。
每轮先写入新 payload，再读取并验证；warmup 不计入统计，输出每轮耗时和平均
`put_GBps/get_GBps`。Put 计时包含 MetaManager 的预留和提交；Get 计时包含
Runtime 提交与 stream 同步，不包含 AccessTable 准备、HBM 上传/回读和逐字节校验，
因此不是纯 SQ 带宽。
失败时退出码非零且不会输出 `AICPU_URMA_PASS`；`--help` 查看完整参数。

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

需要区分共享的 Host-HCCP/WQE 问题与 AICPU 提交路径问题时，用同一个原生客户端
运行 AIV 对照，无需 Python。`--compare-aiv` 只额外编译 AIV kernel object，
不改变默认 AICPU 后端：

```bash
./build.sh aicpu-urma --install-kernel --compare-aiv
build-aicpu-urma/tests/kvaf_aicpu_urma_client \
  --backend aiv --peer 127.0.0.1:28991 --blocks 1 --warmup 0 --iterations 1
```

若 AIV 在同一候选的 WRITE 探针成功而 AICPU 返回 `2/2`，重点检查 AICPU 写 SQ、
缓存可见性和门铃顺序；若两者在同一候选均返回 `2/2`，先检查共享的 WQE 字段、
token、地址与 PeerLink 资源。两种后端的批量 WQE 目前都是单 SGE/WQE，
`--blocks` 不会将一个 WQE 变为多 SGL。

后续 AIV 对照在 `peer_candidate=9` 的两个 NPU EID 上也得到 `status=2`、
`sub_status=2`。AIV 的 `processed=1` 只表示请求已发出，不能视为 WRITE 成功；
旧版 AIV 的 `cqe_word=0` 是因为未保存原始 CQE。由此可以排除 AICPU 独有的提交路径
是唯一原因，但仍不能从该片段确定 26 个 CPU 候选的最终结果。

共享 WQE 控制字与本地 HCOMM AICPU 实现对照后，发现 KVAF 首轮 SQE 使用
`owner=1` 且末条未设置 `fence`，而 HCOMM 使用首轮 `owner=0`、末条 `fence=1`。
现已统一修正 AICPU 和 AIV BatchPut/BatchGet 编码；这是有依据的故障假设，
尚未经过真机验证。重新构建后先用同一 peer 跑两种后端的单 block 探针：

```bash
./build.sh aicpu-urma --install-kernel --compare-aiv
build-aicpu-urma/tests/kvaf_aicpu_urma_client \
  --backend aiv --peer 127.0.0.1:28991 --blocks 1 --warmup 0 --iterations 1
build-aicpu-urma/tests/kvaf_aicpu_urma_client \
  --backend aicpu --peer 127.0.0.1:28991 --blocks 1 --warmup 0 --iterations 1
```

修正后首个 CPU 候选的两个 NPU EID 仍返回 `2/2`，因此 owner/fence 不是该错误的
充分解释。本地 HCOMM A5 状态表将 `status=2, sub_status=2` 标为
`Local Operation Error / Local Access Error`；该表不是当前驱动的实时诊断，
不能仅凭它认定具体的 HBM 地址或 token 有误。新版 AIV 状态会保存原始 CQE，
两端失败日志均打印 `cqe_word`、`sq_pi`、`cq_ci`、`last_wqe`；HBM 注册日志打印
注册接口返回的原始 token ID，若与分配的 token ID 不一致则单独告警，并使用
注册结果填写 WQE 的本地 token ID。下一轮先对比两端同一候选的这些字段，
以及 CPU 服务 `REGION_ALLOCATED` 的 `token_id` / `seg_token_id`，再检查设备日志
中的本地内存访问错误；不要把候选扫描中途的片段当作最终结果。
