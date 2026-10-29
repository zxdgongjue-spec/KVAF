# 同一份 KV DDR 同时用于 K Pull 和 Decode BatchGet

## 当前交付与结论边界

这是一组独立诊断程序，目前尚无 UB/NPU 三端完整通过结果，不能据此宣布单 MR 或双 MR 已获驱动支持。

- `k_service.cc` 使用本工作区真实 `CpuRuntime` 的分配、注册和建链代码，仅在诊断可执行文件中拦截 KV MR 注册。
- `hbm_export.cc` 在 Prefill 进程内导出已有 Tensor HBM，使用现有 Direct HCCP/RA ABI。
- `run_probe.py prefill` 更新 HBM，通知 K 发起 READ。
- `run_probe.py decode` 调用真实 `NpuRuntime.batch_get`，检查 NPU 后继算子和最终字节内容。
- `host_test.cc`、`test_control.py` 验证诊断程序自身。fake provider 的通过不代表硬件通过。

本版明确针对当前 `codex/shared-ddr-two-npu` 基线 `a44e913`、控制协议 v7、AIV BatchGet。
它验证的是所需的共享 DDR 数据路径；并未移植参考聊天中 `e99a50e` 的 AICPU v6 service。
**不能混用 AICPU v6 的 Python 包与本程序的 v7 service。** AICPU gather 还会追加控制区和 staging，
需要额外适配其内存布局和 worker 生命周期；本程序会拒绝以 v6 CPU 源码编译。
原工作区已有的未提交修改不属于本诊断改动。三端必须使用同一份 KVAF 源码构建。

## 本仓库固定的 UMDK 权限语义

`build.sh` 固定的是 **`br_UMDK_24.12.0_LTS`**。2026-10-27 取得的该分支提交为
`8cb685a80bf92e6287c3262d46b3d513fb3a3302`。前一版说明误用了新版 LOCAL_ONLY 的语义，现已纠正。

该版本的定义是 `LOCAL_WRITE=1`、`REMOTE_READ=2`、`REMOTE_WRITE=4`、`REMOTE_ATOMIC=8`；
没有 `LOCAL_ONLY`。因此 `0xf` 是“本地写 + 远端读写原子”，不是矛盾的权限组合。
见 [该提交的 urma_opcode.h](https://gitcode.com/openeuler/umdk/blob/8cb685a80bf92e6287c3262d46b3d513fb3a3302/include/urma_opcode.h)。

更关键的是，原版 `urma_check_seg_cfg()` 在设置 REMOTE_WRITE 或 REMOTE_ATOMIC、
但未设置 LOCAL_WRITE 时返回失败，`urma_register_seg()` 随后设置 `errno=EINVAL`，尚未进入 provider。
所以 **原版库会拒绝 KVAF 当前的 `access=0xe`**；`0xf` 能通过这项 core 校验，仍可能被 provider/驱动拒绝。
见 [该提交的注册校验](https://gitcode.com/openeuler/umdk/blob/8cb685a80bf92e6287c3262d46b3d513fb3a3302/lib/urma/core/urma_cp_api.c)。

这与 KVAF 注释所述“部署驱动拒绝 LOCAL_WRITE，故保留 0xe”存在配套差异。
如果目标机已运行成功，不能只根据头文件来源断言实际 `liburma.so` 就是该分支未修改版本；
需要记录加载的 liburma、provider/驱动、固件、CANN 版本，核对是否有部署补丁或不同版本的库。
程序打印 `header=...` 仅描述编译头文件，不能证明加载库的权限语义。
`access_compat.h` 现在拒绝新版 LOCAL_ONLY 头文件，避免测试时再次混淆。

这版公开头文件的 SGE 是 **24 B**，程序默认按其布局提交。
`-DPROBE_EXTENDED_SGE=ON` 显式选择参考 CPU push 的扩展 **32 B SGE**，尾部 `user_tseg` 为零。
应优先用部署头文件/provider 实现确认布局；若用于 ABI 排查，应保持 case、卡、EID、分块大小
和页模式不变，只比较这一项，并保留两组日志。它是候选布局，不因参考实现使用就自动视为目标机契约。
当前 AIV BatchGet 直接构造硬件 WQE，不通过 host SGE，故其成功不能认证 CPU READ 的 SGE 布局。
segment=48 B、Jetty ID=24 B、NPU Jetty key=32 B、CPU Jetty key=28 B 的约束保持不变。
建链还沿用现有 KVAF 所需的 `urma_get_tp_list`、`urma_import_jetty_ex` expanded CTP 扩展；
本次取得的原版分支源码没有这两个函数定义，不能假定自行编译原版库就具备部署扩展。
先核对实际库的导出符号及 ABI，再运行硬件测试。

## 测试矩阵

每组启动三个新进程。不要在同一个 K 进程中切换 case。

| case | Decode 导出的 MR | K READ 使用的 MR | 用途 |
|---|---|---|---|
| `single-current` | `access=0xe, non_pin=1` | 同一个 MR | 复现 KVAF 配置；原版 24.12 core 预期拒绝 |
| `single-pinned` | `access=0xe, non_pin=0` | 同一个 MR | 单独改变 pin；原版 core 仍预期拒绝 |
| `single-plus-bit0` | `access=0xf, non_pin=1` | 同一个 MR | 加上 LOCAL_WRITE 的单 MR 候选 |
| `single-local-write-pinned` | `access=0xf, non_pin=0` | 同一个 MR | 单 MR 候选，pinned |
| `dual` | `access=0xe, non_pin=1` | 同一 VA/len，`access=1, non_pin=0` | 双 MR，先注册远端视图 |
| `dual-compatible` | `access=0xf, non_pin=1` | 同一 VA/len，`access=1, non_pin=0` | 两个视图均满足原版 core 的权限校验 |
| `dual-reverse` | `access=0xe, non_pin=1` | 同一 VA/len，`access=1, non_pin=0` | 先注册本地视图，检查顺序依赖 |
| `dual-both-pinned` | `access=0xe, non_pin=0` | 同一 VA/len，`access=1, non_pin=0` | 两个视图都 pinned |
| `dual-separate-context` | 原 context，`0xe/non_pin=1` | 同设备/EID 的另一个 context，`1/non_pin=0` | 跨 context 的补充试验 |

双 MR 各用独立 token ID，两个注册始终引用**同一 allocation、同一 VA 和长度**。
首次注册/第二次注册的 `errno`、flags、context 和 MR 地址分别记录。
默认同 context；`dual-separate-context` 也不等价于跨 UB 设备的重复注册。
`dual`、`dual-reverse`、`dual-both-pinned`、`dual-separate-context` 保留原始 0xe 远端视图，
用于检查部署库行为；未修改的 24.12 core 会拒绝该视图。

## 编译

目标为 Linux AArch64 + 部署时配套的 UMDK/CANN。所有命令从 KVAF 仓库根目录执行。
示例路径需要替换成目标机路径；构建不下载或安装驱动。
K 是运行 Linux 服务的 Host CPU，可以是 NPU 服务器的机头；Prefill、Decode 分别使用该机
两张 NPU 和独立进程。三角色不要求三台机器。控制端口可以用 loopback，但 KV payload
仍须经 K 的 URMA READ 进入 DDR，再由 Decode BatchGet 读取；同机通过不能外推三机拓扑。

K 机器：

```bash
export PROBE_UMDK_INCLUDE=/path/to/br_UMDK_24.12.0_LTS/include
export PROBE_URMA_LIBRARY=/usr/lib64/liburma.so
cmake -S tests/diagnostics/shared_ddr_probe -B build-ddr-probe-cpu \
  -DPROBE_ROLE=cpu \
  -DUMDK_INCLUDE_DIR="$PROBE_UMDK_INCLUDE" \
  -DPROBE_URMA="$PROBE_URMA_LIBRARY"
cmake --build build-ddr-probe-cpu --target kvaf_ddr_probe -j8
ldd build-ddr-probe-cpu/kvaf_ddr_probe
nm -D "$PROBE_URMA_LIBRARY" | grep -E 'urma_get_tp_list|urma_import_jetty_ex'
```

Prefill 机器，先加载 CANN 环境，再编译导出库：

```bash
export PROBE_UMDK_INCLUDE=/path/to/br_UMDK_24.12.0_LTS/include
export ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/ascend-toolkit/latest
cmake -S tests/diagnostics/shared_ddr_probe -B build-ddr-probe-npu \
  -DPROBE_ROLE=npu \
  -DUMDK_INCLUDE_DIR="$PROBE_UMDK_INCLUDE" \
  -DASCEND_CANN_PACKAGE_PATH="$ASCEND_CANN_PACKAGE_PATH"
cmake --build build-ddr-probe-npu --target kvaf_probe_hbm -j8
```

Decode 使用同一源码的正常 NPU 构建：

```bash
./build.sh npu
export PYTHONPATH="$PWD/build-npu/python${PYTHONPATH:+:$PYTHONPATH}"
export KVAF_BATCH_GET_KERNEL="$PWD/build-npu/bin/kvaf_execute_batch_get_kernels.o"
export KVAF_BATCH_PUT_KERNEL="$PWD/build-npu/bin/kvaf_execute_batch_put_kernels.o"
```

Decode 建链校验需要 BatchPut kernel，即使本测试的数据读取使用 BatchGet。
脚本会把 `--put-kernel`（或 `KVAF_BATCH_PUT_KERNEL`）显式传入 Runtime；仅设置环境变量
不会让原生 Runtime 自动读取它。启动前检查两个 kernel 文件存在、可读。

Prefill/Decode Python 环境需要 `torch`、`torch_npu`、`numpy`；Decode 还需要这个构建的 `kvaf`。
Prefill 导出库自己加载 Direct HCCP，建议和 Decode 使用独立进程、独立 NPU。
可以先在同一 NPU 主机上用两张卡跑通；三机结果仍须在目标拓扑上复测。

## 三端执行

示例形状：`token_bytes=656, rows=4096`，payload 为 2,686,976 B，
加上首尾各 4096 B 保护区，allocation 为 **2,695,168 B**。
三个进程的形状和 rounds 必须一致。每组先跑 3 轮，再跑 100 轮。

1. K：将 UB 设备和 EID index 替换为部署值。两个监听端口只供本次测试使用。

```bash
set -o pipefail
build-ddr-probe-cpu/kvaf_ddr_probe \
  single-plus-bit0 udma0 0 2695168 0.0.0.0 29990 29991 default \
  --read-chunk-bytes 4096 \
  2>&1 | tee single-plus-bit0.k.log
```

`PAGE_MODE` 参数 `default` 使用现有 service 的普通页模式，改成 `thp` 可复测其 THP 分配路径。
THP 是另一个变量，不能由 `non_pin` 的数值推断。测试端口是无鉴权诊断接口，只在隔离测试网络使用。

`--read-chunk-bytes` 是必填的 K 端实验参数，无默认值；软件允许范围是十进制整数 1..65536 B。
示例的 4096 B 是待验证的分块候选，不代表已经查询或确认设备能力上限。
请按部署资料核对候选单次 READ 长度；本实验每条 READ 固定源、目的各一个 SGE，仍检查
Jetty 创建、post、CQ 和最终数据。若缺少能力资料，全轮通过只能证明该显式候选在本次部署可用。
小分块会增加串行请求数量；其超时不能直接解释为硬件不支持 READ。本程序不做吞吐验收。
启动日志打印 `capability_source=explicit_candidate hardware_limits_queried=0`，
不会自动查询能力或按未知硬件能力下调配置值；最后一块仅读取剩余字节，失败后不改用另一长度重试。

2. Decode：替换 K 的 IP、NPU device ID、该 NPU 的 32 位十六进制 EID、K UB 设备名。

```bash
python3 tests/diagnostics/shared_ddr_probe/run_probe.py decode \
  --control K_IP:29991 --cpu K_IP:29990 \
  --device 0 --eid DECODE_EID_32_HEX --peer-ub-device udma0 \
  --get-kernel "$KVAF_BATCH_GET_KERNEL" \
  --put-kernel "$KVAF_BATCH_PUT_KERNEL" \
  --token-bytes 656 --rows 4096 --rounds 3 \
  2>&1 | tee single-plus-bit0.decode.log
```

3. Prefill：替换自己的 device ID 和可达 EID。

```bash
python3 tests/diagnostics/shared_ddr_probe/run_probe.py prefill \
  --control K_IP:29991 --device 1 --eid PREFILL_EID_32_HEX \
  --exporter "$PWD/build-ddr-probe-npu/libkvaf_probe_hbm.so" \
  --token-bytes 656 --rows 4096 --rounds 3 \
  2>&1 | tee single-plus-bit0.prefill.log
```

两端默认等待 120 秒；如果启动间隔较长，在两条 Python 命令都加 `--timeout 600`。
`--cpu` 必须指向 K 的 KVAF 端口（示例 `29990`），`--control` 指向诊断端口（示例 `29991`）。
脚本会在连接前拒绝两个参数指向同一端点，避免把不同协议发给同一个监听器。
若当前部署需要额外 Runtime 参数（如 `reuse_device_ra`），可通过 Decode 的
`--runtime-options /absolute/path/options.json` 传入 JSON 对象；这不会更改控制协议版本。
kernel 路径优先级是显式命令行参数、JSON 中的对应路径、环境变量。

成功后，Prefill 打印 `PASS`，Decode 打印 `ALL_PASS`，K 每轮打印 `ROUND_PASS`。
两端正常退出后停止 K，再把 case 改成 `dual-compatible`，重复三端启动。
也应单独跑 `single-current` 复现 KVAF 的原始 0xe 行为，以识别实际库与原版源码的差异。
按矩阵继续测试，日志名随 case 更改；任一组若在注册阶段失败，记录阶段和 errno，
不必启动 Prefill。某一组被拒绝不等于所有单 MR 或双 MR 方案失败。

## 一轮实际验证的内容

```text
Prefill 在自己的 HBM 产生 round N 数据并同步完成
    → 控制消息 PULL(N)
K 提交 URMA READ
    src = Prefill imported HBM segment
    dst = KV allocation + 4096，tseg = 本组 local MR
    → 每个分块均等到匹配且成功的 CQE
    → 发布 round N 完成
Decode 真实 batch_get
    addresses[0] = 同一 KV allocation 的导出基址 + 4096
    offsets = 随机排列的 token 字节偏移
    → 完整、空、部分 miss_count
    → 反向 output_indices，首尾输出保护行
    → NPU 后继 bitwise_xor 消费，再 D2H 全量比较
    → ACK(N)
K 此时才读取 CPU mapping 检查 payload 和首尾保护区
    → 允许 Prefill 开始下一轮
```

测试使用显式地址 Tensor 重载，绕过 key→slot 元数据提交，只检验本问题的内存与传输能力。
K READ 直接写入 KV 池，无 payload staging 和 `memcpy` 补拷贝；控制 TCP 不传 KV 数据。
K 不在 Decode 读取前预读或 flush payload，避免检查本身掩盖 DMA→远端 READ 的可见性问题。
后续轮次会在 K CPU 缓存已读取旧数据的条件下复测；这里只观察现有平台一致性行为，不猜测缓存指令。
若可见性检查失败，应查平台缓存契约后再加入必要维护，不能先 memcpy 再宣称直接落池通过。

`PULL_CQ_OK` 仅证明 K 传输完成；`ROUND_PASS` 才包括 Decode、K CPU 与保护区检查。
这是一项串行完成顺序试验：不会让 K 写和 Decode 读同一字节范围同时发生。
跨槽并发、失败恢复、元数据 commit、完整业务 Pull API、超大 DDR 池 pin 资源消耗不在本次通过结论内。

## 结果判定

| 观察 | 可得结论 |
|---|---|
| 单 MR 注册成功，但 READ CQ 错误 | 注册接受不等于支持该数据路径，保留 CQ 状态与版本日志 |
| 单 MR 全轮 `ROUND_PASS` | 此环境/页模式/规模下，同一个 MR 同时承担两个用途 |
| 双 MR 第二次注册失败 | 此参数/context/order 下不允许该组合；不能外推全部双 MR 方案 |
| 双 MR 注册成功但数据检查失败 | 不能宣布双 MR 方案可用 |
| 双 MR 全轮 `ROUND_PASS` | 此环境下，同一 allocation 的两个注册视图可用于这条顺序数据路径 |
| `0xe` 被拒绝、`0xf` 注册成功 | 与 24.12 core 的 LOCAL_WRITE 组合校验一致；仍须完成数据链路检查 |
| `0xf` 被拒绝、`0xe` 成功 | 实际库/部署补丁与本次取得的原版 core 行为不同，需定位差异，不能据此宣布采用 LOCAL_ONLY 语义 |

建议留存表：case、页模式、allocation bytes、头文件宏、liburma/provider/驱动/固件/CANN 版本、
两个注册的 flags/errno、首次失败阶段/CQ 状态、成功轮数、三端日志。

### 独立 DDR 首块对照

当 `dual` 在 SGE24 和 SGE32 下都于首条 READ 返回本地访问错误时，可在 K 命令末尾添加
`--independent-ddr-control`。它是额外诊断开关，不是新的 MR case，默认关闭，仅支持 `default` 页模式。
保留本轮卡、EID、SGE32 和 4096 B 分块，三端启动全新进程；Prefill、Decode 命令不变，无需因此重编。

```bash
# 沿用现有构建缓存的 UMDK include 和 liburma 路径。
cmake -S tests/diagnostics/shared_ddr_probe -B build-ddr-probe-cpu \
  -DPROBE_EXTENDED_SGE=ON
cmake --build build-ddr-probe-cpu --target kvaf_ddr_probe -j8

env URMA_LOG_LEVEL=debug build-ddr-probe-cpu/kvaf_ddr_probe \
  dual udmac0d1e2 0 2695168 0.0.0.0 29990 29991 default \
  --read-chunk-bytes 4096 --independent-ddr-control \
  2>&1 | tee dual.sge32.independent.k.log
```

`PROBE_READY` 应包含 `wire_sge_bytes=32 independent_ddr_control=1`（字段可不相邻）。
收到首次 PULL、Prefill 已完成 bind 和 Pattern 更新后，K 额外分配与原 KV allocation **同大小**
的独立 DDR，仅注册一次 `access=1/non_pin=0`，使用相同 context、token value 和新的 token ID。
与 default KV DDR 一样，在注册后初始化内存；首块预填期望值逐字节取反，其余字节为保护值。
使用同一 HBM 源、源 MR、Jetty、SGE 布局和首 WR 长度发起 READ，并在成功 CQ 后逐字节校验。
它不 CPU 读取原 KV payload，也不把控制结果 memcpy 到原 KV DDR。

- `REGISTER role=independent_ddr`：独立注册结果；不等于 READ 成功。
- `INDEPENDENT_DDR_CONTROL_PASS ... scope=first_chunk_only shared_ddr_verified=0`：首块进入独立 DDR 且 CPU 数据/保护区正确。
- 随后仍执行原 KV DDR READ；错误日志的 `target=independent_ddr/kv_pool` 区分两个目标。
- 只有原来的共享 DDR 全轮 `ROUND_PASS` 及两端最终成功，才满足共享数据路径验收；控制成功不发布 round/ACK。

控制成功而原 KV DDR 失败，可将问题收敛到两个目的 allocation/注册映射等差异，不能唯一认定重复 MR 不支持。
控制也报同样错误，说明“必须使用原共享 allocation 才会触发”不是必要条件；仍需核对通用注册、
地址转换、READ 请求/导入契约和 provider，不能直接判定 HBM 不可读。
原 KV 的两个 MR 此时仍保留在同一 context，因此这个结果也不能排除它们对 context 整体的影响。
该控制会预热源 HBM/队列/映射；启用时 `PULL_CQ_OK` 和 `ROUND_PASS` 都标记
`independent_ddr_control=1`。若共享路径随后通过，应再使用全新进程、不带该开关复测原始路径。
额外 DDR/MR/token 在控制失败时保留；正常 CLOSE 先销毁队列和导入资源，再按 MR、token、内存顺序释放。

### 用户态库日志

真机加载列表已观察到 `/usr/lib64/liburma.so.0.0.3`、`libummu.so.1.0.5`、
`/usr/lib64/urma/liburma-udma.so` 与 `liburma_ubagg.so.0.0.3`。
文件名不能证明具体 UMDK 提交或当前设备实际采用哪个插件；需保留部署配套信息。
内核的 `Adding to iommu group` 是设备入组信息，不是本次 READ 的访问故障证据。

仓库固定提交的 [URMA core 日志实现](https://gitee.com/openeuler/umdk/blob/8cb685a80bf92e6287c3262d46b3d513fb3a3302/lib/urma/core/urma_log.c)
使用字符串 `URMA_LOG_LEVEL=debug`，默认写 syslog，故 `tee` 和 `dmesg` 未必收集到这些日志。
上面的命令启用该开关；复测后及时收集用户态时间窗口：

```bash
journalctl --since '5 minutes ago' --no-pager -o short-precise > dual.sge32.independent.journal.log
grep -Ei 'URMA_LOG_TAG|UDMA_LOG_TAG|urma|udma|ummu' \
  dual.sge32.independent.journal.log > dual.sge32.independent.urma.log
```

若 journal 没有记录，按 `/etc/rsyslog.conf`、`/etc/rsyslog.d/` 中的规则查实际落盘日志；
不能将 journal 或 dmesg 为空解释为 provider 没有错误。
部署 UDMA/UB 聚合插件可能有独立 logger，该开关不能保证覆盖全部插件。

### 失败定位与资源保留

失败或超时时程序不会把超时当作 DMA 已停止。Python 进程保留 Tensor，K 保留已武装的资源；
先停止 K 诊断进程，再中断 Prefill/Decode，最后启动全新三进程。不要强行复用原地址继续测。
该保留策略也适用于 Python 检查失败，需人工结束测试进程；正常成功路径会按依赖顺序释放。
kernel 路径、明显的端口冲突等本地参数错误在 NPU 初始化和 RPC 之前退出，不向 K 发送 FAIL。
如果 K 日志只有 `PROBE_FAIL op=7`，表示某个客户端上报失败，应先看该客户端的首个 traceback；
此时日志附带的 errno 不足以认定发生了 MR 注册失败。`op=1` 是另一个客户端查询状态时读到失败。

例如 `CQ_FAILURE status=4 context=1 expected=1` 中，请求标识匹配；按编译使用的 24.12
头文件，状态 4 是 `URMA_CR_LOC_ACCESS_ERR`（本地内存访问错误），不是 context 不匹配。
现分别报告 `READ CQ returned error status` 和 `READ CQ user_ctx mismatch`，并保留原始状态、
请求标识、轮次、分块范围及两个 SGE 的地址/MR。post 成功仅表示提交被接受，错误完成不发布 round。
CTP 查询与 Jetty 导入返回成功，也不代表实际 READ 已成功访问 HBM 或写入 DDR。
已观察到 `single-current` 与 `dual` 都在首个 READ 返回该状态；这说明增加独立本地 MR 仍未解决问题，
不能只归因于 0xe 缺少 LOCAL_WRITE，也不能据此宣布 CPU READ HBM 不可行。
此时优先核对部署的 SGE/WR/CR 布局、实际本地 MR 的范围/context 和 provider 记录，
避免同时轮换 MR case、EID、卡和分块，导致结果无法归因。
`READ_LOCAL_MR layout=compiled_header` 打印按编译布局解读的本地 MR 范围/context/属性，
用于与 REGISTER 和提交 context 对照；它本身不能认证目标库 ABI。

Prefill 在发送 EXPORT 前打印 `event=HBM_EXPORT`，包含 `configured_eid`、
`segment_eid_prefix`、`jetty_eid_prefix` 和两个完整 key。后两个前缀按本诊断所需 ABI
取 key 的前 16 B；原始 key 用于进一步核对部署 ABI。
`configured_eid`（命令行 `--eid`）供两端查询 CTP 路径使用，可以与实际资源 key 中的 EID 不同。
例如已观察到配置 CLOS EID 为 `00000000003f030000100000df00ab00`，而 HBM MR 和 Jetty
都返回 `000000000001030000100000df00a200`。这不证明任意两种 EID 均可互换或可达。
K 比较的是 HBM segment 与实际导出的 Jetty key 中的资源 EID；不要求它们等于配置 EID。
接收成功后打印 `SOURCE_DESCRIPTOR route_eid=... mr_eid=... jetty_eid=...`，
仍将原配置 EID 用于 CTP 查询，并原样导入两个 key。
若出现 `HBM segment EID differs from exported Jetty EID`，表示两个资源身份确实不一致；
应核对部署 ABI 和源端资源，不能改写 MR key 或删除检查来绕过。
升级这项 K 端校验需要重新编译 `kvaf_ddr_probe`；Prefill 导出库和 Decode 无需因此重编。

若失败于 `query pull CTP`，查看 K 的 `PULL_CTP_QUERY`：它打印实际查询的 local/peer EID、
flag/mode、容量、API 返回码 `rc`、输出 `count`、`tp` 和即时保存的 `errno`。
`result=api_error` 表示 `rc != 0`；`empty_list` 表示 API 成功但 `count=0`；
`zero_handle` 表示有条目但 handle 为 0。成功要求 `rc=0`、`count>0` 且 `tp!=0`，
不能单凭非零 errno 判断失败。查询前清零 errno，避免将先前操作的错误误归到 CTP。
程序仍只查询配置的 EID 一次，不自动改用资源 EID 或切换 K 的 UB 设备。
若查询日志中 flag/mode 和双端 EID 全零，不能继续按普通路径不可达判断。
`DecodeSourceDescriptor` 应构造 flag=1、RM 模式及非零 EID；当前程序在解码后保存独立 heap 快照，
在 import_seg、三个队列创建调用后逐次输出 `PULL_CTP_STATE stage=... unchanged=...`。
首次变化立即失败，绝不重新填回字段继续执行。`PULL_CTP_QUERY_BEGIN` 在实际查询之前打印，
避免把查询返回后已被破坏的输入快照误认为真实入参。

编译头文件来自下载的 24.12 源码、运行库来自系统 `/usr/lib64`，不能单凭头文件宏认定 ABI 配套。
真机已观察到 `urma_query_device` 返回成功，却写出了该头文件声明的 312 B 输出边界；
它没有输出长度参数，旧布局不能安全解释部署库的完整属性结构。
本版删除这次非必需的能力查询，改为显式分块候选；不再分配大缓冲后继续按旧布局读字段。
这消除了已定位的越界调用，不代表其余所有 ABI 已认证兼容；SGE/Jetty/segment 等仍需核对。
CTP config/count/info 和 READ 的 CR 输出使用带前后哨兵的独立 heap 缓冲，打印
`NATIVE_ABI_GUARD op=... value_bytes=... before_ok=... after_ok=...`。
CR 轮询只在哨兵异常时打印边界日志，以免空轮询刷屏；每次返回均先检查哨兵，再解释 CR。
检测到边界改写时，在使用返回字段之前立即停止；哨兵填充仅用于定位，不是 ABI 兼容层，
即使完整也不能证明所有字段偏移与部署库一致。若后续需要恢复自动能力发现，应使用与运行库
准确配套的部署头文件，或以该头编译仅输出固定宽度标量的独立适配层。

Decode 的 CTP 成功或 Prefill→K 的查询成功，不能代替验证 K→Prefill 的查询。
可用全新 K、同一 UB device/EID/case，将 NPU 5 临时作为 Decode（使用它自己的配置 EID）
做普通通道初始化对照；此时不启动 Prefill，出现 `ARMED` 后等待 round 1 是预期行为。
取到阶段日志后先停 K，再中断 Decode，并为完整三端测试重新启动所有进程。

## 无硬件的本地检查

```bash
cmake -S tests/diagnostics/shared_ddr_probe -B build-ddr-probe-host \
  -DPROBE_ROLE=host-test -DUMDK_INCLUDE_DIR="$PROBE_UMDK_INCLUDE"
cmake --build build-ddr-probe-host --target kvaf_probe_host_test -j8
build-ddr-probe-host/kvaf_probe_host_test
python3 -m unittest discover -s tests/diagnostics/shared_ddr_probe -p 'test_*.py'
```

host 测试覆盖九组注册视图、先后顺序、同 allocation 校验、跨 context、分块 READ、
多轮数据更换、post/CQ 错误不发布完成，以及第二个 MR 注册失败时回滚。
READ 完成回归另覆盖状态错误但请求号匹配、成功状态但请求号不匹配、poll 返回负数、
容量为 1 却返回多条以及 CR 输出越界；均验证不继续提交后续分块、不发布完成轮次。
独立 DDR 回归覆盖注册配置与源/首 WR 长度保持、目的 allocation 隔离、成功后继续原共享 READ、
控制 CQ/数据错误时不发布完成且保留资源，以及已停止的 fake DMA 下显式释放资源无泄露。
源描述符回归使用上述真机返回的 48 B segment 和 32 B Jetty key 作为固定输入，
验证配置 EID 与资源 EID 不同但 MR/Jetty 身份一致时可通过，CTP 查询仍保留配置 EID；
同时拒绝真实资源身份不一致、错误 key 长度/RM 模式、越界、地址溢出及全零 EID。
CTP 查询回归覆盖成功和三种失败分类、清除陈旧 errno、立即捕获错误及单次调用保留双端 EID。
内存诊断回归覆盖前后哨兵破坏、逐阶段查询数据损坏、实际查询前拒绝全零入参，
以及 CTP 三个输出缓冲越界和库内正常改写 config 本体的区分。
显式分块回归覆盖合法边界以及缺失、零、负数、非整数和超限/溢出输入；参数在 Runtime 初始化前校验。
另用明确标注的 24.12 core 校验模型检查：0xe 拒绝、0xf 接受。数据搬运 fake provider
只检查测试工具流程，不声明真实 provider 支持这些配置。
控制测试覆盖二进制布局、碎片响应、错误响应、版本不匹配和数据版本变化。
启动测试覆盖 Runtime 构造参数包含 BatchPut kernel、CLI/JSON/环境变量优先级，
以及缺少文件、端口冲突时不发 RPC、不污染 K 会话。
这些检查验证的是测试工具，最终驱动兼容性仍以目标机三端结果为准。

本次本地检查（2026-10-27，macOS）：已改用上述 24.12 提交的真实头文件，
两个 C++ 主体文件通过 `clang++ -fsyntax-only`；fake-provider 可执行测试通过；
Python 控制和启动回归测试于 2026-10-28 全部通过（14 项），控制协议未改变。
macOS 检查仅为编译提供 `MADV_HUGEPAGE` 常量，不执行 Linux 页管理操作。
2026-10-29 补充检查：使用同一 24.12 提交的真实头文件，通过 clang++ 编译并运行新增源描述符
回归及原 fake-provider 测试。当前本地环境没有完整 Linux UMDK/CANN 链接环境，
这些检查不能代替目标机完整构建、建链、READ 和全轮数据验证。
