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
仅在确认部署 provider 使用参考 CPU push 的扩展 **32 B SGE** 时，设置 `-DPROBE_EXTENDED_SGE=ON`。
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
  2>&1 | tee single-plus-bit0.k.log
```

末参数 `default` 使用现有 service 的普通页模式，改成 `thp` 可复测其 THP 分配路径。
THP 是另一个变量，不能由 `non_pin` 的数值推断。测试端口是无鉴权诊断接口，只在隔离测试网络使用。

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

失败或超时时程序不会把超时当作 DMA 已停止。Python 进程保留 Tensor，K 保留已武装的资源；
先停止 K 诊断进程，再中断 Prefill/Decode，最后启动全新三进程。不要强行复用原地址继续测。
该保留策略也适用于 Python 检查失败，需人工结束测试进程；正常成功路径会按依赖顺序释放。
kernel 路径、明显的端口冲突等本地参数错误在 NPU 初始化和 RPC 之前退出，不向 K 发送 FAIL。
如果 K 日志只有 `PROBE_FAIL op=7`，表示某个客户端上报失败，应先看该客户端的首个 traceback；
此时日志附带的 errno 不足以认定发生了 MR 注册失败。`op=1` 是另一个客户端查询状态时读到失败。

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
源描述符回归使用上述真机返回的 48 B segment 和 32 B Jetty key 作为固定输入，
验证配置 EID 与资源 EID 不同但 MR/Jetty 身份一致时可通过，CTP 查询仍保留配置 EID；
同时拒绝真实资源身份不一致、错误 key 长度/RM 模式、越界、地址溢出及全零 EID。
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
