# A5 AIV + SIMT + URMA + H2D 最小集合

## 1. 结论和状态

本分支实现的是完整 KVAF 的第一条数据面最小闭环：

```text
预先准备的 Device AccessTable
    -> AIV 内启动 32-lane SIMT 查表
    -> 生成固定 64B H2D 描述符
    -> Host 从 READY channel 快照 SQ/CQ/EID/TP/token
    -> 单 AIV 填 64B URMA READ WQE
    -> AIV 写 SQ 并敲 SQ doorbell
    -> Host DDR 直接读入本地 HBM
    -> AIV 按 phase/owner 轮询 CQ 并敲 CQ doorbell
    -> Device Completion
```

这里：

- **AIV** 是 AI Vector Core；
- **SIMT** 是 AIV 内的多线程执行模式；
- **URMA 控制面**仍由 Host 侧 HCOMM 创建和交换 Endpoint、MR、EID、TP、
  Jetty、SQ 和 CQ；
- **URMA 数据面**不再调用设备侧 `AscendC::Hcomm`，而是由单个 AIV
  producer 直接写 HCOMM 创建的 SQ/CQ 和 doorbell；
- **H2D** 指 Host DDR 到 NPU HBM，不是先拷入一个 Host staging buffer
  再调用普通 ACL H2D copy。

当前状态必须分层表述：

| 层次 | 状态 | 证据 |
|---|---|---|
| 固定设备 ABI 与 CPU 参考规划器 | 已实现 | 默认库和单元测试 |
| SIMT 查表与 64B 描述符生成 | 已实现代码 | 服务器 B 需用 bisheng 编译 |
| Host DDR HVA→DVA 显式映射 | 已实现代码 | `halHostRegister(HOST_MEM_MAP_DEV)` |
| Host MR 导出、Device endpoint 导入 | 已实现代码 | `HcommMemExport/Import` |
| CPU/AIV UBC_CTP channel | 已实现代码 | 服务器 B 需验证 channel READY |
| SQ/CQ/EID/TP/token 快照与校验 | 已实现代码 | 服务器 B 需验证 `DIRECT_URMA_READY` |
| AIV 直写 SQ/CQ 的 URMA READ | 已实现代码 | 服务器 B 需验证数据与 Completion |
| PC A 本地验证 | 已通过 | Apple clang CPU 单测和脚本语法检查 |
| A5 硬件端到端验证 | 尚未执行 | PC A 没有 CANN、bisheng 和 A5 |

因此，合入前的准确说法是“**代码闭环已实现，本地可验证部分已通过，A5
硬件验收待服务器 B 执行**”，不能把它描述成已经在 A5 上跑通。

本最小集采用设计文档 §3.4 的具体图边界：Replay 从 capture 前已稳定放在
NPU 上的 AccessTable 开始，`PrepareAccessTable`、Key 展开和 Meta 查询在
图外。本分支尚未验证 capture/replay，因此也不能笼统声称已满足文档中的
全部图执行约束。

## 2. 为什么采用 “Host HCOMM 建链、SIMT 规划、AIV 直写队列”

HCOMM 已经提供资源生命周期和对端交换，SHMEM 则证明 AIV 可以消费
HCOMM channel 中的 SQ/CQ/remote-MR 元数据并直接提交 WQE。直接让 32 个
SIMT lane 竞争同一个 SQ head、doorbell 和 CQ tail 会形成多 producer
竞争，所以本实现仍把队列所有权收敛到外层单个 AIV。

本实现使用单 producer 规则：

1. 一个 AIV block，固定 32 个 SIMT lane；
2. 每个 lane 只读取 AccessTable、Index、Mask 和输出偏移；
3. 每个 lane 写自己的 64B 描述符，不操作 SQ/CQ；
4. `asc_vf_call` 返回后，外层 AIV 检查整个 batch；
5. Host 在 kernel 前把 READY channel 的 SQ/CQ、EID、TP、remote token 和
   MR 边界和 HCOMM 软件队列地址快照成 192B
   `AivDirectUrmaChannel`；
6. 只有所有 descriptor 同时落在该 local/remote MR 内，AIV 才提交；
7. AIV 每次填一个带 remote token、strong-order 和 CQE 请求的 64B
   READ WQE，写 SQ 后
   敲门铃；
8. AIV 按真实 queue depth 计算 owner/phase，检查 CQE
   `status/substatus`，成功后更新 CQ doorbell。

这保留了 SIMT 并行查表能力，同时让一个 AIV 成为 SQ/CQ 的唯一软件
producer/consumer。MVP 逐条提交并等待 CQ，优先保证方向、token、回卷和
错误传播正确；批量无 CQE WQE 加尾部 fence 是后续性能优化。

核心文件：

- `access_operator/include/kvaf/access_operator/aiv_simt_urma_h2d.h`
- `access_operator/src/aiv_simt_urma_h2d.cpp`
- `access_operator/a5/src/aiv_simt_urma_h2d_kernel.cpp`

参考实现与本分支的对应关系如下（路径相对于 PC A 的 `agent/`）：

- `shmem/examples/simt_rma/main.cpp`：AIV 外层通过 `asc_vf_call` 启动
  SIMT；
- `shmem/CMakeLists.txt`：新旧 bisheng 的 `xasc/xcce`、A5 架构参数与
  fat-object 链接；
- `shmem/src/host/transport/device_udma/`：从 HCOMM channel 读取
  SQ/CQ/remote-MR 上下文并转成设备数据面 ABI；
- `shmem/src/device/gm2gm/engine/shmem_device_udma.hpp`：64B SQE、remote
  token、doorbell 和带 owner/phase 的 CQ 处理；
- `asc-devkit/impl/adv_api/detail/hcomm/impl/platform_v310/`：用于交叉核对
  raw READ opcode、WQE flag、token/EID/TP 字段和 CQE 状态；
- `UB_MicroBenchmark_new/src/operator/AIV_URMA/new_single_op/aiv_urma.h`：
  用于核对 SIMT 组 WQE、SQ 写入和 doorbell 框架，但不原样复制其固定
  depth、token=0、双 Jetty 和 CQ poll 假设。

A5 Host DDR 分配/注册、HCOMM endpoint/channel/MR 生命周期和 driver EID
发现均在 KVAF 内独立封装；构建和运行不要求下载其他业务代码仓。

### 2.1 `UB_MicroBenchmark_new` 的可复用边界

该项目是“可运行包较完整”，不是“URMA 控制面源码闭环”：

| 环节 | 源码完整性 | KVAF 选择 |
|---|---|---|
| Host DDR 分配和 NPU 映射 | 部分可见 | 保留思路，KVAF 自己管理长度、注册和释放 |
| Context/JFC/Jetty/TP 创建 | 核心在预编译 `liburma_perftest.so` | 不引入黑盒，改由 Host HCOMM 创建 |
| EID/Jetty/segment 交换与 bind | 核心在同一预编译 SO | 不引入黑盒，改由 HCOMM channel 交换 |
| SQ/CQ/doorbell 查询 | wrapper 可见，实际填充在 SO | 从 HCOMM READY channel 直接快照并校验 |
| AIV raw WQE/SQ/CQ | 源码可见 | 只借鉴提交框架，字段语义以 SHMEM/asc-devkit 交叉核对 |

不能直接复用 UB 数据面的原因包括：remote token 被写成 0、owner 使用固定
depth、SQ 回卷存在无符号下溢、CQ 没有完整 phase/错误传播、READ wrapper
方向不一致，以及示例 `is_H2D=false`。它的 H2D 分支还存在固定 16 MiB
映射与 1 GiB 注册配置不一致、硬编码 device ID、地址偏移和错误 allocator
释放等问题。

## 3. 固定 ABI 和最小语义

### 3.1 输入

`DeviceBatchGetParams` 只包含稳定设备地址和整数：

- `DeviceAccessTableView`
  - `remote_addresses[]`：Device endpoint 导入 Host MR 后得到的 NPU
    可访问地址；
  - `lengths[]`：每个 KV block 的字节数；
  - `valid[]`：条目有效位；
  - `capacity`、`generation` 和 ABI version。
- `indices[]`：请求到 AccessTable 行的索引；
- `mask[]`：0 表示跳过；
- `output_offsets[]`：在一块连续 HBM 输出中的字节偏移；
- `output_base/output_bytes`；
- 描述符区、请求数和描述符容量。

当前固定容量是 32 条，且单条长度不超过 `UINT32_MAX`。这是 A5 URMA SGE
长度字段的显式保护，也是 MVP 上限，不是完整 KVAF 的最终上限。
这里的 `output_offsets` 是对设计文档 `output_indices` 的 MVP 简化：它不是
`int32` 多维 Tensor，也不包含 KV tensor 的层/头/块 stride 计算。

### 3.2 描述符和完成态

`UrmaH2DDescriptor` 固定为 64B，包含：

- ABI version、状态、逻辑请求号；
- imported remote address；
- local HBM address；
- byte length、generation 和精确错误码。

`DeviceCompletion` 固定为 64B，包含 batch 状态、规划/提交/完成条数和
`first_failed_index`。规划阶段采用整批成功或整批失败语义；发现越界、
无效 AccessTable 行、stale generation 或地址溢出时，不提交任何 URMA
请求。

## 4. Host DDR 注册与建链流程

### 4.1 Host 角色

1. 动态加载 `libascend_hal.so` 并解析分配、注册和释放符号；
2. 调用 `halMemAlloc(MEM_HOST | MEM_TYPE_DDR | MEM_PAGE_NORMAL)` 分配并
   填充 Host DDR；
3. 调用：

   ```text
   halHostRegister(host_hva, bytes, HOST_MEM_MAP_DEV, logical_device, &dva)
   ```

   失败时直接终止，不允许退回 HVA；
4. 创建 `ENDPOINT_LOC_TYPE_HOST`、`COMM_PROTOCOL_UBC_CTP` endpoint；
5. 以 `COMM_MEM_TYPE_HOST` 将 HVA 注册到 HCOMM；
6. `HcommMemExport` 导出不透明 MR 描述；
7. 等待 Device 完成 `COMM_ENGINE_AIV + SERVER` channel 创建；
8. 创建 `COMM_ENGINE_CPU + CLIENT` channel，并在两端轮询建链状态；
9. 保持 DDR、DVA mapping、MR、endpoint 和 channel 存活，直到 Device
   完成。

`mapped_dva` 会写入 Host 日志，作为 DDR 确实映射进 NPU 地址空间的证据。
它只用于验证并维持 HVA→DVA mapping 生命周期：Host endpoint 仍按
`COMM_MEM_TYPE_HOST` 注册 HVA，真正写入设备 AccessTable、提交给 URMA 的
地址是 Device endpoint 执行 `HcommMemImport` 后得到的 `view.addr`。不能把
这两个地址视为同一个值。

### 4.2 Device 角色

1. `aclrtMalloc` 分配 HBM 输出，并先写 canary；
2. 创建 `ENDPOINT_LOC_TYPE_DEVICE` endpoint；
3. 以 `COMM_MEM_TYPE_DEVICE` 注册 HBM；
4. 创建 `COMM_ENGINE_AIV + SERVER` channel；
5. 调用一次 `HcommChannelGetStatus` 启动 SERVER 建链状态机，并确认状态为
   `CONNECTING` 或 `READY`；
6. 原子发布 server channel 已启动标记，允许 Host 再创建 `CLIENT` channel；
7. Host 和 Device 都使用 `exchangeAllMems=true`，且所有本地 MR 都在建链前
   注册；
8. 轮询 `HcommChannelGetStatus`，只有 READY 才继续；
9. 导入 Host MR 描述，取得 `HcommMemImport` 返回的 `view.addr`；
10. 从 AIV channel 的 Device `ChannelEntity` 读取第一组 SQ/CQ 和全部
    local/remote MR 元数据；
11. 用 `view.addr/bytes` 精确匹配 remote Host DDR MR，用输出 HBM 精确
    匹配 local MR；拒绝错误 token、长度或边界。当前 HCOMM UBC/URMA
    `MemoryImport` 源码只回填 `addr/size`，没有回填 `type`，因此 KVAF
    依据本次明确的 Host 描述符交换契约记录 `kHostDdr`，不读取这个未定义
    的返回字段；
12. 校验 64B WQE/CQE、power-of-two depth、空队列、非零 EID、24-bit TP
    和 20-bit token，再生成 `AivDirectUrmaChannel` 并启动 kernel。

本冒烟程序显式保证 SERVER 先创建。HCOMM 的 `HcommChannelCreate` 只创建
通道对象，后续 `HcommChannelGetStatus` 才推进建链状态机；如果 Host
CLIENT 在 Device SERVER 尚不存在时先发起，Host 可能直接进入
`FAILED(2)`，而 Device 只会等待超时。

`view.addr` 才能进入设备 AccessTable。Host HVA、显式映射 DVA、公开逻辑
地址和 imported view 是不同地址域，不能混用。

Device endpoint 的 `physical_device_id/super_device_id/server_index/
super_pod_index` 必须来自当前服务器的真实 ACL/runtime 拓扑。冒烟程序允许
用环境变量传入，默认 0 只适合这些字段确实全为 0 的单机拓扑。

对应实现：

- `access_operator/a5/include/kvaf/access_operator/a5/hcomm_urma_resources.h`
- `access_operator/a5/src/hcomm_urma_resources.cpp`
- `access_operator/a5/smoke/aiv_simt_urma_h2d_smoke.cpp`

## 5. Device 数据面流程

1. AIV 初始化 Completion，并校验 ABI、容量和所有 GM 地址；
2. `asc_vf_call` 启动 32 个 SIMT lane；
3. lane 并行执行：
   - mask；
   - index bounds；
   - AccessTable valid；
   - remote range 和 output range；
   - 生成 64B 描述符；
4. AIV 扫描描述符。任何一条失败都会设置
   `first_failed_index` 并退出；
5. AIV 校验 192B direct channel ABI，再对所有 active descriptor 做
   local HBM MR 和 remote Host DDR MR 的整批 preflight；
6. 对每个 active descriptor，AIV 在 UB 清零并填一个 64B WQE：

   ```text
   opcode       = URMA READ (6)
   local SGE    = NPU HBM destination
   remote VA    = imported Host DDR source
   tokenEn      = 1
   remote id    = HCOMM remote MR token id
   token value  = HCOMM remote MR token value
   TP / EID     = HCOMM SQ context
   flag         = strong-order + CQE
   owner        = absolute_sq_head / actual_sq_depth
   ```

7. AIV 用 MTE3 把 WQE 写入
   `sq_base + (head % sq_depth) * 64`，先把
   `packed(next_head,next_wqe_count)` 写回 HCOMM `headAddr`，再把新 head
   写入 SQ doorbell；
8. AIV 从 `cq_base + (tail % cq_depth) * 64` 读取 CQE，按实际 depth
   计算 expected phase/owner；超时或 `status/substatus != 0` 会写入
   descriptor/Completion 错误；
9. 成功后把 CQ tail 写回 HCOMM `tailAddr`、写 CQ doorbell，并同步 HCOMM
   SQ `tailAddr`，再标记 descriptor complete；
10. smoke Host 控制程序同步 stream，回读结果、descriptor 和 Completion；
11. 正向 smoke 默认把 payload 分成最多 4 个分段，反向排列 AccessTable
    index，并额外加入一个带非法 output offset 的 masked lane；
12. 校验 SQ head/CQ tail 增量、每个 active descriptor、masked
    descriptor、完整 payload以及输出尾部 canary。

“整批失败”是 API 可见语义，不代表物理写入原子性。所有地址/MR 错误会在
第一条 doorbell 前被拒绝；如果硬件在后续某条 CQE 上报错，HBM 中仍可能
已有前面 descriptor 的数据；
调用方必须在 Completion 失败时丢弃整个输出。

当前逐条 post+poll，使 outstanding WQE 最多为 1；这是正确性 MVP，不是
吞吐最优实现。批量 post、SIMT 并行填 WQE、多 Jetty 和多 QP 需要在硬件
验收后单独增加。

## 6. PC A、服务器 B、远端仓 C 操作

### 6.1 PC A：开发和非硬件测试

```bash
cd /Users/chenshuang/codes/agent/KVAF
git fetch origin
git switch A5_AIV_URMA_cs
git pull --ff-only

cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

如果 PC A 没有 CMake，可直接验证 CPU 参考规划器：

```bash
mkdir -p /tmp/kvaf-aiv-test
clang++ -std=c++17 -Wall -Wextra -Werror \
  -Iinclude -Iaccess_operator/include \
  access_operator/src/aiv_simt_urma_h2d.cpp \
  access_operator/tests/aiv_simt_urma_h2d_test.cpp \
  -o /tmp/kvaf-aiv-test/kvaf_aiv_simt_urma_h2d_test
/tmp/kvaf-aiv-test/kvaf_aiv_simt_urma_h2d_test
```

预期：

```text
KVAF_AIV_SIMT_URMA_H2D_UNIT_OK
```

### 6.2 服务器 B：拉取、构建和 A5 验收

在服务器 B 宿主机更新共享代码；不要再建临时容器：

```bash
cd /data/c00982601/KVAF
git fetch origin
git switch A5_AIV_URMA_cs
git pull --ff-only
git log -1 --oneline
```

构建和硬件运行都在现有 `vllm-ascend-cs-kvaf` 容器内执行。容器和宿主机
均挂载 `/data/c00982601/KVAF`，因此不需要在容器内再次 clone：

```bash
docker exec -it -u 0 vllm-ascend-cs-kvaf bash

cd /data/c00982601/KVAF
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64:${LD_LIBRARY_PATH:-}
unset ASCEND_RT_VISIBLE_DEVICES

export KVAF_A5_BUILD_DIR=/data/c00982601/KVAF/build-a5-direct-urma
export KVAF_ASC_DEVKIT_PATH=/data/c00982601/asc-devkit-9.1.0-20260722
bash scripts/build_a5_aiv_urma.sh
```

本实现不再包含设备侧 `hcomm/hcomm.h`，所以**不要求仅为
`COMM_PROTOCOL_UBC_CTP/ChannelHandle` 升级 CANN**。第一轮应优先使用容器
内的 CANN runtime/`libhcomm.so`/`libascendcl.so`，并沿用此前已证明能与
bisheng 完成 xasc 编译的 `asc-devkit-9.1.0-20260722` 基础/SIMT 头。若该
目录不存在，再检查原路径：

```bash
test -r /data/c00982601/asc-devkit/include/simt_api/asc_simt.h
export KVAF_ASC_DEVKIT_PATH=/data/c00982601/asc-devkit
export KVAF_A5_BUILD_DIR=/data/c00982601/KVAF/build-a5-direct-urma-fallback
bash scripts/build_a5_aiv_urma.sh
```

不能混装另一套完整 CANN 到容器目录。Host `libhcomm.so`、ACL runtime、
driver 和 AICPU 包仍必须彼此兼容；`asc-devkit` 在这里最多只覆盖编译期
AscendC 基础/SIMT 头。

运行脚本会先查询 logical/physical device 映射和 HCOMM endpoint 拓扑。
如果没有显式设置 EID，它会调用 KVAF 自带的最小 resolver，通过
`/run/lock/kvaf_asdrv_ub_pair.lock` 协作锁串行化对
`/proc/asdrv_ub/dev_id` selector 的写入和 `pair_info` 快照读取，再把
`local_eid` 作为 Host EID、`remote_eid` 作为 Device EID：

```bash
unset KVAF_HOST_EID KVAF_DEVICE_EID KVAF_PAIR_CHANNEL
export KVAF_LOGICAL_DEVICE_ID=0
export KVAF_H2D_BYTES=4096
bash scripts/run_a5_aiv_urma_h2d_smoke.sh
```

若驱动为同一物理设备报告多个 active pair，resolver 会拒绝猜测并列出
channel。确认实际 channel 后重跑：

```bash
export KVAF_PAIR_CHANNEL=<实际通道号>
bash scripts/run_a5_aiv_urma_h2d_smoke.sh
```

resolver 也可以单独检查；参数使用物理设备号：

```bash
python3 scripts/resolve_a5_urma_eids.py \
  --physical-device-id 0 \
  --shell
```

运行脚本调用 `scripts/query_a5_topology.py`，从 ACL/runtime 查询 physical
device、SDID、server ID 和 super-pod ID。单独检查：

```bash
python3 scripts/query_a5_topology.py --logical-device-id 0 --shell
```

`KVAF_SUPER_POD_INDEX=4294967295`（`0xffffffff`）表示 runtime 未配置或
无法提供 SuperPod，而不是实际索引。本单机 H2D 用例按 HCOMM 的 uint32 ABI
原样传递该哨兵；跨 Server/SuperPod 的 D2D 路由不能据此判定可达。

只有同时出现以下证据才算正向用例通过：

```text
KVAF_A5_DDR_REGISTERED ... mapped_dva=...
KVAF_A5_DDR_IMPORTED ... npu_view=...
KVAF_A5_CHANNEL_READY role=host engine=CPU
KVAF_A5_CHANNEL_READY role=device engine=AIV
KVAF_A5_DIRECT_URMA_READY ... sq_depth=... cq_depth=... tp_id=... token_id=...
KVAF_A5_H2D_SMOKE_OK ... masked=1 canary=PASS
KVAF_A5_TEARDOWN_OK role=host
KVAF_A5_TEARDOWN_OK role=device
KVAF_A5_AIV_SIMT_URMA_H2D_ACCEPTANCE=PASS
```

结果保存在 `results/a5_aiv_urma_h2d_<run-tag>/`。

### 6.3 远端仓 C：同步确认

```bash
git ls-remote --heads https://gitcode.com/CacheOS/KVAF.git \
  A5_AIV_URMA_cs
```

服务器 B 验收后如需修复，仍按 A→C→B 循环：

```bash
# PC A
git add -A
git commit -m "fix(a5): address server validation finding"
git push origin A5_AIV_URMA_cs

# 服务器 B
cd /data/c00982601/KVAF
git fetch origin
git switch A5_AIV_URMA_cs
git pull --ff-only

# Docker vllm-ascend-cs-kvaf
cd /data/c00982601/KVAF
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/build_a5_aiv_urma.sh
bash scripts/run_a5_aiv_urma_h2d_smoke.sh
```

## 7. 当前没有包含的完整 KVAF 能力

本 MVP 有意不实现：

- Host Meta Manager、分布式 key→AccessTable 构建；
- Runtime `PrepareAccessTable/BatchGet` 接口接线；
- Prefix/Layer/Block 多维布局和 KV tensor 适配；
- BatchPut、Delete、异步多 outstanding request；
- 多 channel、多 QP、跨卡路由和容错；
- ACL Graph capture/replay；
- vLLM/CacheOS adaptor；
- 生产级 bootstrap。冒烟程序只用同机文件交换不透明 MR 描述；
- 性能调优和大 batch 分块。

另外，KVAF、SHMEM 和 UB 源码中没有可验证的 `SMIT` API 调用。
当前能证明的是 Host HCOMM `UBC_CTP` 建链 + AIV raw URMA READ；底层是否由
特定版本的 SMIT 驱动，必须由 CANN/HCOMM provider 文档或服务器 B 的驱动
trace 证明，不能只凭本仓源码断言。

## 8. 进入下一阶段前的硬门槛

1. 服务器 B 正向 H2D smoke 全部 marker 通过；
2. wrong/expired MR token 或越界地址负向用例能失败且不污染 canary；
3. 重复至少 1000 次建链、传输、销毁，无超时和资源泄漏；
4. 将正向 smoke 的多条 descriptor（含 mask）扩大到完整 32-lane 边界；
5. 确认 CQ 完成后 HBM 对下游 AIV kernel 的可见性；
6. 再接入稳定 AccessTable 的 ACL Graph capture/replay。
