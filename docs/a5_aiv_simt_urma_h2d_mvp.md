# A5 AIV + SIMT + URMA + H2D 最小集合

## 1. 结论和状态

本分支实现的是完整 KVAF 的第一条数据面最小闭环：

```text
预先准备的 Device AccessTable
    -> AIV 内启动 32-lane SIMT 查表
    -> 生成固定 64B H2D 描述符
    -> 单 AIV 调用 HCOMM/URMA ReadNbi
    -> Host DDR 直接读入本地 HBM
    -> Commit + Drain
    -> Device Completion
```

这里：

- **AIV** 是 AI Vector Core；
- **SIMT** 是 AIV 内的多线程执行模式；
- **URMA** 通过 CANN `AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP>` 使用，QP、CQ、
  token 和门铃仍由 HCOMM 封装；
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
| AIV URMA READ 到 HBM | 已实现代码 | 服务器 B 需验证数据与 Completion |
| PC A 本地验证 | 已通过 | Apple clang CPU 单测和脚本语法检查 |
| A5 硬件端到端验证 | 尚未执行 | PC A 没有 CANN、bisheng 和 A5 |

因此，合入前的准确说法是“**代码闭环已实现，本地可验证部分已通过，A5
硬件验收待服务器 B 执行**”，不能把它描述成已经在 A5 上跑通。

本最小集采用设计文档 §3.4 的具体图边界：Replay 从 capture 前已稳定放在
NPU 上的 AccessTable 开始，`PrepareAccessTable`、Key 展开和 Meta 查询在
图外。本分支尚未验证 capture/replay，因此也不能笼统声称已满足文档中的
全部图执行约束。

## 2. 为什么采用 “SIMT 组描述符、AIV 提交”

SHMEM 的 SIMT 例子可以从 AIV kernel 内调用 `asc_vf_call`，但当前公开的
HCOMM URMA API 是 `__aicore__` 接口，`hcomm.h` 还明确拒绝 pure-SIMT
编译。让 SIMT lane 直接写同一个 SQ、更新门铃并轮询 CQ，也会形成多个
producer 竞争。

本实现使用单 producer 规则：

1. 一个 AIV block，固定 32 个 SIMT lane；
2. 每个 lane 只读取 AccessTable、Index、Mask 和输出偏移；
3. 每个 lane 写自己的 64B 描述符，不操作 SQ/CQ；
4. `asc_vf_call` 返回后，外层 AIV 检查整个 batch；
5. 只有整个 batch 规划合法，AIV 才逐条调用 `ReadNbi<false>`；
6. AIV 对同一 channel 只执行一次 `Commit` 和一次 `Drain`。

这保留了 SIMT 并行查表能力，同时让 HCOMM channel 保持单生产者。

核心文件：

- `access_operator/include/kvaf/access_operator/aiv_simt_urma_h2d.h`
- `access_operator/src/aiv_simt_urma_h2d.cpp`
- `access_operator/a5/src/aiv_simt_urma_h2d_kernel.cpp`

参考实现与本分支的对应关系如下（路径相对于 PC A 的 `agent/`）：

- `shmem/examples/simt_rma/main.cpp`：AIV 外层通过 `asc_vf_call` 启动
  SIMT；
- `shmem/CMakeLists.txt`：新旧 bisheng 的 `xasc/xcce`、A5 架构参数与
  fat-object 链接；
- `asc-devkit/tests/api/adv_api/hcomm/test_hcomm_urma_950.cpp`：HCOMM URMA
  workspace 清零、`ReadNbi/Commit/Drain` 语义；
- `UB_MicroBenchmark/src/operator/AIV_URMA/new_single_op/aiv_urma.h`：作为
  raw WQE/SIMT 行为的交叉检查。本分支没有复制它的私有 SQ/CQ 写法，而是
  保持在公开 HCOMM API 上。

A5 Host DDR 分配/注册、HCOMM endpoint/channel/MR 生命周期和 driver EID
发现均在 KVAF 内独立封装；构建和运行不要求下载其他业务代码仓。

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
5. 原子发布 server channel 已创建标记，允许 Host 再创建 `CLIENT` channel；
6. Host 和 Device 都使用 `exchangeAllMems=true`，且所有本地 MR 都在建链前
   注册；
7. 轮询 `HcommChannelGetStatus`，只有 READY 才继续；
8. 导入 Host MR 描述，取得 `HcommMemImport` 返回的 `view.addr`；
9. 拒绝空地址、错误 memory type 或不足的长度，再启动 kernel。

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
5. AIV 将 HCOMM 的 512B UB workspace 完整清零，再初始化
   `AscendC::Hcomm<UBC_CTP>`；清零用于保证 SGE token 等保留字段不带
   未初始化值；
6. 对每个 active descriptor：

   ```text
   ReadNbi<false>(channel, local_hbm, imported_host_ddr, bytes)
   ```

7. 一次 `Commit(channel)` 发布本 batch；
8. `Drain(channel)` 等待 CQ 完成并检查 HCOMM 返回码；
9. 标记每条 descriptor complete，再写 batch Completion；
10. smoke Host 控制程序同步 stream，回读结果、descriptor 和 Completion；
11. 正向 smoke 默认把 payload 分成最多 4 个分段，反向排列 AccessTable
    index，并额外加入一个带非法 output offset 的 masked lane；
12. 校验每个 active descriptor、masked descriptor、完整 payload 以及输出
    尾部 canary。

“整批失败”是 API 可见语义，不代表物理写入原子性。如果部分 `ReadNbi`
已经入队后，后续请求、Commit 或 Drain 失败，HBM 中可能已有部分数据；
调用方必须在 Completion 失败时丢弃整个输出。

固定 32 条还隐含一个服务器 B 验证项：当前 HCOMM channel 的有效 SQ 深度
必须容纳本 batch。此假设不能只靠源码证明。

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

```bash
git clone https://gitcode.com/CacheOS/KVAF.git
cd KVAF
git fetch origin
git switch A5_AIV_URMA_cs
git pull --ff-only

source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/build_a5_aiv_urma.sh
```

如果当前 CANN 没有安装 `adv_api/hcomm/hcomm.h`，准备匹配版本的
`asc-devkit`，然后：

```bash
export KVAF_ASC_DEVKIT_PATH=/path/to/asc-devkit
bash scripts/build_a5_aiv_urma.sh
```

运行脚本会先查询 logical/physical device 映射和 HCOMM endpoint 拓扑。
如果没有显式设置 EID，它会调用 KVAF 自带的最小 resolver，通过
`/run/lock/kvaf_asdrv_ub_pair.lock` 协作锁串行化对
`/proc/asdrv_ub/dev_id` selector 的写入和 `pair_info` 快照读取，再把
`local_eid` 作为 Host EID、`remote_eid` 作为 Device EID：

```bash
unset KVAF_HOST_EID KVAF_DEVICE_EID KVAF_PAIR_CHANNEL
export KVAF_LOGICAL_DEVICE_ID=0
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
git fetch origin
git switch A5_AIV_URMA_cs
git pull --ff-only
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

另外，KVAF 和 SHMEM 源码中没有可验证的 `SMIT` API 调用。
当前能证明的是 HCOMM `UBC_CTP` + AIV engine + URMA Read API；底层是否由
特定版本的 SMIT 驱动，必须由 CANN/HCOMM provider 文档或服务器 B 的驱动
trace 证明，不能只凭本仓源码断言。

## 8. 进入下一阶段前的硬门槛

1. 服务器 B 正向 H2D smoke 全部 marker 通过；
2. wrong/expired MR token 或越界地址负向用例能失败且不污染 canary；
3. 重复至少 1000 次建链、传输、销毁，无超时和资源泄漏；
4. 将正向 smoke 的多条 descriptor（含 mask）扩大到完整 32-lane 边界；
5. 确认 `Drain` 后 HBM 对下游 AIV kernel 的可见性；
6. 再接入稳定 AccessTable 的 ACL Graph capture/replay。
