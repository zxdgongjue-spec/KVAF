# CPU Pull / Offload 构建与运行

实现提供独立的 `kvaf.OffloadRuntime`。调用端运行于持有 HBM 的 NPU 进程，CPU 服务主动 READ 到自己的 DDR。正常路径是 `PreparePut → CPU Pull → CommitPut`，不要求 AIV BatchGet/BatchPut kernel。

## 构建

三端卸载与 Decode 读取不在本版范围。以下两个进程使用同一份源码构建。

CPU 机器需要 Linux、UMDK 24.12 配套头文件以及部署时的 liburma/provider。CPU 建链继续使用 KVAF 已有部署要求的 `urma_get_tp_list` 和 `urma_import_jetty_ex` 扩展。

```bash
cmake -S . -B build-cpu-offload \
  -DKVAF_BUILD_REMOTE_CPU_SERVICE=ON \
  -DKVAF_URMA_ROOT=/path/to/umdk \
  -DKVAF_URMA_LIBRARY=/path/to/liburma.so
cmake --build build-cpu-offload -j8
```

默认 CPU READ 使用公开的 24 字节 SGE；如果已确认部署 provider 使用参考 CPU push 中的 32 字节扩展 SGE，则额外设置 `-DKVAF_CPU_PULL_EXTENDED_SGE=ON`。不能只根据头文件版本判断实际加载库的 ABI。

沿用 `aicpu-urma@8bc5a2a` 的 provider 环境时，显式选择其 32 字节 SGE。已有构建目录需重新配置，仅执行 build 不会改变缓存中的 OFF：

```bash
cmake -S . -B build-cpu-offload -DKVAF_CPU_PULL_EXTENDED_SGE=ON
cmake --build build-cpu-offload -j8
```

重新启动 CPU 服务后，下一次分配应打印 `CPU_PULL_DDR ... sge_bytes=32`。这确认当前二进制使用了所选布局；是否匹配实际加载的 provider 仍需以部署版本和硬件验收确认。

NPU 机器需要 CANN、Python、pybind11，以及运行时的 torch/torch_npu：

```bash
cmake -S . -B build-offload \
  -DKVAF_ENABLE_CPU_OFFLOAD=ON \
  -DKVAF_ENABLE_AIV_BATCH_GET=OFF \
  -DKVAF_ENABLE_AIV_BATCH_PUT=OFF \
  -DKVAF_BUILD_PYTHON=ON \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/ascend-toolkit/latest
cmake --build build-offload -j8
export PYTHONPATH="$PWD/build-offload/python${PYTHONPATH:+:$PYTHONPATH}"
```

原有 `NpuRuntime` 和 BatchPut 入口保持独立。Offload 的编译开关不触发 AIV kernel 编译。

## 使用

在 CPU 机器按现有服务方式启动 `build-cpu-offload/runtime/kvaf_cpu_service`，通过其 `--help` 查看绑定地址、端口、UB device 与 EID 选择参数。控制端口供受信任调用方使用；新协议与原 KVCP v6 共用监听入口，但采用独立 magic/version。

调用端显式提供 NPU EID 与 CPU candidate，不执行 BatchGet 自适应选路。初始化会分配目标池，并通过一次小规模 CPU Pull 和 CPU DDR 校验验证所选路径。

`npu_eid` 用于 CTP 路径查询；HCCP 返回的 HBM segment 和 Jetty key 分别携带 provider 的内存与队列标识。按 `aicpu-urma@8bc5a2a` 的 CPU READ 实现，这些导出数据原样用于导入，不要求内嵌 EID 与配置的路径 EID 相等，不应把 segment EID 直接替换成 `npu_eid`。

```python
import torch
import torch_npu
import kvaf

torch.npu.set_device(0)
runtime = kvaf.OffloadRuntime(
    npu_eid="替换为本机 NPU 的 32 位十六进制 EID",
    device_id=0,
    cpu_candidate=0,
    timeout_ms=30000,
)
runtime.link_to("CPU_HOST:PORT")
runtime.initialize(ddr_bytes=16 * 1024 * 1024, slot_size=65536)

source = torch.arange(4096, device="npu:0", dtype=torch.int32)
runtime.offload(
    keys=[1, 3], source=source,
    input_offsets=[0, 8192], lengths=[8192, 4096],
)
checksum = runtime.last_checksum()  # 可选诊断：按请求顺序对 CPU DDR 计算 FNV-1a 64
runtime.batch_free([1, 3])
runtime.shutdown()
```

每个 runtime 只允许一个调用线程和一个在途 Offload。源必须是当前所选 NPU 上的连续 Tensor；元数据使用整数序列或一维连续 CPU int64 Tensor，最多 4096 项。key 沿用当前固定 slot ID 语义，单项长度不超过 slot_size。

`stream` 缺省为当前 NPU stream。Offload 在该 stream 上记录并等待 producer event，再通知 CPU；调用方负责其他 stream 上的依赖，且在调用期间不能覆盖源数据。首版不支持图捕获。

## 失败与恢复

`offload()` 成功才表示 CPU 数据和元数据均已提交。发生异常后检查 `runtime.pending_offload`：

- 为 false：没有仍被该任务保留的源内存，安全失败的预留已回滚。
- 为 true：不要修改或复用 source。使用 `runtime.task_id` 识别任务，再调用 `runtime.resolve_offload()`；它查询原任务，不重新提交 DMA。

Python binding 保持未决任务的源 Tensor 强引用。C++ 调用方必须自行保持源 allocation 和内容有效，直到 `pending_offload()` 为 false。MR 注册不能代替内存所有权。

CQ 错误或超时会隔离该 CPU transport，首版不尝试强制取消/重建 DMA。状态仍未知时 Shutdown 返回错误，目标 slot 和源注册继续保留；异常析构也保留相关资源到进程退出。网络恢复可解决丢失响应，但不能自动修复硬件队列错误。

初始化探测若失败且留下 pending 事务，同样先 resolve，再 shutdown，之后重新 initialize；探测失败时不会将 runtime 标记为就绪。

### 初始化错误定位

`create HBM context` 表示 NPU 侧 `RaCtxInit` 失败，还没有注册源 MR 或发起 CPU READ。异常中包含底层 `ret`、物理设备号、EID 和 `eid_index`。先核对 `--device` 对应的物理设备和 `--npu-eid`；CPU candidate 是后续建链使用的路径，不能修复本地 NPU context 的创建失败。

`invalid exported HBM range` 表示 context 和源 HBM 注册已成功，但导出的 segment 没有通过校验。异常包含：

- `zero_base`：导出地址为零。
- `short_segment`：导出长度小于请求长度。
- `address_overflow`：导出地址加请求长度会溢出。

同时保留 `requested_base/exported_base`、`requested_bytes/exported_bytes`、`npu_eid/segment_eid` 与 `key_bytes`，用于区分范围和 ABI 问题。旧版本还会因 `eid_mismatch=1` 拒绝有效的导出；这项相等检查已在 NPU 导出和 CPU 导入两端移除，需要两端都更新、重新编译并重启 CPU 服务。地址和长度检查仍然保留。验收脚本在初始化失败时也会执行安全清理；未决任务继续保留资源。

### 与 aicpu-urma 的对照

参考版本：`aicpu-urma@8bc5a2a`。

- `access_operator/demo/aicpu_batch_get/host/cpu_push_host.cc`：通过 HCCP 注册 HBM，单独传递配置 EID 和原始 segment，再查询 CTP、创建 NPU target Jetty。
- `runtime/src/cpu_push_get_transport.cc`：检查 HBM 范围，原样导入 segment；使用配置 EID 查询 CPU 侧 CTP，并以原始 key 导入 NPU Jetty。`Transfer(false, ...)` 通过 CPU READ 拉取 HBM 中的通知和描述符。
- `common/include/urma_transport.h`：封装 CTP/Jetty 导入参数，并采用 32 字节扩展 SGE。本分支通过 `KVAF_CPU_PULL_EXTENDED_SGE` 显式选择该 ABI，仍须匹配实际部署的 provider。

Offload 沿用上述 EID 与导出标识的处理方式，将 READ 来源扩展为调用方的 Tensor HBM；源 MR 只开放远端 READ，任务完成后提交元数据。

token 也沿用参考实现的 region 约定：CPU 在 Allocate 时生成非零 `token_value`，用于 DDR 注册并在响应中返回；NPU 用它注册 HBM、创建 Jetty，CPU 用同一个值导入 HBM 和 Jetty、创建本地接收队列。两端的 `token_id` 仍分别由各自 provider 分配，不能将本地 token ID 当作共享的认证值。Open 会检查 token 一致性。

早期 Offload 的 CPU DDR 使用本地 token ID 作为 token value，而 NPU 每次导出另行生成 token。更新此约定后需要两端重新编译并重启；旧服务缺少 Allocate token 会在注册 HBM 前被拒绝。此对齐消除了一项与可用基线的差异，仍需硬件复测确认是否解决 CQ 错误。

### READ 未完成时

Jetty 导入成功只代表建链资源已创建，初始化还要用一次 READ 验证 HBM 到 DDR 的数据路径。CPU 服务按 `region`、`task` 打印 `CPU_PULL_TASK_BEGIN` 和 `CPU_PULL_TASK_END`；错误详情包含以下阶段：

- `stage=post_read`：提交失败，保留 URMA 返回值、`errno` 和 bad WR 信息。
- `stage=poll_cq`：查询 CQ 失败。
- `stage=cq_completion`：CQ 状态非成功或返回的 `user_ctx` 不匹配；打印 `cq_status`、`cq_status_name`、`cq_user_ctx`、预期 `wr_id`、`local_id`、`tpn`。
- `stage=cq_timeout`：提交后 15 秒仍未收到有效完成。

每种 READ 错误都保留 `sge_bytes`、源/目标地址和长度。NPU 异常会区分 CPU transport 已隔离和任务仍在运行但调用方等待超时，并给出相同的 `region/task` 以关联 CPU 日志。`resolve_offload()` 只查询原任务，不重发 READ，也不重建已隔离的硬件队列。

公开 UMDK 24.12 中，`cq_status=4` 是 `URMA_CR_LOC_ACCESS_ERR`，表示本地内存访问错误；`ret=1` 表示取到一条 CQ 记录，不表示 READ 成功。优先核对 CPU 接收 MR、SGE ABI 和保护信息，不应仅凭该状态改换 NPU EID。`CPU_PULL_DDR` 记录驱动实际返回的 `registered_base/bytes/attr/access/token_id`、分配的 token ID 和范围/权限检查结果；检查不通过时在提交 DMA 前失败。READ 错误同时附带 `dst_mr_*` 与 `src_mr_*`，用于检查地址是否落在实际注册范围内。日志不输出共享认证 token 的值。

若使用参考分支的 provider，先确认 `sge_bytes=32`，并保留本轮设备、NPU EID、CPU candidate 参数复测，以便区分 ABI 变化和路径变化。当前独立验收进程已经退出且 CPU transport 被隔离时，应停止旧 CPU 服务，再启动新二进制和新的 Python 验收进程。仅凭 `pending/quarantined` 无法判断 ABI 或链路是哪一个出了问题，需要完整的 CPU 任务错误行。

## 验证与限制

主机测试包含事务提交/回滚、pending 数据不可见、目标范围检查、任务去重，以及真实 TCP 上丢失 Submit/Open/ReleaseSource/Close 响应后的恢复。测试使用 fake ACL/HCCP 和 Host 内存后端，不代表硬件通过。

```bash
cmake -S . -B build-host -DBUILD_TESTING=ON
cmake --build build-host -j8
ctest --test-dir build-host --output-on-failure
```

两机硬件验收运行：

```bash
python tests/python/test_cpu_offload.py \
  --cpu CPU_HOST:PORT --npu-eid NPU_EID_HEX --device 0 --cpu-candidate 0
```

脚本覆盖 producer stream、不同长度、非零源 offset、多次复用及跨 64 KiB WR 的 READ，直接校验 CPU DDR，不依赖 Decode/BatchGet。

目标 DDR 使用 pinned `LOCAL_WRITE` MR；源 HBM 按远端 READ 用途注册。当前后端要求部署平台支持 CQ 完成后的 coherent host-DMA 可见性。注册权限、pin 模式、SGE/设备属性 ABI、驱动配套和 CPU 可见性须在目标硬件验证。本版未验证同一池作为远端 BatchGet 源的行为。
