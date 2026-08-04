# KVAF Runtime

## 模块定位

KVAF Runtime 是对外的 KV 控制面入口。它不处理 Prefix Key 或
Forward 语义；它只拥有内部 DDR 的生命周期，编排 MetaManager 与
AccessOperator，并把调用方已经准备好的输入/输出 buffer 交给数据面。

## 已实现能力

- `Initialize`：经 `DdrMemoryAllocator` 分配内部 DDR，初始化 Operator，
  再向 MetaManager 注册远端 region；任一前置步骤失败会释放已获得资源。
- `BatchPut`：`PreparePut -> ExecuteBatchPut -> CommitPut`；数据面失败时
  自动 `AbortPut`，因此 Key 和 slot 可重新使用。配置 BatchPut kernel 的 NPU
  路径使用同步 AIV Direct-HCCP WRITE；未配置时保留同步
  `aclrtMemcpy(DeviceToHost)` mock。
- `PrepareAccessTable` / `GetAccessTableView` / `ReleaseAccessTable`：持有
  MetaManager 稳定读锁，生成一维 address/length/valid 表，并在 Release 时
  解锁。
- `BatchGet`：校验调用方的 index、mask、output 和（NPU 路径）Device
  AccessTable buffer 后，向调用方 stream 提交 `AccessOperator::ExecuteBatchGet`。
  Python 默认不在 KVAF 内同步；同一 stream 的后续操作自然按序执行。
- `BatchFree`：将删除请求交给 MetaManager；仍被 AccessTable 持有的 Key
  会返回 `kBusy`。

异步接口、BatchPut 图捕获管理及图对象生命周期不在本次 POC 范围。

## 资源所有权

Runtime 持有且负责释放：内部 DDR、AccessTable 状态和 Key 稳定性锁。

调用方持有且必须在其 BatchGet 所在 stream 完成前保持有效：BatchPut 的连续 `input` buffer、
与 Key 一一对应的 Host `input_offsets` / `lengths` 元数据、BatchGet 的
index/mask/output-index/output buffer，以及 NPU 上的 address/valid 表。Runtime
不会为这些调用方 buffer 分配、释放、隐式复制或注册链路。

`BatchPutRequest` 的输入语义为：`keys[i]` 的源数据位于
`input + input_offsets[i]`，长度为 `lengths[i]`。Runtime 校验每个范围位于
`input` 内，以 `lengths` 调用 `PreparePut`，再将相同请求和目标地址描述交给
Operator。因此同一 batch 可以包含不同长度的 KV block；input 可以包含对齐
padding，offset 无需固定步长。

配置 AIV BatchPut 时，`DefaultAccessOperator` 将 Host metadata 复用 staging
buffer 上传到 NPU，并将 `PreparePut` 返回的 destination address 上传为
`uint64[row_count]`；NPU input HBM 则按 `(地址, bytes)` 注册/缓存。BatchPut 使用
调用方 stream 并在该 stream 同步完成后才允许 Runtime Commit；core status 是
Operator 内部诊断状态，不是 Python 参数或返回值。

NPU BatchGet 的 `status` 和 `queue_state` 是 AIV kernel ABI 必需的内部 device
buffer，由 `DefaultAccessOperator.Initialize` 申请并在 `Shutdown` 释放；Python
和 `BatchGetRequest` 都不传它们。显式 `synchronize=True` 时，Operator 可读取
其 completion status 并以 `[KVAF_AIV] BATCH_GET_DEVICE_ERROR` 诊断日志记录
设备侧错误；它不会成为 Python 的额外返回值。

## 当前 POC 的 NPU AccessTable 边界

`PrepareAccessTable` 生成 Runtime-owned 的扁平 Host `address/length/valid` 表并
持有对应 Key 锁。C++ `GetAccessTableView` 返回指向三段连续 Host 内存的 non-owning
view，不是 Tensor，也不会复制数据。

Python `NpuRuntime.access_table_view(handle, device="npu", capacity=N)` 则创建
`DeviceAccessTable`：它持有 `addresses: torch.int64[N]` 和 `valid: torch.uint8[N]`
两个 NPU Tensor，并在图外完成一次 Host-to-NPU materialize。这个对象可直接传给
`NpuRuntime.batch_get`，调用者无需再传 `access_table` handle、裸指针、字节数或
手写 scatter。`capacity` 可大于有效 key 数；有效数据位于前缀，其余 `valid=0`。

`DeviceAccessTable` 及其两个 Tensor 必须存活到关联 BatchGet stream 完成，且其
source handle 必须在此后才可 `ReleaseAccessTable`。Runtime 不额外注册 caller 的
address/valid Tensor。

`valid` 是每个 AccessTable entry 的安全位：`1` 表示可访问，`0` 表示 padding
或无效 entry。AIV kernel 对 `mask=1 && valid=0` 返回设备侧 `kInvalidLookup`，
且不会对该地址发起 UDMA 读；padding 应由 Adapter 同时以 `mask=0` 跳过。
当前 InMemory MetaManager 对不存在的 Key 在 `PrepareAccessTable` 的
`LockKeys` 阶段直接返回 `kNotFound`，因此不会产生一张含该 Key 的
`valid=0` 表。

当前 AIV BatchGet ABI 每次 launch 只有一个 `block_bytes`，所以 NPU AccessTable
的所有 valid entry 必须具有相同长度。Runtime 在 `PrepareAccessTable` 时一次性
计算这一不变量，之后每次 BatchGet 只做 O(1) 比较；不会在数据面热路径重新扫描
整张表。

设备侧 `core_status.result`（例如 `kInvalidLookup`）是 Operator 内部诊断状态，
不是 Python BatchGet 的输入/输出。显式 `synchronize=True` 时会记录错误 marker；
默认 enqueue-only 调用由调用方在同一 stream 上的后续工作或自身同步决定何时观察
完成。参数校验、ACL 分配/提交和显式 stream 同步失败仍以 Runtime 错误返回。

## 测试

默认测试均由 Python 脚本定义场景和断言，并直接调用 pybind 的
`kvaf.Runtime`；没有 C++ Mock driver 或子进程协议：

```bash
cmake -S . -B build -DBUILD_TESTING=ON -DKVAF_BUILD_PYTHON=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

`tests/python/test_runtime_mock.py` 可以直接编辑；其中覆盖 Runtime 初始化、
Put/Get 生命周期、AccessTable 锁定、slot 回收、参数校验和 Operator 失败回滚。
脚本会以 `try/except` 探测 `torch_npu`、MPS、CPU 和无 PyTorch 环境：MPS 可用时
用于在本机生成 caller-side 测试数据，然后直接传给 CPU Mock。MPS 不是 CANN
AIV 或 Direct-HCCP 的替身，因此不能作为 NPU 数据面验证结论。

## A5 / CANN BatchGet 集成

`KVAF_ENABLE_AIV_BATCH_GET=ON` 配合 `KVAF_BUILD_PYTHON=ON` 会构建 merged
demo 的 kernel/host 库、`DefaultAccessOperator` 和 `kvaf.NpuRuntime`。
Python 脚本创建 NPU Tensor；通过 Runtime materialize `DeviceAccessTable`，然后
直接传入 `batch_get`：

```bash
cmake -S . -B build-aiv -DBUILD_TESTING=ON \
  -DKVAF_BUILD_PYTHON=ON \
  -DKVAF_ENABLE_AIV_BATCH_GET=ON \
  -DKVAF_ENABLE_AIV_BATCH_PUT=ON \
  -DASCEND_CANN_PACKAGE_PATH="$ASCEND_CANN_PACKAGE_PATH"
cmake --build build-aiv -j

KVAF_RUN_NPU_TESTS=1 \
KVAF_AIV_BATCH_GET_KERNEL="$PWD/build-aiv/bin/kvaf_execute_batch_get_kernels.o" \
DEVICE_ID=0 \
PYTHONPATH="$PWD/build-aiv/python" \
KVAF_NPU_ACTIVE_PERCENT=5 \
timeout 240s python3 tests/python/test_npu_batch_get.py
```

该脚本用小规模 case 验证 Runtime 初始化、调用方 Device Tensor、Runtime
materialize 的 `DeviceAccessTable`、BatchGet 输出字节、`output A -> B -> A` 的 local-buffer 缓存复用及
teardown。它以 `KVAF_NPU_ACTIVE_PERCENT` 构造稀疏 mask（例如 `5` 表示 5% 请求
有效），只为 `mask=1` 的请求写入 KV；随后可用 Runtime 的 ACL D2H mock
`batch_put` 将 caller-owned NPU source Tensor 写入 Runtime-owned DDR。该小型
BatchGet 用例不传 BatchPut kernel，因而不启动 Direct-HCCP BatchPut transport。

设置 `KVAF_NPU_GRAPH_REPLAYS` 可在同一个 Python 用例中额外运行 Python-owned 的
raw `kvaf.AclGraph` 图模式：

```bash
KVAF_NPU_GRAPH_REPLAYS=2 \
KVAF_NPU_WARMUP=10 \
KVAF_NPU_ITERATIONS=100 \
# 其余 KVAF_RUN_NPU_TESTS / kernel / shape 环境变量与上例相同
timeout 240s python3 tests/python/test_npu_batch_get.py
```

图对象、capture、replay 和销毁都由 Python 测试持有的 `kvaf.AclGraph` 管理；KVAF
Runtime 与 AccessOperator 不创建或保存图。`AclGraph` 是 pybind 暴露的薄封装，直接
调用 ACL Runtime Graph 的 `aclmdlRICaptureBegin`、`aclmdlRICaptureEnd`、
`aclmdlRIExecuteAsync` 和 `aclmdlRIDestroy`。脚本先在图外 eager 执行
`output A -> B -> A`，使 A 的 local HBM 注册进入 transport cache；再在 raw ACL
capture 中只 capture `runtime.batch_get(..., stream=capture_stream)`。

每次 replay 前，Python 在同一 ACL stream 上以
`AclGraph.copy_host_to_device_async` 更新固定地址的 `address_table`、`indices`、
`mask`、`output_indices`；capture 中的 BatchGet 参数和所有 NPU tensor 地址不变。正常
性能口径是 `graph_replay_to_stream_completion`：四项 H2D 更新会在 replay 前完成并同步，
不进入计时窗口；计时仅包含 graph replay 到同一 stream 完成，与 eager 的
`runtime_batch_get_call_to_stream_completion` 口径对齐。默认 graph 的 warmup/iterations
与 eager 一律使用相同的 `KVAF_NPU_WARMUP` / `KVAF_NPU_ITERATIONS` 采样计划。

AIV demo 使用原生
`aclrtMemcpyAsync(fixed_device_inputs, stream) -> aclmdlRIExecuteAsync(graph, stream)`。
Python 测试使用调用方持有的 `torch.npu.Stream`；`AclGraph.replay(stream)` 显式传入
capture 使用的同一 stream。图对象会在 Runtime shutdown 前 reset，避免图仍持有调用方
buffer 时释放 HCCP transport。

Python `batch_get` 的 `stream=None` 默认解析为 `torch.npu.current_stream()`，
`synchronize=False` 默认只提交工作。同一 stream 的后续 Torch/ACL 运算天然等待
BatchGet；如果调用方显式切换 stream，则由调用方建立 event/wait 依赖，并在
teardown 前完成每个使用过的 stream。只有排障或单独等待时才需要
`synchronize=True`。

正确性验证完成后，脚本会输出
`PERF_RESULT metric=runtime_batch_get_call_to_stream_completion`。这个计时窗口
从 `Runtime.BatchGet` 调用开始，到 AIV adapter 中的 stream 同步完成结束，排除
DDR 注册、Device buffer 初始化和 Host-to-Device 表更新。输出包括 mean、p50、
p90、p99 和 useful GiB/s。`ctest` 和上面的命令均设置 240 秒上限；未设置
`KVAF_RUN_NPU_TESTS=1` 时会明确跳过。

`DIRECT_LOCAL_BUFFER_ACQUIRE_NEW` 会始终记录新的 output HBM 注册；高频的
cache-hit / release 日志默认关闭，避免压测污染终端。如需排查 LocalBuffer 引用
计数，可额外设置 `KVAF_AIV_VERBOSE_LOCAL_BUFFER=1`。

## NPU Runtime 性能脚本

`tests/python/test_npu_runtime_perf.py` 是独立的 steady-state 性能脚本，不纳入
`ctest`。它分别测量：固定 128-key batch 的 `BatchPut`、一次全量
`PrepareAccessTable`、一次全量 `access_table_view(device="npu")`、随机 index/mask
的 `BatchGet`、全量 `ReleaseAccessTable`，以及 128-key batch 的 `BatchFree`。
BatchGet/Put 的每个样本均在 Runtime 调用后由调用方 `torch.npu.synchronize()`
结束；随机输入与数据填充不进入样本计时。

```bash
KVAF_RUN_NPU_TESTS=1 \
KVAF_AIV_BATCH_GET_KERNEL="$PWD/build-aiv/bin/kvaf_execute_batch_get_kernels.o" \
KVAF_BATCH_PUT_KERNEL="$PWD/build-aiv/bin/kvaf_execute_batch_put_kernels.o" \
DEVICE_ID=0 \
PYTHONPATH="$PWD/build-aiv/python" \
timeout 1800s python3 tests/python/test_npu_runtime_perf.py
```

默认 `batch=16`，即全表 2,097,152 key、每次 BatchGet 32,768 requests、`block_bytes=576`、
32 个 AIV core、3 次 warmup 与 1000 个样本。可用 `KVAF_RUNTIME_PERF_BATCH`、
`KVAF_RUNTIME_PERF_AIV_CORES`、`KVAF_RUNTIME_PERF_WARMUP`、
`KVAF_RUNTIME_PERF_ITERATIONS`、`KVAF_RUNTIME_PERF_ACTIVE_PERCENT` 和
`KVAF_RUNTIME_PERF_BLOCK_BYTES` 覆盖。全量 data population 会写入 2,097,152 个
逻辑 KV block；脚本显式配置 AIV Direct-HCCP BatchPut kernel，BatchPut 性能样本
测量 Runtime metadata upload、AIV WRITE 及其 caller-stream 完成，不包含全量数据
填充。
脚本以 CPU `torch.int64` Tensor 传入全量 Prepare key，避免 Python `list[int]` 转换污染
这项控制面性能数据。
