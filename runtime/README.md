# KVAF Runtime

## 模块定位

KVAF Runtime 是对外的同步 KV 控制面入口。它不处理 Prefix Key 或
Forward 语义；它只拥有内部 DDR 的生命周期，编排 MetaManager 与
AccessOperator，并把调用方已经准备好的输入/输出 buffer 交给数据面。

## 已实现的同步能力

- `Initialize`：经 `DdrMemoryAllocator` 分配内部 DDR，初始化 Operator，
  再向 MetaManager 注册远端 region；任一前置步骤失败会释放已获得资源。
- `BatchPut`：`PreparePut -> ExecuteBatchPut -> CommitPut`；数据面失败时
  自动 `AbortPut`，因此 Key 和 slot 可重新使用。当前 NPU 路径以同步
  `aclrtMemcpy(DeviceToHost)` 作为测试 mock，真实 BatchPut 数据面尚未合入。
- `PrepareAccessTable` / `GetAccessTableView` / `ReleaseAccessTable`：持有
  MetaManager 稳定读锁，生成一维 address/length/valid 表，并在 Release 时
  解锁。
- `BatchGet`：校验调用方的 index、mask、output 和（NPU 路径）Device
  AccessTable buffer 后，同步调用 `AccessOperator::ExecuteBatchGet`。
- `BatchFree`：将删除请求交给 MetaManager；仍被 AccessTable 持有的 Key
  会返回 `kBusy`。

异步接口、真实 BatchPut 的 NPU/AICPU 数据面及图捕获管理不在本次 POC 范围。

## 资源所有权

Runtime 持有且负责释放：内部 DDR、AccessTable 状态和 Key 稳定性锁。

调用方持有且必须在同步调用完成前保持有效：BatchPut block、BatchGet 的
index/mask/output-index/output buffer，以及 NPU 上的 address/valid 表。Runtime
不会为这些调用方 buffer 分配、释放、隐式复制或注册链路。

调用方也持有 NPU BatchGet 的 `status` 和 `queue_state` buffer。它们与
`output` 一样在每次 `BatchGetRequest` 中传给 Runtime，再由 Operator 在本次
执行中 Acquire 对应的 local HBM MR；没有固定的 output 绑定 API。

## 当前 POC 的 NPU AccessTable 边界

当前 `PrepareAccessTable` 只生成 Runtime-owned 的扁平 Host
`address/length/valid` 表并持有对应 Key 锁。C++ `GetAccessTableView` 返回的是
指向这三段连续 Host 内存的非 owning view，不是 Tensor，也不会复制数据。Python
绑定将它们导出为只读 `memoryview`，因此 Adapter 可用
`torch.frombuffer(...).to("npu")` 进行一次必要的 Host-to-NPU 复制，而不会先构造
数百万 Python 整数。

NPU AccessTable 的 device tensor 由 CacheOS-vLLM Adapter 创建和维护；本仓的
Python NPU 测试脚本暂时承担该 Adapter 角色。它只对已锁定的有效 key 读取 Host
view，并将地址和 valid 值 scatter 到 caller-owned 的稀疏 NPU `addresses` / `valid`
tensor；未使用的 entry 保持 `valid=0`。因此 Device AccessTable 的容量可以大于
Host view 的 entry 数，且 Runtime 不拥有这两个 device buffer。

Adapter 必须保证 Host `memoryview` 在完成其 H2D copy 前、device buffer 在关联
BatchGet 完成且 `ReleaseAccessTable` 前持续有效。Runtime 不分配、复制、释放或
注册这两块 device AccessTable buffer。

`valid` 是每个 AccessTable entry 的安全位：`1` 表示可访问，`0` 表示 padding
或无效 entry。AIV kernel 对 `mask=1 && valid=0` 返回设备侧 `kInvalidLookup`，
且不会对该地址发起 UDMA 读；padding 应由 Adapter 同时以 `mask=0` 跳过。
当前 InMemory MetaManager 对不存在的 Key 在 `PrepareAccessTable` 的
`LockKeys` 阶段直接返回 `kNotFound`，因此不会产生一张含该 Key 的
`valid=0` 表。

当前同步 Operator 尚未把 kernel 写入的 `core_status.result`（例如
`kInvalidLookup`）转换为 `Runtime.BatchGet` 的错误 Status；NPU 集成测试会
显式检查该状态。将设备 completion status 纳入 Runtime 返回语义，是后续把
Adapter 输入扩展为通用 AccessTableOutputSpec 时必须完成的工作。

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
Python 脚本创建 NPU Tensor，并在每次同步 `batch_get` 中直接传入
output/status/queue：

```bash
cmake -S . -B build-aiv -DBUILD_TESTING=ON \
  -DKVAF_BUILD_PYTHON=ON \
  -DKVAF_ENABLE_AIV_BATCH_GET=ON \
  -DASCEND_CANN_PACKAGE_PATH="$ASCEND_CANN_PACKAGE_PATH"
cmake --build build-aiv -j

KVAF_RUN_NPU_TESTS=1 \
KVAF_AIV_BATCH_GET_KERNEL="$PWD/build-aiv/bin/kvaf_execute_batch_get_kernels.o" \
DEVICE_ID=0 \
PYTHONPATH="$PWD/build-aiv/python" \
KVAF_NPU_ACTIVE_PERCENT=5 \
timeout 240s python3 tests/python/test_npu_batch_get.py
```

该脚本用小规模 case 验证 Runtime 初始化、调用方 Device buffer、Runtime
BatchGet、设备完成、输出字节、`output A -> B -> A` 的 local-buffer 缓存复用及
teardown。它以 `KVAF_NPU_ACTIVE_PERCENT` 构造稀疏 mask（例如 `5` 表示 5% 请求
有效），只为 `mask=1` 的请求写入 KV；随后用 Runtime 的临时 ACL D2H mock
`batch_put` 将 caller-owned NPU source Tensor 写入 Runtime-owned DDR，再将少量
有效 address scatter 到可很大的 Device AccessTable。该 mock 验证 Runtime 的
BatchPut 事务编排，但不代表未来的 BatchPut 数据面实现。

设置 `KVAF_NPU_GRAPH_REPLAYS` 可在同一个 Python 用例中额外运行外部
`torch.npu.NPUGraph` 图模式：

```bash
KVAF_NPU_GRAPH_REPLAYS=2 \
KVAF_NPU_GRAPH_WARMUP=5 \
KVAF_NPU_GRAPH_ITERATIONS=20 \
# 其余 KVAF_RUN_NPU_TESTS / kernel / shape 环境变量与上例相同
timeout 240s python3 tests/python/test_npu_batch_get.py
```

图对象、capture、replay 和销毁都由 `torch_npu` 持有；KVAF 不创建或保存图。脚本
先在图外 eager 执行 `output A -> B -> A`，使 A 的 local HBM 注册进入 transport
cache；再在 `torch.npu.graph(...)` 中只 capture
`runtime.batch_get(..., stream, synchronize=False)`。每次 replay 前，调用方在这条
同一 stream 上更新固定地址的 `indices` 与 `output_indices` tensor，再调用
`graph.replay()`。因此固定调用方 input tensor 的地址始终不变，而内容可改变。

AIV demo 使用原生
`aclrtMemcpyAsync(fixed_device_inputs, stream) -> aclmdlRIExecuteAsync(graph, stream)`。
Python 测试也使用同一个 `caller_stream`，但以 NPU tensor 的内容更新代替裸 ACL
copy；KVAF 不拥有这些 input tensor，也不创建、保存或 replay 图。

图性能指标 `external_graph_input_copy_to_stream_completion` 包含输入更新、整张外部图
replay 与 stream 同步；不包含 Python 构造 Host 输入、capture、初始化和最终验证，
因此不能与 eager 指标直接作一一对比。

`torch_npu.NPUGraph.replay()` 会在调用时的 **current stream** 上提交 replay，
不会自动切回 capture 时传入的 stream。因此测试会显式进入 `capture_stream` 后再调用
replay：输入更新、图执行和 completion wait 必须属于同一条 stream 顺序链。上游若选择
另一条 replay stream，必须自行在该 stream 上建立输入依赖并等待该 stream 完成。

Python 调用 `batch_get` 必须显式传入 `torch.npu.current_stream()`（或 capture
stream 对象）。
默认 `synchronize=True` 保持同步接口；只有在外部图 capture 的调用点传
`synchronize=False`，并由图的调用方在 teardown 前完成 capture stream。

正确性验证完成后，脚本会输出
`PERF_RESULT metric=runtime_batch_get_call_to_stream_completion`。这个计时窗口
从 `Runtime.BatchGet` 调用开始，到 AIV adapter 中的 stream 同步完成结束，排除
DDR 注册、Device buffer 初始化和 Host-to-Device 表更新。输出包括 mean、p50、
p90、p99 和 useful GiB/s。`ctest` 和上面的命令均设置 240 秒上限；未设置
`KVAF_RUN_NPU_TESTS=1` 时会明确跳过。

`DIRECT_LOCAL_BUFFER_ACQUIRE_NEW` 会始终记录新的 output HBM 注册；高频的
cache-hit / release 日志默认关闭，避免压测污染终端。如需排查 LocalBuffer 引用
计数，可额外设置 `KVAF_AIV_VERBOSE_LOCAL_BUFFER=1`。
