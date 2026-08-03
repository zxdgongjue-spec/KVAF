# KVAF Runtime

## 模块定位

KVAF Runtime 是对外的同步 KV 控制面入口。它不处理 Prefix Key 或
Forward 语义；它只拥有内部 DDR 的生命周期，编排 MetaManager 与
AccessOperator，并把调用方已经准备好的输入/输出 buffer 交给数据面。

## 已实现的同步能力

- `Initialize`：经 `DdrMemoryAllocator` 分配内部 DDR，初始化 Operator，
  再向 MetaManager 注册远端 region；任一前置步骤失败会释放已获得资源。
- `BatchPut`：`PreparePut -> ExecuteBatchPut -> CommitPut`；数据面失败时
  自动 `AbortPut`，因此 Key 和 slot 可重新使用。
- `PrepareAccessTable` / `GetAccessTableView` / `ReleaseAccessTable`：持有
  MetaManager 稳定读锁，生成一维 address/length/valid 表，并在 Release 时
  解锁。
- `BatchGet`：校验调用方的 index、mask、output 和（NPU 路径）Device
  AccessTable buffer 后，同步调用 `AccessOperator::ExecuteBatchGet`。
- `BatchFree`：将删除请求交给 MetaManager；仍被 AccessTable 持有的 Key
  会返回 `kBusy`。

异步接口、BatchPut 的 NPU/AICPU 数据面及图捕获管理不在本次 POC 范围。

## 资源所有权

Runtime 持有且负责释放：内部 DDR、AccessTable 状态和 Key 稳定性锁。

调用方持有且必须在同步调用完成前保持有效：BatchPut block、BatchGet 的
index/mask/output-index/output buffer，以及 NPU 上的 address/valid 表。Runtime
不会为这些调用方 buffer 分配、释放、隐式复制或注册链路。

调用方也持有 NPU BatchGet 的 `status` 和 `queue_state` buffer。它们与
`output` 一样在每次 `BatchGetRequest` 中传给 Runtime，再由 Operator 在本次
执行中 Acquire 对应的 local HBM MR；没有固定的 output 绑定 API。

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
timeout 240s python3 tests/python/test_npu_batch_get.py
```

该脚本用小规模 case 验证 Runtime 初始化、调用方 Device buffer、Runtime
BatchGet、设备完成、输出字节、`output A -> B -> A` 的 local-buffer 缓存复用及
teardown。当前 demo 尚未实现 `BatchPut`，因此测试的 `seed_fixed_blocks` 只用于
把固定大小的测试数据写入 Runtime-owned DDR；它不宣称验证 BatchPut。BatchPut
operator 合入后应改为 `runtime.batch_put`。

正确性验证完成后，脚本会输出
`PERF_RESULT metric=runtime_batch_get_call_to_stream_completion`。这个计时窗口
从 `Runtime.BatchGet` 调用开始，到 AIV adapter 中的 stream 同步完成结束，排除
DDR 注册、Device buffer 初始化和 Host-to-Device 表更新。输出包括 mean、p50、
p90、p99 和 useful GiB/s。`ctest` 和上面的命令均设置 240 秒上限；未设置
`KVAF_RUN_NPU_TESTS=1` 时会明确跳过。
