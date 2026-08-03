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

NPU `AivBatchGetAccessOperator::BindBatchGetOutput` 必须由调用方或测试在
Graph Capture 前完成；它接收 caller-owned output/status/queue buffer。

## 测试

默认测试均由 Python 脚本定义场景和断言，C++ 只提供无场景的 Mock driver：

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

`tests/python/test_runtime_mock.py` 可以直接编辑；其中覆盖 Runtime 初始化、
Put/Get 生命周期、AccessTable 锁定、slot 回收、参数校验和 Operator 失败回滚。
脚本会以 `try/except` 探测 `torch_npu`、MPS、CPU 和无 PyTorch 环境：MPS 可用时
用于在本机生成 caller-side 测试数据，随后序列化给 CPU Mock。MPS 不是 CANN AIV
或 Direct-HCCP 的替身，因此不能作为 NPU 数据面验证结论。

## A5 / CANN BatchGet 集成

`KVAF_ENABLE_AIV_BATCH_GET=ON` 会构建 merged demo 的 kernel/host 库、
`AivBatchGetAccessOperator`，以及 Runtime→MetaManager→AIV 的 smoke runner：

```bash
cmake -S . -B build-aiv -DBUILD_TESTING=ON \
  -DKVAF_ENABLE_AIV_BATCH_GET=ON \
  -DASCEND_CANN_PACKAGE_PATH="$ASCEND_CANN_PACKAGE_PATH"
cmake --build build-aiv -j

KVAF_RUN_NPU_TESTS=1 \
KVAF_AIV_RUNTIME_BATCH_GET_RUNNER="$PWD/build-aiv/bin/kvaf_runtime_aiv_batch_get_runner" \
KVAF_AIV_BATCH_GET_KERNEL="$PWD/build-aiv/bin/kvaf_execute_batch_get_kernels.o" \
DEVICE_ID=0 \
python3 tests/python/test_aiv_batch_get_npu.py
```

该脚本用小规模 eager case 验证 Runtime 初始化、调用方 Device buffer/链路
注册、Runtime BatchGet、设备完成、输出字节和 teardown。它有 240 秒超时并
输出分阶段标记，未设置 `KVAF_RUN_NPU_TESTS=1` 时会明确跳过。
