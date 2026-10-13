# Q1 使用进程内 host DDR

Q1 非 MTP 测试需要让 NPU 直接读取本机 DDR，同时保持三方 LI tile 内阈值比较与
Post 的实现。本分支为原有 KVAF 增加 `local_host_ddr=True`：DDR 由运行测试的
进程申请、持有并注册，初始化无需连接 CPU 服务。主机仍负责分配、注册和准备
输入；计时路径中的读取由 AIV 内核通过已注册的传输队列发起。

## 分支来源

自有仓库为 `https://gitcode.com/CacheOS/KVAF.git`，工作分支为
`kip-cs-nomtp-tile-matrix`。本分支从以下提交精确派生：

```text
72c3955d7a0d825d641d32958cee969a8f196548
feat: enable THP by default and hide diagnostic page mode option
```

这与原服务器目录 `KVAF-v9-thp-72c3955d7a0d` 的来源一致。本次修改保留这一
分支的 Runtime、Python 参数及 V9 传输结构，不切换到 MTP 分支的 Runtime API。

本地 DDR 分配行为参考已验证的 MTP 提交
`720ed3ecdc7df613afd1a07e85612656c09acc6a`，具体来源是该提交的
`bindings/npu_python_runtime.cc` 中 `RuntimeHostDdrAllocator` 和页状态统计逻辑。
本分支把它们移到 `runtime/include/local_host_ddr.h` 与
`runtime/src/local_host_ddr.cc`，通过显式参数选择页模式，并用 C++ `Status`
返回错误。该提交是实现参考，不是本分支的 Git 基线，也不要求加载其扩展。

另选择性保留 `b3e14eb42c80d3a3dc3d4a5d4fe5dc0ad9ba8d00` 的 BatchPut
有界 CQ 等待和逐队列诊断。CQ 是设备读取传输完成状态的队列；这个修复在
原有快路径超时后提供有界的额外等待，并记录失败队列，不改变 LI 的融合策略。该提交之前
的整段 PeerLink 调试历史没有因此成为本分支的基线。

## 调用与页模式

原有远端服务模式仍默认保留。Q1 调用方必须显式启用本地 DDR，并使用本分支
同一次构建产生的 Python 扩展和 Get/Put 内核对象：

```python
runtime = kvaf.NpuRuntime(
    batch_get_kernel_object_path=get_kernel,
    batch_get_token_bytes=656,
    batch_get_tokens_per_block=128,
    device_id=0,
    npu_core_count=workers,
    batch_put_kernel_object_path=put_kernel,
    jettys_per_core=2,
    local_host_ddr=True,
    local_ddr_page_mode="thp",
)
runtime.initialize(ddr_bytes, slot_size)
assert runtime.local_host_ddr
assert runtime.remote_region["remote_service"] is False
```

`get_kernel`、`put_kernel`、`workers` 和 DDR 大小由上层 runner 提供。这里的
`656` 与 `128` 是当前 Q1 fixture 的 token 字节数和每个 block 的 token 数。
本地模式不调用 `link_to`，如果调用则报错；也不启动 `kvaf_cpu_service`。
`remote_region` 是既有接口名称，在本地模式下仍表示可供设备访问的区域，
并通过 `owner_pid` 标明当前进程所有权。

| `local_ddr_page_mode` | 分配与验证 |
| --- | --- |
| `thp`（默认） | Linux `mmap`，按 2 MiB 对齐，`MADV_HUGEPAGE` 后触碰完整范围；要求 `/proc/self/smaps` 证明存在透明大页。 |
| `base` | 相同范围与对齐方式，使用 `MADV_NOHUGEPAGE`，要求实际大页字节数为零。 |
| `acl` | 使用 `aclrtMallocHost`，不强制大页。 |

`base` 与 `thp` 在非 Linux 主机上明确拒绝执行；`thp` 无实际大页时也报错，
不会悄悄降级。统计会累加所有覆盖 DDR 的虚拟内存区间，因此注册过程切分
区间后仍能检查完整覆盖。Python `remote_region` 暴露 `ddr_page_mode`、
`ddr_anon_huge_bytes` 与 `ddr_smaps_covered_bytes`，初始化后再次验证页状态。

该开关只选择 DDR 来源，不启用 MTP、不增加 query，也不引入窗口轮转。
“与 MTP 一致”指 DDR 属于当前进程、直接注册并由设备访问这一内存路径。

## 构建与无 NPU 测试

以下目标只编译真实分配器与私有 ACL stub；它不链接 PeerLink、CPU 服务或
设备传输实现，因此无需 CANN 或 NPU：

```bash
cmake -S . -B build-local-ddr-host \
  -DBUILD_TESTING=ON -DKVAF_BUILD_PYTHON=OFF \
  -DKVAF_ENABLE_AIV_BATCH_GET=OFF -DKVAF_ENABLE_AIV_BATCH_PUT=OFF \
  -DKVAF_BUILD_REMOTE_CPU_SERVICE=OFF
cmake --build build-local-ddr-host --target kvaf_local_host_ddr_allocator_test -j2
ctest --test-dir build-local-ddr-host -R '^kvaf_local_host_ddr_allocator_' -V
```

测试覆盖错误输入、ACL 申请失败后的重试、重复申请、错误地址释放、完整释放、
页统计清空及 Linux 真实 `mmap`。Linux base 测试主动切分虚拟内存区间，验证
统计仍覆盖完整分配。THP 独立测试只有在非 Linux 或内核明确配置 `[never]`
且分配器拒绝降级时返回 skip；其他分配或统计错误均使测试失败。

NPU 构建沿用该分支的 AIV 配置，显式关闭独立 CPU 服务：

```bash
cmake -S . -B build-local-ddr-npu \
  -DBUILD_TESTING=OFF -DKVAF_BUILD_PYTHON=ON \
  -DKVAF_ENABLE_AIV_BATCH_GET=ON -DKVAF_ENABLE_AIV_BATCH_PUT=ON \
  -DKVAF_BUILD_REMOTE_CPU_SERVICE=OFF \
  -DPython_EXECUTABLE="$PYTHON" -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH"
cmake --build build-local-ddr-npu -j16
```

`PYTHON` 应指向安装了 `pybind11`、`torch` 与 `torch_npu` 的解释器。
host 测试只能证明分配器行为与资源释放，不能代替真实 HCCP 注册、AIV
完成队列检查或 Q1 输出校验。真机结果还必须记录构建来源、
`LOCAL_HOST_DDR_READY ... remote_service=0`、实际页状态和输出正确性。
