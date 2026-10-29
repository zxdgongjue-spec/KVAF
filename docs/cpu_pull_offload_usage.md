# CPU Pull / Offload 构建与验证

本实现位于 `codex/cpu-pull-hbm-reference`，基于 `main@c4375e1`，参考
`aicpu-urma@8bc5a2a`。KVOF 协议为版本 2，固定使用参考的 32 字节 SGE 和四路 Jetty。
请同时重编 CPU 服务和 NPU Python 扩展，使用新构建目录，避免加载旧版本。

## CPU 服务

使用已经跑通 aicpu-urma 的同一套 liburma、provider 和头文件。在 CPU 环境中：

```bash
cmake -S . -B build-cpu-pull-reference \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DKVAF_BUILD_REMOTE_CPU_SERVICE=ON \
  -DKVAF_ENABLE_CPU_OFFLOAD=OFF \
  -DKVAF_ENABLE_AIV_BATCH_GET=OFF \
  -DKVAF_ENABLE_AIV_BATCH_PUT=OFF \
  -DBUILD_TESTING=OFF
cmake --build build-cpu-pull-reference -j8
./build-cpu-pull-reference/runtime/kvaf_cpu_service --listen 0.0.0.0 --port 28999
```

若 URMA 未安装在默认路径，配置时增加 `-DKVAF_URMA_ROOT=/实际安装目录`，或分别设置
`KVAF_URMA_INCLUDE_DIR` 和 `KVAF_URMA_LIBRARY`。保留该环境运行参考分支所需的动态库搜索路径。
确认旧服务已停止后再启动新服务。候选 ID 以新进程的 `CANDIDATE_DISCOVERED` 日志为准。

## NPU Python 扩展

先在当前终端加载已验证环境的 CANN 环境脚本，使用安装了 torch、torch_npu、pybind11
的 Python：

```bash
cmake -S . -B build-offload-reference \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DKVAF_ENABLE_CPU_OFFLOAD=ON \
  -DKVAF_BUILD_PYTHON=ON \
  -DKVAF_ENABLE_AIV_BATCH_GET=OFF \
  -DKVAF_ENABLE_AIV_BATCH_PUT=OFF \
  -DBUILD_TESTING=OFF \
  -DPython_EXECUTABLE="$(command -v python)"
cmake --build build-offload-reference -j8
export PYTHONPATH="$PWD/build-offload-reference/python${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import kvaf; print(kvaf.__file__); print(kvaf.OffloadRuntime)'
```

CANN 默认位置为 `/usr/local/Ascend/ascend-toolkit/latest`；也可通过
`ASCEND_CANN_PACKAGE_PATH` 或 `ASCEND_HOME_PATH` 环境变量、或 CMake
`-DASCEND_CANN_PACKAGE_PATH=/实际目录` 指定。此构建只需要 CANN host runtime，
不编译 AIV kernel。最后一条命令应指向新构建目录内的 kvaf。

## 硬件验收

2026-10-29，用户在测试机器上确认：使用以下配置，当前
`codex/cpu-pull-hbm-reference` 和此前的 `codex/cpu-pull-hbm` 分支均测试成功。
其他机器的 device/EID/candidate 仍需按实际拓扑选择。

```bash
python tests/python/test_cpu_offload.py \
  --cpu 127.0.0.1:28999 \
  --npu-eid 000000000000020000100000df00e100 \
  --device 7 --cpu-candidate 0
```

CPU 服务不在本机时，将 `127.0.0.1` 改为 CPU 服务地址。

此次 CPU candidate 0 对应 `udmac0d1e2`、eid_index=0，CPU EID 为
`00000000003f020000100000df080b00`。
之前固定使用的 NPU CLOS EID `00000000003f030000100000df00eb00` 能完成 Jetty
导入，但首次 READ 返回 `URMA_CR_LOC_ACCESS_ERR`。参考分支按发现顺序尝试 NPU EID：
第一组 session 释放后，第二个 `asdrv_pair_info_device_eid` 候选通过初始化。
Offload 固定指定该第二个 EID 后，两条分支均成功，因此本次恢复成功不依赖于新分支的
传输改写。EID 选择结论仅针对这台机器；CQ 状态本身不能解释底层硬件为何拒绝第一条路径。

检查输出：

1. 两端出现 `CPU_PULL_REFERENCE commit=8bc5a2a protocol=2 jettys=4` 相关字段，
   CPU 日志同时包含 `sge_bytes=32`，表明加载了本实现。
2. 第一个任务为 `phase=probe_service_thread`，READ HBM+64，最多 64 字节；
   成功后打印 `state=3 status=0 detail=completed`，Python 确认启动探测校验和。
3. 后续任务为 `phase=offload_worker`。脚本执行三轮数据，每轮包含 1、656、100001
   字节的三个 span，验证 CPU DDR 实际数据的 FNV-1a 校验和，随后 BatchFree 复用 slots。
4. 三轮都打印 `CPU DDR checksum passed` 并成功 Shutdown，才算该硬件配置验收通过。

`PEER_LINK_JETTY_IMPORT_READY` 只证明导入成功；不能替代 CQ 和 DDR 数据校验。

## 接口

```python
runtime = kvaf.OffloadRuntime(npu_eid, device_id=7, cpu_candidate=0, timeout_ms=30000)
runtime.link_to("127.0.0.1:28999")
runtime.initialize(ddr_bytes=8 * 131072, slot_size=131072)
runtime.offload([1, 3], source, [7, 100003], [656, 100001], stream=producer_stream)
checksum = runtime.last_checksum()
runtime.batch_free([1, 3])
runtime.shutdown()
```

`source` 必须是指定 NPU 上的连续 Tensor；offset 和 length 以字节计。
key 是池内 slot ID，单块不能超过 slot_size，单次最多 4096 块。stream 为数据生产者
使用的流，省略时取当前 NPU stream。函数等待生产者 event，再通知 CPU 读取；返回成功
表示数据已确认完成并 CommitPut。一个 Runtime 内串行调用。

初始化仅用 key 0 做探测，成功后会释放该 key。Checksum 是测试诊断接口，读取 CPU DDR；
本实现不提供 Decode/BatchGet。

## 失败恢复

- `pending_offload=False`：源已安全释放；检查错误后可处理失败或关闭实例。
- `pending_offload=True`：保留 runtime 和源 Tensor，不覆盖源数据，调用
  `runtime.resolve_offload()` 查询原任务。Python binding 也会保留源的强引用。
- TCP ACK 丢失但任务已完成：Resolve 确认 CPU 状态、释放源并完成元数据提交，不重发 READ。
- `CPU Pull transport quarantined`：查看 CPU 的 `CPU_PULL_TASK_END`。READ 提交、CQ
  或超时错误会保留资源，Resolve 不会重新提交或强制释放。保留双方日志后协调停止测试进程，
  重新运行时启动新 CPU 服务，避免旧隔离资源占用。
- 初始化探测未决：Resolve 只解决传输状态。确认不再 pending 后 Shutdown，创建新的
  OffloadRuntime 再初始化；不会把未经过校验和检查的旧实例直接标记为 ready。

CPU 失败日志记录 stage、ret、errno、wr_id、源/目的地址、长度、CQ status/user_ctx、
local_id、tpn 和两端 MR 属性。出现 `cq_status=4` 时仍需在实际 provider 环境定位，
不能仅凭权限值或 Jetty 导入成功判断问题已解决。

## 无硬件回归

```bash
cmake -S . -B build-offload-tests -DBUILD_TESTING=ON \
  -DKVAF_TEST_URMA_INCLUDE_DIR=/公共UMDK24.12头文件目录
cmake --build build-offload-tests -j8
ctest --test-dir build-offload-tests --output-on-failure
```

`KVAF_TEST_URMA_INCLUDE_DIR` 可省略，但会跳过 CPU READ provider 模型测试。
其余测试使用主机内存和 ACL/HCCP 替身，不需要 NPU。TCP 恢复测试需要允许监听本机端口。
这些结果不替代上述硬件验收。
