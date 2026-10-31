# CPU Pull 构建与验证

当前使用统一 `NpuRuntime` 和共享 DDR。CPU 数据面保留参考分支的四 Jetty 批量 READ，公开接口使用 KVRP v4；CPU 服务与 NPU Python 扩展必须同时重编译、重启。

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
python -c 'import kvaf; print(kvaf.__file__); print(kvaf.NpuRuntime.batch_cpu_pull)'
```

CANN 默认位置为 `/usr/local/Ascend/ascend-toolkit/latest`；也可通过
`ASCEND_CANN_PACKAGE_PATH` 或 `ASCEND_HOME_PATH` 环境变量、或 CMake
`-DASCEND_CANN_PACKAGE_PATH=/实际目录` 指定。此构建只需要 CANN host runtime，
不编译 AIV kernel。最后一条命令应指向新构建目录内的 kvaf。

## 本卡 HBM 的 CPU Pull

```python
runtime = kvaf.NpuRuntime(device_id=7, enable_cpu_offload=True)
runtime.link_to("127.0.0.1:28999")
runtime.initialize(ddr_bytes=8 * 131072, slot_size=131072)
runtime.batch_cpu_pull([1, 3], source, [7, 100003], [656, 100001], stream=producer_stream)
checksum = runtime.last_offload_checksum()  # 独立诊断，未计入传输调用
runtime.batch_free([1, 3])
runtime.shutdown()
```

source 是本卡连续 NPU Tensor；元数据使用 Host 整数序列或连续一维 CPU int64 Tensor。偏移和长度均为字节。每个 key 对应一个由 MetaManager 分配的槽位，长度不超过 slot_size；每批最多 4096 个块。省略 stream 时使用当前 NPU stream。

`batch_cpu_pull()` 同步等待源生产、CPU READ 和 Meta Commit；成功返回 None，失败抛异常。注册在源不变时复用，切换源或 shutdown 时注销。相同参数再次调用表示新批次，不是恢复；目标 keys 仍占用时会被拒绝。

明确失败内部 Abort；DMA 未决时保留源和槽位并阻止新操作。应用停止使用该 Runtime，调用 shutdown 核验并清理原操作；shutdown 不发起新 READ，也不能在 DMA 尚未结束时强行释放。Python 会保留源引用；C++ 调用者需保持输入分配有效，直到换源注销成功或 shutdown 成功。已发起 Commit 的批次在确认丢失后仍完成原 Commit 决定。

## 验收命令

```bash
# 本卡便利入口：三轮变化数据，1/656/100001 字节，校验后释放槽位。
python tests/python/test_cpu_offload.py --cpu 127.0.0.1:28999 --device 7

# 应用 Connector 显式接口：两张卡、同一个共享 DDR 池。
python tests/python/test_remote_cpu_offload.py \
  --cpu 127.0.0.1:28999 --device-a 6 --device-b 7
```

本卡测试可用 `--npu-eid` 固定 CPU↔本卡路由、用 `--cpu-candidate` 固定 CPU 设备。共享池首先由 Runtime 分配并固定 CPU candidate，源注册在该池上尝试 NPU EID；不会为每个源重建 DDR 池。

源 EID 自动发现先尝试 CLOS，再尝试 asdrv pair-info。注册含最多 64 字节的 scratch READ 和 checksum 探测，不覆盖已经分配的 DDR 槽位。此前测试机器上 CLOS 路由被拒绝，`000000000000020000100000df00e100` 路由成功；具体选择以当前硬件为准。

本卡验收成功标记为 `LOCAL_CPU_PULL_PASS`，跨卡为 `REMOTE_OFFLOAD_PASS`。建链导入成功不能代替数据校验。

需要普通 BatchPut/BatchGet 和 NPU 直读时，见 [共享 DDR](shared_ddr_remote_offload.md)、[NPU raw READ](hbm_raw_addr_read.md) 和 [Connector 接口设计](connector_pull_api_design.md)。
