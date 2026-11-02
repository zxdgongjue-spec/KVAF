# CPU Pull 构建与验证

当前使用统一 `NpuRuntime` 和共享 DDR。CPU 数据面保留参考分支的四 Jetty 批量 READ，公开接口使用 KVRP v5；CPU 服务与 NPU Python 扩展必须同时重编译、重启。

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

# 78 个独立分配的 tensor，逐个注册 MR、共用连接；每个 tensor 拉取两轮。
python tests/python/test_remote_cpu_offload.py \
  --cpu 127.0.0.1:28999 --device-a 6 --device-b 7 \
  --tensors 78 --rounds 156
```

本卡测试可用 `--npu-eid` 固定 CPU↔本卡路由、用 `--cpu-candidate` 固定 CPU 设备。共享池首先由 Runtime 分配并固定 CPU candidate，源注册在该池上尝试 NPU EID；不会为每个源重建 DDR 池。

源 EID 自动发现先尝试 CLOS，再尝试 asdrv pair-info。注册对每个 MR 各做最多 64 字节的 scratch READ 和 checksum 探测，不覆盖已经分配的 DDR 槽位。此前测试机器上 CLOS 路由被拒绝，`000000000000020000100000df00e100` 路由成功；具体选择以当前硬件为准。

本卡验收成功标记为 `LOCAL_CPU_PULL_PASS`，跨卡为 `REMOTE_OFFLOAD_PASS`。建链导入成功不能代替数据校验。

需要普通 BatchPut/BatchGet 和 NPU 直读时，见 [共享 DDR](shared_ddr_remote_offload.md)、[NPU raw READ](hbm_raw_addr_read.md) 和 [Connector 接口设计](connector_pull_api_design.md)。

## 多 MR CPU Pull 接口

生产端调用 `source.register(ticket, tensors)` 注册列表，再用 `source.ready(tensors[i])`
选择本批源。原单 tensor 的 `register(ticket, tensor)` 和 `ready()` 保持可用。
读取端仍使用 `attach_cpu_source(desc)` 和
`submit_cpu_pull(ready, put, src_offsets, dst_addrs, lengths)`，不新增句柄参数。
`src_offsets` 始终相对选定 tensor，`dst_addrs` 来自本次 `prepare_put()` 分配。

78 个 MR 共用一组 4 Jetty。每个共享 DDR Region 只保留一份 pinned 本地接收 MR，
另外仍保留原先供 NPU BatchGet/BatchPut 访问的 DDR 导出 MR。`ready(tensor)`
生成的 MR 身份由 Connector 原样转发，CPU 内部选择对应的导入段和 token 信息。
注册后 tensor 的地址和大小不能变化，所有分配保留到 shutdown 成功；当前批次未释放前
不能改写选中 tensor。使用同一个 Source 对象的批次依次提交、完成、释放。

成功注册日志包含 `mrs=78 jettys=4 probe=all_mrs_scratch_checksum_pass`。
`REMOTE_PULL_DONE` 中的 `mr` 随所选 tensor 改变；验收终行应有
`REMOTE_OFFLOAD_PASS ... tensors=78 single_mr_per_batch=1`。
每轮 checksum 和重复提交校验继续保留，且不重新注册或重建连接。

## 性能测试与阶段统计

CPU 服务开启统计后，每个源连接独立累计 READ 样本，正常注销源时输出汇总；READ
失败时也会输出当时的汇总。探测与校验 RPC 不计入样本。可沿用已有构建目录，例如：

```bash
cmake --build build-cpu-shared-ddr -j8
KVAF_CPU_PULL_PROFILE=1 \
KVAF_CPU_PULL_PROFILE_WARMUP=10 \
KVAF_CPU_PULL_PROFILE_EACH=0 \
  ./build-cpu-shared-ddr/runtime/kvaf_cpu_service --listen 0.0.0.0 --port 28999
```

`KVAF_CPU_PULL_PROFILE_WARMUP` 默认为 0，按每个源连接的 READ 调用次数排除前 N 批，
应与 Python 的 `--warmup` 一致。`KVAF_CPU_PULL_PROFILE_EACH=0` 关闭逐批日志，保留汇总；
不设此项则仍输出原有 `CPU_PULL_PROFILE` 逐批日志。

```bash
python tests/python/bench_remote_cpu_offload.py \
  --cpu 127.0.0.1:28999 --device-a 6 --device-b 7 \
  --blocks 1024 --block-bytes 83968 --warmup 10 --iterations 100
```

CPU 输出两组 `CPU_PULL_SUMMARY`：`scope=all` 包含预热，`scope=measured` 排除预热。
上例成功执行后分别为 110 和 100 个样本。`profile_id` 区分源连接，头部还包含 source_base、
ddr_base、调用数、失败数、丢弃样本数；失败仅计数，不混入成功延迟分布。
各 `CPU_PULL_SUMMARY_STAGE` 输出 mean/P50/P95/P99/min/max，单位微秒：

| CPU 阶段 | 含义 |
| --- | --- |
| `build` | 构造本轮分段、SGE、WQE 与窗口 |
| `post` | 向 provider 批量提交 WQE 的累计时间 |
| `cq_wait` | 等待各队列完成的累计时间 |
| `other` | READ 总时间减去 build/post/cq_wait 的余量 |
| `total` | CPU backend 的 READ 调用时间，不含摘要采集/打印、校验、上层控制 RPC |

Python 输出 `CPU_PULL_BENCH_PROFILE_STAGE`，同样排除预热并给出全部分位数：

- reader：`prepare_put`、`dst_addrs`（应用组织 DDR 绝对地址）、`submit_cpu_pull`、`wait_cpu_pull`、`commit_put`，其总和为 `total`。
- reader 另列 `last_offload_checksum`、`verification`、`batch_free`，均不包含在传输 `total` 中；checksum 是 verification 的组成部分，不能重复相加。
- producer：`ready`、`release_batch`。源写入与同步、业务通知 IPC、注册/建链均不计入传输总时间。

`CPU_PULL_BENCH_PASS` 的 `metric=python_prepare_put_to_commit` 包含事务/控制开销，
不能与 CPU backend 的 total 等同；也不跨进程相减时间戳。
CPU 与 Python 分位数统一按 `(N-1)*p` 线性插值，各阶段分位数不可相加。
统计只在开启时存储逐批耗时，不保存 payload；汇总与排序在 READ 计时之外完成。
这些脚本不再包含 serial 模式。
