# NPU 直接读取对端 HBM：batch_get_raw_addr

同一个 NpuRuntime 可访问 CPU DDR，并通过独立 fullmesh 链路读取另一张卡的 HBM。本路径不使用 AccessTable、keys 或 DDR MetaManager。B 可分别注册一块供 CPU Pull 的 HBM 和一块供 NPU Pull 的 HBM。

## 生产端 B

```python
source = kvaf.NpuPullHbmSource(device_id=7, npu_eid=b_fullmesh_eid)
ticket = source.register(hbm_for_a, stream=producer_stream)
port = source.listen("127.0.0.1:0")
# 应用把可达的控制地址与 ticket 发给 A，等待 A attach 完成。
# 每批：应用先完成生产同步，再把 ready 与业务元数据发给 A。
producer_stream.synchronize()
ready = source.ready()
# 收到 A 应用的完成通知后：
source.release_batch(ready)
```

register 保留连续 NPU Tensor，初始化实际注册在 attach 时完成。Tensor 起点无需页对齐：底层向下对齐 4 KiB 并覆盖整个 Tensor，逻辑源起点仍为 Tensor 的实际地址。注册至探测完成期间保持数据稳定。

ready 不携带范围、不等待生产 stream；应用负责同步。release_batch(ready) 只在读取完成/已安全取消后成功，cancel_batch(ready) 只允许取消尚未被读取方获取的批次。两者都核验源身份和 sequence。注销前先释放批次、由 A detach，再 source.shutdown()。

## 读取端 A

```python
link = runtime.attach_hbm(b_address, ticket, local_eid=a_fullmesh_eid)
# 收到 ready 后，应用计算三个等长、连续的一维 int64 NPU Tensor。
with torch.npu.stream(a_stream):
    src = torch.tensor([7, 100003, 250019], dtype=torch.int64, device=output.device)
    dst = torch.tensor([100, 500, 4096], dtype=torch.int64, device=output.device)
    sizes = torch.tensor([1, 656, 100001], dtype=torch.int64, device=output.device)
runtime.batch_get_raw_addr(link, ready, src, dst, sizes, output, stream=a_stream)
# 成功后由应用通知 B 释放批次；所有批次结束后 runtime.detach_hbm(link)。
```

源偏移相对于注册 HBM，目的偏移相对于 output，均以字节计。output 和三数组必须在 Runtime 所在 NPU，连续且存储不重叠；各目的段不得重叠，范围须落在注册源和 output 内。应用保证元数据在传入 stream 上就绪，并保持到完成/恢复结束。

当前只保留该 NPU Tensor 重载。成功返回 `{"state": "done", "sequence": ready.sequence}`，包含 READ 完成、输出注册清理、生产端完成确认。未决时用 `resolve_raw_addr(link)` 重试控制或清理，绝不重发 READ；`pending_raw_addr` 表示仍需保留资源。若获取批次的响应丢失而尚未发 READ，恢复会取消并返回错误，不能将 output 视为有效。

双方必须显式配置 fullmesh EID：B 构造源时的 npu_eid 和 A attach_hbm 的 local_eid。CPU↔NPU 路由独立配置，自动 CLOS/pair-info 发现不能替代 fullmesh。

## 算子与资源

raw READ V2 在同一次算子中完成整批校验、分段、WQE 提交和 CQ 等待。三数组不回传 Host，没有独立准备算子。短段直接索引，长段通过 UB 前缀表定位；沿用 BatchGet 的 WQE 编码、双 bank、多 Jetty、inflight 窗口和 CQ 处理。Host 只读取最终状态。

仅建链的 all-lane 探测仍使用内部 Host 绝对地址描述符模式，不再作为公开数据读取接口。探测失败必须确认 kernel 已退出且 initiator 队列成功销毁后才能换候选。

KVHR v4 控制协议只传源/连接身份和批次号，不传三数组或 payload。A 的 HbmReader 保存连接/导入 MR、USER_CTL Jetty/CQ、队列和未决任务；B 的 NpuPullHbmSource 保存 Tensor、导出 MR、目标 Jetty、客户端身份和批次状态。各通道分别协商 TP，CPU Pull 的共享 TP 路径独立。

每批最多 4096 段、拆分后最多 1048576 个 WQE，每段 READ 最多 64 KiB；每个源绑定一个 reader，同时一个活动批次，默认 2 核 × 2 Jetty，总计最多 4 个 Jetty。接口同步，不支持同一 reader 的并发调用或 graph capture。

使用 `KvafExecuteBatchGetRawAddrAivKernelV2`，旧 V1 object 不兼容。NPU 运行库和 raw kernel object 需一起重编译。底层 provider 返回的内存身份与路由 EID 分开保留；不手动改写 MR/Jetty 的导出身份。

## 编译与硬件验证

CPU service 与 NPU 扩展需同时重编译并重启，以使用 KVRP/KVHR v4。
在 Linux 测试机已有的 CANN、torch_npu 和 liburma 环境中更新源码，然后重新编译 NPU 侧：

```bash
cmake -S . -B build-shared-ddr \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DKVAF_BUILD_REMOTE_CPU_SERVICE=OFF \
  -DKVAF_BUILD_PYTHON=ON \
  -DKVAF_ENABLE_CPU_OFFLOAD=ON \
  -DKVAF_ENABLE_AIV_BATCH_GET=ON \
  -DKVAF_ENABLE_AIV_BATCH_PUT=ON \
  -DBUILD_TESTING=OFF \
  -DPython_EXECUTABLE="$(command -v python)"
cmake --build build-shared-ddr -j8
export PYTHONPATH="$PWD/build-shared-ddr/python${PYTHONPATH:+:$PYTHONPATH}"
python tests/python/test_hbm_read_raw_addr.py \
  --cpu 127.0.0.1:28999 --device-a 6 --device-b 7 --rounds 3 \
  --npu-eid-a-raw '<A_FULLMESH_EID>' \
  --npu-eid-b-raw '<B_FULLMESH_EID>'
```

运行前将两个占位符替换为对应卡的实际 fullmesh EID。

脚本自动启动 B 子进程和 B 的临时控制监听端口。在同一个 A runtime 中逐轮验证：

1. A BatchPut 写入 DDR 的 marker 保持正确。
2. CPU Pull B HBM₁ 到共享 DDR，A BatchGet 读回并逐字节比较。
3. A 直接 READ B HBM₂ 的 1/656/100001 字节三段，检查目的位置间隔、padding 和 NPU 消费者可见性。
   HBM₂ 使用页内偏移为 `0xc00` 的连续 Tensor view，覆盖非页对齐的注册场景。
4. B 两块缓冲区分别收到确认后复用，内容每轮变化。
5. 先释放批次、A detach、B shutdown，最后关闭 A runtime。

可独立指定 `--npu-eid-a`（DDR）、`--npu-eid-a-raw`（A—B，A 的 fullmesh）、
`--npu-eid-b-cpu`（CPU—B）、`--npu-eid-b-raw`（A—B，B 的 fullmesh）。
成功输出 `HBM_RAW_PASS two_b_buffers=1 shared_runtime=1 no_access_table_for_hbm=1`。
出现 unresolved 时测试会保留进程/缓冲区，需先检查两端日志和确认传输停止。
若 attach 已明确失败且没有未解决的传输，A 会通知 B 关闭源，避免 B 等待 attached 超时。

本地无 Ascend 硬件时，`kvaf_batch_get_raw_addr_kernel_model_test` 运行实际 AIV 源码的 CPU 模型；
`kvaf_hbm_read_test` 使用真实 TCP/source/client 与假 ACL/HCCP 验证状态、分段、丢 ACK 和清理。
这些测试不替代 Linux CANN 编译和真实 A—B READ 验收。
