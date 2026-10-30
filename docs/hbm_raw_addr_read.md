# A 直接读取 B HBM：batch_get_raw_addr

A 沿用现有 `NpuRuntime`：BatchPut/BatchGet 访问 CPU DDR，
`offload_from` 通知 CPU 拉取 B 的 HBM₁；新增 `batch_get_raw_addr` 直接读取 B 的 HBM₂。
两块 B HBM 独立分配、注册、发布和确认，互不借用 DDR ticket 或冻结状态。

## 接口

### B：HbmReadSource

```python
source = kvaf.HbmReadSource(device_id=7, npu_eid=b_fullmesh_eid)
ticket = source.register(hbm_for_a, stream=producer_stream)
port = source.listen("127.0.0.1:0")
# 上层把可从 A 访问的 host:port 和 ticket 传给 A，并等待 attach 完成。
```

- `register(source, stream=None)`：保留连续的 B NPU Tensor，等待初始化写入完成，返回区域 ticket。
  实际 HCCP MR/target Jetty 在 A—B 协商 EID 时建立。注册至 attach 完成期间不要改写 HBM。
  Tensor 起点无需按页对齐：底层按 4 KiB 向下对齐注册起点并覆盖 Tensor 的全部字节，
  发布的源地址仍是 Tensor 的实际导出起点，范围不包括对齐引入的前缀。
- `listen(address)`：启动 B 进程内的 C++ 控制线程，返回实际端口；端口 0 自动选择。
  当前为 IPv4 `host:port`。跨主机部署可监听 `0.0.0.0`，交给 A 的地址必须是可达地址。
- `ready(input_offsets, lengths, stream=None)`：等待该生产 stream 上的写入完成，冻结本批数据。
  返回 `version/source_id/epoch/sequence/source_addresses/lengths`。
  **source_addresses 是实际导出的 B HBM 绝对地址**，不是查表索引。
- `release_batch()`：只有 A 确认完成/未发 DMA 的取消后才能解冻；收到上层完成通知后调用。
- `cancel_batch()`：仅能取消尚未被 A Acquire 的 Ready 批次。拒绝取消已被 A 获取的批次。
- `shutdown()`：必须先解冻批次、由 A `detach_hbm`，才能关闭服务并释放 B 注册。
- `connected/frozen/selected_npu_eid`：连接、冻结状态和选中的 B 路由 EID。

Python 持有 source Tensor 的强引用直到 shutdown 成功；C++ 调用者必须自行保持存储有效。
冻结属于使用契约，无法阻止上层绕过 API 直接写 Tensor。

### A：现有 NpuRuntime

```python
region = runtime.attach_hbm(b_address, ticket, local_eid=a_fullmesh_eid)
# B 写完后，上层将 ready_batch 传给 A。
runtime.batch_get_raw_addr(
    region, ready_batch, output=a_output,
    output_offsets=[0, 1024, 3072], stream=a_stream,
)
# 成功后通知 B 执行 source.release_batch()，再进入下一轮。
# 所有批次释放后：
runtime.detach_hbm(region)
```

- `attach_hbm(address, export_ticket, local_eid="") -> region`：独立选择 A—B EID 组合，
  导入 B MR、双向绑定 Jetty，执行覆盖所有配置通道的小块 READ/checksum 探测。
  `local_eid` 只约束 A—B 连接，DDR 的 `npu_eids` 配置仍独立。
  **NPU—NPU 直连需要双方的 fullmesh EID**：分别在 B 的 `HbmReadSource(npu_eid=...)`
  和 A 的 `attach_hbm(local_eid=...)` 显式传入。当前自动发现仅包含 CLOS/pair-info，
  尚未发现 fullmesh；留空的候选探测不能替代 fullmesh 配置。
- `batch_get_raw_addr(region, ready_batch, output, output_offsets, stream=None)`：同步执行直接 READ。
  每段源地址、长度取自 ready_batch；目的地址为 `output.data_ptr() + output_offsets[i]`。
  所有偏移/长度以字节为单位，变长段超过 64 KiB 会拆分为多个 READ。
  output 必须为 A 卡连续 Tensor；目的范围必须在该 Tensor 内，且各段不能重叠。
  每个 Ready 批次消费一次，已完成 sequence 的再次提交返回错误，不重放 READ。
- `resolve_raw_addr(region)`：只重试控制确认或资源清理，不重发 READ。
  Acquire 回复不确定且 A 尚未发 DMA 时，恢复通过取消结束，返回错误表示没有产出有效数据。
  Complete 回复丢失时重试 ACK，已完成的数据不会再读取。
- `pending_raw_addr`：是否存在必须保留输出/源的未解决操作。
- `detach_hbm(region)`：无未完成读取时释放 A 资源，再确认 B 释放对端通道和注册。
  丢失 Close 回复可以重复 detach。

`batch_get_raw_addr_kernel_object_path` 是新增的可选 runtime 参数。
缺省使用 BatchGet kernel 同目录下的 `kvaf_execute_batch_get_raw_addr_kernels.o`。
该功能要求 `KVAF_ENABLE_AIV_BATCH_GET=ON`；B 的 `HbmReadSource` 在任一 CANN 功能构建中可用。

当前范围：一个 source 对应一个 A reader；每个 source 同时一批冻结数据；每批最多 4096 段，
拆分后最多 1048576 个 WQE；A—B 合计 1–4 个 Jetty（默认 2 核 × 2 Jetty）。
该限制来自复用的四通道 HBM exporter，不影响原 DDR BatchGet 的 Jetty 配置范围。
新增接口同步完成，不支持 graph capture 或并发调用同一 reader；DDR BatchGet 的现有行为不变。

## 数据面和控制面

数据面直接传入三个设备数组：`source_addresses:uint64[]`、
`destination_addresses:uint64[]`、`lengths:uint32[]`。
没有 AccessTable、key/indices/valid 查询，也不经过 DDR MetaManager。

新 AIV 内核复制当前 BatchGet V2 的 READ WQE 编码、双 bank 提交、多 Jetty 调度、
inflight 窗口和 CQ 处理。独立符号 `KvafExecuteBatchGetRawAddrAivKernelV1` 与独立 object
使现有 BatchGet ABI 保持不变。裸地址范围同时在 host 和内核检查。

KVHR v1 是独立的 A—B TCP 控制协议，只传 ticket、连接描述、范围和确认；不传 payload。
A 的 HbmReader 保存连接/导入 MR、USER_CTL Jetty/CQ、队列状态和未完成任务。
B 的 HbmReadSource 保存 Tensor、导出 MR、目标 Jetty、客户端身份、Ready 范围和版本。
底层复用 HBM Exporter；NPU 的 32 字节 Jetty key 与 CPU 的 28 字节 key 分开校验。
A 的每个发起 Jetty 与 B 对应目标 Jetty 分别交换配对 TP，B 按通道保存 TP；CPU Pull 原有共享 TP 不变。
provider 返回的 segment/Jetty 描述原样保留；CTP 路由 EID 与 WQE 内存身份 EID 分开使用。

成功返回必须同时满足：AIV stream 完成、各核 CQ/状态成功、A 输出 MR 注销成功、B 完成 ACK。
用户输出不会作为零引用缓存留在 transport 中。
数据读取超时/错误会隔离连接并保留资源，不凭超时允许 B 覆盖，也不自动重放 READ。
启动探测失败只有在 AIV kernel 已退出且所有 initiator Jetty 成功销毁后才尝试下一候选；
无法证明停止访问时同样保留 A scratch 和 B 注册。

## 编译与硬件验证

CPU service 不需要新增功能，继续运行当前 `build-cpu-shared-ddr/runtime/kvaf_cpu_service`。
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
