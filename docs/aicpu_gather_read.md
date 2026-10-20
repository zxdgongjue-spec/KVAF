# AICPU UMDK 流水 GatherRead

CPU 把当前源清单 gather 到连续 DDR staging，完成一段连续前缀后发布 `ready_bytes`。
AICPU 每批重新构造 READ WR/SGL，确认整批源数据就绪后立即提交，边提交边继续构造下一批。
每个 worker 的一波最多 4096 条 WR，只在最后一条请求数据 CQE；完成前保留所有 WR/SGL 槽位。

## 编译与运行

在仓库根目录编译 CPU 服务和 NPU 客户端：

```bash
./build.sh cpu
./build.sh aicpu-urma --install-kernel
```

CPU 终端：

```bash
./build-cpu/runtime/kvaf_cpu_service \
  --listen 0.0.0.0 --port 28990 \
  --gather-threads 8 --gather-mode active --gather-affinity on --gather-notify-profile
```

NPU 终端：

```bash
./build-aicpu-urma/tests/kvaf_aicpu_gather_read_client \
  --device-id 7 --peer 127.0.0.1:28990 \
  --urma-device udmac0d1e5 --eid-index 0 --aicpu-cores 1 \
  --blocks 4096 --block-bytes 656 --output-stride 1312 \
  --sgl-size 6 --sq-batch-wqes 32 --pipeline-chunk-bytes 131072 \
  --profile-detail 1 --warmup 10 --iterations 1000
```

设备和 EID 按部署环境调整。`pipeline-chunk-bytes=0` 等完整 CPU ACK 后再 READ。
1..8 个 worker 使用同一对 TP，各有独立数据 Jetty/CQ，并共享一条专用控制 Jetty/CQ。
控制队列负责清单 WRITE、通知、ACK 和前缀查询；CPU 服务为每个通道分别创建 Jetty、RQ 和 CQ。
总 CPU 线程预算由 `gather-threads` 设置，与 AICPU worker 数分别配置。

只构建现有 AIV/Python 路径仍使用 `./build.sh npu`；它的默认选择和原有公共请求结构保持不变。

## C++ 调用

给 `NpuRuntimeOptions::aicpu_urma_kernel_json_path` 设置安装后的 JSON 路径，即选择 UMDK 后端。
`urma_device`、`urma_eid_index` 和 `npu_core_count` 配置该会话；BatchPut、BatchGet、初始化探测和
GatherRead 复用同一会话，不再为 UMDK 创建 Direct-HCCP 连接。
未设置 JSON 的现有调用继续使用原 AIV 后端。

调用 `NpuRuntime::GatherRead(CpuGatherReadRequest)`，提供本轮源 span、DDR staging、
DDR descriptor 邮箱、HBM 输出 span 和调用方 stream。源/目标清单、地址和长度可逐轮改变；
只复用缓冲区容量，每轮覆盖全部有效描述符和 WR/SGL。不会保存上一轮清单结果。

Host 信任调用方的有效布局，不执行逐轮全量校验。源总量与目标总量必须一致，地址及容量必须有效，
DDR 源、staging、descriptor、区域末尾 64 B control 不得重叠。descriptor 地址按 64 B 对齐，
至少保留 `peer::GatherDescriptorBytes(source_count)` 字节。
staging、descriptor 和末尾 control 所在 KV slot 必须由调用方保留，不参与普通数据分配；测试客户端会实际占住这些 slot。
HBM 输出末端额外空间用 `GatherReadPackedOutputBytes(output_extent, source_bytes, target_count, chunk_bytes)`
计算，包含 scratch、源清单与目标清单；辅助函数位于 `gather_read_storage.h`。
CPU/AICPU 边界检查、CQ 错误和超时仍返回失败。

同一 AICPU 进程目前只支持一个活动 UMDK 会话，第二次创建会明确失败。
已注册的调用方 HBM 缓冲在 Runtime Shutdown 前须保持有效；不支持释放后复用地址。
GatherRead 同步返回；BatchPut 同步返回；BatchGet 保留默认只 enqueue、可选 synchronize 的原契约。
调用方负责跨 stream 依赖，并在 Shutdown 前完成异步 BatchGet。
GatherRead 上传或同步失败时保留 pinned metadata，后续调用或 Shutdown 先排空 stream，避免提前复用。
客户端、CPU 服务和 AICPU kernel 一起更新。

## 性能与正确性

抽取前已测基线：device 7，CPU 8 线程 active/绑核，单 AICPU worker，4096 × 656 B，
输出 stride 1312，SGL 6，batch 32，128 KiB 前缀流水，warmup 10、测量 1000 轮。
端到端均值 **258.4 µs / 10.40 GB/s**，P50 **257.0 µs**，P95 **280.0 µs**。
这份整理后的代码还需在相同设备和条件下复测。

客户端每轮更新 payload、改变源清单顺序、毒化输出，并逐字节验证载荷及输出间隙。
输入上传、造数和结果验证不计入 GatherRead 时间。
日志中的 Host wait 包含传输、kernel 与 CPU/AICPU 工作；流水 READ 包含 ready-prefix 等待，
CPU gather 与 READ 重叠；多 worker 阶段取每轮最大值，不能相加。
Provider submit 包含 provider 编码、SQ 写入与 doorbell，不能当作单独的 SQ 拷贝计时。

无 NPU 的 native 回归在存在 UMDK headers 时使用 provider 与 ACL 替身，覆盖平铺 BatchPut、
异步 BatchGet、1/2/4 worker、变化布局、前缀就绪约束及 stream 失败恢复。它们验证行为和生命周期；
上面的 NPU 命令用于确认硬件兼容性和性能。
