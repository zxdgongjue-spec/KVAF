# NPU0 读写 NPU1 HBM

本分支沿用现有 CPU 服务的使用流程：服务分配内存池，客户端 `batch_put`
写入 Key 对应的数据，再通过 `prepare_access_table` 和 `batch_get` 读回。
NPU1 上的池由 `aclrtMalloc` 分配；它不是已有 Tensor 的导出接口。

## 建链与数据路径

```text
NPU0 / Python NpuRuntime                  NPU1 / kvaf_npu_service
RuntimeCore + MetaManager                NpuRuntime (HBM provider)
  Key -> slot                              aclrtMalloc HBM pool
  DeviceAccessTable                        Direct-HCCP MR + token
  AIV BatchPut / BatchGet                   每个通道独立的 target Jetty / CQ
        |                                          |
        +--- TCP: EID / region / segment / CTP -----+
        |         Jetty key / PSN / release         |
        +--- UB: AIV WRITE -> HBM，AIV READ <- HBM -+
```

`link_to(address, peer_device="npu")` 选择 HBM allocator。
`initialize(pool_bytes, slot_size)` 请求服务分配并注册 HBM，交换 segment key、
原始 token ID 和导出的 UB 地址，再交换双方 Jetty/CTP/PSN 建链。
初始化沿用 64B AIV WRITE/CQ 连通性探测。BatchGet 使用同一套 AIV READ 路径，
MetaManager、AccessTable 和在途控制的语义不变。
数据面直接访问远端 HBM；TCP 只承载控制信息。

NPU1 通过 Direct-HCCP 管理 RA context、MR、token 和 target Jetty，
不初始化 CPU `liburma` 数据面。UMDK 公开头文件仅用于解析 MR 返回的
`urma_seg_t`；远端地址取 segment 中的 UB 地址，token ID 在客户端按原有规则
转换一次。每个 session 独立拥有 HBM、MR、token 和通道。

## 构建

两端需要支持现有 A5 AIV/UB 路径的 Ascend 驱动、CANN 和配置好的 UB 网络。
客户端还需要 `torch_npu`。两端使用本分支构建，控制协议版本为 **v7**；
原有 CPU 服务也必须同步重编译，不能与旧 v6 二进制混用。

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/ascend-toolkit/latest
# 如 CANN 或系统 include 中没有 urma_api.h，指定匹配平台 UMDK 的公开头目录：
export KVAF_NPU_URMA_INCLUDE_DIR=/path/to/umdk/include
./build.sh npu
```

也可通过 `KVAF_URMA_ROOT` 指定包含 `include/urma_api.h` 的安装根目录。
构建生成 `build-npu/runtime/kvaf_npu_service`、Python binding 和两个 AIV kernel。
直接使用 CMake 时，服务开关为 `KVAF_BUILD_NPU_HBM_SERVICE`，默认随
`KVAF_ENABLE_AIV_BATCH_GET` 开启；关闭服务开关仍可构建原有 NPU 客户端。

## 双端验证

先在提供 HBM 的进程启动服务，示例使用本机逻辑卡 1：

```bash
./build-npu/runtime/kvaf_npu_service \
  --listen 0.0.0.0 --port 28999 --device-id 1
```

在 NPU0 客户端执行：

```bash
KVAF_NPU_SERVICE_ADDR=NPU1_HOST_IP:28999 DEVICE_ID=0 \
  bash tests/run_remote_npu_e2e.sh /absolute/path/to/KVAF/build-npu
```

同机两卡可用 `127.0.0.1:28999`；跨机时 `--device-id` / `DEVICE_ID` 分别是
各进程可见的本地逻辑卡号，不要求服务端一定为 1。TCP 可达不代表 UB 可达，
必须通过初始化的 RMA 探测和读回校验。

两端默认自动发现 EID。需要固定时，服务使用可重复的 `--npu-eid <32位十六进制EID>`，
客户端使用 `KVAF_NPU_EIDS=<EID>`，多个客户端 EID 用逗号分隔。
服务使用 `--hbm-allocator huge|normal` 选择 ACL 大页优先或普通分配，默认 `huge`。
`--max-sessions` 默认 64，`--max-channels` 为每个 session 的通道上限，默认 4096。
CPU 专用的 `adaptive_ub_device`、`peer_ub_device` 和
`adaptive_peer_ub_devices` 不适用于 NPU peer，设置后会明确报错。

脚本复用现有数据校验，检查 BatchPut 后的 BatchGet 内容、A/B/A 输出切换、
miss-count 边界，并默认执行 3 次 graph replay；可用 `KVAF_NPU_GRAPH_REPLAYS=0`
关闭图验证。缺少 NPU、binding 或 kernel 时脚本会失败。成功时应看到
`RUNTIME_NPU_BATCH_GET_VALIDATION_PASS` 和
`RUNTIME_NPU_DYNAMIC_OUTPUT_VALIDATION_PASS`；服务端最后应出现
`HBM_RELEASED ... remaining=0`（仅一个客户端时）。初始化探测也可能产生临时 session。

## 性能测试

性能测试可复用现有 `run_remote_cpu_batch_get_perf.sh` 和
`run_remote_cpu_batch_get_miss_count_perf.sh`，指定 `KVAF_PEER_DEVICE=npu` 与
`KVAF_PEER_SERVICE_ADDR` 即可；也可使用 `--peer-device npu --peer-service-addr host:port`。
脚本保留原名和 CPU 默认值，计时前仍先 BatchPut 填充远端池并验证读回，计时不包含填充。

先用单组参数验证性能流程：

```bash
KVAF_PEER_DEVICE=npu KVAF_PEER_SERVICE_ADDR=NPU1_HOST_IP:28999 DEVICE_ID=0 \
  bash tests/run_remote_cpu_batch_get_perf.sh /absolute/path/to/KVAF/build-npu \
  --table-capacity 8192 --request-length 8192 --miss-count 8192 \
  --tokens-per-block 128 --token-bytes 656 --aiv-cores 2 --jettys-per-core 1 \
  --warmup 3 --iterations 100 --kernel-profile
```

完整扫描使用相同配置入口，输出各 shape 的日志及汇总 CSV：

```bash
KVAF_PEER_DEVICE=npu KVAF_PEER_SERVICE_ADDR=NPU1_HOST_IP:28999 DEVICE_ID=0 \
  bash tests/run_remote_cpu_batch_get_miss_count_perf.sh /absolute/path/to/KVAF/build-npu \
  --table-capacity 8192 --request-length 8192 --tokens-per-block 128 \
  --warmup 3 --iterations 100
```

扫描包含 2646 组 token-bytes、miss-count、AIV core 与 Jetty 数量组合。
不要为 NPU peer 传 `--adaptive-ub-device` 或 `--peer-ub-device`，它们只用于 CPU 服务。
`table-capacity` 是有效 block 数，池容量为
`table_capacity × tokens_per_block × token_bytes`；上述命令在最大 token 配置下
分配 4 GiB。旧版默认 table-capacity 在该配置下需要 624 GiB，应根据 NPU1
可用 HBM 显式设置。CPU/NPU 对比测试应使用相同容量和访问模式。
各 shape 的 `BATCH_GET_PERF_CONFIG` 日志记录实际 `peer_device`。

## Python 调用示例

设置 `PYTHONPATH=/absolute/path/to/build-npu/python`，并将以下 kernel 路径改为实际路径：

```python
import torch
import torch_npu
import kvaf

torch.npu.set_device(0)
token_bytes, tokens_per_block = 656, 128
block_bytes = token_bytes * tokens_per_block
runtime = kvaf.NpuRuntime(
    "/absolute/path/to/build-npu/bin/kvaf_execute_batch_get_kernels.o",
    batch_get_token_bytes=token_bytes,
    batch_get_tokens_per_block=tokens_per_block,
    device_id=0,
    batch_put_kernel_object_path=
        "/absolute/path/to/build-npu/bin/kvaf_execute_batch_put_kernels.o",
)
runtime.link_to("NPU1_HOST_IP:28999", peer_device="npu")
table = None
try:
    runtime.initialize(2 * block_bytes, block_bytes)
    assert runtime.remote_region["device"] == "npu"
    source = (torch.arange(2 * block_bytes, dtype=torch.int64) % 251).to(
        dtype=torch.uint8, device="npu:0"
    ).reshape(2, tokens_per_block, token_bytes)
    runtime.batch_put([0, 1], source, [0, block_bytes], [block_bytes, block_bytes])
    table = runtime.prepare_access_table([0, 1])
    view = runtime.access_table_view(table, device="npu")
    indices = torch.tensor([0, 1], dtype=torch.int32, device="npu:0")
    offsets = torch.tensor([3 * token_bytes, 7 * token_bytes],
                           dtype=torch.int32, device="npu:0")
    miss_count = torch.tensor([2], dtype=torch.int32, device="npu:0")
    rows = torch.tensor([0, 1], dtype=torch.int32, device="npu:0")
    output = torch.zeros((2, token_bytes), dtype=torch.uint8, device="npu:0")
    runtime.batch_get(view, indices, offsets, miss_count, rows, output)
    torch.npu.synchronize()
    expected = torch.stack([source[0, 3], source[1, 7]]).cpu()
    assert torch.equal(output.cpu(), expected)
finally:
    # 默认 BatchGet 只入队；同步后才能解锁、释放或销毁相关 Tensor。
    torch.npu.synchronize()
    if table is not None:
        runtime.release_access_table(table)
    if runtime.initialized:
        runtime.shutdown()
```

`batch_free(keys)` 回收池内 slot；`shutdown()` 先释放客户端数据面，再通知服务释放
session。服务按 imported Jetty → target Jetty/CQ → MR → token → HBM 的顺序回收。
独立服务进程在客户端正常退出后再停止；异常退出客户端的 session 没有自动租约回收，
需要重启服务。一个客户端 Runtime 当前管理一个 peer region。

## 验证范围

无硬件测试编译生产服务代码并替换 ACL/RA 调用，覆盖协议、指定设备、分配失败回滚、
独立通道、token/PSN 交换、数量限制和资源释放顺序。它不能证明实际 UB 连通性或
设备侧读写成功；完整 CANN 构建及上述双 NPU 脚本仍需在目标环境执行。
