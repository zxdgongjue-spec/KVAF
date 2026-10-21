# Python BatchGet 使用 AICPU UMDK 流水

已有的 `NpuRuntime.batch_get` 可以选择 UMDK 后端。AICPU 从本轮 NPU Tensor 生成源清单，
CPU gather 到连续 DDR staging；每完成 128 KiB 连续前缀，AICPU 就提交对应 READ。
indices、offsets、mask、output_indices 和数据可以每轮变化，不缓存上一轮内容。

## 构建和启动

在仓库根目录执行：

```bash
./build.sh cpu
./build.sh aicpu-urma --install-kernel
export PYTHONPATH="$PWD/build-aicpu-urma/python:$PYTHONPATH"
```

CPU 服务使用新二进制重启：

```bash
./build-cpu/runtime/kvaf_cpu_service \
  --listen 0.0.0.0 --port 28990 \
  --gather-threads 8 --gather-mode active --gather-affinity on
```

NPU 进程先加载 CANN 环境，再使用现有 Tensor 接口：

```python
import os
import torch
import torch_npu
import kvaf

torch.npu.set_device(7)
runtime = kvaf.NpuRuntime(
    batch_get_kernel_object_path="",
    batch_get_token_bytes=656,
    batch_get_tokens_per_block=1,
    device_id=7,
    npu_core_count=1,
    aicpu_urma_kernel_json_path=os.path.join(
        os.environ["ASCEND_HOME_PATH"],
        "opp/vendors/cust/op_impl/aicpu/config/kvaf_aicpu_urma.json",
    ),
    urma_device="udmac0d1e5",
    urma_eid_index=0,
)
runtime.link_to("127.0.0.1:28990")
runtime.initialize(ddr_bytes=656 * 8192, slot_size=656)

keys = list(range(4096))
source = torch.randint(0, 256, (4096, 656), dtype=torch.uint8, device="npu:7")
runtime.batch_put(keys, source, [i * 656 for i in keys], [656] * 4096)
table = runtime.prepare_access_table(keys)
device_table = runtime.access_table_view(table, device="npu")
indices = torch.arange(4096, dtype=torch.int32, device="npu:7")
offsets = torch.zeros_like(indices)  # 表项内的字节偏移。
mask = torch.ones(4096, dtype=torch.uint8, device="npu:7")
output_indices = indices.clone()
output = torch.empty_like(source)
runtime.batch_get(device_table, indices, offsets, mask, output_indices, output,
                  synchronize=True)
assert torch.equal(output, source)
runtime.release_access_table(table)
runtime.shutdown()
```

设备、URMA device、EID 和服务地址按部署环境调整。原来的 addresses/valid Tensor 重载也可直接使用。
未设置 `aicpu_urma_kernel_json_path` 时仍走原 AIV 后端。

## 执行和资源约定

1..8 个 AICPU worker 共用一对 TP，每个 worker 有独立数据 Jetty/CQ；另有一条专用控制 Jetty/CQ。
CPU 侧每条通道也各有独立 Jetty/RQ/CQ。CPU 与 AICPU 共用端点枚举、队列创建和 TP/Jetty 扩展代码；
AICPU 进程内的 liburma 动态加载与会话由 AICPU 自己维护。

每次最多 65536 项；token 大小由构造参数指定，最大总载荷须能用 uint32 表示。
Runtime 在用户指定 DDR 数据容量之外预留 `65536 * token_bytes` 的 staging、源描述符和控制区，
这些空间不进入 KV slot 分配。HBM 描述符和 scratch 使用独立内部缓冲，输出 Tensor 无需额外尾部空间。
READ 使用 SGL 6、每次提交 32 条 WR；每波最多 4096 条 WR，只请求最后一条的有序 CQE，完成后复用 WR/SGL 空间。

BatchPut 同步完成；BatchGet 默认只 enqueue，`synchronize=True` 等待完成并读取错误状态。
内部 workspace 在下一次操作或 Shutdown 前排空；跨 stream 依赖由调用方负责。
调用方须保留元数据 Tensor 至执行结束，已注册的源/输出 Tensor 至 Shutdown 前保持有效。
同一 AICPU 进程只支持一个活动 UMDK 会话。目前该后端使用 eager 调用，不支持 ACL Graph capture/replay。

本地验证使用 ACL/provider 替身；本分支的 NPU 兼容性与性能需要服务器验证。

## 实现位置

| 目录/文件 | 职责 |
| --- | --- |
| `access_operator/src/aicpu_access_operator.cc` | Runtime 适配、Get/Put 共享会话和跨操作同步 |
| `access_operator/demo/aicpu_batch_get/host/` | BatchGet workspace、参数和 kernel 提交 |
| `access_operator/demo/aicpu_batch_get/kernel/gather_read.cc` | 当前流水算法、READ WR/SGL 组装和调度 |
| `access_operator/demo/aicpu_batch_put/` | 原有同步 BatchPut 的 Host/设备实现 |
| `access_operator/src/umdk/host/` | 通用会话和 kernel 加载 |
| `access_operator/src/umdk/kernel/` | 通用队列、注册、提交、CQ 及统一设备包 |

当前 BatchGet 入口仅调用 `gather_read`；没有其他实现的占位代码或选择参数。
两个 demo 均按 `include/host/kernel` 分层，共用一个设备包和会话；Python 参数及构建命令保持不变。
