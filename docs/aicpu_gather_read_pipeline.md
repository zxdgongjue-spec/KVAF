# CPU gather / AICPU READ 分块流水

`--pipeline-chunk-bytes N` 开启同一轮内的流水，`0` 保留整块 gather ACK 后再 READ 的对照路径。流水要求 `--descriptor-transport urma --read-schedule dynamic`，块大小必须为正的 64 B 倍数。建议先比较 `0`、`65536`、`131072`；这些是实验参数，不是硬件限制。

CPU 收到源清单后，一次完成清单检查和前缀偏移计算。拷贝线程动态领取连续 staging 字节块，一个块可以跨多个源 span。线程完成整块拷贝后标记该块完成；使用原子 CAS 推进连续完成前缀，将 `ready_bytes` 发布到 DDR 清单头，不增加块间的互斥锁或线程唤醒。后面的块先完成时，不会越过未完成的前块发布进度，也不会回退已发布进度。设备可见性沿用内存屏障，不增加数据缓存行刷新。

AICPU 控制 kernel 写入清单并通知 CPU 后返回，不等待整块 ACK。后续 READ kernel 的各 worker 使用独立 SQ/CQ 和独立 64 B HBM 轮询缓冲，通过 URMA READ 获取 `ready_bytes`。只有某个 READ 任务的整个 staging 范围都在就绪前缀内时，才提交该任务的数据 READ；已经观察到的进度在 worker 内复用。CPU 拷贝后续块与前面任务的 READ 可以重叠。

CPU 完成所有块并更新 CPU profile 后发布最终 ACK。AICPU worker 0 完成自己的 READ 任务后确认最终 ACK；整个 stream 结束还需要全部 worker 的数据 READ CQE 完成。staging 本轮各块互不覆盖，下一轮必须在当前同步调用返回后才能复用。进度中的错误位、无进度超时和 CQ 错误都会使调用失败。

两个粒度独立：`--pipeline-chunk-bytes` 是 CPU 字节块大小，`--read-task-spans` 是 AICPU 每个 READ 任务最多包含的目标 span 数。若 READ 任务过大或首个任务覆盖全部数据，流水会退化成等待大块完成再 READ。

计时含义：

| 输出 | 区间 / 包含关系 |
| --- | --- |
| CPU read and prepare source list | CPU 检查清单、计算偏移、准备拷贝线程 |
| CPU gather DDR -> staging | CPU 所有块的拷贝及进度发布，包含线程同步 |
| AICPU READ staging -> HBM | 首个 READ worker 进入至最后一个数据 READ worker 完成，包含等待 ready 前缀 |
| Wait for ready prefixes | 每轮各 worker 累计进度轮询时间的最大值，已包含在 READ 总时间内 |
| Wait for READ completions | 数据 READ 的 CQE 等待，不包含进度查询的 CQE 等待 |
| Final CPU ACK confirmation | worker 0 数据 READ 结束后读取最终 ACK/profile，可能与其他 worker 重叠 |

CPU gather 与 AICPU READ 相互重叠，不能相加。进度轮询的 WQE 不计入数据 `read_wqes`。

## 编译和运行

清单头升级为 48 B，magic 升级为 GDR2；CPU service、客户端和 AICPU kernel 必须一起更新。普通串行路径也使用新版清单头。

在 CPU service 所在机器编译并重启服务：

```bash
./build.sh cpu
./build-cpu/runtime/kvaf_cpu_service \
  --listen 0.0.0.0 --port 28990 \
  --gather-threads 8 --gather-mode adaptive --gather-affinity off
```

在 NPU 客户端所在机器编译、安装 kernel，然后比较串行与流水：

```bash
./build.sh aicpu-urma --install-kernel

for chunk_bytes in 0 65536 131072; do
  ./build-aicpu-urma/tests/kvaf_aicpu_gather_read_client \
    --peer 127.0.0.1:28990 --device-id 6 \
    --blocks 4096 --block-bytes 656 --output-stride 1312 \
    --sgl-size 6 --aicpu-cores 5 --sq-batch-wqes 8 --sq-copy memcpy \
    --read-schedule dynamic --read-task-spans 192 \
    --descriptor-transport urma --pipeline-chunk-bytes "$chunk_bytes" \
    --profile-detail 1 --warmup 10 --iterations 1000
done
```

远程 CPU service 时，将 `--peer` 改为实际地址。本地测试只验证小规模内存的正确性、流水顺序、变化布局和失败路径；设备上的性能收益需要用这组对照命令测量。
