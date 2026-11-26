# Host CPU RoCE / HBM Direct RDMA 带宽工具

客户端在 A5 Host 上申请并注册一块 NPU HBM buffer，Host CPU 使用多个提交线程并发驱动多个 RC QP 的 RDMA WRITE/READ，网卡直接在这块 HBM 与服务端注册的 DDR region 之间搬运 payload。每个 QP 有独立 CQ、Host worker 和 HBM/DDR 子区域；所有 QP 共享一个 PD，因此 HBM MR 和服务端 DDR MR 各注册一次。工具分别输出 READ/WRITE 的聚合带宽。

这里的 “HBM Direct RDMA” 指 RoCE RNIC 的 DMA 直接读写已注册的 NPU HBM。WQE 仍由 Host CPU 通过 `ibv_post_send()` 提交；这不是 AIV/NPU 侧构造 WQE、敲 doorbell 的 NDA 直驱路径。它适合验证 HBM 注册后经 RoCE 到远端 DDR 的数据路径及多 QP 聚合带宽。

## 构建

A5 客户端需要 Linux 上的 CANN、Ascend 驱动及 libibverbs 开发库。加载本机 CANN 的 `set_env.sh` 后，从 KVAF 根目录执行：

```bash
cmake -S access_operator/demo/host_roce -B /tmp/kvaf-host-roce \
  -DCANN_ROOT="$ASCEND_HOME_PATH"
cmake --build /tmp/kvaf-host-roce -j
```

驱动装在其他目录时传入 `-DASCEND_DRIVER_PATH=/实际驱动目录`。远端 CPU 机器只需构建 DDR 服务端：

```bash
cmake -S access_operator/demo/host_roce -B /tmp/kvaf-roce-cpu \
  -DBUILD_HOST_ROCE_CLIENT=OFF
cmake --build /tmp/kvaf-roce-cpu -j
```

## 启动

使用本目录构建的 `kvaf_roce_cpu_service`。它处理 bench 的多 QP 握手，
构建业务 Runtime 得到的同名服务使用单 QP 握手。

服务端每次接受一个客户端连接，结束后退出。每轮测试前启动一次：

```bash
/tmp/kvaf-roce-cpu/kvaf_roce_cpu_service \
  --listen 90.96.15.111 --port 29999 \
  --rdma-device hrn5_0 --rdma-port 1 --gid-index 1
```

客户端参数中的第一个位置参数是 ACL 逻辑设备 ID，TCP 地址用于控制面握手；数据面使用 `--rdma-device` 指定的 RoCE 设备：

```bash
/tmp/kvaf-host-roce/kvaf_host_roce 0 90.96.15.111:29999 \
  --rdma-device hrn5_6 --rdma-port 1 --gid-index 1 \
  --bytes 4194304 --depth 32 --qp-count 4 \
  --warmup 100 --iterations 10000 --operation both
```

参数说明：

- `--bytes`：每个 RDMA WR 的字节数，默认 1 MiB；每笔 WR 长度必须能由 verbs SGE 长度表示。
- `--depth`：每个 QP 的 outstanding WR 数，默认 64，最大 128。
- `--qp-count`：并发 RC QP 数，也是 Host 提交线程数，默认 1，最大 32。实际可创建数量由网卡和驱动资源决定。
- `--warmup`：每个 QP 在每种操作计时前的预热 WR 数，默认 100。
- `--iterations`：每个 QP 在每种操作中的采样 WR 数，默认 1000；总 WR 数是 `iterations × qp-count`。
- `--operation read|write|both`：指定测 READ、WRITE 或两者，默认 both。
- HBM buffer 和远端 DDR region 大小均为 `bytes × depth × qp-count`，当前客户端和服务端总上限为 1 GiB。调高 QP 数时可相应降低每 QP depth。

工具会先将固定模式写入所有 DDR 槽位，再执行所选采样。每种操作的采样只统计并发 WR 提交、CQ 轮询和完成等待，不包含线程/QP 初始化、MR 注册、TCP 握手和数据校验。每种操作结束后还会读取全部槽位并校验数据。输出的 `GBps` 是十进制 GB/s，`GiBps` 是二进制 GiB/s，`Gbps` 是十进制 Gbit/s；WRITE 表示 NPU HBM → 远端 DDR，READ 表示远端 DDR → NPU HBM。

建议固定 WR 大小，逐步测试 `--qp-count 1 2 4 8`，并在每个 QP 数下调整 `--depth 16 32 64 128`。READ 深度由设备查询到的 `max_qp_rd_atom`、`max_qp_init_rd_atom` 决定，代码不再额外限制为 16；运行日志中的 `read_depth` 是设备能力协商结果。每次修改客户端参数前都要重启服务端，因为 region 大小由客户端握手申请。达到性能平台后，再检查链路计数器、Host CPU 和服务端资源是否成为瓶颈。

## 8 张 NPU / 8 张 RDMA 网卡并发测试

一条映射行对应一个客户端进程、一个 ACL NPU device、一个本地 RDMA device 和一个服务端控制端点。当前 CPU demo 服务端每个进程只接收一个客户端，因此服务端需要按映射表手动启动 8 个进程，使用不同控制端口和各自对应的 RDMA device。控制面可以使用 A5 可访问的同一个服务端 IP 加不同端口；数据面由 `--rdma-device` 和双方的 GID 选择实际 RoCE 网卡。

在 A5 上复制并填写映射模板，尤其要按实际机器确认 NPU ID、`hrn5_*` 和 GID 的对应关系：

```text
link0  0  <npu0对应的客户端网卡>  <gid>  <server-ip>:29990
link1  1  <npu1对应的客户端网卡>  <gid>  <server-ip>:29991
...
link7  7  <npu7对应的客户端网卡>  <gid>  <server-ip>:29997
```

每行字段依次为 `label ACL_DEVICE_ID CLIENT_RDMA_DEVICE GID_INDEX SERVER_HOST:PORT`，以空格或 Tab 分隔。完成服务端 8 个进程的启动后，在 A5 并发执行一轮 WRITE：

```bash
bash access_operator/demo/host_roce/run_multi_client_bw.sh \
  --demo /tmp/kvaf-host-roce/kvaf_host_roce \
  --mapping ./client_mapping.tsv --operation write \
  --bytes 1048576 --depth 16 --qp-count 1 --warmup 10 --iterations 1000
```

脚本会为每条映射并发启动一个客户端。`--qp-count` 指每个客户端进程在该行映射的 NPU/RDMA 网卡上创建的 QP 数；例如 8 行映射配 `--qp-count 4`，就是 8 个客户端进程、共 32 个 QP，仍然是一张 NPU/网卡对应一台服务端网卡。缓冲区大小按每客户端 `bytes × depth × qp-count` 计算，客户端和服务端单进程的 region 上限均为 1 GiB。保存的各链路日志会包含每个客户端内多个 QP 的聚合带宽；脚本再把各客户端结果相加，生成 `summary.tsv` 和总聚合带宽。

完成 WRITE 后，重新启动服务端 8 个进程，再以 `--operation read` 执行同一命令得到 READ 汇总。READ 的远端数据预置发生在采样计时前。若任一客户端失败，脚本会返回非零并标记汇总不完整；日志保留在输出目录中供排查。默认 `--qp-count 1` 适合先验证并发启动，确认稳定后再逐步增加 QP 数、`--bytes`、`--depth` 和 `--iterations`。
