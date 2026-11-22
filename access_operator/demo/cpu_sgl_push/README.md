# CPU-SGL Push Demo (A5)

验证目标：CPU 侧用 URMA `urma_post_jfs_wr` WRITE + **本端 SGL**（多段散列 DDR）
把 token 行直接推进 **NPU HBM 的连续 run**，与现有 NPU AIV 逐 token READ 拉取
对比同 miss 数下的落地时间。

## 协议约束（openEuler URMA 文档）

- 单边 write/read：**远端只能 1 个 sge**（必须连续）；
- **本端可以是 SGL**：`urma_post_jfs_wr` 支持本端非连续 → "CPU 散列 → NPU 连续"
  只能走 CPU 推（WRITE），不能走 NPU 拉（READ 的远端无法 gather）。

## 组成

- `device/npu_push_target.cpp`：NPU 进程。ACL + direct-hccp（与 BatchGet 同一套
  RA/CtxInit/TokenAlloc/LmemRegister/CqCreate/QpCreate），把 HBM hot buffer 注册成
  可远程写的 segment，把 {eid, mem_key, jetty_key, token_value, hbm_addr, flag_addr}
  写进 keyfile；随后轮询 HBM flag（8B seq），每看到新 seq 记录 CLOCK_MONOTONIC，
  结束后对落地行做自描述模式校验。
- `cpu/cpu_sgl_push.cpp`：host 进程。liburma context，注册本地 DDR（自描述 token 行），
  读 keyfile 导入 NPU 的 segment 和 jetty，每个迭代把 miss 列表按 group（默认 16）
  聚成若干 run，每个 run 一条 WRITE WR（src=sgl[k] 散列 DDR 行，dst=1 sge 连续 run），
  链式一次 post，最后挂一条 8B flag WR（place_order=strong），poll JFC 拿完成时间。
- 两进程同机同时钟（CLOCK_MONOTONIC），用 seq 对齐时间戳。

## 运行

```bash
# 容器内编 NPU 侧（无需 ccec，纯 host 代码）
bash scripts/build_device.sh /path/to/kvaf /path/to/out
# host 编 CPU 侧（需要 urma 头文件，仓内 third_party/umdk/include 已带）
bash scripts/build_host.sh /path/to/kvaf /path/to/out
# 端到端（单机）：起 NPU target -> 等 keyfile -> 跑 CPU pusher -> 收集 CSV
bash scripts/run_sgl_demo.sh 0 102 16 200   # device0, miss=102(5%), group=16, 200 迭代
```
