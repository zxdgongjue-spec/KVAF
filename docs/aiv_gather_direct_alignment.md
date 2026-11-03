# Gather 必需开销与 direct 对齐

本轮保留现有 capture 查询和 eager 限制；不将删除校验误当作支持 ACL graph。
gather 的 Host sequence 会被 capture 固化，replay 安全还需要独立处理设备侧批次
编号。当前 Host/kernel 仍为 V3、232 B args、192 B profile，CPU wire 协议不变。

## 必须保留的 gather 工作

| 工作 | 原因与边界 |
| --- | --- |
| 构造本批源地址清单 | 动态请求元数据在 NPU，CPU 需要清单才能知道本批复制哪些源地址。当前协议为 48 B header + 每请求 16 B span。 |
| 将清单和通知发送到 CPU | NIC doorbell 只启动网络队列，不能代替 CPU 的软件通知。descriptor、command 必须先于 notify 可见。 |
| CPU 将 payload 复制到连续 staging | 多 SGE READ 需要连续远端源，对应离散本地目标；这是当前 gather 算法的核心额外搬运。SGL=1 仍走该算法。 |
| 确认 CPU 数据准备完成 | 当前整批协议必须等最终 ACK；started ACK 和旧 sequence 不能用于启动数据 READ。以后可以改 chunk 流水，但不能取消数据就绪约束。 |
| 维护控制 SQ/CQ、scratch 和 sequence | 控制队列使用 Scalar 写 SQ，NIC 更新 CQ/ACK；必须保留相应写出/失效和有序完成。 |

描述清单的 MTE3 完成等待仍保留，确保 NIC 不读取尚未写完的 HBM 清单。
控制提交现在将 descriptor 分段、command、notify 合并到一个物理 wave，使用
notify 的 strong place order/completion。标准队列只需要一次 doorbell 和尾 CQ；
小队列仍可为遵守 WQEBB/WR 容量在中途 drain。硬件可见性需要目标机验证。

## 本轮删除或对齐的工作

| 原额外工作 | 当前实现 |
| --- | --- |
| 每次 Host 清零 status/profile | 删除；每个 AIV 核在入口完整初始化其记录。 |
| 每核遍历整表/整组数组做 Scalar Flush | 删除 SIMT 输入的全量 Flush。保留 Scalar 请求数的一条缓存行；Scalar 大 token 回退仅 Flush 本 lane 有效 output_indices。 |
| 普通异步调用自动 D2H status | 删除；与 direct 异步路径一样，仅等待 stream 完成不代表检查设备 status。 |
| 下一次同流调用前强制 Drain | 删除；同流顺序保证共享 workspace、staging 和队列不被并发复用。 |
| descriptor 与 command/notify 分两轮提交并等 CQ | 合并；只有队列容量不足时才额外 drain。 |
| READ 每轮等待上一批 MTE 再继续提交 | 对齐 direct：安排 MTE(n-1)、VF(n)，再提交 n-2；最后排空剩余提交。 |

全部 SGL 值均使用通用多 SGE 编码器，不对 SGL=1 单独分流或特化。WQE 大小
由实际 SGE 数按 `48 + 16 * sges` 向上对齐到 64 B；因此 SGL=1 自然编码为
64 B，但仍执行相同的通用编码逻辑，用于验证通用实现相对于 direct 的开销。

普通异步调用的 output 注册引用在 launch 后释放，但 MR/entity 缓存仍保留到
transport shutdown，和 direct 一致。调用方须持有输入/output Tensor 到执行完成。
切换 stream 时仍需要一次完成 fence，因为本实现共享 descriptor/staging/SQ；
显式 Drain、BatchPut 前的 Drain 和 shutdown 也仍等待未完成调用。

`synchronize=True`、profile 和探针继续 D2H 校验当前批状态。需要诊断异步错误时，
可在创建 Runtime 前设置 `KVAF_AIV_GATHER_CHECK_STATUS=1`，恢复逐批状态回收和
复用前的检查；正式性能脚本拒绝该设置，避免把诊断成本记为异步基线。
ACL 同步失败仍返回错误并保留资源，直至成功同步/停止 transport 后才释放。

保留 Scalar 请求数、持久化队列状态、控制 SQ、CQ 和 ACK 的必要缓存维护。
不能为了减少统计中的 Flush 次数而删掉这些跨 launch、跨核或 NIC 可见性操作。

## 对比性能的条件与尚未对齐的资源

gather 仍为每核一个数据 Jetty、一个控制 Jetty；控制 Jetty 不贡献 payload
并行度。对比本轮代码时，将 direct 的数据 Jetty 配成每核一个，并统一核数、
单 EID、软件在途预算关闭、有效请求数、表规模及源/目标访问顺序。多数据 Jetty
支持不属于本轮协议精简，不能把当前队列数量差异解释成 gather 算法必需成本。

数据 wave 仍遵守已配置的逻辑 WR 上限和 WQEBB 容量。标准队列为 8192 WR、
32768 WQEBB，SGL=1 时每 lane 8193 请求需要两个 wave。没有把 direct 的容量
检查差异直接复制过来，也没有加入每 128 WR 等一次 CQ 的软件窗口。

性能脚本的无 profile 基线改为异步调用 + 调用方 stream 完成，与 direct 的
`operation=batch_get` 对比；profile 组保留同步调用和 status/profile D2H。
CSV schema 3 为各行写入对应 timing_contract，避免混合两种口径。gather 带宽
仍按 p50 计算，direct 脚本的 useful_GiBps 按 mean 计算；跨脚本应统一使用同一
延迟统计量重算带宽。SGL 在性能脚本顶部 GATHER_SGL_SIZE 修改，并核对启动日志。

## 验证范围

本地真实 Host 源码配合延迟 ACL 替身验证连续异步提交、stream 切换、缓存引用、
同步/异步状态回收差异、profile 和失败清理。生产 .asc 的队列模型验证清单/输出、
环回、1～13 SGE、空批/重复批次、大小 SQ、控制合批及 READ 双 bank 同步。
这些测试不模拟 CANN 指令生成、实际缓存层级、DMA 远端可见性或真实性能。

在目标 NPU 更新 Host/Runtime/Python 和 kernel 后，先运行功能回归，再采性能：

```bash
cmake --build build-npu -j
export PYTHONPATH="$PWD/build-npu/python"
export KVAF_BATCH_GET_KERNEL="$PWD/build-npu/bin/kvaf_gather_batch_get_kernels.o"
KVAF_RUN_NPU_TESTS=1 python tests/python/test_aiv_gather_batch_get.py
KVAF_RUN_NPU_TESTS=1 python tests/python/test_aiv_gather_batch_get_perf.py
```

沿用已配置的 CPU 服务地址、BatchPut kernel、设备 ID 和 EID 环境；无需修改 CPU
服务协议。复测必须覆盖同一 Runtime 下多个不同输入批次和 SGL=1/6，不能只用
一次固定输入成功替代缓存/复用验证。
