# Get 执行与图模式演进

本文记录请求级执行分支合入 main 后的实现与验证边界。范围是四种 Get：
URMA AIV Direct、UBMEM Direct、AICPU Gather Read、AICPU CPU Push Scatter。
Put 保持同步事务语义；同一张图的多个 Get 节点和混合算法不在本轮范围内。

## 1. 合并 main

合入 `de4fd37e02fa86357f71fe4fcbbbecc3e75d1d58`：Scatter 并行描述符构建、
各 Jetty 独立 Ready 消费，以及 AICPU 设备侧 ticket、图捕获约束、状态检查和
kernel profiling。保留本分支的资源 owner / typed view / Invocation、多 session、
canonical 地址转换，以及按输入/输出绑定复用注册的接口。

设备 ABI 不混用：

- `KvafAicpuUmdkControlSessionsV4`
- `KvafAicpuGatherReadExecutionV3`
- `KvafAicpuCpuPushScatterExecutionV3`

CPU 服务、NPU host 与 AICPU 包一起重编译、安装并启动新进程。CPU 服务使用
main 的线程调度行为，`--gather-affinity` 已移除；性能对比显式固定线程数和模式。

性能入口的两个参数独立：`--execution-mode legacy|per-call` 选择 Runtime API，
`--execution eager|graph` 选择执行方式。不能把 per-call 自动降级成 legacy。

保留了 main 删除的资源关闭、HAL 生命周期、加载器和安装事务回归；新增设备 ticket
重复 replay、零请求、动态请求数、捕获期间不得分配/注册/同步和 stream 匹配检查。
本地测试使用模型或 stub；实际通信、图回放和性能仍需在 NPU 上验证。

## 2. Get 驻留实例

每个 Runtime / DDR 代次内，每种已启用 Get 配置首次使用时建立一个实例。
`GetResourceReference` 保留 owner 和绑定的执行状态；`GetResourceLease` 仅表示本次
执行或图所占用的资源。普通同步调用或成功 Drain 后释放执行占用、清空借用的请求与
表信息，实例继续存在；不重复构造和绑定执行状态。

同一 stream、容量未增长的热调用跳过 Describe 和 Warmup。容量增长或换 stream
重新检查并准备；每次依然验证请求/绑定边界，并在必要时等待上一次完成。
`acquire` 计时在热调用中只包含驻留引用上的轻量执行占用。

输出绑定由具体算法实例保存。URMA AIV Direct 与 AICPU Gather 可各持有一个绑定，
并交替调用；UBMEM / Scatter 仍不要求输出注册。释放空闲算法的绑定不释放另一个
在途算法持有的 Tensor 或 AccessTable。一个图仍独占 Get 执行方向，限制一个 Get
节点；已有输出绑定只阻止相同实例进入图模式。

Recover 要求目标实例无输出/图绑定，先完成在途操作，再关闭并重建资源。关闭失败
保留实例、状态及资源引用，允许重试；成功 Shutdown 才统一释放它们。
