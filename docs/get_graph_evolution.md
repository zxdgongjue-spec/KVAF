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
