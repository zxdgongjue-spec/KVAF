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

## 3. 四种 Get 的图模式

四个后端均通过 `begin_get_graph(config, output, capacity, stream)` 提前准备资源；
捕获时 `batch_get(..., execution=config, graph_lease=lease, synchronize=False)`
只提交已准备的设备任务。固定配置、stream、输出地址和容量后，可在相同输入存储中
更新 indices、offsets、miss_count、output_indices，重放全量、部分和零请求。
图期间禁止同一 Runtime 的额外 Get / Drain / Warmup / Recover / Shutdown。
一个租约仅允许一个 Get 节点，Put 不进入图。

AIV Direct 保持原状态处理语义：不增加每次状态清零或结果回拷。UBMEM 保留自身
完成检查。AICPU Gather/Scatter 使用设备侧 ticket 推进跨 replay 的通信序号；
捕获阶段不注册、不分配、不同步，显式状态检查须在 replay 完成后执行。
未重放的 AICPU 图正常结束会报未完成，可 `discard_result=True` 取消。
同步/注销失败保留资源和租约以供重试；AICPU 协议错误后显式 discard 会关闭该算法
owner，之后必须 `recover(config)`，不会复用可能损坏的队列。

Profiling 和状态查询通过算法配置路由到驻留实例，默认取 Runtime 的默认 Get。
图前启用 profile、每轮完成后读取，销毁图并结束租约后停用。保留最后成功 profile
供结束后读取。AICPU 的 `check_batch_get_status(stream, execution=config)` 检查
实际对应算法；AIV 无需新增这一检查。

更新 CPU、NPU 和 AICPU 包后，先验证全部功能（同一 CPU 服务进程）：

```bash
KVAF_CPU_SERVICE_ADDR=127.0.0.1:38999 DEVICE_ID=6 \
bash tests/run_cache_execution_e2e.sh "$PWD/build-npu" \
  --report "$PWD/validation-four-get-graph.json"
```

`graph` 案例已覆盖四种 Get；聚焦验证可加 `--modes urma+ubmem --cases graph`。
统一性能入口也可逐个比较 eager / graph，不改变接口模式：

```bash
for backend in aiv ubmem gather_read cpu_push_scatter; do
  KVAF_CPU_SERVICE_ADDR=127.0.0.1:38999 DEVICE_ID=6 \
  bash tests/run_remote_cpu_batch_get_perf.sh "$PWD/build-npu" \
    --execution-mode per-call --backend "$backend" --execution both \
    --request-length 32768 --miss-count 4096 --table-capacity 4096 \
    --token-bytes 656 --tokens-per-block 128 --aiv-cores 2 --jettys-per-core 2 \
    --warmup 10 --iterations 100 --graph-replays 100 --kernel-profile || break
done
```

kernel profile 用于设备分析；Host 延迟对照使用两种接口模式相同的 profile 设置与
`--get-timing sync-call`。便携回归覆盖四算法路由、捕获限制、stream 校验、失败重试、
空/部分/全量请求及 profile 收尾；它们不替代真机 E2E 和性能验证。
