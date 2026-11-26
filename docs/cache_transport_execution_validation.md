# 缓存访问执行框架：性能验证步骤

CPU 服务、NPU Host 库、设备 kernel 与安装的 AICPU 包须来自同一提交。
当前 Runtime 只保留 unified 路径，命令不再使用 `--execution-mode`。
接口和资源契约见[实现与使用](cache_transport_execution_implementation.md)。

## 构建并启动匹配的服务

在 CPU 主机仓库根目录构建并启动服务：

```bash
./build.sh cpu
./build-cpu/runtime/kvaf_cpu_service \
  --listen 0.0.0.0 --port 21111 \
  --gather-threads 8 --gather-mode active
```

测试期间共用此服务；Gather 的线程数不小于客户端 worker 数。
按路由需要增加 `--ub-device NAME --eid-index N`。
UBMEM 需要目标硬件和驱动支持 Host P2P DDR/Fabric 映射。

NPU 主机加载实际 CANN 环境后，在仓库根目录执行：

```bash
./build.sh npu
./build.sh npu --check-kernel
```

`npu` 默认构建全部后端和 Python binding，并安装 AICPU 包。
使用新的 CPU 服务和 NPU 客户端进程。默认产物位于 `build-npu/python/kvaf` 与
`build-npu/bin`。AIV Get / Put 入口分别为 V3 / V2；AICPU Control 为 V5，
Gather / Scatter Execution 为 V5。不能混用旧 Host 或设备包。

## 四种 Get 共用一个入口

在 NPU 仓库根目录运行。跨主机时替换 CPU 地址；设备号和核数按实际环境调整。
以下命令测四种后端、三个规模，每组包含 eager 和 graph：

```bash
bash <<'BASH'
set -euo pipefail
perf_root="reports/get-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$perf_root"
for misses in 2048 4096 8192; do
  for backend in aiv ubmem gather_read cpu_push_scatter; do
    case "$backend" in
      aiv) resources=(--aiv-cores 2 --jettys-per-core 2) ;;
      ubmem) resources=(--aiv-cores 16) ;;
      *) resources=(--aicpu-workers 5) ;;
    esac
    env -u KVAF_GET_DISPATCH_TRACE -u KVAF_PUT_DISPATCH_TRACE \
      KVAF_CPU_SERVICE_ADDR=127.0.0.1:21111 DEVICE_ID=2 \
      KVAF_PERF_REUSE_GET_OUTPUT=1 \
      bash tests/run_remote_cpu_batch_get_perf.sh build-npu \
        --backend "$backend" "${resources[@]}" \
        --execution both --get-timing async-drain \
        --graph-replays 5 \
        --request-length 32768 --table-capacity 4096 --miss-count "$misses" \
        --token-bytes 656 --tokens-per-block 128 \
        --seed 20260903 --no-sort-input-indices \
        --source-access-pattern random --output-access-pattern contiguous \
        --warmup 10 --iterations 100 --kernel-profile \
        2>&1 | tee "$perf_root/get-${backend}-${misses}.log"
  done
done
rg '^BATCH_GET_RESULT |^BATCH_GET_GRAPH_PASS ' "$perf_root"
BASH
```

`--graph-replays 5` 用于计时前的动态输入、部分和零请求验证；正式样本数由
`--iterations 100` 决定。数据准备、预检、预热、输出校验和 profile 快照均在性能计时之外。
Get 性能测试的具体计时与带宽定义见[统一 Get 性能测试](batch_get_performance.md)。

| eager 计时参数 | 计时结束位置 |
|---|---|
| `async-drain`（默认） | 异步提交后 `drain_get()` 返回，包括完成处理和请求清理 |
| `sync-call` | `batch_get(synchronize=True)` 同步返回 |

AIV/UBMEM eager 完成路径不清零、回读或检查 Get 状态报告；ACL 失败仍返回错误。
显式 profile 快照在计时外读取并验证设备报告。UBMEM Put 和 graph 释放保留各自检查。
`scope=end_to_end` 在 eager 下报告 Host 时间和带宽，
在 graph 下报告 replay 前后两个 NPU Event 的间隔及对应带宽。
`scope=kernel` 默认报告设备时间与带宽。上面的命令加入 `--kernel-profile` 查看分段明细。
Graph 回放不重新进入 Host Dispatcher，不能代替 eager 调度开销测量。

可追加 `--multi-graph-check` 做计时前的串行多图验证：同一 stream 上交替回放两张图，
其中一张含两个 Get 节点，覆盖完整、部分和零请求；释放双节点图后继续验证另一张图。
Gather/Scatter 还验证只回放 B 时，未回放的 A 节点不能通过完成检查；
加上 `--kernel-profile` 时，A 的 profile 查询必须拒绝，分别回放后各节点报告不能串用，
且只回放 A 不应改写 B 的 profile 快照。另用一张未回放的临时图验证：先销毁图，
普通结束必须拒绝，随后 `discard_result=True` 能释放保留的租约。这些检查均在正式计时之外。
通过时输出 `BATCH_GET_MULTI_GRAPH_PASS`。随后执行自动资源 scope 检查：
在两条 stream 上分别准备/捕获图，共用一个输出，第一张图包含两个使用相同输出的节点；
串行执行 B → A → 普通 Get/Drain → A，销毁 B 后再回放 A，最后结束所有图后执行 eager。
检查输出及 AICPU 图报告隔离，通过时输出 `BATCH_GET_GRAPH_RESOURCES_PASS`。
跨 stream 切换前由脚本明确等待，不依赖 KVAF 自动安排事件。
正式性能图也使用 `graph_resources().prepare/capture` 自动收集，准备及结束均在计时外。
该检查不验收图内混合算法或跨 stream 并发回放，正式性能样本仍为单节点图。

针对四种 Get 的图资源与 AICPU 节点报告隔离回归，可运行以下命令（CPU 服务地址和设备按现场替换）：

```bash
bash <<'BASH'
set -euo pipefail
for backend in aiv ubmem gather_read cpu_push_scatter; do
  KVAF_CPU_SERVICE_ADDR=127.0.0.1:38999 DEVICE_ID=6 \
  bash tests/run_remote_cpu_batch_get_perf.sh build-npu \
    --backend "$backend" --execution graph \
    --request-length 32768 --table-capacity 4096 --miss-count 4096 \
    --token-bytes 656 --tokens-per-block 128 \
    --multi-graph-check --kernel-profile --warmup 10 --iterations 100
done
BASH
```

每个后端均应输出 `BATCH_GET_MULTI_GRAPH_PASS`、`BATCH_GET_GRAPH_RESOURCES_PASS`
和 `BATCH_GET_GRAPH_PASS`。核数/worker 使用该后端默认值；需要对比性能时保持原配置一致。
所有后端默认使用 ACL 图接口。AICPU Scatter 固定使用原生 ACL 图，其他后端可追加 `--graph-api torch` 验证 Torch 图。

## Put 使用同步性能入口

支持 URMA AIV、UBMEM AIV、URMA AICPU。每次调用等待完成并处理事务，
预检、预热、key 释放及最终 shutdown 在计时外：

```bash
bash <<'BASH'
set -euo pipefail
mkdir -p reports/put
run_put() {
  local transport="$1" executor="$2" cores="$3"
  env -u KVAF_NPU_BLOCK_BYTES -u KVAF_GET_DISPATCH_TRACE -u KVAF_PUT_DISPATCH_TRACE \
    KVAF_CPU_SERVICE_ADDR=127.0.0.1:21111 DEVICE_ID=2 \
    KVAF_PERF_REUSE_PUT_INPUT=1 KVAF_NPU_REQUESTS=1024 \
    KVAF_NPU_TOKEN_BYTES=656 KVAF_NPU_TOKENS_PER_BLOCK=128 \
    KVAF_NPU_CORE_COUNT="$cores" KVAF_NPU_JETTYS_PER_CORE=2 \
    KVAF_NPU_WARMUP=10 KVAF_NPU_ITERATIONS=100 \
    bash tests/run_remote_cpu_batch_put_perf.sh build-npu \
      --transport "$transport" --executor "$executor" \
      2>&1 | tee "reports/put/${transport}-${executor}.log"
}
run_put urma aiv 2
run_put ubmem aiv 16
run_put urma aicpu 1
BASH
```

URMA AIV/AICPU 默认复用输入绑定；UBMEM 不需要输入 MR 注册。
设置 `KVAF_PERF_REUSE_PUT_INPUT=0` 测每次注册/注销的普通调用。
Put 的有效载荷为 `rows * token_bytes * tokens_per_block`，不能把 rows 直接当作 Get miss 数。
结果含 `p50_ms`、`p90_ms`、`p99_ms` 和 `p50_GBps`，绑定与解绑耗时单独报告。
Put 预检和测后检查属于 `kind=metadata`，检查 key、长度、有效位及槽位地址。
Get 的数据预检另行逐字节比较读回结果。

## 绑定与分段定位

URMA AIV Direct 和 Gather 默认在预热前绑定输出，测量结束后解绑。
设置 `KVAF_PERF_REUSE_GET_OUTPUT=0` 对照临时输出路径。
绑定/解绑耗时通过 `PERF_OUTPUT_BINDING` 单独报告，不计入复用路径的稳态样本。
绑定期间不能 resize 或更换底层存储；请求 Tensor 和访问表 pin 仍按每次调用管理。
显式输入/输出绑定 API 及失败恢复见[实现与使用](cache_transport_execution_implementation.md)。

设置 `KVAF_GET_DISPATCH_TRACE=get-dispatch.csv` 或
`KVAF_PUT_DISPATCH_TRACE=put-dispatch.csv` 后运行相应性能命令，再汇总：

```bash
python3 tests/python/summarize_get_dispatch_trace.py get-dispatch.csv --last 100
python3 tests/python/summarize_put_dispatch_trace.py put-dispatch.csv --last 100
```

trace 在 shutdown 时落盘。Get 按算法、路径、事件分组；Put 还区分 transport、executor
与成功/失败。provider 内的准备、提交、完成、清理和 Runtime 外层成本可分别观察。
分段嵌套或重叠，各段 P50 不能相加得到总 P50；最终性能对照关闭 trace。

Python Get 同时写出 `<trace路径>.host.csv`，汇总命令会自动读取并输出 `GET_HOST_TIMING`。
只记录成功的提交调用，不增加设备同步或状态回读，未设置 trace 时不读取计时器。
`total` 是绑定函数内的提交耗时；`tensor_views` 读取七个 Tensor 的元数据和指针，
`request_validation` 检查布局并组装请求，`stream` 获取 ACL stream，`refs` 保留 Python 引用，
`binding_lock` 包含绑定层等待锁及对应的 GIL 切换，`set_device` 是 `aclrtSetDevice`，
`owner_update` 是提交后保存请求引用。`binding_other` 包含剩余 GIL 切换和未单独计时的收尾。

`core_call` 包含 RuntimeCore 的全部调用；其中 `core_table` 查找 AccessTable，
`core_direction` 等待 Get 方向锁，`core_previous_drain` 收尾前一笔请求，
`core_validate` 校验请求并保留表引用，`core_dispatch` 是 Dispatcher 的调用，
`core_finish` 是返回前的请求状态处理。`core_*` 是 `core_call` 内部的分段，不能再加到外层。
绑定入口前的 Pybind 参数转换和返回转换不在 `total` 内，仍以 Python 的 `batch_get_submit`
为完整调用计时。异步提交后的等待仍看 Dispatcher `event=drain`，不计入这些提交字段。

多卡并发使用 `tests/run_remote_cpu_batch_get_multicard_perf.sh`，四种 backend 的命令和计时口径见
[多卡带宽测试](batch_get_multicard_performance.md)。

`tests/python/` 保留性能入口与 trace 汇总，公共模块放在 `utils/`。
独立 C++/Python 功能、配置和单元测试及配套 stub/model、shell 入口、CMake 注册已移除。
性能预检覆盖被测数据路径与图回放，不提供原 matrix、overlap、lifetime、resources、
lifecycle 或故障注入测试的完整覆盖。

普通 Get 恢复另有一个不依赖 SDK 的定向 Host 故障注入回归，源码及编译命令见
[诊断目录](../tests/diagnostics/README.md#get-recovery-fault-injection-host-only)。
它覆盖持续性设备错误、关闭/重建失败重试、表 pin 和输出绑定的保留与退役，
不替代真实 DMA、队列关闭或设备错误后的目标机验收。

## 自动图资源变更的验证边界

接口迁移与可运行代码见[实现篇最小示例](cache_transport_execution_implementation.md#python-migration)，
两节点模型图、固定输入更新及关闭顺序见[完整图示例](cache_transport_execution_implementation.md#get-graph-example)。
可将最小示例代码块与图示例代码块按顺序保存为 `kvaf_api_examples.py`，在匹配 NPU 构建的环境中运行：

```bash
PYTHONPATH="$PWD/build-npu/python${PYTHONPATH:+:$PYTHONPATH}" \
KVAF_CPU_SERVICE_ADDR=127.0.0.1:38999 DEVICE_ID=6 \
KVAF_EXAMPLE_BACKEND=aiv python3 kvaf_api_examples.py
```

分别改为 `ubmem`、`gather_read`、`cpu_push_scatter`，各次应同时输出
`KVAF_EAGER_EXAMPLE_PASS` 与 `KVAF_GRAPH_EXAMPLE_PASS`。示例验证小规模真实 Put/Get、
两节点加普通算子捕获、动态 offset、部分/零请求和正常释放，不替代上面的规模与性能回归。

vLLM/adaptor 集成验收还需要使用框架实际 forward：

- 每个图 key 的预热和 capture 具有相同的 Get 顺序、配置、固定输出和请求容量；
  未发生 cache miss 的调用点仍用 `miss_count=0` 执行，不跳过节点。
- 在模型自己的静态输入更新后回放，与相同输入的 eager 有效输出对比；覆盖完整、部分及
  零请求，不能只以单节点 benchmark 的 PASS 代替模型校验。
- 先预热同实例的最大请求容量，再准备多个图；用框架的等待或事件依赖串行验证图 A、
  图 B、eager fallback，以及淘汰 B 后继续回放 A。
- 捕获失败或未回放图按外部图销毁、`close(discard_result=True)` 的顺序清理；失败保留
  graph entry 并重试，不直接清除资源 owner。全部图结束后才能关闭 Runtime。

本次新增的 `graph_resources`、图/eager 共存和不同图独立 stream，需重新执行上述真机回归；
旧版本的 PASS 和性能数字不构成本次变更的硬件验收。prepare 内普通调用真实执行，
capture 不应再产生输出注册或 workspace 扩容；回放仍调用原生 graph，不加入 Host 代理。
故障时先停止/销毁图再 close/discard；设备完成未知不能当作注销成功。

还需在应用的实际模型 forward 中确认预热与捕获路径一致。一次资源 prepare 只覆盖其
收集到的节点顺序，动态 miss 数可变，新增分支、不同配置/输出 allocation 或容量应拒绝；
先 prepare 最大规格，再准备同实例较小规格的图。图仍持有资源时请求更大容量应返回 Busy。

本地检查已完成：非 NPU CMake 构建和 Python 导入检查；新增绑定全文件 C++17
语法检查；实际新增 pybind 类配合临时 native 桩的 20 项正反例（含四算法配置混合
与错误顺序拒绝）；Dispatcher 与 Gather 注册共享/故障释放的 ASan/UBSan 检查。
这些检查没有真实 CANN 队列或 DMA，不能替代上述 NPU 命令和应用模型捕获验收。
