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
Gather / Scatter Execution 为 V3。不能混用旧 Host 或设备包。

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
        --graph-api torch --graph-replays 5 \
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
| `async-wait` | 异步提交后外部 `torch.npu.synchronize()` 返回，drain 单独计时 |
| `sync-call` | `batch_get(synchronize=True)` 同步返回 |

AIV/UBMEM eager 完成路径不清零、回读或检查 Get 状态报告；ACL 失败仍返回错误。
显式 profile 快照在计时外读取并验证设备报告。UBMEM Put 和 graph 释放保留各自检查。
默认关闭 profiling，仅 `scope=end_to_end` 报告 Host 时间和带宽。
上面的命令显式加入 `--kernel-profile`，开启算子打点、profile 回读及详细输出，
并增加 `scope=kernel` 的设备时间与带宽；UBMEM 同时采集地址/MTE/drain 细分时间。
测无 profiling 的端到端性能时去掉该参数，eager 和 graph 均适用。
Graph 回放不重新进入 Host Dispatcher，不能代替 eager 调度开销测量。

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
run_put urma aicpu 5
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

多卡并发使用 `tests/run_remote_cpu_batch_get_multicard_perf.sh`，四种 backend 的命令和计时口径见
[多卡带宽测试](batch_get_multicard_performance.md)。

`tests/python/` 保留性能入口与 trace 汇总，公共模块放在 `utils/`。
独立 C++/Python 功能、配置和单元测试及配套 stub/model、shell 入口、CMake 注册已移除。
性能预检覆盖被测数据路径与图回放，不提供原 matrix、overlap、lifetime、resources、
lifecycle 或故障注入测试的完整覆盖。
