# 缓存访问执行框架：验证步骤

本文对应 `codex/cache-transport-execution` 分支的请求级执行入口。
实现边界见[实现与使用](cache_transport_execution_implementation.md)。

## 1. 验证入口与范围

- 真机验收：[run_cache_execution_e2e.sh](../tests/run_cache_execution_e2e.sh)，调用
  [test_cache_execution_e2e.py](../tests/python/test_cache_execution_e2e.py)。
- 便携回归：CTest；其中 `kvaf_cache_execution_harness_python` 检查验收脚本的
  字节比较、测试计划、失败报告、参数透传、逐 key 缺失断言及清理顺序，不访问设备。
- 旧 `run_remote_cpu_e2e.sh` 主要验证旧 AIV 入口，不覆盖本次新增框架。

真机脚本显式启用 `per_call_execution` 并设置默认配置、enabled 集合和各 provider
资源预算。AIV Gather / Scatter 不在 enabled 集合中；它们的调用用于验证未启用配置
会被拒绝，不计作可用数据路径。

默认依次运行三种模式，每种模式使用自己的 Runtime 和 DDR 池：

| 模式 | Put | Get | 顺序写读组合 |
|---|---|---|---:|
| `urma` | AIV、AICPU | AIV Direct、AICPU Gather、AICPU Scatter | 6 |
| `ubmem` | AIV | AIV Direct | 1 |
| `urma+ubmem` | URMA AIV、URMA AICPU、UBMEM AIV | URMA AIV Direct、URMA AICPU Gather / Scatter、UBMEM AIV Direct | 12 |

每个模式的各执行配置在同一个 Runtime 中共存。混合池不是为每一对组合重新分配。
矩阵阶段不主动预热，覆盖首次调用的资源准备；后续测试覆盖显式预热与重建。

## 2. 两端部署

CPU 与 NPU 两端都更新到同一个分支和提交。不要将新 Host 库、旧 CPU 服务或旧 kernel
混用。需要目标平台的 URMA/UB 驱动、CANN、PyTorch 和 torch_npu；UBMEM 与混合模式还
需要两端支持 Fabric DDR 分配与映射。普通网络连通不等于支持 Fabric。

在 CPU 主机的仓库根目录：

```bash
./build.sh cpu
./build-cpu/runtime/kvaf_cpu_service \
  --listen 0.0.0.0 --port 28999 \
  --gather-threads 8 --gather-mode active --gather-affinity off
```

确认日志出现 `LISTEN`。按实际路由需要增加 `--ub-device NAME --eid-index N`。
验证完整矩阵时不能使用 `--without-urma`。如果使用非默认 HAL 库路径，给 CPU 服务
增加 `--hal-library /absolute/path/to/libascend_hal.so`。

在 NPU 主机的仓库根目录加载实际 CANN 环境，以下路径仅为常见安装示例：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
./build.sh npu
./build.sh npu --check-kernel
```

`./build.sh npu` 默认构建全部已接入后端和 Python binding，并安装 AICPU 包。
安装后使用新的 NPU 客户端进程。默认产物为：

```text
build-npu/python/kvaf/
build-npu/bin/kvaf_execute_batch_get_kernels.o
build-npu/bin/kvaf_execute_batch_put_kernels.o
```

Get 文件包含 URMA AIV 与 UBMEM Get 入口，Put 文件同理；本脚本使用标准合并产物，
不需要额外生成 UBMEM 专用 `.o`。版本入口为 AIV Get V3、Put V2 和
`KvafAicpuUmdkControlSessionsV3`。

## 3. 运行真机验收

先查看计划，无需安装 torch/kvaf 或访问设备：

```bash
python3 tests/python/test_cache_execution_e2e.py --list
bash tests/run_cache_execution_e2e.sh "$PWD/build-npu" --help
```

先验证 URMA 的基础交叉写读，替换 CPU IP：

```bash
export KVAF_CPU_SERVICE_ADDR="CPU_IP:28999"
export DEVICE_ID=0
bash tests/run_cache_execution_e2e.sh "$PWD/build-npu" \
  --modes urma --cases matrix \
  --report "$PWD/validation-urma.json"
```

再运行全部用例。建议在独立测试 CPU 服务上执行；失败时不要与其他业务混用该服务。
下例使用 Linux 的 `timeout` 给包括建链、kernel 等待和关闭在内的整个进程设置上限：

```bash
set -o pipefail
timeout --signal=TERM --kill-after=10s 1800s \
  bash tests/run_cache_execution_e2e.sh "$PWD/build-npu" \
  --peer "$KVAF_CPU_SERVICE_ADDR" --device "$DEVICE_ID" \
  --report "$PWD/validation-full.json" \
  2>&1 | tee "$PWD/validation-full.log"
```

移除 `timeout ...` 可在无该命令的环境运行；外部超时不会自动清理 CPU 远端资源。
脚本按所选案例失败即停止，不继续用可能处于未决状态的 Runtime 测试其他组合。

常用参数：

| 参数 | 默认值 / 作用 |
|---|---|
| `--modes` | `urma ubmem urma+ubmem`；可只指定一种或两种 |
| `--cases` | `matrix overlap lifetime resources graph lifecycle`；固定按此顺序执行所选项 |
| `--rows` | 32；同时作为各 provider 的请求容量预算 |
| `--token-bytes` / `--tokens-per-block` | 656 / 2；offset 按字节解释 |
| `--cores` / `--cpu-threads` | 2 / 8；CPU 线程数至少为核数 |
| `--graph-replays` | 3，至少 3；循环验证全量、部分和空请求前缀 |
| `--cycles` | 2，至少 2；关闭后重用同一个 Runtime 初始化 |
| `--npu-eids` | 逗号分隔的 32 位十六进制 EID；也可用 `KVAF_NPU_EIDS` |
| `--peer-ub-device` | 固定 CPU UB 设备；也可用 `KVAF_PEER_UB_DEVICE` |
| `--peer` / `--device` | 默认读取 `KVAF_CPU_SERVICE_ADDR` / `DEVICE_ID` |
| `--report` | `cache_execution_results.json`；逐项更新，已有同名文件会覆盖 |

可用 `PYTHON=/path/to/python` 选择解释器。脚本优先从指定 build 目录导入 kvaf，并检查
导入位置，防止误测环境中安装的旧包。缺构建产物、torch_npu、设备或接口都返回失败，
不以 `SKIP` 成功退出。此入口无需额外设置 `KVAF_RUN_NPU_TESTS`。

默认数据规模用于功能验收。增加 rows 或 tokens-per-block 会增加 CPU DDR、NPU Tensor
和 Python 字节数据占用；Gather/Scatter 还保留其内部固定容量工作区，不能只按 rows
估算总内存。本脚本不是带宽 benchmark。

### 3.1 HAL/SVM 跨池生命周期回归

现场驱动 `25.6.rc3.1` 曾出现：同一个 CPU 服务第一次 UBMEM matrix 成功，第二次失败；
UBMEM 全用例也会在 `lifecycle/0` 的重新初始化失败。驱动日志确认
`svm_char_dev_open` 的 Host SVM OPEN ioctl 返回 `EEXIST`，继而
`Svm master init failed`、`halMemCreate ret=38`。修复让 HAL 在进程内跨池复用，
避免池关闭时卸载 HAL；实际池内存、映射仍在关闭时释放。

更新代码后在 CPU 端运行 `./build.sh cpu`，停止旧测试服务并启动新二进制一次。
这一步清除旧进程遗留的 SVM 状态；后续验证全程使用这个服务，不在轮次间重启。
NPU 端沿用已构建的同分支接口即可，此修复没有改变协议或设备 kernel。

在 NPU 仓库目录执行（地址和设备号按实际环境修改）：

```bash
(
  set -e
  export KVAF_CPU_SERVICE_ADDR=127.0.0.1:38999
  export DEVICE_ID=6
  for run in 1 2 3; do
    timeout --signal=TERM --kill-after=10s 180s \
      bash tests/run_cache_execution_e2e.sh "$PWD/build-npu" \
      --modes ubmem --cases matrix --report "$PWD/ubmem-reopen-$run.json"
  done
  timeout --signal=TERM --kill-after=10s 600s \
    bash tests/run_cache_execution_e2e.sh "$PWD/build-npu" \
    --modes ubmem --cycles 5 --report "$PWD/ubmem-lifecycle-full.json"
  timeout --signal=TERM --kill-after=10s 1800s \
    bash tests/run_cache_execution_e2e.sh "$PWD/build-npu" \
    --report "$PWD/validation-full.json"
)
```

前三次覆盖不同客户端进程先后使用同一个 CPU 服务；第二条测试命令省略 `--cases`，
运行 UBMEM 全部六类案例并重建池 5 次；最后覆盖全部模式和案例。
任一步失败会停止。所有命令退出码为 0、五份报告均为 PASS，且期间 CPU 服务 PID
不变，才算该回归通过。重启服务后仅一次 matrix 成功不足以确认修复。

## 4. 各案例检查什么

| 案例 | 操作与通过标准 |
|---|---|
| `matrix` | 每个 Put 写非零且随行、字节、seed 变化的 payload；同表依次切换所有 Get，检查 0、半数、全量请求；每个输出字节匹配独立 CPU oracle |
| `overlap` | 每对 Put/Get 分别运行双 stream 和双 Host 线程；Get 读旧 key，Put 写不同新 key，随后验证两份 payload；Get 提交后不先调用 `drain_get` 再 Put |
| `lifetime` | Get 异步提交后释放公共 AccessTable handle；该 handle 随即不可查询，源 key 仍因 pin 返回 Busy；`drain_get` 后数据正确且 key 可释放 |
| `resources` | 对全部 provider 预热；过量 warmup、Get、Put 被拒绝；拒绝的 Put 未提交 key；重复 Put、未启用配置被拒绝；显式 `recover` 重建会话后原 DDR 数据仍可读 |
| `graph` | 两种可用 AIV Direct Get 分别 prepare/capture/replay；同一输出和请求 buffer 地址下更新 indices、offsets、miss_count、output_indices；graph 期间 eager Get、Get 预热/恢复、drain、shutdown 被拒绝，独立 Put 可成功；释放后普通 Get 可继续 |
| `lifecycle` | 每轮连续调用两次 shutdown，再初始化、预热各 provider、写入并读取数据；最终 shutdown 成功 |

请求使用非零 token 偏移和分散、反向的输出行号。有效请求前缀之外故意放入越界索引、
负偏移和负输出行号，确认后端只消费 `miss_count` 指定的前缀。输出先填充 `0xA5`，
同时检查被写区域及没有被请求的区域。两端不能用跳过 payload 的 done 探针替代这些测试。

每次读取完成都检查 `remote_region` 不变；公共表仍有效时检查 Host AccessTable 的
addresses、lengths、valid 不变。这样能发现切换注册或后端时更换池、清零数据等错误。

`overlap` 的 PASS 证明这些调用方式下结果正确，**不证明设备 kernel 确实同时执行**。
报告固定记录 `hardware_overlap_measured: false`。实际重叠需在较大工作量下用目标环境
profiler 检查读写 stream 时间线，且不能把 Host 调用耗时重叠当作 DMA 重叠。

`resources` 在真机上测试的是正常会话的显式关闭/重建，不人为破坏网络或设备。
超容量 Put 被拒绝后，对每个候选 key 单独调用 `prepare_access_table([key])`，
要求得到 `KeyError: 'LockKeys key was not found'`。不存在的 key 不会生成 `valid=0`
表项；也不能只检查整批查询失败，因为查询在首个缺失 key 处停止，可能掩盖后续 key
被部分提交的问题。
创建回复丢失、部分初始化、MR/unmap 清理失败、未决 Put 等故障由下节可控注入测试覆盖；
本脚本 PASS 不代表这些故障已在真实设备上注入。关闭成功也不能单独证明无驱动级资源泄漏，
长期循环还需观察服务进程、设备和驱动资源计数。

## 5. 判断结果与定位失败

每项打印：

```text
KVAF_EXECUTION_BEGIN urma+ubmem/matrix/urma/aiv->ubmem/aiv/direct
KVAF_EXECUTION_RESULT {"case": "...", "status": "PASS", "seconds": ...}
KVAF_EXECUTION_SUMMARY PASS report=/absolute/path/to/validation-full.json
```

通过条件同时满足：

1. 进程退出码为 0，最终 JSON `status` 为 `PASS`。
2. `options` / `plan` 包含期望的模式与案例；只选 `--cases matrix` 的 PASS 仅代表该子集。
3. 所有实际执行的 `results` 均为 `PASS`，包括各模式的最终 shutdown。

返回 1 表示依赖、操作、数据比较或清理失败；argparse 用法错误返回 2。
`timeout` 返回 124（强制 KILL 时也可能为 137）；被外部终止的 JSON 可能停在 `RUNNING`，
属于未完成，不能算通过。保留最后一个 BEGIN、JSON、NPU 完整 traceback 和同时间 CPU 日志。

优先按以下顺序排查：

- 环境阶段失败：指定 build 目录、解释器、torch_npu、设备号和新包是否一致。
- 构造时报 `enabled AIV Put provider was not built`：检查
  `build-npu/CMakeCache.txt` 中 `KVAF_ENABLE_AIV_BATCH_PUT:BOOL=ON`。早期版本即使
  此选项为 ON，也因 `kvaf_npu_runtime` 缺少同名编译宏而误报；更新本分支并重新
  执行 `./build.sh npu`。这是 NPU 本地能力检查，与 CPU 地址、端口无关。
- 初始化/首调失败：CPU 地址与版本、UB/EID 路由、Fabric 能力及 kernel 安装。
- `ubmem/initialize` 或 `ubmem/lifecycle` 报 `halMemCreate(Host P2P DDR) failed, ret=38`：
  错误发生在 CPU 服务调用 HAL 时，尚未导出 Fabric handle 或在 NPU 建立映射；
  也可能在 HAL 内部初始化失败，尚未真正分配物理内存。
  HAL 返回码不等同于 Linux `errno`，不能据此认定设备不支持 UBMEM 或内存泄漏。
  新版错误同时包含申请的 `bytes`、`cpu_service_pid` 和调用后立即保存的 `errno`；
  `errno=0` 只表示 HAL 未提供该信息。查看 CPU 服务用户的驱动 plog，结合 PID 和
  失败时间匹配记录，日志路径被重定向时使用实际目录：

  ```bash
  grep -rniE -C 3 'halMemCreate|Mem create failed|DRV_ERROR_FILE_OPS|Svm master init failed' \
    "$HOME/ascend/log" 2>/dev/null | tail -100
  ```

  若 `errno=17 (File exists)`，保留现有 CPU 服务进程，在另一终端附加 syscall
  跟踪。普通部署使用错误中的 `cpu_service_pid`；若服务在 Docker 中、strace 在宿主机，
  必须使用宿主机看到的该服务 PID，而非容器内 PID：

  ```bash
  strace -f -tt -yy -s 4096 \
    -e trace=openat,close,ioctl,write,writev,sendto,sendmsg \
    -p CPU_PID -o "$PWD/kvaf-cpu-hal-detail.log"
  ```

  在 NPU 端复现失败后，用 Ctrl+C 停止附加的 strace，再提取：

  ```bash
  grep -nE -C 5 'EEXIST|halMemCreate|Svm master init failed|Ioctl failed|Open failed' \
    kvaf-cpu-hal-detail.log | tail -180
  ```

  若 `/dev/davinci_manager` 的 ioctl 返回 EEXIST，并伴随 `svm_char_dev_open`、
  `svm_ioctl_dev_init`、`Svm master init failed`，与上述 Host SVM 重建失败一致，
  按 3.1 节更新、重启一次并复验。strace 可能把该 ioctl 编号显示成
  `ZFS_IOC_POOL_CREATE`，这是编号名称重叠，不能据此判定 ZFS 或普通文件冲突。
  单独的 `errno=17` 不足以确认根因，不删除未知 HAL/共享内存文件。
  跟踪需要目标机安装 strace 且允许附加；若权限不允许，可在独立测试服务停止后用
  同一 strace 前缀启动服务（把 `-p CPU_PID` 换成完整的 `kvaf_cpu_service` 启动命令）。

  若单独 `--modes urma` 和 `--modes ubmem` 通过，但默认顺序失败，分别记录
  同一 CPU 服务进程下 `--modes urma ubmem --cases matrix` 与
  `--modes ubmem urma --cases matrix` 的结果，并另测 `--modes urma+ubmem`。
  前两项检查跨 Runtime 的模式切换，最后一项检查同一 DDR 池上的混合访问；
  单模式通过不能代替这两类验证。更换顺序后通过也不能证明原失败已修复。
- 旧脚本在 `resources` 案例的 `absent = ...prepare_access_table(keys)` 处报
  `KeyError: 'LockKeys key was not found'`：这是脚本把正常的缺失 key 当成失败。
  更新本分支的测试脚本即可，无需重新构建 CPU/NPU；先用
  `--modes urma --cases resources` 复验，再跑完整用例。修正版会逐 key 验证该异常，
  若某个 key 意外可读，则报 `rejected Put committed key ...`，此时应排查 Put 提交路径。
- payload mismatch：报告给出第一个错误字节，结合 case 的 Put/Get 组合定位数据路径。
- Busy / shutdown 失败：先保留日志和状态；正常失败会尝试安全关闭，关闭未确认时不能声称资源已释放。
- 客户端被强制终止：确认没有其他客户端使用该测试 CPU 服务后，由操作者检查并重启测试服务，
  再运行新的验收进程；脚本不会自动杀远端服务。

## 6. 便携回归与故障注入

无需 NPU，在装有 CMake、C++17 编译器、Python 和 pybind11 的环境执行：

```bash
cmake -S . -B build-verify \
  -DBUILD_TESTING=ON -DKVAF_BUILD_PYTHON=ON \
  -DKVAF_BUILD_REMOTE_CPU_SERVICE=ON -DKVAF_CPU_ENABLE_URMA=OFF
cmake --build build-verify --parallel 8
ctest --test-dir build-verify --output-on-failure -j 4
```

若有匹配的真实或 stub 头文件，可在 configure 时增加：

```bash
-DKVAF_TEST_URMA_INCLUDE_DIR=/absolute/path/to/urma/headers
-DKVAF_TEST_ACL_INCLUDE_DIR=/absolute/path/to/include
```

ACL 目录下应存在 `acl/acl.h`。这两项只启用额外 mock 测试，不代表正式 SDK 或设备验收。
用 `ctest --test-dir build-verify -N` 查看当前构建实际启用的用例。它们包含验收脚本
自测和真实 Runtime CMake 编译参数检查；后者无需 CANN，覆盖完整构建及裁剪构建的能力宏。

重点测试可单独运行：

```bash
ctest --test-dir build-verify --output-on-failure \
  -R 'kvaf_(runtime_build_flags_python|cache_execution_harness_python|execution_.*_test|peer_execution_resources_test|runtime_put_recovery_test|cache_region_service_test|runtime_cleanup_test|ubmem_hal_.*_test|ubmem_mapping_cache_test|aicpu_session_registry_test|aicpu_put_resolution_test|.*_provider_resources_test|aiv_.*_transport_close_test)'
```

资源管理与在线 Put 恢复的回归：

- `kvaf_execution_resource_manager_test`：所有权在初始化前移交、租约独占、Put/Get
  首次初始化互不阻塞、容量拒绝、失败后保留 owner、状态解绑与销毁顺序、请求元数据副本；
  DescribeRequirements 在资源初始化之前执行，初始化和预热直接进入 owner，检查容量溢出。
- `kvaf_peer_execution_resources_test`：外层算子销毁后独立初始化、预热和关闭资源；
  本地关闭失败不关闭远端、远端关闭重试、未初始化/超容量请求的清理不创建 CPU session。
- `kvaf_runtime_put_recovery_test`：未知完成后成功提交或安全回滚、配置不匹配、元数据及
  完成确认失败重试、提交异常、绑定失败；断言恢复不重复提交 WRITE，也不重复 Commit/Abort。
- `kvaf_aiv_provider_resources_test`：使用实际 AIV Host provider 和替身设备接口检查
  执行状态绑定、原次 status 校验、迟到完成、临时注册回收及资源关闭失败重试；
  独立 owner 生命周期、准备后的具体视图、错配状态拒绝和 Launch 不创建注册/工作区。
- `kvaf_ubmem_provider_resources_test`：检查 stream/status 收集失败后保留执行状态、
  恢复不重发、完整完成数量校验，以及旧报告不能让新调用误判成功；独立资源初始化、
  Put/Get 具体视图、Launch 无新增设备分配、未绑定/stream 错配拒绝和释放失败重试。
- `kvaf_aicpu_put_resolution_test`：多个 descriptor 由一次 WRITE 提交；检查原次序号
  和设备结果、状态收集失败、WRITE 成功后注销失败的清理重试及结果保留；
  独立资源生命周期和准备阶段注册、具体 Put Launch 无注册/分配、准备输入和调用状态校验。
- `kvaf_aicpu_gather_resources_test`：真实 Gather Host 逻辑的 Prepare/ExecutePrepared
  分离，检查注册和 profiling 工作区只在准备阶段建立，未准备/错配/重复 Launch 被拒绝，
  连续请求复用 Gather 工作区及 CPU worker。
- `kvaf_aiv_get_transport_close_test` / `kvaf_aiv_put_transport_close_test`：逐阶段注入
  Jetty、CQ、MR、token、设备内存、context 释放失败，检查重试不重复释放已完成部分，
  且队列未关闭时不释放其依赖。

本次公共 View / URMA 绑定拆分新增或扩展：

- `kvaf_execution_view_binding_test`：旧 View 在准备/解绑/关闭/同址重建后失效；
  独立方向准备互不误伤；绑定重复导入复用、context 代次和远端注册身份隔离、
  部分导入失败及最后引用关闭失败保留、异常后的清理重试。
- `kvaf_cpu_ddr_registration_test`（需要 URMA 头文件）：CPU 注册标志、原始 token 与
  注册基址导出，注册部分失败、MR/token 释放失败和幂等重试。
- `kvaf_aicpu_session_registry_test`：同一 session 多次导入只发生一次，注册身份
  改变被拒绝，unimport 失败保留句柄；继续检查 canonical → 注册地址转换。
- AICPU Put/Gather 与 UBMEM provider 用例补充旧 View 的重准备、解绑失效检查。

可单独运行本轮相关用例：

```bash
ctest --test-dir build-verify --output-on-failure \
  -R 'kvaf_(execution_view_binding|cpu_ddr_registration|aicpu_session_registry|aicpu_put_resolution|aicpu_gather_resources|cpu_push_shutdown|aiv_.*_transport_close|.*_provider_resources)_test'
```

此次涉及 CPU 服务、NPU Host 和 AICPU kernel 内部结构，需重新构建两端并部署新的
AICPU 包；wire 协议未变。沿用第 3 节原全量命令即可，不设置 `--cases`，并在同一个
CPU 服务进程下重复两轮。原性能脚本的公开参数不变；性能结果应在功能全量通过后采集。

### 设备热路径优化的复验

本轮保持 canonical 地址契约和 ABI，修改 NPU Host、AICPU kernel 和 AIV kernel。
重新运行 `./build.sh npu` 及 `./build.sh npu --check-kernel`，使用新客户端进程。
CPU 端若已部署同分支此前的实现，可沿用该 CPU 服务；本轮没有改 CPU 协议或服务代码。
先运行第 3 节原全量验收（不设置 `--cases`），再运行原性能脚本。

复测此前下降明显的三个点：

```bash
KVAF_AICPU_TOKEN_BYTES=656 \
KVAF_AICPU_MISS_COUNTS="2048 4096 8192" \
KVAF_AICPU_WORKERS=5 \
bash tests/run_aicpu_gather_read_sweep.sh "$PWD/build-npu" \
  --peer 127.0.0.1:38999 --device-id 6 --warmup 10 --iterations 100
```

比较 main、`ef2aec43` 和优化后的分支时，保持构建选项、设备、CPU gather 配置、
table capacity、tokens-per-block、预热和迭代次数一致；若原命令另有参数，继续沿用。
分别使用各分支匹配的产物和 CPU 服务，不混用版本。建议各跑三轮，保存每轮 CSV。
优先比较 `device_p50_us`、`lookup_table_us`、`address_send_us`、`notify_cpu_us`
和 `build_wqe_us`；各阶段是 worker 最大耗时的统计量，不能简单相加当作 device P50。
Host 重复 D2H 的消除不计入 device 耗时。便携回归通过不能代替真机性能对比。

本轮相关便携测试：

- `kvaf_aicpu_gather_execution_test`：运行真实 Gather/transfer/session 实现，替身 URMA
  执行 WQE；检查每个 worker 一次 registry 查询、多次 ready/ACK 轮询、并发 session、
  数据 payload、空 worker、64 KiB 分片、4096 WR 跨轮次、传输错误及 session 读锁保护。
- `kvaf_aicpu_session_registry_test`：保留原注册/范围检查，并补充跨轮次的大 span 和
  后续 span 的地址、offset、尾部 CQE 标志。
- AIV Get/Put kernel model：新增 registered 基址低于 canonical 基址的 payload 检查。
- `kvaf_aicpu_gather_resources_test`：同步完成不重复 D2H；异步 Drain 和同步失败后的
  重试仍收集状态。`kvaf_cpu_push_request_shape_test` 只检查 main 原有的形状/计数校验，
  不向已恢复的直接查表路径注入非法索引。

```bash
ctest --test-dir build-verify --output-on-failure \
  -R 'kvaf_(aicpu_gather_execution|aicpu_gather_resources|aicpu_session_registry|batch_.*_kernel_model|cpu_push_request_shape)_test'
```

故障通过便携替身注入，不在真机上故意破坏正在执行的 DMA。真机应重新运行第 3 节
全量测试，覆盖 matrix、overlap、lifetime、resources、graph 和 lifecycle；保持同一
CPU 服务进程再跑一轮，确认资源跨调用复用及池重建正常。此前 HAL 修复的真机通过
结果以及此前版本的全量 PASS，不能代替本次资源/算子职责拆分后的验收。测试命令、
参数和 JSON 报告格式保持不变；重新编译后使用原全量命令即可。

`kvaf_ubmem_hal_execution_test`、`kvaf_ubmem_hal_legacy_test` 和
`kvaf_ubmem_hal_concurrent_test` 使用动态加载的替身 HAL，模拟用户态随卸载重置、
驱动进程状态继续存在的情况。覆盖连续释放/重建池、CPU 后端重建、新旧入口共享及
并发首次加载，并检查各池的 create/release、reserve/free、map/unmap 数量配对。
测试不通过链接替身库来保持它存活；旧实现会在第二次分配触发模拟的 `ret=38/EEXIST`。
这些是生命周期回归，真实驱动验证仍需执行 3.1 节。

TCP 恢复测试绑定本机回环地址，验证创建、控制、关闭回复丢失后的幂等重试。
`kvaf_cpu_push_scatter_test` 包含超时路径，约需 90 秒，不应把它的正常等待当成挂死。
没有安装设备依赖时，仅检查验收脚本本身可执行：

```bash
python3 tests/python/test_cache_execution_harness.py
bash -n tests/run_cache_execution_e2e.sh
```

这些检查通过只表示脚本和便携模型通过，不会生成真机 PASS。当前开发机没有 Ascend
设备，真机报告需要在目标 CPU/NPU 环境实际运行后取得。

## 7. 新旧执行路径的性能对照

性能脚本新增 `--execution-mode legacy|per-call`，默认仍为 `legacy`。
legacy 不向绑定传入新接口参数，可继续用于旧接口回归；per-call 显式开启 Dispatcher，
只启用本脚本实际使用的 Put/Get 配置，并在每次调用传入对应 execution。
请求数据、随机种子、表容量、worker/core 数和预热规则保持原脚本行为。

旧性能脚本通过只说明兼容路径没有明显回退；全量 E2E 通过说明新路径功能符合预期，
二者都不能代替 per-call 性能测量。本节测试不实施设计文档第 14 节的资源共享方案。

| 入口 | 两种模式的覆盖与计时 |
|---|---|
| `tests/run_aicpu_gather_read_sweep.sh` | Gather 设备分阶段计时及同步 Host 调用耗时 |
| `tests/run_aicpu_cpu_push_scatter_sweep.sh` | Scatter 设备分阶段计时及同步 Host 调用耗时 |
| `tests/run_remote_cpu_batch_get_perf.sh` | URMA AIV / UBMEM Get 的提交、完成等待和调用到完成耗时 |
| `tests/run_remote_cpu_runtime_perf.sh` | 固定配置的 Put/Get 及 Runtime 操作耗时 |
| `tests/python/test_npu_batch_put.py` | 同步 Put 调用耗时；在原构建和运行环境下追加模式参数 |
| `tests/run_remote_cpu_batch_get_miss_count_perf.sh` | 两种模式可使用相同 Host 口径扫描；默认 legacy 保留原设备周期扫描 |
| `tests/run_remote_cpu_batch_get_multicard_perf.sh` | 显式 graph 租约下的多卡重放吞吐；per-call 要求 `--graph-calls 1` |

### 7.1 AICPU Gather / Scatter

在 CPU 服务已启动且两端产物匹配的条件下，使用同一构建对比两个入口：

```bash
for perf_mode in legacy per-call; do
  KVAF_AICPU_TOKEN_BYTES=656 \
  KVAF_AICPU_MISS_COUNTS="2048 4096 8192" \
  KVAF_AICPU_WORKERS=5 \
  KVAF_AICPU_SWEEP_LOG_DIR="$PWD/reports/gather-$perf_mode" \
  bash tests/run_aicpu_gather_read_sweep.sh "$PWD/build-npu" \
    --execution-mode "$perf_mode" \
    --peer 127.0.0.1:38999 --device-id 6 --warmup 10 --iterations 100
done
```

每轮使用尚不存在的输出目录。Scatter 使用相同参数，将入口替换为
`tests/run_aicpu_cpu_push_scatter_sweep.sh`，并为其指定不同输出目录；其日志目录变量
同样为 `KVAF_AICPU_SWEEP_LOG_DIR`。若原命令限制了 `--table-capacity` 等负载参数，
两种模式均沿用相同值。建议交替运行几轮，不将单轮波动当成固定回退。

日志及 CSV 新增 `execution_mode`、`host_p50_us`、`host_p50_GBps`：

- `device_p50_us` / `p50_GBps` 保持原设备总耗时和带宽口径。
- `host_p50_us` 是同步 `batch_get` 调用到返回的中位数，包括 Host 调度、注册/资源
  准备、完成处理和 profile 收集；不包含请求生成、数据初始化及输出正确性比较。
- `host_p50_GBps` 按有效字节数除以 Host P50 计算，采用十进制 GB/s。
- 设备阶段仍为每轮 worker 最大耗时的均值，不能相加得到 Host 或设备 P50。

### 7.2 AIV / UBMEM 的 Host 性能

两个模式都可运行原单点脚本，只需追加 `--execution-mode`。例如：

```bash
for perf_mode in legacy per-call; do
  KVAF_CPU_SERVICE_ADDR=127.0.0.1:38999 DEVICE_ID=6 \
  bash tests/run_remote_cpu_batch_get_perf.sh "$PWD/build-npu" \
    --execution-mode "$perf_mode" --transport urma \
    --request-length 32768 --miss-count 4096 --table-capacity 4096 \
    --token-bytes 656 --tokens-per-block 128 \
    --aiv-cores 5 --jettys-per-core 2 --no-kernel-profile \
    --warmup 10 --iterations 100
done
```

测试 UBMEM 时将 `--transport urma` 替换为 `--transport ubmem`。结果行带
`execution_mode`；`operation=batch_get` 的 `p50_ms` 是调用到完成的 Host 耗时。
per-call 在外部 stream synchronize 后调用 `drain_get()`，其完成检查、注册释放及
租约回收计入当前样本，避免把这部分成本留到下一轮。`batch_get_synchronize` 包含
这段完成处理，不能将其解释成纯设备等待。

使用原 miss-count 扫描入口进行同口径对照：

```bash
for perf_mode in legacy per-call; do
  KVAF_CPU_SERVICE_ADDR=127.0.0.1:38999 DEVICE_ID=6 \
  KVAF_BATCH_GET_TOKEN_BYTES=656 \
  KVAF_BATCH_GET_MISS_COUNTS="2048 4096 8192" \
  KVAF_BATCH_GET_AIV_CORES=5 KVAF_BATCH_GET_JETTYS_PER_CORE=2 \
  KVAF_BATCH_GET_MISS_SWEEP_LOG_DIR="$PWD/reports/aiv-host-$perf_mode" \
  bash tests/run_remote_cpu_batch_get_miss_count_perf.sh "$PWD/build-npu" \
    --execution-mode "$perf_mode" --timing host --transport urma \
    --table-capacity 4096 --request-length 32768 --warmup 10 --iterations 100
done
```

Host 扫描 CSV 标记 `timing=host_call_to_completion`，输出 `host_*_us`、
`submit_*_us`、`completion_*_us` 和 `host_p50_GBps`。UBMEM 不扫描无效的 Jetty 数。
任一子进程失败或结果未标记为所选模式时，扫描失败，不生成 PASS。

当前 per-call AIV/UBMEM 不支持 `start_batch_get_kernel_profile`。脚本遇到
`--kernel-profile` 会明确报错，不退回 legacy。miss-count 扫描默认 legacy 使用
原 kernel cycle 指标，默认 per-call 使用 Host 指标；对比时两边显式设置
`--timing host`。`--timing kernel --execution-mode per-call` 不受支持。
Host 对照扫描采用固定路由，不执行 adaptive 校准；如需固定 CPU UB 设备，两边使用
相同 `--peer-ub-device`。单点 legacy 的原 adaptive 和 cycle profiling 能力保留。

### 7.3 多卡 graph 与便携检查

per-call graph 当前每个租约只允许捕获一次 Get，测试时两种模式均指定
`--graph-calls 1`，例如在原多卡命令后追加：

```text
--execution-mode per-call --graph-calls 1
```

脚本在 capture 前申请 graph 租约，在 synchronize、销毁 graph 后归还。
per-call 不支持默认的 256 次 Get 捕获，会在启动 worker 前提示修改参数。
graph 重放吞吐只覆盖已捕获设备任务，不包含每次 eager 调用的 Host Dispatcher 开销，
不能代替 7.2 节的调用到完成测量。

本次只调整性能脚本及说明，不要求重新编译 kernel 或服务；per-call 需要当前分支
匹配的绑定及 CPU 服务。便携检查覆盖模式及容量配置、计时口径、子进程参数传递、
扫描失败传播和 graph 生命周期，不会宣称真机性能通过：

```bash
PYTHONPATH="$PWD/tests/python" python3 -m unittest \
  test_perf_execution test_batch_get_inflight_config test_batch_get_multicard_perf
```
