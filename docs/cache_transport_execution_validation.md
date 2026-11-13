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
`KvafAicpuUmdkControlSessionsV4`（新增按输出定向注销，旧 V3 包需重新安装）。

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

AIV Direct 的 `end_get_graph` 等待 stream 并释放注册资源，不读取 kernel status；
`discard_result` 对该 provider 仅作为通用接口参数保留。

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
  执行状态绑定、普通 Get 的诊断式 status 读取、迟到完成、临时注册回收及资源关闭失败重试；
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
| `tests/run_remote_cpu_batch_get_perf.sh` | URMA AIV / UBMEM Get 的异步提交、完成等待；`--get-timing sync-call` 测同步调用 |
| `tests/run_remote_cpu_runtime_perf.sh` | 固定配置的 Put/Get 及 Runtime 操作耗时 |
| `tests/python/test_npu_batch_put.py` | 同步 Put 调用耗时；在原构建和运行环境下追加模式参数 |
| `tests/run_remote_cpu_batch_get_miss_count_perf.sh` | 两种模式的 Host 扫描均测同步 `batch_get`；默认 legacy 保留原设备周期扫描 |
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
    --get-timing sync-call \
    --warmup 10 --iterations 100
done
```

测试 UBMEM 时将 `--transport urma` 替换为 `--transport ubmem`。结果行带
`execution_mode`；`--get-timing sync-call` 下，`operation=batch_get` 的 `p50_ms` 是一次
同步 API 调用的 Host 耗时，与 Gather/Scatter 的 `host_p50_us` 使用相同计时边界。
Gather/Scatter 的同步调用还会收集设备分段 profile；AIV/UBMEM Host 扫描禁用
kernel profile。因此应比较同一算法的 legacy 与 per-call，不能把不同算法的
Host 绝对值直接当作仅由算法本身造成的差异。
输出注册是否每轮释放由第 8 节的绑定模式决定。默认 `--get-timing async-drain`
保留原提交和完成等待分段；per-call 的 `drain_get()` 包含完成检查，因此该模式的
`batch_get_synchronize` 不能解释为纯设备等待，也不用于四种算法的 Host 对比。

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

Host 扫描 CSV 标记 `timing=synchronous_batch_get`，输出 `host_*_us` 和
`host_p50_GBps`。`submit_*_us` 和 `completion_*_us` 仅由异步分段模式输出，
不作为同步调用的阶段数据。UBMEM 不扫描无效的 Jetty 数。
任一子进程失败或结果未标记为所选模式时，扫描失败，不生成 PASS。

当前 per-call AIV/UBMEM 不支持 `start_batch_get_kernel_profile`。脚本遇到
`--kernel-profile` 会明确报错，不退回 legacy。miss-count 扫描默认 legacy 使用
原 kernel cycle 指标，默认 per-call 使用 Host 指标；对比时两边显式设置
`--timing host`。`--timing kernel --execution-mode per-call` 不受支持。
Host 对照扫描采用固定路由，不执行 adaptive 校准；如需固定 CPU UB 设备，两边使用
相同 `--peer-ub-device`。单点 legacy 的原 adaptive 和 cycle profiling 能力保留。

### 7.3 按次 Get 调度分段计时

诊断四种 eager Get 的 legacy 与 per-call。per-call 包括 Gather/AIV Direct 的
复用输出路径和 Scatter/UBMEM 的普通路径；legacy 记录同一 provider 的准备、
提交、完成及清理阶段，便于对照变化。设置 `KVAF_GET_DISPATCH_TRACE` 后，Runtime 将样本
缓存在内存中，在 `shutdown()` 时写成 CSV；未设置时不读这些分段时钟，也不写文件。
每次测试使用新的 CSV 路径，避免与旧版列格式混写。需重新构建 NPU 客户端；
AICPU Gather/Scatter 还需重新构建对应 Host 包，CPU 服务无须修改。
UBMEM 同步调用示例：

```bash
KVAF_GET_DISPATCH_TRACE="$PWD/ubmem-dispatch.csv" \
KVAF_CPU_SERVICE_ADDR=127.0.0.1:38999 DEVICE_ID=6 \
bash tests/run_remote_cpu_batch_get_perf.sh "$PWD/build-npu" \
  --execution-mode per-call --transport ubmem --get-timing sync-call \
  --request-length 32768 --miss-count 4096 --table-capacity 4096 \
  --token-bytes 656 --tokens-per-block 128 --aiv-cores 5 \
  --no-kernel-profile --warmup 10 --iterations 100
python3 tests/python/summarize_get_dispatch_trace.py ubmem-dispatch.csv --last 100
```

URMA AIV Direct 使用相同命令，改为 `--transport urma` 和独立的 trace 文件名。
对照 legacy 时改为 `--execution-mode legacy`，也使用独立的 trace 文件；
两种模式都保留 `--get-timing sync-call`，并分别运行摘要脚本。
Gather/Scatter 使用原单点负载；`PYTHONPATH` 指向同一次构建。Scatter 示例：

```bash
PYTHONPATH="$PWD/build-npu/python${PYTHONPATH:+:$PYTHONPATH}" \
KVAF_GET_DISPATCH_TRACE="$PWD/scatter-dispatch.csv" \
python3 access_operator/demo/aicpu_batch_get/python/benchmark_cpu_push_scatter.py \
  --execution-mode per-call --peer 127.0.0.1:38999 --device-id 6 \
  --workers 5 --token-bytes 656 --miss-count 8192 --warmup 10 --iterations 100
python3 tests/python/summarize_get_dispatch_trace.py scatter-dispatch.csv --last 100
```

Gather 将脚本改为 `benchmark_gather_read.py` 并使用独立 trace 文件名。
Gather/Scatter 的 legacy 对照同样只改 `--execution-mode legacy` 与 trace 文件名。
`--last` 对每个算法、执行路径、事件分别取最后 100 个样本，以排除正确性检查
和预热。输出的 `algorithm` 为 `gather`、`scatter`、`aiv_direct` 或 `ubmem`；
`path=output` 表示复用输出绑定，`path=ordinary` 表示每次构造 Get 上下文。
legacy 的标签为 `algorithm=legacy path=legacy`，由文件名区分所测算法。

AIV Direct 还会输出 `aiv_capture_prepare`、`aiv_capture_submit`、
`aiv_status_reset`、`aiv_kernel_launch` 和 `aiv_submit_other`（单位 µs）。
前两项分别是请求准备与提交阶段的 `aclmdlRICaptureGetInfo`；
`aiv_status_reset` 保留为兼容旧 trace 的列；AIV Direct 的普通 Get 和 graph
租约都不逐次清零状态，该字段应为 0。普通 Get 完成后的设备状态仅用于诊断；
graph 释放会等待 stream 完成，不读取设备状态，和旧图捕获路径一致。
`aiv_kernel_launch` 是 `KvafExecuteBatchGetLaunch` 的 Host 调用耗时，
`aiv_submit_other` 为提交阶段剩余的参数组装、校验和计时开销。
这些字段嵌套在 `provider_prepare` / `provider_submit` 内，不应再次加到总耗时。
legacy 同步路径没有状态清零和普通请求的 capture 查询，对应字段为 0。
其他 provider 的这些列为 0，摘要中不显示。旧版 CSV 不含这些列，仍可由摘要脚本读取。
两种模式须使用**新建且不同名**的 trace 文件；修改代码后需重新构建 NPU 客户端，
CPU 服务不需要重建。例如复测 656 字节、4096 miss、2 核、每核 2 Jetty：

```bash
mkdir -p reports
for perf_mode in legacy per-call; do
  trace="$PWD/reports/aiv-${perf_mode}-submit-$(date +%s).csv"
  KVAF_GET_DISPATCH_TRACE="$trace" \
  KVAF_CPU_SERVICE_ADDR=127.0.0.1:38999 DEVICE_ID=6 \
  bash tests/run_remote_cpu_batch_get_perf.sh "$PWD/build-npu" \
    --execution-mode "$perf_mode" --transport urma --get-timing sync-call \
    --request-length 32768 --miss-count 4096 --table-capacity 4096 \
    --token-bytes 656 --tokens-per-block 128 --aiv-cores 2 --jettys-per-core 2 \
    --no-kernel-profile --warmup 10 --iterations 100
  python3 tests/python/summarize_get_dispatch_trace.py "$trace" --last 100
done
```

请对照两种模式的 `aiv_capture_submit`、`aiv_status_reset`、
`aiv_kernel_launch` 和 `aiv_submit_other`，同时保留 `provider_submit` 作为总量检查。

提交样本包含锁等待、配置解析、输出绑定检查、前次完成处理、上下文构造或更新、
资源需求查询、租约获取、状态绑定，以及 provider 的需求查询与 Warmup。
`provider_prepare` 是各 provider 的请求准备；`provider_submit` 包含参数组装、
状态初始化和内核提交；`provider_completion` 包含同步等待、状态回拷与校验；
`provider_cleanup` 是 provider 内的调用后资源释放。`provider_other` 为 provider
总耗时中尚未归入这些阶段的部分；`outside_provider` 是 dispatcher 总耗时减去
provider 执行时间。各字段 P50 来自不同样本，不应相加。

同步调用的 `provider_completion` 包含真实设备等待，不等于新模式独有的纯 Host
开销。各 provider 的具体参数准备方式不同，应在相同算法、相同负载下比较两种模式的
provider 阶段，再看 per-call 的 dispatcher
阶段；必要时继续测设备时间。per-call trace 的 `total` 只覆盖 dispatcher，
legacy trace 的 `total` 覆盖 RuntimeCore 到 provider 返回，两者不能直接相减；
Python 绑定耗时均未覆盖。trace 会增加计时开销。
最终性能比较使用关闭 trace 后的 Host 扫描结果。

### 7.4 多卡 graph 与便携检查

per-call graph 当前每个租约只允许捕获一次 Get，测试时两种模式均指定
`--graph-calls 1`，例如在原多卡命令后追加：

```text
--execution-mode per-call --graph-calls 1
```

脚本在 capture 前申请 graph 租约，在 synchronize、销毁 graph 后归还。
per-call 不支持默认的 256 次 Get 捕获，会在启动 worker 前提示修改参数。
graph 重放吞吐只覆盖已捕获设备任务，不包含每次 eager 调用的 Host Dispatcher 开销，
不能代替 7.2 节的调用到完成测量。

第 8 节的输出复用需要重新构建 NPU 客户端和 AICPU 包；CPU 服务协议没有改变。
便携检查覆盖模式及容量配置、计时口径、子进程参数传递、
扫描失败传播和 graph 生命周期，不会宣称真机性能通过：

```bash
PYTHONPATH="$PWD/tests/python" python3 -m unittest \
  test_perf_execution test_batch_get_inflight_config test_batch_get_multicard_perf
```


## 8. Get 输出绑定与注册复用

本轮优化针对 per-call URMA AIV Direct 和 AICPU Gather 的反复注册/注销：

| 资源 | 当前生命周期 |
| --- | --- |
| Gather 内部 workspace 及其注册 | Get provider 生命周期；普通调用和绑定调用都复用，provider 关闭时注销后释放 |
| Gather 调用方输出注册 | 普通调用完成后定向注销；显式输出绑定跨调用保留，到解绑或 shutdown |
| AIV Direct 输出注册、token 和设备 entity 表 | 普通调用完成后释放；显式输出绑定跨调用保留，到解绑或 shutdown |
| 请求 Tensor、输入地址表、访问表/key pin | 仍按调用保留到完成；不会跟随输出绑定长期占用 |
| UBMEM / Scatter | 沿用已有资源生命周期，不使用这个输出绑定接口 |

地址转换契约和设备数据 kernel 保持现有版本。未删掉流同步、状态收集和错误判断。
AICPU 只新增控制操作 `kUnregisterBuffer`，用来单独释放输出注册；不再为一个普通
Get 的结束连带注销 workspace。对应入口升级到 `KvafAicpuUmdkControlSessionsV4`。

### 8.1 调用方式和约束

Runtime 必须按 `per_call_execution=True` 初始化，并启用对应 Get 配置：

```python
cfg = kvaf.GetExecutionConfig("urma", "aiv", "direct")
# Gather 改成 kvaf.GetExecutionConfig("urma", "aicpu", "gather")
lease = runtime.begin_get_output(cfg, output, request_capacity, stream)
try:
    for indices, offsets, miss_count, output_indices in requests:
        runtime.batch_get(device_table, indices, offsets, miss_count,
                          output_indices, output, stream=stream,
                          execution=cfg, output_lease=lease)
        runtime.drain_get()  # 完成检查；保留绑定的注册和 entity 表
finally:
    runtime.end_get_output(lease)  # 等待未完成 Get，再注销输出
```

- Python 在绑定期间持有输出 Tensor 和传入的 stream 对象。C++ 调用方自行持有底层分配。
  **绑定期间禁止 resize、set_、替换/释放底层 storage，也不能通过别名改变 storage**；
  保留 Python 引用不能防止调用方主动改变 storage。完成后可改写输出内容。
- 绑定固定配置、输出地址和字节数，以及最大请求容量（indices 长度，不是 miss_count）。
  请求、访问表和 active count 可以变化；可跨 stream 顺序提交，Get 仍按原规则串行完成。
  跨 stream 的输入/输出内容依赖由调用方建立。
- 一个 Runtime 同时只允许一个输出绑定；绑定期间占用 Get 方向，所有 Get 必须携带
  相同 lease。切换 Get 算法/输出、recover、额外 warmup 或 graph 前先解绑。
  Put 仍可独立执行。graph 使用原来的 graph lease。
- `drain_get()` 不解绑；`end_get_output()` 成功后 lease 失效。同一地址重新绑定会重新注册，
  不因地址相同就沿用旧分配的注册。`shutdown()` 会关闭绑定和对应资源。
- 等待/注销失败时保留原生资源及 Python 引用。等待失败可重试完成；无法恢复的错误
  通过 shutdown 关闭资源。只有成功解绑或成功 shutdown 后，才能释放/改动原输出分配。
- 不传 `output_lease` 的普通调用仍支持临时输出，完成后注销；不会按裸地址建立跨调用缓存。

### 8.2 性能脚本

第 7 节原命令继续可用。per-call 的 Gather、AIV Direct 及 runtime perf 脚本在预热前
绑定它们一直复用的输出，在测量结束后解绑；legacy、UBMEM、Scatter 和 graph 路径不新增绑定。
所有输出正确性校验仍然保留。

新增 `PERF_OUTPUT_BINDING` 日志，Gather/Scatter 和 Host sweep CSV 也包含：

- `output_binding`：`reused` / `transient` / `legacy` / `not-needed`。
- `binding_setup_us` / `binding_teardown_us`：一次性绑定及解绑耗时。

`reused` 的稳态 Host 延迟不含一次性绑定/解绑；每次调用的数据传输、完成检查仍计入。
需要计算完整任务成本时，再加上 setup/teardown。普通临时输出的对照可通过环境变量运行：

```bash
KVAF_PERF_REUSE_GET_OUTPUT=0 \
KVAF_CPU_SERVICE_ADDR=127.0.0.1:38999 DEVICE_ID=6 \
bash tests/run_remote_cpu_batch_get_perf.sh "$PWD/build-npu" \
  --execution-mode per-call --transport urma --no-kernel-profile \
  --request-length 32768 --miss-count 4096 --table-capacity 4096 \
  --token-bytes 656 --tokens-per-block 128 --aiv-cores 5 --jettys-per-core 2 \
  --warmup 10 --iterations 100
```

删除 `KVAF_PERF_REUSE_GET_OUTPUT=0` 即恢复默认复用。Gather sweep 也读取此变量；
对比两轮时使用不同的日志目录，其余参数保持一致。

### 8.3 构建和验收

```bash
./build.sh npu
./build.sh npu --check-kernel
KVAF_CPU_SERVICE_ADDR=127.0.0.1:38999 DEVICE_ID=6 \
bash tests/run_cache_execution_e2e.sh "$PWD/build-npu" \
  --report "$PWD/validation-output-reuse.json"
```

更新 NPU binding 和已安装 AICPU 包后使用新客户端进程，不能搭配旧 V3 控制包。
CPU 服务无需为本轮改动重新编译或重启。

原全量入口的 `resources` 组已包含 `output-binding`、`output-rebind`、`output-shutdown`：
变 active count/stream、多次复用、错输出/超容量/过期 lease 拒绝、访问表在完成后解锁、
Python Tensor 在绑定期间保活、解绑和 shutdown 后释放、重新绑定及普通调用回归。
可先单跑 `--modes urma --cases resources`，随后仍需执行全量验证。

便携故障测试检查实际注册次数、workspace 不被连带注销、同地址解绑后重新注册、
等待/状态读取/注销失败时资源保留及重试；它们不能替代真机正确性和性能测量。

## 9. Put 的 legacy / per-call 性能对比

使用独立入口 `tests/run_remote_cpu_batch_put_perf.sh`，底层仍运行
`tests/python/test_npu_batch_put.py`。支持以下三组配置；每组只改变
`--execution-mode`，其余参数保持一致：

| Put 路径 | 参数 | 示例核数 |
| --- | --- | --- |
| URMA AIV | `--transport urma --executor aiv` | 2 |
| UBMEM AIV | `--transport ubmem --executor aiv` | 16 |
| URMA AICPU | `--transport urma --executor aicpu` | 5 |

AICPU 使用已安装的配套设备包，允许仅包含 AICPU 的构建，不要求 AIV `.o`。
UBMEM 不支持 `--executor aicpu`。两种执行模式都使用当前分支的同一份构建；
这里比较接口与调度路径，不将旧 main 构建的结果混入对照。CPU 服务保持运行，
不需要在两个模式之间重启；每个命令会创建并关闭自己的 Runtime。

### 9.1 测量口径

`timing=synchronous_batch_put` 测量 Python 调用 `batch_put()` 到同步返回的耗时，
包含绑定层、元数据准备、实际写入、等待完成和提交结果，以及调用内发生的资源清理。
两种模式都不额外在每个样本后追加一次设备同步。初始化、输入构造、预热、正确性检查、
`batch_free()` 和最终 shutdown 位于计时区间外；每次重新 Put 之前先在区间外释放同一组 key。

输入 Tensor 在整轮测量中保持不变并重复使用。URMA AIV 和 URMA AICPU 的 per-call 默认在测量前调用
`begin_put_input()`，预检、预热和正式样本使用同一个 `input_lease`，测后调用
`end_put_input()`。输入注册、workspace 和执行上下文跨调用复用，绑定和解绑单独计时，
均不计入稳态 `batch_put()` 样本。UBMEM 没有本地输入 MR 注册，不需要这个输入绑定。

日志中 `input_binding` 为 `legacy`、`reused`、`not-needed` 或 `transient`。
需要对照每次调用注册/注销的普通路径时，设置 `KVAF_PERF_REUSE_PUT_INPUT=0`；
此时 URMA AIV/AICPU per-call 标记为 `transient`，调用内的注册/注销仍包含在样本耗时内。
两个模式的请求、源数据和设备完成语义保持一致，不应将单次绑定成本当成每次 Put 的成本。

legacy 的两条 URMA Put 都跨调用缓存输入注册，关闭 transport/session 时注销；
AICPU legacy 已恢复 main 的这一行为，不再在每次 `AcknowledgePut` 时注销输入。
因此不能把修复前 legacy 的逐次注册/注销结果当作现在的基线，应使用同一构建重新比较。
`binding_setup_us` 包含首次绑定可能触发的 provider、通信和 workspace 初始化，
不只是 MR 注册耗时；legacy 的此字段为零也不表示它没有初始化成本。

Put 的 `rows` 是写入 block 数，不能直接等同于 Get 的 `miss_count`：

```text
block_bytes = token_bytes * tokens_per_block
payload_bytes_per_call = rows * block_bytes
```

例如 `token_bytes=656`、`tokens_per_block=128` 时，每个 block 为 83,968 B；
512 个 block 的一次 Put 有效载荷为 42,991,616 B。对比模式时应同时固定
rows、block 大小、核数、Jetty 数、输入分配、设备和 CPU 服务参数。

### 9.2 单点及扫描命令

在仓库根目录执行。以下函数将两种模式的日志分别保存在 `reports/put`，
所有测试环境变量仅作用于对应子进程。示例按默认单层布局运行；若此前设置过
`KVAF_DSA_*`，先在新的终端运行，或使用相同的显式 DSA 布局完成两种模式的比较。

```bash
mkdir -p reports/put
set -o pipefail

compare_put() {
  local put_transport="$1" put_executor="$2" put_cores="$3" put_rows="$4"
  local perf_mode
  for perf_mode in legacy per-call; do
    env -u KVAF_NPU_BLOCK_BYTES \
      KVAF_CPU_SERVICE_ADDR=127.0.0.1:38999 DEVICE_ID=6 \
      KVAF_NPU_REQUESTS="$put_rows" \
      KVAF_NPU_TOKEN_BYTES=656 KVAF_NPU_TOKENS_PER_BLOCK=128 \
      KVAF_NPU_CORE_COUNT="$put_cores" KVAF_NPU_JETTYS_PER_CORE=2 \
      KVAF_NPU_WARMUP=10 KVAF_NPU_ITERATIONS=100 \
      bash tests/run_remote_cpu_batch_put_perf.sh "$PWD/build-npu" \
        --transport "$put_transport" --executor "$put_executor" \
        --execution-mode "$perf_mode" \
      | tee "reports/put/${put_transport}-${put_executor}-${put_rows}-${perf_mode}.log" \
      || return
  done
}

# 先固定一个规模，分别比较三种 Put。
compare_put urma aiv 2 512
compare_put ubmem aiv 16 512
compare_put urma aicpu 5 512

# 需要扫描时，使用同一函数；以下命令会覆盖同名的 512 行日志。
for put_rows in 128 512 2048; do
  compare_put urma aiv 2 "$put_rows" || break
  compare_put ubmem aiv 16 "$put_rows" || break
  compare_put urma aicpu 5 "$put_rows" || break
done
```

出现失败时先查看相应日志，不将缺失结果作为零耗时或测试通过。若出现小幅差异，
以反向顺序再跑一轮并使用另一个日志目录，确认结果是否稳定，避免把运行顺序或机器负载
变化当作执行模式差异。`KVAF_REMOTE_CPU_PERF_TIMEOUT` 可调整单进程超时秒数，默认 600。

### 9.3 结果与验证范围

收集每份日志的 `PERF_RESULT metric=runtime_batch_put_sync_call_to_return` 行。
结果同时标记 `operation=batch_put`、`execution_mode`、`transport`、`executor`、
`backend`、`timing`、`input_binding` 和 `payload_bytes_per_call`。
`binding_setup_us`、`binding_teardown_us` 是这一轮仅发生一次的绑定/解绑耗时，
也会分别输出 `PERF_INPUT_BINDING operation=begin/end`。优先比较同配置的 `p50_ms`，
同时观察 `p90_ms`、`p99_ms` 和均值；相对差异按
`(per_call_p50 / legacy_p50 - 1) * 100%` 计算。
`p50_GBps` 使用 p50 耗时计算；兼容字段 `useful_GBps` 使用平均耗时计算，
两者均为十进制 GB/s，不能直接当作同一个指标。

脚本的预检和测后校验检查提交后的 key、长度、有效位及槽位地址，当前属于
`kind=metadata` 校验，不宣称逐字节读取 DDR。数据正确性仍使用原全量入口验证：

```bash
KVAF_CPU_SERVICE_ADDR=127.0.0.1:38999 DEVICE_ID=6 \
bash tests/run_cache_execution_e2e.sh "$PWD/build-npu" \
  --report "$PWD/validation-put-perf.json"
```

`resources` 用例还覆盖 URMA AIV/AICPU 的显式 Put 输入绑定：同一输入存储改变内容、
keys 顺序及 source offsets 后多次 Put，并通过已启用的 Get 路径逐字节读回；
错输入、错 stream、超容量和过期 lease 必须在元数据提交前拒绝；
检查 Python Tensor 在绑定期间保活、解绑和 shutdown 后释放、重新绑定和普通 Put 回归。
可先单跑 `--modes urma --cases resources`，然后执行上面的全量验证。
本地无 NPU 时只能检查脚本和参数路径；是否存在性能劣化以同机实测结果为准。

### 9.4 应用中复用 Put 输入

当前显式输入绑定支持 `Urma + Aiv` 和 `Urma + Aicpu`，使用 per-call Runtime。绑定限定输入的起始地址、
总字节数、配置、stream 和最大请求数；同一次绑定中 keys、offsets、lengths 可以变化。
上次同步 Put 完成后可以修改输入内容，但不能 resize 或替换其底层存储（包括其他别名）。
每次调用仍进行事务准备、数据写入和完成处理，只复用输入注册及相关执行资源。

```python
cfg = kvaf.PutExecutionConfig("urma", "aiv")
# AICPU Put 使用 kvaf.PutExecutionConfig("urma", "aicpu")，其余调用相同。
lease = runtime.begin_put_input(cfg, source, max_requests, stream)
try:
    runtime.batch_put(keys, source, offsets, lengths,
                      stream=stream, execution=cfg, input_lease=lease)
    # 继续使用同一 source 和 stream，传入下一批 keys/offsets/lengths。
finally:
    runtime.end_put_input(lease)
```

绑定生效时，Put 必须携带匹配的 lease；需要切换输入、stream 或 Put 算法时，先解绑。
Python 绑定持有输入和 stream 的强引用，解绑成功才释放。若 Put 完成状态不确定，
先按事务恢复流程调用 `recover(cfg)`；`end_put_input()` 不会强行释放仍被事务使用的资源。
绑定或解绑失败时不能释放或更换输入存储，成功 shutdown 也可以完成清理。

### 9.5 三条 Put 路径的主机分段计时

三条 Put 路径及 legacy 对照都支持 `KVAF_PUT_DISPATCH_TRACE`。以下为 URMA AIV 示例；
UBMEM AIV 和 URMA AICPU 按 9.2 替换 transport、executor 和核数即可。

```bash
mkdir -p reports/put
for perf_mode in legacy per-call; do
  env -u KVAF_NPU_BLOCK_BYTES \
    KVAF_CPU_SERVICE_ADDR=127.0.0.1:38999 DEVICE_ID=6 \
    KVAF_NPU_REQUESTS=1024 KVAF_NPU_TOKEN_BYTES=656 KVAF_NPU_TOKENS_PER_BLOCK=128 \
    KVAF_NPU_CORE_COUNT=2 KVAF_NPU_JETTYS_PER_CORE=2 \
    KVAF_NPU_WARMUP=10 KVAF_NPU_ITERATIONS=100 \
    KVAF_PUT_DISPATCH_TRACE="$PWD/reports/put/urma-aiv-${perf_mode}-dispatch.csv" \
    bash tests/run_remote_cpu_batch_put_perf.sh "$PWD/build-npu" \
      --transport urma --executor aiv --execution-mode "$perf_mode" || break
  python3 tests/python/summarize_put_dispatch_trace.py \
    "reports/put/urma-aiv-${perf_mode}-dispatch.csv" --last 100 || break
done
```

汇总按 transport、executor、path、event 和 result 分组，每组取最后 100 次，分别显示
`p50/p90` 微秒数。`path=input` 是复用输入，`ordinary` 是未绑定的 per-call，
`legacy` 为原接口；`event=recover` 和失败记录单独展示，不混入成功 Put 样本。
`PERF_RESULT` 是 Python 同步端到端时间；CSV 是 C++ 内部分段，两者边界不同。

主要分段为：

| 字段 | 内容 |
| --- | --- |
| `lock`、`validation` | Runtime Put 锁及请求校验 |
| `prepare` | 执行资源和请求准备，可能含首次资源初始化 |
| `metadata_prepare` | 元数据事务准备 |
| `execute` | 执行调度及 provider 调用 |
| `metadata_finalize`、`acknowledge` | 提交/中止元数据事务及完成确认 |
| `dispatch_lock`、`dispatch_resolve` | dispatcher 锁及执行配置解析 |
| `context`、`describe`、`acquire`、`bind` | 上下文准备、需求描述、资源获取和视图绑定 |
| `provider_prepare` | 算子准备，包括必要的输入/通信资源准备 |
| `provider_registration` | prepare 中发生的注册子段，可检查稳态是否仍重复注册 |
| `provider_metadata` | 本次请求的设备元数据构造/复制 |
| `provider_submit`、`provider_completion` | 提交计算及等待完成/结果读取 |
| `provider_cleanup` | 本次调用内的资源清理 |
| `provider_other` | provider 总时间扣除一级子段后的剩余部分 |
| `outside_provider` | Runtime 总时间减去 provider 时间 |

层级包含关系不能直接相加：例如 `provider_registration` 已包含在 `provider_prepare`，
dispatcher 子段也可能出现在 Runtime 的 `prepare/execute` 内。各段 p50 来自各自分布，
其和不保证等于 total p50。绑定/解绑成本查看 `PERF_INPUT_BINDING`；稳态 `path=input`
应避免反复注册输入，但仍需保留完成和事务检查。最终吞吐对比关闭 trace，再运行 9.2，
避免将计时采样开销混入结论；CSV 在 shutdown/析构时落盘，不在每次 Put 内写文件。

Put 调度层在同步 Launch 期间借用本次请求及目标地址描述，不再把 keys、offsets、lengths
和目标地址 vector 完整复制到执行上下文；恢复所需的资源租约和完成状态仍保持到事务结束。
`context` 是整个上下文准备阶段，不能直接解释为纯 memcpy 时间。设备描述符的构造及上传
仍位于 `provider_metadata`，legacy 和 per-call 都保留这部分算法所需的工作。

复验时，两条 URMA per-call 的默认结果应显示 `input_binding=reused` 和 `path=input`，
稳态不再反复执行 MR 注册及注销；AICPU legacy 也不应每次出现约数百微秒的注册和
`acknowledge` 注销成本。对照 `KVAF_PERF_REUSE_PUT_INPUT=0` 的普通调用，确认复用来自
显式绑定，并未取消临时输入的生命周期清理。调用成功、吞吐及错误恢复仍需真机验证。
