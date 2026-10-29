# 用 HAL/Fabric 测试 CPU DDR 与 NPU 之间的 BatchPut / BatchGet

在 CPU 节点启动 `kvaf_ubmem_cpu_service`，在 Ascend950 节点运行 `kvaf_ubmem_test`。
测试程序调用 KVAF 已有的 `RuntimeCore::BatchPut/BatchGet`，验证 NPU 能否通过映射地址直接读写 CPU DDR。
CPU 端使用 Ascend HAL/Fabric，应用代码不调用 OBMM、URMA 或 HCCP。

该实验入口是 C++ 程序；现有 Python `NpuRuntime` 的 URMA 数据面继续使用原有入口。

## 编译、运行

CPU 节点编译不需要 CANN Toolkit 或 UMDK 头文件：

```bash
./build.sh ubmem-cpu
./build-ubmem-cpu/bin/kvaf_ubmem_cpu_service --listen 0.0.0.0 --port 29000
```

运行时需要匹配硬件的 `libascend_hal.so` 及 SVM/UBmem 驱动能力。库不在默认搜索路径时，使用
`--hal-library /actual/path/libascend_hal.so`。可用 `--numa-node 0` 指定 Host DDR 的 NUMA 节点；
不指定时使用 HAL 的 Host 分配策略。服务在收到分配请求时才实际测试 DDR 分配和导出能力。

NPU 节点沿用现有 KVAF AIV 内核使用的 Ascend950 CANN 工具链：

```bash
source /actual/path/to/cann/set_env.sh
./build.sh ubmem-npu
./tests/run_ubmem_e2e.sh "$PWD/build-ubmem-npu" CPU_IP:29000 --device 0
```

也可以直接运行：

```bash
./build-ubmem-npu/bin/kvaf_ubmem_test \
  --peer CPU_IP:29000 \
  --kernel "$PWD/build-ubmem-npu/bin/kvaf_ubmem_kernels.o" \
  --device 0 --cores 2 --rows 17 \
  --token-bytes 656 --tokens-per-block 128 --repeat 2
```

默认每个 KV slot 为 `656 * 128 = 83968` 字节，使用 17 个 slot。HAL 物理分配大小向上取整到
2 MiB；MetaManager 和内核只访问请求的逻辑 region 范围。`--rows` 至少为 3，`--cores` 为 1–32。

## 测试程序实际验证什么

每轮先执行本地 HBM 拷贝对照，复用同一 Put kernel、源数据、偏移、长度和核数，
仅把目的区换成本地 HBM，并逐字节校验。通过后打印 `LOCAL_COPY_VERIFIED`，
随后打印 `MAPPED_PUT_BEGIN` 并开始真实 UBmem 测试。对照失败会停止测试。

每轮测试依次执行：

1. 通过现有 MetaManager 为一批倒序 Key 分配 slot，调用 BatchPut 写入完整 KV block。
   输入采用非对齐起点及块间空隙，避免仅覆盖连续、对齐的源数据。
2. CPU 服务逐字节检查自己的 DDR。期望模式混合 slot 地址的高低位，能识别错 slot、错偏移。
3. 准备现有格式的 AccessTable，随后由 CPU 服务改写 DDR，使用不同的测试模式。
4. BatchGet 读取不同 token 偏移，并按 `output_indices` 写到重排后的 HBM 行。
   请求数组尾部保留无效值，`miss_count` 只覆盖有效前缀。
5. 在同一调用方 stream 上排入 D2H 校验，检查 CPU 改写后的数据以及输出末尾 guard；另测零 miss。
6. 释放 AccessTable 和 Key，再用同一数据面测试长度为 1、655、4097 字节的变长 BatchPut
   （长度超过 slot 时截到 slot 大小），再次由 CPU 检查内容。

`--repeat` 默认 2，复用同一映射，检测跨轮陈旧读。最终成功输出：

```text
[KVAF_UBMEM] BATCH_PUT_CPU_VERIFIED
[KVAF_UBMEM] BATCH_GET_CPU_SOURCE_VERIFIED (...)
[KVAF_UBMEM] VARIABLE_BATCH_PUT_CPU_VERIFIED
[SUCCESS] UBmem BatchPut/BatchGet passed with independent CPU DDR verification
```

CPU 的 fill/check 是测试 oracle，TCP 只携带范围、seed 和校验结果；KV payload 由 UBmem 数据面传输。

## 数据路径与源码

```text
CPU Host P2P DDR
  halMemCreate → Host UVA reserve/map → export FABRIC handle
                                      │
                         TCP PeerLink 交换描述符
                                      │
NPU HAL import → aclrtReserveMemAddress → halMemMap
                                      │
          BatchPut: 本地 HBM → SIMT store → CPU DDR
          BatchGet: CPU DDR → SIMT load → 本地 HBM
```

- `access_operator/src/ubmem/hal.cc`：动态加载 HAL，管理不透明物理句柄、Fabric handle 与映射。
  Host 分配采用 `MEM_P2P_DDR_TYPE`、huge page 和 HAL Host UVA。NPU VA 使用
  `aclrtReserveMemAddress(..., nullptr, 1)`，释放时配对调用 `aclrtReleaseMemAddress`。
- `runtime/src/ubmem_cpu_runtime.cc`：使用已有 RuntimeService 接收控制请求，拥有 CPU DDR 生命周期。
- `runtime/src/peer_link.cc`：增加 UBmem 分配、CPU oracle 请求；沿用原有 region release。
- `access_operator/src/ubmem/access_operator.cc`：实现现有 AccessOperator 接口，保持 BatchPut 同步提交、
  BatchGet 按调用方 stream 入队。每个 stream 独立持有状态和完成事件。
- `access_operator/src/ubmem/batch_kernel.asc`：对已映射的内存执行 SIMT load/store；逐行解析、搬运，
  检查设备端 Get 元数据中的有效请求前缀、表索引、远端 block 范围和输出范围。
- `tests/ubmem/npu_test.cc`：以真实 RuntimeCore、MetaManager 运行上述测试。

NPU 端先将 ACL 用户 device ID 转为 HAL logical device ID，支持可见设备编号重映射。
内核使用 `mapped_base + (remote_address - remote_region_base)` 转换地址，无需对称地址空间。
CPU region 在 NPU 映射持有期间不能释放；解除映射前先等待各 stream 的完成事件。

## 校验与同步的职责

使用 `RuntimeCore::BatchGet/BatchPut` 作为入口。Runtime 负责 buffer 格式、布局及 Put 源范围，
MetaManager 负责按请求顺序分配 Put 目的范围；UBmem adapter 只增加 NPU 类型、kernel ABI
位宽和 stream/state 检查。内核不重复检查 host 构造的参数；Get 的 `miss_count`、索引、
偏移和地址表内容在设备上产生，仍由内核检查，避免越界读写。

每个 core 遇到无效 Get 行时停止自己的循环并报告行号；此前完成的行和其他 core 的工作
不会回滚。异步 Get 只返回入队结果；需要读取内核错误报告时使用 `synchronize=true`。

同步执行沿用现有 `KvafExecuteBatchPutSync` 的顺序：在调用方 stream 上初始化设备状态、
启动 kernel、将状态 D2H 到固定页 host buffer，最后同步 stream。异步 Get 仍用完成事件
跟踪映射生命周期。UBmem 不额外添加入口或
出口 DCCI。每次 VF 调用前使用 `S→V` set/wait，调用后使用 `V→S` set/wait，再处理下一行。
这沿用 KVAF `RunBatchGetValidation` 和 SHMEM `simt_rma_perftest` 的 `VSSync<Simt>`，
交接标量准备参数、向量消费参数及标量复用调用状态的顺序；它不是远端完成通知。

CPU oracle 只在 fill 完成后执行内存屏障，保证 DDR store 先于允许 NPU 开始 Get 的 TCP
响应；只读 check 无需出口屏障。这个屏障不清理 CPU cache。跨节点数据可见性依赖目标
HAL 映射和 UB 通路，需运行 CPU 改写后 NPU Get、NPU Put 后 CPU check 的硬件测试确认。
驱动的 `SVM_FLAG_BY_PASS_CACHE` 在 `malloc_mng.c` 中控制分配器缓存，不能作为硬件绕过
缓存的依据。

## 实现依据

- 构建：`access_operator/cmake/aiv_toolchain.cmake` 从已有 BatchGet/BatchPut 构建配置提取，
  三个内核共享编译器、头文件查找和编译参数。
- SIMT 搬运：SHMEM `src/device_simt/gm2gm/engine/shmem_device_simt_mte.hpp` 的 GM load/store 循环。
- HAL ABI：CANN Driver `pkg_inc/ascend_hal_base.h` 的 `drv_mem_prop` 和函数声明，
  `ascend_hal_define.h` 的内存类型、Host UVA 标志和 Fabric handle。
- NPU VA：SHMEM `hybm_vmm_based_segment.cpp::ReserveEachPeMemorySpace` 和
  `hybm_acl_device_mem_segment.cpp::ReserveEachPeMemorySpace` 的 Ascend950 分支均用
  `aclrtReserveMemAddress(..., nullptr, 1)`。`dl_acl_api.cpp::AclrtReserveMemAddress` 明确要求
  nullptr 自动选址使用 ACL。KVAF 配对使用 ACL 预留/释放，HAL 导入/映射沿用前者的 `Map` 路径。
- Fabric 导出及共享属性：SHMEM `src/host/mem/heap/hybm_vmm_based_segment.cpp::ExportInner`。

SHMEM 的 `rma_d2h_demo` 是标量 p/g，`mte_perftest --memory-type dram` 使用 MTE/UB 搬运。
本分支的 SIMT GM 拷贝并不是这两个 Host DDR 示例的同一数据面；它借鉴的是 SHMEM 的
SIMT GM 循环，Host DDR 可达性必须由实际 Put/Get 校验确认。

独立 CPU 节点使用这组 HAL 接口导出 DDR，是本测试要验证的能力。SHMEM 的 Host DDR
示例运行在 Ascend runtime 环境，不能据此认定独立 CPU 节点已经支持。

## 定位不支持的环节

测试会报告具体失败的 HAL 接口，并通过 TCP 将 CPU 分配错误返回 NPU 端，例如：

- `cannot load libascend_hal.so`：HAL 库未安装或不在加载路径。
- `halMemCreate(Host P2P DDR)`：当前驱动／节点不能分配此类 DDR，或资源不足。
- `reserve virtual address`：CPU 的 HAL Host UVA 或 NPU 的 ACL 设备 VA 预留失败。
- `halMemExportToShareableHandleV2(FABRIC)`：CPU 端 Fabric 导出失败。
- `halMemImportFromShareableHandleV2(FABRIC)` / `halMemMap`：NPU 侧导入、路由、权限或映射失败。
- `CPU_MISMATCH` / `KERNEL_ERROR`：分别为 CPU 内容校验失败、内核请求范围等校验失败。
- `aclrtSynchronizeStream` 返回 `507035`：CANN 定义为 Vector Core 执行异常。
  程序立即输出 `aclGetRecentErrMsg()` 的详细信息；结合运行时日志中的 PC、访问地址和
  vec/mte error info 定位，单凭该错误码不能确定具体原因。

当前设备日志中的 334 表示 SIMT 非法 GM 访问或跨设备访问超时。若错误出现在
`LOCAL_COPY_BEGIN` 后，先排查本地 SIMT 搬运及参数；若本地校验通过、错误出现在
`MAPPED_PUT_BEGIN` 后，重点排查映射目的地址及其可达性。
`DEVICE_MAPPING_READY` 和 `FAILED_LAUNCH` 输出设备编号、映射与输入范围，可与故障地址对照。

同步 BatchPut 输出每核 `PUT_CORE`，包含设备报告的 `completed` 和 host 计算的 `expected`。
默认 17 行、2 核应分别完成 9 行和 8 行；计数不符会在 MetaManager 提交前报错。
状态在每次同步启动前填为 `0xFF`，用于识别本次 kernel 是否写入报告；`seen_capacity`、
`seen_blocks`、`seen_op` 是内核实际收到的启动字段，默认 Put 应为 `17`、`2`、`1`。
host 和 kernel 使用 `KvafUbmemBatchKernelV2` 符号，旧 kernel 文件会在加载时被拒绝。
`CPU_MISMATCH` 同时输出 `expected` / `actual` 字节。默认倒序 Put 首先检查 slot 16，
起点为 `16 × 656 × 128 = 1343488`；该处失败并不表示前面的 slot 已经通过校验。

两端必须具有可达的 UB 内存语义通路；仅 TCP 可达不能建立 UBmem 映射。该测试用于确定当前
CPU-only 节点是否具备这条 HAL/Fabric 能力；不支持时报告失败，不切换为其他传输方式。

## 无硬件的软件检查

```bash
cmake -S . -B build-ubmem-host -DKVAF_BUILD_UBMEM_TEST=ON -DBUILD_TESTING=ON
cmake --build build-ubmem-host --parallel
ctest --test-dir build-ubmem-host --output-on-failure
```

`kvaf_ubmem_control_test` 使用仅测试链接的 fake HAL，共享文件映射模拟同一物理内存的两个 VA，
覆盖故障回滚、真实 loopback TCP 描述符交换、CPU oracle、逻辑长度和映射引用保护。
它需要允许在 loopback 临时监听端口。

`kvaf_ubmem_kernel_test` 包含实际 `.asc` 内核源码，在 CPU 模型中覆盖非对齐、656 字节读、
变长写、无效 offset/地址、零 miss 和输出边界。该模型不模拟硬件缓存、SIMT 时序或 UB 总线。
NPU 编译及跨机映射、数据可见性必须使用上面的硬件程序验证。

参考：[SHMEM PR !285](https://gitcode.com/cann/shmem/pull/285)、
[CANN Driver SVM](https://gitcode.com/cann/driver/tree/master/src/ascend_hal/svm)。
