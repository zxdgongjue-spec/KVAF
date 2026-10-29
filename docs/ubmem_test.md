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

NPU 节点使用匹配 Ascend950 的 CANN Toolkit / asc-devkit：

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
NPU halMemImportFromShareableHandleV2 → Device VA reserve/map
                                      │
          BatchPut: 本地 HBM → SIMT store → CPU DDR
          BatchGet: CPU DDR → SIMT load → 本地 HBM
```

- `access_operator/src/ubmem/hal.cc`：动态加载 HAL，管理不透明物理句柄、Fabric handle 与映射。
  Host 分配采用 `MEM_P2P_DDR_TYPE`、huge page；Host UVA 和 NPU Device-share VA 分别预留。
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

缓存维护仅针对 **AIV 标量 DCache**，直接调用 AscendC 的 `DataCacheCleanAndInvalid`：

| 位置 | 范围 | 生产者 → 消费者及理由 |
| --- | --- | --- |
| 内核入口、读取元数据前 | 本核整个标量 DCache，一次 | H2D 或前序 kernel → 当前标量读取。复用地址时不能读到此前缓存的请求数组、miss count 或地址表。元数据分散在多个数组，地址表还要间接索引，统一失效避免逐行刷新和额外遍历。 |
| 内核出口、写入报告后 | 本核独占的 64 字节状态行，一次 | 标量 status store → host `aclrtMemcpy`。将状态写回，使 host 能读取错误码和完成行数；每核按缓存行隔离，避免写回其他 core 的报告。 |

入口没有本 kernel 的待完成工作，因此不加 `PIPE_ALL`。出口的 `PIPE_ALL` 用于等待本核 VF
工作完成后再发布状态；ACL event/stream 用于 host 等待和后续同 stream 操作排序。
SIMT payload 使用普通 GM load/store，不对每条 payload 或每行数据添加 DCCI。
上述标量缓存操作不能用来证明远端 DDR 的写入完成或 CPU/NPU 的缓存一致性。

CPU oracle 只在 fill 完成后执行内存屏障，保证 DDR store 先于允许 NPU 开始 Get 的 TCP
响应；只读 check 无需出口屏障。这个屏障不清理 CPU cache。跨节点数据可见性依赖目标
HAL 映射和 UB 通路，需运行 CPU 改写后 NPU Get、NPU Put 后 CPU check 的硬件测试确认。
驱动的 `SVM_FLAG_BY_PASS_CACHE` 在 `malloc_mng.c` 中控制分配器缓存，不能作为硬件绕过
缓存的依据。

## 定位不支持的环节

测试会报告具体失败的 HAL 接口，并通过 TCP 将 CPU 分配错误返回 NPU 端，例如：

- `cannot load libascend_hal.so`：HAL 库未安装或不在加载路径。
- `halMemCreate(Host P2P DDR)`：当前驱动／节点不能分配此类 DDR，或资源不足。
- `halMemAddressReserve`：不支持当前 Host UVA／Device-share VA 预留方式，或地址资源不足。
- `halMemExportToShareableHandleV2(FABRIC)`：CPU 端 Fabric 导出失败。
- `halMemImportFromShareableHandleV2(FABRIC)` / `halMemMap`：NPU 侧导入、路由、权限或映射失败。
- `CPU_MISMATCH` / `KERNEL_ERROR`：分别为 CPU 内容校验失败、内核请求范围等校验失败。

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
