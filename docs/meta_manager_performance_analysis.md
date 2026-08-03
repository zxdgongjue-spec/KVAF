# MetaManager 性能分析

## 1. 目标

本文档整理 `MetaManager` 地址管理模块当前版本的性能测试方法、测试场景和实测结果，重点回答三个问题：

- 当前各接口的耗时大概在什么量级
- `resident_keys` 和 `batch_size` 对耗时有什么影响
- 一批请求走完整条控制面链路时，端到端耗时大概是多少

相关代码：

- 接口定义：[meta_manager.h](../meta_manager/include/kvaf/meta_manager/meta_manager.h)
- benchmark：[in_memory_meta_manager_perf.cc](../meta_manager/tests/in_memory_meta_manager_perf.cc)
- 模块说明：[README.md](../meta_manager/README.md)

## 2. 测试环境

- 日期：2026-08-03
- 平台：Ubuntu on WSL
- 编译器：GCC 13.3.0
- 构建类型：`Release`
- 日志开关：关闭

构建与执行命令：

```sh
cd /mnt/c/Urma/KVAF
cmake -S . -B build-wsl-release -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-wsl-release -j
ctest --test-dir build-wsl-release --output-on-failure
./build-wsl-release/meta_manager/kvaf_meta_manager_perf
```

如需同时观察接口级计时日志：

```sh
./build-wsl-release/meta_manager/kvaf_meta_manager_perf --enable-timing-log
```

## 3. 指标定义

benchmark 输出的关键字段如下：

- `call_count`：当前场景下实际测量的调用次数，通常等于 `resident_keys / batch_size`
- `avg_ns`：单次接口调用平均耗时
- `avg_per_key_ns`：把 `avg_ns` 均摊到单个 key 后的耗时
- `total_ms`：当前方法在当前场景下，所有测量调用累计总耗时
- `p50_ns`：中位数延迟
- `p95_ns`：95 分位延迟
- `max_ns`：最大值

建议观察时同时看两类指标：

- 看单次调用成本：关注 `avg_ns`
- 看 batch 摊薄效果：关注 `avg_per_key_ns`

## 4. 测试场景

### 4.1 二维场景矩阵

本次测试按下面两个维度组合：

- `resident_keys = 4096 / 8192 / 16384`
- `batch_size = 1 / 8 / 64`

其中：

- `resident_keys` 表示测量开始前，`MetaManager` 中已经存在的已提交 key 数量
- `batch_size` 表示单次接口调用处理的 key 数量

当前 benchmark 不再为了凑统计样本而重复多轮。  
除 `RegisterRemoteRegion` 外，绝大多数场景都满足：

- `call_count = resident_keys / batch_size`

### 4.2 各接口场景说明

- `PreparePut(short)`：先构造 `resident_keys` 条已提交短记录，再对一批新的短 payload 做地址预留
- `PreparePut(long)`：与上面相同，但 payload 进入 long slot
- `CommitPut`：在已有 resident 集合上持续提交新的 batch，模拟持续写入发布
- `AbortPut`：在 steady-state 集合上重复执行 `prepare + abort`
- `LockKeys / BatchLookup / UnlockKeys`：对 resident 集合按 `batch_size` 分批做稳定读
- `BatchFree`：把 resident 集合按 `batch_size` 逐批回收
- `EndToEnd(short)`：完整链路
  `PreparePut -> CommitPut -> LockKeys -> BatchLookup -> UnlockKeys -> BatchFree`
- `EndToEnd(long)`：与上面相同，但写入长度落到 long slot

## 5. 总体结论

### 5.1 单 key 场景

当 `batch_size = 1` 时：

- `PreparePut(short)` 大致在 `332.99 ns ~ 719.71 ns`
- `PreparePut(long)` 大致在 `407.95 ns ~ 940.35 ns`
- `BatchLookup` 大致在 `99.79 ns ~ 128.86 ns`
- `EndToEnd(short)` 大致在 `1241.92 ns ~ 1765.40 ns`
- `EndToEnd(long)` 大致在 `967.71 ns ~ 1944.63 ns`

结论：

- 读路径里最轻的是 `BatchLookup`
- 写路径里 `PreparePut(long)` 比 `PreparePut(short)` 更重
- 完整控制面链路在单 key 下大致是 `1 us ~ 2 us` 量级

### 5.2 batch 场景

当 `batch_size = 64` 时：

- 单次调用耗时明显变大
- 但 `avg_per_key_ns` 往往下降，说明固定函数调用和状态切换成本被 batch 摊薄了

例如在 `resident_keys = 16384, batch_size = 64` 时：

- `PreparePut(short)`：`30771.98 ns/call`，均摊 `480.81 ns/key`
- `PreparePut(long)`：`43915.32 ns/call`，均摊 `686.18 ns/key`
- `BatchLookup`：`3935.50 ns/call`，均摊 `61.49 ns/key`
- `EndToEnd(short)`：`126428.14 ns/call`，均摊 `1975.44 ns/key`
- `EndToEnd(long)`：`187989.22 ns/call`，均摊 `2937.33 ns/key`

结论：

- 如果上层允许批处理，batch 对单 key 平均成本通常有收益
- `BatchLookup` 的摊薄效果最明显
- `EndToEnd(long)` 是当前最重的完整链路

## 6. 代表性结果

### 6.1 小 batch 边界：`resident_keys = 4096, batch_size = 1`

| 方法 | call_count | total_ms | avg_ns | avg_per_key_ns | p50_ns | p95_ns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `PreparePut(short)` | 4096 | 1.98 | 482.92 | 482.92 | 391.00 | 826.00 |
| `PreparePut(long)` | 4096 | 2.12 | 517.96 | 517.96 | 404.00 | 958.00 |
| `CommitPut` | 4096 | 0.94 | 229.55 | 229.55 | 154.00 | 265.00 |
| `AbortPut` | 4096 | 0.35 | 84.97 | 84.97 | 83.00 | 86.00 |
| `LockKeys` | 4096 | 0.87 | 211.91 | 211.91 | 180.00 | 238.00 |
| `BatchLookup` | 4096 | 0.42 | 103.61 | 103.61 | 96.00 | 120.00 |
| `UnlockKeys` | 4096 | 0.49 | 119.39 | 119.39 | 110.00 | 119.00 |
| `BatchFree` | 4096 | 0.82 | 200.40 | 200.40 | 182.00 | 245.00 |
| `EndToEnd(short)` | 4096 | 5.27 | 1287.49 | 1287.49 | 1169.00 | 1783.00 |
| `EndToEnd(long)` | 4096 | 3.96 | 967.71 | 967.71 | 912.00 | 1133.00 |

观察：

- 单 key 场景下，`EndToEnd(short)` 约 `1.29 us`
- 单 key 场景下，`EndToEnd(long)` 约 `0.97 us`
- `BatchLookup` 和 `UnlockKeys` 较轻

### 6.2 大 batch 边界：`resident_keys = 16384, batch_size = 64`

| 方法 | call_count | total_ms | avg_ns | avg_per_key_ns | p50_ns | p95_ns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `PreparePut(short)` | 256 | 7.88 | 30771.98 | 480.81 | 25584.00 | 50635.00 |
| `PreparePut(long)` | 256 | 11.24 | 43915.32 | 686.18 | 34805.00 | 90444.00 |
| `CommitPut` | 256 | 5.65 | 22089.10 | 345.14 | 11560.00 | 17447.00 |
| `AbortPut` | 256 | 1.26 | 4919.40 | 76.87 | 4520.00 | 6561.00 |
| `LockKeys` | 256 | 7.32 | 28596.88 | 446.83 | 22601.00 | 68105.00 |
| `BatchLookup` | 256 | 1.01 | 3935.50 | 61.49 | 3875.00 | 4851.00 |
| `UnlockKeys` | 256 | 1.27 | 4954.62 | 77.42 | 4725.00 | 6385.00 |
| `BatchFree` | 256 | 6.08 | 23758.08 | 371.22 | 22300.00 | 30604.00 |
| `EndToEnd(short)` | 256 | 32.37 | 126428.14 | 1975.44 | 103477.00 | 228539.00 |
| `EndToEnd(long)` | 256 | 48.13 | 187989.22 | 2937.33 | 106095.00 | 276077.00 |

观察：

- 单次调用成本明显上升，但 `avg_per_key_ns` 整体仍优于很多单 key 写路径
- `BatchLookup` 的单 key 成本最低，约 `61.49 ns/key`
- `EndToEnd(long)` 在大 batch 下明显重于 `EndToEnd(short)`

## 7. 关键趋势分析

### 7.1 `batch_size` 仍然是最强影响因子

从当前结果看，最明显的变化通常来自 `batch_size`：

- `batch_size` 从 `1` 增长到 `64` 后，单次调用耗时显著上升
- 但均摊到每个 key 后，大多数接口的 `avg_per_key_ns` 会下降

这说明当前实现里存在比较明显的固定成本，例如：

- 接口调用本身
- 一次锁保护下的状态切换
- 批量容器构造和遍历

### 7.2 `BatchLookup` 是当前最轻路径

无论在小 batch 还是大 batch 下，`BatchLookup` 的 `avg_per_key_ns` 都最低。  
原因也比较直接：

- 它只读当前映射
- 不涉及 slot 分配
- 不涉及新记录发布
- 不涉及队列回收

### 7.3 `EndToEnd(long)` 是当前最重完整链路

当前最值得关注的重路径主要有：

- `EndToEnd(long)`
- `EndToEnd(short)`
- `PreparePut(long)`
- `BatchFree`

它们的共同特点是涉及更多状态变更，或者落在 long slot 路径。

## 8. 端到端结果解读

`EndToEnd(short)` 和 `EndToEnd(long)` 是本次最接近业务真实流程的场景。

几个代表值如下：

- `EndToEnd(short)`
  - `resident_keys = 4096, batch_size = 1`
    `1287.49 ns/batch`
  - `resident_keys = 8192, batch_size = 8`
    `6831.37 ns/batch`
    `853.92 ns/key`
  - `resident_keys = 16384, batch_size = 64`
    `126428.14 ns/batch`
    `1975.44 ns/key`

- `EndToEnd(long)`
  - `resident_keys = 4096, batch_size = 1`
    `967.71 ns/batch`
  - `resident_keys = 8192, batch_size = 8`
    `7581.21 ns/batch`
    `947.65 ns/key`
  - `resident_keys = 16384, batch_size = 64`
    `187989.22 ns/batch`
    `2937.33 ns/key`

可见：

- 完整链路在 batch 化后，单 batch 总耗时上升是必然的
- long 路径在大 batch 下明显更重
- 对吞吐优先场景，适度增大 batch 仍然有意义，但要关注端到端总延迟上升

## 9. 当前结论

可以先给出一个比较直接的工程判断：

- 当前实现已经能够把单 key 控制面操作压在 `us` 级别
- 对 batch 场景，系统仍具有一定摊薄收益
- `BatchLookup` 已经比较轻
- 后续主要优化空间更可能在 `EndToEnd(long)`、`EndToEnd(short)`、`PreparePut(long)` 和 `BatchFree`

如果后续还要继续完善 benchmark，建议增加三类补充场景：

- short / long 混合 batch
- 更大的 batch size，例如 `128 / 256`
- 并发 benchmark，观察单锁实现的扩展性

## 10. 附录

完整矩阵结果不再手工抄录到本文档中，建议直接运行 benchmark 获取 CSV 风格输出：

```sh
./build-wsl-release/meta_manager/kvaf_meta_manager_perf
```

这样更适合后续导入表格工具做进一步筛选、绘图和横向对比。
