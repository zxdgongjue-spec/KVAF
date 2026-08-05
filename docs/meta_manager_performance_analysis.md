# MetaManager 并发性能测试结果

## 1. 测试范围

本文档只记录 `MetaManager` 当前并发版本的性能结果，重点关注：

- 多线程并发执行 `PreparePut`
- 多线程并发执行 `LockKeys`
- 多线程并发执行 `BatchLookup`
- 多线程并发执行 `LockKeys + BatchLookup`

相关代码：

- 接口定义：[meta_manager.h](../meta_manager/include/kvaf/meta_manager/meta_manager.h)
- benchmark：[in_memory_meta_manager_perf.cc](../meta_manager/tests/in_memory_meta_manager_perf.cc)
- 模块说明：[README.md](../meta_manager/README.md)

## 2. 测试环境与命令

- 日期：2026-08-05
- 平台：Ubuntu on WSL
- 编译器：GCC 13.3.0
- 构建类型：`Release`
- 日志开关：关闭

```sh
cd /mnt/c/Urma/KVAF
cmake -S . -B build-wsl-release -DBUILD_TESTING=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-wsl-release -j
ctest --test-dir build-wsl-release --output-on-failure
./build-wsl-release/meta_manager/kvaf_meta_manager_perf \
  --concurrent-only
```

如需观察执行进度：

```sh
./build-wsl-release/meta_manager/kvaf_meta_manager_perf \
  --concurrent-only --show-progress
```

## 3. 测试场景与统计口径

本次场景固定为：

- `resident_keys = 2097152`
- `batch_size = 32768`
- `thread_count = 4 / 8 / 16`
- 每轮线程处理互不重叠的 key batch

测试接口：

- `ConcurrentPreparePut(short/long)`：多线程并发预留短/长 slot
- `ConcurrentLockKeys`：多线程并发建立锁上下文
- `ConcurrentBatchLookup`：预先建立锁上下文，再并发查询
- `ConcurrentLockLookup`：每个线程连续执行 `LockKeys + BatchLookup`

正式计时前执行 1 个独立 warm-up round。warm-up 用于预热线程、容器容量、
内存页和锁状态，不计入 `call_count`、总耗时或延迟统计。

表格中的总耗时和延迟使用 `ms`，单 key 均摊耗时保留 `ns`：

- `total_ms`：所有正式测量轮次的累计 wall time
- `avg_ms`：单轮平均 wall time
- `avg_per_batch_ms`：`avg_ms / thread_count`，是并发轮次的派生值
- `avg_per_key_ns`：`avg_ns / (batch_size * thread_count)`，保留纳秒单位
- `min_ms / p50_ms / p95_ms / max_ms`：轮次延迟统计

`avg_per_batch_ms` 不等同于单线程实际接口延迟，判断稳定延迟时应优先查看
`p50_ms`、`p95_ms` 和 `max_ms`。

## 4. 性能测试结果

| method | threads | call_count | total_ms | avg_ms | avg_per_batch_ms | avg_per_key_ns | min_ms | p50_ms | p95_ms | max_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ConcurrentPreparePut(short)` | 4 | 16 | 989.60 | 61.85023375 | 15.46255844 | 471.88 | 54.680240 | 63.040590 | 67.816580 | 67.816580 |
| `ConcurrentPreparePut(long)` | 4 | 16 | 971.89 | 60.74342812 | 15.18585703 | 463.44 | 53.280800 | 59.147430 | 77.140600 | 77.140600 |
| `ConcurrentLockKeys` | 4 | 16 | 80.51 | 5.03212438 | 1.25803109 | 38.39 | 4.358320 | 4.913230 | 6.115950 | 6.115950 |
| `ConcurrentBatchLookup` | 4 | 16 | 6.22 | 0.38860750 | 0.09715188 | 2.96 | 0.343960 | 0.393720 | 0.402170 | 0.402170 |
| `ConcurrentLockLookup` | 4 | 16 | 87.21 | 5.45059312 | 1.36264828 | 41.58 | 5.045190 | 5.369040 | 6.769040 | 6.769040 |
| `ConcurrentPreparePut(short)` | 8 | 8 | 918.21 | 114.77594625 | 14.34699328 | 437.84 | 105.429830 | 115.713990 | 123.458600 | 123.458600 |
| `ConcurrentPreparePut(long)` | 8 | 8 | 860.36 | 107.54501375 | 13.44312672 | 410.25 | 94.493310 | 108.194360 | 119.772930 | 119.772930 |
| `ConcurrentLockKeys` | 8 | 8 | 46.19 | 5.77320375 | 0.72165047 | 22.02 | 4.784600 | 5.851260 | 6.394370 | 6.394370 |
| `ConcurrentBatchLookup` | 8 | 8 | 4.23 | 0.52827250 | 0.06603406 | 2.02 | 0.459290 | 0.551380 | 0.638930 | 0.638930 |
| `ConcurrentLockLookup` | 8 | 8 | 51.22 | 6.40275875 | 0.80034484 | 24.42 | 5.632690 | 6.313810 | 7.583100 | 7.583100 |
| `ConcurrentPreparePut(short)` | 16 | 4 | 882.99 | 220.74705500 | 13.79669094 | 421.04 | 202.412500 | 229.201940 | 236.085730 | 236.085730 |
| `ConcurrentPreparePut(long)` | 16 | 4 | 841.71 | 210.42872250 | 13.15179516 | 401.36 | 185.325420 | 218.546700 | 231.065890 | 231.065890 |
| `ConcurrentLockKeys` | 16 | 4 | 24.25 | 6.06298250 | 0.37893641 | 11.56 | 5.584140 | 6.241790 | 6.645900 | 6.645900 |
| `ConcurrentBatchLookup` | 16 | 4 | 3.35 | 0.83729500 | 0.05233094 | 1.60 | 0.657660 | 0.974980 | 1.040100 | 1.040100 |
| `ConcurrentLockLookup` | 16 | 4 | 27.05 | 6.76331750 | 0.42270734 | 12.90 | 6.337930 | 7.071820 | 7.244760 | 7.244760 |

## 5. 结果结论

- warm-up 后，`ConcurrentLockKeys` 的最大轮次耗时稳定在
  `6.12 ~ 6.65 ms`，此前几十到上百毫秒的异常长尾已消失。
- `ConcurrentBatchLookup` 明显轻于 `ConcurrentLockKeys`，当前并发读路径的
  主要成本仍在 shard 查找、共享锁获取和锁上下文创建。
- `ConcurrentLockLookup` 比 `ConcurrentLockKeys` 仅增加少量 lookup 成本，
  4/8/16 threads 的 P50 分别为 `5.37 / 6.31 / 7.07 ms`。
- `ConcurrentPreparePut` 的均摊成本约为
  `401.36 ~ 471.88 ns/key`，仍明显高于 `BatchLookup`。
- `avg_per_batch_ms` 会随线程数增加而下降，但它是 wall time 的均摊值，
  不能替代真实的单线程调用延迟。

当前结果说明 warm-up 能有效降低冷启动和调度抖动，但不能消除操作系统抢占、
线程迁移及共享锁内部竞争。后续如需继续优化，应补充每个线程自身耗时和整轮
wall time 两类指标。
