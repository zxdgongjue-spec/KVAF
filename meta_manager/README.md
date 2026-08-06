# Meta Manager

## 模块职责

`MetaManager` 维护 `KeyId -> 远端地址` 的元数据映射，当前实现覆盖：

- 写路径：`PreparePut / CommitPut / AbortPut`
- 读路径：`LockKeys / BatchLookup / UnlockKeys`
- 回收路径：`BatchFree`

当前版本是内存版 POC，不负责 payload 搬运。

## 当前边界

- 单 `MetaManager` 实例、单远端 `region`
- `KeyId = std::string`
- 支持 short slot 和 long slot，两类 slot 数量相同
- 一个 key 只保留一条当前可见记录
- 不支持多 region、覆盖写和可变长分配

## 核心流程

内部采用 `hashtable + queue`，并按 key 分片：

- `key_shards_`：分片维护 `key -> record`
- `short_free_slots_ / long_free_slots_`：维护两类空闲 slot
- `LockKeys`：建立读阶段 lease，固定 key 直到 `UnlockKeys`
- `BatchLookup`：返回已固定 key 的地址信息
- `BatchFree`：回收未被锁定的记录

写路径：

```text
PreparePut -> payload 写入 -> CommitPut
```

读路径：

```text
LockKeys -> BatchLookup -> UnlockKeys
```

## 结果结构

接口定义见 [meta_manager.h](include/kvaf/meta_manager/meta_manager.h)。

```cpp
struct AccessDesc {
  std::vector<const KeyId*> keys;
  std::vector<Address> addresses;
  std::vector<std::uint64_t> lengths;
  std::vector<std::uint8_t> valid;
};

using PutAccessDesc = AccessDesc;
using FlatAccessDesc = AccessDesc;
```

`PutAccessDesc` 和 `FlatAccessDesc` 保留语义名称，但底层直接使用并行数组。
同一个下标 `i` 的 `keys[i]`、`addresses[i]`、`lengths[i]` 和 `valid[i]`
描述同一条记录。`keys[i]` 指向 MetaManager 内部稳定存储的字符串，避免在
热路径复制 `std::string`；下游可以直接访问 `addresses[i]` 和 `lengths[i]`。

## 构建与测试

Ubuntu/WSL Debug 构建：

```sh
cd /mnt/c/Urma/KVAF
cmake -S . -B build-wsl -DBUILD_TESTING=ON
cmake --build build-wsl -j
ctest --test-dir build-wsl --output-on-failure
```

Release 构建：

```sh
cmake -S . -B build-wsl-release -DBUILD_TESTING=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-wsl-release -j
ctest --test-dir build-wsl-release --output-on-failure
```

## 计时日志

接口级计时日志默认关闭：

```cpp
manager->SetTimingLogEnabled(true);
```

性能测试建议关闭日志，避免 I/O 干扰。

## 并发性能测试

benchmark 程序见 [in_memory_meta_manager_perf.cc](tests/in_memory_meta_manager_perf.cc)。

运行全部 benchmark：

```sh
./build-wsl-release/meta_manager/kvaf_meta_manager_perf
```

只运行并发场景：

```sh
./build-wsl-release/meta_manager/kvaf_meta_manager_perf \
  --concurrent-only
```

显示阶段进度：

```sh
./build-wsl-release/meta_manager/kvaf_meta_manager_perf \
  --concurrent-only --show-progress
```

并发测试固定使用：

- `resident_keys = 2097152`
- `batch_size = 32768`
- `thread_count = 4 / 8 / 16`
- 每个线程处理互不重叠的 key batch

测试接口：

- `ConcurrentPreparePut(short/long)`
- `ConcurrentLockKeys`
- `ConcurrentBatchLookup`
- `ConcurrentLockLookup`

正式计时前会执行 1 个独立 warm-up round。warm-up 用于预热线程、容器容量、
内存页和锁状态，不计入正式统计。

### 结果字段

- `total_ms`：所有正式轮次的累计 wall time
- `avg_ns`：单轮平均 wall time
- `avg_per_batch_ns`：`avg_ns / thread_count` 的派生值
- `avg_per_key_ns`：按本轮所有 key 均摊的耗时
- `p50_ns / p95_ns / max_ns`：轮次延迟统计

benchmark 原始 CSV 仍输出 `ns` 字段；下面的结果摘要统一换算为 `ms`。

### 2026-08-05 结果摘要

| 方法 | threads | avg_ms | avg_per_batch_ms | avg_per_key_ns | p50_ms | max_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ConcurrentLockKeys` | 4 | 5.0321 | 1.2580 | 38.39 | 4.9132 | 6.1160 |
| `ConcurrentBatchLookup` | 4 | 0.3886 | 0.0972 | 2.96 | 0.3937 | 0.4022 |
| `ConcurrentLockLookup` | 4 | 5.4506 | 1.3626 | 41.58 | 5.3690 | 6.7690 |
| `ConcurrentLockKeys` | 8 | 5.7732 | 0.7217 | 22.02 | 5.8513 | 6.3944 |
| `ConcurrentBatchLookup` | 8 | 0.5283 | 0.0660 | 2.02 | 0.5514 | 0.6389 |
| `ConcurrentLockLookup` | 8 | 6.4028 | 0.8003 | 24.42 | 6.3138 | 7.5831 |
| `ConcurrentLockKeys` | 16 | 6.0630 | 0.3789 | 11.56 | 6.2418 | 6.6459 |
| `ConcurrentBatchLookup` | 16 | 0.8373 | 0.0523 | 1.60 | 0.9750 | 1.0401 |
| `ConcurrentLockLookup` | 16 | 6.7633 | 0.4227 | 12.90 | 7.0718 | 7.2448 |

warm-up 后，`ConcurrentLockKeys` 的异常长尾已从几十/上百毫秒收敛到约
`6.1~6.7 ms`。当前并发读路径的主要成本仍集中在 `LockKeys`。

完整结果和详细分析见：

- [meta_manager_performance_analysis.md](../docs/meta_manager_performance_analysis.md)
