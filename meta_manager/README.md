# Meta Manager

## 模块职责

`MetaManager` 负责维护 `KeyId -> 远端地址` 的元数据映射，当前实现聚焦地址管理这条控制面链路：

- 写路径：`PreparePut / CommitPut / AbortPut`
- 读路径：`LockKeys / BatchLookup / UnlockKeys`
- 回收路径：`BatchFree`

当前版本是一个内存版 POC，重点验证 `put / lock / get / free` 语义，不负责 payload 真正搬运。

## 当前边界

当前实现支持：

- 单 `MetaManager` 实例
- 单远端 `region`
- 两类 slot：`short_slot_size` 和 `long_slot_size`
- 两类 slot 数量相同，由 `slot_count_per_class` 指定
- `KeyId = std::string`
- 一个 key 只保留一条当前可见记录

当前实现不负责：

- 多 region 调度
- prefix key 转换
- payload 拷贝
- 版本覆盖

## 核心思路

内部仍然是 `hashtable + queue`：

- `key_index_`：维护 `key -> record`
- `short_free_slots_ / long_free_slots_`：维护两类空闲 slot 队列

写路径分两阶段：

1. `PreparePut` 先预留地址，但结果对读路径不可见
2. `CommitPut` 再把结果发布到 `key_index_`

读路径分三步：

1. `LockKeys` 固定一批 key，并增加 `pin_count`
2. `BatchLookup` 返回这批 key 当前对应的地址信息
3. `UnlockKeys` 释放快照

## 结果结构

接口定义见 [meta_manager.h](include/kvaf/meta_manager/meta_manager.h)。

现在 `PutAccessDesc` 和 `FlatAccessDesc` 都统一复用同一个底层结构：

```cpp
struct AccessEntry {
  KeyId key;
  Address address;
  std::uint64_t length;
  std::uint8_t valid;
};

struct AccessDesc {
  std::vector<AccessEntry> entries;
};

using PutAccessDesc = AccessDesc;
using FlatAccessDesc = AccessDesc;
```

这样做的目的有两个：

- 保留 `PutAccessDesc / FlatAccessDesc` 这两个语义名，调用点更容易理解
- 统一成 `entries[i]` 访问方式，避免多组并行数组带来的错位风险

### 接口示例

`PreparePut` 返回写入地址时：

```cpp
PutContext put_context;
PutAccessDesc put_desc;
manager->PreparePut(keys, lengths, &put_context, &put_desc);

const auto& entry = put_desc.entries[i];
Address write_address = entry.address;
std::uint64_t reserved_length = entry.length;
```

`BatchLookup` 返回读地址时：

```cpp
LockContext lock_context;
FlatAccessDesc access_desc;
manager->LockKeys(keys, kvaf::MetaManagerLockMode::kStableReadAllowAppend,
                  &lock_context);
manager->BatchLookup(keys, lock_context, &access_desc);

const auto& entry = access_desc.entries[i];
if (entry.valid) {
  Address read_address = entry.address;
  std::uint64_t valid_length = entry.length;
}
```

## 构建与验证

在 Ubuntu on WSL 下可直接执行：

```sh
cd /mnt/c/Urma/KVAF
cmake -S . -B build-wsl -DBUILD_TESTING=ON
cmake --build build-wsl -j
ctest --test-dir build-wsl --output-on-failure
```

Release 构建：

```sh
cmake -S . -B build-wsl-release -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-wsl-release -j
ctest --test-dir build-wsl-release --output-on-failure
```

## 计时日志

接口级计时日志默认关闭，可通过：

```cpp
auto manager = kvaf::CreateInMemoryMetaManager();
manager->SetTimingLogEnabled(true);
```

打开后，每次核心接口调用都会打印一行日志，例如：

```text
[MetaManagerTiming] method=PreparePut elapsed_ns=812
```

关闭方式：

```cpp
manager->SetTimingLogEnabled(false);
```

性能测试时建议关闭日志，避免日志 I/O 干扰结果。

## 性能测试

性能测试程序见 [in_memory_meta_manager_perf.cc](tests/in_memory_meta_manager_perf.cc)。

构建与执行：

```sh
cmake -S . -B build-wsl-release -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-wsl-release -j
./build-wsl-release/meta_manager/kvaf_meta_manager_perf
```

如需同时观察接口级计时日志：

```sh
./build-wsl-release/meta_manager/kvaf_meta_manager_perf --enable-timing-log
```

### 测试场景

这版 benchmark 采用二维矩阵：

- `resident_keys = 4096 / 8192 / 16384`
- `batch_size = 1 / 8 / 64`

其中：

- `resident_keys` 表示测量开始前，`MetaManager` 中已经存在的已提交 key 数量
- `batch_size` 表示单次接口调用要处理多少个 key

各接口的具体场景如下：

- `PreparePut(short)`：先构造 `resident_keys` 条已提交短记录，再对一批新的短 payload 做预留
- `PreparePut(long)`：与上面相同，但 payload 落到 long slot
- `CommitPut`：在已有 resident 集合上持续提交新的 batch，模拟持续写入发布
- `AbortPut`：在 steady-state resident 集合上重复执行 `prepare + abort`
- `LockKeys / BatchLookup / UnlockKeys`：对 resident 集合按 `batch_size` 分批做稳定读
- `BatchFree`：把 resident 集合按 `batch_size` 逐批释放
- `EndToEnd(short)`：完整控制面链路
  `PreparePut -> CommitPut -> LockKeys -> BatchLookup -> UnlockKeys -> BatchFree`
- `EndToEnd(long)`：与上面相同，但写入长度落到 long slot

### 输出字段

benchmark 输出列如下：

- `call_count`：当前场景下实际测量的调用次数，通常等于 `resident_keys / batch_size`
- `avg_ns`：单次接口调用平均耗时
- `avg_per_key_ns`：把 `avg_ns` 均摊到单个 key 的耗时
- `total_ms`：当前方法在当前场景下，所有测量调用的总耗时
- `p50_ns / p95_ns / max_ns`：延迟分位数与极值

当前 benchmark 不再为了凑统计样本而重复多轮。

`batch_size` 会明显影响结果，所以观察时建议同时看：

- 单次调用耗时 `avg_ns`
- 单 key 均摊耗时 `avg_per_key_ns`
- 场景总耗时 `total_ms`

### 本次实测环境

- 日期：2026-08-03
- 平台：Ubuntu on WSL
- 编译器：GCC 13.3.0
- 构建类型：`Release`
- 日志开关：关闭

### 代表性结果

完整矩阵会由 benchmark 程序直接打印；下面摘录两组边界场景，便于快速看趋势。

| 方法 | resident_keys | batch_size | call_count | total_ms | avg_ns | avg_per_key_ns | p50_ns | p95_ns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `RegisterRemoteRegion` | 1 | 1 | 1 | 0.00 | 3253.00 | 3253.00 | 3253.00 | 3253.00 |
| `PreparePut(short)` | 4096 | 1 | 4096 | 1.98 | 482.92 | 482.92 | 391.00 | 826.00 |
| `PreparePut(short)` | 16384 | 64 | 256 | 7.88 | 30771.98 | 480.81 | 25584.00 | 50635.00 |
| `PreparePut(long)` | 4096 | 1 | 4096 | 2.12 | 517.96 | 517.96 | 404.00 | 958.00 |
| `PreparePut(long)` | 16384 | 64 | 256 | 11.24 | 43915.32 | 686.18 | 34805.00 | 90444.00 |
| `CommitPut` | 4096 | 1 | 4096 | 0.94 | 229.55 | 229.55 | 154.00 | 265.00 |
| `CommitPut` | 16384 | 64 | 256 | 5.65 | 22089.10 | 345.14 | 11560.00 | 17447.00 |
| `AbortPut` | 4096 | 1 | 4096 | 0.35 | 84.97 | 84.97 | 83.00 | 86.00 |
| `AbortPut` | 16384 | 64 | 256 | 1.26 | 4919.40 | 76.87 | 4520.00 | 6561.00 |
| `LockKeys` | 4096 | 1 | 4096 | 0.87 | 211.91 | 211.91 | 180.00 | 238.00 |
| `LockKeys` | 16384 | 64 | 256 | 7.32 | 28596.88 | 446.83 | 22601.00 | 68105.00 |
| `BatchLookup` | 4096 | 1 | 4096 | 0.42 | 103.61 | 103.61 | 96.00 | 120.00 |
| `BatchLookup` | 16384 | 64 | 256 | 1.01 | 3935.50 | 61.49 | 3875.00 | 4851.00 |
| `UnlockKeys` | 4096 | 1 | 4096 | 0.49 | 119.39 | 119.39 | 110.00 | 119.00 |
| `UnlockKeys` | 16384 | 64 | 256 | 1.27 | 4954.62 | 77.42 | 4725.00 | 6385.00 |
| `BatchFree` | 4096 | 1 | 4096 | 0.82 | 200.40 | 200.40 | 182.00 | 245.00 |
| `BatchFree` | 16384 | 64 | 256 | 6.08 | 23758.08 | 371.22 | 22300.00 | 30604.00 |
| `EndToEnd(short)` | 4096 | 1 | 4096 | 5.27 | 1287.49 | 1287.49 | 1169.00 | 1783.00 |
| `EndToEnd(short)` | 16384 | 64 | 256 | 32.37 | 126428.14 | 1975.44 | 103477.00 | 228539.00 |
| `EndToEnd(long)` | 4096 | 1 | 4096 | 3.96 | 967.71 | 967.71 | 912.00 | 1133.00 |
| `EndToEnd(long)` | 16384 | 64 | 256 | 48.13 | 187989.22 | 2937.33 | 106095.00 | 276077.00 |

### 结果解读

- `batch_size` 对单次调用耗时有明显影响，batch 越大，`avg_ns` 往往越高
- 但从 `avg_per_key_ns` 看，大 batch 往往更划算，因为固定的函数调用和锁开销被摊薄了
- `BatchLookup` 的单 key 成本最低，当前代表值约 `61.49 ns/key`
- `CommitPut / BatchFree / EndToEnd` 相对更重，因为它们涉及更多状态变更
- `EndToEnd(short)` 和 `EndToEnd(long)` 可以直接看一批请求走完整条控制面链路的总成本；例如 `resident_keys=16384, batch_size=64` 时：
  short 路径约 `126.43 us/batch`，`1975.44 ns/key`
  long 路径约 `187.99 us/batch`，`2937.33 ns/key`

如果需要完整矩阵结果，直接运行 benchmark 即可；程序会输出完整 CSV 风格结果，便于复制到表格工具继续分析。
