#include "kvaf/meta_manager/meta_manager.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using kvaf::AccessDesc;
using kvaf::CreateInMemoryMetaManager;
using kvaf::FlatAccessDesc;
using kvaf::KeyId;
using kvaf::LockContext;
using kvaf::MetaManager;
using kvaf::MetaManagerLockMode;
using kvaf::PutAccessDesc;
using kvaf::PutContext;
using kvaf::RemoteRegionDesc;
using kvaf::Status;

bool g_show_progress = false;

constexpr std::uint64_t kBaseAddress = 4096;
constexpr std::uint64_t kShortSlotSize = 128;
constexpr std::uint64_t kLongSlotSize = 256;
constexpr std::uint64_t kShortLength = 64;
constexpr std::uint64_t kLongLength = 200;

// resident_keys 表示测试开始前，已经在 key_index_ 中常驻的已提交 key 数量。
// batch_size 表示单次接口调用处理多少个 key。
struct BenchmarkScenario {
  std::size_t resident_keys{0};
  std::size_t batch_size{0};
  std::size_t thread_count{1};
};

struct Stats {
  std::string method;
  std::size_t resident_keys{0};
  std::size_t batch_size{0};
  std::size_t thread_count{1};
  std::size_t call_count{0};
  double total_ms{0.0};
  double avg_ns{0.0};
  double avg_per_batch_ns{0.0};
  double avg_per_key_ns{0.0};
  double min_ns{0.0};
  double p50_ns{0.0};
  double p95_ns{0.0};
  double max_ns{0.0};
};

struct BenchmarkOptions {
  bool enable_timing_log{false};
  bool include_concurrent{false};
  bool concurrent_only{false};
  bool show_progress{false};
};

void CheckOk(const Status& status, const char* step) {
  if (!status.ok()) {
    std::cerr << "Benchmark failed at " << step
              << " status=" << status.message() << '\n';
    std::exit(1);
  }
}

[[noreturn]] void FailBenchmark(const char* step, const std::string& message) {
  std::cerr << "Benchmark validation failed at " << step << ": " << message
            << '\n';
  std::exit(1);
}

KeyId MakeKey(const std::string& prefix, std::size_t index) {
  return prefix + "_" + std::to_string(index);
}

RemoteRegionDesc MakeRegion(std::uint64_t slot_count_per_class) {
  RemoteRegionDesc region;
  region.base = kBaseAddress;
  region.short_slot_size = kShortSlotSize;
  region.long_slot_size = kLongSlotSize;
  region.slot_count_per_class = slot_count_per_class;
  region.length = slot_count_per_class * (kShortSlotSize + kLongSlotSize);
  return region;
}

std::unique_ptr<MetaManager> MakeInitializedManager(
    std::uint64_t slot_count_per_class,
    bool enable_timing_log) {
  auto manager = CreateInMemoryMetaManager();
  manager->SetTimingLogEnabled(enable_timing_log);
  manager->SetPerformanceModeEnabled(true);
  CheckOk(manager->RegisterRemoteRegion(MakeRegion(slot_count_per_class)),
          "RegisterRemoteRegion");
  return manager;
}

std::vector<BenchmarkScenario> BuildScenarios() {
  return {
      BenchmarkScenario{128*1024, 2*1024, 1},
      BenchmarkScenario{2*128*1024, 2*2*1024, 1},
      BenchmarkScenario{4*128*1024, 4*2*1024, 1},
      BenchmarkScenario{8*128*1024, 8*2*1024, 1},
      BenchmarkScenario{16*128*1024, 16*2*1024, 1},
  };
}

std::vector<BenchmarkScenario> BuildConcurrentScenarios() {
  return {
      BenchmarkScenario{4*128*1024, 4*2*1024, 4},
      BenchmarkScenario{8*128*1024, 8*2*1024, 4},
      BenchmarkScenario{16*128*1024, 16*2*1024, 4},
      BenchmarkScenario{16*128*1024, 16*2*1024, 8},
  };
}

std::size_t BatchesPerRound(const BenchmarkScenario& scenario) {
  if (scenario.batch_size == 0 ||
      scenario.resident_keys % scenario.batch_size != 0) {
    FailBenchmark("BatchesPerRound", "resident_keys must be divisible by batch_size");
  }
  return scenario.resident_keys / scenario.batch_size;
}

std::size_t ConcurrentRounds(const BenchmarkScenario& scenario) {
  const std::size_t batch_count = BatchesPerRound(scenario);
  if (scenario.thread_count == 0 ||
      batch_count % scenario.thread_count != 0) {
    FailBenchmark("ConcurrentRounds",
                  "batch_count must be divisible by thread_count");
  }
  return batch_count / scenario.thread_count;
}

Stats BuildStats(const std::string& method,
                 const BenchmarkScenario& scenario,
                 std::vector<double> call_latencies_ns) {
  std::sort(call_latencies_ns.begin(), call_latencies_ns.end());
  const double total_ns =
      std::accumulate(call_latencies_ns.begin(), call_latencies_ns.end(), 0.0);

  Stats stats;
  stats.method = method;
  stats.resident_keys = scenario.resident_keys;
  stats.batch_size = scenario.batch_size;
  stats.thread_count = scenario.thread_count;
  stats.call_count = call_latencies_ns.size();
  stats.total_ms = total_ns / 1000000.0;
  stats.avg_ns = total_ns / static_cast<double>(call_latencies_ns.size());
  stats.avg_per_batch_ns =
      stats.avg_ns / static_cast<double>(scenario.thread_count);
  stats.avg_per_key_ns =
      stats.avg_ns / static_cast<double>(scenario.batch_size * scenario.thread_count);
  stats.min_ns = call_latencies_ns.front();
  stats.p50_ns = call_latencies_ns[call_latencies_ns.size() / 2];
  stats.p95_ns =
      call_latencies_ns[(call_latencies_ns.size() * 95) / 100];
  stats.max_ns = call_latencies_ns.back();
  return stats;
}

void AppendStats(std::vector<Stats>* stats_list,
                 const Stats& stats,
                 bool show_progress) {
  if (show_progress) {
    std::cerr << "[BenchmarkProgress] finished " << stats.method
              << " resident_keys=" << stats.resident_keys
              << " batch_size=" << stats.batch_size
              << " thread_count=" << stats.thread_count << '\n';
  }
  stats_list->push_back(stats);
}

void ProgressLog(const std::string& message) {
  if (!g_show_progress) {
    return;
  }
  std::cerr << "[BenchmarkProgress] " << message << '\n';
}

std::vector<KeyId> BuildKeyBatch(const std::string& prefix,
                                 std::size_t start_index,
                                 std::size_t batch_size) {
  std::vector<KeyId> keys;
  keys.reserve(batch_size);
  for (std::size_t i = 0; i < batch_size; ++i) {
    keys.push_back(MakeKey(prefix, start_index + i));
  }
  return keys;
}

std::size_t BatchCount(std::size_t total_keys, std::size_t batch_size) {
  if (batch_size == 0 || total_keys % batch_size != 0) {
    FailBenchmark("BatchCount", "total_keys must be divisible by batch_size");
  }
  return total_keys / batch_size;
}

std::vector<std::uint64_t> BuildLengthBatch(std::size_t batch_size,
                                            std::uint64_t length) {
  return std::vector<std::uint64_t>(batch_size, length);
}

void CheckAccessDesc(const AccessDesc& access_desc,
                     const std::vector<KeyId>& keys,
                     std::uint64_t expected_length,
                     const char* step) {
  if (access_desc.entries.size() != keys.size()) {
    FailBenchmark(step, "returned entry count does not match batch size");
  }

  for (std::size_t i = 0; i < keys.size(); ++i) {
    const auto& entry = access_desc.entries[i];
    if (entry.length != expected_length) {
      FailBenchmark(step, "returned length does not match the requested length");
    }
    if (entry.valid != 1) {
      FailBenchmark(step, "returned valid flag is not set");
    }
  }
}

bool ValidateAccessDesc(const AccessDesc& access_desc,
                        const std::vector<KeyId>& keys,
                        std::uint64_t expected_length,
                        std::string* error_message) {
  if (access_desc.entries.size() != keys.size()) {
    *error_message = "returned entry count does not match batch size";
    return false;
  }

  for (std::size_t i = 0; i < keys.size(); ++i) {
    const auto& entry = access_desc.entries[i];
    if (entry.length != expected_length) {
      *error_message = "returned length does not match the requested length";
      return false;
    }
    if (entry.valid != 1) {
      *error_message = "returned valid flag is not set";
      return false;
    }
  }

  return true;
}

void PopulateCommittedKeys(MetaManager* manager,
                           const std::string& prefix,
                           std::size_t total_keys,
                           std::size_t batch_size,
                           const std::vector<std::uint64_t>& lengths,
                           const char* progress_label = nullptr) {
  const std::size_t batch_count = BatchCount(total_keys, batch_size);
  const std::size_t progress_interval =
      std::max<std::size_t>(1, batch_count / 8);
  for (std::size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
    const auto keys =
        BuildKeyBatch(prefix, batch_index * batch_size, batch_size);
    PutContext put_context;
    PutAccessDesc put_desc;
    CheckOk(manager->PreparePut(keys, lengths, &put_context, &put_desc),
            "PreparePut(populate committed keys)");
    CheckAccessDesc(put_desc, keys, lengths.front(),
                    "PreparePut(populate committed keys)");
    CheckOk(manager->CommitPut(put_context), "CommitPut(populate committed keys)");

    if (progress_label != nullptr &&
        ((batch_index + 1) % progress_interval == 0 ||
         batch_index + 1 == batch_count)) {
      ProgressLog(std::string(progress_label) + " populated_batches=" +
                  std::to_string(batch_index + 1) + "/" +
                  std::to_string(batch_count));
    }
  }
}

std::vector<std::vector<KeyId>> BuildRoundKeyBatches(
    const std::string& prefix,
    const BenchmarkScenario& scenario,
    std::size_t round) {
  std::vector<std::vector<KeyId>> round_key_batches;
  round_key_batches.reserve(scenario.thread_count);
  for (std::size_t t = 0; t < scenario.thread_count; ++t) {
    const std::size_t batch_index = round * scenario.thread_count + t;
    round_key_batches.push_back(
        BuildKeyBatch(prefix, batch_index * scenario.batch_size,
                      scenario.batch_size));
  }
  return round_key_batches;
}

template <typename WorkerFunc>
double RunConcurrentRound(std::size_t thread_count,
                          WorkerFunc worker_func,
                          std::string* first_error) {
  std::vector<std::thread> workers;
  workers.reserve(thread_count);

  std::atomic<std::size_t> ready_count{0};
  std::atomic<bool> start_flag{false};
  std::mutex error_mutex;

  for (std::size_t t = 0; t < thread_count; ++t) {
    workers.emplace_back([&, t]() {
      ready_count.fetch_add(1, std::memory_order_acq_rel);
      while (!start_flag.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }

      const std::string worker_error = worker_func(t);
      if (worker_error.empty()) {
        return;
      }

      std::lock_guard<std::mutex> error_lock(error_mutex);
      if (first_error->empty()) {
        *first_error = worker_error;
      }
    });
  }

  while (ready_count.load(std::memory_order_acquire) != thread_count) {
    std::this_thread::yield();
  }

  const auto start = std::chrono::steady_clock::now();
  start_flag.store(true, std::memory_order_release);

  for (auto& worker : workers) {
    worker.join();
  }

  const auto end = std::chrono::steady_clock::now();
  return static_cast<double>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
          .count());
}

Stats BenchmarkRegisterRemoteRegion(bool enable_timing_log) {
  std::vector<double> call_latencies_ns;
  call_latencies_ns.reserve(1);
  auto manager = CreateInMemoryMetaManager();
  manager->SetTimingLogEnabled(enable_timing_log);
  const auto start = std::chrono::steady_clock::now();
  CheckOk(manager->RegisterRemoteRegion(MakeRegion(1)), "RegisterRemoteRegion");
  const auto end = std::chrono::steady_clock::now();
  call_latencies_ns.push_back(static_cast<double>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
          .count()));

  BenchmarkScenario scenario;
  scenario.resident_keys = 1;
  scenario.batch_size = 1;
  scenario.thread_count = 1;
  return BuildStats("RegisterRemoteRegion", scenario,
                    std::move(call_latencies_ns));
}

// PreparePut(short): 在已有 resident_keys 条已提交短记录的前提下，
// 为一批新的短 payload 预留地址；测量的是单次 PreparePut 调用本身。
Stats BenchmarkPreparePutShort(const BenchmarkScenario& scenario,
                               bool enable_timing_log) {
  auto manager = MakeInitializedManager(
      scenario.resident_keys + scenario.batch_size + 1, enable_timing_log);
  const auto resident_lengths =
      BuildLengthBatch(scenario.batch_size, kShortLength);
  const std::size_t batch_count =
      BatchCount(scenario.resident_keys, scenario.batch_size);
  PopulateCommittedKeys(manager.get(), "prepare_short_resident",
                        scenario.resident_keys, scenario.batch_size,
                        resident_lengths);

  const auto work_lengths = BuildLengthBatch(scenario.batch_size, kShortLength);

  std::vector<double> call_latencies_ns;
  call_latencies_ns.reserve(batch_count);
  for (std::size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
    const auto work_keys = BuildKeyBatch("prepare_short_work",
                                         batch_index * scenario.batch_size,
                                         scenario.batch_size);
    PutContext put_context;
    PutAccessDesc put_desc;
    const auto start = std::chrono::steady_clock::now();
    CheckOk(manager->PreparePut(work_keys, work_lengths, &put_context, &put_desc),
            "PreparePut(short)");
    const auto end = std::chrono::steady_clock::now();
    CheckAccessDesc(put_desc, work_keys, kShortLength, "PreparePut(short)");
    CheckOk(manager->AbortPut(put_context), "AbortPut(after PreparePut short)");
    call_latencies_ns.push_back(static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count()));
  }
  return BuildStats("PreparePut(short)", scenario,
                    std::move(call_latencies_ns));
}

// PreparePut(long): 场景与 short 相同，但 payload 落到 long slot。
Stats BenchmarkPreparePutLong(const BenchmarkScenario& scenario,
                              bool enable_timing_log) {
  auto manager = MakeInitializedManager(
      scenario.resident_keys + scenario.batch_size + 1, enable_timing_log);
  const auto resident_lengths =
      BuildLengthBatch(scenario.batch_size, kShortLength);
  const std::size_t batch_count =
      BatchCount(scenario.resident_keys, scenario.batch_size);
  PopulateCommittedKeys(manager.get(), "prepare_long_resident",
                        scenario.resident_keys, scenario.batch_size,
                        resident_lengths);

  const auto work_lengths = BuildLengthBatch(scenario.batch_size, kLongLength);

  std::vector<double> call_latencies_ns;
  call_latencies_ns.reserve(batch_count);
  for (std::size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
    const auto work_keys = BuildKeyBatch("prepare_long_work",
                                         batch_index * scenario.batch_size,
                                         scenario.batch_size);
    PutContext put_context;
    PutAccessDesc put_desc;
    const auto start = std::chrono::steady_clock::now();
    CheckOk(manager->PreparePut(work_keys, work_lengths, &put_context, &put_desc),
            "PreparePut(long)");
    const auto end = std::chrono::steady_clock::now();
    CheckAccessDesc(put_desc, work_keys, kLongLength, "PreparePut(long)");
    CheckOk(manager->AbortPut(put_context), "AbortPut(after PreparePut long)");
    call_latencies_ns.push_back(static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count()));
  }
  return BuildStats("PreparePut(long)", scenario,
                    std::move(call_latencies_ns));
}

// CommitPut: 每轮先构造 resident_keys 条已存在记录，再顺序追加同等规模的新 batch，
// 该场景模拟“持续写入并发布元数据”的成本。
Stats BenchmarkCommitPut(const BenchmarkScenario& scenario,
                         bool enable_timing_log) {
  const auto lengths = BuildLengthBatch(scenario.batch_size, kShortLength);
  const std::size_t batch_count =
      BatchCount(scenario.resident_keys, scenario.batch_size);

  std::vector<double> call_latencies_ns;
  call_latencies_ns.reserve(batch_count);
  auto manager = MakeInitializedManager(
      scenario.resident_keys + scenario.resident_keys + 1, enable_timing_log);
  PopulateCommittedKeys(manager.get(), "commit_resident",
                        scenario.resident_keys, scenario.batch_size, lengths);

  for (std::size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
    const auto keys =
        BuildKeyBatch("commit_work", batch_index * scenario.batch_size,
                      scenario.batch_size);
    PutContext put_context;
    PutAccessDesc put_desc;
    CheckOk(manager->PreparePut(keys, lengths, &put_context, &put_desc),
            "PreparePut(before CommitPut)");
    CheckAccessDesc(put_desc, keys, kShortLength, "PreparePut(before CommitPut)");

    const auto start = std::chrono::steady_clock::now();
    CheckOk(manager->CommitPut(put_context), "CommitPut");
    const auto end = std::chrono::steady_clock::now();
    call_latencies_ns.push_back(static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count()));
  }
  return BuildStats("CommitPut", scenario, std::move(call_latencies_ns));
}

// AbortPut: 在 steady-state resident 集合上重复执行 prepare + abort，
// 用来观察放弃未提交写入时的回滚成本。
Stats BenchmarkAbortPut(const BenchmarkScenario& scenario,
                        bool enable_timing_log) {
  auto manager = MakeInitializedManager(
      scenario.resident_keys + scenario.batch_size + 1, enable_timing_log);
  const auto resident_lengths =
      BuildLengthBatch(scenario.batch_size, kShortLength);
  const std::size_t batch_count =
      BatchCount(scenario.resident_keys, scenario.batch_size);
  PopulateCommittedKeys(manager.get(), "abort_resident",
                        scenario.resident_keys, scenario.batch_size,
                        resident_lengths);

  const auto work_lengths = BuildLengthBatch(scenario.batch_size, kShortLength);

  std::vector<double> call_latencies_ns;
  call_latencies_ns.reserve(batch_count);
  for (std::size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
    const auto work_keys = BuildKeyBatch("abort_work",
                                         batch_index * scenario.batch_size,
                                         scenario.batch_size);
    PutContext put_context;
    PutAccessDesc put_desc;
    CheckOk(manager->PreparePut(work_keys, work_lengths, &put_context, &put_desc),
            "PreparePut(before AbortPut)");
    CheckAccessDesc(put_desc, work_keys, kShortLength, "PreparePut(before AbortPut)");

    const auto start = std::chrono::steady_clock::now();
    CheckOk(manager->AbortPut(put_context), "AbortPut");
    const auto end = std::chrono::steady_clock::now();
    call_latencies_ns.push_back(static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count()));
  }
  return BuildStats("AbortPut", scenario, std::move(call_latencies_ns));
}

// LockKeys: manager 中提前常驻 resident_keys 条记录，随后按 batch_size 分批 lock/unlock。
Stats BenchmarkLockKeys(const BenchmarkScenario& scenario,
                        bool enable_timing_log) {
  const auto lengths = BuildLengthBatch(scenario.batch_size, kShortLength);
  const std::size_t batch_count =
      BatchCount(scenario.resident_keys, scenario.batch_size);

  std::vector<double> call_latencies_ns;
  call_latencies_ns.reserve(batch_count);
  auto manager =
      MakeInitializedManager(scenario.resident_keys + 1, enable_timing_log);
  PopulateCommittedKeys(manager.get(), "lock_resident", scenario.resident_keys,
                        scenario.batch_size, lengths,
                        "ConcurrentLockLookup.populate");

  for (std::size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
    const auto keys =
        BuildKeyBatch("lock_resident", batch_index * scenario.batch_size,
                      scenario.batch_size);
    LockContext lock_context;
    const auto start = std::chrono::steady_clock::now();
    CheckOk(manager->LockKeys(keys, MetaManagerLockMode::kStableReadAllowAppend,
                              &lock_context),
            "LockKeys");
    const auto end = std::chrono::steady_clock::now();
    CheckOk(manager->UnlockKeys(lock_context), "UnlockKeys(after LockKeys)");
    call_latencies_ns.push_back(static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count()));
  }
  return BuildStats("LockKeys", scenario, std::move(call_latencies_ns));
}

// BatchLookup: 对一批已经 lock 的 key 做查询；输出同时校验 entries[i] 语义是否正确。
Stats BenchmarkBatchLookup(const BenchmarkScenario& scenario,
                           bool enable_timing_log) {
  const auto lengths = BuildLengthBatch(scenario.batch_size, kShortLength);
  const std::size_t batch_count =
      BatchCount(scenario.resident_keys, scenario.batch_size);

  std::vector<double> call_latencies_ns;
  call_latencies_ns.reserve(batch_count);
  auto manager =
      MakeInitializedManager(scenario.resident_keys + 1, enable_timing_log);
  PopulateCommittedKeys(manager.get(), "lookup_resident",
                        scenario.resident_keys, scenario.batch_size, lengths);

  for (std::size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
    const auto keys =
        BuildKeyBatch("lookup_resident", batch_index * scenario.batch_size,
                      scenario.batch_size);
    LockContext lock_context;
    FlatAccessDesc access_desc;
    CheckOk(manager->LockKeys(keys, MetaManagerLockMode::kStableReadAllowAppend,
                              &lock_context),
            "LockKeys(before BatchLookup)");
    const auto start = std::chrono::steady_clock::now();
    CheckOk(manager->BatchLookup(keys, lock_context, &access_desc),
            "BatchLookup");
    const auto end = std::chrono::steady_clock::now();
    CheckAccessDesc(access_desc, keys, kShortLength, "BatchLookup");
    CheckOk(manager->UnlockKeys(lock_context), "UnlockKeys(after BatchLookup)");
    call_latencies_ns.push_back(static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count()));
  }
  return BuildStats("BatchLookup", scenario, std::move(call_latencies_ns));
}

// ConcurrentLockLookup: 多个线程共享同一个 manager，
// 每轮并发对 thread_count 个互不重叠的 batch 执行 LockKeys + BatchLookup。
// 计时的是整轮并发 wall time，因此 avg_per_key_ns 会按
// batch_size * thread_count 做均摊。
Stats BenchmarkConcurrentPreparePutShort(const BenchmarkScenario& scenario,
                                         bool enable_timing_log) {
  if (scenario.thread_count <= 1) {
    FailBenchmark("BenchmarkConcurrentPreparePutShort",
                  "thread_count must be greater than 1");
  }

  const auto resident_lengths =
      BuildLengthBatch(scenario.batch_size, kShortLength);
  const auto work_lengths = BuildLengthBatch(scenario.batch_size, kShortLength);
  const std::size_t round_count = ConcurrentRounds(scenario);

  auto manager = MakeInitializedManager(
      scenario.resident_keys + scenario.batch_size * scenario.thread_count + 1,
      enable_timing_log);
  PopulateCommittedKeys(manager.get(), "concurrent_prepare_short_resident",
                        scenario.resident_keys, scenario.batch_size,
                        resident_lengths,
                        "ConcurrentPreparePutShort.populate");

  std::vector<double> call_latencies_ns;
  call_latencies_ns.reserve(round_count);

  for (std::size_t round = 0; round < round_count; ++round) {
    ProgressLog("ConcurrentPreparePutShort round=" +
                std::to_string(round + 1) + "/" + std::to_string(round_count) +
                " resident_keys=" + std::to_string(scenario.resident_keys) +
                " batch_size=" + std::to_string(scenario.batch_size) +
                " thread_count=" + std::to_string(scenario.thread_count));

    std::vector<PutContext> put_contexts(scenario.thread_count);
    std::vector<PutAccessDesc> put_descs(scenario.thread_count);
    const auto round_key_batches =
        BuildRoundKeyBatches("concurrent_prepare_short_work", scenario, round);
    std::string first_error;

    call_latencies_ns.push_back(RunConcurrentRound(
        scenario.thread_count,
        [&](std::size_t t) -> std::string {
          const Status prepare_status = manager->PreparePut(
              round_key_batches[t], work_lengths, &put_contexts[t],
              &put_descs[t]);
          if (!prepare_status.ok()) {
            return "ConcurrentPreparePut(short).PreparePut status=" +
                   prepare_status.message();
          }
          return "";
        },
        &first_error));

    if (!first_error.empty()) {
      FailBenchmark("ConcurrentPreparePut(short)", first_error);
    }

    for (std::size_t t = 0; t < scenario.thread_count; ++t) {
      CheckAccessDesc(put_descs[t], round_key_batches[t], kShortLength,
                      "ConcurrentPreparePut(short)");
      CheckOk(manager->AbortPut(put_contexts[t]),
              "AbortPut(after ConcurrentPreparePut short)");
    }
  }

  return BuildStats("ConcurrentPreparePut(short)", scenario,
                    std::move(call_latencies_ns));
}

Stats BenchmarkConcurrentPreparePutLong(const BenchmarkScenario& scenario,
                                        bool enable_timing_log) {
  if (scenario.thread_count <= 1) {
    FailBenchmark("BenchmarkConcurrentPreparePutLong",
                  "thread_count must be greater than 1");
  }

  const auto resident_lengths =
      BuildLengthBatch(scenario.batch_size, kShortLength);
  const auto work_lengths = BuildLengthBatch(scenario.batch_size, kLongLength);
  const std::size_t round_count = ConcurrentRounds(scenario);

  auto manager = MakeInitializedManager(
      scenario.resident_keys + scenario.batch_size * scenario.thread_count + 1,
      enable_timing_log);
  PopulateCommittedKeys(manager.get(), "concurrent_prepare_long_resident",
                        scenario.resident_keys, scenario.batch_size,
                        resident_lengths,
                        "ConcurrentPreparePutLong.populate");

  std::vector<double> call_latencies_ns;
  call_latencies_ns.reserve(round_count);

  for (std::size_t round = 0; round < round_count; ++round) {
    ProgressLog("ConcurrentPreparePutLong round=" +
                std::to_string(round + 1) + "/" + std::to_string(round_count) +
                " resident_keys=" + std::to_string(scenario.resident_keys) +
                " batch_size=" + std::to_string(scenario.batch_size) +
                " thread_count=" + std::to_string(scenario.thread_count));

    std::vector<PutContext> put_contexts(scenario.thread_count);
    std::vector<PutAccessDesc> put_descs(scenario.thread_count);
    const auto round_key_batches =
        BuildRoundKeyBatches("concurrent_prepare_long_work", scenario, round);
    std::string first_error;

    call_latencies_ns.push_back(RunConcurrentRound(
        scenario.thread_count,
        [&](std::size_t t) -> std::string {
          const Status prepare_status = manager->PreparePut(
              round_key_batches[t], work_lengths, &put_contexts[t],
              &put_descs[t]);
          if (!prepare_status.ok()) {
            return "ConcurrentPreparePut(long).PreparePut status=" +
                   prepare_status.message();
          }
          return "";
        },
        &first_error));

    if (!first_error.empty()) {
      FailBenchmark("ConcurrentPreparePut(long)", first_error);
    }

    for (std::size_t t = 0; t < scenario.thread_count; ++t) {
      CheckAccessDesc(put_descs[t], round_key_batches[t], kLongLength,
                      "ConcurrentPreparePut(long)");
      CheckOk(manager->AbortPut(put_contexts[t]),
              "AbortPut(after ConcurrentPreparePut long)");
    }
  }

  return BuildStats("ConcurrentPreparePut(long)", scenario,
                    std::move(call_latencies_ns));
}

Stats BenchmarkConcurrentLockKeys(const BenchmarkScenario& scenario,
                                  bool enable_timing_log) {
  if (scenario.thread_count <= 1) {
    FailBenchmark("BenchmarkConcurrentLockKeys",
                  "thread_count must be greater than 1");
  }

  const auto lengths = BuildLengthBatch(scenario.batch_size, kShortLength);
  const std::size_t round_count = ConcurrentRounds(scenario);

  auto manager =
      MakeInitializedManager(scenario.resident_keys + 1, enable_timing_log);
  PopulateCommittedKeys(manager.get(), "concurrent_lock_resident",
                        scenario.resident_keys, scenario.batch_size, lengths,
                        "ConcurrentLockKeys.populate");

  std::vector<double> call_latencies_ns;
  call_latencies_ns.reserve(round_count);

  for (std::size_t round = 0; round < round_count; ++round) {
    ProgressLog("ConcurrentLockKeys round=" + std::to_string(round + 1) +
                "/" + std::to_string(round_count) +
                " resident_keys=" + std::to_string(scenario.resident_keys) +
                " batch_size=" + std::to_string(scenario.batch_size) +
                " thread_count=" + std::to_string(scenario.thread_count));

    std::vector<LockContext> lock_contexts(scenario.thread_count);
    const auto round_key_batches =
        BuildRoundKeyBatches("concurrent_lock_resident", scenario, round);
    std::string first_error;

    call_latencies_ns.push_back(RunConcurrentRound(
        scenario.thread_count,
        [&](std::size_t t) -> std::string {
          const Status lock_status = manager->LockKeys(
              round_key_batches[t], MetaManagerLockMode::kStableReadAllowAppend,
              &lock_contexts[t]);
          if (!lock_status.ok()) {
            return "ConcurrentLockKeys.LockKeys status=" +
                   lock_status.message();
          }
          return "";
        },
        &first_error));

    if (!first_error.empty()) {
      FailBenchmark("ConcurrentLockKeys", first_error);
    }

    for (std::size_t t = 0; t < scenario.thread_count; ++t) {
      CheckOk(manager->UnlockKeys(lock_contexts[t]),
              "UnlockKeys(after ConcurrentLockKeys)");
    }
  }

  return BuildStats("ConcurrentLockKeys", scenario,
                    std::move(call_latencies_ns));
}

Stats BenchmarkConcurrentBatchLookup(const BenchmarkScenario& scenario,
                                     bool enable_timing_log) {
  if (scenario.thread_count <= 1) {
    FailBenchmark("BenchmarkConcurrentBatchLookup",
                  "thread_count must be greater than 1");
  }

  const auto lengths = BuildLengthBatch(scenario.batch_size, kShortLength);
  const std::size_t round_count = ConcurrentRounds(scenario);

  auto manager =
      MakeInitializedManager(scenario.resident_keys + 1, enable_timing_log);
  PopulateCommittedKeys(manager.get(), "concurrent_lookup_resident",
                        scenario.resident_keys, scenario.batch_size, lengths,
                        "ConcurrentBatchLookup.populate");

  std::vector<double> call_latencies_ns;
  call_latencies_ns.reserve(round_count);

  for (std::size_t round = 0; round < round_count; ++round) {
    ProgressLog("ConcurrentBatchLookup round=" + std::to_string(round + 1) +
                "/" + std::to_string(round_count) +
                " resident_keys=" + std::to_string(scenario.resident_keys) +
                " batch_size=" + std::to_string(scenario.batch_size) +
                " thread_count=" + std::to_string(scenario.thread_count));

    const auto round_key_batches =
        BuildRoundKeyBatches("concurrent_lookup_resident", scenario, round);
    std::vector<LockContext> lock_contexts(scenario.thread_count);
    std::vector<FlatAccessDesc> access_descs(scenario.thread_count);
    for (std::size_t t = 0; t < scenario.thread_count; ++t) {
      CheckOk(manager->LockKeys(round_key_batches[t],
                                MetaManagerLockMode::kStableReadAllowAppend,
                                &lock_contexts[t]),
              "LockKeys(before ConcurrentBatchLookup)");
    }

    std::string first_error;
    call_latencies_ns.push_back(RunConcurrentRound(
        scenario.thread_count,
        [&](std::size_t t) -> std::string {
          const Status lookup_status = manager->BatchLookup(
              round_key_batches[t], lock_contexts[t], &access_descs[t]);
          if (!lookup_status.ok()) {
            return "ConcurrentBatchLookup.BatchLookup status=" +
                   lookup_status.message();
          }
          return "";
        },
        &first_error));

    if (!first_error.empty()) {
      FailBenchmark("ConcurrentBatchLookup", first_error);
    }

    for (std::size_t t = 0; t < scenario.thread_count; ++t) {
      std::string validation_error;
      if (!ValidateAccessDesc(access_descs[t], round_key_batches[t], kShortLength,
                              &validation_error)) {
        FailBenchmark("ConcurrentBatchLookup", validation_error);
      }
      CheckOk(manager->UnlockKeys(lock_contexts[t]),
              "UnlockKeys(after ConcurrentBatchLookup)");
    }
  }

  return BuildStats("ConcurrentBatchLookup", scenario,
                    std::move(call_latencies_ns));
}

Stats BenchmarkConcurrentLockLookup(const BenchmarkScenario& scenario,
                                    bool enable_timing_log) {
  if (scenario.thread_count <= 1) {
    FailBenchmark("BenchmarkConcurrentLockLookup",
                  "thread_count must be greater than 1");
  }

  const auto lengths = BuildLengthBatch(scenario.batch_size, kShortLength);
  const std::size_t round_count = ConcurrentRounds(scenario);

  auto manager =
      MakeInitializedManager(scenario.resident_keys + 1, enable_timing_log);
  PopulateCommittedKeys(manager.get(), "concurrent_lookup_resident",
                        scenario.resident_keys, scenario.batch_size, lengths,
                        "ConcurrentLockLookup.populate");

  std::vector<double> call_latencies_ns;
  call_latencies_ns.reserve(round_count);

  for (std::size_t round = 0; round < round_count; ++round) {
    ProgressLog("ConcurrentLockLookup round=" + std::to_string(round + 1) +
                "/" + std::to_string(round_count) +
                " resident_keys=" + std::to_string(scenario.resident_keys) +
                " batch_size=" + std::to_string(scenario.batch_size) +
                " thread_count=" + std::to_string(scenario.thread_count));

    std::vector<LockContext> lock_contexts(scenario.thread_count);
    std::vector<FlatAccessDesc> access_descs(scenario.thread_count);
    const auto round_key_batches =
        BuildRoundKeyBatches("concurrent_lookup_resident", scenario, round);
    std::string first_error;

    call_latencies_ns.push_back(RunConcurrentRound(
        scenario.thread_count,
        [&](std::size_t t) -> std::string {
          const Status lock_status = manager->LockKeys(
              round_key_batches[t], MetaManagerLockMode::kStableReadAllowAppend,
              &lock_contexts[t]);
          if (!lock_status.ok()) {
            return "ConcurrentLockLookup.LockKeys status=" +
                   lock_status.message();
          }

          const Status lookup_status = manager->BatchLookup(
              round_key_batches[t], lock_contexts[t], &access_descs[t]);
          if (!lookup_status.ok()) {
            return "ConcurrentLockLookup.BatchLookup status=" +
                   lookup_status.message();
          }
          return "";
        },
        &first_error));

    if (!first_error.empty()) {
      FailBenchmark("ConcurrentLockLookup", first_error);
    }

    for (std::size_t t = 0; t < scenario.thread_count; ++t) {
      std::string validation_error;
      if (!ValidateAccessDesc(access_descs[t], round_key_batches[t], kShortLength,
                              &validation_error)) {
        FailBenchmark("ConcurrentLockLookup", validation_error);
      }
      CheckOk(manager->UnlockKeys(lock_contexts[t]),
              "UnlockKeys(after ConcurrentLockLookup)");
    }
  }

  return BuildStats("ConcurrentLockLookup", scenario,
                    std::move(call_latencies_ns));
}

// UnlockKeys: 先建立 lock，再测量解锁本身的耗时。
Stats BenchmarkUnlockKeys(const BenchmarkScenario& scenario,
                          bool enable_timing_log) {
  const auto lengths = BuildLengthBatch(scenario.batch_size, kShortLength);
  const std::size_t batch_count =
      BatchCount(scenario.resident_keys, scenario.batch_size);

  std::vector<double> call_latencies_ns;
  call_latencies_ns.reserve(batch_count);
  auto manager =
      MakeInitializedManager(scenario.resident_keys + 1, enable_timing_log);
  PopulateCommittedKeys(manager.get(), "unlock_resident",
                        scenario.resident_keys, scenario.batch_size, lengths);

  for (std::size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
    const auto keys =
        BuildKeyBatch("unlock_resident", batch_index * scenario.batch_size,
                      scenario.batch_size);
    LockContext lock_context;
    CheckOk(manager->LockKeys(keys, MetaManagerLockMode::kStableReadAllowAppend,
                              &lock_context),
            "LockKeys(before UnlockKeys)");
    const auto start = std::chrono::steady_clock::now();
    CheckOk(manager->UnlockKeys(lock_context), "UnlockKeys");
    const auto end = std::chrono::steady_clock::now();
    call_latencies_ns.push_back(static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count()));
  }
  return BuildStats("UnlockKeys", scenario, std::move(call_latencies_ns));
}

// BatchFree: 每轮把一整套 resident 数据按 batch_size 逐批释放，模拟回收阶段的开销。
Stats BenchmarkBatchFree(const BenchmarkScenario& scenario,
                         bool enable_timing_log) {
  const auto lengths = BuildLengthBatch(scenario.batch_size, kShortLength);
  const std::size_t batch_count =
      BatchCount(scenario.resident_keys, scenario.batch_size);

  std::vector<double> call_latencies_ns;
  call_latencies_ns.reserve(batch_count);
  auto manager =
      MakeInitializedManager(scenario.resident_keys + 1, enable_timing_log);
  PopulateCommittedKeys(manager.get(), "free_resident",
                        scenario.resident_keys, scenario.batch_size, lengths);

  for (std::size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
    const auto keys =
        BuildKeyBatch("free_resident", batch_index * scenario.batch_size,
                      scenario.batch_size);
    const auto start = std::chrono::steady_clock::now();
    CheckOk(manager->BatchFree(keys), "BatchFree");
    const auto end = std::chrono::steady_clock::now();
    call_latencies_ns.push_back(static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count()));
  }
  return BuildStats("BatchFree", scenario, std::move(call_latencies_ns));
}

// EndToEnd(short): 测量一批短记录从地址预留到最终回收的完整控制面链路：
// PreparePut -> CommitPut -> LockKeys -> BatchLookup -> UnlockKeys -> BatchFree。
Stats BenchmarkEndToEndShort(const BenchmarkScenario& scenario,
                             bool enable_timing_log) {
  auto manager = MakeInitializedManager(
      scenario.resident_keys + scenario.batch_size + 1, enable_timing_log);
  const auto resident_lengths =
      BuildLengthBatch(scenario.batch_size, kShortLength);
  const std::size_t batch_count =
      BatchCount(scenario.resident_keys, scenario.batch_size);
  PopulateCommittedKeys(manager.get(), "e2e_resident", scenario.resident_keys,
                        scenario.batch_size, resident_lengths);

  const auto work_lengths = BuildLengthBatch(scenario.batch_size, kShortLength);

  std::vector<double> call_latencies_ns;
  call_latencies_ns.reserve(batch_count);
  for (std::size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
    const auto work_keys = BuildKeyBatch("e2e_short_work",
                                         batch_index * scenario.batch_size,
                                         scenario.batch_size);
    PutContext put_context;
    PutAccessDesc put_desc;
    LockContext lock_context;
    FlatAccessDesc access_desc;

    const auto start = std::chrono::steady_clock::now();
    CheckOk(manager->PreparePut(work_keys, work_lengths, &put_context, &put_desc),
            "EndToEnd.PreparePut");
    CheckOk(manager->CommitPut(put_context), "EndToEnd.CommitPut");
    CheckOk(manager->LockKeys(work_keys, MetaManagerLockMode::kStableReadAllowAppend,
                              &lock_context),
            "EndToEnd.LockKeys");
    CheckOk(manager->BatchLookup(work_keys, lock_context, &access_desc),
            "EndToEnd.BatchLookup");
    CheckOk(manager->UnlockKeys(lock_context), "EndToEnd.UnlockKeys");
    CheckOk(manager->BatchFree(work_keys), "EndToEnd.BatchFree");
    const auto end = std::chrono::steady_clock::now();

    CheckAccessDesc(put_desc, work_keys, kShortLength, "EndToEnd.PreparePut");
    CheckAccessDesc(access_desc, work_keys, kShortLength, "EndToEnd.BatchLookup");
    call_latencies_ns.push_back(static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count()));
  }
  return BuildStats("EndToEnd(short)", scenario,
                    std::move(call_latencies_ns));
}

// EndToEnd(long): 与 short 端到端流程相同，但写入长度落到 long slot。
Stats BenchmarkEndToEndLong(const BenchmarkScenario& scenario,
                            bool enable_timing_log) {
  auto manager = MakeInitializedManager(
      scenario.resident_keys + scenario.batch_size + 1, enable_timing_log);
  const auto resident_lengths =
      BuildLengthBatch(scenario.batch_size, kShortLength);
  const std::size_t batch_count =
      BatchCount(scenario.resident_keys, scenario.batch_size);
  PopulateCommittedKeys(manager.get(), "e2e_long_resident",
                        scenario.resident_keys, scenario.batch_size,
                        resident_lengths);

  const auto work_lengths = BuildLengthBatch(scenario.batch_size, kLongLength);

  std::vector<double> call_latencies_ns;
  call_latencies_ns.reserve(batch_count);
  for (std::size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
    const auto work_keys = BuildKeyBatch("e2e_long_work",
                                         batch_index * scenario.batch_size,
                                         scenario.batch_size);
    PutContext put_context;
    PutAccessDesc put_desc;
    LockContext lock_context;
    FlatAccessDesc access_desc;

    const auto start = std::chrono::steady_clock::now();
    CheckOk(manager->PreparePut(work_keys, work_lengths, &put_context, &put_desc),
            "EndToEndLong.PreparePut");
    CheckOk(manager->CommitPut(put_context), "EndToEndLong.CommitPut");
    CheckOk(manager->LockKeys(work_keys, MetaManagerLockMode::kStableReadAllowAppend,
                              &lock_context),
            "EndToEndLong.LockKeys");
    CheckOk(manager->BatchLookup(work_keys, lock_context, &access_desc),
            "EndToEndLong.BatchLookup");
    CheckOk(manager->UnlockKeys(lock_context), "EndToEndLong.UnlockKeys");
    CheckOk(manager->BatchFree(work_keys), "EndToEndLong.BatchFree");
    const auto end = std::chrono::steady_clock::now();

    CheckAccessDesc(put_desc, work_keys, kLongLength, "EndToEndLong.PreparePut");
    CheckAccessDesc(access_desc, work_keys, kLongLength,
                    "EndToEndLong.BatchLookup");
    call_latencies_ns.push_back(static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count()));
  }
  return BuildStats("EndToEnd(long)", scenario,
                    std::move(call_latencies_ns));
}

void PrintStatsTable(const std::vector<Stats>& stats_list) {
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "MetaManager performance benchmark (unit: ns, total_ms)\n";
  std::cout << "Scenario notes:\n";
  std::cout << "- resident_keys: committed keys already stored before the measured call\n";
  std::cout << "- batch_size: number of keys handled by one interface call\n";
  std::cout << "- thread_count: concurrent workers in one measured round; serial methods use 1\n";
  std::cout << "- call_count: serial methods use resident_keys / batch_size; concurrent methods use resident_keys / (batch_size * thread_count)\n";
  std::cout << "- avg_ns: average latency per call\n";
  std::cout << "- avg_per_batch_ns: avg_ns amortized to one batch; in concurrent methods one batch is handled by one thread\n";
  std::cout << "- avg_per_key_ns: avg_ns amortized to one key; concurrent methods amortize over all keys of one round\n";
  std::cout << "- total_ms: total measured time of this method/scenario across all measured calls\n";
  std::cout << "- concurrent methods are only emitted when explicitly enabled\n";
  std::cout << "method,resident_keys,batch_size,thread_count,call_count,total_ms,avg_ns,avg_per_batch_ns,avg_per_key_ns,min_ns,p50_ns,p95_ns,max_ns\n";
  for (const Stats& stats : stats_list) {
    std::cout << stats.method << ',' << stats.resident_keys << ','
              << stats.batch_size << ',' << stats.thread_count << ','
              << stats.call_count << ','
              << stats.total_ms << ',' << stats.avg_ns << ','
              << stats.avg_per_batch_ns << ',' << stats.avg_per_key_ns << ','
              << stats.min_ns << ','
              << stats.p50_ns << ',' << stats.p95_ns << ',' << stats.max_ns
              << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  BenchmarkOptions options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--enable-timing-log") {
      options.enable_timing_log = true;
    } else if (arg == "--include-concurrent") {
      options.include_concurrent = true;
    } else if (arg == "--concurrent-only") {
      options.include_concurrent = true;
      options.concurrent_only = true;
    } else if (arg == "--show-progress") {
      options.show_progress = true;
    }
  }

  g_show_progress = options.show_progress;

  const std::vector<BenchmarkScenario> scenarios = BuildScenarios();
  const std::vector<BenchmarkScenario> concurrent_scenarios =
      BuildConcurrentScenarios();
  std::vector<Stats> stats_list;
  stats_list.reserve(1 + scenarios.size() * 10 + concurrent_scenarios.size() * 5);
  AppendStats(&stats_list, BenchmarkRegisterRemoteRegion(options.enable_timing_log),
              options.show_progress);

  if (!options.concurrent_only) {
    for (const auto& scenario : scenarios) {
      AppendStats(&stats_list,
                  BenchmarkPreparePutShort(scenario, options.enable_timing_log),
                  options.show_progress);
      AppendStats(&stats_list,
                  BenchmarkPreparePutLong(scenario, options.enable_timing_log),
                  options.show_progress);
      AppendStats(&stats_list,
                  BenchmarkCommitPut(scenario, options.enable_timing_log),
                  options.show_progress);
      AppendStats(&stats_list,
                  BenchmarkAbortPut(scenario, options.enable_timing_log),
                  options.show_progress);
      AppendStats(&stats_list,
                  BenchmarkLockKeys(scenario, options.enable_timing_log),
                  options.show_progress);
      AppendStats(&stats_list,
                  BenchmarkBatchLookup(scenario, options.enable_timing_log),
                  options.show_progress);
      AppendStats(&stats_list,
                  BenchmarkUnlockKeys(scenario, options.enable_timing_log),
                  options.show_progress);
      AppendStats(&stats_list,
                  BenchmarkBatchFree(scenario, options.enable_timing_log),
                  options.show_progress);
      AppendStats(&stats_list,
                  BenchmarkEndToEndShort(scenario, options.enable_timing_log),
                  options.show_progress);
      AppendStats(&stats_list,
                  BenchmarkEndToEndLong(scenario, options.enable_timing_log),
                  options.show_progress);
    }
  }

  if (options.include_concurrent) {
    for (const auto& scenario : concurrent_scenarios) {
      AppendStats(&stats_list,
                  BenchmarkConcurrentPreparePutShort(
                      scenario, options.enable_timing_log),
                  options.show_progress);
      AppendStats(&stats_list,
                  BenchmarkConcurrentPreparePutLong(
                      scenario, options.enable_timing_log),
                  options.show_progress);
      AppendStats(&stats_list,
                  BenchmarkConcurrentLockKeys(scenario,
                                              options.enable_timing_log),
                  options.show_progress);
      AppendStats(&stats_list,
                  BenchmarkConcurrentBatchLookup(scenario,
                                                 options.enable_timing_log),
                  options.show_progress);
      AppendStats(&stats_list,
                  BenchmarkConcurrentLockLookup(scenario,
                                                options.enable_timing_log),
                  options.show_progress);
    }
  }

  PrintStatsTable(stats_list);
  return 0;
}
