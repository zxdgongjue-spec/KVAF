#include "kvaf/meta_manager/meta_manager.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace {

using kvaf::AccessDesc;
using kvaf::CreateInMemoryMetaManager;
using kvaf::FlatAccessDesc;
using kvaf::FlushMetaManagerTimingSummary;
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
constexpr std::size_t kWarmupRounds = 1;

struct BenchmarkScenario {
  std::size_t resident_keys{0};
  std::size_t batch_size{0};
};

struct Stats {
  std::string method;
  std::size_t resident_keys{0};
  std::size_t batch_size{0};
  std::size_t call_count{0};
  double total_ms{0.0};
  double avg_ns{0.0};
  double avg_per_key_ns{0.0};
  double min_ns{0.0};
  double p50_ns{0.0};
  double p95_ns{0.0};
  double max_ns{0.0};
};

struct BenchmarkResult {
  Stats total;
  std::vector<Stats> details;
};

struct BenchmarkOptions {
  bool enable_timing_log{false};
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
    std::uint64_t slot_count_per_class) {
  auto manager = CreateInMemoryMetaManager();
  manager->SetTimingLogEnabled(false);
  manager->SetPerformanceModeEnabled(true);
  CheckOk(manager->RegisterRemoteRegion(MakeRegion(slot_count_per_class)),
          "RegisterRemoteRegion");
  return manager;
}

std::vector<BenchmarkScenario> BuildScenarios() {
  return {
      BenchmarkScenario{128 * 1024, 2 * 1024},
      BenchmarkScenario{2 * 128 * 1024, 2 * 2 * 1024},
      BenchmarkScenario{4 * 128 * 1024, 4 * 2 * 1024},
      BenchmarkScenario{8 * 128 * 1024, 8 * 2 * 1024},
      BenchmarkScenario{16 * 128 * 1024, 16 * 2 * 1024},
  };
}

std::size_t BatchCount(std::size_t total_keys, std::size_t batch_size) {
  if (batch_size == 0 || total_keys % batch_size != 0) {
    FailBenchmark("BatchCount", "total_keys must be divisible by batch_size");
  }
  return total_keys / batch_size;
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

std::vector<std::uint64_t> BuildLengthBatch(std::size_t batch_size,
                                            std::uint64_t length) {
  return std::vector<std::uint64_t>(batch_size, length);
}

void CheckAccessDesc(const AccessDesc& access_desc,
                     const std::vector<KeyId>& keys,
                     std::uint64_t expected_length,
                     const char* step) {
  if (access_desc.keys.size() != keys.size() ||
      access_desc.addresses.size() != keys.size() ||
      access_desc.lengths.size() != keys.size() ||
      access_desc.valid.size() != keys.size()) {
    FailBenchmark(step, "returned entry count does not match batch size");
  }

  for (std::size_t i = 0; i < keys.size(); ++i) {
    if (access_desc.lengths[i] != expected_length) {
      FailBenchmark(step, "returned length does not match the requested length");
    }
    if (access_desc.valid[i] != 1) {
      FailBenchmark(step, "returned valid flag is not set");
    }
  }
}

double ElapsedNs(const std::chrono::steady_clock::time_point& start) {
  return static_cast<double>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - start)
          .count());
}

template <typename Function>
double MeasureNs(Function&& function) {
  const auto start = std::chrono::steady_clock::now();
  std::forward<Function>(function)();
  return ElapsedNs(start);
}

Stats BuildStats(const std::string& method,
                 const BenchmarkScenario& scenario,
                 std::vector<double> call_latencies_ns) {
  if (call_latencies_ns.empty()) {
    FailBenchmark(method.c_str(), "benchmark produced no timing samples");
  }

  std::sort(call_latencies_ns.begin(), call_latencies_ns.end());
  const double total_ns =
      std::accumulate(call_latencies_ns.begin(), call_latencies_ns.end(), 0.0);

  Stats stats;
  stats.method = method;
  stats.resident_keys = scenario.resident_keys;
  stats.batch_size = scenario.batch_size;
  stats.call_count = call_latencies_ns.size();
  stats.total_ms = total_ns / 1000000.0;
  stats.avg_ns = total_ns / static_cast<double>(stats.call_count);
  stats.avg_per_key_ns =
      stats.avg_ns / static_cast<double>(scenario.batch_size);
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
              << " batch_size=" << stats.batch_size << '\n';
  }
  stats_list->push_back(stats);
}

void AppendStats(std::vector<Stats>* stats_list,
                 const BenchmarkResult& result,
                 bool show_progress) {
  AppendStats(stats_list, result.total, show_progress);
  stats_list->insert(stats_list->end(), result.details.begin(),
                     result.details.end());
}

void ProgressLog(const std::string& message) {
  if (g_show_progress) {
    std::cerr << "[BenchmarkProgress] " << message << '\n';
  }
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

void WarmupPut(MetaManager* manager,
               const std::string& key_prefix,
               const std::vector<std::uint64_t>& lengths,
               bool commit,
               const char* benchmark_name) {
  if (manager == nullptr) {
    FailBenchmark(benchmark_name, "warm-up manager is null");
  }

  for (std::size_t round = 0; round < kWarmupRounds; ++round) {
    ProgressLog(std::string(benchmark_name) + " warm-up round=" +
                std::to_string(round + 1) + "/" +
                std::to_string(kWarmupRounds));
    const auto keys = BuildKeyBatch(key_prefix, round * lengths.size(),
                                    lengths.size());
    PutContext put_context;
    PutAccessDesc put_desc;
    CheckOk(manager->PreparePut(keys, lengths, &put_context, &put_desc),
            "PreparePut(single-thread warm-up)");
    CheckAccessDesc(put_desc, keys, lengths.front(),
                    "PreparePut(single-thread warm-up)");
    if (commit) {
      CheckOk(manager->CommitPut(put_context),
              "CommitPut(single-thread warm-up)");
    } else {
      CheckOk(manager->AbortPut(put_context),
              "AbortPut(single-thread warm-up)");
    }
  }
}

void WarmupLockLookup(MetaManager* manager,
                      const std::string& key_prefix,
                      std::size_t batch_size,
                      std::uint64_t expected_length,
                      bool lookup,
                      const char* benchmark_name) {
  if (manager == nullptr) {
    FailBenchmark(benchmark_name, "warm-up manager is null");
  }

  for (std::size_t round = 0; round < kWarmupRounds; ++round) {
    ProgressLog(std::string(benchmark_name) + " warm-up round=" +
                std::to_string(round + 1) + "/" +
                std::to_string(kWarmupRounds));
    const auto keys = BuildKeyBatch(key_prefix, round * batch_size, batch_size);
    LockContext lock_context;
    CheckOk(manager->LockKeys(keys, MetaManagerLockMode::kStableReadAllowAppend,
                              &lock_context),
            "LockKeys(single-thread warm-up)");
    if (lookup) {
      FlatAccessDesc access_desc;
      CheckOk(manager->BatchLookup(keys, lock_context, &access_desc),
              "BatchLookup(single-thread warm-up)");
      CheckAccessDesc(access_desc, keys, expected_length,
                      "BatchLookup(single-thread warm-up)");
    }
    CheckOk(manager->UnlockKeys(lock_context),
            "UnlockKeys(after single-thread warm-up)");
  }
}

void WarmupBatchFree(MetaManager* manager,
                     const std::string& key_prefix,
                     const std::vector<std::uint64_t>& lengths,
                     const char* benchmark_name) {
  if (manager == nullptr) {
    FailBenchmark(benchmark_name, "warm-up manager is null");
  }

  for (std::size_t round = 0; round < kWarmupRounds; ++round) {
    ProgressLog(std::string(benchmark_name) + " warm-up round=" +
                std::to_string(round + 1) + "/" +
                std::to_string(kWarmupRounds));
    const auto keys = BuildKeyBatch(key_prefix, round * lengths.size(),
                                    lengths.size());
    PutContext put_context;
    PutAccessDesc put_desc;
    CheckOk(manager->PreparePut(keys, lengths, &put_context, &put_desc),
            "PreparePut(before single-thread BatchFree warm-up)");
    CheckOk(manager->CommitPut(put_context),
            "CommitPut(before single-thread BatchFree warm-up)");
    CheckOk(manager->BatchFree(keys), "BatchFree(single-thread warm-up)");
  }
}

void WarmupEndToEnd(MetaManager* manager,
                    const std::string& key_prefix,
                    const std::vector<std::uint64_t>& lengths,
                    std::uint64_t expected_length,
                    const char* benchmark_name) {
  if (manager == nullptr) {
    FailBenchmark(benchmark_name, "warm-up manager is null");
  }

  for (std::size_t round = 0; round < kWarmupRounds; ++round) {
    ProgressLog(std::string(benchmark_name) + " warm-up round=" +
                std::to_string(round + 1) + "/" +
                std::to_string(kWarmupRounds));
    const auto keys = BuildKeyBatch(key_prefix, round * lengths.size(),
                                    lengths.size());
    PutContext put_context;
    PutAccessDesc put_desc;
    CheckOk(manager->PreparePut(keys, lengths, &put_context, &put_desc),
            "PreparePut(single-thread end-to-end warm-up)");
    CheckOk(manager->CommitPut(put_context),
            "CommitPut(single-thread end-to-end warm-up)");

    LockContext lock_context;
    FlatAccessDesc access_desc;
    CheckOk(manager->LockKeys(keys, MetaManagerLockMode::kStableReadAllowAppend,
                              &lock_context),
            "LockKeys(single-thread end-to-end warm-up)");
    CheckOk(manager->BatchLookup(keys, lock_context, &access_desc),
            "BatchLookup(single-thread end-to-end warm-up)");
    CheckAccessDesc(access_desc, keys, expected_length,
                    "BatchLookup(single-thread end-to-end warm-up)");
    CheckOk(manager->UnlockKeys(lock_context),
            "UnlockKeys(single-thread end-to-end warm-up)");
    CheckOk(manager->BatchFree(keys),
            "BatchFree(single-thread end-to-end warm-up)");
  }
}

Stats BenchmarkRegisterRemoteRegion(bool enable_timing_log) {
  auto warmup_manager = CreateInMemoryMetaManager();
  CheckOk(warmup_manager->RegisterRemoteRegion(MakeRegion(1)),
          "RegisterRemoteRegion(single-thread warm-up)");

  std::vector<double> call_latencies_ns;
  call_latencies_ns.reserve(1);
  auto manager = CreateInMemoryMetaManager();
  manager->SetTimingLogEnabled(enable_timing_log);
  call_latencies_ns.push_back(MeasureNs([&]() {
    CheckOk(manager->RegisterRemoteRegion(MakeRegion(1)),
            "RegisterRemoteRegion");
  }));

  return BuildStats("RegisterRemoteRegion", BenchmarkScenario{1, 1},
                    std::move(call_latencies_ns));
}

BenchmarkResult BenchmarkPreparePutShort(const BenchmarkScenario& scenario,
                                         bool enable_timing_log) {
  auto manager = MakeInitializedManager(
      scenario.resident_keys + scenario.batch_size + 1);
  const auto resident_lengths =
      BuildLengthBatch(scenario.batch_size, kShortLength);
  const std::size_t batch_count =
      BatchCount(scenario.resident_keys, scenario.batch_size);
  PopulateCommittedKeys(manager.get(), "prepare_short_resident",
                        scenario.resident_keys, scenario.batch_size,
                        resident_lengths);

  const auto work_lengths = BuildLengthBatch(scenario.batch_size, kShortLength);
  WarmupPut(manager.get(), "warmup_prepare_short", work_lengths, false,
            "PreparePut(short)");
  manager->SetTimingLogEnabled(enable_timing_log);

  std::vector<double> prepare_samples;
  std::vector<double> build_key_samples;
  std::vector<double> check_access_samples;
  std::vector<double> abort_samples;
  prepare_samples.reserve(batch_count);
  build_key_samples.reserve(batch_count);
  check_access_samples.reserve(batch_count);
  abort_samples.reserve(batch_count);
  for (std::size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
    std::vector<KeyId> work_keys;
    build_key_samples.push_back(MeasureNs([&]() {
      work_keys = BuildKeyBatch("prepare_short_work",
                                batch_index * scenario.batch_size,
                                scenario.batch_size);
    }));
    PutContext put_context;
    PutAccessDesc put_desc;
    prepare_samples.push_back(MeasureNs([&]() {
      CheckOk(manager->PreparePut(work_keys, work_lengths, &put_context,
                                  &put_desc),
              "PreparePut(short)");
    }));
    check_access_samples.push_back(MeasureNs([&]() {
      CheckAccessDesc(put_desc, work_keys, kShortLength, "PreparePut(short)");
    }));
    abort_samples.push_back(MeasureNs([&]() {
      CheckOk(manager->AbortPut(put_context),
              "AbortPut(after PreparePut short)");
    }));
  }

  BenchmarkResult result;
  result.total = BuildStats("PreparePut(short)", scenario, prepare_samples);
  result.details.push_back(BuildStats("PreparePut(short).BuildKeyBatch",
                                      scenario, std::move(build_key_samples)));
  result.details.push_back(BuildStats("PreparePut(short).PreparePut", scenario,
                                      std::move(prepare_samples)));
  result.details.push_back(BuildStats("PreparePut(short).CheckAccessDesc",
                                      scenario,
                                      std::move(check_access_samples)));
  result.details.push_back(BuildStats("PreparePut(short).AbortPut", scenario,
                                      std::move(abort_samples)));
  return result;
}

BenchmarkResult BenchmarkPreparePutLong(const BenchmarkScenario& scenario,
                                        bool enable_timing_log) {
  auto manager = MakeInitializedManager(
      scenario.resident_keys + scenario.batch_size + 1);
  const auto resident_lengths =
      BuildLengthBatch(scenario.batch_size, kShortLength);
  const std::size_t batch_count =
      BatchCount(scenario.resident_keys, scenario.batch_size);
  PopulateCommittedKeys(manager.get(), "prepare_long_resident",
                        scenario.resident_keys, scenario.batch_size,
                        resident_lengths);

  const auto work_lengths = BuildLengthBatch(scenario.batch_size, kLongLength);
  WarmupPut(manager.get(), "warmup_prepare_long", work_lengths, false,
            "PreparePut(long)");
  manager->SetTimingLogEnabled(enable_timing_log);

  std::vector<double> prepare_samples;
  std::vector<double> build_key_samples;
  std::vector<double> check_access_samples;
  std::vector<double> abort_samples;
  prepare_samples.reserve(batch_count);
  build_key_samples.reserve(batch_count);
  check_access_samples.reserve(batch_count);
  abort_samples.reserve(batch_count);
  for (std::size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
    std::vector<KeyId> work_keys;
    build_key_samples.push_back(MeasureNs([&]() {
      work_keys = BuildKeyBatch("prepare_long_work",
                                batch_index * scenario.batch_size,
                                scenario.batch_size);
    }));
    PutContext put_context;
    PutAccessDesc put_desc;
    prepare_samples.push_back(MeasureNs([&]() {
      CheckOk(manager->PreparePut(work_keys, work_lengths, &put_context,
                                  &put_desc),
              "PreparePut(long)");
    }));
    check_access_samples.push_back(MeasureNs([&]() {
      CheckAccessDesc(put_desc, work_keys, kLongLength, "PreparePut(long)");
    }));
    abort_samples.push_back(MeasureNs([&]() {
      CheckOk(manager->AbortPut(put_context),
              "AbortPut(after PreparePut long)");
    }));
  }

  BenchmarkResult result;
  result.total = BuildStats("PreparePut(long)", scenario, prepare_samples);
  result.details.push_back(BuildStats("PreparePut(long).BuildKeyBatch",
                                      scenario, std::move(build_key_samples)));
  result.details.push_back(BuildStats("PreparePut(long).PreparePut", scenario,
                                      std::move(prepare_samples)));
  result.details.push_back(BuildStats("PreparePut(long).CheckAccessDesc",
                                      scenario,
                                      std::move(check_access_samples)));
  result.details.push_back(BuildStats("PreparePut(long).AbortPut", scenario,
                                      std::move(abort_samples)));
  return result;
}

Stats BenchmarkCommitPut(const BenchmarkScenario& scenario,
                         bool enable_timing_log) {
  const auto lengths = BuildLengthBatch(scenario.batch_size, kShortLength);
  const std::size_t batch_count =
      BatchCount(scenario.resident_keys, scenario.batch_size);
  auto manager = MakeInitializedManager(
      scenario.resident_keys + scenario.resident_keys + scenario.batch_size + 1);
  PopulateCommittedKeys(manager.get(), "commit_resident",
                        scenario.resident_keys, scenario.batch_size, lengths);
  WarmupPut(manager.get(), "warmup_commit", lengths, true, "CommitPut");
  manager->SetTimingLogEnabled(enable_timing_log);

  std::vector<double> call_latencies_ns;
  call_latencies_ns.reserve(batch_count);
  for (std::size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
    const auto keys =
        BuildKeyBatch("commit_work", batch_index * scenario.batch_size,
                      scenario.batch_size);
    PutContext put_context;
    PutAccessDesc put_desc;
    CheckOk(manager->PreparePut(keys, lengths, &put_context, &put_desc),
            "PreparePut(before CommitPut)");
    CheckAccessDesc(put_desc, keys, kShortLength,
                    "PreparePut(before CommitPut)");
    call_latencies_ns.push_back(MeasureNs([&]() {
      CheckOk(manager->CommitPut(put_context), "CommitPut");
    }));
  }
  return BuildStats("CommitPut", scenario, std::move(call_latencies_ns));
}

Stats BenchmarkAbortPut(const BenchmarkScenario& scenario,
                        bool enable_timing_log) {
  auto manager = MakeInitializedManager(
      scenario.resident_keys + scenario.batch_size + 1);
  const auto resident_lengths =
      BuildLengthBatch(scenario.batch_size, kShortLength);
  const std::size_t batch_count =
      BatchCount(scenario.resident_keys, scenario.batch_size);
  PopulateCommittedKeys(manager.get(), "abort_resident",
                        scenario.resident_keys, scenario.batch_size,
                        resident_lengths);
  const auto work_lengths = BuildLengthBatch(scenario.batch_size, kShortLength);
  WarmupPut(manager.get(), "warmup_abort", work_lengths, false, "AbortPut");
  manager->SetTimingLogEnabled(enable_timing_log);

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
    CheckAccessDesc(put_desc, work_keys, kShortLength,
                    "PreparePut(before AbortPut)");
    call_latencies_ns.push_back(MeasureNs([&]() {
      CheckOk(manager->AbortPut(put_context), "AbortPut");
    }));
  }
  return BuildStats("AbortPut", scenario, std::move(call_latencies_ns));
}

BenchmarkResult BenchmarkLockKeys(const BenchmarkScenario& scenario,
                                  bool enable_timing_log) {
  const auto lengths = BuildLengthBatch(scenario.batch_size, kShortLength);
  const std::size_t batch_count =
      BatchCount(scenario.resident_keys, scenario.batch_size);
  auto manager =
      MakeInitializedManager(scenario.resident_keys + 1);
  PopulateCommittedKeys(manager.get(), "lock_resident", scenario.resident_keys,
                        scenario.batch_size, lengths, "LockKeys.populate");
  WarmupLockLookup(manager.get(), "lock_resident", scenario.batch_size,
                   kShortLength, false, "LockKeys");
  manager->SetTimingLogEnabled(enable_timing_log);

  std::vector<double> lock_samples;
  std::vector<double> build_key_samples;
  std::vector<double> unlock_samples;
  lock_samples.reserve(batch_count);
  build_key_samples.reserve(batch_count);
  unlock_samples.reserve(batch_count);
  for (std::size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
    std::vector<KeyId> keys;
    build_key_samples.push_back(MeasureNs([&]() {
      keys = BuildKeyBatch("lock_resident", batch_index * scenario.batch_size,
                           scenario.batch_size);
    }));
    LockContext lock_context;
    lock_samples.push_back(MeasureNs([&]() {
      CheckOk(manager->LockKeys(keys, MetaManagerLockMode::kStableReadAllowAppend,
                                &lock_context),
              "LockKeys");
    }));
    unlock_samples.push_back(MeasureNs([&]() {
      CheckOk(manager->UnlockKeys(lock_context),
              "UnlockKeys(after LockKeys)");
    }));
  }

  BenchmarkResult result;
  result.total = BuildStats("LockKeys", scenario, lock_samples);
  result.details.push_back(BuildStats("LockKeys.BuildKeyBatch", scenario,
                                      std::move(build_key_samples)));
  result.details.push_back(BuildStats("LockKeys.LockKeys", scenario,
                                      std::move(lock_samples)));
  result.details.push_back(BuildStats("LockKeys.UnlockKeys", scenario,
                                      std::move(unlock_samples)));
  return result;
}

Stats BenchmarkBatchLookup(const BenchmarkScenario& scenario,
                           bool enable_timing_log) {
  const auto lengths = BuildLengthBatch(scenario.batch_size, kShortLength);
  const std::size_t batch_count =
      BatchCount(scenario.resident_keys, scenario.batch_size);
  auto manager =
      MakeInitializedManager(scenario.resident_keys + 1);
  PopulateCommittedKeys(manager.get(), "lookup_resident",
                        scenario.resident_keys, scenario.batch_size, lengths);
  WarmupLockLookup(manager.get(), "lookup_resident", scenario.batch_size,
                   kShortLength, true, "BatchLookup");
  manager->SetTimingLogEnabled(enable_timing_log);

  std::vector<double> call_latencies_ns;
  call_latencies_ns.reserve(batch_count);
  for (std::size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
    const auto keys =
        BuildKeyBatch("lookup_resident", batch_index * scenario.batch_size,
                      scenario.batch_size);
    LockContext lock_context;
    FlatAccessDesc access_desc;
    CheckOk(manager->LockKeys(keys, MetaManagerLockMode::kStableReadAllowAppend,
                              &lock_context),
            "LockKeys(before BatchLookup)");
    call_latencies_ns.push_back(MeasureNs([&]() {
      CheckOk(manager->BatchLookup(keys, lock_context, &access_desc),
              "BatchLookup");
    }));
    CheckAccessDesc(access_desc, keys, kShortLength, "BatchLookup");
    CheckOk(manager->UnlockKeys(lock_context), "UnlockKeys(after BatchLookup)");
  }
  return BuildStats("BatchLookup", scenario, std::move(call_latencies_ns));
}

Stats BenchmarkUnlockKeys(const BenchmarkScenario& scenario,
                          bool enable_timing_log) {
  const auto lengths = BuildLengthBatch(scenario.batch_size, kShortLength);
  const std::size_t batch_count =
      BatchCount(scenario.resident_keys, scenario.batch_size);
  auto manager =
      MakeInitializedManager(scenario.resident_keys + 1);
  PopulateCommittedKeys(manager.get(), "unlock_resident",
                        scenario.resident_keys, scenario.batch_size, lengths);
  WarmupLockLookup(manager.get(), "unlock_resident", scenario.batch_size,
                   kShortLength, false, "UnlockKeys");
  manager->SetTimingLogEnabled(enable_timing_log);

  std::vector<double> call_latencies_ns;
  call_latencies_ns.reserve(batch_count);
  for (std::size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
    const auto keys =
        BuildKeyBatch("unlock_resident", batch_index * scenario.batch_size,
                      scenario.batch_size);
    LockContext lock_context;
    CheckOk(manager->LockKeys(keys, MetaManagerLockMode::kStableReadAllowAppend,
                              &lock_context),
            "LockKeys(before UnlockKeys)");
    call_latencies_ns.push_back(MeasureNs([&]() {
      CheckOk(manager->UnlockKeys(lock_context), "UnlockKeys");
    }));
  }
  return BuildStats("UnlockKeys", scenario, std::move(call_latencies_ns));
}

BenchmarkResult BenchmarkBatchFree(const BenchmarkScenario& scenario,
                                   bool enable_timing_log) {
  const auto lengths = BuildLengthBatch(scenario.batch_size, kShortLength);
  const std::size_t batch_count =
      BatchCount(scenario.resident_keys, scenario.batch_size);
  auto manager = MakeInitializedManager(
      scenario.resident_keys + scenario.batch_size + 1);
  PopulateCommittedKeys(manager.get(), "free_resident", scenario.resident_keys,
                        scenario.batch_size, lengths);
  WarmupBatchFree(manager.get(), "warmup_free", lengths, "BatchFree");
  manager->SetTimingLogEnabled(enable_timing_log);

  std::vector<double> free_samples;
  std::vector<double> build_key_samples;
  free_samples.reserve(batch_count);
  build_key_samples.reserve(batch_count);
  for (std::size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
    std::vector<KeyId> keys;
    build_key_samples.push_back(MeasureNs([&]() {
      keys = BuildKeyBatch("free_resident", batch_index * scenario.batch_size,
                           scenario.batch_size);
    }));
    free_samples.push_back(MeasureNs([&]() {
      CheckOk(manager->BatchFree(keys), "BatchFree");
    }));
  }

  BenchmarkResult result;
  result.total = BuildStats("BatchFree", scenario, free_samples);
  result.details.push_back(BuildStats("BatchFree.BuildKeyBatch", scenario,
                                      std::move(build_key_samples)));
  result.details.push_back(BuildStats("BatchFree.BatchFree", scenario,
                                      std::move(free_samples)));
  return result;
}

BenchmarkResult BenchmarkEndToEnd(const BenchmarkScenario& scenario,
                                  bool enable_timing_log,
                                  const std::string& method_name,
                                  const std::string& resident_prefix,
                                  const std::string& work_prefix,
                                  const std::vector<std::uint64_t>& lengths,
                                  std::uint64_t expected_length) {
  auto manager = MakeInitializedManager(
      scenario.resident_keys + scenario.batch_size + 1);
  const auto resident_lengths =
      BuildLengthBatch(scenario.batch_size, kShortLength);
  const std::size_t batch_count =
      BatchCount(scenario.resident_keys, scenario.batch_size);
  PopulateCommittedKeys(manager.get(), resident_prefix, scenario.resident_keys,
                        scenario.batch_size, resident_lengths);
  WarmupEndToEnd(manager.get(), "warmup_" + work_prefix, lengths,
                 expected_length, method_name.c_str());
  manager->SetTimingLogEnabled(enable_timing_log);

  std::vector<double> total_samples;
  std::vector<double> prepare_samples;
  std::vector<double> commit_samples;
  std::vector<double> lock_samples;
  std::vector<double> lookup_samples;
  std::vector<double> unlock_samples;
  std::vector<double> free_samples;
  total_samples.reserve(batch_count);
  prepare_samples.reserve(batch_count);
  commit_samples.reserve(batch_count);
  lock_samples.reserve(batch_count);
  lookup_samples.reserve(batch_count);
  unlock_samples.reserve(batch_count);
  free_samples.reserve(batch_count);

  for (std::size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
    const auto work_keys =
        BuildKeyBatch(work_prefix, batch_index * scenario.batch_size,
                      scenario.batch_size);
    PutContext put_context;
    PutAccessDesc put_desc;
    LockContext lock_context;
    FlatAccessDesc access_desc;

    const auto total_start = std::chrono::steady_clock::now();

    const auto prepare_start = std::chrono::steady_clock::now();
    CheckOk(manager->PreparePut(work_keys, lengths, &put_context, &put_desc),
            (method_name + ".PreparePut").c_str());
    prepare_samples.push_back(ElapsedNs(prepare_start));

    const auto commit_start = std::chrono::steady_clock::now();
    CheckOk(manager->CommitPut(put_context),
            (method_name + ".CommitPut").c_str());
    commit_samples.push_back(ElapsedNs(commit_start));

    const auto lock_start = std::chrono::steady_clock::now();
    CheckOk(manager->LockKeys(work_keys, MetaManagerLockMode::kStableReadAllowAppend,
                              &lock_context),
            (method_name + ".LockKeys").c_str());
    lock_samples.push_back(ElapsedNs(lock_start));

    const auto lookup_start = std::chrono::steady_clock::now();
    CheckOk(manager->BatchLookup(work_keys, lock_context, &access_desc),
            (method_name + ".BatchLookup").c_str());
    lookup_samples.push_back(ElapsedNs(lookup_start));

    const auto unlock_start = std::chrono::steady_clock::now();
    CheckOk(manager->UnlockKeys(lock_context),
            (method_name + ".UnlockKeys").c_str());
    unlock_samples.push_back(ElapsedNs(unlock_start));

    const auto free_start = std::chrono::steady_clock::now();
    CheckOk(manager->BatchFree(work_keys),
            (method_name + ".BatchFree").c_str());
    free_samples.push_back(ElapsedNs(free_start));

    total_samples.push_back(ElapsedNs(total_start));
    CheckAccessDesc(put_desc, work_keys, expected_length,
                    (method_name + ".PreparePut").c_str());
    CheckAccessDesc(access_desc, work_keys, expected_length,
                    (method_name + ".BatchLookup").c_str());
  }

  BenchmarkResult result;
  result.total = BuildStats(method_name, scenario, std::move(total_samples));
  result.details.push_back(BuildStats(
      method_name + ".PreparePut", scenario, std::move(prepare_samples)));
  result.details.push_back(BuildStats(
      method_name + ".CommitPut", scenario, std::move(commit_samples)));
  result.details.push_back(BuildStats(
      method_name + ".LockKeys", scenario, std::move(lock_samples)));
  result.details.push_back(BuildStats(
      method_name + ".BatchLookup", scenario, std::move(lookup_samples)));
  result.details.push_back(BuildStats(
      method_name + ".UnlockKeys", scenario, std::move(unlock_samples)));
  result.details.push_back(BuildStats(
      method_name + ".BatchFree", scenario, std::move(free_samples)));
  return result;
}

BenchmarkResult BenchmarkEndToEndShort(const BenchmarkScenario& scenario,
                                       bool enable_timing_log) {
  return BenchmarkEndToEnd(
      scenario, enable_timing_log, "EndToEnd(short)", "e2e_short_resident",
      "e2e_short_work", BuildLengthBatch(scenario.batch_size, kShortLength),
      kShortLength);
}

BenchmarkResult BenchmarkEndToEndLong(const BenchmarkScenario& scenario,
                                      bool enable_timing_log) {
  return BenchmarkEndToEnd(
      scenario, enable_timing_log, "EndToEnd(long)", "e2e_long_resident",
      "e2e_long_work", BuildLengthBatch(scenario.batch_size, kLongLength),
      kLongLength);
}

void PrintStatsTable(const std::vector<Stats>& stats_list) {
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "MetaManager single-thread performance benchmark "
               "(unit: ns, total_ms)\n";
  std::cout << "Scenario notes:\n";
  std::cout << "- all benchmark calls run on one thread\n";
  std::cout << "- one independent warm-up round is excluded from statistics\n";
  std::cout << "- resident_keys: committed keys already stored before the measured call\n";
  std::cout << "- batch_size: number of keys handled by one interface call\n";
  std::cout << "- call_count: resident_keys / batch_size\n";
  std::cout << "- avg_ns: average latency per measured call\n";
  std::cout << "- avg_per_key_ns: avg_ns amortized to one key\n";
  std::cout << "- EndToEnd rows include per-stage PreparePut/LockKeys/BatchFree timing\n";
  std::cout << "- --enable-timing-log emits internal phase timing for "
               "PreparePut/LockKeys/BatchFree\n";
  std::cout << "method,resident_keys,batch_size,call_count,total_ms,avg_ns,"
               "avg_per_key_ns,min_ns,p50_ns,p95_ns,max_ns\n";
  for (const Stats& stats : stats_list) {
    std::cout << stats.method << ',' << stats.resident_keys << ','
              << stats.batch_size << ',' << stats.call_count << ','
              << stats.total_ms << ',' << stats.avg_ns << ','
              << stats.avg_per_key_ns << ',' << stats.min_ns << ','
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
    } else if (arg == "--show-progress") {
      options.show_progress = true;
    }
  }

  g_show_progress = options.show_progress;
  const std::vector<BenchmarkScenario> scenarios = BuildScenarios();
  std::vector<Stats> stats_list;
  stats_list.reserve(1 + scenarios.size() * 35);

  AppendStats(&stats_list,
              BenchmarkRegisterRemoteRegion(options.enable_timing_log),
              options.show_progress);
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

  PrintStatsTable(stats_list);
  if (options.enable_timing_log) {
    FlushMetaManagerTimingSummary();
  }
  return 0;
}
