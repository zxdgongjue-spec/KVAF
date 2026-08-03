#include "kvaf/meta_manager/meta_manager.h"

#include <algorithm>
#include <chrono>
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

using kvaf::Address;
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

constexpr std::uint64_t kBaseAddress = 4096;
constexpr std::uint64_t kShortSlotSize = 128;
constexpr std::uint64_t kLongSlotSize = 256;
constexpr std::uint64_t kShortLength = 64;
constexpr std::uint64_t kLongLength = 200;
constexpr std::size_t kRegisterIterations = 5000;
constexpr std::size_t kMethodIterations = 20000;

struct Stats {
  std::string name;
  double avg_ns{0.0};
  double min_ns{0.0};
  double p50_ns{0.0};
  double p95_ns{0.0};
  double max_ns{0.0};
};

void CheckOk(const Status& status, const char* step) {
  if (!status.ok()) {
    std::cerr << "Benchmark failed at " << step
              << " status=" << status.message() << '\n';
    std::exit(1);
  }
}

KeyId MakeKey(const char* prefix, std::size_t index) {
  return std::string(prefix) + "_" + std::to_string(index);
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
  CheckOk(manager->RegisterRemoteRegion(MakeRegion(slot_count_per_class)),
          "RegisterRemoteRegion");
  return manager;
}

Stats BuildStats(const std::string& name, std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  const double total_ns =
      std::accumulate(samples.begin(), samples.end(), 0.0);

  Stats stats;
  stats.name = name;
  stats.avg_ns = total_ns / static_cast<double>(samples.size());
  stats.min_ns = samples.front();
  stats.p50_ns = samples[samples.size() / 2];
  stats.p95_ns = samples[(samples.size() * 95) / 100];
  stats.max_ns = samples.back();
  return stats;
}

std::vector<KeyId> OneKeyBatch(const KeyId& key) {
  return std::vector<KeyId>{key};
}

std::vector<std::uint64_t> OneLengthBatch(std::uint64_t length) {
  return std::vector<std::uint64_t>{length};
}

Stats BenchmarkRegisterRemoteRegion(bool enable_timing_log) {
  std::vector<double> samples;
  samples.reserve(kRegisterIterations);
  for (std::size_t i = 0; i < kRegisterIterations; ++i) {
    auto manager = CreateInMemoryMetaManager();
    manager->SetTimingLogEnabled(enable_timing_log);
    const auto start = std::chrono::steady_clock::now();
    CheckOk(manager->RegisterRemoteRegion(MakeRegion(1)), "RegisterRemoteRegion");
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    samples.push_back(static_cast<double>(elapsed_ns));
  }
  return BuildStats("RegisterRemoteRegion", std::move(samples));
}

Stats BenchmarkPreparePutShort(bool enable_timing_log) {
  auto manager = MakeInitializedManager(1, enable_timing_log);
  std::vector<double> samples;
  samples.reserve(kMethodIterations);
  for (std::size_t i = 0; i < kMethodIterations; ++i) {
    PutContext put_context;
    PutAccessDesc put_desc;
    const auto start = std::chrono::steady_clock::now();
    CheckOk(manager->PreparePut(OneKeyBatch(MakeKey("prepare_short", i)),
                                OneLengthBatch(kShortLength), &put_context,
                                &put_desc),
            "PreparePut(short)");
    const auto end = std::chrono::steady_clock::now();
    CheckOk(manager->AbortPut(put_context), "AbortPut(after PreparePut short)");
    const auto elapsed_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    samples.push_back(static_cast<double>(elapsed_ns));
  }
  return BuildStats("PreparePut(short)", std::move(samples));
}

Stats BenchmarkPreparePutLong(bool enable_timing_log) {
  auto manager = MakeInitializedManager(1, enable_timing_log);
  std::vector<double> samples;
  samples.reserve(kMethodIterations);
  for (std::size_t i = 0; i < kMethodIterations; ++i) {
    PutContext put_context;
    PutAccessDesc put_desc;
    const auto start = std::chrono::steady_clock::now();
    CheckOk(manager->PreparePut(OneKeyBatch(MakeKey("prepare_long", i)),
                                OneLengthBatch(kLongLength), &put_context,
                                &put_desc),
            "PreparePut(long)");
    const auto end = std::chrono::steady_clock::now();
    CheckOk(manager->AbortPut(put_context), "AbortPut(after PreparePut long)");
    const auto elapsed_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    samples.push_back(static_cast<double>(elapsed_ns));
  }
  return BuildStats("PreparePut(long)", std::move(samples));
}

Stats BenchmarkCommitPut(bool enable_timing_log) {
  auto manager = MakeInitializedManager(kMethodIterations + 1, enable_timing_log);
  std::vector<double> samples;
  samples.reserve(kMethodIterations);
  for (std::size_t i = 0; i < kMethodIterations; ++i) {
    PutContext put_context;
    PutAccessDesc put_desc;
    CheckOk(manager->PreparePut(OneKeyBatch(MakeKey("commit", i)),
                                OneLengthBatch(kShortLength), &put_context,
                                &put_desc),
            "PreparePut(before CommitPut)");
    const auto start = std::chrono::steady_clock::now();
    CheckOk(manager->CommitPut(put_context), "CommitPut");
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    samples.push_back(static_cast<double>(elapsed_ns));
  }
  return BuildStats("CommitPut", std::move(samples));
}

Stats BenchmarkAbortPut(bool enable_timing_log) {
  auto manager = MakeInitializedManager(1, enable_timing_log);
  std::vector<double> samples;
  samples.reserve(kMethodIterations);
  for (std::size_t i = 0; i < kMethodIterations; ++i) {
    PutContext put_context;
    PutAccessDesc put_desc;
    CheckOk(manager->PreparePut(OneKeyBatch(MakeKey("abort", i)),
                                OneLengthBatch(kShortLength), &put_context,
                                &put_desc),
            "PreparePut(before AbortPut)");
    const auto start = std::chrono::steady_clock::now();
    CheckOk(manager->AbortPut(put_context), "AbortPut");
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    samples.push_back(static_cast<double>(elapsed_ns));
  }
  return BuildStats("AbortPut", std::move(samples));
}

std::unique_ptr<MetaManager> MakeCommittedManager(std::size_t key_count,
                                                  bool enable_timing_log,
                                                  const char* prefix) {
  auto manager = MakeInitializedManager(key_count + 1, enable_timing_log);
  for (std::size_t i = 0; i < key_count; ++i) {
    PutContext put_context;
    PutAccessDesc put_desc;
    CheckOk(manager->PreparePut(OneKeyBatch(MakeKey(prefix, i)),
                                OneLengthBatch(kShortLength), &put_context,
                                &put_desc),
            "PreparePut(populate committed manager)");
    CheckOk(manager->CommitPut(put_context),
            "CommitPut(populate committed manager)");
  }
  return manager;
}

Stats BenchmarkLockKeys(bool enable_timing_log) {
  auto manager = MakeCommittedManager(kMethodIterations, enable_timing_log, "lock");
  std::vector<double> samples;
  samples.reserve(kMethodIterations);
  for (std::size_t i = 0; i < kMethodIterations; ++i) {
    LockContext lock_context;
    const auto keys = OneKeyBatch(MakeKey("lock", i));
    const auto start = std::chrono::steady_clock::now();
    CheckOk(manager->LockKeys(keys, MetaManagerLockMode::kStableReadAllowAppend,
                              &lock_context),
            "LockKeys");
    const auto end = std::chrono::steady_clock::now();
    CheckOk(manager->UnlockKeys(lock_context), "UnlockKeys(after LockKeys)");
    const auto elapsed_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    samples.push_back(static_cast<double>(elapsed_ns));
  }
  return BuildStats("LockKeys", std::move(samples));
}

Stats BenchmarkBatchLookup(bool enable_timing_log) {
  auto manager =
      MakeCommittedManager(kMethodIterations, enable_timing_log, "lookup");
  std::vector<double> samples;
  samples.reserve(kMethodIterations);
  for (std::size_t i = 0; i < kMethodIterations; ++i) {
    LockContext lock_context;
    FlatAccessDesc access_desc;
    const auto keys = OneKeyBatch(MakeKey("lookup", i));
    CheckOk(manager->LockKeys(keys, MetaManagerLockMode::kStableReadAllowAppend,
                              &lock_context),
            "LockKeys(before BatchLookup)");
    const auto start = std::chrono::steady_clock::now();
    CheckOk(manager->BatchLookup(keys, lock_context, &access_desc),
            "BatchLookup");
    const auto end = std::chrono::steady_clock::now();
    CheckOk(manager->UnlockKeys(lock_context),
            "UnlockKeys(after BatchLookup)");
    const auto elapsed_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    samples.push_back(static_cast<double>(elapsed_ns));
  }
  return BuildStats("BatchLookup", std::move(samples));
}

Stats BenchmarkUnlockKeys(bool enable_timing_log) {
  auto manager =
      MakeCommittedManager(kMethodIterations, enable_timing_log, "unlock");
  std::vector<double> samples;
  samples.reserve(kMethodIterations);
  for (std::size_t i = 0; i < kMethodIterations; ++i) {
    LockContext lock_context;
    const auto keys = OneKeyBatch(MakeKey("unlock", i));
    CheckOk(manager->LockKeys(keys, MetaManagerLockMode::kStableReadAllowAppend,
                              &lock_context),
            "LockKeys(before UnlockKeys)");
    const auto start = std::chrono::steady_clock::now();
    CheckOk(manager->UnlockKeys(lock_context), "UnlockKeys");
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    samples.push_back(static_cast<double>(elapsed_ns));
  }
  return BuildStats("UnlockKeys", std::move(samples));
}

Stats BenchmarkBatchFree(bool enable_timing_log) {
  auto manager = MakeCommittedManager(kMethodIterations, enable_timing_log, "free");
  std::vector<double> samples;
  samples.reserve(kMethodIterations);
  for (std::size_t i = 0; i < kMethodIterations; ++i) {
    const auto start = std::chrono::steady_clock::now();
    CheckOk(manager->BatchFree(OneKeyBatch(MakeKey("free", i))), "BatchFree");
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    samples.push_back(static_cast<double>(elapsed_ns));
  }
  return BuildStats("BatchFree", std::move(samples));
}

void PrintStatsTable(const std::vector<Stats>& stats_list) {
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "MetaManager performance benchmark (unit: ns)\n";
  std::cout << "method,avg_ns,min_ns,p50_ns,p95_ns,max_ns\n";
  for (const Stats& stats : stats_list) {
    std::cout << stats.name << ',' << stats.avg_ns << ',' << stats.min_ns << ','
              << stats.p50_ns << ',' << stats.p95_ns << ',' << stats.max_ns
              << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  bool enable_timing_log = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--enable-timing-log") {
      enable_timing_log = true;
    }
  }

  const std::vector<Stats> stats_list{
      BenchmarkRegisterRemoteRegion(enable_timing_log),
      BenchmarkPreparePutShort(enable_timing_log),
      BenchmarkPreparePutLong(enable_timing_log),
      BenchmarkCommitPut(enable_timing_log),
      BenchmarkAbortPut(enable_timing_log),
      BenchmarkLockKeys(enable_timing_log),
      BenchmarkBatchLookup(enable_timing_log),
      BenchmarkUnlockKeys(enable_timing_log),
      BenchmarkBatchFree(enable_timing_log),
  };

  PrintStatsTable(stats_list);
  return 0;
}
