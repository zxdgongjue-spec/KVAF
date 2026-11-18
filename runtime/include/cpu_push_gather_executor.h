#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "cpu_gather_executor.h"
#include "cpu_push_gather_schedule.h"

namespace kvaf::peer {

// CPU-push Scatter owns this executor. AIV Gather READ keeps CpuGatherExecutor's
// descriptor validation, prefix publication and worker scheduling unchanged.
class CpuPushGatherExecutor {
 public:
  CpuPushGatherExecutor(void* memory, std::uint64_t base, std::size_t max_threads,
                        std::size_t min_bytes_per_thread = 256U * 1024U,
                        GatherWaitMode wait_mode = GatherWaitMode::kPassive,
                        bool bind_cores = false, std::size_t cpu_offset = 0U)
      : memory_(static_cast<std::uint8_t*>(memory)), base_(base),
        cpu_ids_(bind_cores ? AllowedCpus(cpu_offset) : std::vector<int>{}),
        max_threads_(std::min(std::max<std::size_t>(1U, max_threads),
            cpu_ids_.empty() ? std::max<std::size_t>(1U, max_threads) : cpu_ids_.size())),
        min_bytes_per_thread_(std::max<std::size_t>(1U, min_bytes_per_thread)),
        wait_mode_(wait_mode) {}

  ~CpuPushGatherExecutor() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_.store(true, std::memory_order_release);
    }
    ready_.notify_all();
    for (auto& helper : helpers_) helper.join();
  }
  CpuPushGatherExecutor(const CpuPushGatherExecutor&) = delete;
  CpuPushGatherExecutor& operator=(const CpuPushGatherExecutor&) = delete;

  struct FixedSuffix {
    const std::uint64_t* ready_sequence{};
    std::uint64_t sequence{};
    const std::atomic<bool>* stop{};
    std::chrono::milliseconds timeout{};
    std::size_t first_wave{};
  };
  // The caller retains the address list and staging through Copy.
  std::size_t PrepareFixed(void* staging, const std::uint64_t* addresses,
                          std::uint32_t count, std::uint32_t span_bytes,
                          std::uint64_t source_bytes,
                          const std::vector<cpu_push::GatherWave>& waves = {},
                          const FixedSuffix* suffix = nullptr,
                          std::size_t first_shard_wave = SIZE_MAX,
                          bool jetty_supply = false,
                          std::uint32_t jettys = cpu_push::kJettyCount);
  using WaveReady = std::function<bool(std::size_t)>;
  using ShardProgress = std::function<bool()>;
  void Copy(const WaveReady& on_wave = {}, const ShardProgress& on_shards = {});
  bool ShardReady(std::size_t wave, std::size_t queue) const;
  bool UsesJettySupply() const { return fixed_jetty_supply_; }

  void BindWorker(std::size_t worker) const {
#if defined(__linux__)
    const long tid = syscall(SYS_gettid);
    if (worker >= cpu_ids_.size()) {
      return;
    }
    cpu_set_t mask{};
    CPU_ZERO(&mask);
    CPU_SET(cpu_ids_[worker], &mask);
    const int result = sched_setaffinity(0, sizeof(mask), &mask);
    if (result)
      std::fprintf(stderr, "[KVAF_CPU] gather affinity failed tid=%ld cpu=%d errno=%d\n", tid,
                   cpu_ids_[worker], errno);
#else
    (void)worker;
#endif
  }

 private:
  std::size_t PrepareJob(void* staging, const std::uint64_t* addresses,
                         std::uint64_t total_bytes) {
    const auto threads =
        std::min(max_threads_, std::max<std::size_t>(1U, total_bytes / min_bytes_per_thread_));
    while (helpers_.size() + 1U < threads) {
      const auto worker = helpers_.size() + 1U;
      helpers_.emplace_back(&CpuPushGatherExecutor::RunHelper, this, worker);
    }
    staging_ = static_cast<std::uint8_t*>(staging);
    addresses_ = addresses;
    total_bytes_ = total_bytes;
    active_threads_ = threads;
    for (auto& wave : fixed_progress_) wave.remaining = threads;
    PrepareShardProgress();
    return threads;
  }

  static std::vector<int> AllowedCpus(std::size_t offset) {
    std::vector<int> cpus;
#if defined(__linux__)
    cpu_set_t mask{};
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0)
      for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
        if (CPU_ISSET(cpu, &mask)) cpus.push_back(cpu);
#endif
    if (offset < cpus.size())
      cpus.erase(cpus.begin(), cpus.begin() + offset);
    else
      cpus.clear();
    return cpus;
  }

  std::uint64_t Boundary(std::size_t index, std::uint64_t begin, std::uint64_t end,
                         std::size_t threads) const {
    if (index == 0U) return begin;
    if (index == threads) return end;
    const auto target = begin + (end - begin) * index / threads;
    const auto base = reinterpret_cast<std::uintptr_t>(staging_);
    const auto boundary = ((base + target + 63U) & ~std::uintptr_t{63U}) - base;
    return std::min<std::uint64_t>(boundary, end);
  }

  std::uint64_t CopyFixedRange(std::uint64_t begin, std::uint64_t end);
  void PrepareShardProgress();
  void FinishFixedRange(std::size_t& remaining);
  template <typename OnProgress>
  std::uint64_t CopyFixedWave(std::size_t worker, std::size_t wave, OnProgress on_progress);
  template <typename OnWait>
  bool WaitFixedSuffix(OnWait on_wait);
  std::uint64_t ProcessSource(std::size_t worker);
  void DispatchJob();
  void RunHelper(std::size_t worker);

  std::uint8_t* memory_;
  std::uint64_t base_;
  std::vector<int> cpu_ids_;
  std::size_t max_threads_, min_bytes_per_thread_;
  GatherWaitMode wait_mode_;
  std::vector<std::thread> helpers_;
  std::mutex mutex_;
  std::condition_variable ready_;
  static constexpr std::uint64_t kDispatchThreadMask = UINT32_MAX;
  static constexpr std::uint64_t kDispatchGenerationStep = std::uint64_t{1} << 32;
  const std::uint64_t* addresses_{};
  std::uint32_t span_bytes_{};
  std::uint8_t* staging_{};
  std::uint64_t total_bytes_{};
  std::size_t active_threads_{1U};
  std::uint64_t fixed_last_source_offset_{};
  FixedSuffix fixed_suffix_{};
  struct alignas(64) FixedWaveProgress {
    std::size_t remaining{}, first_shard{};
    std::uint64_t aligned_bytes{};
    cpu_push::GatherWave rows{};
  };
  std::vector<FixedWaveProgress> fixed_progress_;
  struct alignas(64) FixedShardProgress { std::size_t remaining{}; };
  std::size_t fixed_first_shard_wave_{SIZE_MAX};
  bool fixed_jetty_supply_{};
  std::uint32_t fixed_jetty_count_{cpu_push::kJettyCount};
  std::vector<FixedShardProgress> fixed_shards_;
  enum class FixedError { kNone, kSource, kSuffix, kCancelled };
  std::atomic<FixedError> fixed_error_{FixedError::kNone};
  void SetFixedError(FixedError error) {
    auto expected = FixedError::kNone;
    fixed_error_.compare_exchange_strong(expected, error, std::memory_order_relaxed);
  }
  std::atomic<bool> stopping_{false};
  alignas(64) std::atomic<std::uint64_t> dispatch_{0U};
  alignas(64) std::atomic<std::size_t> completed_{0U};
};

}  // namespace kvaf::peer
