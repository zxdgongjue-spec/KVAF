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

#include "cpu_gather.h"
#include "cpu_push_gather_schedule.h"

#if defined(__linux__)
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#endif

namespace kvaf::peer {

inline void GatherBarrier() {
#if defined(__aarch64__)
  asm volatile("dsb sy" ::: "memory");
#else
  std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

inline void GatherSpinPause() {
#if defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#elif defined(__x86_64__)
  asm volatile("pause" ::: "memory");
#else
  std::this_thread::yield();
#endif
}

enum class GatherWaitMode { kActive, kAdaptive, kPassive };

struct alignas(64) CpuGatherThreadProfile {
  std::uint64_t ready_ns{}, start_ns{}, copy_done_ns{}, done_ns{};
  std::uint64_t thread_cpu_ns{}, bytes{}, tid{};
  int cpu{-1};
  bool wait_path{}, cpu_time_valid{};
};

struct CpuGatherCopyProfile {
  std::uint64_t begin_ns{}, coordinator_done_ns{}, done_ns{};
  std::uint64_t suffix_wait_ns{}, suffix_seen_ns{};
};

class CpuGatherExecutor {
 public:
  CpuGatherExecutor(void* memory, std::uint64_t base, std::size_t max_threads,
                std::size_t min_bytes_per_thread = 256U * 1024U,
                GatherWaitMode wait_mode = GatherWaitMode::kPassive, bool bind_cores = false,
                std::size_t cpu_offset = 0U)
      : memory_(static_cast<std::uint8_t*>(memory)),
        base_(base),
        cpu_ids_(bind_cores ? AllowedCpus(cpu_offset) : std::vector<int>{}),
        max_threads_(
            std::min(std::max<std::size_t>(1U, max_threads),
                     cpu_ids_.empty() ? std::max<std::size_t>(1U, max_threads) : cpu_ids_.size())),
        min_bytes_per_thread_(std::max<std::size_t>(1U, min_bytes_per_thread)),
        wait_mode_(wait_mode) {}

  ~CpuGatherExecutor() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_.store(true, std::memory_order_release);
    }
    ready_.notify_all();
    for (std::thread& helper : helpers_) helper.join();
  }

  CpuGatherExecutor(const CpuGatherExecutor&) = delete;
  CpuGatherExecutor& operator=(const CpuGatherExecutor&) = delete;

  std::size_t PrepareDirect(std::uint64_t region_bytes, std::uint64_t descriptor_address,
                            std::uint64_t sequence, std::size_t max_spans);
  struct FixedSuffix {
    const std::uint64_t* ready_sequence{};
    std::uint64_t sequence{};
    const std::atomic<bool>* stop{};
    std::chrono::milliseconds timeout{};
    std::size_t first_wave{};
  };
  // Keep the address list and destination stable through Copy. Copy reports
  // invalid fixed source addresses after all participating workers finish.
  // Wave ranges partition [0, count). Suffix guards first_wave and all later
  // waves. An empty plan gathers the entire batch before returning.
  std::size_t PrepareFixed(void* staging, const std::uint64_t* addresses,
                           std::uint32_t count, std::uint32_t span_bytes,
                           std::uint64_t source_bytes,
                           const std::vector<cpu_push::GatherWave>& waves = {},
                           const FixedSuffix* suffix = nullptr,
                           std::size_t first_shard_wave = SIZE_MAX,
                           bool jetty_supply = false,
                           std::uint32_t jettys = cpu_push::kJettyCount);

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

  // The coordinator calls on_wave with the number of completed waves in
  // plan order. Returning false stops publication and joins the helpers.
  using WaveReady = std::function<bool(std::size_t)>;
  // After the whole-wave prefix, on_shards can inspect the completion of
  // each shard independently. Wave indices passed to ShardReady are zero-based.
  using ShardProgress = std::function<bool()>;
  void Copy(const WaveReady& on_wave = {}, const ShardProgress& on_shards = {});
  bool ShardReady(std::size_t wave, std::size_t queue) const;
  std::uint64_t FixedReadyNs(std::size_t wave, std::size_t queue, bool shard) const;
  bool UsesJettySupply() const { return fixed_jetty_supply_; }
  void EnableProfiling(bool enabled, bool trace_ready = false) {
    if (enabled) thread_profiles_.resize(max_threads_);
    profiling_enabled_ = enabled;
    trace_ready_ = enabled && trace_ready;
  }
  const CpuGatherCopyProfile& CopyProfile() const { return copy_profile_; }
  const std::vector<CpuGatherThreadProfile>& ThreadProfiles() const { return thread_profiles_; }

 private:
  std::size_t PrepareJob(void* staging, const GatherSpan* spans, const std::uint64_t* addresses,
                         std::uint32_t span_count, std::uint64_t total_bytes) {
    const std::size_t threads =
        std::min(max_threads_, std::max<std::size_t>(1U, total_bytes / min_bytes_per_thread_));
    while (helpers_.size() + 1U < threads) {
      const std::size_t index = helpers_.size() + 1U;
      helpers_.emplace_back(&CpuGatherExecutor::RunHelper, this, index);
    }
    spans_ = spans;
    addresses_ = addresses;
    span_count_ = span_count;
    staging_ = static_cast<std::uint8_t*>(staging);
    total_bytes_ = total_bytes;
    active_threads_ = threads;
    for (auto& wave : fixed_progress_) {
      wave.remaining = threads;
      wave.ready_ns = 0;
    }
    PrepareShardProgress();
    if (profiling_enabled_) {
      std::fill_n(thread_profiles_.begin(), threads, CpuGatherThreadProfile{});
      copy_profile_ = {};
    }
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

  std::uint64_t CopyRange(std::uint64_t offset, std::uint64_t end) const;
  std::uint64_t CopyFixedRange(std::uint64_t begin, std::uint64_t end);
  void PrepareShardProgress();
  void FinishFixedRange(std::size_t& remaining, std::uint64_t& ready_ns);
  void RecordSuffixSeen();
  template <typename OnProgress>
  std::uint64_t CopyFixedWave(std::size_t worker, std::size_t wave, OnProgress on_progress);
  template <typename OnWait>
  bool WaitFixedSuffix(OnWait on_wait);

  std::uint64_t ProcessSource(std::size_t worker);

  static std::uint64_t NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
  }
  void BeginWorker(std::size_t worker);
  void FinishWorker(std::size_t worker, std::uint64_t bytes);
  void RunHelper(std::size_t worker);

  std::uint8_t* memory_;
  std::uint64_t base_;
  std::vector<int> cpu_ids_;
  std::size_t max_threads_;
  std::size_t min_bytes_per_thread_;
  GatherWaitMode wait_mode_;
  std::vector<std::thread> helpers_;
  std::mutex mutex_;
  std::condition_variable ready_;
  static constexpr std::uint64_t kDispatchThreadMask = UINT32_MAX;
  static constexpr std::uint64_t kDispatchGenerationStep = std::uint64_t{1} << 32;
  // The sender retains this descriptor until all copy threads finish and ACK is published.
  const GatherSpan* spans_{nullptr};
  const std::uint64_t* addresses_{nullptr};
  std::uint32_t span_count_{0U}, span_bytes_{0U};
  std::uint8_t* staging_{nullptr};
  std::uint64_t total_bytes_{0U};
  std::size_t active_threads_{1U};
  std::uint64_t fixed_last_source_offset_{0U};
  FixedSuffix fixed_suffix_{};
  struct alignas(64) FixedWaveProgress {
    std::size_t remaining{};
    std::size_t first_shard{};
    std::uint64_t aligned_bytes{};
    cpu_push::GatherWave rows{};
    std::uint64_t ready_ns{};
  };
  std::vector<FixedWaveProgress> fixed_progress_;
  struct alignas(64) FixedShardProgress {
    std::size_t remaining{};
    std::uint64_t ready_ns{};
  };
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
  std::vector<std::uint64_t> variable_offsets_;
  std::uint64_t* pipeline_ready_{nullptr};
  std::uint32_t pipeline_chunk_bytes_{0U};
  std::vector<std::uint32_t> chunk_done_;
  std::atomic<std::size_t> next_chunk_{0U};
  bool profiling_enabled_{}, trace_ready_{};
  std::vector<CpuGatherThreadProfile> thread_profiles_;
  CpuGatherCopyProfile copy_profile_{};
  std::atomic<bool> stopping_{false};
  // The low bits specify the participants of this generation. An inactive
  // helper can skip a job without reading fields the coordinator may reuse.
  alignas(64) std::atomic<std::uint64_t> dispatch_{0U};
  alignas(64) std::atomic<std::size_t> completed_{0U};
};

}  // namespace kvaf::peer
