#pragma once

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

#include "cpu_gather.h"

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
enum class GatherReadMode { kAicpu, kAiv };

// CPU steady-clock timestamps only; they do not share the AIV clock.
struct alignas(64) CpuGatherThreadProfile {
  std::uint64_t ready_ns{}, start_ns{}, copy_done_ns{}, done_ns{};
  std::uint64_t thread_cpu_ns{}, bytes{}, tid{}, progress_publications{};
  int cpu{-1};
  bool wait_path{}, cpu_time_valid{};
};

struct CpuGatherCopyProfile {
  std::uint64_t publish_ns{}, begin_ns{}, coordinator_done_ns{}, done_ns{};
};

class CpuGatherExecutor {
 public:
  CpuGatherExecutor(void* memory, std::uint64_t base, std::size_t max_threads,
                std::size_t min_bytes_per_thread = 256U * 1024U,
                GatherWaitMode wait_mode = GatherWaitMode::kPassive,
                void* workspace_memory = nullptr, std::uint64_t workspace_base = 0U,
                bool bind_cores = false, std::size_t cpu_offset = 0U,
                GatherReadMode read_mode = GatherReadMode::kAicpu)
      : memory_(static_cast<std::uint8_t*>(memory)),
        base_(base),
        workspace_memory_(static_cast<std::uint8_t*>(workspace_memory ? workspace_memory : memory)),
        workspace_base_(workspace_memory ? workspace_base : base),
        cpu_ids_(bind_cores ? AllowedCpus(cpu_offset) : std::vector<int>{}),
        max_threads_(std::min(std::max<std::size_t>(1U, max_threads),
                              cpu_ids_.empty() ? std::max<std::size_t>(1U, max_threads)
                                               : cpu_ids_.size())),
        min_bytes_per_thread_(std::max<std::size_t>(1U, min_bytes_per_thread)),
        wait_mode_(wait_mode), read_mode_(read_mode) {}

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
                            std::uint64_t sequence, std::size_t max_spans,
                            const GatherControl* control = nullptr,
                            const std::atomic<bool>* stop = nullptr);

  void BindWorker(std::size_t worker) const {
#if defined(__linux__)
    if (worker >= cpu_ids_.size()) {
      return;
    }
    cpu_set_t mask{};
    CPU_ZERO(&mask);
    CPU_SET(cpu_ids_[worker], &mask);
    const int result = sched_setaffinity(0, sizeof(mask), &mask);
    if (result)
      std::fprintf(stderr, "[KVAF_CPU] gather affinity failed cpu=%d errno=%d\n",
                   cpu_ids_[worker], errno);
#else
    (void)worker;
#endif
  }

  void Copy();
  // Enable before the first PrepareDirect. Disabled execution reads no clocks.
  void EnableProfiling() { thread_profiles_.resize(max_threads_); }
  std::size_t MaxThreads() const { return max_threads_; }
  const CpuGatherCopyProfile& CopyProfile() const { return copy_profile_; }
  // Read active entries only, after Copy has joined all helpers.
  const std::vector<CpuGatherThreadProfile>& ThreadProfiles() const { return thread_profiles_; }

 private:
  std::size_t Publish(std::uint64_t staging, const GatherSpan* spans,
                      std::uint32_t span_count, std::uint32_t total_bytes) {
    const std::size_t threads =
        std::min(max_threads_, std::max<std::size_t>(1U, total_bytes / min_bytes_per_thread_));
    std::unique_lock<std::mutex> lock(mutex_);
    while (helpers_.size() + 1U < threads) {
      const std::size_t index = helpers_.size() + 1U;
      helpers_.emplace_back(&CpuGatherExecutor::RunHelper, this, index);
    }
    spans_ = spans;
    span_count_ = span_count;
    staging_ = workspace_memory_ + (staging - workspace_base_);
    total_bytes_ = total_bytes;
    active_threads_ = threads;
    if (pipeline_ready_ != nullptr && read_mode_ == GatherReadMode::kAicpu) {
      chunk_done_.assign(GatherProgressChunkCount(total_bytes_, pipeline_chunk_bytes_), 0U);
      next_chunk_.store(0U, std::memory_order_relaxed);
    } else if (cooperative_) {
      cooperative_count_ = GatherProgressChunkCount(total_bytes_, pipeline_chunk_bytes_, first_chunk_divisor_);
      for (std::uint32_t chunk = 0; chunk < cooperative_count_; ++chunk) {
        cooperative_progress_[chunk].remaining.store(threads, std::memory_order_relaxed);
        cooperative_progress_[chunk].complete.store(false, std::memory_order_relaxed);
      }
    } else if (pipeline_ready_ != nullptr) {
      // Balanced static rounds, capped at 256 KiB per copy task. A 2.5625-MiB
      // lane with 16 threads gets 16 x 164-KiB tasks, independent of ready chunk.
      constexpr std::uint64_t kMaxTaskBytes = 256U * 1024U;
      const auto round_bytes = threads * kMaxTaskBytes;
      const auto rounds = (total_bytes_ + round_bytes - 1U) / round_bytes;
      const auto slots = threads * rounds;
      task_bytes_ = ((total_bytes_ + slots - 1U) / slots + 63U) & ~std::uint64_t{63U};
      task_progress_.assign((total_bytes_ + task_bytes_ - 1U) / task_bytes_, TaskProgress{});
      ready_tasks_ = 0U;
      published_bytes_ = 0U;
    }
    phase_.store(0U, std::memory_order_release);
    if (!thread_profiles_.empty()) {
      std::fill_n(thread_profiles_.begin(), threads, CpuGatherThreadProfile{});
      copy_profile_ = {};
      copy_profile_.publish_ns = NowNs();
    }
    generation_.fetch_add(1U, std::memory_order_release);
    lock.unlock();
    ready_.notify_all();
    return threads;
  }

  std::uint64_t Boundary(std::size_t index) const {
    if (index == 0U) return 0U;
    if (index == active_threads_) return total_bytes_;
    const std::uint64_t target = total_bytes_ * index / active_threads_;
    const auto address = reinterpret_cast<std::uintptr_t>(staging_) + target;
    return ((address + 63U) & ~std::uintptr_t{63U}) - reinterpret_cast<std::uintptr_t>(staging_);
  }

  std::uint64_t CopyRange(std::uint64_t offset, std::uint64_t end) const;
  std::uint64_t CopyReadyRange(std::uint64_t offset, std::uint64_t end) const;
  bool WaitSuffix() const;

  std::uint64_t ProcessSource(std::size_t worker);
  std::uint64_t CopyDynamicChunks();
  void PublishTaskReady();
  void PublishCooperativeReady();
  void AdvanceReadyPrefix(std::uint64_t observed, std::uint64_t ready);

  static std::uint64_t NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
  }
  void CopyWorker(std::size_t worker);
  void RunHelper(std::size_t worker);

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

  std::uint8_t* memory_;
  std::uint64_t base_;
  // Source addresses remain cache-canonical; descriptor/staging addresses are
  // relative to the independently registered Gather workspace.
  std::uint8_t* workspace_memory_;
  std::uint64_t workspace_base_;
  std::vector<int> cpu_ids_;
  std::size_t max_threads_;
  std::size_t min_bytes_per_thread_;
  GatherWaitMode wait_mode_;
  const GatherReadMode read_mode_;
  std::vector<std::thread> helpers_;
  std::mutex mutex_;
  std::condition_variable ready_;
  std::atomic<std::uint64_t> generation_{0U};
  // The sender retains this descriptor until all copy threads finish and ACK is published.
  const GatherSpan* spans_{nullptr};
  const std::uint64_t* descriptor_suffix_{nullptr};
  const std::atomic<bool>* request_stop_{nullptr};
  std::uint64_t sequence_{}, head_bytes_{};
  mutable std::atomic<bool> copy_failed_{false};
  std::uint32_t span_count_{0U}, span_bytes_{0U};
  std::uint8_t* staging_{nullptr};
  std::uint64_t total_bytes_{0U};
  std::size_t active_threads_{1U};
  std::vector<std::uint64_t> variable_offsets_;
  std::uint64_t* pipeline_ready_{nullptr};
  std::uint32_t pipeline_chunk_bytes_{0U};
  // AICPU helpers claim and publish completed uniform chunks independently.
  std::vector<std::uint32_t> chunk_done_;
  std::atomic<std::size_t> next_chunk_{0U};
  std::uint32_t first_chunk_divisor_{0U};
  bool cooperative_{};
  std::uint32_t cooperative_count_{};
  // One completion RMW per worker/block; only the last worker scans/publishes.
  // Blocks have separate lines and no worker waits for another block's users.
  struct alignas(64) CooperativeProgress {
    std::atomic<std::size_t> remaining{0};
    std::atomic<bool> complete{false};
  };
  std::array<CooperativeProgress, kGatherMaxProgressChunks> cooperative_progress_;
  // Separate balanced task ownership from partial DMA-visible progress. Each
  // slot has one writer; only the coordinator publishes the contiguous prefix.
  struct alignas(64) TaskProgress { std::uint64_t bytes{}; };
  std::vector<TaskProgress> task_progress_;
  std::uint64_t task_bytes_{0U};
  std::size_t ready_tasks_{0U};
  std::uint64_t published_bytes_{0U};
  std::vector<CpuGatherThreadProfile> thread_profiles_;
  CpuGatherCopyProfile copy_profile_{};
  alignas(64) std::atomic<std::uint32_t> phase_{0U};
  alignas(64) std::atomic<std::size_t> completed_{0U};
  std::atomic<bool> stopping_{false};
};

}  // namespace kvaf::peer
