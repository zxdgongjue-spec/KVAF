#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

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

struct GatherCopyProfile {
  std::uint64_t bytes{}, first_ready_ns{}, copy_ns{};
  std::size_t threads{};
};

class CpuGatherExecutor {
 public:
  CpuGatherExecutor(void* memory, std::uint64_t base, std::size_t max_threads,
                std::size_t min_bytes_per_thread = 256U * 1024U,
                GatherWaitMode wait_mode = GatherWaitMode::kPassive, bool profile = false)
      : memory_(static_cast<std::uint8_t*>(memory)),
        base_(base),
        max_threads_(std::max<std::size_t>(1U, max_threads)),
        min_bytes_per_thread_(std::max<std::size_t>(1U, min_bytes_per_thread)),
        wait_mode_(wait_mode), profile_enabled_(profile) {}

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

  void Copy();

  const GatherCopyProfile& profile() const { return profile_; }

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
    staging_ = memory_ + (staging - base_);
    total_bytes_ = total_bytes;
    active_threads_ = threads;
    phase_.store(0U, std::memory_order_release);
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

  std::uint64_t ProcessSource(std::size_t worker);

  void RunHelper(std::size_t worker);

  std::uint8_t* memory_;
  std::uint64_t base_;
  std::size_t max_threads_;
  std::size_t min_bytes_per_thread_;
  GatherWaitMode wait_mode_;
  bool profile_enabled_;
  GatherCopyProfile profile_;
  std::uint64_t copy_begin_ns_{0U};
  std::atomic<std::uint64_t> first_ready_ns_{0U};
  std::vector<std::thread> helpers_;
  std::mutex mutex_;
  std::condition_variable ready_;
  std::atomic<std::uint64_t> generation_{0U};
  // The sender retains this descriptor until all copy threads finish and ACK is published.
  const GatherSpan* spans_{nullptr};
  std::uint32_t span_count_{0U}, span_bytes_{0U};
  std::uint8_t* staging_{nullptr};
  std::uint64_t total_bytes_{0U};
  std::size_t active_threads_{1U};
  std::vector<std::uint64_t> variable_offsets_;
  std::uint64_t* pipeline_ready_{nullptr};
  std::uint32_t pipeline_chunk_bytes_{0U};
  std::vector<std::uint32_t> chunk_done_;
  std::atomic<std::size_t> next_chunk_{0U};
  alignas(64) std::atomic<std::uint32_t> phase_{0U};
  alignas(64) std::atomic<std::size_t> completed_{0U};
  std::atomic<bool> stopping_{false};
};

}  // namespace kvaf::peer
