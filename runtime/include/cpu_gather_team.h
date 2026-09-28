#pragma once

#include "cpu_gather.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <cerrno>
#include <cstdio>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace kvaf::peer {

inline void GatherCacheLine(const void* address, bool invalidate) {
#if defined(__aarch64__)
  if (invalidate) asm volatile("dc civac, %0" :: "r"(address) : "memory");
  else asm volatile("dc cvac, %0" :: "r"(address) : "memory");
#else
  (void)address;
  (void)invalidate;
#endif
}

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

enum class GatherWaitMode { kActive, kPassive };

class CpuGatherTeam {
 public:
  CpuGatherTeam(void* memory, std::uint64_t base, std::size_t max_threads,
                std::size_t min_bytes_per_thread = 256U * 1024U,
                GatherWaitMode wait_mode = GatherWaitMode::kPassive,
                bool bind_cores = false)
      : memory_(static_cast<std::uint8_t*>(memory)), base_(base),
        cpu_ids_(bind_cores ? AllowedCpus() : std::vector<int>{}),
        max_threads_(std::min(std::max<std::size_t>(1U, max_threads),
                              cpu_ids_.empty() ? std::max<std::size_t>(1U, max_threads)
                                               : cpu_ids_.size())),
        min_bytes_per_thread_(std::max<std::size_t>(1U, min_bytes_per_thread)),
        wait_mode_(wait_mode) {}

  ~CpuGatherTeam() {
    stopping_.store(true, std::memory_order_release);
    ready_.notify_all();
    for (std::thread& helper : helpers_) helper.join();
  }

  CpuGatherTeam(const CpuGatherTeam&) = delete;
  CpuGatherTeam& operator=(const CpuGatherTeam&) = delete;

  std::size_t max_threads() const { return max_threads_; }

  void BindWorker(std::size_t worker) const {
#if defined(__linux__)
    const long tid = syscall(SYS_gettid);
    if (worker >= cpu_ids_.size()) {
      std::fprintf(stderr, "[KVAF_CPU] GATHER_THREAD worker=%zu tid=%ld cpu=%d pinned=0\n",
                   worker, tid, sched_getcpu());
      return;
    }
    cpu_set_t mask{};
    CPU_ZERO(&mask);
    CPU_SET(cpu_ids_[worker], &mask);
    const int result = sched_setaffinity(0, sizeof(mask), &mask);
    std::fprintf(stderr,
                 "[KVAF_CPU] GATHER_THREAD worker=%zu tid=%ld cpu=%d pinned=%d error=%d\n",
                 worker, tid, cpu_ids_[worker], result == 0, result == 0 ? 0 : errno);
#else
    (void)worker;
#endif
  }

  std::size_t Prepare(std::uint64_t staging, const std::vector<GatherSpan>& spans,
                      std::uint32_t total_bytes) {
    const std::size_t threads = std::min(max_threads_, std::max<std::size_t>(
        1U, total_bytes / min_bytes_per_thread_));
    std::unique_lock<std::mutex> lock(mutex_);
    while (helpers_.size() + 1U < threads) {
      const std::size_t index = helpers_.size() + 1U;
      helpers_.emplace_back(&CpuGatherTeam::RunHelper, this, index);
    }
    spans_ = &spans;
    staging_ = memory_ + (staging - base_);
    total_bytes_ = total_bytes;
    active_threads_ = threads;
    offsets_.clear();
    offsets_.reserve(spans.size() + 1U);
    offsets_.push_back(0U);
    for (const auto& span : spans) offsets_.push_back(offsets_.back() + span.length);
    phase_.store(0U, std::memory_order_release);
    generation_.fetch_add(1U, std::memory_order_release);
    lock.unlock();
    ready_.notify_all();
    return threads;
  }

  void InvalidateSources() { RunPhase(kInvalidatePhase); }
  void Copy() { RunPhase(kCopyPhase); }
  void FlushStaging() { RunPhase(kFlushPhase); }

 private:
  static constexpr std::uint32_t kInvalidatePhase = 1U;
  static constexpr std::uint32_t kCopyPhase = 2U;
  static constexpr std::uint32_t kFlushPhase = 3U;

  static std::vector<int> AllowedCpus() {
    std::vector<int> cpus;
#if defined(__linux__)
    cpu_set_t mask{};
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0)
      for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
        if (CPU_ISSET(cpu, &mask)) cpus.push_back(cpu);
#endif
    return cpus;
  }

  std::uint64_t Boundary(std::size_t index) const {
    if (index == 0U) return 0U;
    if (index == active_threads_) return total_bytes_;
    const std::uint64_t target = total_bytes_ * index / active_threads_;
    const auto address = reinterpret_cast<std::uintptr_t>(staging_) + target;
    return ((address + 63U) & ~std::uintptr_t{63U}) -
           reinterpret_cast<std::uintptr_t>(staging_);
  }

  void ProcessSource(std::size_t worker, bool invalidate) const {
    std::uint64_t offset = Boundary(worker);
    const std::uint64_t end = Boundary(worker + 1U);
    auto found = std::upper_bound(offsets_.begin(), offsets_.end(), offset);
    std::size_t index = static_cast<std::size_t>(found - offsets_.begin() - 1U);
    while (offset < end) {
      const GatherSpan& span = (*spans_)[index];
      const std::uint64_t in_span = offset - offsets_[index];
      const std::uint64_t bytes = std::min<std::uint64_t>(
          end - offset, span.length - in_span);
      auto* source = memory_ + (span.address - base_) + in_span;
      if (invalidate) {
        const auto first = reinterpret_cast<std::uintptr_t>(source) & ~std::uintptr_t{63U};
        const auto last = reinterpret_cast<std::uintptr_t>(source) + bytes;
        for (auto line = first; line < last; line += 64U)
          GatherCacheLine(reinterpret_cast<const void*>(line), true);
      } else {
        std::memcpy(staging_ + offset, source, bytes);
      }
      offset += bytes;
      ++index;
    }
  }

  void FlushShard(std::size_t worker) const {
    const auto begin = reinterpret_cast<std::uintptr_t>(staging_) + Boundary(worker);
    const auto end = reinterpret_cast<std::uintptr_t>(staging_) + Boundary(worker + 1U);
    for (auto line = begin & ~std::uintptr_t{63U}; line < end; line += 64U)
      GatherCacheLine(reinterpret_cast<const void*>(line), false);
  }

  void ProcessShard(std::size_t worker, std::uint32_t phase) const {
    if (phase == kInvalidatePhase) ProcessSource(worker, true);
    else if (phase == kCopyPhase) ProcessSource(worker, false);
    else FlushShard(worker);
    if (phase != kCopyPhase) GatherBarrier();
  }

  void RunPhase(std::uint32_t phase) {
    // A phase completes on every core before the next one can touch staging.
    completed_.store(0U, std::memory_order_relaxed);
    phase_.store(phase, std::memory_order_release);
    ProcessShard(0U, phase);
    while (completed_.load(std::memory_order_acquire) != active_threads_ - 1U)
      GatherSpinPause();
  }

  void RunHelper(std::size_t worker) {
    BindWorker(worker);
    std::uint64_t observed_generation = 0U;
    bool was_active = false;
    for (;;) {
      bool active = false;
      if (wait_mode_ == GatherWaitMode::kActive && was_active) {
        std::uint64_t generation = generation_.load(std::memory_order_acquire);
        while (generation == observed_generation) {
          if (stopping_.load(std::memory_order_acquire)) return;
          GatherSpinPause();
          generation = generation_.load(std::memory_order_acquire);
        }
        std::lock_guard<std::mutex> lock(mutex_);
        observed_generation = generation_.load(std::memory_order_acquire);
        active = worker < active_threads_;
      } else {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [this, &observed_generation] {
          return stopping_.load(std::memory_order_acquire) ||
                 generation_.load(std::memory_order_acquire) != observed_generation;
        });
        if (stopping_.load(std::memory_order_acquire)) return;
        observed_generation = generation_.load(std::memory_order_acquire);
        active = worker < active_threads_;
      }
      was_active = active;
      if (!active) continue;
      for (std::uint32_t expected = kInvalidatePhase;
           expected <= kFlushPhase; ++expected) {
        while (phase_.load(std::memory_order_acquire) < expected) {
          if (stopping_.load(std::memory_order_acquire)) return;
          GatherSpinPause();
        }
        ProcessShard(worker, expected);
        completed_.fetch_add(1U, std::memory_order_release);
      }
    }
  }

  std::uint8_t* memory_;
  std::uint64_t base_;
  std::vector<int> cpu_ids_;
  std::size_t max_threads_;
  std::size_t min_bytes_per_thread_;
  GatherWaitMode wait_mode_;
  std::vector<std::thread> helpers_;
  std::mutex mutex_;
  std::condition_variable ready_;
  std::atomic<std::uint64_t> generation_{0U};
  const std::vector<GatherSpan>* spans_{nullptr};
  std::uint8_t* staging_{nullptr};
  std::uint64_t total_bytes_{0U};
  std::size_t active_threads_{1U};
  std::vector<std::uint64_t> offsets_;
  alignas(64) std::atomic<std::uint32_t> phase_{0U};
  alignas(64) std::atomic<std::size_t> completed_{0U};
  std::atomic<bool> stopping_{false};
};

}  // namespace kvaf::peer
