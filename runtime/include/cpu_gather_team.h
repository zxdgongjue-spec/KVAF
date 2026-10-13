#pragma once

#include "cpu_gather.h"

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

#if defined(__linux__)
#include <cerrno>
#include <cstdio>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
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
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_.store(true, std::memory_order_release);
    }
    ready_.notify_all();
    for (std::thread& helper : helpers_) helper.join();
  }

  CpuGatherTeam(const CpuGatherTeam&) = delete;
  CpuGatherTeam& operator=(const CpuGatherTeam&) = delete;

  std::size_t max_threads() const { return max_threads_; }
  std::uint64_t total_bytes() const { return total_bytes_; }

  std::size_t PrepareDirect(std::uint64_t region_bytes, std::uint64_t descriptor_address,
                            std::uint64_t sequence, std::size_t max_spans);

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
    pipeline_ready_ = nullptr;
    offsets_.resize(spans.size() + 1U);
    offsets_[0U] = 0U;
    for (std::size_t index = 0U; index < spans.size(); ++index)
      offsets_[index + 1U] = offsets_[index] + spans[index].length;
    return Publish(staging, spans, total_bytes);
  }

  void Copy() {
    completed_.store(0U, std::memory_order_relaxed);
    phase_.store(1U, std::memory_order_release);
    ProcessSource(0U);
    GatherBarrier();
    while (completed_.load(std::memory_order_acquire) != active_threads_ - 1U)
      GatherSpinPause();
  }

 private:
  std::size_t Publish(std::uint64_t staging, const std::vector<GatherSpan>& spans,
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
    phase_.store(0U, std::memory_order_release);
    generation_.fetch_add(1U, std::memory_order_release);
    lock.unlock();
    ready_.notify_all();
    return threads;
  }

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

  void CopyRange(std::uint64_t offset, std::uint64_t end) const {
    auto found = std::upper_bound(offsets_.begin(), offsets_.end(), offset);
    std::size_t index = static_cast<std::size_t>(found - offsets_.begin() - 1U);
    while (offset < end) {
      const GatherSpan& span = (*spans_)[index];
      const std::uint64_t in_span = offset - offsets_[index];
      const std::uint64_t bytes = std::min<std::uint64_t>(
          end - offset, span.length - in_span);
      auto* source = memory_ + (span.address - base_) + in_span;
      std::memcpy(staging_ + offset, source, bytes);
      offset += bytes;
      ++index;
    }
  }

  void ProcessSource(std::size_t worker) {
    if (pipeline_ready_ == nullptr) {
      CopyRange(Boundary(worker), Boundary(worker + 1U));
      return;
    }
    for (;;) {
      const auto chunk = next_chunk_.fetch_add(1U, std::memory_order_relaxed);
      if (chunk >= chunk_done_.size()) return;
      const auto begin = static_cast<std::uint64_t>(chunk) * pipeline_chunk_bytes_;
      CopyRange(begin, std::min(begin + pipeline_chunk_bytes_, total_bytes_));
      GatherBarrier();
      __atomic_store_n(&chunk_done_[chunk], 1U, __ATOMIC_RELEASE);
      auto observed = __atomic_load_n(pipeline_ready_, __ATOMIC_ACQUIRE);
      auto prefix = static_cast<std::size_t>(observed / pipeline_chunk_bytes_);
      while (prefix < chunk_done_.size() &&
             __atomic_load_n(&chunk_done_[prefix], __ATOMIC_ACQUIRE) != 0U)
        ++prefix;
      const auto ready_bytes = std::min(total_bytes_,
          static_cast<std::uint64_t>(prefix) * pipeline_chunk_bytes_);
      while (observed < ready_bytes) {
        if (!__atomic_compare_exchange_n(pipeline_ready_, &observed, ready_bytes, false,
                                         __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) continue;
        GatherBarrier();
        break;
      }
    }
  }

  void RunHelper(std::size_t worker) {
    BindWorker(worker);
    std::uint64_t observed_generation = 0U;
    bool was_active = false;
    for (;;) {
      bool active = false;
      if (wait_mode_ != GatherWaitMode::kPassive && was_active) {
        const auto spin_start = std::chrono::steady_clock::now();
        std::uint32_t spin_count = 0U;
        std::uint64_t generation = generation_.load(std::memory_order_acquire);
        while (generation == observed_generation) {
          if (stopping_.load(std::memory_order_acquire)) return;
          if (wait_mode_ == GatherWaitMode::kAdaptive &&
              (++spin_count & 255U) == 0U &&
              std::chrono::steady_clock::now() - spin_start >=
                  std::chrono::microseconds(100)) break;
          GatherSpinPause();
          generation = generation_.load(std::memory_order_acquire);
        }
        std::unique_lock<std::mutex> lock(mutex_);
        if (generation_.load(std::memory_order_acquire) == observed_generation)
          ready_.wait(lock, [this, &observed_generation] {
            return stopping_.load(std::memory_order_acquire) ||
                   generation_.load(std::memory_order_acquire) != observed_generation;
          });
        if (stopping_.load(std::memory_order_acquire)) return;
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
      while (phase_.load(std::memory_order_acquire) == 0U) {
        if (stopping_.load(std::memory_order_acquire)) return;
        GatherSpinPause();
      }
      ProcessSource(worker);
      GatherBarrier();
      completed_.fetch_add(1U, std::memory_order_release);
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
  std::vector<GatherSpan> direct_spans_;
  std::uint8_t* staging_{nullptr};
  std::uint64_t total_bytes_{0U};
  std::size_t active_threads_{1U};
  std::vector<std::uint64_t> offsets_;
  std::uint64_t* pipeline_ready_{nullptr};
  std::uint32_t pipeline_chunk_bytes_{0U};
  std::vector<std::uint32_t> chunk_done_;
  std::atomic<std::size_t> next_chunk_{0U};
  alignas(64) std::atomic<std::uint32_t> phase_{0U};
  alignas(64) std::atomic<std::size_t> completed_{0U};
  std::atomic<bool> stopping_{false};
};

}  // namespace kvaf::peer
