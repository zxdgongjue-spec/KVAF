#pragma once

#include <time.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "cpu_gather.h"

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

struct alignas(64) GatherCopyWorkerProfile {
  std::uint64_t bytes{0U};
  std::uint64_t begin_ns{0U}, end_ns{0U};
  std::uint64_t thread_cpu_ns{0U};
  bool thread_cpu_available{false};
  int start_cpu{-1}, end_cpu{-1};
};

struct GatherCopyProfile {
  std::uint64_t begin_ns{0U}, end_ns{0U}, join_wait_ns{0U};
  std::size_t workers{0U};
};

class CpuGatherTeam {
 public:
  CpuGatherTeam(void* memory, std::uint64_t base, std::size_t max_threads,
                std::size_t min_bytes_per_thread = 256U * 1024U,
                GatherWaitMode wait_mode = GatherWaitMode::kPassive, bool bind_cores = false,
                bool profile_copy = false, std::size_t cpu_offset = 0U)
      : memory_(static_cast<std::uint8_t*>(memory)),
        base_(base),
        cpu_ids_(bind_cores ? AllowedCpus(cpu_offset) : std::vector<int>{}),
        max_threads_(
            std::min(std::max<std::size_t>(1U, max_threads),
                     cpu_ids_.empty() ? std::max<std::size_t>(1U, max_threads) : cpu_ids_.size())),
        min_bytes_per_thread_(std::max<std::size_t>(1U, min_bytes_per_thread)),
        wait_mode_(wait_mode),
        copy_profiles_(profile_copy ? new GatherCopyWorkerProfile[max_threads_] : nullptr) {}

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
  const GatherCopyProfile& copy_profile() const { return copy_profile_; }
  const GatherCopyWorkerProfile& copy_worker_profile(std::size_t worker) const {
    return copy_profiles_[worker];
  }

  std::size_t PrepareDirect(std::uint64_t region_bytes, std::uint64_t descriptor_address,
                            std::uint64_t sequence, std::size_t max_spans);

  void BindWorker(std::size_t worker) const {
#if defined(__linux__)
    const long tid = syscall(SYS_gettid);
    if (worker >= cpu_ids_.size()) {
      std::fprintf(stderr, "[KVAF_CPU] GATHER_THREAD worker=%zu tid=%ld cpu=%d pinned=0\n", worker,
                   tid, sched_getcpu());
      return;
    }
    cpu_set_t mask{};
    CPU_ZERO(&mask);
    CPU_SET(cpu_ids_[worker], &mask);
    const int result = sched_setaffinity(0, sizeof(mask), &mask);
    std::fprintf(stderr, "[KVAF_CPU] GATHER_THREAD worker=%zu tid=%ld cpu=%d pinned=%d error=%d\n",
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

  void Copy();

 private:
  std::size_t Publish(std::uint64_t staging, const std::vector<GatherSpan>& spans,
                      std::uint32_t total_bytes) {
    const std::size_t threads =
        std::min(max_threads_, std::max<std::size_t>(1U, total_bytes / min_bytes_per_thread_));
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

  std::uint64_t Boundary(std::size_t index) const {
    if (index == 0U) return 0U;
    if (index == active_threads_) return total_bytes_;
    const std::uint64_t target = total_bytes_ * index / active_threads_;
    const auto address = reinterpret_cast<std::uintptr_t>(staging_) + target;
    return ((address + 63U) & ~std::uintptr_t{63U}) - reinterpret_cast<std::uintptr_t>(staging_);
  }

  std::uint64_t CopyRange(std::uint64_t offset, std::uint64_t end) const;

  std::uint64_t ProcessSource(std::size_t worker);
  void RunSource(std::size_t worker);

  void RunHelper(std::size_t worker);

  std::uint8_t* memory_;
  std::uint64_t base_;
  std::vector<int> cpu_ids_;
  std::size_t max_threads_;
  std::size_t min_bytes_per_thread_;
  GatherWaitMode wait_mode_;
  std::unique_ptr<GatherCopyWorkerProfile[]> copy_profiles_;
  GatherCopyProfile copy_profile_{};
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
