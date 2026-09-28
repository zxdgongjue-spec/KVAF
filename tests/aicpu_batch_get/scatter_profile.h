#pragma once

#include "abi.h"
#include <algorithm>
#include <cstddef>

namespace kvaf::aicpu_test {
constexpr std::uint32_t kProfileCpuLimit = 1024;

// Each worker owns a cache-line-aligned record. CPU hits are boundary samples,
// not time spent on a CPU; migrations away and back between samples are unseen.
struct alignas(64) WorkerProfile {
  std::uint64_t rounds{}, row_begin{}, row_end{};
  std::uint64_t invalidate_ticks{}, memcpy_ticks{}, clean_ticks{};
  std::uint64_t start_offset_ticks{}, finish_offset_ticks{};
  std::uint64_t last_start{}, last_finish{};
  std::uint64_t thread_cpu_ns{}, cpu_time_rounds{}, cpu_time_errors{}, cpu_time_errno{};
  std::uint64_t affinity_errno{}, affinity_capacity{};
  std::uint64_t cpu_sample_errors{}, cpu_sample_errno{}, cpu_out_of_range{};
  std::uint64_t boundary_cpu_changes{};
  std::int64_t first_cpu{-1}, last_cpu{-1};
  std::uint64_t allowed_cpus[kProfileCpuLimit / 64]{};
  std::uint64_t cpu_hits[kProfileCpuLimit]{};
};

struct alignas(64) ScatterProfile {
  std::uint64_t revision{}, workers{}, rounds{};
  std::uint64_t pool_ticks{}, dispatch_ticks{}, envelope_ticks{}, join_ticks{};
  WorkerProfile worker[kMaxScatterThreads];
};
static_assert(sizeof(WorkerProfile) % 64 == 0 && offsetof(ScatterProfile, worker) % 64 == 0);

inline void RecordCpuSample(WorkerProfile& p, int cpu, int error) {
  if (cpu < 0) {
    ++p.cpu_sample_errors;
    if (!p.cpu_sample_errno) p.cpu_sample_errno = error;
    return;
  }
  if (p.first_cpu < 0) p.first_cpu = cpu;
  p.last_cpu = cpu;
  if (static_cast<unsigned>(cpu) < kProfileCpuLimit) ++p.cpu_hits[cpu];
  else ++p.cpu_out_of_range;
}

inline void RecordWorkerTiming(WorkerProfile& p, std::uint64_t dispatch,
                               std::uint64_t start, std::uint64_t invalidated,
                               std::uint64_t copied, std::uint64_t cleaned) {
  ++p.rounds;
  p.invalidate_ticks += invalidated - start;
  p.memcpy_ticks += copied - invalidated;
  p.clean_ticks += cleaned - copied;
  p.start_offset_ticks += start - dispatch;
  p.finish_offset_ticks += cleaned - dispatch;
  p.last_start = start;
  p.last_finish = cleaned;
}

// Called only after all workers have completed this round. Do not add worker
// phase times to get wall time: their intervals can overlap.
inline void RecordPoolTiming(ScatterProfile& p, std::uint64_t dispatch,
                             std::uint64_t returned) {
  auto first = returned;
  auto last = dispatch;
  for (std::uint64_t i = 0; i < p.workers; ++i) {
    first = std::min(first, p.worker[i].last_start);
    last = std::max(last, p.worker[i].last_finish);
  }
  ++p.rounds;
  p.pool_ticks += returned - dispatch;
  p.dispatch_ticks += first - dispatch;
  p.envelope_ticks += last - first;
  p.join_ticks += returned - last;
}
}  // namespace kvaf::aicpu_test
