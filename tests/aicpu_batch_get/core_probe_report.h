#pragma once

#include "abi.h"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace kvaf::aicpu_test {
inline std::string CoreProbeCpuList(const CoreProbeSlot& slot) {
  if (slot.affinity_errno) return "unavailable";
  std::string list;
  for (unsigned cpu = 0; cpu < 1024; ++cpu) {
    if (!(slot.allowed_cpus[cpu / 64] & (std::uint64_t{1} << (cpu % 64)))) continue;
    if (!list.empty()) list += ',';
    list += std::to_string(cpu);
  }
  return list.empty() ? "none" : list;
}

inline std::uint32_t CoreProbeDistinctCpus(const CoreProbeResult& result) {
  bool seen[1024]{};
  std::uint32_t count = 0;
  for (std::uint64_t i = 0; i < result.completed && i < kMaxScatterThreads; ++i) {
    const std::int64_t samples[] = {result.slot[i].first_cpu, result.slot[i].last_cpu};
    for (const auto cpu : samples) {
      if (cpu < 0 || cpu >= 1024 || seen[cpu]) continue;
      seen[cpu] = true; ++count;
    }
  }
  return count;
}

inline std::uint32_t CoreProbeMaxOverlap(const CoreProbeResult& result) {
  std::uint32_t maximum = 0;
  for (std::uint64_t i = 0; i < result.completed && i < kMaxScatterThreads; ++i) {
    const auto tick = result.slot[i].start_tick;
    if (result.slot[i].finish_tick <= tick) continue;
    std::uint32_t concurrent = 0;
    for (std::uint64_t j = 0; j < result.completed && j < kMaxScatterThreads; ++j)
      if (result.slot[j].start_tick <= tick && tick < result.slot[j].finish_tick) ++concurrent;
    if (concurrent > maximum) maximum = concurrent;
  }
  return maximum;
}

inline void PrintCoreProbe(std::FILE* out, const CoreProbeResult& result,
                           std::uint32_t requested_blocks) {
  if (!IsValidCoreProbeResult(result, requested_blocks))
    throw std::runtime_error("invalid AICPU core probe result");
  const auto complete = result.claimed == requested_blocks && result.completed == requested_blocks;
  const auto distinct = CoreProbeDistinctCpus(result);
  const auto overlap = CoreProbeMaxOverlap(result);
  std::fprintf(out, "[AICPU_CORE_PROBE] SUMMARY requested_blocks=%u claimed=%llu completed=%llu "
                    "frequency=%llu hold_ticks=%llu distinct_cpus=%u max_overlap=%u complete=%u multicore_ready=%u status=%llu\n",
               requested_blocks, (unsigned long long)result.claimed,
               (unsigned long long)result.completed, (unsigned long long)result.counter_frequency,
               (unsigned long long)result.hold_ticks, distinct, overlap, complete ? 1U : 0U,
               complete && distinct >= requested_blocks && overlap >= requested_blocks ? 1U : 0U,
               (unsigned long long)result.status);
  for (std::uint64_t i = 0; i < result.completed && i < kMaxScatterThreads; ++i) {
    const auto& slot = result.slot[i];
    std::fprintf(out, "[AICPU_CORE_PROBE] BLOCK id=%llu start_tick=%llu finish_tick=%llu "
                      "first_cpu=%lld last_cpu=%lld allowed_cpus=%s affinity_errno=%llu "
                      "cpu_sample_errors=%llu cpu_sample_errno=%llu cpu_out_of_range=%llu\n",
                 (unsigned long long)slot.slot, (unsigned long long)slot.start_tick,
                 (unsigned long long)slot.finish_tick, (long long)slot.first_cpu,
                 (long long)slot.last_cpu, CoreProbeCpuList(slot).c_str(),
                 (unsigned long long)slot.affinity_errno,
                 (unsigned long long)slot.cpu_sample_errors,
                 (unsigned long long)slot.cpu_sample_errno,
                 (unsigned long long)slot.cpu_out_of_range);
  }
}
}  // namespace kvaf::aicpu_test
