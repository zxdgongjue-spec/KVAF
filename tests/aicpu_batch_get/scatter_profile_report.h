#pragma once

#include "scatter_profile.h"
#include <cstdio>
#include <stdexcept>
#include <string>

namespace kvaf::aicpu_test {
inline void ValidateScatterProfile(const Config& config, const ScatterProfile& p) {
  if (p.revision != kKernelRevision || p.workers != ActiveThreads(config) ||
      p.workers == 0 || p.workers > kMaxScatterThreads || p.rounds != config.iterations ||
      p.rounds == 0 || p.pool_ticks == 0 ||
      p.pool_ticks != p.dispatch_ticks + p.envelope_ticks + p.join_ticks)
    throw std::runtime_error("invalid/incomplete scatter profile");
  for (std::uint64_t i = 0; i < p.workers; ++i) {
    const auto& w = p.worker[i];
    const auto range = ScatterRange(config, i, p.workers);
    std::uint64_t hits = w.cpu_sample_errors + w.cpu_out_of_range;
    for (auto count : w.cpu_hits) hits += count;
    if (w.rounds != p.rounds || w.row_begin != range.begin || w.row_end != range.end ||
        hits != 2 * w.rounds || w.cpu_time_rounds + w.cpu_time_errors != w.rounds ||
        w.affinity_capacity > kProfileCpuLimit ||
        w.finish_offset_ticks < w.start_offset_ticks ||
        w.finish_offset_ticks - w.start_offset_ticks !=
            w.invalidate_ticks + w.memcpy_ticks + w.clean_ticks)
      throw std::runtime_error("invalid/incomplete worker profile");
  }
}

inline std::string ProfileCpuList(const WorkerProfile& w, bool allowed) {
  std::string list;
  if (allowed && w.affinity_errno) return "unavailable";
  for (unsigned cpu = 0; cpu < kProfileCpuLimit; ++cpu) {
    const bool present = allowed ? (w.allowed_cpus[cpu / 64] & (std::uint64_t{1} << (cpu % 64))) != 0
                                 : w.cpu_hits[cpu] != 0;
    if (!present) continue;
    if (!list.empty()) list += ',';
    list += std::to_string(cpu);
    if (!allowed) list += ':' + std::to_string(w.cpu_hits[cpu]);
  }
  return list.empty() ? "none" : list;
}

inline void PrintScatterProfile(std::FILE* out, const Config& config,
                                std::uint64_t frequency, const ScatterProfile& p) {
  ValidateScatterProfile(config, p);
  if (!frequency) throw std::runtime_error("missing scatter profile counter frequency");
  if (IsScatterPipeline(config))
    std::fprintf(out, "[AICPU_PROFILE] SEMANTICS mode=scatter_pipeline "
        "start_origin=request pool_includes=range_ready_wait "
        "envelope=scatter_span thread_cpu=scatter_only\n");
  const double us_per_tick_per_round = 1e6 / static_cast<double>(frequency) / p.rounds;
  std::uint64_t cpu_ns = 0;
  bool cpu_time_valid = true;
  for (std::uint64_t i = 0; i < p.workers; ++i) {
    cpu_ns += p.worker[i].thread_cpu_ns;
    cpu_time_valid &= p.worker[i].cpu_time_rounds == p.rounds;
  }
  const auto parallelism = cpu_time_valid ?
      std::to_string(static_cast<double>(cpu_ns) / (p.pool_ticks * (1e9 / frequency))) :
      std::string("unavailable");
  std::fprintf(out, "[AICPU_PROFILE] SUMMARY threads=%llu rounds=%llu avg_pool_us=%.3f "
                   "avg_dispatch_us=%.3f avg_envelope_us=%.3f avg_join_us=%.3f cpu_parallelism=%s\n",
               (unsigned long long)p.workers, (unsigned long long)p.rounds,
               p.pool_ticks * us_per_tick_per_round, p.dispatch_ticks * us_per_tick_per_round,
               p.envelope_ticks * us_per_tick_per_round, p.join_ticks * us_per_tick_per_round,
               parallelism.c_str());
  for (std::uint64_t i = 0; i < p.workers; ++i) {
    const auto& w = p.worker[i];
    const auto allowed = ProfileCpuList(w, true), hits = ProfileCpuList(w, false);
    const auto thread_cpu = w.cpu_time_rounds ?
        std::to_string(w.thread_cpu_ns / (1000.0 * w.cpu_time_rounds)) : std::string("unavailable");
    std::fprintf(out, "[AICPU_PROFILE] WORKER id=%llu rows=[%llu,%llu) rounds=%llu "
                     "avg_invalidate_us=%.3f avg_memcpy_us=%.3f avg_clean_us=%.3f "
                     "avg_start_us=%.3f avg_finish_us=%.3f avg_thread_cpu_us=%s "
                     "allowed_cpus=%s affinity_errno=%llu affinity_capacity=%llu "
                     "cpu_hits=%s first_cpu=%lld last_cpu=%lld boundary_cpu_changes=%llu "
                     "cpu_sample_errors=%llu cpu_sample_errno=%llu cpu_out_of_range=%llu "
                     "cpu_time_rounds=%llu cpu_time_errors=%llu cpu_time_errno=%llu\n",
                 (unsigned long long)i, (unsigned long long)w.row_begin, (unsigned long long)w.row_end,
                 (unsigned long long)w.rounds, w.invalidate_ticks * us_per_tick_per_round,
                 w.memcpy_ticks * us_per_tick_per_round, w.clean_ticks * us_per_tick_per_round,
                 w.start_offset_ticks * us_per_tick_per_round, w.finish_offset_ticks * us_per_tick_per_round,
                 thread_cpu.c_str(), allowed.c_str(), (unsigned long long)w.affinity_errno,
                 (unsigned long long)w.affinity_capacity, hits.c_str(),
                 (long long)w.first_cpu, (long long)w.last_cpu,
                 (unsigned long long)w.boundary_cpu_changes, (unsigned long long)w.cpu_sample_errors,
                 (unsigned long long)w.cpu_sample_errno, (unsigned long long)w.cpu_out_of_range,
                 (unsigned long long)w.cpu_time_rounds, (unsigned long long)w.cpu_time_errors,
                 (unsigned long long)w.cpu_time_errno);
  }
}
}  // namespace kvaf::aicpu_test
