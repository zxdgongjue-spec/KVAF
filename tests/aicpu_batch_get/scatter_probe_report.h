#pragma once

#include "scatter_probe.h"
#include "core_probe_report.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace kvaf::aicpu_test {
inline std::uint64_t ProbeMaxOverlap(const ScatterProbeSample* s, std::uint64_t workers) {
  std::uint64_t maximum = 0;
  for (std::uint64_t i = 0; i < workers; ++i) {
    std::uint64_t count = 0;
    for (std::uint64_t j = 0; j < workers; ++j)
      if (s[j].op_begin <= s[i].op_begin && s[i].op_begin < s[j].op_end) ++count;
    maximum = std::max(maximum, count);
  }
  return maximum;
}
inline void ValidateScatterProbe(const ScatterProbeConfig& c, const ScatterProbeResult& r,
                                 const std::vector<ScatterProbeSample>& samples) {
  ScatterProbeLayout l{};
  if (!MakeScatterProbeLayout(c, &l) || r.magic != c.magic || r.revision != c.revision ||
      !r.counter_frequency || r.completed_workers != c.workers ||
      samples.size() != c.iterations * c.workers)
    throw std::runtime_error("invalid/incomplete scatter probe result");
  for (const auto& s : samples)
    if (!s.invalidate_begin || s.invalidate_issued < s.invalidate_begin || s.invalidate_end < s.invalidate_issued ||
        s.op_begin < s.invalidate_end || s.op_end <= s.op_begin ||
        s.drain_begin < s.op_end || s.drain_end < s.drain_begin ||
        s.clean_begin < s.drain_end || s.clean_issued < s.clean_begin || s.clean_end < s.clean_issued)
      throw std::runtime_error("invalid scatter probe sample timeline");
  if (ProbeSerial(c.mode)) for (std::uint64_t round = 0; round < c.iterations; ++round)
    if (ProbeMaxOverlap(samples.data() + round * c.workers, c.workers) != 1)
      throw std::runtime_error("serialized scatter probe operations overlapped");
}

inline void PrintScatterProbe(std::FILE* out, const ScatterProbeConfig& c,
                              const ScatterProbeResult& r,
                              const std::vector<ScatterProbeSample>& samples,
                              std::uint64_t trace_rounds) {
  ValidateScatterProbe(c, r, samples);
  ScatterProbeLayout l{}; MakeScatterProbeLayout(c, &l);
  const double us_per_tick = 1e6 / static_cast<double>(r.counter_frequency);
  double envelope_ticks = 0, sum_ticks = 0, round_ticks = 0, overlap = 0;
  std::uint64_t maximum_overlap = 0;
  bool seen_cpus[1024]{};
  unsigned distinct_cpus = 0;
  for (std::uint64_t round = 0; round < c.iterations; ++round) {
    const auto* s = samples.data() + round * c.workers;
    auto first = s[0].op_begin, last = s[0].op_end;
    auto begin = s[0].invalidate_begin, end = s[0].clean_end;
    for (std::uint64_t w = 0; w < c.workers; ++w) {
      first = std::min(first, s[w].op_begin); last = std::max(last, s[w].op_end);
      begin = std::min(begin, s[w].invalidate_begin); end = std::max(end, s[w].clean_end);
      sum_ticks += s[w].op_end - s[w].op_begin;
      for (auto cpu : {s[w].first_cpu, s[w].last_cpu})
        if (cpu >= 0 && cpu < 1024 && !seen_cpus[cpu]) { seen_cpus[cpu] = true; ++distinct_cpus; }
    }
    envelope_ticks += last - first; round_ticks += end - begin;
    const auto concurrent = ProbeMaxOverlap(s, c.workers);
    overlap += concurrent; maximum_overlap = std::max(maximum_overlap, concurrent);
    if (round < trace_rounds) for (std::uint64_t w = 0; w < c.workers; ++w)
      std::fprintf(out, "[AICPU_SCATTER_PROBE] TRACE case=%s round=%llu worker=%llu "
          "invalidate_begin_us=%.3f invalidate_issued_us=%.3f invalidate_end_us=%.3f op_begin_us=%.3f op_end_us=%.3f "
          "drain_begin_us=%.3f drain_end_us=%.3f clean_begin_us=%.3f clean_issued_us=%.3f clean_end_us=%.3f "
          "first_cpu=%lld last_cpu=%lld\n",
          ScatterProbeModeName(c.mode), (unsigned long long)round, (unsigned long long)w,
          (s[w].invalidate_begin - begin) * us_per_tick, (s[w].invalidate_issued - begin) * us_per_tick,
          (s[w].invalidate_end - begin) * us_per_tick,
          (s[w].op_begin - begin) * us_per_tick, (s[w].op_end - begin) * us_per_tick,
          (s[w].drain_begin - begin) * us_per_tick, (s[w].drain_end - begin) * us_per_tick,
          (s[w].clean_begin - begin) * us_per_tick, (s[w].clean_issued - begin) * us_per_tick,
          (s[w].clean_end - begin) * us_per_tick,
          (long long)s[w].first_cpu, (long long)s[w].last_cpu);
  }
  const double envelope_us = envelope_ticks * us_per_tick / c.iterations;
  // Payload/envelope is an operation diagnostic, not completed HBM bandwidth.
  // Stores may still be buffered at op_end; fenced copies report drain separately.
  const double payload_gbps = ProbeComputesOnly(c.mode) ? 0 : l.source_bytes / (envelope_us * 1000);
  std::fprintf(out, "[AICPU_SCATTER_PROBE] SUMMARY case=%s workers=%llu rows_per_worker=%llu "
      "row_bytes=%llu output_stride=%llu rounds=%llu frequency=%llu avg_op_envelope_us=%.3f "
      "avg_sum_worker_op_us=%.3f avg_round_envelope_us=%.3f payload_GBps=%.6f "
      "avg_max_overlap=%.3f max_overlap=%llu distinct_cpus=%u memory=acl_normal "
      "source=cold output=cold transport=none phase_waits=outside_operation\n",
      ScatterProbeModeName(c.mode), (unsigned long long)c.workers,
      (unsigned long long)c.rows_per_worker, (unsigned long long)c.row_bytes,
      (unsigned long long)l.stride, (unsigned long long)c.iterations,
      (unsigned long long)r.counter_frequency, envelope_us, sum_ticks * us_per_tick / c.iterations,
      round_ticks * us_per_tick / c.iterations, payload_gbps, overlap / c.iterations,
      (unsigned long long)maximum_overlap, distinct_cpus);
  for (std::uint64_t w = 0; w < c.workers; ++w) {
    double inv = 0, op = 0, drain = 0, clean = 0, cpu_ns = 0;
    double inv_issue = 0, inv_dsb = 0, clean_issue = 0, clean_dsb = 0;
    std::uint64_t cpu_time_errors = 0, cpu_sample_errors = 0, cpu_changes = 0, checksum = 0;
    std::uint64_t cpu_time_errno = 0, cpu_sample_errno = 0;
    std::array<std::uint64_t, 1024> cpu_hits{};
    std::vector<std::uint64_t> durations;
    for (std::uint64_t round = 0; round < c.iterations; ++round) {
      const auto& s = samples[round * c.workers + w];
      inv += s.invalidate_end - s.invalidate_begin; op += s.op_end - s.op_begin;
      inv_issue += s.invalidate_issued - s.invalidate_begin; inv_dsb += s.invalidate_end - s.invalidate_issued;
      clean_issue += s.clean_issued - s.clean_begin; clean_dsb += s.clean_end - s.clean_issued;
      drain += s.drain_end - s.drain_begin; clean += s.clean_end - s.clean_begin;
      cpu_ns += s.op_cpu_ns; cpu_time_errors += s.cpu_time_errno != 0;
      cpu_sample_errors += s.cpu_sample_errno != 0;
      if (!cpu_time_errno) cpu_time_errno = s.cpu_time_errno;
      if (!cpu_sample_errno) cpu_sample_errno = s.cpu_sample_errno;
      for (auto cpu_id : {s.first_cpu, s.last_cpu})
        if (cpu_id >= 0 && cpu_id < 1024) ++cpu_hits[cpu_id];
      cpu_changes += s.first_cpu >= 0 && s.last_cpu >= 0 && s.first_cpu != s.last_cpu;
      checksum ^= s.checksum; durations.push_back(s.op_end - s.op_begin);
    }
    std::sort(durations.begin(), durations.end());
    std::string hits;
    for (std::size_t cpu_id = 0; cpu_id < cpu_hits.size(); ++cpu_id) if (cpu_hits[cpu_id]) {
      if (!hits.empty()) hits += ',';
      hits += std::to_string(cpu_id) + ':' + std::to_string(cpu_hits[cpu_id]);
    }
    const auto cpu = cpu_time_errors ? std::string("unavailable") : std::to_string(cpu_ns / c.iterations / 1000);
    std::fprintf(out, "[AICPU_SCATTER_PROBE] WORKER case=%s id=%llu rows=[%llu,%llu) "
        "avg_invalidate_us=%.3f avg_invalidate_issue_us=%.3f avg_invalidate_dsb_us=%.3f "
        "avg_op_us=%.3f p50_op_us=%.3f p99_op_us=%.3f "
        "avg_drain_us=%.3f avg_clean_us=%.3f avg_op_cpu_us=%s "
        "avg_clean_issue_us=%.3f avg_clean_dsb_us=%.3f "
        "first_cpu=%lld last_cpu=%lld allowed_cpus=%s affinity_errno=%llu "
        "cpu_hits=%s cpu_time_error_rounds=%llu cpu_time_errno=%llu "
        "cpu_sample_error_rounds=%llu cpu_sample_errno=%llu boundary_cpu_changes=%llu checksum=%llu\n",
        ScatterProbeModeName(c.mode), (unsigned long long)w,
        (unsigned long long)(w * c.rows_per_worker), (unsigned long long)((w + 1) * c.rows_per_worker),
        inv * us_per_tick / c.iterations, inv_issue * us_per_tick / c.iterations,
        inv_dsb * us_per_tick / c.iterations, op * us_per_tick / c.iterations,
        durations[(durations.size() - 1) * 50 / 100] * us_per_tick,
        durations[(durations.size() - 1) * 99 / 100] * us_per_tick,
        drain * us_per_tick / c.iterations, clean * us_per_tick / c.iterations, cpu.c_str(),
        clean_issue * us_per_tick / c.iterations, clean_dsb * us_per_tick / c.iterations,
        (long long)r.placement[w].first_cpu, (long long)r.placement[w].last_cpu,
        CoreProbeCpuList(r.placement[w]).c_str(), (unsigned long long)r.placement[w].affinity_errno,
        hits.empty() ? "none" : hits.c_str(), (unsigned long long)cpu_time_errors,
        (unsigned long long)cpu_time_errno, (unsigned long long)cpu_sample_errors,
        (unsigned long long)cpu_sample_errno,
        (unsigned long long)cpu_changes, (unsigned long long)checksum);
    if (c.pmu) for (unsigned event = 0; event < kScatterProbePmuEvents; ++event) {
      if (!(c.pmu & (1ULL << event))) continue;
      std::uint64_t valid = 0, event_errno = 0, read_errno = 0, read_errors = 0;
      double raw = 0, scaled = 0, running_fraction = 0;
      for (std::uint64_t round = 0; round < c.iterations; ++round) {
        const auto& pmu = samples[round * c.workers + w].pmu;
        if (!event_errno) event_errno = pmu.event_errno[event];
        if (!read_errno) read_errno = pmu.read_errno;
        read_errors += pmu.read_errno != 0;
        if (!(pmu.available_mask & (std::uint64_t{1} << event)) || pmu.read_errno ||
            !pmu.enabled_ns || !pmu.running_ns) continue;
        ++valid; raw += pmu.value[event];
        scaled += static_cast<double>(pmu.value[event]) * pmu.enabled_ns / pmu.running_ns;
        running_fraction += static_cast<double>(pmu.running_ns) / pmu.enabled_ns;
      }
      const auto raw_count = valid ? std::to_string(raw / valid) : std::string("unavailable");
      const auto scaled_count = valid ? std::to_string(scaled / valid) : std::string("unavailable");
      std::fprintf(out, "[AICPU_SCATTER_PROBE] PMU case=%s worker=%llu event=%s valid_rounds=%llu "
          "avg_raw=%s avg_scaled=%s avg_running_fraction=%.6f open_errno=%llu read_error_rounds=%llu read_errno=%llu\n",
          ScatterProbeModeName(c.mode), (unsigned long long)w, ScatterProbePmuName(event),
          (unsigned long long)valid, raw_count.c_str(), scaled_count.c_str(),
          valid ? running_fraction / valid : 0, (unsigned long long)event_errno,
          (unsigned long long)read_errors, (unsigned long long)read_errno);
    }
  }
}
}  // namespace kvaf::aicpu_test
