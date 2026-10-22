#pragma once
#include "control.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace kvaf::aicpu_test {
inline Report Summarize(const Config& config, const KernelResult& result,
                        const std::vector<Sample>& samples) {
  Layout layout{};
  Require(MakeLayout(config, &layout) && result.status == Result::kOk &&
          result.completed_rounds == layout.rounds && result.counter_frequency > 0 &&
          result.kernel_revision == kKernelRevision &&
          result.active_threads == ActiveThreads(config) &&
          samples.size() == config.iterations, "invalid/incomplete device timings");
  Report report{};
  report.active_threads = result.active_threads;
  std::vector<double> times;
  const double ns_per_tick = 1e9 / static_cast<double>(result.counter_frequency);
  for (const auto& sample : samples) {
    Require(sample.total_ticks > 0 && sample.wait_ticks <= sample.total_ticks &&
            sample.scatter_ticks == sample.total_ticks - sample.wait_ticks,
            "invalid timing sample");
    const auto ns = sample.total_ticks * ns_per_tick;
    times.push_back(ns);
    report.mean_ns += ns;
    report.wait_ns += sample.wait_ticks * ns_per_tick;
    report.scatter_ns += sample.scatter_ticks * ns_per_tick;
    report.prepare_ns += sample.prepare_ticks * ns_per_tick;
    report.verify_ns += sample.verify_ticks * ns_per_tick;
  }
  std::sort(times.begin(), times.end());
  report.mean_ns /= times.size(); report.wait_ns /= times.size(); report.scatter_ns /= times.size();
  report.p50_ns = times[static_cast<std::size_t>(std::ceil(times.size() * .50)) - 1];
  report.p99_ns = times[static_cast<std::size_t>(std::ceil(times.size() * .99)) - 1];
  report.prepare_ns /= times.size(); report.verify_ns /= times.size();
  report.gbps = static_cast<double>(layout.payload_bytes) / report.mean_ns;
  report.rounds = layout.rounds; report.status = 0;
  return report;
}
}  // namespace kvaf::aicpu_test
