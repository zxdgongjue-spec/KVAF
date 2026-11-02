#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <utility>
#include <vector>

namespace kvaf::offload {

struct TimingDistribution {
  double mean{}, p50{}, p95{}, p99{}, min{}, max{};
};

// Linear interpolation at (N - 1) * p, also used by the Python benchmark.
inline TimingDistribution SummarizeTimings(std::vector<double> values) {
  if (values.empty()) return {};
  std::sort(values.begin(), values.end());
  auto percentile = [&](double p) {
    const double index = (values.size() - 1) * p;
    const auto lower = static_cast<std::size_t>(index);
    const auto upper = std::min(lower + 1, values.size() - 1);
    return values[lower] + (values[upper] - values[lower]) * (index - lower);
  };
  return {std::accumulate(values.begin(), values.end(), 0.0) / values.size(),
          percentile(.50), percentile(.95), percentile(.99), values.front(), values.back()};
}

// One source connection. Probes and checksum RPCs never enter this collector.
// Samples retain only timings; percentile sorting happens outside READ timing.
class CpuPullProfileSummary {
 public:
  void Reset(std::uint64_t warmup) {
    warmup_ = warmup;
    attempts_ = measured_attempts_ = failures_ = measured_failures_ = dropped_ = measured_dropped_ = reported_ = 0;
    samples_.clear();
  }

  void Record(bool success, double build, double post, double cq, double total) noexcept {
    const bool measured = ++attempts_ > warmup_;
    measured_attempts_ += measured;
    if (!success) { ++failures_; measured_failures_ += measured; return; }
    // Profiling must not change the transfer result if storage cannot grow.
    try {
      samples_.push_back({{build, post, cq, std::max(0.0, total-build-post-cq), total}, measured});
    } catch (...) { ++dropped_; measured_dropped_ += measured; }
  }

  void Report(FILE* output, std::uint64_t id, std::uint64_t source_base,
              std::uint64_t ddr_base, const char* reason) noexcept {
    if (!attempts_ || attempts_ == reported_) return;
    try {
      for (bool measured : {false, true}) {
        const char* scope = measured ? "measured" : "all";
        std::size_t count = 0;
        for (const auto& sample : samples_) count += !measured || sample.measured;
        std::fprintf(output, "[KVAF_CPU] CPU_PULL_SUMMARY profile_id=%llu scope=%s reason=%s "
            "source_base=0x%llx ddr_base=0x%llx attempts=%llu samples=%zu failed=%llu "
            "warmup_config=%llu warmup_seen=%llu dropped_samples=%llu percentile=linear "
            "units=us checksum_included=0 probes_included=0\n",
            (unsigned long long)id, scope, reason, (unsigned long long)source_base,
            (unsigned long long)ddr_base, (unsigned long long)(measured ? measured_attempts_ : attempts_),
            count, (unsigned long long)(measured ? measured_failures_ : failures_),
            (unsigned long long)warmup_, (unsigned long long)std::min(warmup_, attempts_),
            (unsigned long long)(measured ? measured_dropped_ : dropped_));
        if (!count) continue;
        constexpr const char* names[] = {"build", "post", "cq_wait", "other", "total"};
        for (std::size_t stage = 0; stage < 5; ++stage) {
          std::vector<double> values;
          values.reserve(count);
          for (const auto& sample : samples_)
            if (!measured || sample.measured) values.push_back(sample.us[stage]);
          const auto stats = SummarizeTimings(std::move(values));
          std::fprintf(output, "[KVAF_CPU] CPU_PULL_SUMMARY_STAGE profile_id=%llu scope=%s "
              "stage=%s samples=%zu mean_us=%.3f p50_us=%.3f p95_us=%.3f p99_us=%.3f "
              "min_us=%.3f max_us=%.3f\n", (unsigned long long)id, scope, names[stage], count,
              stats.mean, stats.p50, stats.p95, stats.p99, stats.min, stats.max);
        }
      }
      reported_ = attempts_;
    } catch (...) {
      std::fprintf(output, "[KVAF_CPU] CPU_PULL_SUMMARY_ERROR profile_id=%llu reason=allocation_failed\n",
                   (unsigned long long)id);
    }
  }

  void ReleaseSamples() { std::vector<Sample>().swap(samples_); }

 private:
  struct Sample { std::array<double, 5> us; bool measured; };
  std::uint64_t warmup_{}, attempts_{}, measured_attempts_{}, failures_{}, measured_failures_{};
  std::uint64_t dropped_{}, measured_dropped_{}, reported_{};
  std::vector<Sample> samples_;
};
}  // namespace kvaf::offload
