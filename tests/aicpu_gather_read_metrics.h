#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace kvaf::aicpu_urma_client {

struct TimingStats {
  double mean_ms;
  double min_ms;
  double p50_ms;
  double p95_ms;
  double p99_ms;
  double max_ms;
  double mean_gbps;
};

inline TimingStats SummarizeTiming(std::vector<double> samples, std::size_t payload_bytes) {
  if (samples.empty() || payload_bytes == 0)
    throw std::invalid_argument("timing statistics need samples and a payload");
  long double total_ms = 0;
  for (double sample : samples) {
    if (!std::isfinite(sample) || sample <= 0)
      throw std::invalid_argument("timing samples must be positive and finite");
    total_ms += sample;
  }
  std::sort(samples.begin(), samples.end());
  const auto percentile = [&](double fraction) {
    const double position = fraction * (samples.size() - 1U);
    const auto lower = static_cast<std::size_t>(position);
    const auto upper = std::min(lower + 1U, samples.size() - 1U);
    return samples[lower] + (samples[upper] - samples[lower]) * (position - lower);
  };
  const double mean_ms = static_cast<double>(total_ms / samples.size());
  return {mean_ms,
          samples.front(),
          percentile(0.50),
          percentile(0.95),
          percentile(0.99),
          samples.back(),
          static_cast<double>(payload_bytes) / (mean_ms * 1.0e6)};
}

}  // namespace kvaf::aicpu_urma_client
