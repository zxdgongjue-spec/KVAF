#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace kvaf::eid_perf {

struct Stats {
  double mean_ns{0};
  double p50_ns{0};
  double p99_ns{0};
  double mean_gbps{0};
  double p50_gbps{0};
};

inline Stats Summarize(std::vector<std::uint64_t> samples, std::uint64_t payload_bytes) {
  if (samples.empty() || payload_bytes == 0U ||
      std::find(samples.begin(), samples.end(), 0U) != samples.end()) {
    throw std::invalid_argument("bandwidth needs positive payload and timing samples");
  }
  long double total = 0;
  for (auto ns : samples) total += ns;
  std::sort(samples.begin(), samples.end());
  const auto percentile = [&](double fraction) {
    const double position = fraction * (samples.size() - 1U);
    const auto lower = static_cast<std::size_t>(position);
    const auto upper = std::min(lower + 1U, samples.size() - 1U);
    return static_cast<double>(samples[lower]) +
        (samples[upper] - samples[lower]) * (position - lower);
  };
  Stats result;
  result.mean_ns = static_cast<double>(total / samples.size());
  result.p50_ns = percentile(0.50);
  result.p99_ns = percentile(0.99);
  // bytes/ns is decimal GB/s. Divide total bytes by total time, not an
  // arithmetic mean of per-sample bandwidths.
  result.mean_gbps = static_cast<double>(payload_bytes) / result.mean_ns;
  result.p50_gbps = static_cast<double>(payload_bytes) / result.p50_ns;
  return result;
}

inline std::string CsvCell(const std::string& value) {
  if (value.find_first_of(",\"\r\n") == std::string::npos) return value;
  std::string quoted = "\"";
  for (char c : value) { if (c == '"') quoted += '"'; quoted += c; }
  return quoted + '"';
}

}  // namespace kvaf::eid_perf
