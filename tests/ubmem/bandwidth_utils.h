#pragma once
#include "ubmem/raw_copy_abi.h"

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

namespace kvaf::ubmem::perf {
// Uniform token sampling is equivalent to the reference benchmark's random
// block + random token selection. Retain offsets only for untimed verification.
inline std::vector<std::uint64_t> MakeRequests(
    std::vector<RawCopyRequest>& requests, std::uint64_t remote_base, std::uint64_t local_base,
    std::uint64_t remote_tokens, std::uint32_t row_bytes, bool random,
    std::uint64_t iteration, std::mt19937_64& generator) {
  std::uniform_int_distribution<std::uint64_t> pick(0, remote_tokens - 1);
  std::vector<std::uint64_t> offsets(requests.size());
  for (std::size_t row = 0; row < requests.size(); ++row) {
    const auto token = random ? pick(generator) : (iteration * requests.size() + row) % remote_tokens;
    offsets[row] = token * row_bytes;
    requests[row] = {remote_base + offsets[row], local_base + row * row_bytes};
  }
  return offsets;
}

// Matches the reference benchmark's linearly interpolated percentiles.
inline double Percentile(const std::vector<double>& sorted, double fraction) {
  const double position = fraction * (sorted.size() - 1);
  const auto lower = static_cast<std::size_t>(position);
  const auto upper = std::min(lower + 1, sorted.size() - 1);
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}
inline double GBps(std::uint64_t payload_bytes, double microseconds) {
  return static_cast<double>(payload_bytes) / microseconds / 1000.0;
}
}  // namespace kvaf::ubmem::perf
