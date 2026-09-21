#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include "status.h"

namespace kvaf::detail {

constexpr std::uint32_t kStartupWarmup = 3U;
constexpr std::uint32_t kStartupSamples = 9U;
constexpr std::uint32_t kStartupTokenBytes = 656U;
constexpr std::uint32_t kStartupTokensPerBlock = 128U;
constexpr std::uint32_t kStartupAivCores = 2U;
constexpr std::uint32_t kStartupJettysPerCore = 1U;
constexpr std::uint64_t kStartupHugePageBytes = 2ULL * 1024U * 1024U;

struct StartupProbeConfig {
  std::uint32_t token_bytes;
  std::uint32_t tokens_per_block;
  std::uint32_t miss_count;
};

struct StartupProbeLayout {
  std::uint64_t block_bytes{0U};
  std::uint64_t source_bytes{0U};
  std::uint64_t ddr_bytes{0U};
};

inline Status BuildStartupProbeLayout(const StartupProbeConfig& config,
                                      StartupProbeLayout* layout) {
  if (layout == nullptr || config.token_bytes == 0U ||
      config.tokens_per_block == 0U || config.miss_count == 0U ||
      config.miss_count > static_cast<std::uint32_t>(
                              std::numeric_limits<std::int32_t>::max())) {
    return Status::InvalidArgument("invalid adaptive UB probe shape");
  }
  const std::uint64_t block_bytes =
      static_cast<std::uint64_t>(config.token_bytes) * config.tokens_per_block;
  if ((static_cast<std::uint64_t>(config.tokens_per_block) - 1U) *
          config.token_bytes >
      static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
    return Status::InvalidArgument("adaptive UB probe token offset exceeds int32");
  }
  const std::uint64_t blocks =
      (static_cast<std::uint64_t>(config.miss_count) +
       config.tokens_per_block - 1U) /
      config.tokens_per_block;
  if (blocks > static_cast<std::uint64_t>(
                   std::numeric_limits<std::int32_t>::max()) + 1U ||
      block_bytes > std::numeric_limits<std::uint64_t>::max() / blocks) {
    return Status::InvalidArgument("adaptive UB probe source size overflows");
  }
  const std::uint64_t source_bytes = blocks * block_bytes;
  const std::uint64_t gcd = std::gcd(block_bytes, kStartupHugePageBytes);
  const std::uint64_t alignment_factor = block_bytes / gcd;
  if (alignment_factor >
      std::numeric_limits<std::uint64_t>::max() / kStartupHugePageBytes) {
    return Status::InvalidArgument("adaptive UB probe alignment overflows");
  }
  const std::uint64_t alignment = alignment_factor * kStartupHugePageBytes;
  if (source_bytes > std::numeric_limits<std::uint64_t>::max() -
                         (alignment - 1U)) {
    return Status::InvalidArgument("adaptive UB probe DDR size overflows");
  }
  const std::uint64_t ddr_bytes =
      ((source_bytes + alignment - 1U) / alignment) * alignment;
  const std::uint64_t payload_bytes =
      static_cast<std::uint64_t>(config.token_bytes) * config.miss_count;
  if (source_bytes > std::numeric_limits<std::size_t>::max() ||
      ddr_bytes > std::numeric_limits<std::size_t>::max() ||
      payload_bytes > std::numeric_limits<std::size_t>::max()) {
    return Status::InvalidArgument("adaptive UB probe buffers exceed size_t");
  }
  *layout = StartupProbeLayout{block_bytes, source_bytes, ddr_bytes};
  return Status::OK();
}

struct StartupCandidate {
  std::uint32_t peer;
  std::size_t local_group;
  double cycles{0.0};
};

// Adaptive selection treats a peer device as the performance candidate. The
// entries within each device are connectivity fallbacks (K EID x N EID group),
// so only the first working entry for that device is benchmarked.
template <class Initialize, class Cleanup>
Status SelectStartupDeviceCandidate(
    const std::vector<std::vector<StartupCandidate>>& devices,
    Initialize initialize, Cleanup cleanup) {
  Status last = Status::NotFound("no usable peer device");
  std::vector<StartupCandidate> ranked;
  for (const auto& device : devices) {
    for (auto candidate : device) {
      bool device_ready = false;
      last = initialize(candidate, true, &candidate.cycles);
      if (last.ok()) {
        if (std::isfinite(candidate.cycles) && candidate.cycles > 0.0) {
          ranked.push_back(candidate);
          device_ready = true;
        } else {
          last = Status::Internal("startup probe returned invalid kernel timing");
        }
      }
      const Status released = cleanup();
      if (!released.ok()) return released;
      if (device_ready) break;
    }
  }
  std::stable_sort(ranked.begin(), ranked.end(),
                   [](const auto& a, const auto& b) { return a.cycles < b.cycles; });
  for (auto candidate : ranked) {
    last = initialize(candidate, false, &candidate.cycles);
    if (last.ok()) return last;
    const Status released = cleanup();
    if (!released.ok()) return released;
  }
  return last;
}

}  // namespace kvaf::detail
