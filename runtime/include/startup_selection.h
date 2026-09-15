#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "status.h"

namespace kvaf::detail {

constexpr std::uint32_t kStartupTokenBytes = 656U;
constexpr std::uint32_t kStartupTokensPerBlock = 128U;
constexpr std::uint32_t kStartupMissCount = 2048U;
constexpr std::uint32_t kStartupWarmup = 1U;
constexpr std::uint32_t kStartupSamples = 5U;
// LCM(656 * 128, 2 MiB): valid both for slot allocation and hugetlb-2m.
constexpr std::uint64_t kStartupDdrBytes = 82ULL * 1024U * 1024U;
static_assert(kStartupDdrBytes % (kStartupTokenBytes * kStartupTokensPerBlock) == 0U);
static_assert(kStartupDdrBytes % (2U * 1024U * 1024U) == 0U);

struct StartupCandidate {
  std::uint32_t peer;
  std::size_t local_group;
  double cycles{0.0};
};

inline constexpr bool ShouldTuneStartupCandidates(bool adaptive_requested,
                                                  bool device_filter_applied) {
  return adaptive_requested && device_filter_applied;
}

// The callback leaves resources alive, even on failure. Cleanup must finish
// before trying another pair; a cleanup failure is never treated as a slow link.
// Successful final initialization retains resources for the caller's session.
template <class Initialize, class Cleanup>
Status SelectStartupCandidate(bool adaptive,
                              const std::vector<StartupCandidate>& candidates,
                              Initialize initialize, Cleanup cleanup) {
  Status last = Status::NotFound("no usable peer candidate");
  std::vector<StartupCandidate> ranked;
  for (auto candidate : candidates) {
    last = initialize(candidate, adaptive, &candidate.cycles);
    if (last.ok() && !adaptive) return last;
    if (last.ok()) {
      if (std::isfinite(candidate.cycles) && candidate.cycles > 0.0) {
        ranked.push_back(candidate);
      } else {
        last = Status::Internal("startup probe returned invalid kernel timing");
      }
    }
    const Status released = cleanup();
    if (!released.ok()) return released;
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
