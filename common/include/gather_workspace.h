#pragma once
#include <cstdint>
#include <optional>
#include <string_view>
#include "get_window.h"

namespace kvaf {
// A pair of zeros selects the fixed 128-KiB layout. Sized chunks use
// per-lane bytes, ceil division and cache-line rounding, as in AIV Gather.
struct GatherChunkConfig {
  std::uint32_t first_divisor{}, tail_chunks{};
  constexpr bool enabled() const { return first_divisor != 0; }
  constexpr bool valid() const {
    return (!first_divisor && !tail_chunks) || (first_divisor >= 2 && tail_chunks >= 1 && tail_chunks <= 32);
  }
  static constexpr std::uint64_t AlignedPart(std::uint64_t bytes, std::uint32_t parts) {
    return ((bytes + parts - 1) / parts + 63) & ~std::uint64_t{63};
  }
  constexpr std::uint64_t First(std::uint64_t bytes) const {
    const auto first = AlignedPart(bytes, first_divisor);
    return first < bytes ? first : bytes;
  }
  constexpr std::uint32_t Tail(std::uint64_t bytes) const {
    const auto left = bytes - First(bytes);
    return left ? static_cast<std::uint32_t>(AlignedPart(left, tail_chunks)) : 64U;
  }
};
// Legacy default workspace capacity, not a total BatchGet limit.
inline constexpr std::uint32_t kGatherMaxRequests = kDefaultGetWindow;
inline constexpr std::uint32_t kGatherMaxWorkers = 8U;
inline constexpr std::uint32_t kGatherChunkBytes = 131072U;
inline constexpr std::uint32_t kGatherMaxReadWaveWrs = 4096U;
// Internal zero selects the per-launch policy; public ExecutionResources uses None.
inline constexpr std::uint32_t kGatherDefaultReadWaveWrs = 0U;
inline constexpr std::uint32_t kGatherDefaultAddressGroupsPerPost = 3U;
inline bool ResolveGatherAddressGroupsPerPost(std::optional<std::uint32_t> configured,
                                             const char* environment, std::uint32_t* groups) {
  if (configured) {
    *groups = *configured;
  } else if (environment) {
    *groups = 0;
    for (const char digit : std::string_view(environment)) {
      if (digit < '0' || digit > '9') return false;
      const auto next = std::uint64_t{*groups} * 10U + static_cast<std::uint32_t>(digit - '0');
      // Larger settings already publish the whole remaining window.
      *groups = next > UINT32_MAX ? UINT32_MAX : static_cast<std::uint32_t>(next);
    }
  } else {
    *groups = kGatherDefaultAddressGroupsPerPost;
  }
  return *groups != 0;
}
inline constexpr bool ValidGatherReadWaveWrs(std::uint32_t wrs) {
  return wrs != 0U && wrs <= kGatherMaxReadWaveWrs;
}
inline constexpr bool ValidGatherReadWaveConfig(std::uint32_t wrs) {
  return wrs == kGatherDefaultReadWaveWrs || ValidGatherReadWaveWrs(wrs);
}
inline constexpr std::uint32_t ResolveGatherReadWaveWrs(std::uint32_t configured,
                                                      std::uint32_t miss_count) {
  if (configured != kGatherDefaultReadWaveWrs) return configured;
  return miss_count <= 8192U ? 512U : 64U;
}
inline constexpr std::uint32_t GatherLaneCapacity(
    std::uint32_t workers, std::uint32_t capacity = kDefaultGetWindow) {
  return capacity / workers + (capacity % workers != 0U);
}
inline constexpr std::uint64_t GatherLaneStagingBytes(std::uint32_t token_bytes,
                                                    std::uint32_t workers,
                                                    std::uint32_t capacity = kDefaultGetWindow) {
  return (static_cast<std::uint64_t>(token_bytes) * GatherLaneCapacity(workers, capacity) + 63U) &
         ~std::uint64_t{63U};
}
inline constexpr std::uint64_t GatherLaneDescriptorBytes(
    std::uint32_t workers, std::uint32_t capacity = kDefaultGetWindow) {
  return (64U + GatherLaneCapacity(workers, capacity) * 16ULL + 63U) & ~std::uint64_t{63U};
}
inline constexpr std::uint64_t GatherStagingBytes(std::uint32_t token_bytes,
                                                std::uint32_t workers = 1U,
                                                std::uint32_t capacity = kDefaultGetWindow) {
  return workers * GatherLaneStagingBytes(token_bytes, workers, capacity);
}
inline constexpr std::uint64_t GatherRemoteWorkspaceBytes(std::uint32_t token_bytes,
                                                        std::uint32_t workers = 1U,
                                                        std::uint32_t capacity = kDefaultGetWindow) {
  return GatherStagingBytes(token_bytes, workers, capacity) +
         workers * (GatherLaneDescriptorBytes(workers, capacity) + 64U) + 128U;
}
inline bool ValidGatherWindow(std::uint32_t token_bytes, std::uint32_t workers,
                              std::uint32_t capacity) {
  return token_bytes && workers && workers <= kGatherMaxWorkers && capacity &&
      capacity <= INT32_MAX && capacity % kGetWindowGranularity == 0U &&
      std::uint64_t{token_bytes} * GatherLaneCapacity(workers, capacity) <= UINT32_MAX;
}
inline std::uint32_t GatherWindowCapacity(std::uint32_t token_bytes, std::uint32_t workers,
                                        std::uint64_t budget, std::uint64_t alignment = 1U) {
  if (!budget)
    return ValidGatherWindow(token_bytes, workers, kDefaultGetWindow) ? kDefaultGetWindow : 0U;
  return LargestGetWindow([&](std::uint32_t capacity) {
    std::uint64_t bytes{};
    return ValidGatherWindow(token_bytes, workers, capacity) &&
        GetWindowBytes(GatherRemoteWorkspaceBytes(token_bytes, workers, capacity), alignment, &bytes) &&
        bytes <= budget;
  });
}
}  // namespace kvaf
