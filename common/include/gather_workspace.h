#pragma once
#include <cstdint>
#include "get_window.h"

namespace kvaf {
// Legacy default workspace capacity, not a total BatchGet limit.
inline constexpr std::uint32_t kGatherMaxRequests = kDefaultGetWindow;
inline constexpr std::uint32_t kGatherMaxWorkers = 8U;
inline constexpr std::uint32_t kGatherChunkBytes = 131072U;
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
