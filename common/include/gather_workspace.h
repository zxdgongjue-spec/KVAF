#pragma once
#include <cstdint>

namespace kvaf {
inline constexpr std::uint32_t kGatherMaxRequests = 65536U;
inline constexpr std::uint32_t kGatherMaxWorkers = 8U;
inline constexpr std::uint32_t kGatherChunkBytes = 131072U;
inline constexpr std::uint32_t GatherLaneCapacity(std::uint32_t workers) {
  return (kGatherMaxRequests + workers - 1U) / workers;
}
inline constexpr std::uint64_t GatherLaneStagingBytes(std::uint32_t token_bytes,
                                                     std::uint32_t workers,
                                                     std::uint64_t min_bytes = 0U) {
  const auto fixed_bytes = static_cast<std::uint64_t>(token_bytes) * GatherLaneCapacity(workers);
  // The descriptor can describe at most UINT32_MAX bytes. The aligned stride
  // itself is uint64 and may include one more byte of padding.
  const auto range_bytes = min_bytes < UINT32_MAX ? min_bytes : UINT32_MAX;
  return ((fixed_bytes > range_bytes ? fixed_bytes : range_bytes) + 63U) & ~std::uint64_t{63U};
}
inline constexpr std::uint64_t GatherLaneDescriptorBytes(std::uint32_t workers) {
  return (64U + GatherLaneCapacity(workers) * 16ULL + 63U) & ~std::uint64_t{63U};
}
inline constexpr std::uint64_t GatherStagingBytes(std::uint32_t token_bytes,
                                                std::uint32_t workers = 1U) {
  return workers * GatherLaneStagingBytes(token_bytes, workers);
}
inline constexpr std::uint64_t GatherRemoteWorkspaceBytes(std::uint32_t token_bytes,
                                                         std::uint32_t workers = 1U,
                                                         std::uint64_t min_lane_bytes = 0U) {
  return workers * GatherLaneStagingBytes(token_bytes, workers, min_lane_bytes) +
         workers * (GatherLaneDescriptorBytes(workers) + 64U) + 128U;
}
}  // namespace kvaf
