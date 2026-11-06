#pragma once
#include <cstdint>
#include "gather_shard_layout.h"

namespace kvaf {
inline constexpr std::uint32_t kGatherMaxRequests = 65536U;
inline constexpr std::uint32_t kGatherMaxWorkers = 8U;
inline constexpr std::uint32_t kGatherChunkBytes = 131072U;
inline constexpr std::uint32_t GatherLaneCapacity(std::uint32_t workers) {
  return (kGatherMaxRequests + workers - 1U) / workers;
}
inline constexpr std::uint64_t GatherLaneStagingBytes(std::uint32_t token_bytes,
                                                    std::uint32_t workers) {
  return (static_cast<std::uint64_t>(token_bytes) * GatherLaneCapacity(workers) + 63U) &
         ~std::uint64_t{63U};
}
inline constexpr std::uint64_t GatherLaneDescriptorBytes(std::uint32_t workers) {
  return ((64U + GatherLaneCapacity(workers) * 16ULL + 63U) & ~std::uint64_t{63U}) +
         kGatherProgressBytes;
}
inline constexpr std::uint64_t GatherStagingBytes(std::uint32_t token_bytes,
                                                std::uint32_t workers = 1U) {
  return workers * GatherLaneStagingBytes(token_bytes, workers);
}
inline constexpr std::uint64_t GatherRemoteWorkspaceBytes(std::uint32_t token_bytes,
                                                        std::uint32_t workers = 1U) {
  return GatherStagingBytes(token_bytes, workers) +
         workers * (GatherLaneDescriptorBytes(workers) + 64U) + 128U;
}
}  // namespace kvaf
