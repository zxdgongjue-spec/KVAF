#pragma once
#include <cstdint>

namespace kvaf {
inline constexpr std::uint32_t kGatherMaxRequests = 65536U;
inline constexpr std::uint32_t kGatherChunkBytes = 131072U;
inline constexpr std::uint64_t GatherStagingBytes(std::uint32_t token_bytes) {
  return (static_cast<std::uint64_t>(token_bytes) * kGatherMaxRequests + 63U) & ~std::uint64_t{63U};
}
inline constexpr std::uint64_t GatherRemoteWorkspaceBytes(std::uint32_t token_bytes) {
  // Staging, descriptor spans/header, control, and both alignment paddings.
  return GatherStagingBytes(token_bytes) + kGatherMaxRequests * 16ULL + 256U;
}
}  // namespace kvaf
