#pragma once
#include <cstdint>

namespace kvaf {
// Gather-only GDR3 wire format. The 48-byte envelope and spans retain their
// offsets; its last word carries the immutable READ group size instead of a
// prefix. An aligned trailer contains generation-tagged, independent progress.
inline constexpr std::uint64_t kGatherShardMagic = 0x4b56414647445233ULL;
inline constexpr std::uint32_t kGatherMaxShards = 64;
inline constexpr std::uint32_t kGatherProgressBytes = kGatherMaxShards * 8;
inline constexpr std::uint32_t kGatherScratchBytes = 64 + kGatherProgressBytes;
#if defined(__CCE_AICORE__)
#define KVAF_GATHER_INLINE __aicore__ inline
#else
#define KVAF_GATHER_INLINE inline
#endif
KVAF_GATHER_INLINE std::uint64_t GatherProgressOffset(std::uint32_t spans) {
  return (48ULL + std::uint64_t(spans) * 16 + 63) & ~std::uint64_t{63};
}
KVAF_GATHER_INLINE std::uint32_t GatherShardCount(std::uint32_t wrs) {
  return wrs < kGatherMaxShards ? wrs : kGatherMaxShards;
}
KVAF_GATHER_INLINE std::uint32_t GatherShardBoundary(
    std::uint32_t wrs, std::uint32_t shards, std::uint32_t shard) {
  return static_cast<std::uint32_t>(std::uint64_t(wrs) * shard / shards);
}
KVAF_GATHER_INLINE std::uint64_t GatherShardProgress(
    std::uint64_t sequence, std::uint32_t completed_wrs) {
  return (sequence << 32) | completed_wrs;
}
#undef KVAF_GATHER_INLINE
}  // namespace kvaf
