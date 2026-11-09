#pragma once
#include <cstdint>

namespace kvaf {
// GDR4 changes CPU copy ownership only. The 48-byte GDR2 envelope, mutable
// byte prefix and final ACK keep their meanings.
inline constexpr std::uint64_t kGatherCooperativeMagic = 0x4b56414647445234ULL;
inline constexpr std::uint32_t kGatherMaxCooperativeChunks = 8;
inline constexpr std::uint32_t kGatherScratchBytes = 64;
#if defined(__CCE_AICORE__)
#define KVAF_PREFIX_INLINE __aicore__ inline
#else
#define KVAF_PREFIX_INLINE inline
#endif
KVAF_PREFIX_INLINE bool GatherCooperativeCountValid(std::uint32_t chunks) {
  return chunks == 0 || chunks == 2 || chunks == 4 || chunks == 8;
}
KVAF_PREFIX_INLINE std::uint32_t GatherCooperativeChunkBytes(
    std::uint64_t total_bytes, std::uint32_t chunks) {
  return chunks ? static_cast<std::uint32_t>(((total_bytes + chunks - 1) / chunks + 63) & ~std::uint64_t{63}) : 0;
}
#undef KVAF_PREFIX_INLINE
}  // namespace kvaf
