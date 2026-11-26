#pragma once
#include <cstddef>
#include <cstdint>

namespace kvaf::peer {
inline constexpr std::uint64_t kGatherDescriptorMagic = 0x4b56414647445232ULL;

struct GatherDescriptorHeader {
  std::uint64_t magic;
  std::uint64_t sequence;
  std::uint64_t staging_address;
  std::uint32_t span_count;
  std::uint32_t total_bytes;
  std::uint32_t pipeline_chunk_bytes{0U};
  // Zero selects variable-length source spans.
  std::uint32_t span_bytes{0U};
  std::uint64_t ready_bytes{0U};
};
static_assert(sizeof(GatherDescriptorHeader) == 48U, "gather descriptor ABI changed");
static_assert(offsetof(GatherDescriptorHeader, ready_bytes) == 40U, "gather progress offset changed");
}  // namespace kvaf::peer

namespace kvaf {
// Initial notify commits the first 1024 spans; control+24 commits the suffix.
// Older CPU services reject the flagged magic instead of consuming an incomplete list.
inline constexpr std::uint64_t kGatherHeadStartFlag = std::uint64_t{1} << 63;
inline constexpr std::uint32_t kGatherHeadRows = 1024;
inline constexpr std::uint32_t kGatherSuffixOffset = 24;
// GDR4 changes CPU copy ownership only. The 48-byte GDR2 envelope, mutable
// byte prefix and final ACK keep their meanings.
inline constexpr std::uint64_t kGatherCooperativeMagic = 0x4b56414647445234ULL;
// GDR5: first payload/32, followed by four equal-sized (64-B aligned) tails.
// Keep a distinct magic so older CPU services reject the nonuniform layout.
inline constexpr std::uint64_t kGatherBootstrapMagic = 0x4b56414647445235ULL;
inline constexpr std::uint32_t kGatherBootstrapDivisor = 32;
inline constexpr std::uint32_t kGatherBootstrapTailChunks = 4;
// GDR6 appends an explicit 16-byte layout config after the unchanged 48-byte
// prefix header. Sequence/ready offsets and source span records stay unchanged.
inline constexpr std::uint64_t kGatherConfiguredMagic = 0x4b56414647445236ULL;
struct GatherChunkConfig {
  std::uint32_t first_chunk_divisor, tail_chunks;
  std::uint64_t reserved;
};
static_assert(sizeof(GatherChunkConfig) == 16);
inline constexpr std::uint32_t kGatherMaxCooperativeChunks = 32;
inline constexpr std::uint32_t kGatherMaxProgressChunks = kGatherMaxCooperativeChunks + 1;
inline constexpr std::uint32_t kGatherScratchBytes = 64;
#if defined(__CCE_AICORE__)
#define KVAF_PREFIX_INLINE __aicore__ inline
#else
#define KVAF_PREFIX_INLINE inline
#endif
KVAF_PREFIX_INLINE bool GatherCooperativeCountValid(std::uint32_t chunks) {
  return chunks == 0 || chunks == 2 || chunks == 4 || chunks == 8 ||
         chunks == 16 || chunks == 32;
}
KVAF_PREFIX_INLINE bool GatherCooperativeLayoutValid(std::uint32_t chunks, std::uint32_t first_divisor) {
  return first_divisor ? first_divisor >= 2 && chunks >= 1 && chunks <= kGatherMaxCooperativeChunks :
      GatherCooperativeCountValid(chunks);
}
KVAF_PREFIX_INLINE std::uint32_t GatherPrefixHeaderBytes(bool configurable) {
  return sizeof(peer::GatherDescriptorHeader) + (configurable ? sizeof(GatherChunkConfig) : 0U);
}
KVAF_PREFIX_INLINE std::uint64_t GatherFirstChunkBytes(
    std::uint64_t total_bytes, std::uint32_t first_divisor) {
  if (!first_divisor) return 0;
  const auto first = ((total_bytes + first_divisor - 1) / first_divisor + 63) & ~std::uint64_t{63};
  return first < total_bytes ? first : total_bytes;
}
KVAF_PREFIX_INLINE std::uint32_t GatherCooperativeChunkBytes(
    std::uint64_t total_bytes, std::uint32_t chunks, std::uint32_t first_divisor = 0) {
  const auto remaining = total_bytes - GatherFirstChunkBytes(total_bytes, first_divisor);
  const auto bytes = chunks ? ((remaining + chunks - 1) / chunks + 63) & ~std::uint64_t{63} : 0;
  // A nonempty request fully covered by its short first chunk still uses the
  // nonzero chunk-mode envelope; no tail will actually be copied or published.
  return static_cast<std::uint32_t>(bytes ? bytes : first_divisor && total_bytes ? 64 : 0);
}
KVAF_PREFIX_INLINE std::uint32_t GatherProgressChunkCount(
    std::uint64_t total_bytes, std::uint32_t chunk_bytes, std::uint32_t first_divisor = 0) {
  if (!total_bytes || !chunk_bytes) return 0;
  const auto first = GatherFirstChunkBytes(total_bytes, first_divisor);
  return static_cast<std::uint32_t>((first ? 1 : 0) +
      (total_bytes - first + chunk_bytes - 1) / chunk_bytes);
}
KVAF_PREFIX_INLINE std::uint64_t GatherProgressChunkEnd(
    std::uint64_t total_bytes, std::uint32_t chunk_bytes, std::uint32_t first_divisor,
    std::uint32_t index) {
  const auto first = GatherFirstChunkBytes(total_bytes, first_divisor);
  const auto end = first ? first + std::uint64_t(index) * chunk_bytes :
      std::uint64_t(index + 1) * chunk_bytes;
  return end < total_bytes ? end : total_bytes;
}
KVAF_PREFIX_INLINE bool GatherProgressBytesValid(
    std::uint64_t total_bytes, std::uint32_t chunk_bytes, std::uint32_t first_divisor,
    std::uint64_t ready) {
  if (ready > total_bytes || !chunk_bytes) return false;
  if (!ready || ready == total_bytes) return true;
  const auto first = GatherFirstChunkBytes(total_bytes, first_divisor);
  return ready >= first && (ready - first) % chunk_bytes == 0;
}
#undef KVAF_PREFIX_INLINE
}  // namespace kvaf
