#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "gather_span.h"
#include "gather_prefix_layout.h"
#include "gather_workspace.h"

namespace kvaf::peer {

struct alignas(64) GatherControl {
  std::uint64_t notify, ack, command, suffix_sequence;
  std::uint64_t reserved[4];
};
static_assert(sizeof(GatherControl) == 64U);
static_assert(offsetof(GatherControl, suffix_sequence) == kGatherSuffixOffset);

constexpr std::uint64_t kGatherHead256Magic = 0x4b56414648443236ULL;
constexpr std::uint64_t kGatherSizedChunksMagic = 0x4b56414653435a31ULL;
constexpr std::uint32_t kGatherHeadRows = 256U;
constexpr std::uint64_t kGatherErrorBit = std::uint64_t{1U} << 63U;

inline std::uint64_t GatherStartedAck(std::uint64_t sequence) { return ~sequence; }

inline std::uint64_t GatherDescriptorBytes(std::size_t span_count, bool configurable = false) {
  return GatherPrefixHeaderBytes(configurable) + span_count * sizeof(GatherSpan);
}

inline bool GatherRange(std::uint64_t base, std::uint64_t bytes, std::uint64_t address,
                        std::uint64_t length) {
  return length != 0U && address >= base && address - base <= bytes &&
         length <= bytes - (address - base);
}

inline std::uint64_t GatherControlOffset(std::uint64_t bytes) {
  return (bytes - 64U) & ~std::uint64_t{63U};
}

}  // namespace kvaf::peer
