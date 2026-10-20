#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "gather_span.h"

namespace kvaf::peer {

struct alignas(64) GatherControl {
  std::uint64_t notify, ack, command;
  std::uint64_t reserved[5];
};
static_assert(sizeof(GatherControl) == 64U);

static_assert(sizeof(GatherSpan) == 16U, "gather span ABI changed");

constexpr std::uint64_t kGatherDescriptorMagic = 0x4b56414647445232ULL;
constexpr std::uint64_t kGatherErrorBit = std::uint64_t{1U} << 63U;

inline std::uint64_t GatherStartedAck(std::uint64_t sequence) { return ~sequence; }

struct GatherDescriptorHeader {
  std::uint64_t magic;
  std::uint64_t sequence;
  std::uint64_t staging_address;
  std::uint32_t span_count;
  std::uint32_t total_bytes;
  std::uint32_t pipeline_chunk_bytes{0U};
  std::uint32_t reserved{0U};
  std::uint64_t ready_bytes{0U};
};

static_assert(sizeof(GatherDescriptorHeader) == 48U, "gather descriptor ABI changed");
static_assert(offsetof(GatherDescriptorHeader, ready_bytes) == 40U,
              "gather progress offset changed");

inline std::uint64_t GatherDescriptorBytes(std::size_t span_count) {
  return sizeof(GatherDescriptorHeader) + span_count * sizeof(GatherSpan);
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
