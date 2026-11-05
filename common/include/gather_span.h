#pragma once
#include <cstdint>
namespace kvaf::peer {
struct GatherSpan {
  std::uint64_t address;
  std::uint32_t length;
  std::uint32_t reserved;
};

static_assert(sizeof(GatherSpan) == 16U);
}  // namespace kvaf::peer
