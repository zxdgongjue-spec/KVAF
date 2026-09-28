#pragma once

#include <cstdint>

namespace kvaf {

constexpr std::uint32_t kGatherMaxSglSize = 8U;

struct GatherOutputSpan {
  std::uint64_t offset;
  std::uint32_t length;
  std::uint32_t reserved;
};

static_assert(sizeof(GatherOutputSpan) == 16U, "gather output span ABI changed");

}  // namespace kvaf
