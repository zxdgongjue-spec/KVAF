#pragma once

#include <cstdint>

namespace kvaf {

// Physical SGE limit per READ WQE, not a limit on the requested gather size.
constexpr std::uint32_t kGatherMaxSglSize = 6U;

enum class GatherSqCopyMode : std::uint32_t {
  kMemcpySp = 0U,
  kMemcpy = kMemcpySp,
};

inline bool ValidGatherSqCopyMode(GatherSqCopyMode mode) {
  return mode == GatherSqCopyMode::kMemcpySp;
}

struct GatherOutputSpan {
  std::uint64_t offset;
  std::uint32_t length;
  std::uint32_t reserved;
};

static_assert(sizeof(GatherOutputSpan) == 16U, "gather output span ABI changed");

}  // namespace kvaf
