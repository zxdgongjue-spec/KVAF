#pragma once
#include <cstdint>
namespace kvaf {
constexpr std::uint32_t kGatherMaxSglSize = 6U;
struct GatherOutputSpan {
  std::uint64_t offset;
  std::uint32_t length;
  std::uint32_t reserved{0U};
};
static_assert(sizeof(GatherOutputSpan) == 16U);
}  // namespace kvaf
