#pragma once
#include <cstdint>

namespace kvaf {
inline constexpr std::uint32_t kGetWindowGranularity = 4096U;
inline constexpr std::uint32_t kDefaultGetWindow = 65536U;
// miss_count remains int32. Search quantized capacities, never floating-point
// estimates: metadata, per-worker padding and page rounding all count.
template <class Fits>
std::uint32_t LargestGetWindow(Fits fits) {
  std::uint32_t low = 0U, high = INT32_MAX / kGetWindowGranularity;
  while (low < high) {
    const auto middle = low + (high - low + 1U) / 2U;
    if (fits(middle * kGetWindowGranularity)) low = middle;
    else high = middle - 1U;
  }
  return low * kGetWindowGranularity;
}
inline bool GetWindowBytes(std::uint64_t bytes, std::uint64_t alignment,
                          std::uint64_t* rounded) {
  if (!alignment || !rounded) return false;
  const auto padding = (alignment - bytes % alignment) % alignment;
  if (bytes > UINT64_MAX - padding) return false;
  *rounded = bytes + padding;
  return true;
}
}  // namespace kvaf
