#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#if defined(KVAF_AICPU_TEST_DEVICE_KERNEL)
#include <securec.h>
#endif

namespace kvaf::aicpu_test {
// Packed staging is already in output-slot order. Preserve per-row output
// gaps; use the same securec performance entry as the HBM-to-HBM benchmark.
inline bool CopyScatterRows(std::uint8_t* output, const std::uint8_t* source,
                             std::uint64_t rows, std::uint64_t bytes, std::uint64_t stride) {
  for (std::uint64_t row = 0; row < rows; ++row) {
#if defined(KVAF_AICPU_TEST_DEVICE_KERNEL)
    if (memcpy_sp(output + row * stride, bytes, source + row * bytes, bytes) != EOK) return false;
#else
    std::memcpy(output + row * stride, source + row * bytes, static_cast<std::size_t>(bytes));
#endif
  }
  return true;
}
}  // namespace kvaf::aicpu_test
