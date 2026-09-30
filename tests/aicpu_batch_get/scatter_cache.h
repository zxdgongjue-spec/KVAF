#pragma once

#include <cstdint>

namespace kvaf::aicpu_test {
// The output stride is a multiple of 64 (MakeLayout). Visit only cache lines
// intersecting a row's payload, including its partially written last line.
// A shared completion callback keeps the device path to one DSB per worker.
template <class CleanLine, class Complete>
inline void CleanScatterPayload(std::uintptr_t first_row, std::uint64_t row_count,
                                std::uint64_t row_bytes, std::uint64_t stride,
                                CleanLine clean_line, Complete complete) {
  const auto first_line = first_row & ~std::uintptr_t{63};
  const auto bytes_to_cover = first_row - first_line + row_bytes;
  auto row_line = first_line;
  for (std::uint64_t row = 0; row < row_count; ++row, row_line += stride) {
    const auto end = row_line + bytes_to_cover;
    for (auto line = row_line; line < end; line += 64) clean_line(line);
  }
  complete();
}
}  // namespace kvaf::aicpu_test
