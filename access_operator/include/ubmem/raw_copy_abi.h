#pragma once
#include <cstdint>

namespace kvaf::ubmem {
struct RawCopyRequest {
  std::uint64_t source, destination;
};
struct RawCopyArgs {
  std::uint64_t requests, reports;
  std::uint32_t rows, row_bytes, blocks;
};
struct alignas(64) RawCopyReport {
  std::uint64_t cycles, completed;
  std::uint64_t reserved[6];
};
static_assert(sizeof(RawCopyRequest) == 16);
static_assert(sizeof(RawCopyArgs) == 32);
static_assert(sizeof(RawCopyReport) == 64);
}  // namespace kvaf::ubmem
