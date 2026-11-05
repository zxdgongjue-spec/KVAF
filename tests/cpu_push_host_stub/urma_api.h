#pragma once

#include <cstdint>

// Only the exported segment descriptor is used by the Host Scatter transport.
struct urma_seg_t {
  struct { std::uint8_t eid[16]{}; std::uint64_t va{}; } ubva;
  std::uint64_t len{};
  std::uint8_t reserved[16]{};
};
static_assert(sizeof(urma_seg_t) == 48);
