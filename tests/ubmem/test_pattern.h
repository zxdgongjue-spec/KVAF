#pragma once
#include <cstdint>

namespace kvaf::ubmem {
// Mix high address bits so aligned KV slots do not all contain identical data.
inline std::uint8_t TestPattern(std::uint64_t offset, std::uint32_t seed) {
  return static_cast<std::uint8_t>((offset * 31) ^ (offset >> 8) ^ (offset >> 16) ^ seed);
}
}
