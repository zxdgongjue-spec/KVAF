#pragma once

// Host-only dependency stub: verifies the device wrapper calls memcpy_sp with
// exact byte lengths and propagates library failures. Not a performance test.
#include <cstddef>
#include <cstring>
#include <vector>

constexpr int EOK = 0;
namespace kvaf_securec_stub {
struct Call { void* destination; std::size_t capacity; const void* source; std::size_t bytes; };
inline std::vector<Call> calls;
inline std::size_t fail_call{};
inline int Copy(void* dst, std::size_t capacity, const void* src, std::size_t bytes) {
  calls.push_back({dst, capacity, src, bytes});
  if (calls.size() == fail_call || !dst || !src || capacity < bytes) return 1;
  std::memcpy(dst, src, bytes);
  return EOK;
}
}
#define memcpy_sp kvaf_securec_stub::Copy
