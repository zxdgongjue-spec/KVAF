#pragma once

// Host-memory implementation used only by the probe and transport unit tests.
#include <cstdlib>
#include <cstring>
#include <cstdint>

using aclrtStream = void*;
constexpr int ACL_SUCCESS = 0;
constexpr int ACL_MEM_MALLOC_NORMAL_ONLY = 0;
constexpr int ACL_MEMCPY_HOST_TO_DEVICE = 0;
constexpr int ACL_MEMCPY_DEVICE_TO_HOST = 1;
namespace fake_acl { inline size_t allocations = 0U; }
inline int aclrtGetDevice(int32_t* device) { *device = 0; return 0; }
inline int aclrtMalloc(void** data, size_t bytes, int) {
  *data = std::malloc(bytes);
  if (*data == nullptr) return 1;
  ++fake_acl::allocations;
  return 0;
}
inline int aclrtFree(void* data) {
  if (data != nullptr) { --fake_acl::allocations; std::free(data); }
  return 0;
}
inline int aclrtCreateStream(aclrtStream* stream) { *stream = new int; return 0; }
inline int aclrtDestroyStream(aclrtStream stream) { delete static_cast<int*>(stream); return 0; }
inline int aclrtMemcpy(void* dest, size_t capacity, const void* source, size_t bytes, int) {
  if (bytes > capacity) return 1;
  std::memcpy(dest, source, bytes);
  return 0;
}
inline int aclrtMemset(void* dest, size_t capacity, int value, size_t bytes) {
  if (bytes > capacity) return 1;
  std::memset(dest, value, bytes);
  return 0;
}
