#pragma once

#include <cstddef>

#if defined(KVAF_AICPU_URMA_HOST_TEST)
#include <cstring>
#else
#include <securec.h>
#ifndef memcpy_sp
#error "The AICPU SQ copy requires securec performance addons (memcpy_sp)"
#endif
#endif

namespace kvaf::aicpu_urma {

inline bool CopySqBytes(void* destination, const void* source, std::size_t bytes) {
  if (bytes == 0U) return true;
#if defined(KVAF_AICPU_URMA_HOST_TEST)
  if (destination == nullptr || source == nullptr) return false;
  std::memcpy(destination, source, bytes);
  return true;
#else
  return memcpy_sp(destination, bytes, source, bytes) == EOK;
#endif
}

}
