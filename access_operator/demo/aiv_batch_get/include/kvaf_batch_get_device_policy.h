#ifndef KVAF_BATCH_GET_DEVICE_POLICY_H_
#define KVAF_BATCH_GET_DEVICE_POLICY_H_

#include <stdint.h>

#if defined(__CCE_AICORE__)
#define KVAF_BATCH_GET_POLICY_INLINE __aicore__ inline
#else
#define KVAF_BATCH_GET_POLICY_INLINE inline
#endif

namespace kvaf_device {

// Subtraction avoids overflow for malformed UINT32_MAX caller arguments.
KVAF_BATCH_GET_POLICY_INLINE bool BatchGetRequestRangeValid(
    uint32_t capacity, uint32_t begin, uint32_t count) {
  return begin <= capacity && count <= capacity - begin;
}

// Successful completion of the previous window leaves no outstanding SQ
// entries. Keep one entry unused and use the smallest Jetty depth.
KVAF_BATCH_GET_POLICY_INLINE uint32_t BatchGetWindowRequestCount(
    uint32_t remaining, uint32_t minimum_sq_depth) {
  if (minimum_sq_depth < 2U) {
    return 0U;
  }
  const uint32_t capacity = minimum_sq_depth - 1U;
  return remaining < capacity ? remaining : capacity;
}

}  // namespace kvaf_device

#undef KVAF_BATCH_GET_POLICY_INLINE

#endif  // KVAF_BATCH_GET_DEVICE_POLICY_H_
