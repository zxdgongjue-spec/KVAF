#ifndef KVAF_BATCH_GET_COMPACT_PREFIX_H_
#define KVAF_BATCH_GET_COMPACT_PREFIX_H_

#include <stdint.h>

// Shared executable range arithmetic, usable by the AIV kernel and host
// behavioral tests. The caller loads the compact count on device; this
// function never dereferences an NPU address. Define KVAF_PREFIX_INLINE as
// __aicore__ inline in the kernel translation unit.
#ifndef KVAF_PREFIX_INLINE
#define KVAF_PREFIX_INLINE inline
#define KVAF_PREFIX_INLINE_DEFINED_HERE
#endif

KVAF_PREFIX_INLINE bool KvafBatchGetCompactPrefixRange(
    int32_t active_count, uint32_t request_capacity,
    uint32_t core, uint32_t block_count,
    uint32_t* begin, uint32_t* end) {
  if (begin == nullptr || end == nullptr) return false;
  *begin = 0U;
  *end = 0U;
  if (active_count < 0 ||
      static_cast<uint32_t>(active_count) > request_capacity ||
      block_count == 0U || core >= block_count) {
    return false;
  }
  const uint64_t count = static_cast<uint32_t>(active_count);
  *begin = static_cast<uint32_t>((count * core) / block_count);
  *end = static_cast<uint32_t>((count * (static_cast<uint64_t>(core) + 1U)) /
                               block_count);
  return true;
}

#ifdef KVAF_PREFIX_INLINE_DEFINED_HERE
#undef KVAF_PREFIX_INLINE_DEFINED_HERE
#undef KVAF_PREFIX_INLINE
#endif

#endif  // KVAF_BATCH_GET_COMPACT_PREFIX_H_
