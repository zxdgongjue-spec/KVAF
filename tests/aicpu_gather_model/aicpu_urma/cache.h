#pragma once
#include <atomic>
#include <cstdint>
namespace kvaf::aicpu_urma {
inline void FlushUmdkCache(const void*, std::uint64_t) {
  std::atomic_thread_fence(std::memory_order_seq_cst);
}
}
