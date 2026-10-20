#pragma once

#include <atomic>
#include <cstdint>

#if defined(KVAF_AICPU_URMA_HOST_TEST_CACHE)
extern "C" void KvafAicpuUrmaTestFlush(const void* address, std::uint64_t bytes);
#endif

namespace kvaf::aicpu_urma {

// Keep control/prefix DMA buffers under the same cache policy in both backends.
inline void FlushGatherCacheLines(const void* address, std::uint64_t bytes) {
#if defined(KVAF_AICPU_URMA_HOST_TEST_CACHE)
  KvafAicpuUrmaTestFlush(address, bytes);
#elif defined(__linux__) && defined(__aarch64__) && !defined(KVAF_AICPU_URMA_HOST_TEST)
  const auto first = reinterpret_cast<std::uintptr_t>(address) & ~std::uintptr_t{63};
  const auto end = reinterpret_cast<std::uintptr_t>(address) + bytes;
  for (auto line = first; line < end; line += 64U)
    asm volatile("dc civac, %0" ::"r"(line) : "memory");
#else
  (void)address;
  (void)bytes;
#endif
}

inline void FlushGatherCache(const void* address, std::uint64_t bytes) {
  FlushGatherCacheLines(address, bytes);
  std::atomic_signal_fence(std::memory_order_seq_cst);
}

}  // namespace kvaf::aicpu_urma
