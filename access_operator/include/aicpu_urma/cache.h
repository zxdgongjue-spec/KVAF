#pragma once

#include <atomic>
#include <cstdint>

namespace kvaf::aicpu_urma {

// Maintain AICPU cache visibility for DMA buffers.
inline void FlushUmdkCacheLines(const void* address, std::uint64_t bytes) {
#if defined(__linux__) && defined(__aarch64__)
  if (bytes == 0) return;
  const auto first = reinterpret_cast<std::uintptr_t>(address) & ~std::uintptr_t{63};
  const auto end = reinterpret_cast<std::uintptr_t>(address) + bytes;
  for (auto line = first; line < end; line += 64U)
    asm volatile("dc civac, %0" ::"r"(line) : "memory");
#else
  (void)address;
  (void)bytes;
#endif
}

inline void FlushUmdkCache(const void* address, std::uint64_t bytes) {
  FlushUmdkCacheLines(address, bytes);
  std::atomic_signal_fence(std::memory_order_seq_cst);
}

}  // namespace kvaf::aicpu_urma
