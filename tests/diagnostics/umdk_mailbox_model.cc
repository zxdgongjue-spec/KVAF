#include "aicpu_urma/control_abi.h"
#include <atomic>
#include <cstring>
namespace kvaf::aicpu_urma {
// CPU-only protocol model: ordered writes become visible before the final
// release word. Production uses the real asynchronous UMDK queue implementation.
std::uint32_t UmdkTransferPost(const UmdkContext*, const UmdkTransfer* transfers,
                               std::uint32_t count, std::uint32_t) {
  for (std::uint32_t i = 0; i < count; ++i) {
    const auto& t = transfers[i];
    if (t.bytes == 8)
      __atomic_store_n(reinterpret_cast<std::uint64_t*>(t.remote),
          *reinterpret_cast<const std::uint64_t*>(t.local), __ATOMIC_RELEASE);
    else std::memcpy(reinterpret_cast<void*>(t.remote), reinterpret_cast<void*>(t.local), t.bytes);
  }
  return 0;
}
std::uint32_t UmdkTransferWait(const UmdkContext*, std::uint32_t) { return 0; }
}
