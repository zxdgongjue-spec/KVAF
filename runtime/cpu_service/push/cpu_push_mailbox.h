#pragma once

#include "cpu_push_get_abi.h"

namespace kvaf::cpu_push {

inline std::uint64_t LoadMailboxSequence(const void* notify_address, bool& profile) {
  // The device strongly orders its notify WRITE after the request data.
  const auto notify = __atomic_load_n(static_cast<const std::uint64_t*>(notify_address), __ATOMIC_ACQUIRE);
  profile = (notify & kProfileNotify) != 0;
  return notify & ~kProfileNotify;
}

}  // namespace kvaf::cpu_push
