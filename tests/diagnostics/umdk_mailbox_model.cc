#include "aicpu_urma/control_abi.h"
#include <atomic>
#include <cstring>
#include <chrono>
#include <cassert>
#include <deque>
#include <functional>
#include <vector>
namespace mailbox_model {
std::uint64_t first_wait_ns{};
std::deque<std::vector<std::function<void()>>> pending;
}
namespace kvaf::aicpu_urma {
// CPU-only protocol model: ordered writes become visible before the final
// release word. Production uses the real asynchronous UMDK queue implementation.
std::uint32_t UmdkTransferPost(const UmdkContext*, const UmdkTransfer* transfers,
                               std::uint32_t count, std::uint32_t) {
  assert(mailbox_model::pending.size() < 4);
  std::vector<std::function<void()>> checks;
  for (std::uint32_t i = 0; i < count; ++i) {
    const auto& t = transfers[i];
    const auto local = t.local, bytes = t.bytes;
    const std::vector<std::uint8_t> saved(reinterpret_cast<const std::uint8_t*>(local), reinterpret_cast<const std::uint8_t*>(local) + bytes);
    checks.push_back([saved, local, bytes] { assert(!std::memcmp(saved.data(), reinterpret_cast<void*>(local), bytes)); });
    if (t.bytes == 8)
      __atomic_store_n(reinterpret_cast<std::uint64_t*>(t.remote),
          *reinterpret_cast<const std::uint64_t*>(t.local), __ATOMIC_RELEASE);
    else std::memcpy(reinterpret_cast<void*>(t.remote), reinterpret_cast<void*>(t.local), t.bytes);
  }
  mailbox_model::pending.push_back(std::move(checks));
  return 0;
}
std::uint32_t UmdkTransferWait(const UmdkContext*, std::uint32_t) {
  assert(!mailbox_model::pending.empty());
  for (const auto& check : mailbox_model::pending.front()) check();
  mailbox_model::pending.pop_front();
  if (!mailbox_model::first_wait_ns)
    mailbox_model::first_wait_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
  return 0;
}
}
