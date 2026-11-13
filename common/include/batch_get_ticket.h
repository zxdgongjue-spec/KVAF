#pragma once

#include <cstdint>

namespace kvaf {
struct BatchGetTicket {
  std::uint32_t worker;
  std::uint64_t sequence;
};

// A transport has a fixed worker count and serializes its launches. Every
// instance claims once, including empty slices. Never reset between replays.
inline BatchGetTicket ClaimBatchGetTicket(std::uint64_t* claimed, std::uint32_t workers) {
  const auto ticket = __atomic_fetch_add(claimed, std::uint64_t{1}, __ATOMIC_ACQ_REL);
  return {static_cast<std::uint32_t>(ticket % workers), ticket / workers + 1U};
}
}  // namespace kvaf
