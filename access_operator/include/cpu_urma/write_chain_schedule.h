#pragma once

#include <algorithm>
#include <cstddef>

namespace kvaf::cpu_urma_detail {
// No completion waits between these posts. Completing every first prefix
// before any suffix prevents a long queue-0 chain delaying first-ready.
template <class Prefix, class Count, class Post>
bool DispatchWriteChain(std::size_t queues, Prefix prefix, Count count, Post post) {
  for (unsigned phase = 0; phase < 2; ++phase)
    for (std::size_t q = 0; q < queues; ++q) {
      const auto begin = phase ? prefix(q) : 0;
      const auto end = phase ? count(q) : prefix(q);
      if (begin < end && !post(q, begin, end)) return false;
    }
  return true;
}
// Build immutable post boundaries once. No next-pointer mutation while live.
template <class Emit>
void ForEachChainPost(std::size_t begin, std::size_t end, std::size_t limit, Emit emit) {
  if (!limit) limit = end - begin;
  while (begin < end) {
    const auto next = begin + std::min(limit, end - begin);
    emit(begin, next); begin = next;
  }
}
}  // namespace kvaf::cpu_urma_detail
