#pragma once

#include <algorithm>
#include <cstddef>

namespace kvaf::cpu_urma_detail {

// Called by the sole owner of a Jetty. Preparation and submission alternate
// within that worker; a failed submission prevents later chunks being built.
// The caller must terminate the submitted WR list at end - 1 before posting.
template <class Prepare, class Post>
void DispatchJettyChunks(std::size_t wr_count, std::size_t chunk_size,
                         Prepare prepare, Post post) {
  if (wr_count == 0U) return;
  if (chunk_size == 0U) chunk_size = wr_count;
  for (std::size_t begin = 0U; begin < wr_count;) {
    const std::size_t end = begin + std::min(chunk_size, wr_count - begin);
    prepare(begin, end);
    if (!post(begin, end)) return;
    begin = end;
  }
}

// Post every queue's first prefix before any suffix to avoid delaying first-ready.
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

// Fix doorbell boundaries before submission; live WR links must remain unchanged.
template <class Emit>
void ForEachChainPost(std::size_t begin, std::size_t end, std::size_t limit, Emit emit) {
  if (!limit) limit = end - begin;
  while (begin < end) {
    const auto next = begin + std::min(limit, end - begin);
    emit(begin, next); begin = next;
  }
}
}  // namespace kvaf::cpu_urma_detail
