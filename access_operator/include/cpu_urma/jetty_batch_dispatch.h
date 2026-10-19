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

}  // namespace kvaf::cpu_urma_detail
