#pragma once

#include "cpu_urma/cpu_urma_transport.h"
#include <algorithm>
#include <limits>
#include <vector>

namespace kvaf::cpu_urma_detail {
// No payload copy and no sorting: merge only consecutive source AND target
// spans in the same registered segment. Apply within each existing Jetty
// shard so queue ownership, output order and completion fences stay intact.
template <class Segment>
Status CoalesceWriteSpans(const CpuUrmaTransport::HostSge* sources,
    const std::uint64_t* offsets, std::size_t count, std::uint64_t remote_bytes,
    std::uint64_t max_bytes, Segment segment,
    std::vector<CpuUrmaTransport::HostSge>* merged,
    std::vector<std::uint64_t>* merged_offsets) {
  if (!sources || !offsets || !count || !max_bytes || !merged || !merged_offsets)
    return Status::InvalidArgument("invalid contiguous WRITE spans");
  merged->clear(); merged_offsets->clear();
  max_bytes = std::min<std::uint64_t>(max_bytes, UINT32_MAX);
  for (std::size_t i = 0; i < count; ++i) {
    const auto& s = sources[i];
    const auto address = reinterpret_cast<std::uintptr_t>(s.data);
    if (!address || !s.bytes || s.bytes > max_bytes ||
        s.bytes > std::numeric_limits<std::uintptr_t>::max() - address ||
        offsets[i] > remote_bytes || s.bytes > remote_bytes - offsets[i])
      return Status::InvalidArgument("invalid contiguous WRITE span");
    const auto registration = segment(s.data, s.bytes);
    if (!registration) return Status::InvalidArgument("unregistered contiguous WRITE span");
    if (!merged->empty()) {
      auto& tail = merged->back();
      const auto tail_address = reinterpret_cast<std::uintptr_t>(tail.data);
      if (tail_address + tail.bytes == address &&
          merged_offsets->back() + tail.bytes == offsets[i] &&
          s.bytes <= max_bytes - tail.bytes &&
          segment(tail.data, tail.bytes) == registration &&
          segment(tail.data, tail.bytes + s.bytes) == registration) {
        tail.bytes += s.bytes; continue;
      }
    }
    merged->push_back(s); merged_offsets->push_back(offsets[i]);
  }
  return Status::OK();
}
}  // namespace kvaf::cpu_urma_detail
