#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "cpu_urma/cpu_urma_transport.h"

namespace kvaf::cpu_urma_detail {

struct SglGroup {
  std::size_t first{0};
  std::size_t count{0};
  std::uint64_t bytes{0};
};

// Preserve request order. One remote SGE can describe only a contiguous run;
// local SGEs may refer to unrelated registered allocations. Bound the aggregate
// length as well as each row because the remote SGE length is uint32_t.
inline Status PlanSglGroups(const CpuUrmaTransport::HostSge* buffers,
                            const std::uint64_t* remote_offsets,
                            std::size_t count, std::size_t sgl_size,
                            std::uint64_t remote_bytes,
                            std::vector<SglGroup>* groups) {
  if (groups == nullptr) return Status::InvalidArgument("missing SGL plan output");
  groups->clear();
  if (buffers == nullptr || remote_offsets == nullptr || count == 0 || sgl_size == 0)
    return Status::InvalidArgument("invalid SGL plan input");
  constexpr auto max_bytes = std::numeric_limits<std::uint32_t>::max();
  for (std::size_t i = 0; i < count; ++i) {
    const auto bytes = buffers[i].bytes;
    const auto offset = remote_offsets[i];
    if (buffers[i].data == nullptr || bytes == 0 || bytes > max_bytes ||
        offset > remote_bytes || bytes > remote_bytes - offset) {
      groups->clear();
      return Status::InvalidArgument("SGL row is outside its buffer or remote region");
    }
    if (!groups->empty()) {
      auto& previous = groups->back();
      if (previous.count < sgl_size && bytes <= max_bytes - previous.bytes &&
          offset == remote_offsets[previous.first] + previous.bytes) {
        ++previous.count;
        previous.bytes += bytes;
        continue;
      }
    }
    groups->push_back(SglGroup{i, 1, bytes});
  }
  return Status::OK();
}

}  // namespace kvaf::cpu_urma_detail
