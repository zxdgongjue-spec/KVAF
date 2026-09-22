#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "cpu_urma/cpu_urma_transport.h"

namespace kvaf::cpu_urma_detail {

#if defined(__GNUC__) && !defined(__clang__)
#define KVAF_CPU_URMA_BUILD_HOT \
  __attribute__((hot, noinline, flatten, optimize("O3")))
#else
#define KVAF_CPU_URMA_BUILD_HOT
#endif

struct WriteGatherGroup {
  std::size_t first{0U};
  std::size_t count{0U};
  std::uint64_t bytes{0U};
};

// Build one Jetty owner's WRITE descriptors in row order. Every input row is
// emitted as one local SGE; only the remote layout and device limits determine
// WRITE WR boundaries.
template <class EmitRow, class EmitGroup>
KVAF_CPU_URMA_BUILD_HOT Status BuildWriteGatherShard(
    const CpuUrmaTransport::HostSge* buffers,
    const std::uint64_t* remote_offsets,
    std::size_t row_count,
    std::size_t first,
    std::size_t count,
    std::size_t local_sgl_size,
    std::uint64_t remote_bytes,
    std::uint64_t max_group_bytes,
    EmitRow emit_row,
    EmitGroup emit_group,
    std::size_t* wr_count) {
  if (wr_count == nullptr) {
    return Status::InvalidArgument("missing WRITE gather WR count");
  }
  *wr_count = 0U;
  if (buffers == nullptr || remote_offsets == nullptr || count == 0U ||
      local_sgl_size == 0U || max_group_bytes == 0U || first > row_count ||
      count > row_count - first) {
    return Status::InvalidArgument("invalid WRITE gather shard");
  }

  const std::uint64_t max_bytes = std::min<std::uint64_t>(
      max_group_bytes, std::numeric_limits<std::uint32_t>::max());
  WriteGatherGroup group{};
  std::size_t built_wrs = 0U;

  for (std::size_t local_index = 0U; local_index < count; ++local_index) {
    const std::size_t row = first + local_index;
    const auto& buffer = buffers[row];
    const std::uintptr_t address =
        reinterpret_cast<std::uintptr_t>(buffer.data);
    const std::uint64_t bytes = buffer.bytes;
    const std::uint64_t offset = remote_offsets[row];
    if (address == 0U || bytes == 0U || bytes > max_bytes ||
        bytes > std::numeric_limits<std::uintptr_t>::max() - address ||
        offset > remote_bytes || bytes > remote_bytes - offset) {
      return Status::InvalidArgument("invalid WRITE gather row");
    }

    const bool joins_group = group.count != 0U &&
        group.count < local_sgl_size && bytes <= max_bytes - group.bytes &&
        offset == remote_offsets[group.first] + group.bytes;
    if (!joins_group) {
      if (group.count != 0U) {
        emit_group(built_wrs++, group);
      }
      group = WriteGatherGroup{row, 1U, bytes};
    } else {
      ++group.count;
      group.bytes += bytes;
    }
    emit_row(local_index, row, buffer);
  }

  emit_group(built_wrs++, group);
  *wr_count = built_wrs;
  return Status::OK();
}

#undef KVAF_CPU_URMA_BUILD_HOT

}  // namespace kvaf::cpu_urma_detail
