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

// A one-to-one batch needs validation, but no scatter-group array.
inline Status ValidateBatchRows(const CpuUrmaTransport::HostSge* buffers,
                                const std::uint64_t* offsets,
                                std::size_t count,
                                std::uint64_t remote_bytes) {
  if (buffers == nullptr || offsets == nullptr || count == 0U) {
    return Status::InvalidArgument("invalid batch rows");
  }
  for (std::size_t i = 0U; i < count; ++i) {
    const auto address = reinterpret_cast<std::uintptr_t>(buffers[i].data);
    const auto bytes = buffers[i].bytes;
    if (address == 0U || bytes == 0U ||
        bytes > std::numeric_limits<std::uint32_t>::max() ||
        bytes > std::numeric_limits<std::uintptr_t>::max() - address ||
        offsets[i] > remote_bytes || bytes > remote_bytes - offsets[i]) {
      return Status::InvalidArgument(
          "batch row is outside its buffer or remote region");
    }
  }
  return Status::OK();
}

// Coalesce up to remote_sgl_size rows only when both local and remote ranges
// are contiguous. RC WRITE consumes one remote SGE, so remote gaps must start
// a new WR even when the local source remains contiguous.
inline Status PlanWriteCoalescedGroups(const CpuUrmaTransport::HostSge* buffers,
                                       const std::uint64_t* remote_offsets,
                                       std::size_t count, std::size_t remote_sgl_size,
                                       std::uint64_t remote_bytes,
                                       std::vector<SglGroup>* groups) {
  if (groups == nullptr) return Status::InvalidArgument("missing SGL plan output");
  groups->clear();
  if (buffers == nullptr || remote_offsets == nullptr || count == 0 ||
      remote_sgl_size == 0)
    return Status::InvalidArgument("invalid WRITE coalescing plan input");
  constexpr auto max_bytes = std::numeric_limits<std::uint32_t>::max();
  for (std::size_t i = 0; i < count; ++i) {
    const auto bytes = buffers[i].bytes;
    const auto address = reinterpret_cast<std::uintptr_t>(buffers[i].data);
    const auto offset = remote_offsets[i];
    if (address == 0 || bytes == 0 || bytes > max_bytes ||
        bytes > std::numeric_limits<std::uintptr_t>::max() - address ||
        offset > remote_bytes || bytes > remote_bytes - offset) {
      groups->clear();
      return Status::InvalidArgument("WRITE row is outside its buffer or remote region");
    }
    if (!groups->empty()) {
      auto& previous = groups->back();
      const auto base = reinterpret_cast<std::uintptr_t>(buffers[previous.first].data);
      const auto remote_base = remote_offsets[previous.first];
      if (previous.count < remote_sgl_size && bytes <= max_bytes - previous.bytes &&
          address == base + previous.bytes &&
          previous.bytes <= remote_bytes - remote_base &&
          offset == remote_base + previous.bytes) {
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
