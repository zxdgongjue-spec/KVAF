#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "cpu_urma/cpu_urma_transport.h"

namespace kvaf::cpu_urma_detail {

// Global Release optimization triggers SIGBUS in the deployed URMA provider
// ABI path. These helpers are pure CPU planning code, so optimize only them
// without changing compilation of initialization or provider calls.
#if defined(__GNUC__) && !defined(__clang__)
#define KVAF_CPU_URMA_PLAN_HOT __attribute__((hot, noinline, optimize("O3")))
#else
#define KVAF_CPU_URMA_PLAN_HOT
#endif

struct SglGroup {
  std::size_t first{0};
  std::size_t count{0};
  std::uint64_t bytes{0};
};

// One independently planned contiguous range. A caller may process different
// slices concurrently because each worker writes only its own status/groups.
// Group indices remain relative to the slice until MergeSglPlanSlices().
struct SglPlanSlice {
  std::size_t first{0};
  std::size_t count{0};
  Status status{};
  std::vector<SglGroup> groups;
};

inline Status PrepareSglPlanSlices(std::size_t row_count,
                                   std::size_t requested_slice_count,
                                   std::vector<SglPlanSlice>* slices) {
  if (slices == nullptr) {
    return Status::InvalidArgument("missing SGL plan slices");
  }
  if (row_count == 0U || requested_slice_count == 0U) {
    slices->clear();
    return Status::InvalidArgument("invalid SGL plan slice count");
  }

  const std::size_t slice_count = std::min(row_count, requested_slice_count);
  slices->resize(slice_count);
  const std::size_t rows_per_slice = row_count / slice_count;
  const std::size_t remainder = row_count % slice_count;
  std::size_t first = 0U;
  for (std::size_t i = 0U; i < slice_count; ++i) {
    auto& slice = (*slices)[i];
    slice.first = first;
    slice.count = rows_per_slice + (i < remainder ? 1U : 0U);
    slice.status = Status::OK();
    slice.groups.clear();
    first += slice.count;
  }
  return Status::OK();
}

KVAF_CPU_URMA_PLAN_HOT inline bool GroupsFitRegisteredRegion(
    const CpuUrmaTransport::HostSge* buffers,
    const SglGroup* groups,
    std::size_t group_count,
    const void* region_address,
    std::size_t region_bytes) {
  if (buffers == nullptr || groups == nullptr || group_count == 0U ||
      region_address == nullptr || region_bytes == 0U) {
    return false;
  }
  const auto base = reinterpret_cast<std::uintptr_t>(region_address);
  for (std::size_t i = 0U; i < group_count; ++i) {
    const std::size_t first = groups[i].first - groups[0].first;
    const auto requested = reinterpret_cast<std::uintptr_t>(buffers[first].data);
    if (requested < base || groups[i].bytes > region_bytes ||
        requested - base > region_bytes - groups[i].bytes) {
      return false;
    }
  }
  return true;
}

// A one-to-one batch needs validation, but no scatter-group array.
KVAF_CPU_URMA_PLAN_HOT inline Status ValidateBatchRows(
    const CpuUrmaTransport::HostSge* buffers,
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
KVAF_CPU_URMA_PLAN_HOT inline Status PlanWriteCoalescedGroups(
    const CpuUrmaTransport::HostSge* buffers,
    const std::uint64_t* remote_offsets,
    std::size_t count, std::size_t remote_sgl_size,
    std::uint64_t remote_bytes,
    std::vector<SglGroup>* groups,
    std::uint64_t max_group_bytes =
        std::numeric_limits<std::uint32_t>::max()) {
  if (groups == nullptr) return Status::InvalidArgument("missing SGL plan output");
  groups->clear();
  if (buffers == nullptr || remote_offsets == nullptr || count == 0 ||
      remote_sgl_size == 0 || max_group_bytes == 0)
    return Status::InvalidArgument("invalid WRITE coalescing plan input");
  const auto max_bytes = std::min<std::uint64_t>(
      max_group_bytes, std::numeric_limits<std::uint32_t>::max());
  const auto group_is_valid = [&](const SglGroup& group) {
    const auto address = reinterpret_cast<std::uintptr_t>(buffers[group.first].data);
    const auto offset = remote_offsets[group.first];
    return group.bytes <= std::numeric_limits<std::uintptr_t>::max() - address &&
        offset <= remote_bytes && group.bytes <= remote_bytes - offset;
  };
  for (std::size_t i = 0; i < count; ++i) {
    const auto bytes = buffers[i].bytes;
    const auto address = reinterpret_cast<std::uintptr_t>(buffers[i].data);
    const auto offset = remote_offsets[i];
    if (address == 0 || bytes == 0 || bytes > max_bytes) {
      groups->clear();
      return Status::InvalidArgument("invalid WRITE row");
    }
    if (!groups->empty()) {
      auto& previous = groups->back();
      const auto base = reinterpret_cast<std::uintptr_t>(buffers[previous.first].data);
      const auto remote_base = remote_offsets[previous.first];
      if (previous.count < remote_sgl_size && bytes <= max_bytes - previous.bytes &&
          address == base + previous.bytes &&
          offset == remote_base + previous.bytes) {
        ++previous.count;
        previous.bytes += bytes;
        continue;
      }
      if (!group_is_valid(previous)) {
        groups->clear();
        return Status::InvalidArgument("WRITE group is outside its buffer or remote region");
      }
    }
    groups->push_back(SglGroup{i, 1, bytes});
  }
  if (!group_is_valid(groups->back())) {
    groups->clear();
    return Status::InvalidArgument("WRITE group is outside its buffer or remote region");
  }
  return Status::OK();
}

inline Status PlanWriteCoalescedSlice(
    const CpuUrmaTransport::HostSge* buffers,
    const std::uint64_t* remote_offsets,
    std::size_t row_count, std::size_t remote_sgl_size,
    std::uint64_t remote_bytes, SglPlanSlice* slice,
    std::uint64_t max_group_bytes =
        std::numeric_limits<std::uint32_t>::max()) {
  if (slice == nullptr) {
    return Status::InvalidArgument("missing SGL plan slice");
  }
  slice->groups.clear();
  if (buffers == nullptr || remote_offsets == nullptr || slice->count == 0U ||
      slice->first > row_count || slice->count > row_count - slice->first) {
    slice->status = Status::InvalidArgument("invalid SGL plan slice");
    return slice->status;
  }
  slice->status = PlanWriteCoalescedGroups(
      buffers + slice->first, remote_offsets + slice->first, slice->count,
      remote_sgl_size, remote_bytes, &slice->groups, max_group_bytes);
  return slice->status;
}

// Merge only after every planning worker has finished. Slice boundaries are
// intentionally retained: this avoids another row scan and costs at most one
// extra WR per boundary compared with a single serial plan.
KVAF_CPU_URMA_PLAN_HOT inline Status MergeSglPlanSlices(
    const std::vector<SglPlanSlice>& slices,
    std::size_t row_count,
    std::vector<SglGroup>* groups) {
  if (groups == nullptr) {
    return Status::InvalidArgument("missing merged SGL plan output");
  }
  groups->clear();
  if (slices.empty() || row_count == 0U) {
    return Status::InvalidArgument("invalid SGL plan slices");
  }

  std::size_t expected_first = 0U;
  std::size_t group_count = 0U;
  for (const auto& slice : slices) {
    if (!slice.status.ok()) return slice.status;
    if (expected_first > row_count || slice.first != expected_first ||
        slice.count == 0U ||
        slice.count > row_count - expected_first) {
      return Status::InvalidArgument("SGL plan slices do not cover the batch");
    }
    if (slice.groups.empty() || slice.groups.front().first != 0U ||
        slice.groups.back().first > slice.count ||
        slice.groups.back().count >
            slice.count - slice.groups.back().first ||
        slice.groups.back().first + slice.groups.back().count != slice.count ||
        slice.groups.size() >
            std::numeric_limits<std::size_t>::max() - group_count) {
      return Status::InvalidArgument("incomplete SGL plan slice");
    }
    group_count += slice.groups.size();
    expected_first += slice.count;
  }
  if (expected_first != row_count) {
    return Status::InvalidArgument("SGL plan slices do not cover the batch");
  }

  groups->reserve(group_count);
  for (const auto& slice : slices) {
    for (auto group : slice.groups) {
      group.first += slice.first;
      groups->push_back(group);
    }
  }
  return Status::OK();
}

#undef KVAF_CPU_URMA_PLAN_HOT

}  // namespace kvaf::cpu_urma_detail
