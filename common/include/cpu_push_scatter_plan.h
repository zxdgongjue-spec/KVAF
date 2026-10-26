#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>

#include "cpu_push_get_layout.h"

namespace kvaf::cpu_push {
// Sort request positions, not caller-owned tensors. Most slot IDs need only
// one or two radix passes; already ordered requests skip the radix passes.
inline const std::uint32_t* OrderByOutput(const std::int32_t* rows, std::uint32_t count,
                                        std::uint32_t* order, std::uint32_t* scratch) {
  std::uint32_t highest = 0;
  bool ordered = true;
  for (std::uint32_t i = 0; i < count; ++i) {
    order[i] = i;
    highest |= static_cast<std::uint32_t>(rows[i]);
    if (i && rows[i] < rows[i - 1]) ordered = false;
  }
  if (ordered) return order;
  auto* source = order;
  auto* destination = scratch;
  for (unsigned shift = 0; shift < 32 && (highest >> shift); shift += 8) {
    std::uint32_t positions[256]{};
    for (std::uint32_t i = 0; i < count; ++i)
      ++positions[(static_cast<std::uint32_t>(rows[source[i]]) >> shift) & 255];
    std::uint32_t next = 0;
    for (auto& position : positions) {
      const auto size = position;
      position = next;
      next += size;
    }
    for (std::uint32_t i = 0; i < count; ++i) {
      const auto index = source[i];
      destination[positions[(static_cast<std::uint32_t>(rows[index]) >> shift) & 255]++] = index;
    }
    std::swap(source, destination);
  }
  return source;
}

// Each staging row and its destination follow the same permutation. Rebuild
// every round: neither addresses nor the active prefix are assumed immutable.
inline void BuildScatterPlan(const std::uint64_t* addresses, const std::int32_t* indices,
                             const std::int32_t* offsets, const std::int32_t* rows,
                             std::uint32_t count, std::uint64_t* sources,
                             std::uint32_t* destinations, std::uint32_t* scratch) {
  const auto* order = OrderByOutput(rows, count, destinations, scratch);
  for (std::uint32_t i = 0; i < count; ++i) {
    const auto input = order[i];
    sources[i] = addresses[indices[input]] + static_cast<std::uint32_t>(offsets[input]);
    destinations[i] = static_cast<std::uint32_t>(rows[input]);
  }
}

// Consume only the ready range. Adjacent target slots are one contiguous copy;
// gaps remain untouched. Bound individual securec calls even for large tokens.
template <class Copy>
bool CopyScatterRuns(const std::uint32_t* rows, Range range, std::uint32_t token_bytes,
                     Copy copy) {
  constexpr std::uint64_t kMaxCopyBytes = 1024 * 1024;
  for (auto begin = range.begin; begin < range.end;) {
    auto end = begin + 1;
    while (end < range.end && rows[end] == std::uint64_t{rows[end - 1]} + 1) ++end;
    auto source = std::uint64_t{begin} * token_bytes;
    auto destination = std::uint64_t{rows[begin]} * token_bytes;
    auto remaining = std::uint64_t{end - begin} * token_bytes;
    while (remaining) {
      const auto bytes = std::min(remaining, kMaxCopyBytes);
      if (!copy(destination, source, bytes)) return false;
      source += bytes;
      destination += bytes;
      remaining -= bytes;
    }
    begin = end;
  }
  return true;
}
}  // namespace kvaf::cpu_push
