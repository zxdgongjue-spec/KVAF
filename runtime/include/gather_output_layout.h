#pragma once

#include "gather_output.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace kvaf {

inline bool ValidateGatherOutput(std::uint64_t output_bytes,
                                 const std::vector<GatherOutputSpan>& spans,
                                 std::uint32_t expected_bytes,
                                 std::uint64_t* output_extent) {
  if (output_extent == nullptr || spans.empty() || expected_bytes == 0U) return false;
  std::uint64_t total = 0U;
  std::uint64_t extent = 0U;
  std::uint64_t previous_offset = 0U;
  std::uint64_t previous_end = 0U;
  bool ascending = true;
  bool descending = true;
  bool first = true;
  for (const auto& span : spans) {
    if (span.reserved != 0U || span.length == 0U ||
        span.offset > UINT64_MAX - span.length ||
        span.length > expected_bytes - total) return false;
    const std::uint64_t end = span.offset + span.length;
    if (!first) {
      ascending = ascending && previous_end <= span.offset;
      descending = descending && end <= previous_offset;
    }
    previous_offset = span.offset;
    previous_end = end;
    first = false;
    total += span.length;
    extent = std::max(extent, end);
  }
  if (total != expected_bytes || extent > UINT64_MAX - 63U) return false;
  const std::uint64_t scratch_offset = (extent + 63U) & ~std::uint64_t{63U};
  if (scratch_offset > output_bytes || output_bytes - scratch_offset < 64U) return false;
  if (!ascending && !descending) {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    ranges.reserve(spans.size());
    for (const auto& span : spans)
      ranges.emplace_back(span.offset, span.offset + span.length);
    std::sort(ranges.begin(), ranges.end());
    for (std::size_t index = 1U; index < ranges.size(); ++index)
      if (ranges[index].first < ranges[index - 1U].second) return false;
  }
  *output_extent = extent;
  return true;
}

}  // namespace kvaf
