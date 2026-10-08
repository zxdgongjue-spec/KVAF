#pragma once

#include "gather_output.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace kvaf {

inline bool ValidateGatherOutput(std::uint64_t output_bytes,
                                 const GatherOutputSpan* spans, std::size_t span_count,
                                 std::uint32_t expected_bytes,
                                 std::uint64_t* output_extent,
                                 std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges) {
  if (output_extent == nullptr || spans == nullptr || span_count == 0U || expected_bytes == 0U)
    return false;
  std::uint64_t total = 0U;
  std::uint64_t extent = 0U;
  std::uint64_t previous_offset = 0U;
  std::uint64_t previous_end = 0U;
  bool ascending = true;
  bool descending = true;
  bool first = true;
  for (std::size_t index = 0U; index < span_count; ++index) {
    const auto& span = spans[index];
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
    ranges.clear();
    ranges.reserve(span_count);
    for (std::size_t index = 0U; index < span_count; ++index) {
      const auto& span = spans[index];
      ranges.emplace_back(span.offset, span.offset + span.length);
    }
    std::sort(ranges.begin(), ranges.end());
    for (std::size_t index = 1U; index < ranges.size(); ++index)
      if (ranges[index].first < ranges[index - 1U].second) return false;
  }
  *output_extent = extent;
  return true;
}

inline bool ValidateGatherOutput(std::uint64_t output_bytes,
                                 const std::vector<GatherOutputSpan>& spans,
                                 std::uint32_t expected_bytes,
                                 std::uint64_t* output_extent) {
  std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
  return ValidateGatherOutput(output_bytes, spans.data(), spans.size(), expected_bytes,
                              output_extent, ranges);
}

}  // namespace kvaf
