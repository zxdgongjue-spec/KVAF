#include "host_plan.h"

#include <algorithm>
namespace kvaf::aicpu_urma {
GatherDimensions ComputeGatherDimensions(const CpuGatherReadRequest& request) {
  std::uint64_t bytes = 0U, extent = 0U;
  for (const auto& span : request.spans) bytes += span.length;
  for (const auto& span : request.output_spans)
    extent = std::max(extent, span.offset + span.length);
  return {static_cast<std::uint32_t>(bytes), request.output_spans.empty() ? bytes : extent};
}
}  // namespace kvaf::aicpu_urma
