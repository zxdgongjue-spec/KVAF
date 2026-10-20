#pragma once
#include "gather_read_request.h"
namespace kvaf::aicpu_urma {
struct GatherDimensions {
  std::uint32_t bytes;
  std::uint64_t extent;
};
GatherDimensions ComputeGatherDimensions(const CpuGatherReadRequest& request);
}  // namespace kvaf::aicpu_urma
