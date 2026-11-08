#pragma once

#include <limits>

#include "access_operator.h"
#include "aicpu_batch_get/include/batch_get_abi.h"

namespace kvaf::detail {
struct BatchGetExecutionState {
  void* status{nullptr};
  std::size_t status_bytes{0};
};
inline std::uint64_t PointerValue(const void* pointer) {
  return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pointer));
}

inline Status BuildBatchGetKernelArgs(const BatchGetExecRequest& exec,
                                      aicpu_urma::AicpuBatchGetInput* args) {
  if (!exec.request || !args) return Status::InvalidArgument("BatchGet request is missing");
  const auto& request = *exec.request;
  const auto count = request.indices.bytes / sizeof(std::int32_t);
  const auto entries = request.device_access_table.addresses.bytes / sizeof(Address);
  const auto limit = std::numeric_limits<std::uint32_t>::max();
  if (!request.indices.has_elements<std::int32_t>() ||
      request.indices.device != DeviceKind::kNpu || request.offsets.device != DeviceKind::kNpu ||
      request.output_indices.device != DeviceKind::kNpu || request.output.device != DeviceKind::kNpu ||
      !request.offsets.data || !request.output_indices.data || !request.output.valid() ||
      request.offsets.bytes != request.indices.bytes || request.output_indices.bytes != request.indices.bytes ||
      !request.output_block_bytes || request.output.bytes % request.output_block_bytes ||
      !request.device_access_table.addresses.data || !entries ||
      request.device_access_table.addresses.bytes % sizeof(Address) ||
      request.device_access_table.addresses.device != DeviceKind::kNpu ||
      count > limit || entries > limit || request.output_block_bytes > limit)
    return Status::InvalidArgument("invalid AICPU BatchGet layout");
  if (!request.miss_count.has_elements<std::int32_t>() ||
      request.miss_count.bytes != sizeof(std::int32_t) ||
      request.miss_count.device != DeviceKind::kNpu) {
    return Status::InvalidArgument("AICPU BatchGet requires an int32 miss_count");
  }
  *args = {PointerValue(request.device_access_table.addresses.data), PointerValue(request.indices.data),
           PointerValue(request.offsets.data), PointerValue(request.miss_count.data),
           PointerValue(request.output_indices.data),
           static_cast<std::uint32_t>(entries), static_cast<std::uint32_t>(count),
           static_cast<std::uint32_t>(request.output_block_bytes)};
  args->valid_table = PointerValue(request.device_access_table.valid.data);
  if (!request.tokens_per_block ||
      request.tokens_per_block > UINT64_MAX / request.output_block_bytes)
    return Status::InvalidArgument("invalid AICPU block layout");
  args->block_bytes = request.tokens_per_block * request.output_block_bytes;
  return Status::OK();
}
}  // namespace kvaf::detail
