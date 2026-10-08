#pragma once

#include <limits>

#include "requests.h"
#include "status.h"

namespace kvaf {

// Host-side descriptor/layout checks only; never read device memory or resolve
// addresses. Shared by Runtime and the independently callable AccessOperator.
inline Status ValidateRawBatchGetRequest(const BatchGetRequest& request) {
  if (!request.remote_addresses.has_elements<Address>() ||
      !request.local_addresses.has_elements<Address>() ||
      request.remote_addresses.bytes != request.local_addresses.bytes ||
      request.remote_addresses.device != DeviceKind::kNpu ||
      request.local_addresses.device != DeviceKind::kNpu ||
      !request.miss_count.has_elements<std::int32_t>() ||
      request.miss_count.bytes != sizeof(std::int32_t) ||
      request.miss_count.device != DeviceKind::kNpu ||
      !request.output.valid() || request.output.device != DeviceKind::kNpu ||
      request.output_block_bytes == 0 ||
      request.output_block_bytes > request.output.bytes) {
    return Status::InvalidArgument(
        "raw BatchGet requires equal non-empty NPU uint64 address arrays, "
        "one NPU int32 miss_count, and a local output registration");
  }
  if (request.access_table.opaque != 0 || !request.device_access_table.empty() ||
      !request.indices.empty() || !request.offsets.empty() ||
      !request.output_indices.empty() || request.tokens_per_block != 0) {
    return Status::InvalidArgument("raw BatchGet must not include lookup fields");
  }
  if (request.remote_addresses.bytes / sizeof(Address) >
          static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
      request.output_block_bytes > std::numeric_limits<std::uint32_t>::max()) {
    return Status::InvalidArgument("raw BatchGet dimensions exceed the device ABI");
  }
  return Status::OK();
}

}  // namespace kvaf
