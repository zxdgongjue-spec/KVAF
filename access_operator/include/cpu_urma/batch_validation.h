#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "cpu_urma/cpu_urma_transport.h"

namespace kvaf::cpu_urma_detail {

#if defined(__GNUC__) && !defined(__clang__)
#define KVAF_CPU_URMA_VALIDATE_HOT __attribute__((hot, noinline, optimize("O3")))
#else
#define KVAF_CPU_URMA_VALIDATE_HOT
#endif

struct BatchValidationSlice {
  std::size_t first{0U};
  std::size_t count{0U};
  Status status{};
};

inline Status PrepareBatchValidationSlices(
    std::size_t row_count, std::size_t requested_slice_count,
    std::vector<BatchValidationSlice>* slices) {
  if (slices == nullptr) {
    return Status::InvalidArgument("missing batch validation slices");
  }
  if (row_count == 0U || requested_slice_count == 0U) {
    slices->clear();
    return Status::InvalidArgument("invalid batch validation slice count");
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
    first += slice.count;
  }
  return Status::OK();
}

KVAF_CPU_URMA_VALIDATE_HOT inline Status ValidateBatchRows(
    const CpuUrmaTransport::HostSge* buffers,
    const std::uint64_t* offsets,
    std::size_t count,
    std::uint64_t remote_bytes,
    std::uint64_t max_row_bytes =
        std::numeric_limits<std::uint32_t>::max()) {
  if (buffers == nullptr || offsets == nullptr || count == 0U) {
    return Status::InvalidArgument("invalid batch rows");
  }
  const auto max_bytes = std::min<std::uint64_t>(
      max_row_bytes, std::numeric_limits<std::uint32_t>::max());
  for (std::size_t i = 0U; i < count; ++i) {
    const auto address = reinterpret_cast<std::uintptr_t>(buffers[i].data);
    const auto bytes = buffers[i].bytes;
    if (address == 0U || bytes == 0U || bytes > max_bytes ||
        bytes > std::numeric_limits<std::uintptr_t>::max() - address ||
        offsets[i] > remote_bytes || bytes > remote_bytes - offsets[i]) {
      return Status::InvalidArgument(
          "batch row is outside its buffer or remote region");
    }
  }
  return Status::OK();
}

#undef KVAF_CPU_URMA_VALIDATE_HOT

}  // namespace kvaf::cpu_urma_detail
