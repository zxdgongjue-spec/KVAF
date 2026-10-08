#pragma once

#include "gather_output.h"

#include <cstdint>

namespace kvaf::aicpu_urma {

struct alignas(64) GatherTaskState {
  std::uint64_t next_task;
};

inline std::uint32_t GatherTaskCount(std::uint32_t spans, std::uint32_t task_spans) {
  return task_spans == 0U ? 0U : spans / task_spans + (spans % task_spans != 0U);
}

inline std::uint64_t GatherTaskStateOffset(std::uint32_t spans) {
  return (static_cast<std::uint64_t>(spans) * sizeof(GatherOutputSpan) + 63U) &
         ~std::uint64_t{63U};
}

inline std::uint64_t GatherOutputStorageBytes(std::uint32_t spans, std::uint32_t task_spans) {
  return task_spans == 0U ? static_cast<std::uint64_t>(spans) * sizeof(GatherOutputSpan)
      : GatherTaskStateOffset(spans) + sizeof(GatherTaskState) +
          static_cast<std::uint64_t>(GatherTaskCount(spans, task_spans)) * sizeof(std::uint64_t);
}

inline std::uint64_t GatherPackedTargetOffset(std::uint64_t source_bytes) {
  return (source_bytes + 63U) & ~std::uint64_t{63U};
}

inline std::uint64_t GatherReadPackedOutputBytes(std::uint64_t output_extent,
                                                std::uint64_t source_bytes,
                                                std::uint32_t target_count,
                                                std::uint32_t task_spans) {
  if (output_extent > UINT64_MAX - 63U || source_bytes > UINT64_MAX - 63U) return 0U;
  const auto scratch_offset = (output_extent + 63U) & ~std::uint64_t{63U};
  const auto target_offset = GatherPackedTargetOffset(source_bytes);
  const auto target_storage = GatherOutputStorageBytes(target_count, task_spans);
  if (target_offset > UINT64_MAX - target_storage) return 0U;
  const auto metadata_bytes = target_offset + target_storage;
  if (metadata_bytes > UINT64_MAX - 64U ||
      scratch_offset > UINT64_MAX - 64U - metadata_bytes) return 0U;
  return scratch_offset + 64U + metadata_bytes;
}

}
