#pragma once
#include "gather_output.h"
namespace kvaf::aicpu_urma {
inline std::uint64_t GatherScratchBytes(std::uint32_t pipeline_chunk_bytes) {
  return pipeline_chunk_bytes ? 8U * 64U : 64U;
}
inline std::uint64_t GatherPackedTargetOffset(std::uint64_t source_bytes) {
  return (source_bytes + 63U) & ~std::uint64_t{63U};
}
inline std::uint64_t GatherReadPackedOutputBytes(std::uint64_t extent, std::uint64_t source_bytes,
                                                 std::uint32_t targets,
                                                 std::uint32_t pipeline_chunk_bytes = 131072U) {
  if (extent > UINT64_MAX - 63U || source_bytes > UINT64_MAX - 63U) return 0;
  const auto scratch = (extent + 63U) & ~std::uint64_t{63U};
  const auto target_offset = GatherPackedTargetOffset(source_bytes);
  const auto target_bytes = static_cast<std::uint64_t>(targets) * sizeof(GatherOutputSpan);
  if (target_offset > UINT64_MAX - target_bytes) return 0;
  const auto metadata = target_offset + target_bytes;
  const auto scratch_bytes = GatherScratchBytes(pipeline_chunk_bytes);
  if (metadata > UINT64_MAX - scratch_bytes || scratch > UINT64_MAX - scratch_bytes - metadata)
    return 0;
  return scratch + scratch_bytes + metadata;
}
}  // namespace kvaf::aicpu_urma
