#pragma once
#include "sequential_batch_get_abi.h"
#include "scatter_batch_get_profile.h"

namespace kvaf {
// Versioned runtime options in the existing reserved word. Unknown versions or
// bits must fail validation rather than silently select a different algorithm.
constexpr uint32_t kScatterOptionsV1 = 0x53430100U;
constexpr uint32_t kScatterFusedBuild = 1U;
constexpr uint32_t kScatterDoubleBuffer = 2U;
constexpr uint32_t kScatterProfiling = 4U;
constexpr uint32_t kScatterOptionsMask = 7U;
inline constexpr bool ScatterOptionsValid(uint32_t value) {
  return value == 0 || value == kScatterProfileFlag ||
      (value & ~kScatterOptionsMask) == kScatterOptionsV1;
}
inline constexpr bool ScatterOptionEnabled(uint32_t value, uint32_t option) {
  return (value & ~kScatterOptionsMask) == kScatterOptionsV1 && (value & option);
}
inline constexpr bool ScatterProfileEnabled(uint32_t value) {
  return value == kScatterProfileFlag || ScatterOptionEnabled(value, kScatterProfiling);
}
inline constexpr uint32_t ScatterWithProfiling(uint32_t value, bool enabled) {
  // Preserve the two-pass/single-bank behavior of legacy Host arguments.
  return value == 0 || value == kScatterProfileFlag ? (enabled ? kScatterProfileFlag : 0U) :
      (value & ~kScatterProfiling) | (enabled ? kScatterProfiling : 0U);
}
}  // namespace kvaf

struct KvafScatterBatchGetArgs {
  KvafExecuteBatchGetKernelArgs request;
  uint64_t workspace, workspace_bytes, descriptor_offset, staging_offset;
  uint64_t source_base, source_bytes, sequence;
  // Keep the 176-byte ABI: versioned options, or legacy zero/profile-only tag.
  uint32_t max_polls, reserved;
};
static_assert(sizeof(KvafScatterBatchGetArgs) == 176);
