#pragma once
#include "sequential_batch_get_abi.h"
#include "scatter_batch_get_profile.h"

// The kernel supplies the AIV qualifier; Host users keep ordinary C++ helpers.
#ifndef KVAF_SCATTER_OPTIONS_INLINE
#define KVAF_SCATTER_OPTIONS_INLINE inline
#endif

namespace kvaf {
// Shared UB footprint of this mixed SIMD/SIMT binary, reserved by its loader.
// LocalTensor/get_imm address views do not reserve shared memory for SIMT.
constexpr uint32_t kScatterLocalMemoryBytes = 64U * 1024U;
// Versioned runtime options in the existing reserved word. Unknown versions or
// bits must fail validation rather than silently select a different algorithm.
constexpr uint32_t kScatterOptionsV1 = 0x53430100U;
constexpr uint32_t kScatterProfiling = 4U;
// Host's existing trusted-input contract also applies to device list contents.
constexpr uint32_t kScatterTrustedInputs = 8U;
constexpr uint32_t kScatterPipeline = 32U;
constexpr uint32_t kScatterOptionsMask = kScatterProfiling | kScatterTrustedInputs | kScatterPipeline;
KVAF_SCATTER_OPTIONS_INLINE constexpr bool ScatterOptionsValid(uint32_t value) {
  return value == 0 || value == kScatterProfileFlag ||
      (value & ~kScatterOptionsMask) == kScatterOptionsV1;
}
KVAF_SCATTER_OPTIONS_INLINE constexpr bool ScatterOptionEnabled(uint32_t value, uint32_t option) {
  return (value & ~kScatterOptionsMask) == kScatterOptionsV1 && (value & option);
}
KVAF_SCATTER_OPTIONS_INLINE constexpr bool ScatterProfileEnabled(uint32_t value) {
  return value == kScatterProfileFlag || ScatterOptionEnabled(value, kScatterProfiling);
}
KVAF_SCATTER_OPTIONS_INLINE constexpr uint32_t ScatterWithProfiling(uint32_t value, bool enabled) {
  // Profiling does not change the fixed fused-build/double-buffer algorithm.
  return value == 0 || value == kScatterProfileFlag ? (enabled ? kScatterProfileFlag : 0U) :
      (value & ~kScatterProfiling) | (enabled ? kScatterProfiling : 0U);
}
}  // namespace kvaf

// Persistent private workspace state, referenced through the existing request
// entity/queue pointers. The public 176-byte launch ABI is unchanged.
struct alignas(64) KvafScatterMailboxContext {
  KvafBatchGetDirectUrmaEntity entity;
  KvafBatchGetQueueState queue;
  uint32_t max_write_bytes;
  uint32_t reserved[15];
};
static_assert(sizeof(KvafScatterMailboxContext) == 320);

struct KvafScatterBatchGetArgs {
  KvafExecuteBatchGetKernelArgs request;
  uint64_t workspace, workspace_bytes, descriptor_offset, staging_offset;
  uint64_t source_base, source_bytes, sequence;
  // Keep the 176-byte ABI: versioned options, or legacy zero/profile-only tag.
  uint32_t max_polls, reserved;
};
static_assert(sizeof(KvafScatterBatchGetArgs) == 176);
