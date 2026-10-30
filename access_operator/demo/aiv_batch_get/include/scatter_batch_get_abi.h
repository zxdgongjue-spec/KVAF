#pragma once
#include "sequential_batch_get_abi.h"
#include "scatter_batch_get_profile.h"

struct KvafScatterBatchGetArgs {
  KvafExecuteBatchGetKernelArgs request;
  uint64_t workspace, workspace_bytes, descriptor_offset, staging_offset;
  uint64_t source_base, source_bytes, sequence;
  // Keep the 176-byte ABI: zero or the versioned kScatterProfileFlag tag.
  uint32_t max_polls, reserved;
};
static_assert(sizeof(KvafScatterBatchGetArgs) == 176);
