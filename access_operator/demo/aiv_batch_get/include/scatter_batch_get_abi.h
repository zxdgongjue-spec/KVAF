#pragma once
#include "sequential_batch_get_abi.h"

struct KvafScatterBatchGetArgs {
  KvafExecuteBatchGetKernelArgs request;
  uint64_t workspace, workspace_bytes, descriptor_offset, staging_offset;
  uint64_t source_base, source_bytes, sequence;
  // Keep the 176-byte launch ABI; reserved must be zero.
  uint32_t max_polls, reserved;
};
static_assert(sizeof(KvafScatterBatchGetArgs) == 176);
