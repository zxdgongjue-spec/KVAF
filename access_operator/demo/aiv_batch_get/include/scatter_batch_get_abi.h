#pragma once
#include "sequential_batch_get_abi.h"

enum KvafSequentialScatterProbe : uint32_t {
  KVAF_SEQ_PROBE_NONE = 0, KVAF_SEQ_PROBE_DONE = 1,
  KVAF_SEQ_PROBE_FLUSH = 2, KVAF_SEQ_PROBE_LOAD = 3, KVAF_SEQ_PROBE_COPY = 4,
};
struct KvafScatterBatchGetArgs {
  KvafExecuteBatchGetKernelArgs request;
  uint64_t workspace, workspace_bytes, descriptor_offset, staging_offset;
  uint64_t source_base, source_bytes, sequence;
  uint32_t max_polls, scatter_probe;
};
static_assert(sizeof(KvafScatterBatchGetArgs) == 176);
