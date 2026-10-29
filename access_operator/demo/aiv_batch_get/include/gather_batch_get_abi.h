#pragma once
#include "sequential_batch_get_abi.h"

struct KvafGatherBatchGetArgs {
  KvafExecuteBatchGetKernelArgs request;
  uint64_t workspace, workspace_bytes, workspace_entity, descriptor_offset;
  uint64_t lane_descriptor_bytes, lane_staging_bytes;
  uint64_t remote_staging, remote_descriptor, remote_control;
  uint64_t source_base, source_bytes, sequence;
  uint32_t sgl_size, max_polls;
};
static_assert(sizeof(KvafGatherBatchGetArgs) == 216);

// Device-friendly subset of the shared CPU gather wire protocol.
struct KvafSequentialGatherHeader {
  uint64_t magic, sequence, staging_address;
  uint32_t span_count, total_bytes, pipeline_chunk_bytes, span_bytes;
  uint64_t ready_bytes;
};
struct KvafSequentialGatherSpan { uint64_t address; uint32_t length, reserved; };
static_assert(sizeof(KvafSequentialGatherHeader) == 48);
static_assert(sizeof(KvafSequentialGatherSpan) == 16);
