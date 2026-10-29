#pragma once

#include "batch_get_abi.h"

// A separate eager ABI: the V2 direct-read/graph ABI remains unchanged.
enum {
  KVAF_SEQ_SCATTER = 1,
  KVAF_SEQ_GATHER = 2,
  KVAF_SEQ_MAX_REQUESTS = 65536,
  KVAF_SEQ_MAX_WORKERS = 8,
  KVAF_SEQ_MAX_SGES = 6,
};
static const uint64_t KVAF_SEQ_STATUS_MAGIC = 0x4b56414653455131ULL;
enum KvafSequentialResult : uint64_t {
  KVAF_SEQ_OK = 0, KVAF_SEQ_INVALID = 1, KVAF_SEQ_TIMEOUT = 2,
  KVAF_SEQ_REMOTE_ERROR = 3, KVAF_SEQ_CQ_ERROR = 4, KVAF_SEQ_QUEUE_ERROR = 5,
};
enum KvafSequentialPhase : uint64_t {
  KVAF_SEQ_VALIDATE = 1, KVAF_SEQ_BUILD = 2, KVAF_SEQ_NOTIFY = 3,
  KVAF_SEQ_WAIT_REMOTE = 4, KVAF_SEQ_PAYLOAD = 5, KVAF_SEQ_COMPLETE = 6,
};
struct alignas(64) KvafSequentialStatus {
  uint64_t magic, sequence, result, phase, requests, data_wrs, sq_pi, cq_ci;
};
struct KvafSequentialBatchGetArgs {
  KvafExecuteBatchGetKernelArgs request;
  uint64_t workspace;
  uint64_t workspace_bytes;
  uint64_t workspace_entity;
  uint64_t descriptor_offset;
  uint64_t staging_offset;
  uint64_t lane_descriptor_bytes;
  uint64_t lane_staging_bytes;
  uint64_t remote_staging;
  uint64_t remote_descriptor;
  uint64_t remote_control;
  uint64_t source_base;
  uint64_t source_bytes;
  uint64_t sequence;
  uint32_t mode;
  uint32_t sgl_size;
  uint32_t max_polls;
  uint32_t reserved;
};
static_assert(sizeof(KvafSequentialStatus) == 64);
static_assert(sizeof(KvafSequentialBatchGetArgs) == 232);

// Device-friendly subset of the shared CPU gather wire protocol.
struct KvafSequentialGatherHeader {
  uint64_t magic, sequence, staging_address;
  uint32_t span_count, total_bytes, pipeline_chunk_bytes, span_bytes;
  uint64_t ready_bytes;
};
struct KvafSequentialGatherSpan { uint64_t address; uint32_t length, reserved; };
static_assert(sizeof(KvafSequentialGatherHeader) == 48);
static_assert(sizeof(KvafSequentialGatherSpan) == 16);
