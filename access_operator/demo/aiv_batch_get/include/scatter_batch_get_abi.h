#pragma once
#include "sequential_batch_get_abi.h"

enum KvafSequentialScatterProbe : uint32_t {
  KVAF_SEQ_PROBE_NONE = 0, KVAF_SEQ_PROBE_DONE = 1,
  KVAF_SEQ_PROBE_FLUSH = 2, KVAF_SEQ_PROBE_LOAD = 3, KVAF_SEQ_PROBE_COPY = 4,
  // Append IDs so existing Host/kernel probe values retain their meaning.
  KVAF_SEQ_PROBE_SETUP = 5, KVAF_SEQ_PROBE_DMA = 6,
  KVAF_SEQ_PROBE_GM = 7, KVAF_SEQ_PROBE_BODY = 8, KVAF_SEQ_PROBE_UB = 9,
};
// Distinguish a byte-verified MTE2 load from the legacy AFTER_LOAD checkpoint.
// Launch/status layouts stay unchanged; old Host/kernel combinations fail closed.
constexpr uint64_t KVAF_SCATTER_AFTER_LOAD_VERIFIED = 11;
constexpr uint64_t KVAF_SCATTER_AFTER_SETUP = 12;
constexpr uint64_t KVAF_SCATTER_AFTER_DMA = 13;
constexpr uint64_t KVAF_SCATTER_MTE2_MISMATCH = 7;
constexpr uint64_t KVAF_SCATTER_AFTER_GM = 14;
constexpr uint64_t KVAF_SCATTER_AFTER_BODY = 15;
constexpr uint64_t KVAF_SCATTER_AFTER_UB = 16;
// Scatter probes use data_wrs as a diagnostic revision, sq_pi as a checksum
// (or mismatch byte offset), cq_ci as bytes read (or packed GM/UB bytes).
// Gather keeps its original status interpretation. Launch/status sizes do not change.
constexpr uint64_t KVAF_SCATTER_DIAGNOSTIC_REVISION = 1;
struct KvafScatterBatchGetArgs {
  KvafExecuteBatchGetKernelArgs request;
  uint64_t workspace, workspace_bytes, descriptor_offset, staging_offset;
  uint64_t source_base, source_bytes, sequence;
  uint32_t max_polls, scatter_probe;
};
static_assert(sizeof(KvafScatterBatchGetArgs) == 176);
