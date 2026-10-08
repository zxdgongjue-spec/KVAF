#ifndef KVAF_AIV_BATCH_PUT_ABI_H_
#define KVAF_AIV_BATCH_PUT_ABI_H_

#include <stdint.h>

enum {
  KVAF_AIV_BATCH_PUT_STATUS_BYTES = 384,
  KVAF_AIV_BATCH_PUT_MAX_AIV_CORES = 48,
  KVAF_AIV_BATCH_PUT_MAX_JETTYS_PER_CORE = 2,
};

static const uint64_t KVAF_AIV_BATCH_PUT_STATUS_MAGIC = 0x4b5641464250444fULL;
// AICPU-only error snapshot in reserved[]. Existing status size is unchanged.
static const uint64_t KVAF_AICPU_BATCH_PUT_DIAGNOSTIC_MAGIC = 0x4b56414644494147ULL;
enum {
  KVAF_PUT_DIAG_MAGIC = 1,
  KVAF_PUT_DIAG_BATCH_COUNT = 2,
  KVAF_PUT_DIAG_FIRST_WQE = 4,  // 64 bytes
  KVAF_PUT_DIAG_LAST_WQE = 12,  // 64 bytes
  KVAF_PUT_DIAG_CQE = 20,      // first 32 bytes
};

enum KvafBatchPutCoreResult {
  KVAF_BATCH_PUT_CORE_SUCCESS = 0,
  KVAF_BATCH_PUT_CORE_INVALID_ARGUMENT = 1,
  KVAF_BATCH_PUT_CORE_INVALID_ENTITY = 2,
  KVAF_BATCH_PUT_CORE_INVALID_QUEUE = 3,
  KVAF_BATCH_PUT_CORE_INVALID_SOURCE_RANGE = 4,
  KVAF_BATCH_PUT_CORE_INVALID_DESTINATION_RANGE = 5,
  KVAF_BATCH_PUT_CORE_QUEUE_FULL = 6,
  KVAF_BATCH_PUT_CORE_CQ_TIMEOUT = 7,
  KVAF_BATCH_PUT_CORE_CQ_ERROR = 8,
  KVAF_BATCH_PUT_CORE_INVALID_LENGTH = 9,
};

typedef struct KvafBatchPutCoreStatus {
  uint64_t magic;
  uint64_t processed_request_count;
  uint64_t result;
  uint64_t completed_batch_count;
  uint64_t sq_producer_index;
  uint64_t cq_consumer_index;
  uint64_t failed_request_index;
  uint64_t last_wqe_protocol;
  uint64_t cq_status;
  uint64_t cq_sub_status;
  uint64_t reserved[38];
} KvafBatchPutCoreStatus;

typedef struct KvafBatchPutSourceDescriptor {
  uint64_t base_address;
  uint32_t local_token_id;
  uint32_t reserved;
} KvafBatchPutSourceDescriptor;

// All pointer fields are device addresses. Per-row arrays contain row_count
// entries; source_descriptors contains one entry per caller input tensor.
typedef struct KvafExecuteBatchPutKernelArgs {
  uint64_t source_descriptors;    // KvafBatchPutSourceDescriptor[source_count]
  uint64_t source_indices;        // uint32[row_count]
  uint64_t input_offsets;         // uint64[row_count]
  uint64_t lengths;               // uint64[row_count]
  uint64_t destination_addresses; // uint64[row_count]
  uint64_t core_status;           // KvafBatchPutCoreStatus[aiv_core_count]
  uint32_t row_count;
  uint32_t aiv_core_count;
} KvafExecuteBatchPutKernelArgs;

#if defined(__cplusplus)
static_assert(sizeof(KvafBatchPutCoreStatus) == KVAF_AIV_BATCH_PUT_STATUS_BYTES,
              "KvafBatchPutCoreStatus ABI changed");
static_assert(sizeof(KvafBatchPutSourceDescriptor) == 16,
              "KvafBatchPutSourceDescriptor ABI changed");
static_assert(sizeof(KvafExecuteBatchPutKernelArgs) == 56,
              "KvafExecuteBatchPutKernelArgs ABI changed");
#endif

#endif  // KVAF_AIV_BATCH_PUT_ABI_H_
