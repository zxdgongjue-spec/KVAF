#ifndef KVAF_AIV_BATCH_PUT_ABI_H_
#define KVAF_AIV_BATCH_PUT_ABI_H_

#include <stdint.h>

enum {
  KVAF_AIV_BATCH_PUT_STATUS_BYTES = 384,
  KVAF_AIV_BATCH_PUT_MAX_AIV_CORES = 48,
  KVAF_AIV_BATCH_PUT_MAX_JETTYS_PER_CORE = 2,
};

static const uint64_t KVAF_AIV_BATCH_PUT_STATUS_MAGIC = 0x4b5641464250444fULL;

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

// Runtime-facing eager BatchPut V1 ABI. All pointer fields are device
// addresses. The three metadata arrays contain exactly row_count entries.
typedef struct KvafExecuteBatchPutKernelArgs {
  uint64_t input;                 // uint8[input_bytes]
  uint64_t input_offsets;         // uint64[row_count]
  uint64_t lengths;               // uint64[row_count]
  uint64_t destination_addresses; // uint64[row_count]
  uint64_t core_status;           // KvafBatchPutCoreStatus[aiv_core_count]
  uint64_t input_bytes;
  uint32_t row_count;
  uint32_t aiv_core_count;
} KvafExecuteBatchPutKernelArgs;

#if defined(__cplusplus)
static_assert(sizeof(KvafBatchPutCoreStatus) == KVAF_AIV_BATCH_PUT_STATUS_BYTES,
              "KvafBatchPutCoreStatus ABI changed");
static_assert(sizeof(KvafExecuteBatchPutKernelArgs) == 56,
              "KvafExecuteBatchPutKernelArgs ABI changed");
#endif

#endif  // KVAF_AIV_BATCH_PUT_ABI_H_
