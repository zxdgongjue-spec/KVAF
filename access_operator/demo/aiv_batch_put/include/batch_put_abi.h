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
  KVAF_BATCH_PUT_CORE_SQ_PUBLICATION_ERROR = 10,
};

// Scalar SQ publication uses 0..7 with its own marker in word 37. Legacy
// one-row SGE snapshots have no SQ marker and remain distinguishable.
static const uint64_t KVAF_BATCH_PUT_SQ_DIAGNOSTIC_MAGIC = 0x5351505542303031ULL;
enum {
  KVAF_BATCH_PUT_SQ_DIAGNOSTIC_MAGIC_WORD = 37,
  KVAF_BATCH_PUT_SQ_DIAGNOSTIC_WORDS = 4,
  KVAF_BATCH_PUT_SQ_DIAG_PUBLISHED_PI = 0,
  KVAF_BATCH_PUT_SQ_DIAG_FIRST_CONTROL = 1,
  KVAF_BATCH_PUT_SQ_DIAG_LAST_CONTROL = 2,
  KVAF_BATCH_PUT_SQ_DIAG_MISMATCH_WORD = 3,
};

// CQ wait diagnostics use only existing reserved words: one version marker
// followed by two 14-word queue records. SQ publication occupies 0..7 and 37.
static const uint64_t KVAF_BATCH_PUT_CQ_DIAGNOSTIC_MAGIC = 0x4351444941473031ULL;
enum {
  KVAF_BATCH_PUT_CQ_DIAGNOSTIC_MAGIC_WORD = 8,
  KVAF_BATCH_PUT_CQ_DIAGNOSTIC_BASE_WORD = 9,
  KVAF_BATCH_PUT_CQ_DIAGNOSTIC_WORDS_PER_JETTY = 14,
  KVAF_BATCH_PUT_CQ_DIAG_RESULT = 0,
  KVAF_BATCH_PUT_CQ_DIAG_POLLS = 1,
  KVAF_BATCH_PUT_CQ_DIAG_ELAPSED_TICKS = 2,
  KVAF_BATCH_PUT_CQ_DIAG_RAW_WORD = 3,
  KVAF_BATCH_PUT_CQ_DIAG_EXPECTED_OWNER = 4,
  KVAF_BATCH_PUT_CQ_DIAG_TIMEOUT_REASON = 5,
  KVAF_BATCH_PUT_CQ_DIAG_SQ_BEFORE = 6,
  KVAF_BATCH_PUT_CQ_DIAG_CQ_BEFORE = 7,
  KVAF_BATCH_PUT_CQ_DIAG_SQ_AFTER = 8,
  KVAF_BATCH_PUT_CQ_DIAG_CQ_AFTER = 9,
  KVAF_BATCH_PUT_CQ_DIAG_ASSIGNED_ROWS = 10,
  KVAF_BATCH_PUT_CQ_DIAG_FAST_BUDGET_EXCEEDED = 11,
  KVAF_BATCH_PUT_CQ_DIAG_OBSERVED_OWNER = 12,
  KVAF_BATCH_PUT_CQ_DIAG_SLOW_ELAPSED_TICKS = 13,
  KVAF_BATCH_PUT_CQ_TIMEOUT_NONE = 0,
  KVAF_BATCH_PUT_CQ_TIMEOUT_DEVICE_DEADLINE = 1,
  KVAF_BATCH_PUT_CQ_TIMEOUT_POLL_FALLBACK = 2,
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
static_assert(KVAF_BATCH_PUT_CQ_DIAGNOSTIC_BASE_WORD +
                  KVAF_AIV_BATCH_PUT_MAX_JETTYS_PER_CORE *
                      KVAF_BATCH_PUT_CQ_DIAGNOSTIC_WORDS_PER_JETTY <= 38,
              "BatchPut CQ diagnostics overflow reserved status storage");
#endif

#endif  // KVAF_AIV_BATCH_PUT_ABI_H_
