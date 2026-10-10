#ifndef KVAF_AIV_BATCH_PUT_INTERNAL_H_
#define KVAF_AIV_BATCH_PUT_INTERNAL_H_

#include <stdint.h>

#include "batch_put_abi.h"

enum {
  KVAF_AIV_BATCH_PUT_QUEUE_STATE_BYTES = 256,
  KVAF_AIV_BATCH_PUT_DIAG_MAGIC = 0x44494133,
};

typedef struct KvafBatchPutQueueState {
  uint32_t magic;
  uint32_t initialized;
  uint64_t sq_producer_index;
  uint64_t cq_consumer_index;
  uint64_t completed_batch_count;
  uint64_t last_error;
  uint64_t reserved[3];
  // Private host/kernel ABI: rebuild both libraries after changing this state.
  uint32_t diag_magic;
  uint32_t diag_flags;  // bit 0: CQE captured; bit 1: matching outstanding SQE captured
  uint32_t diag_cqe_bytes;
  uint32_t diag_sq_slot;
  uint32_t diag_cqe[16];
  uint32_t diag_wqe[16];
  uint64_t diag_reserved[6];
} KvafBatchPutQueueState;

typedef struct KvafBatchPutDirectUrmaEntity {
  uint32_t magic;
  uint32_t version;
  uint64_t local_hbm_address;
  uint64_t local_hbm_size;
  uint64_t remote_host_ddr_address;
  uint64_t remote_host_ddr_size;
  uint32_t local_token_id;
  uint32_t local_token_value;
  uint32_t remote_token_id;
  uint32_t remote_token_value;
  uint64_t sq_va;
  uint64_t sq_db_va;
  uint32_t jetty_id;
  uint32_t sq_depth;
  uint32_t tp_id;
  uint32_t wqe_size;
  uint8_t remote_eid[16];
  uint64_t cq_va;
  uint64_t cq_db_va;
  uint32_t jfc_id;
  uint32_t cq_depth;
  uint32_t cqe_size;
  uint32_t reserved0;
  uint64_t reserved[7];
} KvafBatchPutDirectUrmaEntity;

// Private launch envelope. Runtime callers never construct or own these
// transport fields; KvafExecuteBatchPutSync injects them from the transport.
typedef struct KvafExecuteBatchPutLaunchArgs {
  KvafExecuteBatchPutKernelArgs request;
  uint64_t channel_entity;  // Entity[aiv_core_count * jettys_per_core]
  uint64_t queue_state;     // QueueState[aiv_core_count * jettys_per_core]
  uint32_t jettys_per_core;
  uint32_t reserved0;
} KvafExecuteBatchPutLaunchArgs;

#if defined(__cplusplus)
static_assert(sizeof(KvafBatchPutQueueState) == KVAF_AIV_BATCH_PUT_QUEUE_STATE_BYTES,
              "KvafBatchPutQueueState ABI changed");
static_assert(sizeof(KvafBatchPutDirectUrmaEntity) == 192,
              "KvafBatchPutDirectUrmaEntity ABI changed");
static_assert(sizeof(KvafExecuteBatchPutLaunchArgs) == 80,
              "KvafExecuteBatchPutLaunchArgs ABI changed");
#endif

#endif  // KVAF_AIV_BATCH_PUT_INTERNAL_H_
