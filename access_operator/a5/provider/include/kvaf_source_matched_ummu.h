#ifndef KVAF_SOURCE_MATCHED_UMMU_H
#define KVAF_SOURCE_MATCHED_UMMU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KVAF_SOURCE_UMMU_STATUS_ABI_VERSION UINT32_C(1)
#define KVAF_SOURCE_UMMU_CAPABILITY_MAGIC \
  UINT64_C(0x4b564146554d4d55) /* "KVAFUMMU" */
#define KVAF_SOURCE_UMMU_CHECKED_RETRYABLE_V1 \
  UINT64_C(0x0000000000010000)
#define KVAF_SOURCE_UMMU_PINNED_FULL_UAPI \
  UINT64_C(0x0000000000000001)

enum kvaf_source_ummu_failure_stage {
  KVAF_SOURCE_UMMU_FAILURE_NONE = 0,
  KVAF_SOURCE_UMMU_FAILURE_INIT,
  KVAF_SOURCE_UMMU_FAILURE_ALLOC_TID_IOCTL,
  KVAF_SOURCE_UMMU_FAILURE_MAPT_CREATE,
  KVAF_SOURCE_UMMU_FAILURE_MAP_INSERT,
  KVAF_SOURCE_UMMU_FAILURE_MAPT_UPDATE,
  KVAF_SOURCE_UMMU_FAILURE_MAP_DELETE,
  KVAF_SOURCE_UMMU_FAILURE_PLBI_QUEUE_FULL,
  KVAF_SOURCE_UMMU_FAILURE_PLBI_PSYNC_TIMEOUT,
  KVAF_SOURCE_UMMU_FAILURE_PLBI_COMPLETION,
  KVAF_SOURCE_UMMU_FAILURE_PLBI_KERNEL_IOCTL,
  KVAF_SOURCE_UMMU_FAILURE_FREE_TID_IOCTL,
  KVAF_SOURCE_UMMU_FAILURE_MUNMAP,
  KVAF_SOURCE_UMMU_FAILURE_LIVE_MAPPING,
  KVAF_SOURCE_UMMU_FAILURE_FORKED_PROCESS,
  KVAF_SOURCE_UMMU_FAILURE_INTERNAL
};

enum kvaf_source_ummu_retryability {
  KVAF_SOURCE_UMMU_RETRY_NOT_APPLICABLE = 0,
  KVAF_SOURCE_UMMU_RETRY_SAFE = 1,
  KVAF_SOURCE_UMMU_RETRY_UNSAFE = 2
};

enum kvaf_source_ummu_pending_operation {
  KVAF_SOURCE_UMMU_PENDING_NONE = 0,
  KVAF_SOURCE_UMMU_PENDING_GRANT_TOKEN_UPDATE,
  KVAF_SOURCE_UMMU_PENDING_UNGRANT,
  KVAF_SOURCE_UMMU_PENDING_UNGRANT_TOKEN_UPDATE,
  KVAF_SOURCE_UMMU_PENDING_FREE_TID
};

/*
 * Stable diagnostic contract for the private source-built UMMU instance.
 * A nonzero live/pending/retained count means the owning provider DSO must not
 * unload.  CHECKED_RETRYABLE_V1 guarantees that every successful core-buffer
 * mmap publishes a preallocated ownership record before it is returned.  A
 * void rollback therefore retains the exact address after munmap failure
 * without relying on a post-failure allocation.
 */
struct kvaf_source_ummu_status {
  uint32_t abi_version;
  uint32_t structure_size;
  uint64_t capability_magic;
  uint64_t capability_bits;
  uint32_t live_tid_count;
  uint32_t pending_operation_count;
  uint32_t retained_mapping_count;
  uint32_t last_failure_stage;
  int32_t last_error;
  uint32_t last_retryability;
  uint32_t last_state_unknown;
  uint32_t last_tid;
  uint32_t last_pending_operation;
  uint64_t last_address;
  uint64_t last_length;
};

uint64_t kvaf_source_ummu_capability(void);
int kvaf_source_ummu_get_status(struct kvaf_source_ummu_status *status);
uint32_t kvaf_source_ummu_live_tid_count(void);
uint32_t kvaf_source_ummu_pending_operation_count(void);
uint32_t kvaf_source_ummu_retained_mapping_count(void);

/*
 * Process-lifetime fail-closed latch.  Once a UMMU command or ownership
 * transition becomes unknowable, a later retry-safe diagnostic must never
 * downgrade Close back to RETRYABLE.
 */
int kvaf_source_ummu_cleanup_is_unknown(void);

/* Retry only work that is explicitly classified KVAF_SOURCE_UMMU_RETRY_SAFE. */
int kvaf_source_ummu_retry_pending(uint32_t tid);

/* True only when this source implementation, rather than libummu.so, owns it. */
int kvaf_source_ummu_checked_contract_ready(void);

/* Private C replacements for the two upstream C++ containers. */
int kvaf_source_ummu_seg_tree_is_empty(uint64_t seg_tree);

#ifdef __cplusplus
}
#endif

#endif
