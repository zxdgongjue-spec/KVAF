#ifndef KVAF_SOURCE_MATCHED_UDMA_H
#define KVAF_SOURCE_MATCHED_UDMA_H

#include <stddef.h>
#include <stdint.h>

#include "kvaf_urma_provider_abi.h"
#include "urma_provider.h"
#include "urma_types.h"

/* Runtime evidence written into provider result reserved fields. */
#define KVAF_UDMA_TEARDOWN_RUNTIME_CAPABILITY \
  KVAF_UDMA_TEARDOWN_SOURCE_OWNED_CHECKED_RETRYABLE_V1

#ifdef __cplusplus
extern "C" {
#endif

static inline uint64_t kvaf_urma_public_abi_signature(void)
{
  const uint64_t fields[] = {
    sizeof(urma_device_t),
    offsetof(urma_device_t, ops),
    offsetof(urma_device_t, sysfs_dev),
    sizeof(urma_device_cap_t),
    offsetof(urma_device_cap_t, max_jfc_depth),
    offsetof(urma_device_cap_t, ceq_cnt),
    sizeof(urma_context_t),
    offsetof(urma_context_t, ops),
    offsetof(urma_context_t, dev_fd),
    offsetof(urma_context_t, eid),
    offsetof(urma_context_t, aggr_mode),
    sizeof(urma_ops_t),
    offsetof(urma_ops_t, alloc_jfc),
    offsetof(urma_ops_t, free_jfc),
    offsetof(urma_ops_t, alloc_jetty),
    offsetof(urma_ops_t, free_jetty),
    offsetof(urma_ops_t, get_dmac),
    sizeof(urma_jfc_t),
    offsetof(urma_jfc_t, jfc_cfg),
    offsetof(urma_jfc_t, handle),
    offsetof(urma_jfc_t, event_mutex),
    sizeof(urma_jetty_t),
    offsetof(urma_jetty_t, remote_jetty),
    offsetof(urma_jetty_t, jetty_cfg),
    offsetof(urma_jetty_t, handle),
    offsetof(urma_jetty_t, event_mutex),
  };
  uint64_t hash = UINT64_C(1469598103934665603);
  size_t i;

  for (i = 0; i < sizeof(fields) / sizeof(fields[0]); ++i) {
    hash ^= fields[i];
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

/*
 * These functions are private to libkvaf_urma_provider_{host,device}.so.
 * The source-built UDMA provider registers into KVAF's private liburma core;
 * that core is compiled with dynamic provider discovery disabled. KVAF
 * refuses to export queue addresses unless the resulting context is owned by
 * this exact provider instance.
 */
int kvaf_udma_prepare_provider_ops(void);
urma_provider_ops_t *kvaf_udma_source_provider_ops(void);
int kvaf_udma_register_source_provider(void);
void kvaf_udma_unregister_source_provider(void);
int kvaf_udma_source_provider_is_registered(void);
int kvaf_udma_source_runtime_init(void);
int kvaf_udma_source_runtime_uninit(void);
int kvaf_udma_source_runtime_is_owned(void);
/*
 * Process-lifetime fail-closed latch for a userspace UDMA mutation whose
 * completion/lock ownership can no longer be proved.  Ordinary ioctl and
 * munmap failures must not set this latch when their exact retry handle is
 * still retained.
 */
void kvaf_udma_source_mark_cleanup_unknown(void);
int kvaf_udma_source_cleanup_is_unknown(void);
int kvaf_udma_source_provider_owns_device(const urma_device_t *device);
int kvaf_udma_source_provider_owns_context(
    const urma_context_t *context);
uint64_t kvaf_udma_source_public_abi_signature(void);
int kvaf_udma_source_public_abi_matches(uint64_t caller_signature);

/* Number of retryable private objects still retained by the provider. */
uint32_t kvaf_udma_source_live_segment_count(void);
uint32_t kvaf_udma_source_live_token_count(void);
uint32_t kvaf_udma_source_live_context_count(void);
uint32_t kvaf_udma_source_live_jetty_count(void);
uint32_t kvaf_udma_source_live_jfr_count(void);
uint32_t kvaf_udma_source_live_jfc_count(void);

/*
 * Count sidecars owned by one exact public context.  Context deletion must
 * use these predicates rather than process-global counts: the direct data
 * context and the long-lived UB-management context intentionally coexist.
 * UINT32_MAX is returned for a NULL context or a lock failure so every
 * caller fails closed.
 */
uint32_t kvaf_udma_source_live_segment_count_for_context(
    const urma_context_t *context);
uint32_t kvaf_udma_source_live_token_count_for_context(
    const urma_context_t *context);
uint32_t kvaf_udma_source_live_jetty_count_for_context(
    const urma_context_t *context);
uint32_t kvaf_udma_source_live_jfr_count_for_context(
    const urma_context_t *context);
uint32_t kvaf_udma_source_live_jfc_count_for_context(
    const urma_context_t *context);
int kvaf_udma_source_context_has_live_resources(
    const urma_context_t *context);

/*
 * The public urma_register_seg() wrapper may allocate/free an implicit token
 * around the provider call.  Those cleanup calls are allowed to overwrite
 * errno, so errno observed by the management layer cannot identify whether
 * validation, UMMU grant, or the kernel register command failed.  Keep the
 * provider's exact per-thread result out of the public provider ABI.
 */
#define KVAF_UDMA_SEGMENT_FAILURE_ABI_VERSION UINT32_C(1)

enum kvaf_udma_segment_failure_stage {
    KVAF_UDMA_SEGMENT_FAILURE_NONE = 0,
    KVAF_UDMA_SEGMENT_FAILURE_VALIDATE,
    KVAF_UDMA_SEGMENT_FAILURE_ALLOCATE_OWNER,
    KVAF_UDMA_SEGMENT_FAILURE_LOCK_OWNER,
    KVAF_UDMA_SEGMENT_FAILURE_UMMU_GRANT,
    KVAF_UDMA_SEGMENT_FAILURE_KERNEL_REGISTER,
};

struct kvaf_udma_segment_failure {
    uint32_t abi_version;
    uint32_t structure_size;
    uint32_t stage;
    int32_t operation_status;
    int32_t operation_errno;
    int32_t rollback_status;
    int32_t rollback_errno;
    uint32_t ummu_tid;
    uint32_t token_id;
    uint64_t token_handle;
    uint64_t address;
    uint64_t length;
};

int kvaf_udma_source_get_segment_failure(
    struct kvaf_udma_segment_failure *failure);

/*
 * The public urma_create_jfr() API returns only a pointer.  The pinned UDMA
 * provider has several userspace allocation and kernel-command stages behind
 * that pointer, and its rollback path can overwrite errno.  Preserve the
 * first per-thread leaf so management Open can report the real failing stage
 * without exposing private UDMA layouts through the provider ABI.
 */
#define KVAF_UDMA_JFR_FAILURE_ABI_VERSION UINT32_C(2)

enum kvaf_udma_jfr_command_flag {
    KVAF_UDMA_JFR_COMMAND_RQ_ADDRESS_VALID = 1U << 0,
    KVAF_UDMA_JFR_COMMAND_RQ_HW_PAGE_ALIGNED = 1U << 1,
    KVAF_UDMA_JFR_COMMAND_RQ_RANGE_VALID = 1U << 2,
    KVAF_UDMA_JFR_COMMAND_INDEX_ADDRESS_VALID = 1U << 3,
    KVAF_UDMA_JFR_COMMAND_INDEX_HW_PAGE_ALIGNED = 1U << 4,
    KVAF_UDMA_JFR_COMMAND_INDEX_RANGE_VALID = 1U << 5,
    KVAF_UDMA_JFR_COMMAND_RECORD_DB_VALID = 1U << 6,
    KVAF_UDMA_JFR_COMMAND_SLEEP_DB_VALID = 1U << 7,
    KVAF_UDMA_JFR_COMMAND_JFC_HANDLE_VALID = 1U << 8,
    KVAF_UDMA_JFR_COMMAND_INPUT_SIZE_MATCH = 1U << 9,
    KVAF_UDMA_JFR_COMMAND_OUTPUT_SIZE_MATCH = 1U << 10,
    KVAF_UDMA_JFR_COMMAND_INPUT_ADDRESS_MATCH = 1U << 11,
    KVAF_UDMA_JFR_COMMAND_OUTPUT_ADDRESS_MATCH = 1U << 12,
    KVAF_UDMA_JFR_COMMAND_CONTEXT_FD_VALID = 1U << 13,
    KVAF_UDMA_JFR_COMMAND_JFC_CONTEXT_MATCH = 1U << 14,
    KVAF_UDMA_JFR_COMMAND_JFC_FD_MATCH = 1U << 15,
    KVAF_UDMA_JFR_COMMAND_JFC_REGISTRY_IDENTITY_MATCH = 1U << 16,
    KVAF_UDMA_JFR_COMMAND_NON_PIN = 1U << 17,
    KVAF_UDMA_JFR_COMMAND_DTU = 1U << 18,
    KVAF_UDMA_JFR_COMMAND_HUGEPAGE = 1U << 19,
    /* Full leaf log only; the compact result carries bits 0..19. */
    KVAF_UDMA_JFR_COMMAND_CONTEXT_SVA_SEPARATE = 1U << 20,
};

#define KVAF_UDMA_JFC_IDENTITY_ABI_VERSION UINT32_C(1)

enum kvaf_udma_jfc_identity_flag {
    KVAF_UDMA_JFC_IDENTITY_REGISTRY_FOUND = 1U << 0,
    KVAF_UDMA_JFC_IDENTITY_KERNEL_CREATED = 1U << 1,
    KVAF_UDMA_JFC_IDENTITY_DELETE_IDLE = 1U << 2,
    KVAF_UDMA_JFC_IDENTITY_NOT_ORPHAN = 1U << 3,
    KVAF_UDMA_JFC_IDENTITY_OWNER_CONTEXT_MATCH = 1U << 4,
    KVAF_UDMA_JFC_IDENTITY_PUBLIC_CONTEXT_MATCH = 1U << 5,
    KVAF_UDMA_JFC_IDENTITY_CREATED_CONTEXT_MATCH = 1U << 6,
    KVAF_UDMA_JFC_IDENTITY_HANDLE_MATCH = 1U << 7,
    KVAF_UDMA_JFC_IDENTITY_ID_MATCH = 1U << 8,
    KVAF_UDMA_JFC_IDENTITY_DEV_FD_MATCH = 1U << 9,
};

struct kvaf_udma_jfc_identity {
    uint32_t abi_version;
    uint32_t structure_size;
    uint32_t flags;
    uint32_t created_id;
    uint32_t current_id;
    int32_t created_dev_fd;
    int32_t current_dev_fd;
    uint64_t created_handle;
    uint64_t current_handle;
};

int kvaf_udma_source_snapshot_jfc_identity(
    const urma_context_t *expected_context, const urma_jfc_t *jfc,
    struct kvaf_udma_jfc_identity *identity);

enum kvaf_udma_jfr_failure_stage {
    KVAF_UDMA_JFR_FAILURE_NONE = 0,
    KVAF_UDMA_JFR_FAILURE_VALIDATE_ARGUMENTS,
    KVAF_UDMA_JFR_FAILURE_ALLOCATE_OWNER,
    KVAF_UDMA_JFR_FAILURE_LOCK_REGISTRY,
    KVAF_UDMA_JFR_FAILURE_VALIDATE_CONFIG,
    KVAF_UDMA_JFR_FAILURE_VALIDATE_CONTEXT,
    KVAF_UDMA_JFR_FAILURE_INIT_SPIN,
    KVAF_UDMA_JFR_FAILURE_ALLOCATE_INDEX,
    KVAF_UDMA_JFR_FAILURE_COMPUTE_RQ,
    KVAF_UDMA_JFR_FAILURE_ALLOCATE_RQ,
    KVAF_UDMA_JFR_FAILURE_ALLOCATE_RECORD_DB,
    KVAF_UDMA_JFR_FAILURE_ALLOCATE_SLEEP_DB,
    KVAF_UDMA_JFR_FAILURE_CREATE_COMMAND,
    KVAF_UDMA_JFR_FAILURE_INSERT_TABLE,
    KVAF_UDMA_JFR_FAILURE_UNLOCK_REGISTRY,
    KVAF_UDMA_JFR_FAILURE_ROLLBACK_KERNEL,
    KVAF_UDMA_JFR_FAILURE_ROLLBACK_LOCAL,
};

struct kvaf_udma_jfr_failure {
    uint32_t abi_version;
    uint32_t structure_size;
    uint32_t stage;
    int32_t operation_status;
    int32_t operation_errno;
    int32_t rollback_status;
    int32_t rollback_errno;
    uint32_t requested_depth;
    uint32_t requested_max_sge;
    uint32_t device_max_depth;
    uint32_t device_max_sge;
    uint32_t context_jfr_sge;
    uint32_t context_dtu_enable;
    uint32_t context_sq_reserved;
    uint32_t context_table_shift;
    uint32_t context_table_entries;
    uint32_t context_page_size;
    uint32_t context_sva_sep_mode;
    uint32_t command_size;
    uint32_t response_size;
    uint32_t udata_in_len;
    uint32_t udata_out_len;
    uint32_t command_flags;
    uint32_t jfc_id;
    int32_t context_dev_fd;
    int32_t context_fd_flags;
    int32_t context_fd_errno;
    int32_t jfc_context_dev_fd;
    int32_t jfc_registry_query_status;
    int32_t jfc_registry_query_errno;
    uint32_t jfc_registry_flags;
    uint32_t jfc_created_id;
    int32_t jfc_created_dev_fd;
    uint32_t requested_trans_mode;
    uint32_t requested_flags;
    uint32_t requested_min_rnr_timer;
    uint32_t requested_token;
    uint32_t requested_id;
    uint64_t jfc_handle;
    uint64_t jfc_created_handle;
    uint64_t udata_in_address;
    uint64_t udata_out_address;
    uint64_t rq_address;
    uint64_t rq_length;
    uint64_t index_address;
    uint64_t index_length;
    uint64_t record_db_address;
    uint64_t sleep_db_address;
};

int kvaf_udma_source_get_jfr_failure(
    struct kvaf_udma_jfr_failure *failure);
void kvaf_udma_source_reset_jfr_failure(void);

/*
 * Management JFRs retain the public URMA validation/refcount path.  The
 * Device provider supplies a one-call RQ allocation policy so its small
 * control-plane receive queue uses the pinned UDMA_MMAP_KERNEL_BUF path;
 * Host management and every ordinary/data JFR keep the provider's default
 * capability-driven hugepage policy.
 */
urma_jfr_t *kvaf_udma_source_create_management_jfr(
    urma_context_t *context, urma_jfr_cfg_t *config);

const char *kvaf_udma_source_management_jfr_rq_memory(void);

/*
 * A create operation can return NULL after the kernel or UMMU has already
 * acquired a resource.  Such an object is not visible to the caller and must
 * therefore be retried by its owning context before context destruction.
 * The drain only touches objects which were never handed to the caller.
 */
uint32_t kvaf_udma_source_orphan_segment_count(void);
uint32_t kvaf_udma_source_orphan_token_count(void);
uint32_t kvaf_udma_source_orphan_jetty_count(void);
uint32_t kvaf_udma_source_orphan_jfr_count(void);
uint32_t kvaf_udma_source_orphan_context_count(void);
int kvaf_udma_source_retry_orphan_segments(urma_context_t *context);
int kvaf_udma_source_retry_orphan_tokens(urma_context_t *context);
int kvaf_udma_source_retry_orphan_jetties(urma_context_t *context);
int kvaf_udma_source_retry_orphan_jfrs(urma_context_t *context);
int kvaf_udma_source_retry_orphan_jfcs(urma_context_t *context);
int kvaf_udma_source_retry_orphan_cleanup(urma_context_t *context);
int kvaf_udma_source_retry_orphan_contexts(void);

/* Installed into the private UDMA urma_ops table before registration. */
urma_jetty_t *kvaf_udma_create_jetty_checked(
    urma_context_t *context, urma_jetty_cfg_t *cfg);
urma_status_t kvaf_udma_delete_jetty_checked(urma_jetty_t *jetty);
urma_jfr_t *kvaf_udma_create_jfr_checked(
    urma_context_t *context, urma_jfr_cfg_t *cfg);
urma_status_t kvaf_udma_delete_jfr_checked(urma_jfr_t *jfr);
urma_jfc_t *kvaf_udma_create_jfc_checked(
    urma_context_t *context, urma_jfc_cfg_t *cfg);
urma_status_t kvaf_udma_delete_jfc_checked(urma_jfc_t *jfc);

#ifdef __cplusplus
}
#endif

#endif
