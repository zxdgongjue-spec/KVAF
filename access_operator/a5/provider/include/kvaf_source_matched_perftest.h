#ifndef KVAF_SOURCE_MATCHED_PERFTEST_H
#define KVAF_SOURCE_MATCHED_PERFTEST_H

#include <stdint.h>

#include "perftest_parameters.h"
#include "perftest_resources.h"

#ifdef __cplusplus
extern "C" {
#endif

enum kvaf_perftest_checked_status {
  KVAF_PERFTEST_CHECKED_OK = 0,
  // Runtime acquisition failed or the peer could not confirm it before any
  // URMA device/context/MR was created. Local runtime state may be released.
  KVAF_PERFTEST_CHECKED_PRE_RESOURCE_FAILED = -1,
  // Creation failed before a usable context was returned. No generic unwind
  // is attempted after a local MR may have been exposed to the peer; callers
  // must retain the process and backing memory until reset.
  KVAF_PERFTEST_CHECKED_CREATE_UNSAFE = -2,
  // A public URMA destroy operation retained its handle and can be retried
  // by this same provider instance for the same session.
  KVAF_PERFTEST_CHECKED_DESTROY_RETRYABLE = -3,
  // The peer teardown protocol became ambiguous. Retrying or releasing the
  // backing memory could race a peer that still owns an imported resource.
  KVAF_PERFTEST_CHECKED_DESTROY_UNSAFE = -4,
  // A matching peer phase frame proved that creation stopped before the next
  // protocol step. Every locally created handle remains in ctx for the
  // coordinated checked teardown state machine.
  KVAF_PERFTEST_CHECKED_CREATE_CLEANUP_REQUIRED = -5,
};

/*
 * Thread-local leaf failure captured by the source-matched UB management
 * backend.  Outer wrappers such as client_handshake must not replace an
 * already-recorded leaf stage.  Open snapshots the record before checked
 * Close can execute another management operation on the same thread.
 */
struct kvaf_perftest_management_diagnostic_v1 {
  uint32_t structure_size;
  uint32_t abi_version;
  uint32_t stage;
  uint32_t flags;
  int32_t raw_errno;
  int32_t effective_errno;
  int32_t operation_status;
  /*
   * The upper status word is provider-specific auxiliary evidence.  It is
   * not a hardware completion code unless the matching semantic flag says
   * so.  In particular CREATE_JFR stores the local provider failure-stage
   * id here.
   */
  int32_t auxiliary_status;
};

void kvaf_perftest_management_diagnostic_reset(void);
int kvaf_perftest_management_diagnostic_get(
    struct kvaf_perftest_management_diagnostic_v1 *diagnostic);

enum kvaf_perftest_teardown_phase {
  KVAF_PERFTEST_TEARDOWN_NOT_STARTED = 0,
  KVAF_PERFTEST_TEARDOWN_RUN_CONTEXT = 1,
  KVAF_PERFTEST_TEARDOWN_DISCONNECT_LOCAL = 2,
  KVAF_PERFTEST_TEARDOWN_DISCONNECT_PEER = 3,
  KVAF_PERFTEST_TEARDOWN_UNIMPORT_SEG_LOCAL = 4,
  KVAF_PERFTEST_TEARDOWN_UNIMPORT_SEG_PEER = 5,
  KVAF_PERFTEST_TEARDOWN_CONNECTION_METADATA = 6,
  KVAF_PERFTEST_TEARDOWN_LOCAL_SEGMENTS = 7,
  KVAF_PERFTEST_TEARDOWN_TOKEN_IDS = 8,
  KVAF_PERFTEST_TEARDOWN_LOCAL_BUFFER_VECTOR = 9,
  KVAF_PERFTEST_TEARDOWN_JETTYS = 10,
  KVAF_PERFTEST_TEARDOWN_JFRS = 11,
  KVAF_PERFTEST_TEARDOWN_RECV_JFCS = 12,
  KVAF_PERFTEST_TEARDOWN_SEND_JFCS = 13,
  KVAF_PERFTEST_TEARDOWN_CONTEXT = 14,
  KVAF_PERFTEST_TEARDOWN_COMPLETE = 15,
  KVAF_PERFTEST_TEARDOWN_UNSAFE = 16,
  // Drains source-owned objects whose create returned NULL after acquiring a
  // lower kernel/UMMU resource. This must precede public segment/token frees.
  KVAF_PERFTEST_TEARDOWN_ORPHAN_RESOURCES = 17,
};

enum kvaf_perftest_teardown_object {
  KVAF_PERFTEST_TEARDOWN_OBJECT_NONE = 0,
  KVAF_PERFTEST_TEARDOWN_OBJECT_CONTROL_PROTOCOL = 1,
  KVAF_PERFTEST_TEARDOWN_OBJECT_TARGET_JETTY = 2,
  KVAF_PERFTEST_TEARDOWN_OBJECT_TARGET_SEGMENT = 3,
  KVAF_PERFTEST_TEARDOWN_OBJECT_LOCAL_SEGMENT = 4,
  KVAF_PERFTEST_TEARDOWN_OBJECT_TOKEN_ID = 5,
  KVAF_PERFTEST_TEARDOWN_OBJECT_JETTY = 6,
  KVAF_PERFTEST_TEARDOWN_OBJECT_JFR = 7,
  KVAF_PERFTEST_TEARDOWN_OBJECT_RECV_JFC = 8,
  KVAF_PERFTEST_TEARDOWN_OBJECT_SEND_JFC = 9,
  KVAF_PERFTEST_TEARDOWN_OBJECT_CONTEXT = 10,
  KVAF_PERFTEST_TEARDOWN_OBJECT_ORPHAN_RESOURCE = 11,
};

// Persistent only inside the provider instance that created the session.
// Public URMA handles remain in perftest_context_t until their corresponding
// operation succeeds. phase/object/index describe the next incomplete work,
// while object_handle is the retained public URMA handle and
// disconnect_action_mask records sub-actions already completed for the
// current target Jetty. unsafe_from_phase preserves the last safe phase when
// peer protocol ambiguity moves phase to TEARDOWN_UNSAFE.
struct kvaf_perftest_teardown_state {
  uint32_t magic;
  uint32_t phase;
  uint32_t object;
  uint32_t object_index;
  uint32_t disconnect_action_mask;
  uint32_t attempts;
  int32_t last_status;
  uint32_t unsafe_from_phase;
  uint64_t object_handle;
};

// Arms one monotonic deadline for the upstream connect/accept helper. It must
// be called immediately before kvaf_perftest_establish_connection().
int kvaf_perftest_set_control_timeout(unsigned int timeout_seconds);

// Re-arm the same bounded interval for the coordinated teardown exchange.
// A decode epoch may outlive the bootstrap deadline, so Close must not reuse
// the already-expired Open deadline.
int kvaf_perftest_rearm_control_timeout(void);

// Both backends are source-matched to the pinned UMDK snapshot.  In UB mode
// expected_peer_eid is mandatory on both roles: the server compares it with
// the remote identity learned from the receive CQE, while the client compares
// it with the imported management target before any metadata is exchanged.
int kvaf_perftest_establish_connection(const perftest_config_t *cfg,
    const char *expected_peer_eid);

enum kvaf_perftest_management_close_status {
  KVAF_PERFTEST_MANAGEMENT_CLOSE_OK = 0,
  // Every failed public destroy retains the exact handle in the process-local
  // UB context. Calling Close again resumes at that handle.
  KVAF_PERFTEST_MANAGEMENT_CLOSE_RETRYABLE = -1,
  // The provider still owns at least one management resource, but the last
  // operation was either non-retryable or could not be disambiguated.  The
  // caller must retain the owner and quarantine the session; treating this
  // as EAGAIN would turn a permanent driver/API error into an infinite loop.
  KVAF_PERFTEST_MANAGEMENT_CLOSE_UNSAFE = -2,
};

int kvaf_perftest_close_connection(perftest_config_t *cfg);
int kvaf_perftest_sync_data(const perftest_config_t *cfg, uint32_t index,
    int size, char *local_data, char *remote_data);
int kvaf_perftest_control_fd(uint32_t index);

int kvaf_perftest_create_duplex_ctx_checked(
    perftest_context_t *ctx, perftest_config_t *cfg);

int kvaf_perftest_confirm_provider_ready(
    perftest_context_t *ctx, perftest_config_t *cfg, int local_status);

int kvaf_perftest_confirm_runtime_ready(
    perftest_config_t *cfg, int local_status);

void kvaf_perftest_teardown_state_init(
    struct kvaf_perftest_teardown_state *state);

int kvaf_perftest_destroy_duplex_ctx_checked(
    perftest_context_t *ctx, perftest_config_t *cfg,
    struct kvaf_perftest_teardown_state *state);

#ifdef __cplusplus
}
#endif

#endif
