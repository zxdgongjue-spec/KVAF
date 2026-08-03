#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <malloc.h>
#include <poll.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "urma_api.h"
#include "kvaf_source_matched_perftest.h"
#include "kvaf_source_matched_udma.h"
#include "kvaf_source_matched_ummu.h"
#include "kvaf_urma_provider_abi.h"
#include "kvaf_ub_close_policy.h"

static struct timespec g_kvaf_control_deadline;
static int g_kvaf_control_deadline_valid;
static unsigned int g_kvaf_control_timeout_seconds;

static _Thread_local struct kvaf_perftest_management_diagnostic_v1
    g_kvaf_management_diagnostic = {
        .structure_size = sizeof(g_kvaf_management_diagnostic),
        .abi_version = KVAF_URMA_MANAGEMENT_DIAGNOSTIC_VERSION,
    };

struct kvaf_ub_management_stage_entry {
    const char *name;
    uint32_t stage;
};

static uint32_t kvaf_ub_management_stage_from_name(const char *name)
{
    static const struct kvaf_ub_management_stage_entry stages[] = {
        {"validate_establish_arguments",
            KVAF_URMA_MANAGEMENT_STAGE_VALIDATE_ESTABLISH_ARGUMENTS},
        {"validate_arguments",
            KVAF_URMA_MANAGEMENT_STAGE_VALIDATE_ARGUMENTS},
        {"allocate_owner", KVAF_URMA_MANAGEMENT_STAGE_ALLOCATE_OWNER},
        {"resolve_device_eid",
            KVAF_URMA_MANAGEMENT_STAGE_RESOLVE_DEVICE_EID},
        {"validate_device_eid",
            KVAF_URMA_MANAGEMENT_STAGE_VALIDATE_DEVICE_EID},
        {"create_context", KVAF_URMA_MANAGEMENT_STAGE_CREATE_CONTEXT},
        {"create_jfc", KVAF_URMA_MANAGEMENT_STAGE_CREATE_JFC},
        {"allocate_send_buffer",
            KVAF_URMA_MANAGEMENT_STAGE_ALLOCATE_SEND_BUFFER},
        {"allocate_receive_buffer",
            KVAF_URMA_MANAGEMENT_STAGE_ALLOCATE_RECEIVE_BUFFER},
        {"allocate_handshake_buffer",
            KVAF_URMA_MANAGEMENT_STAGE_ALLOCATE_HANDSHAKE_BUFFER},
        {"allocate_send_token",
            KVAF_URMA_MANAGEMENT_STAGE_ALLOCATE_SEND_TOKEN},
        {"register_send_segment",
            KVAF_URMA_MANAGEMENT_STAGE_REGISTER_SEND_SEGMENT},
        {"allocate_receive_token",
            KVAF_URMA_MANAGEMENT_STAGE_ALLOCATE_RECEIVE_TOKEN},
        {"register_receive_segment",
            KVAF_URMA_MANAGEMENT_STAGE_REGISTER_RECEIVE_SEGMENT},
        {"allocate_handshake_token",
            KVAF_URMA_MANAGEMENT_STAGE_ALLOCATE_HANDSHAKE_TOKEN},
        {"register_handshake_segment",
            KVAF_URMA_MANAGEMENT_STAGE_REGISTER_HANDSHAKE_SEGMENT},
        {"create_jfr", KVAF_URMA_MANAGEMENT_STAGE_CREATE_JFR},
        {"create_jetty", KVAF_URMA_MANAGEMENT_STAGE_CREATE_JETTY},
        {"validate_server_jetty_id",
            KVAF_URMA_MANAGEMENT_STAGE_VALIDATE_SERVER_JETTY_ID},
        {"server_handshake",
            KVAF_URMA_MANAGEMENT_STAGE_SERVER_HANDSHAKE},
        {"client_parse_peer_eid",
            KVAF_URMA_MANAGEMENT_STAGE_CLIENT_PARSE_PEER_EID},
        {"client_get_tp_list",
            KVAF_URMA_MANAGEMENT_STAGE_CLIENT_GET_TP_LIST},
        {"client_validate_tp_list",
            KVAF_URMA_MANAGEMENT_STAGE_CLIENT_VALIDATE_TP_LIST},
        {"client_import_peer_jetty",
            KVAF_URMA_MANAGEMENT_STAGE_CLIENT_IMPORT_PEER_JETTY_EX},
        {"client_import_peer_jetty_ex",
            KVAF_URMA_MANAGEMENT_STAGE_CLIENT_IMPORT_PEER_JETTY_EX},
        {"client_post_handshake_send",
            KVAF_URMA_MANAGEMENT_STAGE_CLIENT_POST_HANDSHAKE_SEND},
        {"client_poll_handshake_cqe",
            KVAF_URMA_MANAGEMENT_STAGE_CLIENT_POLL_HANDSHAKE_CQE},
        {"client_handshake",
            KVAF_URMA_MANAGEMENT_STAGE_CLIENT_HANDSHAKE},
        {"validate_peer_eid",
            KVAF_URMA_MANAGEMENT_STAGE_VALIDATE_PEER_EID},
        {"post_client_receive",
            KVAF_URMA_MANAGEMENT_STAGE_POST_CLIENT_RECEIVE},
        {"server_validate_peer_ctp",
            KVAF_URMA_MANAGEMENT_STAGE_SERVER_VALIDATE_PEER_CTP},
        {"server_get_tp_list",
            KVAF_URMA_MANAGEMENT_STAGE_SERVER_GET_TP_LIST},
        {"server_validate_tp_list",
            KVAF_URMA_MANAGEMENT_STAGE_SERVER_VALIDATE_TP_LIST},
        {"server_import_peer_jetty_ex",
            KVAF_URMA_MANAGEMENT_STAGE_SERVER_IMPORT_PEER_JETTY_EX},
        {"client_validate_peer_ctp",
            KVAF_URMA_MANAGEMENT_STAGE_CLIENT_VALIDATE_PEER_CTP},
    };
    size_t index;

    if (name == NULL) {
        return KVAF_URMA_MANAGEMENT_STAGE_NONE;
    }
    for (index = 0; index < sizeof(stages) / sizeof(stages[0]); ++index) {
        if (strcmp(name, stages[index].name) == 0) {
            return stages[index].stage;
        }
    }
    return KVAF_URMA_MANAGEMENT_STAGE_NONE;
}

static void kvaf_ub_record_management_diagnostic(
    const char *stage, int raw_errno, int effective_errno,
    int operation_status, bool operation_status_valid,
    int completion_status, bool completion_status_valid)
{
    uint32_t stage_id;

    /* Keep the first concrete leaf.  Outer handshake wrappers and checked
     * teardown run later on the same thread and must not hide the cause. */
    if (g_kvaf_management_diagnostic.stage !=
        KVAF_URMA_MANAGEMENT_STAGE_NONE) {
        return;
    }
    stage_id = kvaf_ub_management_stage_from_name(stage);
    if (stage_id == KVAF_URMA_MANAGEMENT_STAGE_NONE) {
        return;
    }
    g_kvaf_management_diagnostic.stage = stage_id;
    g_kvaf_management_diagnostic.raw_errno = raw_errno;
    g_kvaf_management_diagnostic.effective_errno = effective_errno;
    g_kvaf_management_diagnostic.operation_status = operation_status;
    g_kvaf_management_diagnostic.completion_status = completion_status;
    g_kvaf_management_diagnostic.flags =
        KVAF_URMA_MANAGEMENT_DIAGNOSTIC_RAW_ERRNO_VALID |
        KVAF_URMA_MANAGEMENT_DIAGNOSTIC_EFFECTIVE_ERRNO_VALID;
    if (operation_status_valid) {
        g_kvaf_management_diagnostic.flags |=
            KVAF_URMA_MANAGEMENT_DIAGNOSTIC_OPERATION_STATUS_VALID;
    }
    if (completion_status_valid) {
        g_kvaf_management_diagnostic.flags |=
            KVAF_URMA_MANAGEMENT_DIAGNOSTIC_COMPLETION_STATUS_VALID;
    }
}

void kvaf_perftest_management_diagnostic_reset(void)
{
    (void)memset(&g_kvaf_management_diagnostic, 0,
        sizeof(g_kvaf_management_diagnostic));
    g_kvaf_management_diagnostic.structure_size =
        sizeof(g_kvaf_management_diagnostic);
    g_kvaf_management_diagnostic.abi_version =
        KVAF_URMA_MANAGEMENT_DIAGNOSTIC_VERSION;
}

int kvaf_perftest_management_diagnostic_get(
    struct kvaf_perftest_management_diagnostic_v1 *diagnostic)
{
    if (diagnostic == NULL ||
        diagnostic->structure_size != sizeof(*diagnostic) ||
        diagnostic->abi_version !=
            KVAF_URMA_MANAGEMENT_DIAGNOSTIC_VERSION ||
        g_kvaf_management_diagnostic.stage ==
            KVAF_URMA_MANAGEMENT_STAGE_NONE) {
        return -1;
    }
    *diagnostic = g_kvaf_management_diagnostic;
    return 0;
}

int kvaf_perftest_set_control_timeout(unsigned int timeout_seconds)
{
    if (timeout_seconds == 0 ||
        clock_gettime(CLOCK_MONOTONIC, &g_kvaf_control_deadline) != 0) {
        return -1;
    }
    g_kvaf_control_deadline.tv_sec += (time_t)timeout_seconds;
    g_kvaf_control_deadline_valid = 1;
    g_kvaf_control_timeout_seconds = timeout_seconds;
    return 0;
}

int kvaf_perftest_rearm_control_timeout(void)
{
    if (g_kvaf_control_timeout_seconds == 0) {
        errno = EINVAL;
        return -1;
    }
    return kvaf_perftest_set_control_timeout(
        g_kvaf_control_timeout_seconds);
}

static int kvaf_deadline_remaining_ms(void)
{
    struct timespec now;
    int64_t seconds;
    int64_t nanoseconds;
    int64_t milliseconds;

    if (!g_kvaf_control_deadline_valid ||
        clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        errno = ETIMEDOUT;
        return -1;
    }
    seconds = (int64_t)g_kvaf_control_deadline.tv_sec -
        (int64_t)now.tv_sec;
    nanoseconds = (int64_t)g_kvaf_control_deadline.tv_nsec -
        (int64_t)now.tv_nsec;
    milliseconds = seconds * 1000 + nanoseconds / 1000000;
    if (nanoseconds > 0 && nanoseconds % 1000000 != 0) {
        ++milliseconds;
    }
    if (milliseconds <= 0) {
        errno = ETIMEDOUT;
        return -1;
    }
    return milliseconds > INT_MAX ? INT_MAX : (int)milliseconds;
}

static int kvaf_wait_socket(int fd, short events)
{
    struct pollfd descriptor = {
        .fd = fd,
        .events = events,
        .revents = 0,
    };

    for (;;) {
        int timeout_ms = kvaf_deadline_remaining_ms();
        int result;

        if (timeout_ms < 0) {
            return -1;
        }
        result = poll(&descriptor, 1, timeout_ms);
        if (result > 0) {
            return 0;
        }
        if (result == 0) {
            errno = ETIMEDOUT;
            return -1;
        }
        if (errno != EINTR) {
            return -1;
        }
    }
}

static int kvaf_deadline_connect(
    int fd, const struct sockaddr *address, socklen_t address_length)
{
    int original_flags = fcntl(fd, F_GETFL, 0);
    int result;
    int socket_error = 0;
    socklen_t socket_error_bytes = sizeof(socket_error);

    /* Upstream retries ECONNREFUSED on the same socket.  Reject every retry
     * after the shared monotonic deadline, including failures that do not
     * enter poll(), so its fixed retry loop cannot extend the contract. */
    if (kvaf_deadline_remaining_ms() < 0 || original_flags < 0 ||
        fcntl(fd, F_SETFL, original_flags | O_NONBLOCK) != 0) {
        return -1;
    }
    result = connect(fd, address, address_length);
    if (result != 0 && errno == EINPROGRESS &&
        kvaf_wait_socket(fd, POLLOUT) == 0 &&
        getsockopt(fd, SOL_SOCKET, SO_ERROR, &socket_error,
            &socket_error_bytes) == 0) {
        if (socket_error == 0) {
            result = 0;
        } else {
            errno = socket_error;
            result = -1;
        }
    }
    if (fcntl(fd, F_SETFL, original_flags) != 0) {
        return -1;
    }
    return result;
}

static int kvaf_deadline_usleep(useconds_t requested_microseconds)
{
    const int remaining_milliseconds = kvaf_deadline_remaining_ms();
    uint64_t remaining_microseconds;
    useconds_t bounded_microseconds = requested_microseconds;

    if (remaining_milliseconds < 0) {
        return -1;
    }
    remaining_microseconds = (uint64_t)remaining_milliseconds * 1000U;
    if ((uint64_t)bounded_microseconds > remaining_microseconds) {
        bounded_microseconds = (useconds_t)remaining_microseconds;
    }
    return usleep(bounded_microseconds);
}

static int kvaf_deadline_accept(
    int fd, struct sockaddr *address, socklen_t *address_length)
{
    int original_flags = fcntl(fd, F_GETFL, 0);
    int accepted = -1;

    if (original_flags < 0 ||
        fcntl(fd, F_SETFL, original_flags | O_NONBLOCK) != 0) {
        return -1;
    }
    for (;;) {
        accepted = accept(fd, address, address_length);
        if (accepted >= 0) {
            break;
        }
        if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK) {
            break;
        }
        if (kvaf_wait_socket(fd, POLLIN) != 0) {
            break;
        }
    }
    if (fcntl(fd, F_SETFL, original_flags) != 0) {
        if (accepted >= 0) {
            (void)close(accepted);
        }
        return -1;
    }
    return accepted;
}

#include "perftest_mgmt_tcp.h"
#include "perftest_mgmt_ub.h"

#ifndef KVAF_PERFTEST_MGMT_TCP_SOURCE
#error "KVAF_PERFTEST_MGMT_TCP_SOURCE must name perftest_mgmt_tcp.c"
#endif
/*
 * UMDK 26.06 already uses a complete MSG_NOSIGNAL send loop.  Only replace
 * connect/accept so its process-global TCP context observes KVAF's one-shot
 * monotonic deadline as well.
 */
#define connect kvaf_deadline_connect
#define accept kvaf_deadline_accept
#define usleep kvaf_deadline_usleep
#include KVAF_PERFTEST_MGMT_TCP_SOURCE
#undef usleep
#undef accept
#undef connect

#ifndef KVAF_PERFTEST_MGMT_UB_SOURCE
#error "KVAF_PERFTEST_MGMT_UB_SOURCE must name perftest_mgmt_ub.c"
#endif

/*
 * Keep UMDK's official UB management implementation in the same translation
 * unit so KVAF can retain its private context across checked Close retries.
 * The provider has already acquired the source-linked URMA runtime before it
 * enters this backend; suppress the upstream helper's second urma_init call.
 */
static volatile sig_atomic_t g_kvaf_ub_exit_flag;
static bool g_kvaf_ub_server_arming;
static uint32_t g_kvaf_ub_server_recv_posts;
static uint32_t g_kvaf_ub_server_jetty_id;
static char g_kvaf_ub_server_local_eid[64];
/* Every management SEND requests a CQE.  Exact accounting lets failed-Open
 * teardown distinguish the server's recv-only handshake timeout from a
 * client whose SQ still owns the registered send buffer.  A failed post with
 * no usable bad-WR boundary is deliberately treated as ambiguous until a
 * hardware flush completion proves that the whole SQ is quiescent. */
static struct kvaf_ub_send_tracker g_kvaf_ub_send_tracker;
#define KVAF_UB_MANAGEMENT_SEND_WR_SCAN_LIMIT 1024U

enum kvaf_ub_management_close_phase {
    KVAF_UB_CLOSE_RETRY_ORPHANS = 0,
    KVAF_UB_CLOSE_QUIESCE_JETTY,
    KVAF_UB_CLOSE_QUIESCE_JFR,
    KVAF_UB_CLOSE_DRAIN_JFC,
    KVAF_UB_CLOSE_UNIMPORT_TJETTY,
    KVAF_UB_CLOSE_DELETE_JETTY,
    KVAF_UB_CLOSE_DELETE_JFR,
    KVAF_UB_CLOSE_UNREGISTER_HANDSHAKE,
    KVAF_UB_CLOSE_FREE_HANDSHAKE_TOKEN,
    KVAF_UB_CLOSE_UNREGISTER_RECEIVE,
    KVAF_UB_CLOSE_FREE_RECEIVE_TOKEN,
    KVAF_UB_CLOSE_UNREGISTER_SEND,
    KVAF_UB_CLOSE_FREE_SEND_TOKEN,
    KVAF_UB_CLOSE_DELETE_JFC,
    KVAF_UB_CLOSE_DELETE_CONTEXT,
    KVAF_UB_CLOSE_RELEASE_LOCAL,
};

/* One provider instance owns at most one UMDK management context.  Keep the
 * teardown cursor beside that context so a same-process Close retry never
 * repeats an already successful state transition or destroy operation. */
static enum kvaf_ub_management_close_phase g_kvaf_ub_close_phase =
    KVAF_UB_CLOSE_RETRY_ORPHANS;
static uint64_t g_kvaf_ub_close_attempt;

static urma_status_t kvaf_ub_reuse_runtime_init(
    urma_init_attr_t *attribute)
{
    (void)attribute;
    return URMA_EEXIST;
}

static void kvaf_ub_account_management_completions(
    int count, const urma_cr_t *completions)
{
    int index;

    if (count <= 0 || completions == NULL) {
        return;
    }
    for (index = 0; index < count; ++index) {
        const urma_cr_t *completion = &completions[index];

        if (completion->flag.bs.s_r != 0U) {
            continue;
        }
        /* The pinned UDMA provider does not initialize every CR flag bit.
         * Match official perftest and use the status value as the only
         * full-SQ flush proof. */
        if (completion->status == URMA_CR_WR_FLUSH_ERR_DONE) {
            kvaf_ub_send_tracker_record_completion(
                &g_kvaf_ub_send_tracker, true);
            continue;
        }
        kvaf_ub_send_tracker_record_completion(
            &g_kvaf_ub_send_tracker, false);
    }
}

/* All handshake and metadata CQ polling consumes the single monotonic Open
 * deadline.  The upstream loops remain source-identical, but can no longer
 * spin forever if a peer disappears. */
static int kvaf_deadline_urma_poll_jfc(
    urma_jfc_t *jfc, int cr_cnt, urma_cr_t *cr)
{
    int count;

    if (kvaf_deadline_remaining_ms() < 0) {
        return -1;
    }
    count = urma_poll_jfc(jfc, cr_cnt, cr);
    kvaf_ub_account_management_completions(count, cr);
    return count;
}

static urma_status_t kvaf_tracking_urma_post_jetty_send_wr(
    urma_jetty_t *jetty, urma_jfs_wr_t *wr, urma_jfs_wr_t **bad_wr)
{
    urma_status_t status = urma_post_jetty_send_wr(jetty, wr, bad_wr);
    urma_jfs_wr_t *stop = NULL;
    urma_jfs_wr_t *current;
    uint32_t submitted_count = 0;
    uint32_t index;

    if (status != URMA_SUCCESS) {
        if (bad_wr == NULL || *bad_wr == NULL) {
            kvaf_ub_send_tracker_mark_post_unknown(
                &g_kvaf_ub_send_tracker);
            return status;
        }
        /* URMA reports the first WR that was not posted.  Earlier list nodes
         * remain real in-flight submissions and must be counted. */
        stop = *bad_wr;
    }
    for (current = wr;
         current != NULL && current != stop &&
             submitted_count < KVAF_UB_MANAGEMENT_SEND_WR_SCAN_LIMIT;
         current = current->next) {
        ++submitted_count;
    }
    if (current != stop) {
        /* A provider-owned pointer outside the submitted chain, a cycle or
         * an unexpectedly large list cannot define a trustworthy partial
         * post boundary.  A later flush barrier is the only safe proof. */
        kvaf_ub_send_tracker_mark_post_unknown(
            &g_kvaf_ub_send_tracker);
        return status;
    }
    current = wr;
    for (index = 0; index < submitted_count; ++index) {
        kvaf_ub_send_tracker_record_post(&g_kvaf_ub_send_tracker,
            current->flag.bs.complete_enable != 0U);
        current = current->next;
    }
    return status;
}

/* The server is connectable only after both FIFO receive WQEs (handshake and
 * first metadata frame) reached the hardware RQ.  Publish a line-buffer
 * independent readiness record at exactly that point. */
static urma_status_t kvaf_ready_urma_post_jetty_recv_wr(
    urma_jetty_t *jetty, urma_jfr_wr_t *wr, urma_jfr_wr_t **bad_wr)
{
    const urma_status_t status =
        urma_post_jetty_recv_wr(jetty, wr, bad_wr);

    if (status == URMA_SUCCESS && g_kvaf_ub_server_arming &&
        ++g_kvaf_ub_server_recv_posts == 2U) {
        (void)fprintf(stdout,
            "KVAF_URMA_MGMT_READY role=server jetty_id=%u "
            "local_eid=%s\n",
            g_kvaf_ub_server_jetty_id,
            g_kvaf_ub_server_local_eid);
        (void)fflush(stdout);
    }
    return status;
}

static int kvaf_ub_management_error(
    const char *stage, int operation_errno, int fallback_errno);

/*
 * The pinned UDMA provider exposes import_jetty_ex rather than the legacy
 * import_jetty operation.  The generic liburma compatibility path supplies
 * an all-zero active-TP configuration, which is not sufficient for UDMA
 * CTP.  Keep the upstream management wire protocol, but replace every peer
 * Jetty import in it (server and client) with the checked public-API path
 * implemented below.
 */
static urma_target_jetty_t *kvaf_ub_import_peer_jetty_tp_aware(
    urma_context_t *ctx, urma_rjetty_t *rjetty,
    urma_token_t *token_value);

#define g_exit_flag g_kvaf_ub_exit_flag
#define urma_init kvaf_ub_reuse_runtime_init
#define urma_poll_jfc kvaf_deadline_urma_poll_jfc
#define urma_post_jetty_send_wr kvaf_tracking_urma_post_jetty_send_wr
#define urma_post_jetty_recv_wr kvaf_ready_urma_post_jetty_recv_wr
#define urma_import_jetty kvaf_ub_import_peer_jetty_tp_aware
#define ub_establish_connection kvaf_upstream_ub_establish_connection
#define ub_close_connection kvaf_upstream_ub_close_connection
#define ub_client_handshake kvaf_upstream_ub_client_handshake
#define ub_sync_data kvaf_upstream_ub_sync_data
#define ub_sync_time kvaf_upstream_ub_sync_time
#define ub_comm_send kvaf_upstream_ub_comm_send
#define ub_comm_recv kvaf_upstream_ub_comm_recv
#define ub_comm_poll kvaf_upstream_ub_comm_poll
#include KVAF_PERFTEST_MGMT_UB_SOURCE
#undef ub_client_handshake
#undef ub_comm_poll
#undef ub_comm_recv
#undef ub_comm_send
#undef ub_sync_time
#undef ub_sync_data
#undef ub_close_connection
#undef ub_establish_connection
#undef urma_import_jetty
#undef urma_post_jetty_recv_wr
#undef urma_post_jetty_send_wr
#undef urma_poll_jfc
#undef urma_init
#undef g_exit_flag

/*
 * Upstream ub_create_local_resources() performs best-effort rollback and can
 * discard a public handle after a failed destroy.  KVAF builds the identical
 * resource graph into an owning context first.  On any partial-create error
 * that context is installed in g_ub_ctx and the ordinary checked Close path
 * retains/retries every successfully returned public handle.
 */
static int kvaf_ub_management_error(
    const char *stage, int operation_errno, int fallback_errno)
{
    const int concrete_errno =
        operation_errno == 0 ? fallback_errno : operation_errno;

    kvaf_ub_record_management_diagnostic(
        stage, operation_errno, concrete_errno, 0, false, 0, false);

    /* source_linked_urma_main deliberately reports ENOENT while suppressing
     * dynamic provider scans. Reset errno before every checked operation and
     * name the exact failing stage so that such an internal errno can never
     * be mistaken for a missing endpoint, context, queue or Jetty. */
    (void)fprintf(stderr,
        "KVAF_URMA_MGMT_ERROR stage=%s raw_errno=%d "
        "fallback_errno=%d effective_errno=%d\n",
        stage, operation_errno, fallback_errno, concrete_errno);
    (void)fflush(stderr);
    errno = concrete_errno;
    return -1;
}

static urma_target_jetty_t *kvaf_ub_import_peer_jetty_tp_aware(
    urma_context_t *ctx, urma_rjetty_t *rjetty,
    urma_token_t *token_value)
{
    urma_get_tp_cfg_t get_tp_cfg = {0};
    urma_tp_info_t tp_info = {0};
    urma_import_jetty_ex_cfg_t active_tp_cfg = {0};
    urma_target_jetty_t *tjetty;
    urma_status_t status;
    uint32_t tp_count = 1;
    int operation_errno;
    const bool server = g_kvaf_ub_server_arming;
    const char *get_stage = server ?
        "server_get_tp_list" : "client_get_tp_list";
    const char *validate_stage = server ?
        "server_validate_tp_list" : "client_validate_tp_list";
    const char *import_stage = server ?
        "server_import_peer_jetty_ex" :
        "client_import_peer_jetty_ex";

    if (ctx == NULL || rjetty == NULL || token_value == NULL ||
        rjetty->trans_mode != URMA_TM_RM ||
        rjetty->tp_type != URMA_CTP) {
        (void)kvaf_ub_management_error(
            server ? "server_validate_peer_ctp" :
                     "client_validate_peer_ctp",
            EINVAL, EINVAL);
        return NULL;
    }

    /* Match the known-working public perftest compatibility sequence:
     * resolve exactly one RM+CTP transport path for local/peer EIDs, carry
     * its handle and a local initial PSN into import_jetty_ex.  CTP does not
     * use the RTP peer-handle/PSN exchange step. */
    get_tp_cfg.flag.bs.ctp = 1;
    get_tp_cfg.trans_mode = URMA_TM_RM;
    get_tp_cfg.local_eid = ctx->eid;
    get_tp_cfg.peer_eid = rjetty->jetty_id.eid;
    errno = 0;
    status = urma_get_tp_list(
        ctx, &get_tp_cfg, &tp_count, &tp_info);
    operation_errno = errno;
    if (status != URMA_SUCCESS || tp_count != 1U) {
        const char *failure_stage = status == URMA_SUCCESS ?
            validate_stage : get_stage;
        const int effective_errno = operation_errno == 0 ?
            (status == URMA_SUCCESS ? EPROTO : EIO) : operation_errno;

        (void)fprintf(stderr,
            "KVAF_URMA_MGMT_TP_ERROR stage=%s urma_status=%d "
            "tp_count=%u raw_errno=%d\n",
            failure_stage, (int)status, tp_count, operation_errno);
        (void)fflush(stderr);
        kvaf_ub_record_management_diagnostic(
            failure_stage, operation_errno, effective_errno,
            (int)status, true, (int)tp_count, true);
        (void)kvaf_ub_management_error(
            failure_stage, operation_errno,
            status == URMA_SUCCESS ? EPROTO : EIO);
        return NULL;
    }

    active_tp_cfg.tp_handle = tp_info.tp_handle;
    active_tp_cfg.tp_attr.tx_psn = (uint32_t)rand();
    errno = 0;
    tjetty = urma_import_jetty_ex(
        ctx, rjetty, token_value, &active_tp_cfg);
    operation_errno = errno;
    if (tjetty == NULL) {
        (void)fprintf(stderr,
            "KVAF_URMA_MGMT_TP_ERROR stage=%s tp_handle=%llu "
            "tx_psn=%u raw_errno=%d\n",
            import_stage,
            (unsigned long long)active_tp_cfg.tp_handle,
            active_tp_cfg.tp_attr.tx_psn, operation_errno);
        (void)fflush(stderr);
        (void)kvaf_ub_management_error(
            import_stage, operation_errno, EIO);
        return NULL;
    }
    return tjetty;
}

#define KVAF_UB_CLOSE_DRAIN_BATCH 1
#define KVAF_UB_CLOSE_DRAIN_PASSES 8

static const char *kvaf_ub_close_phase_name(
    enum kvaf_ub_management_close_phase phase)
{
    switch (phase) {
        case KVAF_UB_CLOSE_RETRY_ORPHANS:
            return "retry_orphan_cleanup";
        case KVAF_UB_CLOSE_QUIESCE_JETTY:
            return "quiesce_jetty";
        case KVAF_UB_CLOSE_QUIESCE_JFR:
            return "quiesce_jfr";
        case KVAF_UB_CLOSE_DRAIN_JFC:
            return "drain_jfc";
        case KVAF_UB_CLOSE_UNIMPORT_TJETTY:
            return "unimport_tjetty";
        case KVAF_UB_CLOSE_DELETE_JETTY:
            return "delete_jetty";
        case KVAF_UB_CLOSE_DELETE_JFR:
            return "delete_jfr";
        case KVAF_UB_CLOSE_UNREGISTER_HANDSHAKE:
            return "unregister_handshake_segment";
        case KVAF_UB_CLOSE_FREE_HANDSHAKE_TOKEN:
            return "free_handshake_token";
        case KVAF_UB_CLOSE_UNREGISTER_RECEIVE:
            return "unregister_receive_segment";
        case KVAF_UB_CLOSE_FREE_RECEIVE_TOKEN:
            return "free_receive_token";
        case KVAF_UB_CLOSE_UNREGISTER_SEND:
            return "unregister_send_segment";
        case KVAF_UB_CLOSE_FREE_SEND_TOKEN:
            return "free_send_token";
        case KVAF_UB_CLOSE_DELETE_JFC:
            return "delete_jfc";
        case KVAF_UB_CLOSE_DELETE_CONTEXT:
            return "delete_context";
        case KVAF_UB_CLOSE_RELEASE_LOCAL:
            return "release_local";
        default:
            return "unknown";
    }
}

static void kvaf_ub_reset_close_cursor(void)
{
    g_kvaf_ub_close_phase = KVAF_UB_CLOSE_RETRY_ORPHANS;
    g_kvaf_ub_close_attempt = 0;
}

static bool kvaf_ub_should_log_close_attempt(uint64_t attempt)
{
    return attempt <= 3U || (attempt & (attempt - 1U)) == 0U ||
        attempt % UINT64_C(256) == 0U;
}

static int kvaf_ub_close_operation_failed(
    enum kvaf_ub_management_close_phase phase,
    int operation_status, int operation_errno,
    int query_status, int query_errno, bool state_ambiguous,
    enum kvaf_ub_close_retry_class retry_class,
    bool explicit_retry_proven)
{
    const bool timed_out = operation_status == URMA_ETIMEOUT ||
        operation_errno == ETIMEDOUT || query_status == URMA_ETIMEOUT ||
        query_errno == ETIMEDOUT;
    const bool retryable = kvaf_ub_close_retry_is_proven(
        operation_status, operation_errno, retry_class,
        explicit_retry_proven, state_ambiguous, timed_out);

    if (kvaf_ub_should_log_close_attempt(g_kvaf_ub_close_attempt) ||
        !retryable) {
        (void)fprintf(stderr,
            "KVAF_URMA_MGMT_CLOSE_ERROR stage=%s status=%d errno=%d "
            "query_status=%d query_errno=%d attempt=%llu retryable=%u "
            "state_ambiguous=%u send_inflight=%u "
            "send_accounting_unknown=%u flush_barrier_required=%u "
            "flush_barrier_seen=%u\n",
            kvaf_ub_close_phase_name(phase), operation_status,
            operation_errno, query_status, query_errno,
            (unsigned long long)g_kvaf_ub_close_attempt,
            retryable ? 1U : 0U, state_ambiguous ? 1U : 0U,
            g_kvaf_ub_send_tracker.inflight,
            g_kvaf_ub_send_tracker.accounting_unknown ? 1U : 0U,
            g_kvaf_ub_send_tracker.flush_barrier_required ? 1U : 0U,
            g_kvaf_ub_send_tracker.flush_barrier_seen ? 1U : 0U);
        (void)fflush(stderr);
    }
    if (state_ambiguous || operation_status == URMA_ETIMEOUT ||
        operation_errno == ETIMEDOUT || query_status == URMA_ETIMEOUT ||
        query_errno == ETIMEDOUT) {
        kvaf_udma_source_mark_cleanup_unknown();
    }
    errno = operation_errno == 0 ? EIO : operation_errno;
    return retryable
        ? KVAF_PERFTEST_MANAGEMENT_CLOSE_RETRYABLE
        : KVAF_PERFTEST_MANAGEMENT_CLOSE_UNSAFE;
}

static bool kvaf_ub_close_phase_uses_ummu(
    enum kvaf_ub_management_close_phase phase)
{
    return phase == KVAF_UB_CLOSE_UNREGISTER_HANDSHAKE ||
        phase == KVAF_UB_CLOSE_FREE_HANDSHAKE_TOKEN ||
        phase == KVAF_UB_CLOSE_UNREGISTER_RECEIVE ||
        phase == KVAF_UB_CLOSE_FREE_RECEIVE_TOKEN ||
        phase == KVAF_UB_CLOSE_UNREGISTER_SEND ||
        phase == KVAF_UB_CLOSE_FREE_SEND_TOKEN;
}

static int kvaf_ub_destroy_operation_failed(
    enum kvaf_ub_management_close_phase phase,
    int operation_status, int operation_errno)
{
    struct kvaf_source_ummu_status ummu = {
        .structure_size = sizeof(ummu),
    };
    int diagnostic_status = URMA_SUCCESS;
    int diagnostic_errno = 0;
    bool state_ambiguous = false;
    bool retry_proven = false;

    if (kvaf_ub_close_phase_uses_ummu(phase)) {
        diagnostic_status = kvaf_source_ummu_get_thread_status(&ummu);
        diagnostic_errno = ummu.last_error == INT32_MIN ? EIO :
            (ummu.last_error < 0
                ? -ummu.last_error : ummu.last_error);
        if (diagnostic_status != 0 || ummu.last_state_unknown != 0U ||
            ummu.last_retryability == KVAF_SOURCE_UMMU_RETRY_UNSAFE) {
            state_ambiguous = true;
        } else if (ummu.last_retryability ==
                   KVAF_SOURCE_UMMU_RETRY_SAFE) {
            /* The source-matched UMMU layer retained the exact TID/token
             * operation and explicitly classified replay as safe. */
            retry_proven = true;
        }
        if (kvaf_ub_should_log_close_attempt(g_kvaf_ub_close_attempt) ||
            state_ambiguous) {
            (void)fprintf(stderr,
                "KVAF_URMA_MGMT_CLOSE_UMMU stage=%s failure_stage=%u "
                "tid=%u pending_operation=%u live_tids=%u pending=%u "
                "retained=%u address=0x%llx length=%llu\n",
                kvaf_ub_close_phase_name(phase),
                ummu.last_failure_stage, ummu.last_tid,
                ummu.last_pending_operation, ummu.live_tid_count,
                ummu.pending_operation_count, ummu.retained_mapping_count,
                (unsigned long long)ummu.last_address,
                (unsigned long long)ummu.last_length);
            (void)fflush(stderr);
        }
    }
    return kvaf_ub_close_operation_failed(
        phase, operation_status, operation_errno,
        diagnostic_status, diagnostic_errno, state_ambiguous,
        KVAF_UB_CLOSE_RETRY_CHECKED_DESTROY, retry_proven);
}

static void kvaf_ub_close_phase_completed(
    enum kvaf_ub_management_close_phase completed,
    enum kvaf_ub_management_close_phase next)
{
    (void)fprintf(stderr,
        "KVAF_URMA_MGMT_CLOSE_PROGRESS stage=%s attempt=%llu\n",
        kvaf_ub_close_phase_name(completed),
        (unsigned long long)g_kvaf_ub_close_attempt);
    (void)fflush(stderr);
    g_kvaf_ub_close_phase = next;
}

static int kvaf_ub_quiesce_management_jetty(ub_mgmt_ctx_t *ctx)
{
    urma_jetty_attr_t requested = {
        .mask = JETTY_STATE,
        .state = URMA_JETTY_STATE_ERROR,
    };
    urma_jetty_cfg_t observed_cfg = {0};
    urma_jetty_attr_t observed_attr = {
        .mask = JETTY_STATE,
    };
    int operation_errno;
    int query_errno;
    urma_status_t status;
    urma_status_t query_status;

    if (ctx->pair.jetty == NULL) {
        return KVAF_PERFTEST_MANAGEMENT_CLOSE_OK;
    }
    errno = 0;
    status = urma_modify_jetty(ctx->pair.jetty, &requested);
    operation_errno = errno;
    if (status == URMA_SUCCESS) {
        return KVAF_PERFTEST_MANAGEMENT_CLOSE_OK;
    }

    /* A modify completion can be lost after hardware already entered ERROR.
     * Query before deciding whether the transition is retryable. */
    errno = 0;
    query_status = urma_query_jetty(
        ctx->pair.jetty, &observed_cfg, &observed_attr);
    query_errno = errno;
    if (query_status == URMA_SUCCESS &&
        observed_attr.state == URMA_JETTY_STATE_ERROR) {
        return KVAF_PERFTEST_MANAGEMENT_CLOSE_OK;
    }
    return kvaf_ub_close_operation_failed(
        KVAF_UB_CLOSE_QUIESCE_JETTY, status, operation_errno,
        query_status, query_errno, query_status != URMA_SUCCESS,
        KVAF_UB_CLOSE_RETRY_CONTROL, false);
}

static int kvaf_ub_quiesce_management_jfr(ub_mgmt_ctx_t *ctx)
{
    urma_jfr_attr_t requested = {
        .mask = JFR_STATE,
        .state = URMA_JFR_STATE_ERROR,
    };
    urma_jfr_cfg_t observed_cfg = {0};
    urma_jfr_attr_t observed_attr = {
        .mask = JFR_STATE,
    };
    int operation_errno;
    int query_errno;
    urma_status_t status;
    urma_status_t query_status;

    if (ctx->jfr == NULL) {
        return KVAF_PERFTEST_MANAGEMENT_CLOSE_OK;
    }
    errno = 0;
    status = urma_modify_jfr(ctx->jfr, &requested);
    operation_errno = errno;
    if (status == URMA_SUCCESS) {
        return KVAF_PERFTEST_MANAGEMENT_CLOSE_OK;
    }

    errno = 0;
    query_status = urma_query_jfr(
        ctx->jfr, &observed_cfg, &observed_attr);
    query_errno = errno;
    if (query_status == URMA_SUCCESS &&
        observed_attr.state == URMA_JFR_STATE_ERROR) {
        return KVAF_PERFTEST_MANAGEMENT_CLOSE_OK;
    }
    return kvaf_ub_close_operation_failed(
        KVAF_UB_CLOSE_QUIESCE_JFR, status, operation_errno,
        query_status, query_errno, query_status != URMA_SUCCESS,
        KVAF_UB_CLOSE_RETRY_CONTROL, false);
}

static int kvaf_ub_drain_management_jfc(ub_mgmt_ctx_t *ctx)
{
    urma_cr_t completions[KVAF_UB_CLOSE_DRAIN_BATCH] = {{0}};
    int pass;

    if (ctx->jfc == NULL) {
        return KVAF_PERFTEST_MANAGEMENT_CLOSE_OK;
    }
    for (pass = 0; pass < KVAF_UB_CLOSE_DRAIN_PASSES; ++pass) {
        int count;
        int operation_errno;

        errno = 0;
        /* Pinned UDMA may consume an earlier CQE and still return a negative
         * batch result when a later CQE is malformed.  Poll one CR at a time
         * so a SEND completion or WR_FLUSH_ERR_DONE barrier can never be
         * consumed without reaching the ownership tracker. */
        count = urma_poll_jfc(ctx->jfc,
            KVAF_UB_CLOSE_DRAIN_BATCH, completions);
        operation_errno = errno;
        if (count == 0) {
            if (kvaf_ub_send_tracker_close_is_quiesced(
                    &g_kvaf_ub_send_tracker)) {
                return KVAF_PERFTEST_MANAGEMENT_CLOSE_OK;
            }
            return kvaf_ub_close_operation_failed(
                KVAF_UB_CLOSE_DRAIN_JFC, URMA_EAGAIN, EAGAIN,
                URMA_SUCCESS, 0, false,
                KVAF_UB_CLOSE_RETRY_CONTROL, false);
        }
        if (count < 0) {
            return kvaf_ub_close_operation_failed(
                KVAF_UB_CLOSE_DRAIN_JFC, count, operation_errno,
                URMA_SUCCESS, 0, false,
                KVAF_UB_CLOSE_RETRY_CONTROL, false);
        }
        kvaf_ub_account_management_completions(count, completions);
    }
    errno = EAGAIN;
    return kvaf_ub_close_operation_failed(
        KVAF_UB_CLOSE_DRAIN_JFC, URMA_EAGAIN, EAGAIN,
        URMA_SUCCESS, 0, false,
        KVAF_UB_CLOSE_RETRY_CONTROL, false);
}

#undef KVAF_UB_CLOSE_DRAIN_PASSES
#undef KVAF_UB_CLOSE_DRAIN_BATCH

static const char *kvaf_segment_failure_stage_name(uint32_t stage)
{
    switch (stage) {
        case KVAF_UDMA_SEGMENT_FAILURE_NONE:
            return "none";
        case KVAF_UDMA_SEGMENT_FAILURE_VALIDATE:
            return "validate";
        case KVAF_UDMA_SEGMENT_FAILURE_ALLOCATE_OWNER:
            return "allocate_owner";
        case KVAF_UDMA_SEGMENT_FAILURE_LOCK_OWNER:
            return "lock_owner";
        case KVAF_UDMA_SEGMENT_FAILURE_UMMU_GRANT:
            return "ummu_grant";
        case KVAF_UDMA_SEGMENT_FAILURE_KERNEL_REGISTER:
            return "kernel_register";
        default:
            return "unknown";
    }
}

static const char *kvaf_jfr_failure_stage_name(uint32_t stage)
{
    switch (stage) {
        case KVAF_UDMA_JFR_FAILURE_NONE:
            return "none";
        case KVAF_UDMA_JFR_FAILURE_VALIDATE_ARGUMENTS:
            return "validate_arguments";
        case KVAF_UDMA_JFR_FAILURE_ALLOCATE_OWNER:
            return "allocate_owner";
        case KVAF_UDMA_JFR_FAILURE_LOCK_REGISTRY:
            return "lock_registry";
        case KVAF_UDMA_JFR_FAILURE_VALIDATE_CONFIG:
            return "validate_config";
        case KVAF_UDMA_JFR_FAILURE_VALIDATE_CONTEXT:
            return "validate_context";
        case KVAF_UDMA_JFR_FAILURE_INIT_SPIN:
            return "init_spin";
        case KVAF_UDMA_JFR_FAILURE_ALLOCATE_INDEX:
            return "allocate_index";
        case KVAF_UDMA_JFR_FAILURE_COMPUTE_RQ:
            return "compute_rq";
        case KVAF_UDMA_JFR_FAILURE_ALLOCATE_RQ:
            return "allocate_rq";
        case KVAF_UDMA_JFR_FAILURE_ALLOCATE_RECORD_DB:
            return "allocate_record_db";
        case KVAF_UDMA_JFR_FAILURE_ALLOCATE_SLEEP_DB:
            return "allocate_sleep_db";
        case KVAF_UDMA_JFR_FAILURE_CREATE_COMMAND:
            return "create_command";
        case KVAF_UDMA_JFR_FAILURE_INSERT_TABLE:
            return "insert_table";
        case KVAF_UDMA_JFR_FAILURE_UNLOCK_REGISTRY:
            return "unlock_registry";
        case KVAF_UDMA_JFR_FAILURE_ROLLBACK_KERNEL:
            return "rollback_kernel";
        case KVAF_UDMA_JFR_FAILURE_ROLLBACK_LOCAL:
            return "rollback_local";
        default:
            return "unknown";
    }
}

static int kvaf_ub_jfr_creation_error(
    const char *stage, int wrapper_errno, int fallback_errno)
{
    struct kvaf_udma_jfr_failure failure = {
        .structure_size = sizeof(failure),
    };
    const int query_status =
        kvaf_udma_source_get_jfr_failure(&failure);
    const bool failure_valid = query_status == 0 &&
        failure.abi_version == KVAF_UDMA_JFR_FAILURE_ABI_VERSION &&
        failure.structure_size == sizeof(failure) &&
        failure.stage != KVAF_UDMA_JFR_FAILURE_NONE;
    int effective_errno = wrapper_errno;

    if (failure_valid && failure.operation_errno != 0) {
        effective_errno = failure.operation_errno;
    }
    if (effective_errno == 0) {
        effective_errno = fallback_errno;
    }
    (void)fprintf(stderr,
        "KVAF_URMA_JFR_ERROR management_stage=%s "
        "provider_stage=%s provider_stage_id=%u query_status=%d "
        "operation_status=%d operation_errno=%d "
        "rollback_status=%d rollback_errno=%d "
        "requested_depth=%u requested_max_sge=%u "
        "device_max_depth=%u device_max_sge=%u "
        "context_jfr_sge=%u context_dtu_enable=%u "
        "context_sq_reserved=%u context_table_shift=%u "
        "context_table_entries=%u context_page_size=%u "
        "context_sva_sep_mode=%u command_size=%u response_size=%u "
        "udata_in_len=%u udata_out_len=%u "
        "udata_in_address=0x%" PRIx64 " "
        "udata_out_address=0x%" PRIx64 " command_flags=0x%x "
        "jfc_id=%u jfc_handle=%" PRIu64 " context_dev_fd=%d "
        "context_fd_flags=%d context_fd_errno=%d "
        "jfc_context_dev_fd=%d jfc_registry_query_status=%d "
        "jfc_registry_query_errno=%d jfc_registry_flags=0x%x "
        "jfc_created_id=%u jfc_created_dev_fd=%d "
        "jfc_created_handle=%" PRIu64 " "
        "requested_trans_mode=%u requested_flags=0x%x "
        "requested_min_rnr_timer=%u requested_token=%u requested_id=%u "
        "rq_address=0x%" PRIx64 " rq_length=%" PRIu64 " "
        "index_address=0x%" PRIx64 " index_length=%" PRIu64 " "
        "record_db_address=0x%" PRIx64 " "
        "sleep_db_address=0x%" PRIx64 "\n",
        stage, kvaf_jfr_failure_stage_name(failure.stage),
        failure.stage, query_status, failure.operation_status,
        failure.operation_errno, failure.rollback_status,
        failure.rollback_errno, failure.requested_depth,
        failure.requested_max_sge, failure.device_max_depth,
        failure.device_max_sge, failure.context_jfr_sge,
        failure.context_dtu_enable, failure.context_sq_reserved,
        failure.context_table_shift, failure.context_table_entries,
        failure.context_page_size, failure.context_sva_sep_mode,
        failure.command_size, failure.response_size,
        failure.udata_in_len, failure.udata_out_len,
        failure.udata_in_address, failure.udata_out_address,
        failure.command_flags, failure.jfc_id, failure.jfc_handle,
        failure.context_dev_fd, failure.context_fd_flags,
        failure.context_fd_errno, failure.jfc_context_dev_fd,
        failure.jfc_registry_query_status,
        failure.jfc_registry_query_errno,
        failure.jfc_registry_flags, failure.jfc_created_id,
        failure.jfc_created_dev_fd, failure.jfc_created_handle,
        failure.requested_trans_mode,
        failure.requested_flags, failure.requested_min_rnr_timer,
        failure.requested_token, failure.requested_id,
        failure.rq_address, failure.rq_length,
        failure.index_address, failure.index_length,
        failure.record_db_address, failure.sleep_db_address);
    (void)fflush(stderr);
    kvaf_ub_record_management_diagnostic(
        stage, wrapper_errno, effective_errno,
        failure.operation_status, failure_valid,
        (int)failure.stage, failure_valid);
    return kvaf_ub_management_error(
        stage, effective_errno, fallback_errno);
}

static int kvaf_ub_segment_registration_error(
    const char *stage, int wrapper_errno, int fallback_errno)
{
    struct kvaf_udma_segment_failure segment = {
        .structure_size = sizeof(segment),
    };
    struct kvaf_source_ummu_status ummu = {
        .structure_size = sizeof(ummu),
    };
    const int segment_status =
        kvaf_udma_source_get_segment_failure(&segment);
    const int ummu_status =
        kvaf_source_ummu_get_thread_status(&ummu);
    int effective_errno = wrapper_errno;

    if (segment_status == 0 && segment.stage !=
        KVAF_UDMA_SEGMENT_FAILURE_NONE) {
        effective_errno = segment.operation_errno;
    }
    (void)fprintf(stderr,
        "KVAF_URMA_SEGMENT_ERROR management_stage=%s "
        "provider_stage=%s operation_status=%d operation_errno=%d "
        "rollback_status=%d rollback_errno=%d ummu_tid=%u "
        "token_id=%u token_handle=%llu address=0x%llx length=%llu "
        "ummu_query_status=%d ummu_failure_stage=%u "
        "ummu_error=%d ummu_retryability=%u ummu_state_unknown=%u "
        "ummu_live_tids=%u ummu_pending=%u ummu_retained=%u\n",
        stage, kvaf_segment_failure_stage_name(segment.stage),
        segment.operation_status, segment.operation_errno,
        segment.rollback_status, segment.rollback_errno,
        segment.ummu_tid, segment.token_id,
        (unsigned long long)segment.token_handle,
        (unsigned long long)segment.address,
        (unsigned long long)segment.length,
        ummu_status, ummu.last_failure_stage, ummu.last_error,
        ummu.last_retryability, ummu.last_state_unknown,
        ummu.live_tid_count, ummu.pending_operation_count,
        ummu.retained_mapping_count);
    (void)fflush(stderr);
    kvaf_ub_record_management_diagnostic(
        stage, wrapper_errno,
        effective_errno == 0 ? fallback_errno : effective_errno,
        segment.operation_status,
        segment_status == 0 &&
            segment.stage != KVAF_UDMA_SEGMENT_FAILURE_NONE,
        segment.rollback_status,
        segment_status == 0 && segment.rollback_status != 0);
    return kvaf_ub_management_error(
        stage, effective_errno, fallback_errno);
}

enum kvaf_ub_management_segment_index {
    KVAF_UB_MANAGEMENT_SEND = 0,
    KVAF_UB_MANAGEMENT_RECEIVE,
    KVAF_UB_MANAGEMENT_HANDSHAKE,
    KVAF_UB_MANAGEMENT_SEGMENT_COUNT,
};

/*
 * Do not use the public Segment-registration implicit-token convenience path.
 * The public unregister wrapper frees an implicit token even when the
 * provider's checked segment teardown returns failure.  Explicit ownership
 * lets Close retry the Segment first and free its Token/TID only afterwards.
 */
static urma_token_id_t *g_kvaf_ub_management_tokens[
    KVAF_UB_MANAGEMENT_SEGMENT_COUNT];

static int kvaf_ub_token_allocation_error(
    const char *stage, int wrapper_errno, int fallback_errno)
{
    struct kvaf_source_ummu_status ummu = {
        .structure_size = sizeof(ummu),
    };
    const int ummu_status =
        kvaf_source_ummu_get_thread_status(&ummu);
    int effective_errno = wrapper_errno;

    if (effective_errno == 0 && ummu_status == 0 &&
        ummu.last_error < 0) {
        effective_errno = -ummu.last_error;
    }
    (void)fprintf(stderr,
        "KVAF_URMA_TOKEN_ERROR management_stage=%s wrapper_errno=%d "
        "ummu_query_status=%d ummu_failure_stage=%u ummu_error=%d "
        "ummu_retryability=%u ummu_state_unknown=%u ummu_live_tids=%u "
        "ummu_pending=%u ummu_retained=%u\n",
        stage, wrapper_errno, ummu_status, ummu.last_failure_stage,
        ummu.last_error, ummu.last_retryability,
        ummu.last_state_unknown, ummu.live_tid_count,
        ummu.pending_operation_count, ummu.retained_mapping_count);
    (void)fflush(stderr);
    kvaf_ub_record_management_diagnostic(
        stage, wrapper_errno,
        effective_errno == 0 ? fallback_errno : effective_errno,
        ummu_status, ummu_status != 0, ummu.last_error,
        ummu_status == 0 && ummu.last_error != 0);
    return kvaf_ub_management_error(
        stage, effective_errno, fallback_errno);
}

static int kvaf_ub_create_local_resources_checked(
    const comm_ub_cfg_t *cfg, const char *expected_device_name,
    uint32_t expected_eid_index, ub_mgmt_ctx_t **ctx_out)
{
    urma_jfc_cfg_t jfc_cfg = {0};
    urma_seg_cfg_t seg_cfg = {0};
    urma_jfs_cfg_t jfs_cfg = {0};
    urma_jfr_cfg_t jfr_cfg = {0};
    urma_jetty_cfg_t jetty_cfg = {0};
    ub_mgmt_ctx_t *ctx;
    uint32_t eid_idx = 0;

    if (cfg == NULL || expected_device_name == NULL ||
        expected_device_name[0] == '\0' || ctx_out == NULL) {
        return kvaf_ub_management_error(
            "validate_arguments", EINVAL, EINVAL);
    }
    *ctx_out = NULL;
    errno = 0;
    ctx = (ub_mgmt_ctx_t *)calloc(1, sizeof(*ctx));
    if (ctx == NULL) {
        return kvaf_ub_management_error(
            "allocate_owner", errno, ENOMEM);
    }
    *ctx_out = ctx;

    errno = 0;
    if (ub_resolve_dev_and_eid_idx(
            cfg->src_eid, &ctx->dev, &eid_idx) != 0) {
        return kvaf_ub_management_error(
            "resolve_device_eid", errno, ENODEV);
    }
    /* Management and data resources are one endpoint lifetime. Silently
     * resolving the management EID onto a different UDMA device/index would
     * create a second, unproven ownership domain and break teardown proofs. */
    if (strcmp(ctx->dev->name, expected_device_name) != 0 ||
        eid_idx != expected_eid_index) {
        return kvaf_ub_management_error(
            "validate_device_eid", EACCES, EACCES);
    }
    errno = 0;
    ctx->urma_ctx = urma_create_context(ctx->dev, eid_idx);
    if (ctx->urma_ctx == NULL) {
        return kvaf_ub_management_error(
            "create_context", errno, EIO);
    }

    jfc_cfg.depth = UB_MGMT_JFC_DEPTH;
    errno = 0;
    ctx->jfc = urma_create_jfc(ctx->urma_ctx, &jfc_cfg);
    if (ctx->jfc == NULL) {
        return kvaf_ub_management_error(
            "create_jfc", errno, EIO);
    }

    errno = 0;
    ctx->send_buf = (char *)memalign(
        UB_MGMT_SEG_ALIGN, UB_MGMT_MSG_MAX_SIZE);
    if (ctx->send_buf == NULL) {
        return kvaf_ub_management_error(
            "allocate_send_buffer", errno, ENOMEM);
    }
    errno = 0;
    ctx->recv_buf = (char *)memalign(
        UB_MGMT_SEG_ALIGN, UB_MGMT_MSG_MAX_SIZE);
    if (ctx->recv_buf == NULL) {
        return kvaf_ub_management_error(
            "allocate_receive_buffer", errno, ENOMEM);
    }
    errno = 0;
    ctx->recv_buf_hs = (char *)memalign(
        UB_MGMT_SEG_ALIGN, UB_MGMT_MSG_MAX_SIZE);
    if (ctx->recv_buf_hs == NULL) {
        return kvaf_ub_management_error(
            "allocate_handshake_buffer", errno, ENOMEM);
    }
    (void)memset(ctx->send_buf, 0, UB_MGMT_MSG_MAX_SIZE);
    (void)memset(ctx->recv_buf, 0, UB_MGMT_MSG_MAX_SIZE);
    (void)memset(ctx->recv_buf_hs, 0, UB_MGMT_MSG_MAX_SIZE);

    seg_cfg.len = UB_MGMT_MSG_MAX_SIZE;
    seg_cfg.token_id = NULL;
    seg_cfg.token_value = g_ub_mgmt_token;
    seg_cfg.flag.value = 0;
    seg_cfg.flag.bs.access = URMA_ACCESS_LOCAL_ONLY;
    seg_cfg.flag.bs.token_id_valid = URMA_TOKEN_ID_VALID;
    seg_cfg.user_ctx = (uint64_t)NULL;
    seg_cfg.iova = 0;

    errno = 0;
    g_kvaf_ub_management_tokens[KVAF_UB_MANAGEMENT_SEND] =
        urma_alloc_token_id(ctx->urma_ctx);
    if (g_kvaf_ub_management_tokens[KVAF_UB_MANAGEMENT_SEND] == NULL) {
        return kvaf_ub_token_allocation_error(
            "allocate_send_token", errno, EIO);
    }
    seg_cfg.token_id =
        g_kvaf_ub_management_tokens[KVAF_UB_MANAGEMENT_SEND];
    seg_cfg.va = (uint64_t)ctx->send_buf;
    errno = 0;
    ctx->tseg_send = urma_register_seg(ctx->urma_ctx, &seg_cfg);
    if (ctx->tseg_send == NULL) {
        return kvaf_ub_segment_registration_error(
            "register_send_segment", errno, EIO);
    }
    errno = 0;
    g_kvaf_ub_management_tokens[KVAF_UB_MANAGEMENT_RECEIVE] =
        urma_alloc_token_id(ctx->urma_ctx);
    if (g_kvaf_ub_management_tokens[KVAF_UB_MANAGEMENT_RECEIVE] == NULL) {
        return kvaf_ub_token_allocation_error(
            "allocate_receive_token", errno, EIO);
    }
    seg_cfg.token_id =
        g_kvaf_ub_management_tokens[KVAF_UB_MANAGEMENT_RECEIVE];
    seg_cfg.va = (uint64_t)ctx->recv_buf;
    errno = 0;
    ctx->tseg_recv = urma_register_seg(ctx->urma_ctx, &seg_cfg);
    if (ctx->tseg_recv == NULL) {
        return kvaf_ub_segment_registration_error(
            "register_receive_segment", errno, EIO);
    }
    errno = 0;
    g_kvaf_ub_management_tokens[KVAF_UB_MANAGEMENT_HANDSHAKE] =
        urma_alloc_token_id(ctx->urma_ctx);
    if (g_kvaf_ub_management_tokens[
            KVAF_UB_MANAGEMENT_HANDSHAKE] == NULL) {
        return kvaf_ub_token_allocation_error(
            "allocate_handshake_token", errno, EIO);
    }
    seg_cfg.token_id =
        g_kvaf_ub_management_tokens[KVAF_UB_MANAGEMENT_HANDSHAKE];
    seg_cfg.va = (uint64_t)ctx->recv_buf_hs;
    errno = 0;
    ctx->tseg_recv_hs = urma_register_seg(ctx->urma_ctx, &seg_cfg);
    if (ctx->tseg_recv_hs == NULL) {
        return kvaf_ub_segment_registration_error(
            "register_handshake_segment", errno, EIO);
    }

    jfs_cfg.depth = UB_MGMT_JFS_DEPTH;
    /* A management SEND or route error must suspend this SQ.  Continuing
     * after a failed WQE can advance queue ownership beyond the checked
     * teardown cursor and turn a recoverable Open failure into reset-only
     * hardware state. */
    jfs_cfg.flag.bs.error_suspend = 1;
    jfs_cfg.trans_mode = URMA_TM_RM;
    jfs_cfg.max_sge = 1;
    jfs_cfg.max_rsge = 1;
    jfs_cfg.rnr_retry = URMA_TYPICAL_RNR_RETRY;
    jfs_cfg.err_timeout = URMA_TYPICAL_ERR_TIMEOUT;
    jfs_cfg.jfc = ctx->jfc;

    jfr_cfg.depth = UB_MGMT_JFR_DEPTH;
    jfr_cfg.trans_mode = URMA_TM_RM;
    jfr_cfg.max_sge = 1;
    jfr_cfg.min_rnr_timer = URMA_TYPICAL_MIN_RNR_TIMER;
    jfr_cfg.jfc = ctx->jfc;
    jfr_cfg.token_value = g_ub_mgmt_token;
    /* The public URMA wrapper can reject a request before dispatching into
     * the checked UDMA provider.  Reset first so that such a failure cannot
     * be paired with a previous create attempt's thread-local leaf. */
    kvaf_udma_source_reset_jfr_failure();
    errno = 0;
    ctx->jfr = kvaf_udma_source_create_management_jfr(
        ctx->urma_ctx, &jfr_cfg);
    if (ctx->jfr == NULL) {
        return kvaf_ub_jfr_creation_error(
            "create_jfr", errno, EIO);
    }
    (void)fprintf(stderr,
        "KVAF_URMA_MGMT_JFR_RQ_MEMORY role=%s backing=%s\n",
        cfg->dst_eid == NULL ? "server" : "client",
        kvaf_udma_source_management_jfr_rq_memory());
    (void)fflush(stderr);

    ctx->is_server = cfg->dst_eid == NULL;
    jetty_cfg.id = ctx->is_server ? cfg->dst_jetty_id : 0;
    jetty_cfg.flag.bs.share_jfr = URMA_SHARE_JFR;
    jetty_cfg.jfs_cfg = jfs_cfg;
    jetty_cfg.shared.jfr = ctx->jfr;
    jetty_cfg.shared.jfc = ctx->jfc;
    errno = 0;
    ctx->pair.jetty = urma_create_jetty(ctx->urma_ctx, &jetty_cfg);
    if (ctx->pair.jetty == NULL) {
        return kvaf_ub_management_error(
            "create_jetty", errno, EIO);
    }
    if (ctx->is_server &&
        ctx->pair.jetty->jetty_id.id != cfg->dst_jetty_id) {
        (void)fprintf(stderr,
            "KVAF_URMA_MGMT_JETTY_ID_MISMATCH "
            "requested=%u actual=%u\n",
            cfg->dst_jetty_id, ctx->pair.jetty->jetty_id.id);
        (void)fflush(stderr);
        return kvaf_ub_management_error(
            "validate_server_jetty_id", EPROTO, EPROTO);
    }
    kvaf_ub_send_tracker_require_flush_barrier(
        &g_kvaf_ub_send_tracker);
    return 0;
}

static bool kvaf_ub_peer_matches(const char *expected_peer_eid)
{
    urma_eid_t expected = {0};

    return expected_peer_eid != NULL &&
        urma_str_to_eid(expected_peer_eid, &expected) == 0 &&
        g_ub_ctx != NULL && g_ub_ctx->pair.tjetty != NULL &&
        memcmp(&expected, &g_ub_ctx->pair.tjetty->id.eid,
            sizeof(expected)) == 0;
}

/* The upstream UB helper collapses every client-handshake failure to -1 and
 * commonly leaves errno unset.  Keep its wire protocol source-identical while
 * naming the exact failing operation so AICPU Open evidence can distinguish a
 * route/import failure from a SEND or CQ failure. */
static int kvaf_ub_client_handshake_checked(
    ub_mgmt_ctx_t *ctx, const char *peer_eid_str, uint32_t peer_jetty_id)
{
    urma_eid_t peer_eid = {0};
    urma_rjetty_t rjetty = {0};
    urma_sge_t sge = {0};
    urma_jfs_wr_t wr = {0};
    urma_jfs_wr_t *bad = NULL;
    urma_cr_t cr = {0};
    urma_status_t post_status;
    int poll_status;
    int operation_errno;

    errno = 0;
    if (urma_str_to_eid(peer_eid_str, &peer_eid) != 0) {
        return kvaf_ub_management_error(
            "client_parse_peer_eid", errno, EINVAL);
    }
    fill_rjetty(peer_eid, peer_jetty_id, &rjetty);
    ctx->pair.tjetty = kvaf_ub_import_peer_jetty_tp_aware(
        ctx->urma_ctx, &rjetty, &g_ub_mgmt_token);
    if (ctx->pair.tjetty == NULL) {
        return kvaf_ub_management_error(
            "client_import_peer_jetty", errno, EIO);
    }

    ctx->send_buf[0] = 'H';
    sge.addr = (uint64_t)ctx->send_buf;
    sge.len = UB_MGMT_HANDSHAKE_SIZE;
    sge.tseg = ctx->tseg_send;
    wr.opcode = URMA_OPC_SEND;
    wr.flag.value = 0;
    wr.flag.bs.complete_enable = 1;
    wr.tjetty = ctx->pair.tjetty;
    wr.user_ctx = 0;
    wr.send.src.sge = &sge;
    wr.send.src.num_sge = 1;
    wr.send.imm_data = 0;
    wr.next = NULL;
    errno = 0;
    post_status = kvaf_tracking_urma_post_jetty_send_wr(
        ctx->pair.jetty, &wr, &bad);
    operation_errno = errno;
    if (post_status != URMA_SUCCESS) {
        kvaf_ub_record_management_diagnostic(
            "client_post_handshake_send", operation_errno,
            operation_errno == 0 ? EIO : operation_errno,
            (int)post_status, true, 0, false);
        return kvaf_ub_management_error(
            "client_post_handshake_send", operation_errno, EIO);
    }
    errno = 0;
    poll_status = poll_one_cqe(ctx->jfc, &cr, true);
    if (poll_status != 0) {
        operation_errno = errno;
        kvaf_ub_record_management_diagnostic(
            "client_poll_handshake_cqe", operation_errno,
            operation_errno == 0 ? EIO : operation_errno,
            poll_status, true, (int)cr.status,
            cr.status != URMA_CR_SUCCESS);
        return kvaf_ub_management_error(
            "client_poll_handshake_cqe", operation_errno, EIO);
    }
    return 0;
}

static int kvaf_ub_establish_connection(
    const perftest_config_t *cfg, const char *expected_peer_eid)
{
    const comm_ub_cfg_t ub_cfg = {
        .src_eid = cfg->mgmt_addr,
        .dst_eid = cfg->server_ip,
        .dst_jetty_id = cfg->port,
    };
    ub_mgmt_ctx_t *ctx = NULL;

    if (g_ub_ctx != NULL ||
        g_kvaf_ub_management_tokens[KVAF_UB_MANAGEMENT_SEND] != NULL ||
        g_kvaf_ub_management_tokens[KVAF_UB_MANAGEMENT_RECEIVE] != NULL ||
        g_kvaf_ub_management_tokens[KVAF_UB_MANAGEMENT_HANDSHAKE] != NULL ||
        cfg->pair_num != 1U ||
        expected_peer_eid == NULL || expected_peer_eid[0] == '\0') {
        return kvaf_ub_management_error(
            "validate_establish_arguments", EINVAL, EINVAL);
    }
    kvaf_ub_reset_close_cursor();
    kvaf_ub_send_tracker_reset(&g_kvaf_ub_send_tracker);
    if (kvaf_ub_create_local_resources_checked(
            &ub_cfg, cfg->dev_name, cfg->eid_idx, &ctx) != 0) {
        /* A partial owner is deliberately published for checked rollback. */
        g_ub_ctx = ctx;
        return -1;
    }
    g_ub_ctx = ctx;

    if (ctx->is_server) {
        g_kvaf_ub_server_arming = true;
        g_kvaf_ub_server_recv_posts = 0;
        g_kvaf_ub_server_jetty_id = ctx->pair.jetty->jetty_id.id;
        (void)snprintf(g_kvaf_ub_server_local_eid,
            sizeof(g_kvaf_ub_server_local_eid), "%s", ub_cfg.src_eid);
        errno = 0;
        if (ub_server_handshake(ctx) != 0) {
            const int handshake_errno = errno;

            g_kvaf_ub_server_arming = false;
            return kvaf_ub_management_error(
                "server_handshake", handshake_errno, EIO);
        }
        g_kvaf_ub_server_arming = false;
    } else {
        errno = 0;
        if (kvaf_ub_client_handshake_checked(
                ctx, ub_cfg.dst_eid, ub_cfg.dst_jetty_id) != 0) {
            return kvaf_ub_management_error(
                "client_handshake", errno, EIO);
        }
    }

    /* Never exchange an MR, token or data-Jetty descriptor with an endpoint
     * other than the EID selected by the caller.  On the server this checks
     * the hardware-filled receive CQE identity, not client-supplied bytes. */
    if (!kvaf_ub_peer_matches(expected_peer_eid)) {
        return kvaf_ub_management_error(
            "validate_peer_eid", EACCES, EACCES);
    }

    if (!ctx->is_server) {
        errno = 0;
        if (ub_post_one_recv(ctx) != 0) {
            return kvaf_ub_management_error(
                "post_client_receive", errno, EIO);
        }
    }
    ctx->pair.have_pending_recv = !ctx->is_server;
    return 0;
}

static int kvaf_ub_close_connection_checked(void)
{
    ub_mgmt_ctx_t *ctx = g_ub_ctx;
    int status;

    if (ctx == NULL) {
        return KVAF_PERFTEST_MANAGEMENT_CLOSE_OK;
    }
    if (g_kvaf_ub_close_attempt != UINT64_MAX) {
        ++g_kvaf_ub_close_attempt;
    }

    /* A create API can fail after the source-linked provider acquired a
     * lower kernel object. Drain those context-owned orphan sidecars before
     * changing the state of successfully returned parent handles. */
    if (g_kvaf_ub_close_phase == KVAF_UB_CLOSE_RETRY_ORPHANS) {
        int orphan_status = 0;
        int orphan_errno = 0;

        if (kvaf_udma_source_cleanup_is_unknown() != 0 ||
            kvaf_source_ummu_cleanup_is_unknown() != 0) {
            return kvaf_ub_close_operation_failed(
                KVAF_UB_CLOSE_RETRY_ORPHANS, URMA_FAIL, EUCLEAN,
                URMA_FAIL, EUCLEAN, true,
                KVAF_UB_CLOSE_RETRY_NONE, false);
        }
        if (ctx->urma_ctx != NULL) {
            errno = 0;
            orphan_status =
                kvaf_udma_source_retry_orphan_cleanup(ctx->urma_ctx);
            orphan_errno = errno;
        }
        if (orphan_status != 0) {
            if (kvaf_udma_source_cleanup_is_unknown() != 0 ||
                kvaf_source_ummu_cleanup_is_unknown() != 0) {
                return kvaf_ub_close_operation_failed(
                    KVAF_UB_CLOSE_RETRY_ORPHANS, orphan_status,
                    orphan_errno == 0 ? EIO : orphan_errno,
                    URMA_FAIL, EIO, true,
                    KVAF_UB_CLOSE_RETRY_NONE, false);
            }
            return kvaf_ub_close_operation_failed(
                KVAF_UB_CLOSE_RETRY_ORPHANS, orphan_status,
                orphan_errno == 0 ? EIO : orphan_errno,
                URMA_SUCCESS, 0, false,
                KVAF_UB_CLOSE_RETRY_CHECKED_DESTROY, false);
        }
        kvaf_ub_close_phase_completed(KVAF_UB_CLOSE_RETRY_ORPHANS,
            KVAF_UB_CLOSE_QUIESCE_JETTY);
    }

    /* UMDK's data teardown first transitions Jetty/JFR to ERROR and drains
     * their shared JFC.  The management server also needs that precondition:
     * it publishes READY only after two receive WQEs are resident, and a
     * missing client otherwise leaves those buffers referenced by the RQ. */
    if (g_kvaf_ub_close_phase == KVAF_UB_CLOSE_QUIESCE_JETTY) {
        status = kvaf_ub_quiesce_management_jetty(ctx);
        if (status != KVAF_PERFTEST_MANAGEMENT_CLOSE_OK) {
            return status;
        }
        kvaf_ub_close_phase_completed(KVAF_UB_CLOSE_QUIESCE_JETTY,
            KVAF_UB_CLOSE_QUIESCE_JFR);
    }
    if (g_kvaf_ub_close_phase == KVAF_UB_CLOSE_QUIESCE_JFR) {
        status = kvaf_ub_quiesce_management_jfr(ctx);
        if (status != KVAF_PERFTEST_MANAGEMENT_CLOSE_OK) {
            return status;
        }
        kvaf_ub_close_phase_completed(KVAF_UB_CLOSE_QUIESCE_JFR,
            KVAF_UB_CLOSE_DRAIN_JFC);
    }
    if (g_kvaf_ub_close_phase == KVAF_UB_CLOSE_DRAIN_JFC) {
        status = kvaf_ub_drain_management_jfc(ctx);
        if (status != KVAF_PERFTEST_MANAGEMENT_CLOSE_OK) {
            return status;
        }
        kvaf_ub_close_phase_completed(KVAF_UB_CLOSE_DRAIN_JFC,
            KVAF_UB_CLOSE_UNIMPORT_TJETTY);
    }

#define KVAF_UB_DESTROY_STEP(phase, next_phase, handle, operation) do { \
    if (g_kvaf_ub_close_phase == (phase)) { \
        if ((handle) != NULL) { \
            int kvaf_saved_errno; \
            urma_status_t kvaf_destroy_status; \
            if (kvaf_ub_close_phase_uses_ummu((phase))) { \
                kvaf_source_ummu_reset_thread_status(); \
            } \
            errno = 0; \
            kvaf_destroy_status = (operation)(handle); \
            kvaf_saved_errno = errno; \
            if (kvaf_destroy_status != URMA_SUCCESS) { \
                return kvaf_ub_destroy_operation_failed((phase), \
                    kvaf_destroy_status, kvaf_saved_errno); \
            } \
            (handle) = NULL; \
        } \
        kvaf_ub_close_phase_completed((phase), (next_phase)); \
    } \
} while (0)

    KVAF_UB_DESTROY_STEP(KVAF_UB_CLOSE_UNIMPORT_TJETTY,
        KVAF_UB_CLOSE_DELETE_JETTY,
        ctx->pair.tjetty, urma_unimport_jetty);
    KVAF_UB_DESTROY_STEP(KVAF_UB_CLOSE_DELETE_JETTY,
        KVAF_UB_CLOSE_DELETE_JFR,
        ctx->pair.jetty, urma_delete_jetty);
    KVAF_UB_DESTROY_STEP(KVAF_UB_CLOSE_DELETE_JFR,
        KVAF_UB_CLOSE_UNREGISTER_HANDSHAKE,
        ctx->jfr, urma_delete_jfr);
    KVAF_UB_DESTROY_STEP(KVAF_UB_CLOSE_UNREGISTER_HANDSHAKE,
        KVAF_UB_CLOSE_FREE_HANDSHAKE_TOKEN,
        ctx->tseg_recv_hs, urma_unregister_seg);
    KVAF_UB_DESTROY_STEP(KVAF_UB_CLOSE_FREE_HANDSHAKE_TOKEN,
        KVAF_UB_CLOSE_UNREGISTER_RECEIVE,
        g_kvaf_ub_management_tokens[KVAF_UB_MANAGEMENT_HANDSHAKE],
        urma_free_token_id);
    KVAF_UB_DESTROY_STEP(KVAF_UB_CLOSE_UNREGISTER_RECEIVE,
        KVAF_UB_CLOSE_FREE_RECEIVE_TOKEN,
        ctx->tseg_recv, urma_unregister_seg);
    KVAF_UB_DESTROY_STEP(KVAF_UB_CLOSE_FREE_RECEIVE_TOKEN,
        KVAF_UB_CLOSE_UNREGISTER_SEND,
        g_kvaf_ub_management_tokens[KVAF_UB_MANAGEMENT_RECEIVE],
        urma_free_token_id);
    KVAF_UB_DESTROY_STEP(KVAF_UB_CLOSE_UNREGISTER_SEND,
        KVAF_UB_CLOSE_FREE_SEND_TOKEN,
        ctx->tseg_send, urma_unregister_seg);
    KVAF_UB_DESTROY_STEP(KVAF_UB_CLOSE_FREE_SEND_TOKEN,
        KVAF_UB_CLOSE_DELETE_JFC,
        g_kvaf_ub_management_tokens[KVAF_UB_MANAGEMENT_SEND],
        urma_free_token_id);
    KVAF_UB_DESTROY_STEP(KVAF_UB_CLOSE_DELETE_JFC,
        KVAF_UB_CLOSE_DELETE_CONTEXT,
        ctx->jfc, urma_delete_jfc);
    KVAF_UB_DESTROY_STEP(KVAF_UB_CLOSE_DELETE_CONTEXT,
        KVAF_UB_CLOSE_RELEASE_LOCAL,
        ctx->urma_ctx, urma_delete_context);
#undef KVAF_UB_DESTROY_STEP

    if (g_kvaf_ub_close_phase != KVAF_UB_CLOSE_RELEASE_LOCAL) {
        kvaf_udma_source_mark_cleanup_unknown();
        errno = EIO;
        return KVAF_PERFTEST_MANAGEMENT_CLOSE_UNSAFE;
    }
    free(ctx->send_buf);
    free(ctx->recv_buf);
    free(ctx->recv_buf_hs);
    free(ctx);
    g_ub_ctx = NULL;
    kvaf_ub_send_tracker_reset(&g_kvaf_ub_send_tracker);
    kvaf_ub_reset_close_cursor();
    return KVAF_PERFTEST_MANAGEMENT_CLOSE_OK;
}

int kvaf_perftest_establish_connection(const perftest_config_t *cfg,
    const char *expected_peer_eid)
{
    comm_tcp_cfg_t tcp_cfg;
    int result;
    int saved_errno;

    if (cfg == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (cfg->mgmt_type == PERFTEST_MGMT_UB) {
        errno = 0;
        result = kvaf_ub_establish_connection(cfg, expected_peer_eid);
        if (result == 0) {
            return 0;
        }
        saved_errno = errno == 0 ? EIO : errno;
        errno = saved_errno;
        return -1;
    }
    if (cfg->mgmt_type != PERFTEST_MGMT_TCP) {
        errno = EINVAL;
        return -1;
    }
    tcp_cfg.server_ip = cfg->server_ip;
    tcp_cfg.bind_ip = cfg->bind_ip;
    tcp_cfg.enable_ipv6 = cfg->enable_ipv6;
    tcp_cfg.port = cfg->port;
    tcp_cfg.sock_num = cfg->pair_num;
    errno = 0;
    result = tcp_establish_connection(&tcp_cfg);
    if (result == 0) {
        return 0;
    }
    /*
     * Upstream collapses DNS/bind/connect/accept failures to -1 and its
     * cleanup can overwrite errno. Preserve the best concrete socket errno;
     * use EIO only when the upstream path supplied no errno at all.
     */
    saved_errno = errno == 0 ? EIO : errno;
    errno = saved_errno;
    return -1;
}

int kvaf_perftest_close_connection(perftest_config_t *cfg)
{
    if (cfg == NULL) {
        errno = EINVAL;
        return KVAF_PERFTEST_MANAGEMENT_CLOSE_UNSAFE;
    }
    if (cfg->mgmt_type == PERFTEST_MGMT_UB) {
        return kvaf_ub_close_connection_checked();
    }
    if (cfg->mgmt_type == PERFTEST_MGMT_TCP) {
        tcp_close_connection();
        return KVAF_PERFTEST_MANAGEMENT_CLOSE_OK;
    }
    errno = EINVAL;
    return KVAF_PERFTEST_MANAGEMENT_CLOSE_UNSAFE;
}

int kvaf_perftest_sync_data(const perftest_config_t *cfg, uint32_t index,
    int size, char *local_data, char *remote_data)
{
    if (cfg == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (cfg->mgmt_type == PERFTEST_MGMT_UB) {
        return kvaf_upstream_ub_sync_data(
            index, size, local_data, remote_data);
    }
    if (cfg->mgmt_type == PERFTEST_MGMT_TCP) {
        return tcp_sync_data(index, size, local_data, remote_data);
    }
    errno = EINVAL;
    return -1;
}

/* Used only for SO_RCVTIMEO/SO_SNDTIMEO and emergency shutdown. */
int kvaf_perftest_control_fd(uint32_t index)
{
    if (comm_ctx.sock_fd == NULL || index >= comm_ctx.sock_num) {
        errno = EINVAL;
        return -1;
    }
    return comm_ctx.sock_fd[index];
}

/* Official perftest_resources.c consumes the generic management symbols. */
int sync_data(const perftest_config_t *cfg, uint32_t index, int size,
    char *local_data, char *remote_data)
{
    return kvaf_perftest_sync_data(
        cfg, index, size, local_data, remote_data);
}

int sync_time(const perftest_config_t *cfg, uint32_t index, const char *value)
{
    if (cfg == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (cfg->mgmt_type == PERFTEST_MGMT_UB) {
        return kvaf_upstream_ub_sync_time(index, value);
    }
    if (cfg->mgmt_type == PERFTEST_MGMT_TCP) {
        return tcp_sync_time(index, value);
    }
    errno = EINVAL;
    return -1;
}

ssize_t comm_send(const perftest_config_t *cfg, uint32_t index,
    const void *buffer, size_t bytes)
{
    if (cfg == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (cfg->mgmt_type == PERFTEST_MGMT_UB) {
        return kvaf_upstream_ub_comm_send(index, buffer, bytes);
    }
    if (cfg->mgmt_type == PERFTEST_MGMT_TCP) {
        return tcp_comm_send(index, buffer, bytes);
    }
    errno = EINVAL;
    return -1;
}

ssize_t comm_recv(const perftest_config_t *cfg, uint32_t index,
    void *buffer, size_t bytes)
{
    if (cfg == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (cfg->mgmt_type == PERFTEST_MGMT_UB) {
        return kvaf_upstream_ub_comm_recv(index, buffer, bytes);
    }
    if (cfg->mgmt_type == PERFTEST_MGMT_TCP) {
        return tcp_comm_recv(index, buffer, bytes);
    }
    errno = EINVAL;
    return -1;
}

int comm_poll(const perftest_config_t *cfg, uint32_t index, int timeout_ms)
{
    if (cfg == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (cfg->mgmt_type == PERFTEST_MGMT_UB) {
        return kvaf_upstream_ub_comm_poll(index, timeout_ms);
    }
    if (cfg->mgmt_type == PERFTEST_MGMT_TCP) {
        return tcp_comm_poll(index, timeout_ms);
    }
    errno = EINVAL;
    return -1;
}
