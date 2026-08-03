#include <errno.h>
#include <fcntl.h>
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
#include "kvaf_source_matched_udma.h"

static struct timespec g_kvaf_control_deadline;
static int g_kvaf_control_deadline_valid;
static unsigned int g_kvaf_control_timeout_seconds;

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

#include "kvaf_source_matched_perftest.h"
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

static urma_status_t kvaf_ub_reuse_runtime_init(
    urma_init_attr_t *attribute)
{
    (void)attribute;
    return URMA_EEXIST;
}

/* All handshake and metadata CQ polling consumes the single monotonic Open
 * deadline.  The upstream loops remain source-identical, but can no longer
 * spin forever if a peer disappears. */
static int kvaf_deadline_urma_poll_jfc(
    urma_jfc_t *jfc, int cr_cnt, urma_cr_t *cr)
{
    if (kvaf_deadline_remaining_ms() < 0) {
        return -1;
    }
    return urma_poll_jfc(jfc, cr_cnt, cr);
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

#define g_exit_flag g_kvaf_ub_exit_flag
#define urma_init kvaf_ub_reuse_runtime_init
#define urma_poll_jfc kvaf_deadline_urma_poll_jfc
#define urma_post_jetty_recv_wr kvaf_ready_urma_post_jetty_recv_wr
#define ub_establish_connection kvaf_upstream_ub_establish_connection
#define ub_close_connection kvaf_upstream_ub_close_connection
#define ub_sync_data kvaf_upstream_ub_sync_data
#define ub_sync_time kvaf_upstream_ub_sync_time
#define ub_comm_send kvaf_upstream_ub_comm_send
#define ub_comm_recv kvaf_upstream_ub_comm_recv
#define ub_comm_poll kvaf_upstream_ub_comm_poll
#include KVAF_PERFTEST_MGMT_UB_SOURCE
#undef ub_comm_poll
#undef ub_comm_recv
#undef ub_comm_send
#undef ub_sync_time
#undef ub_sync_data
#undef ub_close_connection
#undef ub_establish_connection
#undef urma_post_jetty_recv_wr
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
        errno = EINVAL;
        return -1;
    }
    *ctx_out = NULL;
    ctx = (ub_mgmt_ctx_t *)calloc(1, sizeof(*ctx));
    if (ctx == NULL) {
        errno = ENOMEM;
        return -1;
    }
    *ctx_out = ctx;

    if (ub_resolve_dev_and_eid_idx(
            cfg->src_eid, &ctx->dev, &eid_idx) != 0) {
        return -1;
    }
    /* Management and data resources are one endpoint lifetime. Silently
     * resolving the management EID onto a different UDMA device/index would
     * create a second, unproven ownership domain and break teardown proofs. */
    if (strcmp(ctx->dev->name, expected_device_name) != 0 ||
        eid_idx != expected_eid_index) {
        errno = EACCES;
        return -1;
    }
    ctx->urma_ctx = urma_create_context(ctx->dev, eid_idx);
    if (ctx->urma_ctx == NULL) {
        return -1;
    }

    jfc_cfg.depth = UB_MGMT_JFC_DEPTH;
    ctx->jfc = urma_create_jfc(ctx->urma_ctx, &jfc_cfg);
    if (ctx->jfc == NULL) {
        return -1;
    }

    ctx->send_buf = (char *)memalign(
        UB_MGMT_SEG_ALIGN, UB_MGMT_MSG_MAX_SIZE);
    ctx->recv_buf = (char *)memalign(
        UB_MGMT_SEG_ALIGN, UB_MGMT_MSG_MAX_SIZE);
    ctx->recv_buf_hs = (char *)memalign(
        UB_MGMT_SEG_ALIGN, UB_MGMT_MSG_MAX_SIZE);
    if (ctx->send_buf == NULL || ctx->recv_buf == NULL ||
        ctx->recv_buf_hs == NULL) {
        errno = ENOMEM;
        return -1;
    }
    (void)memset(ctx->send_buf, 0, UB_MGMT_MSG_MAX_SIZE);
    (void)memset(ctx->recv_buf, 0, UB_MGMT_MSG_MAX_SIZE);
    (void)memset(ctx->recv_buf_hs, 0, UB_MGMT_MSG_MAX_SIZE);

    seg_cfg.len = UB_MGMT_MSG_MAX_SIZE;
    seg_cfg.token_id = NULL;
    seg_cfg.token_value = g_ub_mgmt_token;
    seg_cfg.flag.value = 0;
    seg_cfg.flag.bs.access = URMA_ACCESS_LOCAL_ONLY;
    seg_cfg.user_ctx = (uint64_t)NULL;
    seg_cfg.iova = 0;

    seg_cfg.va = (uint64_t)ctx->send_buf;
    ctx->tseg_send = urma_register_seg(ctx->urma_ctx, &seg_cfg);
    if (ctx->tseg_send == NULL) {
        return -1;
    }
    seg_cfg.va = (uint64_t)ctx->recv_buf;
    ctx->tseg_recv = urma_register_seg(ctx->urma_ctx, &seg_cfg);
    if (ctx->tseg_recv == NULL) {
        return -1;
    }
    seg_cfg.va = (uint64_t)ctx->recv_buf_hs;
    ctx->tseg_recv_hs = urma_register_seg(ctx->urma_ctx, &seg_cfg);
    if (ctx->tseg_recv_hs == NULL) {
        return -1;
    }

    jfs_cfg.depth = UB_MGMT_JFS_DEPTH;
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
    ctx->jfr = urma_create_jfr(ctx->urma_ctx, &jfr_cfg);
    if (ctx->jfr == NULL) {
        return -1;
    }

    ctx->is_server = cfg->dst_eid == NULL;
    jetty_cfg.id = ctx->is_server ? cfg->dst_jetty_id : 0;
    jetty_cfg.flag.bs.share_jfr = URMA_SHARE_JFR;
    jetty_cfg.jfs_cfg = jfs_cfg;
    jetty_cfg.shared.jfr = ctx->jfr;
    jetty_cfg.shared.jfc = ctx->jfc;
    ctx->pair.jetty = urma_create_jetty(ctx->urma_ctx, &jetty_cfg);
    if (ctx->pair.jetty == NULL) {
        return -1;
    }
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

static int kvaf_ub_establish_connection(
    const perftest_config_t *cfg, const char *expected_peer_eid)
{
    const comm_ub_cfg_t ub_cfg = {
        .src_eid = cfg->mgmt_addr,
        .dst_eid = cfg->server_ip,
        .dst_jetty_id = cfg->port,
    };
    ub_mgmt_ctx_t *ctx = NULL;

    if (g_ub_ctx != NULL || cfg->pair_num != 1U ||
        expected_peer_eid == NULL || expected_peer_eid[0] == '\0') {
        errno = EINVAL;
        return -1;
    }
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
        g_kvaf_ub_server_jetty_id = ub_cfg.dst_jetty_id;
        (void)snprintf(g_kvaf_ub_server_local_eid,
            sizeof(g_kvaf_ub_server_local_eid), "%s", ub_cfg.src_eid);
        if (ub_server_handshake(ctx) != 0) {
            g_kvaf_ub_server_arming = false;
            return -1;
        }
        g_kvaf_ub_server_arming = false;
    } else if (ub_client_handshake(
            ctx, ub_cfg.dst_eid, ub_cfg.dst_jetty_id) != 0) {
        return -1;
    }

    /* Never exchange an MR, token or data-Jetty descriptor with an endpoint
     * other than the EID selected by the caller.  On the server this checks
     * the hardware-filled receive CQE identity, not client-supplied bytes. */
    if (!kvaf_ub_peer_matches(expected_peer_eid)) {
        errno = EACCES;
        return -1;
    }

    if (!ctx->is_server && ub_post_one_recv(ctx) != 0) {
        return -1;
    }
    ctx->pair.have_pending_recv = !ctx->is_server;
    return 0;
}

static int kvaf_ub_close_connection_checked(void)
{
    ub_mgmt_ctx_t *ctx = g_ub_ctx;

    if (ctx == NULL) {
        return KVAF_PERFTEST_MANAGEMENT_CLOSE_OK;
    }
    /* A create API can fail after the source-linked provider acquired a
     * lower kernel object. Drain those context-owned orphan sidecars before
     * deleting the successfully returned parent handles below. */
    if (ctx->urma_ctx != NULL &&
        kvaf_udma_source_retry_orphan_cleanup(ctx->urma_ctx) != 0) {
        errno = EAGAIN;
        return KVAF_PERFTEST_MANAGEMENT_CLOSE_RETRYABLE;
    }
#define KVAF_UB_DESTROY(handle, operation) do { \
    if ((handle) != NULL) { \
        if ((operation)(handle) != URMA_SUCCESS) { \
            errno = EAGAIN; \
            return KVAF_PERFTEST_MANAGEMENT_CLOSE_RETRYABLE; \
        } \
        (handle) = NULL; \
    } \
} while (0)
    KVAF_UB_DESTROY(ctx->pair.tjetty, urma_unimport_jetty);
    KVAF_UB_DESTROY(ctx->pair.jetty, urma_delete_jetty);
    KVAF_UB_DESTROY(ctx->jfr, urma_delete_jfr);
    KVAF_UB_DESTROY(ctx->tseg_recv, urma_unregister_seg);
    KVAF_UB_DESTROY(ctx->tseg_recv_hs, urma_unregister_seg);
    KVAF_UB_DESTROY(ctx->tseg_send, urma_unregister_seg);
    KVAF_UB_DESTROY(ctx->jfc, urma_delete_jfc);
    KVAF_UB_DESTROY(ctx->urma_ctx, urma_delete_context);
#undef KVAF_UB_DESTROY

    free(ctx->send_buf);
    free(ctx->recv_buf);
    free(ctx->recv_buf_hs);
    free(ctx);
    g_ub_ctx = NULL;
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
        return KVAF_PERFTEST_MANAGEMENT_CLOSE_RETRYABLE;
    }
    if (cfg->mgmt_type == PERFTEST_MGMT_UB) {
        return kvaf_ub_close_connection_checked();
    }
    if (cfg->mgmt_type == PERFTEST_MGMT_TCP) {
        tcp_close_connection();
        return KVAF_PERFTEST_MANAGEMENT_CLOSE_OK;
    }
    errno = EINVAL;
    return KVAF_PERFTEST_MANAGEMENT_CLOSE_RETRYABLE;
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
