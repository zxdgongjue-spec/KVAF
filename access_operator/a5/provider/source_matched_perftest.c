#include "kvaf_source_matched_perftest.h"
#include "kvaf_source_matched_udma.h"

#include <errno.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>

/*
 * Compile the selected perftest resource implementation into the provider
 * instead of linking a separately built liburma_perftest.so. This keeps
 * perftest_context_t, the ai_jettys sidecar and the code that fills them in
 * one compilation unit. The renamed legacy exports remain private to this
 * shared object; KVAF calls only the checked entry points below.
 */
#ifndef KVAF_PERFTEST_RESOURCES_SOURCE
#error "KVAF_PERFTEST_RESOURCES_SOURCE must name perftest_resources.c"
#endif
/*
 * The selected perftest snapshot calls urma_uninit() from init_device()'s
 * failure labels even though its matching urma_init() is commented out.
 * Runtime ownership belongs exclusively to kvaf_udma_source_runtime_*;
 * prevent the included legacy source from mutating that process-global state.
 */
static urma_status_t kvaf_legacy_urma_uninit_noop(void)
{
    return URMA_SUCCESS;
}

static urma_device_t *kvaf_owned_urma_get_device_by_name(char *name)
{
    urma_device_t *device = urma_get_device_by_name(name);

    return kvaf_udma_source_provider_owns_device(device) == 1
        ? device : NULL;
}

#define create_duplex_ctx kvaf_legacy_create_duplex_ctx
#define destroy_ctx kvaf_legacy_destroy_ctx
#define init_device kvaf_legacy_init_device
#define urma_get_device_by_name kvaf_owned_urma_get_device_by_name
#define urma_uninit kvaf_legacy_urma_uninit_noop
#include KVAF_PERFTEST_RESOURCES_SOURCE
#undef urma_uninit
#undef urma_get_device_by_name
#undef init_device
#undef destroy_ctx
#undef create_duplex_ctx

static int kvaf_init_device_checked(
    perftest_context_t *ctx, perftest_config_t *cfg)
{
    urma_device_t *device;

    device = urma_get_device_by_name(cfg->dev_name);
    if (device == NULL ||
        kvaf_udma_source_provider_owns_device(device) != 1) {
        return -1;
    }
    if (urma_query_device(device, &ctx->dev_attr) != URMA_SUCCESS) {
        return -1;
    }
    if (cfg->enable_user_tp &&
        (device->type == URMA_TRANSPORT_UB ||
         (ctx->dev_attr.dev_cap.sub_trans_mode_cap &
          URMA_RC_USER_TP) == 0)) {
        return -1;
    }
    ctx->urma_ctx = urma_create_context(device, cfg->eid_idx);
    if (ctx->urma_ctx == NULL) {
        return -1;
    }
    ctx->eid = ctx->urma_ctx->eid;
    /* Keep the checked context handle intact when capability checks fail. */
    return check_dev_cap(ctx, cfg);
}

/*
 * The selected urma_perftest fork carries a copied udma_u_jetty/jfc layout
 * solely to populate its optional urma_ai_jetty_t sidecar.  That copy is not
 * an ABI. KVAF creates the same public URMA resources without calling
 * fill_ai_jetty(); the only private queue access is the isolated accessor
 * compiled with the same source-built UDMA snapshot and checked by the public
 * ABI probe after all public resource creation has completed.
 */
static int kvaf_create_duplex_queues_public(
    perftest_context_t *ctx, perftest_config_t *cfg)
{
    urma_jetty_flag_t jetty_flag = {0};
    urma_jfc_cfg_t jfc_cfg = {0};
    urma_jfs_cfg_t jfs_cfg = {0};
    urma_jfr_cfg_t jfr_cfg = {0};
    urma_jetty_cfg_t jetty_cfg = {0};

    if (ctx == NULL || cfg == NULL || cfg->jettys != 1U ||
        cfg->share_jfr ||
        cfg->inline_size >
            ctx->dev_attr.dev_cap.max_jfs_inline_len) {
        return -1;
    }

    ctx->jetty_num = cfg->jettys;
    ctx->jfc_s = calloc(cfg->jettys, sizeof(*ctx->jfc_s));
    ctx->jfc_r = calloc(cfg->jettys, sizeof(*ctx->jfc_r));
    if (ctx->jfc_s == NULL || ctx->jfc_r == NULL) {
        return -1;
    }
    jfc_cfg.depth = cfg->jfc_depth;
    jfc_cfg.flag.bs.lock_free = cfg->lock_free ? 1 : 0;
    jfc_cfg.flag.bs.jfc_inline = cfg->jfc_inline ? 1 : 0;
    jfc_cfg.jfce = NULL;
    jfc_cfg.user_ctx = (uint64_t)NULL;
    ctx->jfc_s[0] = urma_create_jfc(ctx->urma_ctx, &jfc_cfg);
    if (ctx->jfc_s[0] == NULL) {
        return -1;
    }
    ctx->jfc_r[0] = urma_create_jfc(ctx->urma_ctx, &jfc_cfg);
    if (ctx->jfc_r[0] == NULL) {
        return -1;
    }

    fill_jfs_cfg(ctx, cfg, cfg->inline_size, &jfs_cfg);
    fill_jfr_cfg(ctx, cfg, &jfr_cfg);
    jetty_flag.bs.share_jfr = 0;
    jetty_cfg.flag = jetty_flag;
    jetty_cfg.jfs_cfg = jfs_cfg;
    jetty_cfg.jfr_cfg = &jfr_cfg;
    jetty_cfg.id = cfg->jetty_id;

    ctx->jetty = calloc(cfg->jettys, sizeof(*ctx->jetty));
    ctx->ai_jettys =
        calloc(cfg->jettys, sizeof(*ctx->ai_jettys));
    if (ctx->jetty == NULL || ctx->ai_jettys == NULL) {
        return -1;
    }

    jetty_cfg.jfs_cfg.jfc = ctx->jfc_s[0];
    jetty_cfg.jfr_cfg->jfc = ctx->jfc_r[0];
    ctx->jetty[0] = urma_create_jetty(ctx->urma_ctx, &jetty_cfg);
    if (ctx->jetty[0] == NULL) {
        return -1;
    }
    return 0;
}

static int kvaf_register_memory_checked(
    perftest_context_t *ctx, const perftest_config_t *cfg)
{
    uint64_t maximum_size;
    urma_reg_seg_flag_t flag = {0};
    urma_seg_cfg_t seg_cfg = {0};

    if (ctx == NULL || cfg == NULL || cfg->jettys != 1U ||
        cfg->seg_pre_jetty || cfg->dev_buf == NULL ||
        cfg->cache_line_size == 0) {
        return -1;
    }
    ctx->local_buf = calloc(1, sizeof(*ctx->local_buf));
    if (ctx->local_buf == NULL) {
        return -1;
    }
    ctx->local_buf[0] = cfg->dev_buf;
    ctx->page_size = cfg->page_size;
    maximum_size = cfg->size > ctx->page_size
        ? cfg->size : ctx->page_size;
    ctx->buf_size = PERFTEST_ALIGN_CACHELINE(
        maximum_size, cfg->cache_line_size);
    if (ctx->buf_size > UINT64_MAX / PERFTEST_BUF_NUM) {
        return -1;
    }
    ctx->buf_len = ctx->buf_size * PERFTEST_BUF_NUM;

    if (ctx->urma_ctx->dev->type == URMA_TRANSPORT_UB) {
        ctx->token_id = calloc(1, sizeof(*ctx->token_id));
        if (ctx->token_id == NULL) {
            return -1;
        }
        ctx->token_id[0] = urma_alloc_token_id(ctx->urma_ctx);
        if (ctx->token_id[0] == NULL) {
            return -1;
        }
        ctx->ai_jettys[0].sq.token_id =
            ctx->token_id[0]->token_id;
    }

    ctx->local_tseg = calloc(1, sizeof(*ctx->local_tseg));
    if (ctx->local_tseg == NULL) {
        return -1;
    }
    flag.bs.token_policy = cfg->token_policy;
    flag.bs.cacheable = URMA_NON_CACHEABLE;
    flag.bs.access = PERFTEST_DEF_ACCESS;
    flag.bs.token_id_valid = URMA_TOKEN_ID_VALID;
    flag.bs.non_pin = 1;
    seg_cfg.va = (uint64_t)(uintptr_t)ctx->local_buf[0];
    seg_cfg.len = ctx->buf_len;
    seg_cfg.token_value = g_perftest_token;
    seg_cfg.flag = flag;
    seg_cfg.user_ctx = (uintptr_t)NULL;
    if (ctx->token_id != NULL) {
        seg_cfg.token_id = ctx->token_id[0];
    }
    ctx->local_tseg[0] = urma_register_seg(ctx->urma_ctx, &seg_cfg);
    return ctx->local_tseg[0] == NULL ? -1 : 0;
}

static void kvaf_teardown_set_cursor(
    struct kvaf_perftest_teardown_state *state,
    enum kvaf_perftest_teardown_object object, uint32_t index,
    const void *handle)
{
    state->object = (uint32_t)object;
    state->object_index = index;
    state->object_handle = (uint64_t)(uintptr_t)handle;
}

static void kvaf_teardown_advance(
    struct kvaf_perftest_teardown_state *state,
    enum kvaf_perftest_teardown_phase phase)
{
    state->phase = (uint32_t)phase;
    state->object = KVAF_PERFTEST_TEARDOWN_OBJECT_NONE;
    state->object_index = 0;
    state->disconnect_action_mask = 0;
    state->last_status = 0;
    state->object_handle = 0;
}

static int kvaf_unimport_segments_checked(
    perftest_context_t *ctx,
    struct kvaf_perftest_teardown_state *state)
{
    uint32_t i;

    if (ctx->import_tseg == NULL) {
        return 0;
    }
    for (i = state->object_index; i < ctx->jetty_num; ++i) {
        if (ctx->import_tseg[i] == NULL) {
            continue;
        }
        kvaf_teardown_set_cursor(state,
            KVAF_PERFTEST_TEARDOWN_OBJECT_TARGET_SEGMENT, i,
            ctx->import_tseg[i]);
        state->last_status = urma_unimport_seg(ctx->import_tseg[i]);
        if (state->last_status != URMA_SUCCESS) {
            return state->last_status;
        }
        ctx->import_tseg[i] = NULL;
        state->object_index = i + 1U;
    }
    free(ctx->import_tseg);
    ctx->import_tseg = NULL;
    return 0;
}

#define KVAF_DISCONNECT_UNBIND_DONE (1U << 0)
#define KVAF_DISCONNECT_UNADVISE_DONE (1U << 1)

static int kvaf_disconnect_jettys_checked(
    perftest_context_t *ctx, const perftest_config_t *cfg,
    struct kvaf_perftest_teardown_state *state)
{
    uint32_t i;

    if (ctx->import_tjetty == NULL) {
        return 0;
    }
    /*
     * KVAF deliberately disables async import. Its CTP path may use RC or UB
     * RM depending on the source-matched parser/provider combination, so
     * preserve the same disconnect policy as disconnect_jetty_default().
     */
    for (i = state->object_index; i < ctx->jetty_num; ++i) {
        if (ctx->import_tjetty[i] == NULL) {
            state->object_index = i + 1U;
            state->disconnect_action_mask = 0;
            continue;
        }
        kvaf_teardown_set_cursor(state,
            KVAF_PERFTEST_TEARDOWN_OBJECT_TARGET_JETTY, i,
            ctx->import_tjetty[i]);
        if (cfg->trans_mode == URMA_TM_RC &&
            (state->disconnect_action_mask &
             KVAF_DISCONNECT_UNBIND_DONE) == 0) {
            state->last_status = urma_unbind_jetty(ctx->jetty[i]);
            if (state->last_status != URMA_SUCCESS) {
                return state->last_status;
            }
            state->disconnect_action_mask |=
                KVAF_DISCONNECT_UNBIND_DONE;
        }
        if (cfg->trans_mode == URMA_TM_RM &&
            ctx->urma_ctx->dev->type != URMA_TRANSPORT_UB &&
            (state->disconnect_action_mask &
             KVAF_DISCONNECT_UNADVISE_DONE) == 0) {
            state->last_status = urma_unadvise_jetty(
                ctx->jetty[i], ctx->import_tjetty[i]);
            if (state->last_status != URMA_SUCCESS) {
                return state->last_status;
            }
            state->disconnect_action_mask |=
                KVAF_DISCONNECT_UNADVISE_DONE;
        }
        state->last_status =
            urma_unimport_jetty(ctx->import_tjetty[i]);
        if (state->last_status != URMA_SUCCESS) {
            return state->last_status;
        }
        ctx->import_tjetty[i] = NULL;
        state->object_index = i + 1U;
        state->disconnect_action_mask = 0;
    }
    free(ctx->import_tjetty);
    ctx->import_tjetty = NULL;
    return 0;
}

static int kvaf_unregister_segments_checked(
    perftest_context_t *ctx, const perftest_config_t *cfg,
    struct kvaf_perftest_teardown_state *state)
{
    uint32_t i;
    const uint32_t segment_count =
        cfg->seg_pre_jetty ? cfg->jettys : 1U;

    if (ctx->local_tseg == NULL) {
        return 0;
    }
    for (i = state->object_index; i < segment_count; ++i) {
        if (ctx->local_tseg[i] == NULL) {
            state->object_index = i + 1U;
            continue;
        }
        kvaf_teardown_set_cursor(state,
            KVAF_PERFTEST_TEARDOWN_OBJECT_LOCAL_SEGMENT, i,
            ctx->local_tseg[i]);
        state->last_status = urma_unregister_seg(ctx->local_tseg[i]);
        if (state->last_status != URMA_SUCCESS) {
            return state->last_status;
        }
        ctx->local_tseg[i] = NULL;
        state->object_index = i + 1U;
    }
    free(ctx->local_tseg);
    ctx->local_tseg = NULL;
    return 0;
}

static int kvaf_free_tokens_checked(
    perftest_context_t *ctx, const perftest_config_t *cfg,
    struct kvaf_perftest_teardown_state *state)
{
    uint32_t i;
    const uint32_t segment_count =
        cfg->seg_pre_jetty ? cfg->jettys : 1U;

    if (ctx->token_id == NULL) {
        return 0;
    }
    if (ctx->urma_ctx == NULL) {
        return -1;
    }
    if (ctx->urma_ctx->dev->type == URMA_TRANSPORT_UB) {
        for (i = state->object_index; i < segment_count; ++i) {
            if (ctx->token_id[i] == NULL) {
                state->object_index = i + 1U;
                continue;
            }
            kvaf_teardown_set_cursor(state,
                KVAF_PERFTEST_TEARDOWN_OBJECT_TOKEN_ID, i,
                ctx->token_id[i]);
            state->last_status = urma_free_token_id(ctx->token_id[i]);
            if (state->last_status != URMA_SUCCESS) {
                return state->last_status;
            }
            ctx->token_id[i] = NULL;
            state->object_index = i + 1U;
        }
    }
    free(ctx->token_id);
    ctx->token_id = NULL;
    return 0;
}

static void kvaf_release_local_buffer_vector(perftest_context_t *ctx)
{
    /*
     * cfg->dev_buf is owned by KVAF. perftest allocates only this pointer
     * vector, never the external Host DVA/HBM allocation itself.
     */
    free(ctx->local_buf);
    ctx->local_buf = NULL;
}

static int kvaf_delete_jettys_checked(
    perftest_context_t *ctx,
    struct kvaf_perftest_teardown_state *state)
{
    uint32_t i;

    if (ctx->jetty != NULL) {
        for (i = state->object_index; i < ctx->jetty_num; ++i) {
            if (ctx->jetty[i] == NULL) {
                state->object_index = i + 1U;
                continue;
            }
            kvaf_teardown_set_cursor(state,
                KVAF_PERFTEST_TEARDOWN_OBJECT_JETTY, i,
                ctx->jetty[i]);
            state->last_status = urma_delete_jetty(ctx->jetty[i]);
            if (state->last_status != URMA_SUCCESS) {
                return state->last_status;
            }
            ctx->jetty[i] = NULL;
            state->object_index = i + 1U;
        }
        free(ctx->jetty);
        ctx->jetty = NULL;
    }
    free(ctx->ai_jettys);
    ctx->ai_jettys = NULL;
    return 0;
}

static int kvaf_delete_jfrs_checked(
    perftest_context_t *ctx, const perftest_config_t *cfg,
    struct kvaf_perftest_teardown_state *state)
{
    uint32_t i;

    if (ctx->jfr != NULL) {
        const uint32_t jfr_count =
            cfg->jettys / cfg->jettys_pre_jfr;
        for (i = state->object_index; i < jfr_count; ++i) {
            if (ctx->jfr[i] == NULL) {
                state->object_index = i + 1U;
                continue;
            }
            kvaf_teardown_set_cursor(state,
                KVAF_PERFTEST_TEARDOWN_OBJECT_JFR, i,
                ctx->jfr[i]);
            state->last_status = urma_delete_jfr(ctx->jfr[i]);
            if (state->last_status != URMA_SUCCESS) {
                return state->last_status;
            }
            ctx->jfr[i] = NULL;
            state->object_index = i + 1U;
        }
        free(ctx->jfr);
        ctx->jfr = NULL;
    }
    return 0;
}

static int kvaf_delete_recv_jfcs_checked(
    perftest_context_t *ctx, const perftest_config_t *cfg,
    struct kvaf_perftest_teardown_state *state)
{
    uint32_t i;

    if (ctx->jfc_r != NULL) {
        for (i = state->object_index; i < cfg->jettys; ++i) {
            if (i > 0 &&
                (!cfg->pair_flag || cfg->type == PERFTEST_BW)) {
                break;
            }
            if (ctx->jfc_r[i] == NULL) {
                state->object_index = i + 1U;
                continue;
            }
            kvaf_teardown_set_cursor(state,
                KVAF_PERFTEST_TEARDOWN_OBJECT_RECV_JFC, i,
                ctx->jfc_r[i]);
            state->last_status = urma_delete_jfc(ctx->jfc_r[i]);
            if (state->last_status != URMA_SUCCESS) {
                return state->last_status;
            }
            ctx->jfc_r[i] = NULL;
            state->object_index = i + 1U;
        }
        free(ctx->jfc_r);
        ctx->jfc_r = NULL;
    }
    return 0;
}

static int kvaf_delete_send_jfcs_checked(
    perftest_context_t *ctx, const perftest_config_t *cfg,
    struct kvaf_perftest_teardown_state *state)
{
    uint32_t i;

    if (ctx->jfc_s != NULL) {
        for (i = state->object_index; i < cfg->jettys; ++i) {
            if (i > 0 &&
                (!cfg->pair_flag || cfg->type == PERFTEST_BW)) {
                break;
            }
            if (ctx->jfc_s[i] == NULL) {
                state->object_index = i + 1U;
                continue;
            }
            kvaf_teardown_set_cursor(state,
                KVAF_PERFTEST_TEARDOWN_OBJECT_SEND_JFC, i,
                ctx->jfc_s[i]);
            state->last_status = urma_delete_jfc(ctx->jfc_s[i]);
            if (state->last_status != URMA_SUCCESS) {
                return state->last_status;
            }
            ctx->jfc_s[i] = NULL;
            state->object_index = i + 1U;
        }
        free(ctx->jfc_s);
        ctx->jfc_s = NULL;
    }
    ctx->jetty_num = 0;
    return 0;
}

struct kvaf_checked_phase_status {
    uint32_t magic;
    uint32_t phase;
    int32_t status;
    uint32_t reserved;
};

#define KVAF_CHECKED_PHASE_MAGIC 0x4b435053U
#define KVAF_CHECKED_PHASE_INIT_DEVICE 1U
#define KVAF_CHECKED_PHASE_CREATE_QUEUES 2U
#define KVAF_CHECKED_PHASE_REGISTER_MR 3U
#define KVAF_CHECKED_PHASE_EXCHANGE_DONE 4U
#define KVAF_CHECKED_PHASE_IMPORT_SEG 5U
#define KVAF_CHECKED_PHASE_CONNECT_DONE 6U
#define KVAF_CHECKED_PHASE_CREATE_RUN_CTX 7U
#define KVAF_CHECKED_PHASE_PROVIDER_READY 8U
#define KVAF_CHECKED_PHASE_RUNTIME_READY 9U
#define KVAF_CHECKED_PHASE_DISCONNECT 101U
#define KVAF_CHECKED_PHASE_UNIMPORT_SEG 102U
#define KVAF_TEARDOWN_STATE_MAGIC 0x4b545344U

enum kvaf_phase_sync_result {
    KVAF_PHASE_SYNC_OK = 0,
    KVAF_PHASE_SYNC_RETRYABLE = 1,
    KVAF_PHASE_SYNC_UNCERTAIN = 2,
};

static int kvaf_write_full(int fd, const void *buffer, size_t bytes)
{
    const uint8_t *cursor = (const uint8_t *)buffer;
    size_t written = 0;

    while (written < bytes) {
        ssize_t result = send(
            fd, cursor + written, bytes - written, MSG_NOSIGNAL);
        if (result < 0 && errno == EINTR) {
            continue;
        }
        if (result <= 0) {
            return -1;
        }
        written += (size_t)result;
    }
    return 0;
}

static int kvaf_read_full(int fd, void *buffer, size_t bytes)
{
    uint8_t *cursor = (uint8_t *)buffer;
    size_t consumed = 0;

    while (consumed < bytes) {
        ssize_t result = read(fd, cursor + consumed, bytes - consumed);
        if (result < 0 && errno == EINTR) {
            continue;
        }
        if (result <= 0) {
            return -1;
        }
        consumed += (size_t)result;
    }
    return 0;
}

static enum kvaf_phase_sync_result kvaf_sync_phase_status_detailed(
    const perftest_config_t *cfg, uint32_t phase, int local_status)
{
    uint32_t i;
    bool retryable = local_status != 0;
    const struct kvaf_checked_phase_status local = {
        .magic = KVAF_CHECKED_PHASE_MAGIC,
        .phase = phase,
        .status = local_status,
        .reserved = 0,
    };

    for (i = 0; i < cfg->pair_num; ++i) {
        struct kvaf_checked_phase_status peer = {0};
        if (kvaf_write_full(
                cfg->comm.sock_fd[i], &local, sizeof(local)) != 0 ||
            kvaf_read_full(
                cfg->comm.sock_fd[i], &peer, sizeof(peer)) != 0 ||
            peer.magic != KVAF_CHECKED_PHASE_MAGIC ||
            peer.phase != phase) {
            return KVAF_PHASE_SYNC_UNCERTAIN;
        }
        if (peer.status != 0) {
            retryable = true;
        }
    }
    return retryable ? KVAF_PHASE_SYNC_RETRYABLE : KVAF_PHASE_SYNC_OK;
}

static void kvaf_abort_control_sockets(const perftest_config_t *cfg)
{
    uint32_t i;

    if (cfg == NULL || cfg->comm.sock_fd == NULL) {
        return;
    }
    /*
     * Closing the control protocol is safe even when DMA teardown is not.
     * It wakes a peer blocked in an upstream exchange helper while both
     * processes retain every URMA object and backing allocation for reset.
     */
    for (i = 0; i < cfg->pair_num; ++i) {
        if (cfg->comm.sock_fd[i] >= 0) {
            (void)shutdown(cfg->comm.sock_fd[i], SHUT_RDWR);
        }
    }
}

static int kvaf_sync_local_create_phase(
    const perftest_config_t *cfg, uint32_t phase, int local_status)
{
    enum kvaf_phase_sync_result sync_result =
        kvaf_sync_phase_status_detailed(cfg, phase, local_status);

    if (sync_result == KVAF_PHASE_SYNC_OK) {
        return KVAF_PERFTEST_CHECKED_OK;
    }
    if (sync_result == KVAF_PHASE_SYNC_RETRYABLE) {
        return KVAF_PERFTEST_CHECKED_CREATE_CLEANUP_REQUIRED;
    }
    kvaf_abort_control_sockets(cfg);
    return KVAF_PERFTEST_CHECKED_CREATE_UNSAFE;
}

static int kvaf_teardown_retryable(
    struct kvaf_perftest_teardown_state *state, int status)
{
    state->last_status = status;
    return KVAF_PERFTEST_CHECKED_DESTROY_RETRYABLE;
}

static int kvaf_teardown_mark_unsafe(
    struct kvaf_perftest_teardown_state *state, int status)
{
    state->unsafe_from_phase = state->phase;
    state->phase = KVAF_PERFTEST_TEARDOWN_UNSAFE;
    if (state->object == KVAF_PERFTEST_TEARDOWN_OBJECT_NONE) {
        state->object =
            KVAF_PERFTEST_TEARDOWN_OBJECT_CONTROL_PROTOCOL;
    }
    state->last_status = status;
    return KVAF_PERFTEST_CHECKED_DESTROY_UNSAFE;
}

void kvaf_perftest_teardown_state_init(
    struct kvaf_perftest_teardown_state *state)
{
    if (state == NULL) {
        return;
    }
    (void)memset(state, 0, sizeof(*state));
    state->magic = KVAF_TEARDOWN_STATE_MAGIC;
    state->phase = KVAF_PERFTEST_TEARDOWN_NOT_STARTED;
}

static int kvaf_destroy_partial_duplex_ctx_checked(
    perftest_context_t *ctx, perftest_config_t *cfg,
    struct kvaf_perftest_teardown_state *state)
{
    int status;
    enum kvaf_phase_sync_result sync_result;

    if (ctx == NULL || cfg == NULL || state == NULL ||
        state->magic != KVAF_TEARDOWN_STATE_MAGIC) {
        return KVAF_PERFTEST_CHECKED_DESTROY_UNSAFE;
    }
    if (state->attempts != UINT32_MAX) {
        ++state->attempts;
    }
    if (cfg->enable_async_import || cfg->enable_credit ||
        cfg->enable_user_tp || cfg->use_jfce) {
        /*
         * Those modes are outside the direct MVP and need additional checked
         * teardown APIs. Never silently fall back to the legacy void path.
         */
        return kvaf_teardown_mark_unsafe(state, -1);
    }

    for (;;) {
        switch ((enum kvaf_perftest_teardown_phase)state->phase) {
        case KVAF_PERFTEST_TEARDOWN_NOT_STARTED:
            kvaf_teardown_advance(
                state, KVAF_PERFTEST_TEARDOWN_RUN_CONTEXT);
            continue;
        case KVAF_PERFTEST_TEARDOWN_RUN_CONTEXT:
            destroy_run_ctx(ctx);
            kvaf_teardown_advance(
                state, KVAF_PERFTEST_TEARDOWN_DISCONNECT_LOCAL);
            continue;
        case KVAF_PERFTEST_TEARDOWN_DISCONNECT_LOCAL:
            status = kvaf_disconnect_jettys_checked(ctx, cfg, state);
            if (status != URMA_SUCCESS) {
                /*
                 * Report the local failure to the peer before returning. A
                 * valid matching frame means both endpoints can retry this
                 * same phase. An I/O or framing error makes ownership
                 * ambiguous and is deliberately not retried.
                 */
                sync_result = kvaf_sync_phase_status_detailed(
                    cfg, KVAF_CHECKED_PHASE_DISCONNECT, status);
                if (sync_result == KVAF_PHASE_SYNC_UNCERTAIN) {
                    return kvaf_teardown_mark_unsafe(state, status);
                }
                return kvaf_teardown_retryable(state, status);
            }
            kvaf_teardown_advance(
                state, KVAF_PERFTEST_TEARDOWN_DISCONNECT_PEER);
            continue;
        case KVAF_PERFTEST_TEARDOWN_DISCONNECT_PEER:
            kvaf_teardown_set_cursor(state,
                KVAF_PERFTEST_TEARDOWN_OBJECT_CONTROL_PROTOCOL, 0, NULL);
            sync_result = kvaf_sync_phase_status_detailed(
                cfg, KVAF_CHECKED_PHASE_DISCONNECT, 0);
            if (sync_result == KVAF_PHASE_SYNC_UNCERTAIN) {
                return kvaf_teardown_mark_unsafe(state, -1);
            }
            if (sync_result == KVAF_PHASE_SYNC_RETRYABLE) {
                return kvaf_teardown_retryable(state, -1);
            }
            kvaf_teardown_advance(
                state, KVAF_PERFTEST_TEARDOWN_UNIMPORT_SEG_LOCAL);
            continue;
        case KVAF_PERFTEST_TEARDOWN_UNIMPORT_SEG_LOCAL:
            status = kvaf_unimport_segments_checked(ctx, state);
            if (status != URMA_SUCCESS) {
                sync_result = kvaf_sync_phase_status_detailed(
                    cfg, KVAF_CHECKED_PHASE_UNIMPORT_SEG, status);
                if (sync_result == KVAF_PHASE_SYNC_UNCERTAIN) {
                    return kvaf_teardown_mark_unsafe(state, status);
                }
                return kvaf_teardown_retryable(state, status);
            }
            kvaf_teardown_advance(
                state, KVAF_PERFTEST_TEARDOWN_UNIMPORT_SEG_PEER);
            continue;
        case KVAF_PERFTEST_TEARDOWN_UNIMPORT_SEG_PEER:
            kvaf_teardown_set_cursor(state,
                KVAF_PERFTEST_TEARDOWN_OBJECT_CONTROL_PROTOCOL, 0, NULL);
            sync_result = kvaf_sync_phase_status_detailed(
                cfg, KVAF_CHECKED_PHASE_UNIMPORT_SEG, 0);
            if (sync_result == KVAF_PHASE_SYNC_UNCERTAIN) {
                return kvaf_teardown_mark_unsafe(state, -1);
            }
            if (sync_result == KVAF_PHASE_SYNC_RETRYABLE) {
                return kvaf_teardown_retryable(state, -1);
            }
            kvaf_teardown_advance(state,
                KVAF_PERFTEST_TEARDOWN_CONNECTION_METADATA);
            continue;
        case KVAF_PERFTEST_TEARDOWN_CONNECTION_METADATA:
            destroy_connection_info(ctx);
            /*
             * The selected source's free_remote_tp_info() does not release
             * remote_tp_info when CTP is used. Keep these explicit, nulling
             * frees in the same one-shot phase.
             */
            free(ctx->remote_tp_info);
            ctx->remote_tp_info = NULL;
            free(ctx->local_tp_info);
            ctx->local_tp_info = NULL;
            kvaf_teardown_advance(
                state, KVAF_PERFTEST_TEARDOWN_ORPHAN_RESOURCES);
            continue;
        case KVAF_PERFTEST_TEARDOWN_ORPHAN_RESOURCES:
            if (ctx->urma_ctx != NULL) {
                kvaf_teardown_set_cursor(state,
                    KVAF_PERFTEST_TEARDOWN_OBJECT_ORPHAN_RESOURCE, 0,
                    ctx->urma_ctx);
                status = kvaf_udma_source_retry_orphan_cleanup(
                    ctx->urma_ctx);
                if (status != 0) {
                    /*
                     * The exact source sidecars and lower handles remain
                     * owned by this context. A same-session Close retries this
                     * phase before any normal token/context is released.
                     */
                    return kvaf_teardown_retryable(state, status);
                }
            }
            kvaf_teardown_advance(
                state, KVAF_PERFTEST_TEARDOWN_LOCAL_SEGMENTS);
            continue;
        case KVAF_PERFTEST_TEARDOWN_LOCAL_SEGMENTS:
            status = kvaf_unregister_segments_checked(ctx, cfg, state);
            if (status != URMA_SUCCESS) {
                return kvaf_teardown_retryable(state, status);
            }
            kvaf_teardown_advance(
                state, KVAF_PERFTEST_TEARDOWN_TOKEN_IDS);
            continue;
        case KVAF_PERFTEST_TEARDOWN_TOKEN_IDS:
            status = kvaf_free_tokens_checked(ctx, cfg, state);
            if (status != URMA_SUCCESS) {
                return kvaf_teardown_retryable(state, status);
            }
            kvaf_teardown_advance(state,
                KVAF_PERFTEST_TEARDOWN_LOCAL_BUFFER_VECTOR);
            continue;
        case KVAF_PERFTEST_TEARDOWN_LOCAL_BUFFER_VECTOR:
            kvaf_release_local_buffer_vector(ctx);
            kvaf_teardown_advance(
                state, KVAF_PERFTEST_TEARDOWN_JETTYS);
            continue;
        case KVAF_PERFTEST_TEARDOWN_JETTYS:
            status = kvaf_delete_jettys_checked(ctx, state);
            if (status != URMA_SUCCESS) {
                return kvaf_teardown_retryable(state, status);
            }
            kvaf_teardown_advance(
                state, KVAF_PERFTEST_TEARDOWN_JFRS);
            continue;
        case KVAF_PERFTEST_TEARDOWN_JFRS:
            status = kvaf_delete_jfrs_checked(ctx, cfg, state);
            if (status != URMA_SUCCESS) {
                return kvaf_teardown_retryable(state, status);
            }
            kvaf_teardown_advance(
                state, KVAF_PERFTEST_TEARDOWN_RECV_JFCS);
            continue;
        case KVAF_PERFTEST_TEARDOWN_RECV_JFCS:
            status = kvaf_delete_recv_jfcs_checked(ctx, cfg, state);
            if (status != URMA_SUCCESS) {
                return kvaf_teardown_retryable(state, status);
            }
            kvaf_teardown_advance(
                state, KVAF_PERFTEST_TEARDOWN_SEND_JFCS);
            continue;
        case KVAF_PERFTEST_TEARDOWN_SEND_JFCS:
            status = kvaf_delete_send_jfcs_checked(ctx, cfg, state);
            if (status != URMA_SUCCESS) {
                return kvaf_teardown_retryable(state, status);
            }
            kvaf_teardown_advance(
                state, KVAF_PERFTEST_TEARDOWN_CONTEXT);
            continue;
        case KVAF_PERFTEST_TEARDOWN_CONTEXT:
            if (ctx->urma_ctx != NULL) {
                kvaf_teardown_set_cursor(state,
                    KVAF_PERFTEST_TEARDOWN_OBJECT_CONTEXT, 0,
                    ctx->urma_ctx);
                status = urma_delete_context(ctx->urma_ctx);
                if (status != URMA_SUCCESS) {
                    return kvaf_teardown_retryable(state, status);
                }
                ctx->urma_ctx = NULL;
            }
            kvaf_teardown_advance(
                state, KVAF_PERFTEST_TEARDOWN_COMPLETE);
            continue;
        case KVAF_PERFTEST_TEARDOWN_COMPLETE:
            return KVAF_PERFTEST_CHECKED_OK;
        case KVAF_PERFTEST_TEARDOWN_UNSAFE:
        default:
            return KVAF_PERFTEST_CHECKED_DESTROY_UNSAFE;
        }
    }
}

int kvaf_perftest_create_duplex_ctx_checked(
    perftest_context_t *ctx, perftest_config_t *cfg)
{
    int status;
    int phase_status;

    if (ctx == NULL || cfg == NULL ||
        cfg->jetty_mode != PERFTEST_JETTY_DUPLEX ||
        cfg->jettys != 1U || cfg->pair_num != 1U ||
        cfg->dev_buf == NULL || !cfg->tp_aware || !cfg->use_ctp ||
        cfg->trans_mode != URMA_TM_RM || cfg->share_jfr ||
        cfg->seg_pre_jetty ||
        cfg->enable_async_import || cfg->enable_credit ||
        cfg->enable_user_tp || cfg->use_jfce) {
        return KVAF_PERFTEST_CHECKED_CREATE_UNSAFE;
    }

    (void)memset(ctx, 0, sizeof(*ctx));
    status = kvaf_init_device_checked(ctx, cfg);
    phase_status = kvaf_sync_local_create_phase(
        cfg, KVAF_CHECKED_PHASE_INIT_DEVICE, status);
    if (phase_status != KVAF_PERFTEST_CHECKED_OK) {
        return phase_status;
    }
    status = kvaf_create_duplex_queues_public(ctx, cfg);
    phase_status = kvaf_sync_local_create_phase(
        cfg, KVAF_CHECKED_PHASE_CREATE_QUEUES, status);
    if (phase_status != KVAF_PERFTEST_CHECKED_OK) {
        return phase_status;
    }
    status = kvaf_register_memory_checked(ctx, cfg);
    phase_status = kvaf_sync_local_create_phase(
        cfg, KVAF_CHECKED_PHASE_REGISTER_MR, status);
    if (phase_status != KVAF_PERFTEST_CHECKED_OK) {
        return phase_status;
    }

    /*
     * The next two upstream helpers contain their own peer exchanges. If one
     * side fails locally before completing such an exchange, sending a new
     * phase frame would be interpreted as old-protocol payload. Shut down the
     * control socket immediately instead; that makes the peer's full read
     * fail and drives both providers into the poisoned state.
     */
    if (exchange_connection_info(ctx, cfg) != 0) {
        goto unsafe;
    }
    phase_status = kvaf_sync_local_create_phase(
        cfg, KVAF_CHECKED_PHASE_EXCHANGE_DONE, 0);
    if (phase_status != KVAF_PERFTEST_CHECKED_OK) {
        if (phase_status ==
            KVAF_PERFTEST_CHECKED_CREATE_CLEANUP_REQUIRED) {
            return phase_status;
        }
        goto unsafe;
    }

    status = import_seg_for_duplex(ctx, cfg);
    phase_status = kvaf_sync_local_create_phase(
        cfg, KVAF_CHECKED_PHASE_IMPORT_SEG, status);
    if (phase_status != KVAF_PERFTEST_CHECKED_OK) {
        return phase_status;
    }
    if (connect_jetty(ctx, cfg) != 0) {
        goto unsafe;
    }
    if (kvaf_sync_local_create_phase(
            cfg, KVAF_CHECKED_PHASE_CONNECT_DONE, 0) !=
        KVAF_PERFTEST_CHECKED_OK) {
        goto unsafe;
    }
    status = create_run_ctx(ctx, cfg);
    phase_status = kvaf_sync_local_create_phase(
        cfg, KVAF_CHECKED_PHASE_CREATE_RUN_CTX, status);
    if (phase_status != KVAF_PERFTEST_CHECKED_OK) {
        return phase_status;
    }
    return KVAF_PERFTEST_CHECKED_OK;

unsafe:
    /*
     * Do not run the legacy single-sided unwind. Once a local MR has been
     * exchanged, the peer may already hold an imported view; unregistering it
     * without a phase agreement is unsafe. The provider poisons this session
     * and KVAF keeps all backing memory live until reset.
     */
    kvaf_abort_control_sockets(cfg);
    return KVAF_PERFTEST_CHECKED_CREATE_UNSAFE;
}

int kvaf_perftest_confirm_provider_ready(
    perftest_context_t *ctx, perftest_config_t *cfg, int local_status)
{
    int phase_status;

    if (ctx == NULL || cfg == NULL) {
        kvaf_abort_control_sockets(cfg);
        return KVAF_PERFTEST_CHECKED_CREATE_UNSAFE;
    }
    phase_status = kvaf_sync_local_create_phase(
        cfg, KVAF_CHECKED_PHASE_PROVIDER_READY, local_status);
    return phase_status;
}

int kvaf_perftest_confirm_runtime_ready(
    perftest_config_t *cfg, int local_status)
{
    if (cfg == NULL ||
        kvaf_sync_phase_status_detailed(
            cfg, KVAF_CHECKED_PHASE_RUNTIME_READY,
            local_status) != KVAF_PHASE_SYNC_OK) {
        /* No URMA object exists yet; abort only the control exchange. */
        kvaf_abort_control_sockets(cfg);
        return KVAF_PERFTEST_CHECKED_PRE_RESOURCE_FAILED;
    }
    return KVAF_PERFTEST_CHECKED_OK;
}

int kvaf_perftest_destroy_duplex_ctx_checked(
    perftest_context_t *ctx, perftest_config_t *cfg,
    struct kvaf_perftest_teardown_state *state)
{
    return kvaf_destroy_partial_duplex_ctx_checked(ctx, cfg, state);
}
