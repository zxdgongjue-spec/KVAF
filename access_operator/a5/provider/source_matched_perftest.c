#include "kvaf_source_matched_perftest.h"

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
#define create_duplex_ctx kvaf_legacy_create_duplex_ctx
#define destroy_ctx kvaf_legacy_destroy_ctx
#include KVAF_PERFTEST_RESOURCES_SOURCE
#undef destroy_ctx
#undef create_duplex_ctx

/*
 * The selected urma_perftest fork carries a copied udma_u_jetty/jfc layout
 * solely to populate its optional urma_ai_jetty_t sidecar.  That copy is not
 * an ABI and may not match the installed liburma-udma object.  KVAF creates
 * the same public URMA resources without calling fill_ai_jetty(); the only
 * private queue access is the isolated, ABI-pinned accessor after all
 * public resource creation has completed.
 */
static int kvaf_create_duplex_queues_public(
    perftest_context_t *ctx, perftest_config_t *cfg)
{
    urma_jetty_flag_t jetty_flag = {0};
    urma_jfs_cfg_t jfs_cfg = {0};
    urma_jfr_cfg_t jfr_cfg = {0};
    urma_jetty_cfg_t jetty_cfg = {0};
    uint32_t created = 0;

    if (ctx == NULL || cfg == NULL || cfg->jettys != 1U ||
        cfg->share_jfr ||
        cfg->inline_size >
            ctx->dev_attr.dev_cap.max_jfs_inline_len) {
        return -1;
    }

    ctx->jetty_num = cfg->jettys;
    if (create_jfc(ctx, cfg) != 0) {
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
        goto error;
    }

    for (created = 0; created < cfg->jettys; ++created) {
        jetty_cfg.jfs_cfg.jfc = ctx->jfc_s[created];
        jetty_cfg.jfr_cfg->jfc = ctx->jfc_r[created];
        ctx->jetty[created] =
            urma_create_jetty(ctx->urma_ctx, &jetty_cfg);
        if (ctx->jetty[created] == NULL) {
            goto error;
        }
    }
    return 0;

error:
    while (created > 0) {
        --created;
        if (ctx->jetty[created] != NULL) {
            (void)urma_delete_jetty(ctx->jetty[created]);
            ctx->jetty[created] = NULL;
        }
    }
    free(ctx->jetty);
    ctx->jetty = NULL;
    free(ctx->ai_jettys);
    ctx->ai_jettys = NULL;
    destroy_jfc(ctx, cfg);
    ctx->jetty_num = 0;
    return -1;
}

static int kvaf_unimport_segments_checked(perftest_context_t *ctx)
{
    uint32_t i;

    if (ctx->import_tseg == NULL) {
        return 0;
    }
    for (i = 0; i < ctx->jetty_num; ++i) {
        if (ctx->import_tseg[i] == NULL) {
            continue;
        }
        if (urma_unimport_seg(ctx->import_tseg[i]) != URMA_SUCCESS) {
            return -1;
        }
        ctx->import_tseg[i] = NULL;
    }
    free(ctx->import_tseg);
    ctx->import_tseg = NULL;
    return 0;
}

static int kvaf_disconnect_jettys_checked(
    perftest_context_t *ctx, const perftest_config_t *cfg)
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
    for (i = 0; i < ctx->jetty_num; ++i) {
        if (ctx->import_tjetty[i] == NULL) {
            continue;
        }
        if (cfg->trans_mode == URMA_TM_RC &&
            urma_unbind_jetty(ctx->jetty[i]) != URMA_SUCCESS) {
            return -1;
        }
        if (cfg->trans_mode == URMA_TM_RM &&
            ctx->urma_ctx->dev->type != URMA_TRANSPORT_UB &&
            urma_unadvise_jetty(
                ctx->jetty[i], ctx->import_tjetty[i]) != URMA_SUCCESS) {
            return -1;
        }
        if (urma_unimport_jetty(ctx->import_tjetty[i]) != URMA_SUCCESS) {
            return -1;
        }
        ctx->import_tjetty[i] = NULL;
    }
    free(ctx->import_tjetty);
    ctx->import_tjetty = NULL;
    return 0;
}

static int kvaf_unregister_memory_checked(
    perftest_context_t *ctx, const perftest_config_t *cfg)
{
    uint32_t i;
    const uint32_t segment_count =
        cfg->seg_pre_jetty ? cfg->jettys : 1U;

    if (ctx->local_tseg != NULL) {
        for (i = 0; i < segment_count; ++i) {
            if (ctx->local_tseg[i] == NULL) {
                continue;
            }
            if (urma_unregister_seg(ctx->local_tseg[i]) != URMA_SUCCESS) {
                return -1;
            }
            ctx->local_tseg[i] = NULL;
        }
        free(ctx->local_tseg);
        ctx->local_tseg = NULL;
    }

    if (ctx->token_id != NULL &&
        ctx->urma_ctx != NULL &&
        ctx->urma_ctx->dev->type == URMA_TRANSPORT_UB) {
        for (i = 0; i < segment_count; ++i) {
            if (ctx->token_id[i] == NULL) {
                continue;
            }
            if (urma_free_token_id(ctx->token_id[i]) != URMA_SUCCESS) {
                return -1;
            }
            ctx->token_id[i] = NULL;
        }
        free(ctx->token_id);
        ctx->token_id = NULL;
    }

    /*
     * cfg->dev_buf is owned by KVAF. perftest allocates only this pointer
     * vector, never the external Host DVA/HBM allocation itself.
     */
    free(ctx->local_buf);
    ctx->local_buf = NULL;
    return 0;
}

static int kvaf_delete_queues_checked(
    perftest_context_t *ctx, const perftest_config_t *cfg)
{
    uint32_t i;

    if (ctx->jetty != NULL) {
        for (i = 0; i < ctx->jetty_num; ++i) {
            if (ctx->jetty[i] == NULL) {
                continue;
            }
            if (urma_delete_jetty(ctx->jetty[i]) != URMA_SUCCESS) {
                return -1;
            }
            ctx->jetty[i] = NULL;
        }
        free(ctx->jetty);
        ctx->jetty = NULL;
    }
    free(ctx->ai_jettys);
    ctx->ai_jettys = NULL;

    if (ctx->jfr != NULL) {
        const uint32_t jfr_count =
            cfg->jettys / cfg->jettys_pre_jfr;
        for (i = 0; i < jfr_count; ++i) {
            if (ctx->jfr[i] == NULL) {
                continue;
            }
            if (urma_delete_jfr(ctx->jfr[i]) != URMA_SUCCESS) {
                return -1;
            }
            ctx->jfr[i] = NULL;
        }
        free(ctx->jfr);
        ctx->jfr = NULL;
    }

    if (ctx->jfc_r != NULL) {
        for (i = 0; i < cfg->jettys; ++i) {
            if (i > 0 &&
                (!cfg->pair_flag || cfg->type == PERFTEST_BW)) {
                break;
            }
            if (ctx->jfc_r[i] != NULL &&
                urma_delete_jfc(ctx->jfc_r[i]) != URMA_SUCCESS) {
                return -1;
            }
            ctx->jfc_r[i] = NULL;
        }
        free(ctx->jfc_r);
        ctx->jfc_r = NULL;
    }
    if (ctx->jfc_s != NULL) {
        for (i = 0; i < cfg->jettys; ++i) {
            if (i > 0 &&
                (!cfg->pair_flag || cfg->type == PERFTEST_BW)) {
                break;
            }
            if (ctx->jfc_s[i] != NULL &&
                urma_delete_jfc(ctx->jfc_s[i]) != URMA_SUCCESS) {
                return -1;
            }
            ctx->jfc_s[i] = NULL;
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
#define KVAF_CHECKED_PHASE_DISCONNECT 101U
#define KVAF_CHECKED_PHASE_UNIMPORT_SEG 102U

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

static int kvaf_sync_phase_status(
    const perftest_config_t *cfg, uint32_t phase, int local_status)
{
    uint32_t i;
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
            peer.phase != phase || peer.status != 0) {
            return -1;
        }
    }
    return local_status == 0 ? 0 : -1;
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
    if (kvaf_sync_phase_status(cfg, phase, local_status) == 0) {
        return 0;
    }
    kvaf_abort_control_sockets(cfg);
    return -1;
}

static int kvaf_destroy_partial_duplex_ctx_checked(
    perftest_context_t *ctx, perftest_config_t *cfg)
{
    int status = 0;
    const bool had_imported_jetty =
        ctx != NULL && ctx->import_tjetty != NULL;

    if (ctx == NULL || cfg == NULL) {
        return -1;
    }
    destroy_run_ctx(ctx);

    if (cfg->enable_async_import || cfg->enable_credit ||
        cfg->enable_user_tp || cfg->use_jfce) {
        /*
         * Those modes are outside the direct MVP and need additional checked
         * teardown APIs. Never silently fall back to the legacy void path.
         */
        return -1;
    }
    if (!had_imported_jetty) {
        return -1;
    }

    if (kvaf_disconnect_jettys_checked(ctx, cfg) != 0) {
        status = -1;
    }
    if (ctx->import_tjetty != NULL) {
        /*
         * A failed disconnect leaves the target pointer alive for reset.
         * Still enter the peer barrier so the other endpoint does not hang
         * forever in its own checked teardown.
         */
        status = -1;
    }
    if (had_imported_jetty &&
        kvaf_sync_phase_status(
            cfg, KVAF_CHECKED_PHASE_DISCONNECT, status) != 0) {
        return -1;
    }
    if (status != 0) {
        return -1;
    }
    status = kvaf_unimport_segments_checked(ctx);
    /*
     * Both endpoints must prove that their imported view of the peer MR is
     * gone before either endpoint unregisters its own local MR.
     */
    if (had_imported_jetty &&
        kvaf_sync_phase_status(
            cfg, KVAF_CHECKED_PHASE_UNIMPORT_SEG, status) != 0) {
        return -1;
    }
    if (status != 0) {
        return -1;
    }
    destroy_connection_info(ctx);
    free(ctx->remote_tp_info);
    ctx->remote_tp_info = NULL;
    free(ctx->local_tp_info);
    ctx->local_tp_info = NULL;
    if (kvaf_unregister_memory_checked(ctx, cfg) != 0) {
        return -1;
    }
    if (kvaf_delete_queues_checked(ctx, cfg) != 0) {
        return -1;
    }
    if (ctx->urma_ctx != NULL) {
        if (urma_delete_context(ctx->urma_ctx) != URMA_SUCCESS) {
            return -1;
        }
        ctx->urma_ctx = NULL;
    }
    return 0;
}

int kvaf_perftest_create_duplex_ctx_checked(
    perftest_context_t *ctx, perftest_config_t *cfg)
{
    int status;

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
    status = init_device(ctx, cfg);
    if (kvaf_sync_local_create_phase(
            cfg, KVAF_CHECKED_PHASE_INIT_DEVICE, status) != 0) {
        goto unsafe;
    }
    status = kvaf_create_duplex_queues_public(ctx, cfg);
    if (kvaf_sync_local_create_phase(
            cfg, KVAF_CHECKED_PHASE_CREATE_QUEUES, status) != 0) {
        goto unsafe;
    }
    status = register_mem(ctx, cfg);
    if (kvaf_sync_local_create_phase(
            cfg, KVAF_CHECKED_PHASE_REGISTER_MR, status) != 0) {
        goto unsafe;
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
    if (kvaf_sync_local_create_phase(
            cfg, KVAF_CHECKED_PHASE_EXCHANGE_DONE, 0) != 0) {
        goto unsafe;
    }

    status = import_seg_for_duplex(ctx, cfg);
    if (kvaf_sync_local_create_phase(
            cfg, KVAF_CHECKED_PHASE_IMPORT_SEG, status) != 0) {
        goto unsafe;
    }
    if (connect_jetty(ctx, cfg) != 0) {
        goto unsafe;
    }
    if (kvaf_sync_local_create_phase(
            cfg, KVAF_CHECKED_PHASE_CONNECT_DONE, 0) != 0) {
        goto unsafe;
    }
    status = create_run_ctx(ctx, cfg);
    if (kvaf_sync_local_create_phase(
            cfg, KVAF_CHECKED_PHASE_CREATE_RUN_CTX, status) != 0) {
        goto unsafe;
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
    if (ctx == NULL || cfg == NULL ||
        kvaf_sync_local_create_phase(
            cfg, KVAF_CHECKED_PHASE_PROVIDER_READY,
            local_status) != 0) {
        kvaf_abort_control_sockets(cfg);
        return KVAF_PERFTEST_CHECKED_CREATE_UNSAFE;
    }
    return KVAF_PERFTEST_CHECKED_OK;
}

int kvaf_perftest_destroy_duplex_ctx_checked(
    perftest_context_t *ctx, perftest_config_t *cfg)
{
    if (kvaf_destroy_partial_duplex_ctx_checked(ctx, cfg) != 0) {
        return KVAF_PERFTEST_CHECKED_DESTROY_UNSAFE;
    }
    return KVAF_PERFTEST_CHECKED_OK;
}
