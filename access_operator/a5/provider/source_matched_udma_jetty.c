#include "kvaf_source_matched_udma.h"

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/*
 * The upstream delete path removes the SQ lookup entry before issuing the
 * kernel delete.  liburma also frees jetty_cfg.jfr_cfg even when that ioctl
 * fails.  Both mutations make a second delete unsafe.  Keep the complete
 * source-matched implementation for creation and non-direct operations, but
 * replace the direct RM/non-shared delete with a failure-atomic wrapper.
 */
#ifndef KVAF_UDMA_JETTY_SOURCE
#error "KVAF_UDMA_JETTY_SOURCE must name udma_u_jetty.c"
#endif
#define udma_u_create_jetty kvaf_legacy_udma_u_create_jetty
#define udma_u_delete_jetty kvaf_legacy_udma_u_delete_jetty
#include KVAF_UDMA_JETTY_SOURCE
#undef udma_u_delete_jetty
#undef udma_u_create_jetty

#include "kvaf_source_matched_udma_cleanup.h"

struct kvaf_udma_jetty_state {
    struct udma_u_jetty *jetty;
    urma_context_t *owner_context;
    bool kernel_created;
    bool sq_created;
    bool doorbell_allocated;
    bool queue_inserted;
    bool jfc_flushed;
    bool orphan;
    struct kvaf_udma_jetty_state *next;
};

static pthread_mutex_t g_kvaf_jetty_lock = PTHREAD_MUTEX_INITIALIZER;
static struct kvaf_udma_jetty_state *g_kvaf_jetties;
static uint32_t g_kvaf_jetty_count;
static uint32_t g_kvaf_jetty_orphan_count;

static struct kvaf_udma_jetty_state *kvaf_find_jetty_locked(
    struct udma_u_jetty *jetty)
{
    struct kvaf_udma_jetty_state *state;

    for (state = g_kvaf_jetties; state != NULL; state = state->next) {
        if (state->jetty == jetty) {
            return state;
        }
    }
    return NULL;
}

static void kvaf_remove_jetty_locked(
    struct kvaf_udma_jetty_state *state)
{
    struct kvaf_udma_jetty_state **cursor = &g_kvaf_jetties;

    while (*cursor != NULL) {
        if (*cursor == state) {
            *cursor = state->next;
            --g_kvaf_jetty_count;
            if (state->orphan) {
                --g_kvaf_jetty_orphan_count;
            }
            return;
        }
        cursor = &(*cursor)->next;
    }
}

uint32_t kvaf_udma_source_live_jetty_count(void)
{
    uint32_t count;

    if (pthread_mutex_lock(&g_kvaf_jetty_lock) != 0) {
        return UINT32_MAX;
    }
    count = g_kvaf_jetty_count;
    (void)pthread_mutex_unlock(&g_kvaf_jetty_lock);
    return count;
}

uint32_t kvaf_udma_source_orphan_jetty_count(void)
{
    uint32_t count;

    if (pthread_mutex_lock(&g_kvaf_jetty_lock) != 0) {
        return UINT32_MAX;
    }
    count = g_kvaf_jetty_orphan_count;
    (void)pthread_mutex_unlock(&g_kvaf_jetty_lock);
    return count;
}

static int kvaf_delete_kernel_jetty_preserve_cfg(
    urma_jetty_t *public_jetty)
{
    urma_jfr_cfg_t *original_jfr_cfg;
    urma_jfr_cfg_t *command_jfr_cfg;
    int status;

    original_jfr_cfg = public_jetty->jetty_cfg.jfr_cfg;
    if (original_jfr_cfg == NULL) {
        return -1;
    }
    command_jfr_cfg = malloc(sizeof(*command_jfr_cfg));
    if (command_jfr_cfg == NULL) {
        return -1;
    }
    (void)memcpy(command_jfr_cfg, original_jfr_cfg,
        sizeof(*command_jfr_cfg));
    public_jetty->jetty_cfg.jfr_cfg = command_jfr_cfg;
    status = urma_cmd_delete_jetty(public_jetty);
    public_jetty->jetty_cfg.jfr_cfg = original_jfr_cfg;
    return status;
}

static int kvaf_create_direct_jetty_command(
    urma_context_t *context, struct udma_u_jetty *jetty,
    urma_jetty_cfg_t *cfg, bool *kernel_created)
{
    struct udma_create_jetty_resp response = {0};
    struct udma_create_jetty_ucmd command = {0};
    urma_cmd_udrv_priv_t udata = {0};
    int status;

    *kernel_created = false;
    if (cfg->jfs_cfg.priority >= UDMA_MAX_PRIORITY) {
        return -1;
    }
    command.buf_addr = (uintptr_t)jetty->sq.qbuf;
    command.buf_len = jetty->sq.qbuf_size;
    command.db_addr = (uintptr_t)jetty->sq.db.addr;
    command.jetty_addr = (uintptr_t)&jetty->sq;
    command.sqe_bb_cnt = jetty->sq.sqe_bb_cnt;
    command.pi_type = jetty->pi_type;
    command.jetty_type = jetty->jetty_type;
    command.non_pin = jetty->sq.cstm;
    command.is_hugepage = jetty->sq.hugepage != NULL;
#ifdef TGID_VA
    if (jetty->is_tgid_va) {
        command.is_db_tgid_va = true;
        command.tg_id = jetty->tgid;
    }
#endif
    udma_u_set_udata(&udata, &command, (uint32_t)sizeof(command),
        &response, (uint32_t)sizeof(response));
    status = urma_cmd_create_jetty(
        context, &jetty->base, cfg, &udata);
    if (status != 0) {
        return status;
    }
    *kernel_created = true;
#ifdef TGID_VA
    jetty->sq.dwqe_addr_va = response.tgid_va;
#endif
    status = udma_u_set_sq_by_resp(&jetty->sq, &response);
    if (status != 0) {
        return status;
    }
    jetty->sq.idx = jetty->base.jetty_id.id;
    jetty->sq.db.id = jetty->base.jetty_id.id;
    return 0;
}

static int kvaf_create_direct_sq_checked(
    struct kvaf_udma_jetty_state *state, urma_jfs_cfg_t *config)
{
    struct udma_u_jetty_queue *sq = &state->jetty->sq;
    uint32_t sqe_base_block_count;
    uint64_t sqe_bytes;

    sq->lock_free = config->flag.bs.lock_free;
    if (!sq->lock_free &&
        pthread_spin_init(&sq->lock, PTHREAD_PROCESS_PRIVATE) != 0) {
        return -1;
    }
    /* From this point even a failed allocation has a checked local phase. */
    state->sq_created = true;
    udma_u_init_sq_param(sq, config);
    if (config->max_sge == 0U) {
        return -1;
    }
    sqe_bytes = (uint64_t)SQE_WRITE_NOTIFY_CTL_LEN +
        ((uint64_t)config->max_sge - 1U) * UDMA_SGE_SIZE;
    sqe_base_block_count = (uint32_t)(sqe_bytes / UDMA_JFS_WQEBB + 1U);
    if (sqe_base_block_count > MAX_SQE_BB_NUM) {
        sqe_base_block_count = MAX_SQE_BB_NUM;
    }
    if (config->depth == 0U ||
        sqe_base_block_count > UINT32_MAX / config->depth) {
        return -1;
    }
    sq->sqe_bb_cnt = sqe_base_block_count;
    sq->max_sge_num = config->max_sge;
    return kvaf_udma_checked_alloc_queue_buffer(
        sq, sqe_base_block_count * config->depth,
        UDMA_JFS_WQEBB, UDMA_HW_PAGE_SIZE, true);
}

static int kvaf_cleanup_local_jetty_checked(
    struct kvaf_udma_jetty_state *state)
{
    struct udma_u_jetty *jetty = state->jetty;

    if (!state->orphan && !state->jfc_flushed) {
        struct udma_u_context *context = jetty->sq.ctx;
        urma_jfc_t *send_jfc = jetty->base.jetty_cfg.jfs_cfg.jfc;
        urma_jfc_t *receive_jfc = jetty->base.jetty_cfg.jfr_cfg == NULL
            ? NULL : jetty->base.jetty_cfg.jfr_cfg->jfc;

        if (context != NULL && context->set_ta_flush && send_jfc != NULL) {
            udma_u_clean_jfc(send_jfc, jetty->base.jetty_id.id);
        }
        if (context != NULL && context->set_ta_flush &&
            receive_jfc != NULL && receive_jfc != send_jfc) {
            udma_u_clean_jfc(receive_jfc, jetty->base.jetty_id.id);
        }
        state->jfc_flushed = true;
    }

    if (state->queue_inserted) {
        udma_u_jetty_queue_remove(
            jetty->sq.ctx, &jetty->sq, UDMA_U_JETTY_TBL);
        state->queue_inserted = false;
    }
    if (state->doorbell_allocated) {
        if (kvaf_udma_checked_free_doorbell(
                jetty->base.urma_ctx, &jetty->sq.db) != 0) {
            return -1;
        }
        state->doorbell_allocated = false;
    }
    if (state->sq_created) {
        if (kvaf_udma_checked_free_queue_buffer(&jetty->sq) != 0) {
            return -1;
        }
        if (!jetty->sq.lock_free &&
            pthread_spin_destroy(&jetty->sq.lock) != 0) {
            return -1;
        }
        state->sq_created = false;
    }
    remove_jetty_from_grp(jetty);
    return 0;
}

static void kvaf_free_local_jetty(
    struct kvaf_udma_jetty_state *state)
{
    urma_jfr_cfg_t *owned_jfr_cfg =
        state->jetty->base.jetty_cfg.jfr_cfg;

    free(state->jetty);
    free(owned_jfr_cfg);
}

int kvaf_udma_source_retry_orphan_jetties(urma_context_t *context)
{
    struct kvaf_udma_jetty_state *state;
    struct kvaf_udma_jetty_state *next;
    int result = 0;

    if (context == NULL || pthread_mutex_lock(&g_kvaf_jetty_lock) != 0) {
        return -1;
    }
    for (state = g_kvaf_jetties; state != NULL; state = next) {
        next = state->next;
        if (!state->orphan || state->owner_context != context) {
            continue;
        }
        if (state->kernel_created &&
            kvaf_delete_kernel_jetty_preserve_cfg(
                &state->jetty->base) != 0) {
            result = -1;
            continue;
        }
        state->kernel_created = false;
        if (kvaf_cleanup_local_jetty_checked(state) != 0) {
            result = -1;
            continue;
        }
        kvaf_remove_jetty_locked(state);
        kvaf_free_local_jetty(state);
        free(state);
    }
    (void)pthread_mutex_unlock(&g_kvaf_jetty_lock);
    return result;
}

urma_jetty_t *kvaf_udma_create_jetty_checked(
    urma_context_t *context, urma_jetty_cfg_t *cfg)
{
    struct kvaf_udma_jetty_state *state;
    struct udma_u_context *udma_context;
    struct udma_u_jetty *jetty;
    int status;

    state = calloc(1, sizeof(*state));
    jetty = calloc(1, sizeof(*jetty));
    if (state == NULL || jetty == NULL || context == NULL || cfg == NULL ||
        pthread_mutex_lock(&g_kvaf_jetty_lock) != 0) {
        free(state);
        free(jetty);
        return NULL;
    }
    state->jetty = jetty;
    state->owner_context = context;
    state->orphan = true;
    state->next = g_kvaf_jetties;
    g_kvaf_jetties = state;
    ++g_kvaf_jetty_count;
    ++g_kvaf_jetty_orphan_count;

    if (cfg->flag.bs.share_jfr != URMA_NO_SHARE_JFR ||
        cfg->jfr_cfg == NULL || cfg->jfs_cfg.jfc == NULL ||
        cfg->jfr_cfg->jfc == NULL || cfg->jetty_grp != NULL ||
        cfg->jfs_cfg.trans_mode != URMA_TM_RM ||
        cfg->jfr_cfg->trans_mode != URMA_TM_RM) {
        goto fail_local;
    }
    udma_context = to_udma_u_ctx(context);
    jetty->sq.ctx = udma_context;
    jetty->sq.sq_reserved = udma_context->sq_reserved;
    jetty->sq.trans_mode = URMA_TM_RM;
    status = kvaf_create_direct_sq_checked(state, &cfg->jfs_cfg);
    if (status != 0) {
        goto fail_local;
    }
    jetty->jfr = NULL; /* The direct H2D path never posts a receive WR. */
    jetty->jetty_type = UDMA_URMA_NORMAL_JETTY_TYPE;
    status = kvaf_create_direct_jetty_command(
        context, jetty, cfg, &state->kernel_created);
    if (status != 0) {
        goto fail_after_command;
    }
    jetty->sq.db.type = UDMA_MMAP_JETTY_DSQE;
    if (udma_u_alloc_db(context, &jetty->sq.db) != 0) {
        goto fail_after_command;
    }
    state->doorbell_allocated = true;
    jetty->sq.dwqe_addr = (void *)jetty->sq.db.addr;
    if (udma_u_jetty_queue_insert(
            udma_context, &jetty->sq, UDMA_U_JETTY_TBL) != 0) {
        goto fail_after_command;
    }
    state->queue_inserted = true;
    state->orphan = false;
    --g_kvaf_jetty_orphan_count;
    (void)pthread_mutex_unlock(&g_kvaf_jetty_lock);
    return &jetty->base;

fail_after_command:
    if (state->kernel_created &&
        kvaf_delete_kernel_jetty_preserve_cfg(&jetty->base) != 0) {
        /* No public handle escapes; the live count forces quarantine. */
        (void)pthread_mutex_unlock(&g_kvaf_jetty_lock);
        return NULL;
    }
    state->kernel_created = false;
fail_local:
    if (kvaf_cleanup_local_jetty_checked(state) != 0) {
        /* Retain every address and phase for the context-owned retry pass. */
        (void)pthread_mutex_unlock(&g_kvaf_jetty_lock);
        return NULL;
    }
    kvaf_remove_jetty_locked(state);
    kvaf_free_local_jetty(state);
    (void)pthread_mutex_unlock(&g_kvaf_jetty_lock);
    free(state);
    return NULL;
}

urma_status_t kvaf_udma_delete_jetty_checked(urma_jetty_t *public_jetty)
{
    struct kvaf_udma_jetty_state *state;
    struct udma_u_jetty *jetty;
    struct udma_u_jetty_queue *sq;
    urma_jfr_cfg_t *original_jfr_cfg;
    int status;

    if (public_jetty == NULL ||
        pthread_mutex_lock(&g_kvaf_jetty_lock) != 0) {
        return URMA_FAIL;
    }
    jetty = to_udma_u_jetty(public_jetty);
    state = kvaf_find_jetty_locked(jetty);
    if (state == NULL) {
        (void)pthread_mutex_unlock(&g_kvaf_jetty_lock);
        return URMA_FAIL;
    }

    sq = &jetty->sq;
    if (public_jetty->jetty_cfg.flag.bs.share_jfr !=
            URMA_NO_SHARE_JFR ||
        (sq->trans_mode == URMA_TM_RC && sq->tjetty != NULL)) {
        /* The current direct path is RM with a private JFR. */
        (void)pthread_mutex_unlock(&g_kvaf_jetty_lock);
        return URMA_FAIL;
    }
    original_jfr_cfg = public_jetty->jetty_cfg.jfr_cfg;
    if (original_jfr_cfg == NULL) {
        (void)pthread_mutex_unlock(&g_kvaf_jetty_lock);
        return URMA_FAIL;
    }
    /*
     * urma_cmd_delete_jetty always uninitializes the cfg.  Give it a
     * disposable deep-copy, then restore the original object for retry or
     * for the userspace cleanup that follows a successful command.
     */
    if (state->kernel_created) {
        status = kvaf_delete_kernel_jetty_preserve_cfg(public_jetty);
        if (status != 0) {
            (void)pthread_mutex_unlock(&g_kvaf_jetty_lock);
            return URMA_FAIL;
        }
        state->kernel_created = false;
    }

    if (kvaf_cleanup_local_jetty_checked(state) != 0) {
        /* Transport is gone; retry resumes only local queue/mapping phases. */
        (void)pthread_mutex_unlock(&g_kvaf_jetty_lock);
        return URMA_FAIL;
    }
    kvaf_remove_jetty_locked(state);
    (void)pthread_mutex_unlock(&g_kvaf_jetty_lock);

    kvaf_free_local_jetty(state);
    free(state);
    return URMA_SUCCESS;
}

urma_jetty_t *udma_u_create_jetty(
    urma_context_t *context, urma_jetty_cfg_t *cfg)
{
    return kvaf_udma_create_jetty_checked(context, cfg);
}

urma_status_t udma_u_delete_jetty(urma_jetty_t *jetty)
{
    return kvaf_udma_delete_jetty_checked(jetty);
}
