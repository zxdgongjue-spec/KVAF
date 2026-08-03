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
 * replace the direct RM delete with a failure-atomic wrapper. Both a private
 * JFR config and UMDK's mandatory UB shared-JFR form are supported without
 * ever interpreting the inactive member of urma_jetty_cfg_t's union.
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
#include "urma_cmd_tlv.h"

struct kvaf_udma_jetty_state {
    struct udma_u_jetty *jetty;
    urma_context_t *owner_context;
    bool kernel_created;
    bool kernel_delete_pending;
    uint32_t delete_async_events_reported;
    bool event_condition_initialized;
    bool event_mutex_initialized;
    bool sq_created;
    bool doorbell_allocated;
    bool queue_inserted;
    bool table_lock_unknown;
    bool jfc_flushed;
    bool orphan;
    struct kvaf_udma_jetty_state *next;
};

static pthread_mutex_t g_kvaf_jetty_lock = PTHREAD_MUTEX_INITIALIZER;
static struct kvaf_udma_jetty_state *g_kvaf_jetties;
static uint32_t g_kvaf_jetty_count;
static uint32_t g_kvaf_jetty_orphan_count;

static int kvaf_lock_jetty_registry_checked(void)
{
    const int status = pthread_mutex_lock(&g_kvaf_jetty_lock);

    return status == 0 ? 0 : kvaf_udma_pthread_cleanup_failed(status);
}

static int kvaf_unlock_jetty_registry_checked(void)
{
    const int status = pthread_mutex_unlock(&g_kvaf_jetty_lock);

    return status == 0 ? 0 : kvaf_udma_pthread_cleanup_failed(status);
}

static int kvaf_unlock_jetty_table_checked(
    struct kvaf_udma_jetty_state *state,
    struct udma_u_context *context)
{
    if (pthread_rwlock_unlock(&context->jetty_table_lock) == 0) {
        return 0;
    }
    /* Never retry a lock whose ownership after unlock is unknowable. */
    state->table_lock_unknown = true;
    kvaf_udma_source_mark_cleanup_unknown();
    return -1;
}

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

uint32_t kvaf_udma_source_live_jetty_count_for_context(
    const urma_context_t *context)
{
    const struct kvaf_udma_jetty_state *state;
    uint32_t count = 0;

    if (context == NULL || pthread_mutex_lock(&g_kvaf_jetty_lock) != 0) {
        return UINT32_MAX;
    }
    for (state = g_kvaf_jetties; state != NULL; state = state->next) {
        if (state->owner_context == context) {
            ++count;
        }
    }
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

static int kvaf_delete_kernel_jetty_nonblocking(
    struct kvaf_udma_jetty_state *state)
{
    urma_jetty_t *jetty;
    uint32_t events_acked;
    enum kvaf_udma_destroy_decision decision;

    if (state == NULL || state->jetty == NULL ||
        state->owner_context == NULL) {
        return kvaf_udma_cleanup_invariant_failed();
    }
    jetty = &state->jetty->base;
    decision = kvaf_udma_destroy_decide(
        state->kernel_created, state->kernel_delete_pending,
        0U, state->delete_async_events_reported, 0U, 0U, false);
    if (decision == KVAF_UDMA_DESTROY_FAIL_UNSAFE) {
        errno = EUCLEAN;
        kvaf_udma_source_mark_cleanup_unknown();
        return -1;
    }
    if (decision == KVAF_UDMA_DESTROY_RELEASE_LOCAL) {
        state->kernel_delete_pending = false;
        return 0;
    }
    if (decision == KVAF_UDMA_DESTROY_ISSUE_IOCTL) {
        urma_cmd_delete_jetty_t command = {0};
        int status;

        if (jetty->urma_ctx == NULL ||
            jetty->urma_ctx != state->owner_context ||
            jetty->urma_ctx->dev_fd < 0) {
            return kvaf_udma_cleanup_invariant_failed();
        }
        command.in.handle = jetty->handle;
        errno = 0;
        status = urma_ioctl_delete_jetty(
            jetty->urma_ctx->dev_fd, &command);
        if (status != 0) {
            if (!kvaf_udma_destroy_ioctl_retry_is_proven(
                    status, errno)) {
                kvaf_udma_source_mark_cleanup_unknown();
            }
            return status;
        }
        /* Keep the original cfg and every local queue mapping alive while
         * the same sidecar drains the events reported by this one ioctl. */
        state->kernel_created = false;
        state->kernel_delete_pending = true;
        state->delete_async_events_reported =
            command.out.async_events_reported;
    }

    if (kvaf_udma_snapshot_event_counters_checked(
            &jetty->event_mutex, NULL, &jetty->async_events_acked,
            NULL, &events_acked) != 0) {
        return -1;
    }
    decision = kvaf_udma_destroy_decide(
        state->kernel_created, state->kernel_delete_pending,
        0U, state->delete_async_events_reported, 0U, events_acked,
        false);
    if (decision == KVAF_UDMA_DESTROY_FAIL_UNSAFE) {
        errno = EUCLEAN;
        kvaf_udma_source_mark_cleanup_unknown();
        return -1;
    }
    if (decision == KVAF_UDMA_DESTROY_DRAIN_ASYNC) {
        if (kvaf_udma_drain_async_events_bounded(
                jetty->urma_ctx) != 0) {
            return -1;
        }
        if (kvaf_udma_snapshot_event_counters_checked(
                &jetty->event_mutex, NULL,
                &jetty->async_events_acked, NULL,
                &events_acked) != 0) {
            return -1;
        }
        decision = kvaf_udma_destroy_decide(
            state->kernel_created, state->kernel_delete_pending,
            0U, state->delete_async_events_reported, 0U,
            events_acked, true);
        if (decision == KVAF_UDMA_DESTROY_FAIL_UNSAFE) {
            errno = EUCLEAN;
            kvaf_udma_source_mark_cleanup_unknown();
            return -1;
        }
        if (decision == KVAF_UDMA_DESTROY_RETRY_LATER) {
            errno = EAGAIN;
            return URMA_EAGAIN;
        }
    }
    if (decision != KVAF_UDMA_DESTROY_RELEASE_LOCAL) {
        errno = EUCLEAN;
        kvaf_udma_source_mark_cleanup_unknown();
        return -1;
    }
    state->kernel_delete_pending = false;
    return 0;
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
    command.pi_type = jetty->sq.pi_type;
    command.jetty_type = jetty->jetty_type;
    command.non_pin = jetty->sq.cstm;
    command.dtu_en = jetty->sq.dtu_en;
    command.is_hugepage = jetty->sq.hugepage != NULL;
    udma_u_set_udata(&udata, &command, (uint32_t)sizeof(command),
        &response, (uint32_t)sizeof(response));
    status = urma_cmd_create_jetty(
        context, &jetty->base, cfg, &udata);
    if (status != 0) {
        return status;
    }
    *kernel_created = true;
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
    uint32_t max_sge_count;
    uint32_t sqe_base_block_count;

    sq->lock_free = config->flag.bs.lock_free;
    if (!sq->lock_free &&
        pthread_spin_init(&sq->lock, PTHREAD_PROCESS_PRIVATE) != 0) {
        return -1;
    }
    /* From this point even a failed allocation has a checked local phase. */
    state->sq_created = true;
    udma_u_init_sq_param(sq, config);
    max_sge_count = get_max_sge_num(
        config->max_sge, sq->max_inline_size);
    sqe_base_block_count = sq_cal_wqebb_num(
        SQE_WRITE_NOTIFY_CTL_LEN, max_sge_count, UDMA_JFS_WQEBB);
    if (sqe_base_block_count > MAX_SQE_BB_NUM) {
        sqe_base_block_count = MAX_SQE_BB_NUM;
    }
    if (config->depth == 0U ||
        sqe_base_block_count > UINT32_MAX / config->depth) {
        return -1;
    }
    sq->sqe_bb_cnt = sqe_base_block_count;
    sq->max_sge_num = config->max_sge;
    sq->aligned_size = UDMA_HW_PAGE_SIZE;
    return kvaf_udma_checked_alloc_queue_buffer(
        sq, sqe_base_block_count * config->depth,
        UDMA_JFS_WQEBB, UDMA_HW_PAGE_SIZE, true);
}

static int kvaf_insert_jetty_table_checked(
    struct kvaf_udma_jetty_state *state)
{
    struct udma_u_jetty *jetty = state->jetty;
    struct udma_u_context *context = jetty->sq.ctx;
    struct udma_u_jetty **array;
    uint32_t table_id;
    uint32_t slot;

    if (context == NULL || context->jettys_in_tbl_shift == 0U ||
        context->jettys_in_tbl == 0U ||
        (context->jettys_in_tbl &
         (context->jettys_in_tbl - 1U)) != 0U) {
        return -1;
    }
    table_id = jetty->sq.idx >> context->jettys_in_tbl_shift;
    slot = jetty->sq.idx & (context->jettys_in_tbl - 1U);
    if (table_id >= UDMA_JETTY_TABLE_NUM ||
        pthread_rwlock_wrlock(&context->jetty_table_lock) != 0) {
        return -1;
    }
    array = context->jetty_table[table_id].jetty_array;
    if (context->jetty_table[table_id].refcnt == 0) {
        if (array != NULL) {
            (void)kvaf_unlock_jetty_table_checked(state, context);
            return -1;
        }
        array = calloc(context->jettys_in_tbl, sizeof(*array));
        if (array == NULL) {
            (void)kvaf_unlock_jetty_table_checked(state, context);
            return -1;
        }
        context->jetty_table[table_id].jetty_array = array;
    } else if (array == NULL) {
        (void)kvaf_unlock_jetty_table_checked(state, context);
        return -1;
    }
    if (array[slot] != NULL) {
        if (context->jetty_table[table_id].refcnt == 0) {
            free(array);
            context->jetty_table[table_id].jetty_array = NULL;
        }
        (void)kvaf_unlock_jetty_table_checked(state, context);
        return -1;
    }
    array[slot] = jetty;
    ++context->jetty_table[table_id].refcnt;
    state->queue_inserted = true;
    if (kvaf_unlock_jetty_table_checked(state, context) != 0) {
        return -1;
    }
    return 0;
}

static int kvaf_remove_jetty_table_checked(
    struct kvaf_udma_jetty_state *state)
{
    struct udma_u_jetty *jetty = state->jetty;
    struct udma_u_context *context = jetty->sq.ctx;
    struct udma_u_jetty **array;
    uint32_t table_id;
    uint32_t slot;

    if (!state->queue_inserted) {
        return 0;
    }
    if (state->table_lock_unknown || context == NULL ||
        context->jettys_in_tbl_shift == 0U ||
        context->jettys_in_tbl == 0U) {
        return -1;
    }
    table_id = jetty->sq.idx >> context->jettys_in_tbl_shift;
    slot = jetty->sq.idx & (context->jettys_in_tbl - 1U);
    if (table_id >= UDMA_JETTY_TABLE_NUM ||
        pthread_rwlock_wrlock(&context->jetty_table_lock) != 0) {
        return -1;
    }
    array = context->jetty_table[table_id].jetty_array;
    if (context->jetty_table[table_id].refcnt <= 0 ||
        array == NULL || array[slot] != jetty) {
        (void)kvaf_unlock_jetty_table_checked(state, context);
        return -1;
    }
    array[slot] = NULL;
    --context->jetty_table[table_id].refcnt;
    if (context->jetty_table[table_id].refcnt == 0) {
        free(array);
        context->jetty_table[table_id].jetty_array = NULL;
    }
    if (kvaf_unlock_jetty_table_checked(state, context) != 0) {
        return -1;
    }
    state->queue_inserted = false;
    return 0;
}

static int kvaf_cleanup_local_jetty_checked(
    struct kvaf_udma_jetty_state *state)
{
    struct udma_u_jetty *jetty;

    if (state == NULL || state->jetty == NULL ||
        state->owner_context == NULL) {
        return kvaf_udma_cleanup_invariant_failed();
    }
    jetty = state->jetty;
    if ((state->sq_created || state->doorbell_allocated ||
         state->queue_inserted) &&
        (jetty->sq.ctx == NULL ||
         jetty->sq.ctx != to_udma_u_ctx(state->owner_context))) {
        return kvaf_udma_cleanup_invariant_failed();
    }
    if ((state->kernel_created || state->kernel_delete_pending ||
         state->doorbell_allocated ||
         state->event_condition_initialized ||
         state->event_mutex_initialized) &&
        jetty->base.urma_ctx != state->owner_context) {
        return kvaf_udma_cleanup_invariant_failed();
    }

    if (!state->orphan && !state->jfc_flushed) {
        struct udma_u_context *context = jetty->sq.ctx;
        urma_jfc_t *send_jfc = jetty->base.jetty_cfg.jfs_cfg.jfc;
        urma_jfc_t *receive_jfc;

        if (jetty->base.jetty_cfg.flag.bs.share_jfr ==
                URMA_SHARE_JFR) {
            receive_jfc = jetty->base.jetty_cfg.shared.jfc;
            if (receive_jfc == NULL &&
                jetty->base.jetty_cfg.shared.jfr != NULL) {
                receive_jfc =
                    jetty->base.jetty_cfg.shared.jfr->jfr_cfg.jfc;
            }
        } else {
            receive_jfc = jetty->base.jetty_cfg.jfr_cfg == NULL
                ? NULL : jetty->base.jetty_cfg.jfr_cfg->jfc;
        }

        if (context != NULL && send_jfc != NULL) {
            udma_u_clean_jfc(send_jfc, jetty->base.jetty_id.id);
        }
        if (context != NULL && receive_jfc != NULL &&
            receive_jfc != send_jfc) {
            udma_u_clean_jfc(receive_jfc, jetty->base.jetty_id.id);
        }
        state->jfc_flushed = true;
    }

    if (state->queue_inserted) {
        if (kvaf_remove_jetty_table_checked(state) != 0) {
            return -1;
        }
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
        if (!jetty->sq.lock_free) {
            const int spin_status =
                pthread_spin_destroy(&jetty->sq.lock);

            if (spin_status != 0) {
                return kvaf_udma_pthread_cleanup_failed(spin_status);
            }
        }
        state->sq_created = false;
    }
    if (kvaf_udma_destroy_event_sync_checked(
            &jetty->base.event_mutex, &jetty->base.event_cond,
            &state->event_condition_initialized,
            &state->event_mutex_initialized) != 0) {
        return -1;
    }
    remove_jetty_from_grp(jetty);
    return 0;
}

static void kvaf_free_local_jetty(
    struct kvaf_udma_jetty_state *state)
{
    urma_jfr_cfg_t *owned_jfr_cfg = NULL;

    if (state->jetty->base.jetty_cfg.flag.bs.share_jfr ==
            URMA_NO_SHARE_JFR) {
        owned_jfr_cfg = state->jetty->base.jetty_cfg.jfr_cfg;
    }

    free(state->jetty);
    free(owned_jfr_cfg);
}

int kvaf_udma_source_retry_orphan_jetties(urma_context_t *context)
{
    struct kvaf_udma_jetty_state *state;
    struct kvaf_udma_jetty_state *next;
    int result = 0;
    int saved_errno = 0;

    if (context == NULL) {
        return kvaf_udma_cleanup_invariant_failed();
    }
    if (kvaf_lock_jetty_registry_checked() != 0) {
        return -1;
    }
    for (state = g_kvaf_jetties; state != NULL; state = next) {
        next = state->next;
        if (!state->orphan || state->owner_context != context) {
            continue;
        }
        if ((state->kernel_created || state->kernel_delete_pending) &&
            kvaf_delete_kernel_jetty_nonblocking(state) != 0) {
            result = -1;
            kvaf_udma_preserve_cleanup_errno(&saved_errno);
            continue;
        }
        errno = 0;
        if (kvaf_cleanup_local_jetty_checked(state) != 0) {
            (void)kvaf_udma_classify_cleanup_failure();
            result = -1;
            kvaf_udma_preserve_cleanup_errno(&saved_errno);
            continue;
        }
        kvaf_remove_jetty_locked(state);
        kvaf_free_local_jetty(state);
        free(state);
    }
    if (kvaf_unlock_jetty_registry_checked() != 0) {
        return -1;
    }
    if (result != 0) {
        errno = saved_errno == 0 ? EIO : saved_errno;
    }
    return result;
}

urma_jetty_t *kvaf_udma_create_jetty_checked(
    urma_context_t *context, urma_jetty_cfg_t *cfg)
{
    struct kvaf_udma_jetty_state *state;
    struct udma_u_context *udma_context;
    struct udma_u_jetty *jetty;
    urma_jfc_t *receive_jfc;
    bool shared_jfr;
    int status;

    state = calloc(1, sizeof(*state));
    jetty = calloc(1, sizeof(*jetty));
    if (state == NULL || jetty == NULL || context == NULL || cfg == NULL) {
        free(state);
        free(jetty);
        return NULL;
    }
    if (kvaf_lock_jetty_registry_checked() != 0) {
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

    shared_jfr = cfg->flag.bs.share_jfr == URMA_SHARE_JFR;
    if (cfg->flag.bs.non_blocking || cfg->jfs_cfg.jfc == NULL ||
        cfg->jfs_cfg.jfc->urma_ctx != context ||
        cfg->jetty_grp != NULL ||
        cfg->jfs_cfg.trans_mode != URMA_TM_RM) {
        goto fail_local;
    }
    if (shared_jfr) {
        if (cfg->shared.jfr == NULL ||
            cfg->shared.jfr->urma_ctx != context ||
            cfg->shared.jfr->jfr_cfg.trans_mode != URMA_TM_RM ||
            (cfg->shared.jfc != NULL &&
             cfg->shared.jfc->urma_ctx != context)) {
            goto fail_local;
        }
        receive_jfc = cfg->shared.jfc != NULL
            ? cfg->shared.jfc : cfg->shared.jfr->jfr_cfg.jfc;
    } else {
        if (cfg->jfr_cfg == NULL || cfg->jfr_cfg->jfc == NULL ||
            cfg->jfr_cfg->jfc->urma_ctx != context ||
            cfg->jfr_cfg->trans_mode != URMA_TM_RM) {
            goto fail_local;
        }
        receive_jfc = cfg->jfr_cfg->jfc;
    }
    if (receive_jfc == NULL || receive_jfc->urma_ctx != context) {
        goto fail_local;
    }
    udma_context = to_udma_u_ctx(context);
    if (udma_context->dtu_enable || udma_context->sq_reserved ||
        udma_context->jettys_in_tbl_shift == 0U ||
        udma_context->jettys_in_tbl == 0U) {
        goto fail_local;
    }
    jetty->sq.ctx = udma_context;
    jetty->sq.sq_reserved = false;
    jetty->sq.dtu_en = false;
    jetty->sq.is_jetty = true;
    jetty->sq.aligned_size = UDMA_HW_PAGE_SIZE;
    jetty->sq.trans_mode = URMA_TM_RM;
    status = kvaf_create_direct_sq_checked(state, &cfg->jfs_cfg);
    if (status != 0) {
        goto fail_local;
    }
    jetty->jfr = shared_jfr
        ? to_udma_u_jfr(cfg->shared.jfr) : NULL;
    jetty->jetty_type = UDMA_URMA_NORMAL_JETTY_TYPE;
    status = kvaf_create_direct_jetty_command(
        context, jetty, cfg, &state->kernel_created);
    if (state->kernel_created) {
        /* urma_cmd_create_jetty initializes both event primitives before
         * returning the kernel handle.  A later userspace SQ setup failure
         * still has to destroy those primitives after the kernel object is
         * deleted; recording them only on complete success leaked them. */
        state->event_condition_initialized = true;
        state->event_mutex_initialized = true;
    }
    if (status != 0) {
        goto fail_after_command;
    }
    jetty->sq.db.type = UDMA_MMAP_JETTY_DSQE;
    if (udma_u_alloc_db(context, &jetty->sq.db) != 0) {
        goto fail_after_command;
    }
    state->doorbell_allocated = true;
    jetty->sq.dwqe_addr = (void *)jetty->sq.db.addr;
    if (kvaf_insert_jetty_table_checked(state) != 0) {
        goto fail_after_command;
    }
    state->orphan = false;
    --g_kvaf_jetty_orphan_count;
    if (kvaf_unlock_jetty_registry_checked() != 0) {
        /* Keep the non-published handle under the orphan sidecar; exposing it
         * after a registry unlock ambiguity would permit unsafe reuse. */
        state->orphan = true;
        ++g_kvaf_jetty_orphan_count;
        return NULL;
    }
    return &jetty->base;

fail_after_command:
    if ((state->kernel_created || state->kernel_delete_pending) &&
        kvaf_delete_kernel_jetty_nonblocking(state) != 0) {
        /* No public handle escapes; the live count forces quarantine. */
        (void)kvaf_unlock_jetty_registry_checked();
        return NULL;
    }
fail_local:
    errno = 0;
    if (kvaf_cleanup_local_jetty_checked(state) != 0) {
        (void)kvaf_udma_classify_cleanup_failure();
        /* Retain every address and phase for the context-owned retry pass. */
        (void)kvaf_unlock_jetty_registry_checked();
        return NULL;
    }
    kvaf_remove_jetty_locked(state);
    kvaf_free_local_jetty(state);
    (void)kvaf_unlock_jetty_registry_checked();
    free(state);
    return NULL;
}

urma_status_t kvaf_udma_delete_jetty_checked(urma_jetty_t *public_jetty)
{
    struct kvaf_udma_jetty_state *state;
    struct udma_u_jetty *jetty;
    struct udma_u_jetty_queue *sq;
    int status;

    if (public_jetty == NULL) {
        (void)kvaf_udma_cleanup_invariant_failed();
        return URMA_FAIL;
    }
    if (kvaf_lock_jetty_registry_checked() != 0) {
        return URMA_FAIL;
    }
    jetty = to_udma_u_jetty(public_jetty);
    state = kvaf_find_jetty_locked(jetty);
    if (state == NULL || state->owner_context == NULL ||
        public_jetty->urma_ctx != state->owner_context ||
        jetty->sq.ctx != to_udma_u_ctx(state->owner_context)) {
        (void)kvaf_udma_cleanup_invariant_failed();
        (void)kvaf_unlock_jetty_registry_checked();
        return URMA_FAIL;
    }

    sq = &jetty->sq;
    if (public_jetty->jetty_cfg.flag.bs.non_blocking ||
        (sq->trans_mode == URMA_TM_RC && sq->tjetty != NULL)) {
        /* The current direct path is blocking RM. */
        (void)kvaf_udma_cleanup_invariant_failed();
        (void)kvaf_unlock_jetty_registry_checked();
        return URMA_FAIL;
    }
    if ((public_jetty->jetty_cfg.flag.bs.share_jfr ==
            URMA_SHARE_JFR &&
         (public_jetty->jetty_cfg.shared.jfr == NULL ||
          jetty->jfr != to_udma_u_jfr(
              public_jetty->jetty_cfg.shared.jfr))) ||
        (public_jetty->jetty_cfg.flag.bs.share_jfr ==
            URMA_NO_SHARE_JFR &&
         public_jetty->jetty_cfg.jfr_cfg == NULL)) {
        (void)kvaf_udma_cleanup_invariant_failed();
        (void)kvaf_unlock_jetty_registry_checked();
        return URMA_FAIL;
    }
    /*
     * A successful private-JFR delete uninitializes its cfg, so that form
     * receives a disposable deep-copy. The shared-JFR union member is never
     * freed or rewritten by urma_uninit_jetty_cfg.
     */
    if (state->kernel_created || state->kernel_delete_pending) {
        status = kvaf_delete_kernel_jetty_nonblocking(state);
        if (status != 0) {
            (void)kvaf_unlock_jetty_registry_checked();
            return URMA_FAIL;
        }
    }

    errno = 0;
    if (kvaf_cleanup_local_jetty_checked(state) != 0) {
        (void)kvaf_udma_classify_cleanup_failure();
        /* Transport is gone; retry resumes only local queue/mapping phases. */
        (void)kvaf_unlock_jetty_registry_checked();
        return URMA_FAIL;
    }
    kvaf_remove_jetty_locked(state);
    if (kvaf_unlock_jetty_registry_checked() != 0) {
        return URMA_FAIL;
    }

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
