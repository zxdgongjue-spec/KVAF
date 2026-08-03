#include "kvaf_source_matched_udma.h"

#include <errno.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/* Track every direct-path JFC so context teardown cannot run underneath it. */
#ifndef KVAF_UDMA_JFC_SOURCE
#error "KVAF_UDMA_JFC_SOURCE must name udma_u_jfc.c"
#endif
#define udma_u_create_jfc kvaf_legacy_udma_u_create_jfc
#define udma_u_delete_jfc kvaf_legacy_udma_u_delete_jfc
#include KVAF_UDMA_JFC_SOURCE
#undef udma_u_delete_jfc
#undef udma_u_create_jfc

#include "kvaf_source_matched_udma_cleanup.h"
#include "urma_cmd_tlv.h"

struct kvaf_udma_jfc_state {
    struct udma_u_jfc *jfc;
    urma_context_t *owner_context;
    urma_context_t *created_owner_context;
    uint64_t created_handle;
    uint32_t created_id;
    int created_dev_fd;
    bool kernel_created;
    bool kernel_delete_pending;
    uint32_t delete_comp_events_reported;
    uint32_t delete_async_events_reported;
    bool event_condition_initialized;
    bool event_mutex_initialized;
    bool sw_db_allocated;
    bool cq_created;
    bool orphan;
    struct kvaf_udma_jfc_state *next;
};

static pthread_mutex_t g_kvaf_jfc_lock = PTHREAD_MUTEX_INITIALIZER;
static struct kvaf_udma_jfc_state *g_kvaf_jfcs;
static uint32_t g_kvaf_jfc_count;
static uint32_t g_kvaf_jfc_orphan_count;

static int kvaf_lock_jfc_registry_checked(void)
{
    const int status = pthread_mutex_lock(&g_kvaf_jfc_lock);

    return status == 0 ? 0 : kvaf_udma_pthread_cleanup_failed(status);
}

static int kvaf_unlock_jfc_registry_checked(void)
{
    const int status = pthread_mutex_unlock(&g_kvaf_jfc_lock);

    return status == 0 ? 0 : kvaf_udma_pthread_cleanup_failed(status);
}

static int kvaf_delete_kernel_jfc_nonblocking(
    struct kvaf_udma_jfc_state *state)
{
    urma_jfc_t *jfc;
    uint32_t comp_events_acked;
    uint32_t async_events_acked;
    enum kvaf_udma_destroy_decision decision;

    if (state == NULL || state->jfc == NULL ||
        state->owner_context == NULL) {
        return kvaf_udma_cleanup_invariant_failed();
    }
    jfc = &state->jfc->base;
    decision = kvaf_udma_destroy_decide(
        state->kernel_created, state->kernel_delete_pending,
        state->delete_comp_events_reported,
        state->delete_async_events_reported, 0U, 0U, false);
    if (decision == KVAF_UDMA_DESTROY_FAIL_UNSAFE &&
        state->kernel_created && state->kernel_delete_pending) {
        errno = EUCLEAN;
        kvaf_udma_source_mark_cleanup_unknown();
        return -1;
    }
    if (decision == KVAF_UDMA_DESTROY_RELEASE_LOCAL) {
        state->kernel_delete_pending = false;
        return 0;
    }
    if (decision == KVAF_UDMA_DESTROY_ISSUE_IOCTL) {
        urma_cmd_delete_jfc_t command = {0};
        int status;

        if (jfc->urma_ctx == NULL ||
            jfc->urma_ctx != state->owner_context ||
            jfc->urma_ctx->dev_fd < 0) {
            return kvaf_udma_cleanup_invariant_failed();
        }
        command.in.handle = jfc->handle;
        errno = 0;
        status = urma_ioctl_delete_jfc(
            jfc->urma_ctx->dev_fd, &command);
        /* Preserve the exact pinned UMDK destroy convention. */
        if (status == EIO) {
            status = 0;
        }
        if (status != 0) {
            if (!kvaf_udma_destroy_ioctl_retry_is_proven(
                    status, errno)) {
                kvaf_udma_source_mark_cleanup_unknown();
            }
            return status;
        }
        state->kernel_created = false;
        state->kernel_delete_pending = true;
        state->delete_comp_events_reported =
            command.out.comp_events_reported;
        state->delete_async_events_reported =
            command.out.async_events_reported;
    }

    if (kvaf_udma_snapshot_event_counters_checked(
            &jfc->event_mutex, &jfc->comp_events_acked,
            &jfc->async_events_acked, &comp_events_acked,
            &async_events_acked) != 0) {
        return -1;
    }
    decision = kvaf_udma_destroy_decide(
        state->kernel_created, state->kernel_delete_pending,
        state->delete_comp_events_reported,
        state->delete_async_events_reported,
        comp_events_acked, async_events_acked, false);
    if (decision == KVAF_UDMA_DESTROY_FAIL_UNSAFE) {
        errno = EUCLEAN;
        kvaf_udma_source_mark_cleanup_unknown();
        return -1;
    }
    if (decision == KVAF_UDMA_DESTROY_DRAIN_ASYNC) {
        if (kvaf_udma_drain_async_events_bounded(jfc->urma_ctx) != 0) {
            return -1;
        }
        if (kvaf_udma_snapshot_event_counters_checked(
                &jfc->event_mutex, &jfc->comp_events_acked,
                &jfc->async_events_acked, &comp_events_acked,
                &async_events_acked) != 0) {
            return -1;
        }
        decision = kvaf_udma_destroy_decide(
            state->kernel_created, state->kernel_delete_pending,
            state->delete_comp_events_reported,
            state->delete_async_events_reported,
            comp_events_acked, async_events_acked, true);
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

static struct kvaf_udma_jfc_state *kvaf_find_jfc_locked(
    struct udma_u_jfc *jfc)
{
    struct kvaf_udma_jfc_state *state;

    for (state = g_kvaf_jfcs; state != NULL; state = state->next) {
        if (state->jfc == jfc) {
            return state;
        }
    }
    return NULL;
}

int kvaf_udma_source_snapshot_jfc_identity(
    const urma_context_t *expected_context, const urma_jfc_t *public_jfc,
    struct kvaf_udma_jfc_identity *identity)
{
    struct kvaf_udma_jfc_state *state;
    int lock_status;
    struct kvaf_udma_jfc_identity snapshot = {
        .abi_version = KVAF_UDMA_JFC_IDENTITY_ABI_VERSION,
        .structure_size = sizeof(snapshot),
        .created_dev_fd = -1,
        .current_dev_fd = -1,
    };

    if (expected_context == NULL || public_jfc == NULL || identity == NULL ||
        identity->structure_size < sizeof(*identity)) {
        errno = EINVAL;
        return -1;
    }
    lock_status = pthread_mutex_trylock(&g_kvaf_jfc_lock);
    if (lock_status != 0) {
        errno = lock_status;
        return -1;
    }
    state = kvaf_find_jfc_locked(
        to_udma_u_jfc((urma_jfc_t *)public_jfc));
    if (state != NULL) {
        snapshot.flags |= KVAF_UDMA_JFC_IDENTITY_REGISTRY_FOUND;
        snapshot.current_handle = public_jfc->handle;
        snapshot.current_id = public_jfc->jfc_id.id;
        if (public_jfc->urma_ctx != NULL) {
            snapshot.current_dev_fd = public_jfc->urma_ctx->dev_fd;
        }
        snapshot.created_handle = state->created_handle;
        snapshot.created_id = state->created_id;
        snapshot.created_dev_fd = state->created_dev_fd;
        if (state->kernel_created) {
            snapshot.flags |= KVAF_UDMA_JFC_IDENTITY_KERNEL_CREATED;
        }
        if (!state->kernel_delete_pending) {
            snapshot.flags |= KVAF_UDMA_JFC_IDENTITY_DELETE_IDLE;
        }
        if (!state->orphan) {
            snapshot.flags |= KVAF_UDMA_JFC_IDENTITY_NOT_ORPHAN;
        }
        if (state->owner_context == expected_context) {
            snapshot.flags |=
                KVAF_UDMA_JFC_IDENTITY_OWNER_CONTEXT_MATCH;
        }
        if (public_jfc->urma_ctx == expected_context) {
            snapshot.flags |=
                KVAF_UDMA_JFC_IDENTITY_PUBLIC_CONTEXT_MATCH;
        }
        if (state->created_owner_context == expected_context) {
            snapshot.flags |=
                KVAF_UDMA_JFC_IDENTITY_CREATED_CONTEXT_MATCH;
        }
        if (state->created_handle != 0U &&
            state->created_handle == public_jfc->handle) {
            snapshot.flags |= KVAF_UDMA_JFC_IDENTITY_HANDLE_MATCH;
        }
        if (state->created_id == public_jfc->jfc_id.id) {
            snapshot.flags |= KVAF_UDMA_JFC_IDENTITY_ID_MATCH;
        }
        if (state->created_dev_fd >= 0 &&
            state->created_dev_fd == expected_context->dev_fd &&
            state->created_dev_fd == snapshot.current_dev_fd) {
            snapshot.flags |= KVAF_UDMA_JFC_IDENTITY_DEV_FD_MATCH;
        }
    }
    lock_status = pthread_mutex_unlock(&g_kvaf_jfc_lock);
    if (lock_status != 0) {
        errno = lock_status;
        return -1;
    }
    *identity = snapshot;
    return 0;
}

static void kvaf_remove_jfc_locked(struct kvaf_udma_jfc_state *state)
{
    struct kvaf_udma_jfc_state **cursor = &g_kvaf_jfcs;

    while (*cursor != NULL) {
        if (*cursor == state) {
            *cursor = state->next;
            --g_kvaf_jfc_count;
            if (state->orphan) {
                --g_kvaf_jfc_orphan_count;
            }
            return;
        }
        cursor = &(*cursor)->next;
    }
}

static int kvaf_cleanup_local_jfc_checked(
    struct kvaf_udma_jfc_state *state)
{
    struct udma_u_jfc *jfc;
    struct udma_u_context *context;

    if (state == NULL || state->jfc == NULL ||
        state->owner_context == NULL) {
        return kvaf_udma_cleanup_invariant_failed();
    }
    jfc = state->jfc;
    context = to_udma_u_ctx(state->owner_context);
    if ((state->sw_db_allocated || state->cq_created) &&
        jfc->cq.ctx != context) {
        return kvaf_udma_cleanup_invariant_failed();
    }
    if ((state->kernel_created || state->kernel_delete_pending ||
         state->event_condition_initialized ||
         state->event_mutex_initialized) &&
        jfc->base.urma_ctx != state->owner_context) {
        return kvaf_udma_cleanup_invariant_failed();
    }

    if (state->sw_db_allocated) {
        if (kvaf_udma_checked_free_sw_db(
                context, jfc->sw_db, UDMA_JFC_TYPE_DB) != 0) {
            return -1;
        }
        jfc->sw_db = NULL;
        state->sw_db_allocated = false;
    }
    if (state->cq_created) {
        if (kvaf_udma_checked_free_queue_buffer(&jfc->cq) != 0) {
            return -1;
        }
        if (!jfc->cq.lock_free) {
            const int spin_status =
                pthread_spin_destroy(&jfc->cq.lock);

            if (spin_status != 0) {
                return kvaf_udma_pthread_cleanup_failed(spin_status);
            }
        }
        state->cq_created = false;
    }
    if (kvaf_udma_destroy_event_sync_checked(
            &jfc->base.event_mutex, &jfc->base.event_cond,
            &state->event_condition_initialized,
            &state->event_mutex_initialized) != 0) {
        return -1;
    }
    return 0;
}

static int kvaf_create_cq_checked(
    struct kvaf_udma_jfc_state *state, urma_jfc_cfg_t *config)
{
    struct udma_u_jetty_queue *cq = &state->jfc->cq;
    const uint32_t depth = config->depth < UDMA_U_MIN_JFC_DEPTH
        ? UDMA_U_MIN_JFC_DEPTH : config->depth;

    cq->lock_free = config->flag.bs.lock_free;
    if (!cq->lock_free &&
        pthread_spin_init(&cq->lock, PTHREAD_PROCESS_PRIVATE) != 0) {
        return -1;
    }
    state->cq_created = true;
    cq->cstm = false;
    cq->sq_reserved = false;
    cq->dtu_en = false;
    cq->is_jetty = false;
    cq->aligned_size = UDMA_HW_PAGE_SIZE;
    return kvaf_udma_checked_alloc_queue_buffer(
        cq, depth, cq->ctx->cqe_size, UDMA_HW_PAGE_SIZE, false);
}

int kvaf_udma_source_retry_orphan_jfcs(urma_context_t *context)
{
    struct kvaf_udma_jfc_state *state;
    struct kvaf_udma_jfc_state *next;
    int result = 0;
    int saved_errno = 0;

    if (context == NULL) {
        return kvaf_udma_cleanup_invariant_failed();
    }
    if (kvaf_lock_jfc_registry_checked() != 0) {
        return -1;
    }
    for (state = g_kvaf_jfcs; state != NULL; state = next) {
        next = state->next;
        if (!state->orphan || state->owner_context != context) {
            continue;
        }
        if ((state->kernel_created || state->kernel_delete_pending) &&
            kvaf_delete_kernel_jfc_nonblocking(state) != 0) {
            result = -1;
            kvaf_udma_preserve_cleanup_errno(&saved_errno);
            continue;
        }
        errno = 0;
        if (kvaf_cleanup_local_jfc_checked(state) != 0) {
            (void)kvaf_udma_classify_cleanup_failure();
            result = -1;
            kvaf_udma_preserve_cleanup_errno(&saved_errno);
            continue;
        }
        kvaf_remove_jfc_locked(state);
        free(state->jfc);
        free(state);
    }
    if (kvaf_unlock_jfc_registry_checked() != 0) {
        return -1;
    }
    if (result != 0) {
        errno = saved_errno == 0 ? EIO : saved_errno;
    }
    return result;
}

uint32_t kvaf_udma_source_live_jfc_count(void)
{
    uint32_t count;

    if (pthread_mutex_lock(&g_kvaf_jfc_lock) != 0) {
        return UINT32_MAX;
    }
    count = g_kvaf_jfc_count;
    (void)pthread_mutex_unlock(&g_kvaf_jfc_lock);
    return count;
}

uint32_t kvaf_udma_source_live_jfc_count_for_context(
    const urma_context_t *context)
{
    const struct kvaf_udma_jfc_state *state;
    uint32_t count = 0;

    if (context == NULL || pthread_mutex_lock(&g_kvaf_jfc_lock) != 0) {
        return UINT32_MAX;
    }
    for (state = g_kvaf_jfcs; state != NULL; state = state->next) {
        if (state->owner_context == context) {
            ++count;
        }
    }
    (void)pthread_mutex_unlock(&g_kvaf_jfc_lock);
    return count;
}

urma_jfc_t *kvaf_udma_create_jfc_checked(
    urma_context_t *context, urma_jfc_cfg_t *cfg)
{
    struct kvaf_udma_jfc_state *state;
    struct udma_u_context *udma_context;
    struct udma_u_jfc *jfc;
    int status;

    state = calloc(1, sizeof(*state));
    jfc = calloc(1, sizeof(*jfc));
    if (state == NULL || jfc == NULL || context == NULL || cfg == NULL) {
        free(state);
        free(jfc);
        return NULL;
    }
    if (kvaf_lock_jfc_registry_checked() != 0) {
        free(state);
        free(jfc);
        return NULL;
    }
    state->jfc = jfc;
    state->owner_context = context;
    state->orphan = true;
    state->next = g_kvaf_jfcs;
    g_kvaf_jfcs = state;
    ++g_kvaf_jfc_count;
    ++g_kvaf_jfc_orphan_count;

    if (udma_u_check_jfc_cfg(context, cfg) != 0 ||
        cfg->flag.bs.non_blocking || cfg->jfce != NULL) {
        goto fail;
    }
    udma_context = to_udma_u_ctx(context);
    if (udma_context->dtu_enable || udma_context->sq_reserved) {
        goto fail;
    }
    jfc->cq.ctx = udma_context;
    jfc->cq.dtu_en = false;
    jfc->db_cstm = false;
    jfc->ccu_flag = 0U;
    if (kvaf_create_cq_checked(state, cfg) != 0) {
        goto fail;
    }
    status = kvaf_udma_checked_alloc_sw_db(
        udma_context, UDMA_JFC_TYPE_DB, &jfc->sw_db);
    if (jfc->sw_db != NULL) {
        state->sw_db_allocated = true;
    }
    if (status != 0) {
        goto fail;
    }
    status = udma_u_jfc_cmd(context, jfc, cfg);
    if (status != 0) {
        goto fail;
    }
    state->created_owner_context = context;
    state->created_handle = jfc->base.handle;
    state->created_id = jfc->base.jfc_id.id;
    state->created_dev_fd = context->dev_fd;
    state->kernel_created = true;
    state->event_condition_initialized = true;
    state->event_mutex_initialized = true;
    jfc->cq_shift = align_power2(jfc->cq.baseblk_cnt);
    jfc->cq.idx = jfc->base.jfc_id.id;
    jfc->arm_sn = 1;
    state->orphan = false;
    --g_kvaf_jfc_orphan_count;
    if (kvaf_unlock_jfc_registry_checked() != 0) {
        /* No public handle escaped. Keep the fully created object owned by the
         * orphan sidecar while the process-lifetime unsafe latch blocks reuse. */
        state->orphan = true;
        ++g_kvaf_jfc_orphan_count;
        return NULL;
    }
    return &jfc->base;

fail:
    errno = 0;
    if (kvaf_cleanup_local_jfc_checked(state) == 0) {
        kvaf_remove_jfc_locked(state);
        free(jfc);
        free(state);
    } else {
        (void)kvaf_udma_classify_cleanup_failure();
    }
    (void)kvaf_unlock_jfc_registry_checked();
    return NULL;
}

urma_status_t kvaf_udma_delete_jfc_checked(urma_jfc_t *public_jfc)
{
    struct kvaf_udma_jfc_state *state;
    int status;

    if (public_jfc == NULL) {
        (void)kvaf_udma_cleanup_invariant_failed();
        return URMA_FAIL;
    }
    if (kvaf_lock_jfc_registry_checked() != 0) {
        return URMA_FAIL;
    }
    state = kvaf_find_jfc_locked(to_udma_u_jfc(public_jfc));
    if (state == NULL ||
        state->owner_context == NULL ||
        public_jfc->urma_ctx != state->owner_context ||
        public_jfc->jfc_cfg.flag.bs.non_blocking) {
        (void)kvaf_udma_cleanup_invariant_failed();
        (void)kvaf_unlock_jfc_registry_checked();
        return URMA_FAIL;
    }
    if (state->kernel_created || state->kernel_delete_pending) {
        status = kvaf_delete_kernel_jfc_nonblocking(state);
        if (status != 0) {
            (void)kvaf_unlock_jfc_registry_checked();
            return URMA_FAIL;
        }
    }
    errno = 0;
    if (kvaf_cleanup_local_jfc_checked(state) != 0) {
        (void)kvaf_udma_classify_cleanup_failure();
        /* Kernel CQ is gone; retain local mapping handles for exact retry. */
        (void)kvaf_unlock_jfc_registry_checked();
        return URMA_FAIL;
    }
    kvaf_remove_jfc_locked(state);
    if (kvaf_unlock_jfc_registry_checked() != 0) {
        return URMA_FAIL;
    }
    free(state->jfc);
    free(state);
    return URMA_SUCCESS;
}

urma_jfc_t *udma_u_create_jfc(
    urma_context_t *context, urma_jfc_cfg_t *cfg)
{
    return kvaf_udma_create_jfc_checked(context, cfg);
}

urma_status_t udma_u_delete_jfc(urma_jfc_t *jfc)
{
    return kvaf_udma_delete_jfc_checked(jfc);
}
