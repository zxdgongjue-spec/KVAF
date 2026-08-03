#include "kvaf_source_matched_udma.h"

#include <errno.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "kvaf_source_matched_ummu.h"
#include "ummu_api.h"

/*
 * Keep the source-matched create/import implementation, but replace the
 * unregister operation whose upstream void ungrant helper discards errors.
 * Including the selected source also keeps all private layout and helper
 * definitions in the same translation unit.
 */
#ifndef KVAF_UDMA_SEGMENT_SOURCE
#error "KVAF_UDMA_SEGMENT_SOURCE must name udma_u_segment.c"
#endif
#define udma_u_register_seg kvaf_legacy_udma_u_register_seg
#define udma_u_unregister_seg kvaf_legacy_udma_u_unregister_seg
#define udma_u_import_seg kvaf_legacy_udma_u_import_seg
#define udma_u_unimport_seg kvaf_legacy_udma_u_unimport_seg
#include KVAF_UDMA_SEGMENT_SOURCE
#undef udma_u_unimport_seg
#undef udma_u_import_seg
#undef udma_u_unregister_seg
#undef udma_u_register_seg

struct kvaf_udma_segment_state {
    struct udma_u_segment *segment;
    urma_context_t *owner_context;
    uint32_t ummu_tid;
    bool kernel_registered;
    bool ummu_granted;
    bool grant_retry_needed;
    bool grant_state_unknown;
    bool orphan;
    struct kvaf_udma_segment_state *next;
};

static pthread_mutex_t g_kvaf_segment_lock = PTHREAD_MUTEX_INITIALIZER;
static struct kvaf_udma_segment_state *g_kvaf_segments;
static uint32_t g_kvaf_segment_count;
static uint32_t g_kvaf_segment_orphan_count;
static _Thread_local struct kvaf_udma_segment_failure
    g_kvaf_segment_failure;

static void kvaf_clear_segment_failure(void)
{
    g_kvaf_segment_failure = (struct kvaf_udma_segment_failure) {
        .abi_version = KVAF_UDMA_SEGMENT_FAILURE_ABI_VERSION,
        .structure_size = sizeof(g_kvaf_segment_failure),
    };
}

static int kvaf_effective_errno(int status, int operation_errno)
{
    if (status < 0 && status != -1) {
        return -status;
    }
    if (status > 0) {
        return status;
    }
    if (operation_errno != 0) {
        return operation_errno;
    }
    return EIO;
}

static void kvaf_record_segment_failure(
    enum kvaf_udma_segment_failure_stage stage, int operation_status,
    int operation_errno, int rollback_status, int rollback_errno,
    const struct udma_u_segment *segment)
{
    const struct udma_u_tid *token = segment == NULL ||
        segment->urma_tseg.token_id == NULL ? NULL :
        to_udma_u_tid(segment->urma_tseg.token_id);

    g_kvaf_segment_failure = (struct kvaf_udma_segment_failure) {
        .abi_version = KVAF_UDMA_SEGMENT_FAILURE_ABI_VERSION,
        .structure_size = sizeof(g_kvaf_segment_failure),
        .stage = (uint32_t)stage,
        .operation_status = operation_status,
        .operation_errno = kvaf_effective_errno(
            operation_status, operation_errno),
        .rollback_status = rollback_status,
        .rollback_errno = rollback_status == 0 ? 0 :
            kvaf_effective_errno(rollback_status, rollback_errno),
        .ummu_tid = token == NULL ? UINT32_MAX : token->tid,
        .token_id = token == NULL ? UINT32_MAX : token->base.token_id,
        .token_handle = token == NULL ? 0 : token->base.handle,
        .address = segment == NULL ? 0 :
            segment->urma_tseg.seg.ubva.va,
        .length = segment == NULL ? 0 : segment->urma_tseg.seg.len,
    };
}

int kvaf_udma_source_get_segment_failure(
    struct kvaf_udma_segment_failure *failure)
{
    if (failure == NULL ||
        failure->structure_size < sizeof(*failure)) {
        return -EINVAL;
    }
    *failure = g_kvaf_segment_failure;
    failure->abi_version = KVAF_UDMA_SEGMENT_FAILURE_ABI_VERSION;
    failure->structure_size = sizeof(*failure);
    return 0;
}

static bool kvaf_udma_segment_access_is_supported(uint32_t access)
{
    return access == URMA_ACCESS_LOCAL_ONLY ||
        access == URMA_ACCESS_READ ||
        access == (URMA_ACCESS_READ | URMA_ACCESS_WRITE) ||
        access == (URMA_ACCESS_READ | URMA_ACCESS_WRITE |
            URMA_ACCESS_ATOMIC);
}

static void kvaf_add_segment_locked(
    struct kvaf_udma_segment_state *state)
{
    state->next = g_kvaf_segments;
    g_kvaf_segments = state;
    ++g_kvaf_segment_count;
    if (state->orphan) {
        ++g_kvaf_segment_orphan_count;
    }
}

static void kvaf_remove_segment_locked(
    struct kvaf_udma_segment_state *state)
{
    struct kvaf_udma_segment_state **cursor = &g_kvaf_segments;

    while (*cursor != NULL) {
        if (*cursor == state) {
            *cursor = state->next;
            --g_kvaf_segment_count;
            if (state->orphan) {
                --g_kvaf_segment_orphan_count;
            }
            return;
        }
        cursor = &(*cursor)->next;
    }
}

static struct kvaf_udma_segment_state *kvaf_find_segment_locked(
    struct udma_u_segment *segment)
{
    struct kvaf_udma_segment_state *state;

    for (state = g_kvaf_segments; state != NULL; state = state->next) {
        if (state->segment == segment) {
            return state;
        }
    }
    return NULL;
}

uint32_t kvaf_udma_source_live_segment_count(void)
{
    uint32_t count;

    if (pthread_mutex_lock(&g_kvaf_segment_lock) != 0) {
        return UINT32_MAX;
    }
    count = g_kvaf_segment_count;
    (void)pthread_mutex_unlock(&g_kvaf_segment_lock);
    return count;
}

uint32_t kvaf_udma_source_live_segment_count_for_context(
    const urma_context_t *context)
{
    const struct kvaf_udma_segment_state *state;
    uint32_t count = 0;

    if (context == NULL ||
        pthread_mutex_lock(&g_kvaf_segment_lock) != 0) {
        return UINT32_MAX;
    }
    for (state = g_kvaf_segments; state != NULL; state = state->next) {
        if (state->owner_context == context) {
            ++count;
        }
    }
    (void)pthread_mutex_unlock(&g_kvaf_segment_lock);
    return count;
}

uint32_t kvaf_udma_source_orphan_segment_count(void)
{
    uint32_t count;

    if (pthread_mutex_lock(&g_kvaf_segment_lock) != 0) {
        return UINT32_MAX;
    }
    count = g_kvaf_segment_orphan_count;
    (void)pthread_mutex_unlock(&g_kvaf_segment_lock);
    return count;
}

static int kvaf_retry_orphan_segment_locked(
    struct kvaf_udma_segment_state *state)
{
    int status;

    if (!state->orphan) {
        return 0;
    }
    if (state->grant_retry_needed) {
        struct kvaf_source_ummu_status ummu = {
            .structure_size = sizeof(ummu),
        };

        if (state->grant_state_unknown ||
            kvaf_source_ummu_cleanup_is_unknown() != 0) {
            return -1;
        }
        errno = 0;
        status = udma_u_grant_segment(state->segment);
        if (status != 0) {
            if (kvaf_source_ummu_get_thread_status(&ummu) != 0 ||
                ummu.last_failure_stage ==
                    KVAF_SOURCE_UMMU_FAILURE_NONE ||
                ummu.last_state_unknown != 0U ||
                ummu.last_retryability ==
                    KVAF_SOURCE_UMMU_RETRY_UNSAFE) {
                state->grant_state_unknown = true;
                kvaf_udma_source_mark_cleanup_unknown();
            }
            return -1;
        }
        state->ummu_tid = to_udma_u_tid(
            state->segment->urma_tseg.token_id)->tid;
        state->ummu_granted = true;
        state->grant_retry_needed = false;
    }
    if (state->kernel_registered) {
        if (urma_cmd_unregister_seg(&state->segment->urma_tseg) != 0) {
            return -1;
        }
        state->kernel_registered = false;
    }
    if (state->ummu_granted) {
        if (ummu_ungrant(state->ummu_tid,
                (void *)state->segment->urma_tseg.seg.ubva.va,
                state->segment->urma_tseg.seg.len) != 0) {
            return -1;
        }
        state->ummu_granted = false;
    }
    return 0;
}

int kvaf_udma_source_retry_orphan_segments(urma_context_t *context)
{
    struct kvaf_udma_segment_state *state;
    struct kvaf_udma_segment_state *next;
    int result = 0;

    if (context == NULL ||
        pthread_mutex_lock(&g_kvaf_segment_lock) != 0) {
        return -1;
    }
    for (state = g_kvaf_segments; state != NULL; state = next) {
        next = state->next;
        if (!state->orphan || state->owner_context != context) {
            continue;
        }
        if (kvaf_retry_orphan_segment_locked(state) != 0) {
            result = -1;
            continue;
        }
        kvaf_remove_segment_locked(state);
        free(state->segment);
        free(state);
    }
    (void)pthread_mutex_unlock(&g_kvaf_segment_lock);
    return result;
}

urma_target_seg_t *udma_u_register_seg(
    urma_context_t *urma_ctx, urma_seg_cfg_t *seg_cfg)
{
    struct kvaf_udma_segment_state *state;
    struct udma_u_segment *segment;
    int status;

    kvaf_clear_segment_failure();
    if (urma_ctx == NULL || seg_cfg == NULL ||
        !kvaf_udma_segment_access_is_supported(
            seg_cfg->flag.bs.access) ||
        udma_u_check_seg_cfg(seg_cfg) != 0) {
        kvaf_record_segment_failure(
            KVAF_UDMA_SEGMENT_FAILURE_VALIDATE, -EINVAL, EINVAL,
            0, 0, NULL);
        errno = EINVAL;
        return NULL;
    }
    state = calloc(1, sizeof(*state));
    segment = calloc(1, sizeof(*segment));
    if (state == NULL || segment == NULL) {
        kvaf_record_segment_failure(
            KVAF_UDMA_SEGMENT_FAILURE_ALLOCATE_OWNER, -ENOMEM, ENOMEM,
            0, 0, NULL);
        free(state);
        free(segment);
        errno = ENOMEM;
        return NULL;
    }
    state->segment = segment;
    state->owner_context = urma_ctx;
    state->orphan = true;
    if (pthread_mutex_lock(&g_kvaf_segment_lock) != 0) {
        kvaf_record_segment_failure(
            KVAF_UDMA_SEGMENT_FAILURE_LOCK_OWNER, -EDEADLK, EDEADLK,
            0, 0, segment);
        free(segment);
        free(state);
        errno = EDEADLK;
        return NULL;
    }
    /* Own the object before the first lower-level resource is acquired. */
    kvaf_add_segment_locked(state);
    udma_u_init_seg_cfg(urma_ctx, seg_cfg, segment);

    errno = 0;
    status = udma_u_grant_segment(segment);
    if (status != 0) {
        const int operation_errno = errno;
        struct kvaf_source_ummu_status ummu = {
            .structure_size = sizeof(ummu),
        };
        const int ummu_status =
            kvaf_source_ummu_get_thread_status(&ummu);
        const bool no_grant_resource_acquired =
            ummu_status == 0 &&
            ummu.last_pending_operation ==
                KVAF_SOURCE_UMMU_PENDING_NONE &&
            ummu.last_state_unknown == 0U &&
            (ummu.last_failure_stage ==
                 KVAF_SOURCE_UMMU_FAILURE_GRANT_VALIDATE ||
             ummu.last_failure_stage ==
                 KVAF_SOURCE_UMMU_FAILURE_GRANT_STATE_LOOKUP);

        kvaf_record_segment_failure(
            KVAF_UDMA_SEGMENT_FAILURE_UMMU_GRANT, status,
            operation_errno, 0, 0, segment);
        if (no_grant_resource_acquired) {
            kvaf_remove_segment_locked(state);
        } else {
            /*
             * A retryable PLBI failure can leave a completed local MAPT
             * mutation plus PENDING_GRANT_TOKEN_UPDATE.  The Segment address
             * is the only exact owner which can finish that grant and then
             * ungrant it.  Keep the orphan sidecar until Close retries both.
             */
            state->grant_retry_needed = true;
            state->grant_state_unknown = ummu_status != 0 ||
                ummu.last_failure_stage ==
                    KVAF_SOURCE_UMMU_FAILURE_NONE ||
                ummu.last_state_unknown != 0U ||
                ummu.last_retryability ==
                    KVAF_SOURCE_UMMU_RETRY_UNSAFE;
            if (state->grant_state_unknown) {
                kvaf_udma_source_mark_cleanup_unknown();
            }
        }
        (void)pthread_mutex_unlock(&g_kvaf_segment_lock);
        if (no_grant_resource_acquired) {
            free(segment);
            free(state);
        }
        errno = g_kvaf_segment_failure.operation_errno;
        return NULL;
    }
    state->ummu_granted = true;
    state->ummu_tid =
        to_udma_u_tid(segment->urma_tseg.token_id)->tid;

    errno = 0;
    status = udma_exec_register_seg_cmd(
        urma_ctx, seg_cfg, &segment->urma_tseg);
    if (status != 0) {
        const int operation_status = status;
        const int operation_errno = errno;
        int rollback_status;
        int rollback_errno;

        errno = 0;
        rollback_status = ummu_ungrant(
            state->ummu_tid,
            (void *)segment->urma_tseg.seg.ubva.va,
            segment->urma_tseg.seg.len);
        rollback_errno = errno;
        kvaf_record_segment_failure(
            KVAF_UDMA_SEGMENT_FAILURE_KERNEL_REGISTER,
            operation_status, operation_errno,
            rollback_status, rollback_errno, segment);
        if (rollback_status == 0) {
            state->ummu_granted = false;
            kvaf_remove_segment_locked(state);
            (void)pthread_mutex_unlock(&g_kvaf_segment_lock);
            free(segment);
            free(state);
            errno = g_kvaf_segment_failure.operation_errno;
            return NULL;
        }
        /* Preserve an ungrantable mapping instead of losing its owner. */
        (void)pthread_mutex_unlock(&g_kvaf_segment_lock);
        errno = g_kvaf_segment_failure.operation_errno;
        return NULL;
    }
    state->kernel_registered =
        to_udma_u_ctx(urma_ctx)->sva_sep_mode_en ||
        !seg_cfg->flag.bs.non_pin;
    state->orphan = false;
    --g_kvaf_segment_orphan_count;
    (void)pthread_mutex_unlock(&g_kvaf_segment_lock);
    return &segment->urma_tseg;
}

urma_status_t udma_u_unregister_seg(urma_target_seg_t *target_seg)
{
    struct udma_u_segment *segment;
    struct kvaf_udma_segment_state *state;

    if (target_seg == NULL ||
        pthread_mutex_lock(&g_kvaf_segment_lock) != 0) {
        return URMA_FAIL;
    }
    segment = to_udma_u_seg(target_seg);
    state = kvaf_find_segment_locked(segment);
    if (state == NULL) {
        (void)pthread_mutex_unlock(&g_kvaf_segment_lock);
        return URMA_FAIL;
    }

    if (state->kernel_registered) {
        if (urma_cmd_unregister_seg(target_seg) != 0) {
            (void)pthread_mutex_unlock(&g_kvaf_segment_lock);
            return URMA_FAIL;
        }
        state->kernel_registered = false;
    }
    if (state->ummu_granted) {
        if (ummu_ungrant(
                state->ummu_tid,
                (void *)segment->urma_tseg.seg.ubva.va,
                segment->urma_tseg.seg.len) != 0) {
            /* A retry skips the completed kernel command. */
            (void)pthread_mutex_unlock(&g_kvaf_segment_lock);
            return URMA_FAIL;
        }
        state->ummu_granted = false;
    }

    kvaf_remove_segment_locked(state);
    (void)pthread_mutex_unlock(&g_kvaf_segment_lock);
    segment->token_value.token = 0;
    free(segment);
    free(state);
    return URMA_SUCCESS;
}

urma_target_seg_t *udma_u_import_seg(
    urma_context_t *context, urma_seg_t *segment, urma_token_t *token,
    uint64_t address, urma_import_seg_flag_t flag)
{
    return kvaf_legacy_udma_u_import_seg(
        context, segment, token, address, flag);
}

urma_status_t udma_u_unimport_seg(urma_target_seg_t *target_segment)
{
    return kvaf_legacy_udma_u_unimport_seg(target_segment);
}
