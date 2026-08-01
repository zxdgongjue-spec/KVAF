#include "kvaf_source_matched_udma.h"

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

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
    bool orphan;
    struct kvaf_udma_segment_state *next;
};

static pthread_mutex_t g_kvaf_segment_lock = PTHREAD_MUTEX_INITIALIZER;
static struct kvaf_udma_segment_state *g_kvaf_segments;
static uint32_t g_kvaf_segment_count;
static uint32_t g_kvaf_segment_orphan_count;

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
    if (!state->orphan) {
        return 0;
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

    if (udma_u_check_seg_cfg(seg_cfg) != 0) {
        return NULL;
    }
    state = calloc(1, sizeof(*state));
    segment = calloc(1, sizeof(*segment));
    if (state == NULL || segment == NULL) {
        free(state);
        free(segment);
        return NULL;
    }
    state->segment = segment;
    state->owner_context = urma_ctx;
    state->orphan = true;
    if (pthread_mutex_lock(&g_kvaf_segment_lock) != 0) {
        free(segment);
        free(state);
        return NULL;
    }
    /* Own the object before the first lower-level resource is acquired. */
    kvaf_add_segment_locked(state);
    udma_u_init_seg_cfg(urma_ctx, seg_cfg, segment);

    status = udma_u_grant_segment(segment);
    if (status != 0) {
        kvaf_remove_segment_locked(state);
        (void)pthread_mutex_unlock(&g_kvaf_segment_lock);
        free(segment);
        free(state);
        return NULL;
    }
    state->ummu_granted = true;
    state->ummu_tid =
        to_udma_u_tid(segment->urma_tseg.token_id)->tid;

    status = udma_exec_register_seg_cmd(
        urma_ctx, seg_cfg, &segment->urma_tseg);
    if (status != 0) {
        status = ummu_ungrant(
            state->ummu_tid,
            (void *)segment->urma_tseg.seg.ubva.va,
            segment->urma_tseg.seg.len);
        if (status == 0) {
            state->ummu_granted = false;
            kvaf_remove_segment_locked(state);
            (void)pthread_mutex_unlock(&g_kvaf_segment_lock);
            free(segment);
            free(state);
            return NULL;
        }
        /* Preserve an ungrantable mapping instead of losing its owner. */
        (void)pthread_mutex_unlock(&g_kvaf_segment_lock);
        return NULL;
    }
    state->kernel_registered = !seg_cfg->flag.bs.non_pin;
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
