#include "kvaf_source_matched_udma.h"

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

struct kvaf_udma_jfc_state {
    struct udma_u_jfc *jfc;
    urma_context_t *owner_context;
    bool kernel_created;
    bool sw_db_allocated;
    bool cq_created;
    bool orphan;
    struct kvaf_udma_jfc_state *next;
};

static pthread_mutex_t g_kvaf_jfc_lock = PTHREAD_MUTEX_INITIALIZER;
static struct kvaf_udma_jfc_state *g_kvaf_jfcs;
static uint32_t g_kvaf_jfc_count;
static uint32_t g_kvaf_jfc_orphan_count;

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
    struct udma_u_jfc *jfc = state->jfc;
    struct udma_u_context *context =
        to_udma_u_ctx(state->owner_context);

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
        if (!jfc->cq.lock_free &&
            pthread_spin_destroy(&jfc->cq.lock) != 0) {
            return -1;
        }
        state->cq_created = false;
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

    if (context == NULL || pthread_mutex_lock(&g_kvaf_jfc_lock) != 0) {
        return -1;
    }
    for (state = g_kvaf_jfcs; state != NULL; state = next) {
        next = state->next;
        if (!state->orphan || state->owner_context != context) {
            continue;
        }
        if (state->kernel_created &&
            urma_cmd_delete_jfc(&state->jfc->base) != 0) {
            result = -1;
            continue;
        }
        state->kernel_created = false;
        if (kvaf_cleanup_local_jfc_checked(state) != 0) {
            result = -1;
            continue;
        }
        kvaf_remove_jfc_locked(state);
        free(state->jfc);
        free(state);
    }
    (void)pthread_mutex_unlock(&g_kvaf_jfc_lock);
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

urma_jfc_t *kvaf_udma_create_jfc_checked(
    urma_context_t *context, urma_jfc_cfg_t *cfg)
{
    struct kvaf_udma_jfc_state *state;
    struct udma_u_context *udma_context;
    struct udma_u_jfc *jfc;
    int status;

    state = calloc(1, sizeof(*state));
    jfc = calloc(1, sizeof(*jfc));
    if (state == NULL || jfc == NULL || context == NULL || cfg == NULL ||
        pthread_mutex_lock(&g_kvaf_jfc_lock) != 0) {
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
        cfg->flag.bs.non_blocking) {
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
    state->kernel_created = true;
    jfc->cq_shift = align_power2(jfc->cq.baseblk_cnt);
    jfc->cq.idx = jfc->base.jfc_id.id;
    jfc->arm_sn = 1;
    state->orphan = false;
    --g_kvaf_jfc_orphan_count;
    (void)pthread_mutex_unlock(&g_kvaf_jfc_lock);
    return &jfc->base;

fail:
    if (kvaf_cleanup_local_jfc_checked(state) == 0) {
        kvaf_remove_jfc_locked(state);
        free(jfc);
        free(state);
    }
    (void)pthread_mutex_unlock(&g_kvaf_jfc_lock);
    return NULL;
}

urma_status_t kvaf_udma_delete_jfc_checked(urma_jfc_t *public_jfc)
{
    struct kvaf_udma_jfc_state *state;
    int status;

    if (public_jfc == NULL ||
        pthread_mutex_lock(&g_kvaf_jfc_lock) != 0) {
        return URMA_FAIL;
    }
    state = kvaf_find_jfc_locked(to_udma_u_jfc(public_jfc));
    if (state == NULL ||
        public_jfc->jfc_cfg.flag.bs.non_blocking) {
        (void)pthread_mutex_unlock(&g_kvaf_jfc_lock);
        return URMA_FAIL;
    }
    if (state->kernel_created) {
        status = urma_cmd_delete_jfc(public_jfc);
        if (status != 0) {
            (void)pthread_mutex_unlock(&g_kvaf_jfc_lock);
            return URMA_FAIL;
        }
        state->kernel_created = false;
    }
    if (kvaf_cleanup_local_jfc_checked(state) != 0) {
        /* Kernel CQ is gone; retain local mapping handles for exact retry. */
        (void)pthread_mutex_unlock(&g_kvaf_jfc_lock);
        return URMA_FAIL;
    }
    kvaf_remove_jfc_locked(state);
    (void)pthread_mutex_unlock(&g_kvaf_jfc_lock);
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
