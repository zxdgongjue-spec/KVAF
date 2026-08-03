#include "kvaf_source_matched_udma.h"

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "ummu_api.h"

#ifndef KVAF_UDMA_TID_SOURCE
#error "KVAF_UDMA_TID_SOURCE must name udma_u_tid.c"
#endif
#define udma_u_alloc_tid kvaf_legacy_udma_u_alloc_tid
#define udma_u_alloc_tid_ex kvaf_legacy_udma_u_alloc_tid_ex
#define udma_u_free_tid kvaf_legacy_udma_u_free_tid
#include KVAF_UDMA_TID_SOURCE
#undef udma_u_free_tid
#undef udma_u_alloc_tid_ex
#undef udma_u_alloc_tid

struct kvaf_udma_token_state {
    struct udma_u_tid *token;
    urma_context_t *owner_context;
    bool kernel_allocated;
    bool ummu_allocated;
    bool orphan;
    struct kvaf_udma_token_state *next;
};

static pthread_mutex_t g_kvaf_token_lock = PTHREAD_MUTEX_INITIALIZER;
static struct kvaf_udma_token_state *g_kvaf_tokens;
static uint32_t g_kvaf_token_count;
static uint32_t g_kvaf_token_orphan_count;

static void kvaf_add_token_locked(struct kvaf_udma_token_state *state)
{
    state->next = g_kvaf_tokens;
    g_kvaf_tokens = state;
    ++g_kvaf_token_count;
    if (state->orphan) {
        ++g_kvaf_token_orphan_count;
    }
}

static void kvaf_remove_token_locked(struct kvaf_udma_token_state *state)
{
    struct kvaf_udma_token_state **cursor = &g_kvaf_tokens;

    while (*cursor != NULL) {
        if (*cursor == state) {
            *cursor = state->next;
            --g_kvaf_token_count;
            if (state->orphan) {
                --g_kvaf_token_orphan_count;
            }
            return;
        }
        cursor = &(*cursor)->next;
    }
}

static struct kvaf_udma_token_state *kvaf_find_token_locked(
    struct udma_u_tid *token)
{
    struct kvaf_udma_token_state *state;

    for (state = g_kvaf_tokens; state != NULL; state = state->next) {
        if (state->token == token) {
            return state;
        }
    }
    return NULL;
}

uint32_t kvaf_udma_source_live_token_count(void)
{
    uint32_t count;

    if (pthread_mutex_lock(&g_kvaf_token_lock) != 0) {
        return UINT32_MAX;
    }
    count = g_kvaf_token_count;
    (void)pthread_mutex_unlock(&g_kvaf_token_lock);
    return count;
}

uint32_t kvaf_udma_source_live_token_count_for_context(
    const urma_context_t *context)
{
    const struct kvaf_udma_token_state *state;
    uint32_t count = 0;

    if (context == NULL || pthread_mutex_lock(&g_kvaf_token_lock) != 0) {
        return UINT32_MAX;
    }
    for (state = g_kvaf_tokens; state != NULL; state = state->next) {
        if (state->owner_context == context) {
            ++count;
        }
    }
    (void)pthread_mutex_unlock(&g_kvaf_token_lock);
    return count;
}

uint32_t kvaf_udma_source_orphan_token_count(void)
{
    uint32_t count;

    if (pthread_mutex_lock(&g_kvaf_token_lock) != 0) {
        return UINT32_MAX;
    }
    count = g_kvaf_token_orphan_count;
    (void)pthread_mutex_unlock(&g_kvaf_token_lock);
    return count;
}

static int kvaf_retry_orphan_token_locked(
    struct kvaf_udma_token_state *state)
{
    if (!state->orphan) {
        return 0;
    }
    if (state->kernel_allocated) {
        if (urma_cmd_free_token_id(&state->token->base) != 0) {
            return -1;
        }
        state->kernel_allocated = false;
    }
    if (state->ummu_allocated) {
        if (ummu_free_tid(state->token->tid) != 0) {
            return -1;
        }
        state->ummu_allocated = false;
    }
    return 0;
}

int kvaf_udma_source_retry_orphan_tokens(urma_context_t *context)
{
    struct kvaf_udma_token_state *state;
    struct kvaf_udma_token_state *next;
    int result = 0;

    if (context == NULL || pthread_mutex_lock(&g_kvaf_token_lock) != 0) {
        return -1;
    }
    for (state = g_kvaf_tokens; state != NULL; state = next) {
        next = state->next;
        if (!state->orphan || state->owner_context != context) {
            continue;
        }
        if (kvaf_retry_orphan_token_locked(state) != 0) {
            result = -1;
            continue;
        }
        kvaf_remove_token_locked(state);
        free(state->token);
        free(state);
    }
    (void)pthread_mutex_unlock(&g_kvaf_token_lock);
    return result;
}

static urma_token_id_t *kvaf_udma_alloc_tid_checked(
    urma_context_t *context, enum ummu_mapt_mode mode)
{
    struct ummu_tid_attr tid_attr = {.mode = mode};
    struct kvaf_udma_token_state *state;
    struct udma_u_tid *token;
    uint32_t tid = UINT32_MAX;
    int status;

    state = calloc(1, sizeof(*state));
    token = calloc(1, sizeof(*token));
    if (state == NULL || token == NULL) {
        free(state);
        free(token);
        return NULL;
    }
    state->token = token;
    state->owner_context = context;
    state->orphan = true;
    if (pthread_mutex_lock(&g_kvaf_token_lock) != 0) {
        free(token);
        free(state);
        return NULL;
    }
    kvaf_add_token_locked(state);

    status = ummu_allocate_tid(&tid_attr, &tid);
    if (status != 0 || tid > UDMA_MAX_TID) {
        if (tid != UINT32_MAX) {
            state->ummu_allocated = true;
            token->tid = tid;
            if (ummu_free_tid(tid) == 0) {
                state->ummu_allocated = false;
            }
        }
        if (!state->ummu_allocated) {
            kvaf_remove_token_locked(state);
            (void)pthread_mutex_unlock(&g_kvaf_token_lock);
            free(token);
            free(state);
        } else {
            /* Retain an orphan whose UMMU TID could not be released. */
            (void)pthread_mutex_unlock(&g_kvaf_token_lock);
        }
        return NULL;
    }
    token->tid = tid;
    state->ummu_allocated = true;
    token->base.token_id = tid << UDMA_TID_SHIFT;
    status = udma_exec_alloc_tid_cmd(
        context, token->base.token_id, &token->base);
    if (status != 0) {
        if (ummu_free_tid(tid) == 0) {
            state->ummu_allocated = false;
            kvaf_remove_token_locked(state);
            (void)pthread_mutex_unlock(&g_kvaf_token_lock);
            free(token);
            free(state);
        } else {
            (void)pthread_mutex_unlock(&g_kvaf_token_lock);
        }
        return NULL;
    }
    state->kernel_allocated = true;
    token->base.urma_ctx = context;
    state->orphan = false;
    --g_kvaf_token_orphan_count;
    (void)pthread_mutex_unlock(&g_kvaf_token_lock);
    return &token->base;
}

urma_token_id_t *udma_u_alloc_tid(urma_context_t *context)
{
    return kvaf_udma_alloc_tid_checked(context, MAPT_MODE_TABLE);
}

urma_token_id_t *udma_u_alloc_tid_ex(
    urma_context_t *context, urma_token_id_flag_t flag)
{
    return kvaf_udma_alloc_tid_checked(
        context,
        flag.bs.multi_seg != 0 ? MAPT_MODE_TABLE : MAPT_MODE_ENTRY);
}

urma_status_t udma_u_free_tid(urma_token_id_t *token_id)
{
    struct udma_u_tid *token;
    struct kvaf_udma_token_state *state;

    if (token_id == NULL ||
        pthread_mutex_lock(&g_kvaf_token_lock) != 0) {
        return URMA_FAIL;
    }
    token = to_udma_u_tid(token_id);
    state = kvaf_find_token_locked(token);
    if (state == NULL) {
        (void)pthread_mutex_unlock(&g_kvaf_token_lock);
        return URMA_FAIL;
    }

    if (state->kernel_allocated) {
        if (urma_cmd_free_token_id(token_id) != 0) {
            (void)pthread_mutex_unlock(&g_kvaf_token_lock);
            return URMA_FAIL;
        }
        state->kernel_allocated = false;
    }
    if (state->ummu_allocated) {
        if (ummu_free_tid(token->tid) != 0) {
            /* A retry skips the completed kernel free. */
            (void)pthread_mutex_unlock(&g_kvaf_token_lock);
            return URMA_FAIL;
        }
        state->ummu_allocated = false;
    }

    kvaf_remove_token_locked(state);
    (void)pthread_mutex_unlock(&g_kvaf_token_lock);
    free(token);
    free(state);
    return URMA_SUCCESS;
}
