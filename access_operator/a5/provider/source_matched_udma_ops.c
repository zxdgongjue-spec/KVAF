#include "kvaf_source_matched_udma.h"

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/mman.h>

/*
 * The upstream context delete frees the userspace context even when the
 * kernel delete command fails. Rename that private implementation and install
 * the checked implementation below into g_udma_provider_ops before provider
 * registration.
 */
#ifndef KVAF_UDMA_OPS_SOURCE
#error "KVAF_UDMA_OPS_SOURCE must name udma_u_ops.c"
#endif
#define udma_u_create_context kvaf_legacy_udma_u_create_context
#define udma_u_delete_context kvaf_legacy_udma_u_delete_context
#include KVAF_UDMA_OPS_SOURCE
#undef udma_u_delete_context
#undef udma_u_create_context

#include "kvaf_source_matched_udma_cleanup.h"

struct kvaf_udma_context_state {
    struct udma_u_context *context;
    bool orphan;
    bool db_list_mutex_initialized;
    bool hugepage_mutex_initialized;
    bool kernel_context_created;
    bool db_allocated;
    bool node_table_initialized;
    struct kvaf_udma_context_state *next;
};

static pthread_mutex_t g_kvaf_context_lock = PTHREAD_MUTEX_INITIALIZER;
static struct kvaf_udma_context_state *g_kvaf_contexts;
static uint32_t g_kvaf_context_count;
static uint32_t g_kvaf_context_orphan_count;

static void kvaf_add_context_locked(
    struct kvaf_udma_context_state *state)
{
    state->next = g_kvaf_contexts;
    g_kvaf_contexts = state;
    ++g_kvaf_context_count;
    if (state->orphan) {
        ++g_kvaf_context_orphan_count;
    }
}

static struct kvaf_udma_context_state *kvaf_find_context_locked(
    struct udma_u_context *context)
{
    struct kvaf_udma_context_state *state;

    for (state = g_kvaf_contexts; state != NULL; state = state->next) {
        if (state->context == context) {
            return state;
        }
    }
    return NULL;
}

static void kvaf_remove_context_locked(
    struct kvaf_udma_context_state *state)
{
    struct kvaf_udma_context_state **cursor = &g_kvaf_contexts;

    while (*cursor != NULL) {
        if (*cursor == state) {
            *cursor = state->next;
            --g_kvaf_context_count;
            if (state->orphan) {
                --g_kvaf_context_orphan_count;
            }
            return;
        }
        cursor = &(*cursor)->next;
    }
}

uint32_t kvaf_udma_source_live_context_count(void)
{
    uint32_t count;

    if (pthread_mutex_lock(&g_kvaf_context_lock) != 0) {
        return UINT32_MAX;
    }
    count = g_kvaf_context_count;
    (void)pthread_mutex_unlock(&g_kvaf_context_lock);
    return count;
}

uint32_t kvaf_udma_source_orphan_context_count(void)
{
    uint32_t count;

    if (pthread_mutex_lock(&g_kvaf_context_lock) != 0) {
        return UINT32_MAX;
    }
    count = g_kvaf_context_orphan_count;
    (void)pthread_mutex_unlock(&g_kvaf_context_lock);
    return count;
}

/*
 * Roll back only a context which create never handed to liburma's caller.
 * Every completed phase is recorded before moving on, so a later retry does
 * not repeat munmap, context deletion, or mutex destruction.
 */
static int kvaf_retry_orphan_context_locked(
    struct kvaf_udma_context_state *state)
{
    struct udma_u_context *context = state->context;

    if (!state->orphan) {
        return 0;
    }
    /* Finish transport ownership before releasing any mapped control page. */
    if (state->kernel_context_created) {
        if (urma_cmd_delete_context(&context->urma_ctx) != 0) {
            return -1;
        }
        state->kernel_context_created = false;
    }
    if (state->node_table_initialized) {
        udma_u_uninit_node_tbl(context->src_idx_tbl);
        state->node_table_initialized = false;
    }
    if (state->db_allocated) {
        if (kvaf_udma_checked_free_doorbell(
                &context->urma_ctx, &context->db) != 0) {
            return -1;
        }
        state->db_allocated = false;
    }
    /* No queue was returned, so a create orphan cannot own hugepage data. */
    if (state->hugepage_mutex_initialized) {
        if (context->hugepage_list != NULL ||
            pthread_mutex_destroy(&context->hugepage_lock) != 0) {
            return -1;
        }
        state->hugepage_mutex_initialized = false;
    }
    if (state->db_list_mutex_initialized) {
        if (kvaf_udma_context_has_db_pages(context) ||
            pthread_mutex_destroy(&context->db_list_mutex) != 0) {
            return -1;
        }
        state->db_list_mutex_initialized = false;
    }
    return 0;
}

int kvaf_udma_source_retry_orphan_contexts(void)
{
    struct kvaf_udma_context_state *state;
    struct kvaf_udma_context_state *next;
    int result = 0;

    if (pthread_mutex_lock(&g_kvaf_context_lock) != 0) {
        return -1;
    }
    for (state = g_kvaf_contexts; state != NULL; state = next) {
        next = state->next;
        if (!state->orphan) {
            continue;
        }
        if (kvaf_retry_orphan_context_locked(state) != 0) {
            result = -1;
            continue;
        }
        kvaf_remove_context_locked(state);
        free(state->context);
        free(state);
    }
    (void)pthread_mutex_unlock(&g_kvaf_context_lock);
    return result;
}

static urma_context_t *kvaf_udma_create_context_checked(
    urma_device_t *device, uint32_t eid_index, int device_fd)
{
    struct udma_create_ctx_resp response = {0};
    urma_cmd_udrv_priv_t private_data = {0};
    urma_context_cfg_t config = {
        .dev = device,
        .ops = &g_udma_ops,
        .eid_index = eid_index,
        .dev_fd = device_fd,
    };
    struct kvaf_udma_context_state *state;
    struct udma_u_context *context;

    if (kvaf_udma_source_provider_owns_device(device) == 0 ||
        kvaf_udma_source_retry_orphan_contexts() != 0) {
        return NULL;
    }
    state = calloc(1, sizeof(*state));
    context = calloc(1, sizeof(*context));
    if (state == NULL || context == NULL ||
        pthread_mutex_lock(&g_kvaf_context_lock) != 0) {
        free(state);
        free(context);
        return NULL;
    }
    state->context = context;
    state->orphan = true;
    kvaf_add_context_locked(state);

    if (pthread_mutex_init(&context->db_list_mutex, NULL) != 0) {
        goto rollback;
    }
    state->db_list_mutex_initialized = true;
    if (pthread_mutex_init(&context->hugepage_lock, NULL) != 0) {
        goto rollback;
    }
    state->hugepage_mutex_initialized = true;
    context->hugepage_list = NULL;

    udma_u_set_udata(&private_data, NULL, 0,
        &response, (uint32_t)sizeof(response));
    if (urma_cmd_create_context(
            &context->urma_ctx, &config, &private_data) != 0) {
        goto rollback;
    }
    state->kernel_context_created = true;
    udma_u_init_context(context, &response);

    if (udma_u_alloc_db(&context->urma_ctx, &context->db) != 0) {
        goto rollback;
    }
    state->db_allocated = true;
    if (udma_u_init_node_tbl(context->src_idx_tbl) != 0) {
        goto rollback;
    }
    state->node_table_initialized = true;
    state->orphan = false;
    --g_kvaf_context_orphan_count;
    (void)pthread_mutex_unlock(&g_kvaf_context_lock);
    return &context->urma_ctx;

rollback:
    if (kvaf_retry_orphan_context_locked(state) == 0) {
        kvaf_remove_context_locked(state);
        free(context);
        free(state);
    }
    (void)pthread_mutex_unlock(&g_kvaf_context_lock);
    return NULL;
}

static urma_status_t kvaf_udma_delete_context_checked(
    urma_context_t *public_context)
{
    struct udma_u_context *context;
    struct kvaf_udma_context_state *state;

    if (public_context == NULL ||
        pthread_mutex_lock(&g_kvaf_context_lock) != 0) {
        return URMA_FAIL;
    }
    context = to_udma_u_ctx(public_context);
    state = kvaf_find_context_locked(context);
    if (state == NULL) {
        (void)pthread_mutex_unlock(&g_kvaf_context_lock);
        return URMA_FAIL;
    }

    /*
     * Retry rollback objects which create returned as NULL.  They cannot be
     * reached through the perftest context and must be drained here, before
     * the normal live-object gate.  Each kind only visits orphan sidecars
     * owned by this exact public context.
     */
    if (kvaf_udma_source_retry_orphan_cleanup(public_context) != 0 ||
        kvaf_udma_source_live_segment_count() != 0U ||
        kvaf_udma_source_live_token_count() != 0U ||
        kvaf_udma_source_live_jetty_count() != 0U ||
        kvaf_udma_source_live_jfc_count() != 0U) {
        (void)pthread_mutex_unlock(&g_kvaf_context_lock);
        return URMA_FAIL;
    }

    /* Kernel/transport deletion is a separate, one-way phase. */
    if (state->kernel_context_created) {
        if (urma_cmd_delete_context(&context->urma_ctx) != 0) {
            /* Keep context, dev_fd and sidecar for an exact retry. */
            (void)pthread_mutex_unlock(&g_kvaf_context_lock);
            return URMA_FAIL;
        }
        state->kernel_context_created = false;
    }

    /* Local mappings are cleared only after their munmap succeeds. */
    if (state->node_table_initialized) {
        udma_u_uninit_node_tbl(context->src_idx_tbl);
        state->node_table_initialized = false;
    }
    if (state->db_allocated) {
        if (kvaf_udma_checked_free_doorbell(
                public_context, &context->db) != 0) {
            (void)pthread_mutex_unlock(&g_kvaf_context_lock);
            return URMA_FAIL;
        }
        state->db_allocated = false;
    }
    if (state->hugepage_mutex_initialized) {
        if (context->hugepage_list != NULL ||
            pthread_mutex_destroy(&context->hugepage_lock) != 0) {
            (void)pthread_mutex_unlock(&g_kvaf_context_lock);
            return URMA_FAIL;
        }
        state->hugepage_mutex_initialized = false;
    }

    if (state->db_list_mutex_initialized) {
        if (kvaf_udma_context_has_db_pages(context) ||
            pthread_mutex_destroy(&context->db_list_mutex) != 0) {
            (void)pthread_mutex_unlock(&g_kvaf_context_lock);
            return URMA_FAIL;
        }
        state->db_list_mutex_initialized = false;
    }
    kvaf_remove_context_locked(state);
    (void)pthread_mutex_unlock(&g_kvaf_context_lock);
    free(context);
    free(state);
    return URMA_SUCCESS;
}

int kvaf_udma_source_retry_orphan_cleanup(urma_context_t *context)
{
    int result = 0;

    if (context == NULL) {
        return -1;
    }
    if (kvaf_udma_source_retry_orphan_jetties(context) != 0) {
        result = -1;
    }
    if (kvaf_udma_source_retry_orphan_jfcs(context) != 0) {
        result = -1;
    }
    if (kvaf_udma_source_retry_orphan_segments(context) != 0) {
        result = -1;
    }
    if (kvaf_udma_source_retry_orphan_tokens(context) != 0) {
        result = -1;
    }
    return result;
}

int kvaf_udma_prepare_provider_ops(void)
{
    g_udma_ops.create_jetty = kvaf_udma_create_jetty_checked;
    g_udma_ops.delete_jetty = kvaf_udma_delete_jetty_checked;
    /* Upstream batch delete mutates every SQ lookup before the ioctl. */
    g_udma_ops.delete_jetty_batch = NULL;
    g_udma_ops.create_jfc = kvaf_udma_create_jfc_checked;
    g_udma_ops.delete_jfc = kvaf_udma_delete_jfc_checked;
    g_udma_provider_ops.create_context =
        kvaf_udma_create_context_checked;
    g_udma_provider_ops.delete_context =
        kvaf_udma_delete_context_checked;
    return g_udma_provider_ops.create_context ==
                kvaf_udma_create_context_checked &&
            g_udma_provider_ops.delete_context ==
                kvaf_udma_delete_context_checked &&
            g_udma_ops.create_jetty ==
                kvaf_udma_create_jetty_checked &&
            g_udma_ops.delete_jetty ==
                kvaf_udma_delete_jetty_checked &&
            g_udma_ops.create_jfc ==
                kvaf_udma_create_jfc_checked &&
            g_udma_ops.delete_jfc ==
                kvaf_udma_delete_jfc_checked
        ? 0
        : -1;
}

int kvaf_udma_source_provider_owns_context(
    const urma_context_t *public_context)
{
    return public_context != NULL &&
        kvaf_udma_source_provider_is_registered() != 0 &&
        kvaf_udma_source_runtime_is_owned() != 0 &&
        public_context->dev != NULL &&
        public_context->dev->ops == &g_udma_provider_ops;
}
