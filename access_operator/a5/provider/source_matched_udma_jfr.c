#include "kvaf_source_matched_udma.h"

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/mman.h>

/*
 * UMDK's ordinary JFR teardown releases the table entry, two software
 * doorbells, the RQ, and the index queue with void helpers.  A failed munmap
 * therefore loses the only retry handle.  Keep the official implementation
 * for all non-direct operations, but replace direct create/delete with an
 * ownership sidecar whose phases advance only after checked cleanup.
 */
#ifndef KVAF_UDMA_JFR_SOURCE
#error "KVAF_UDMA_JFR_SOURCE must name udma_u_jfr.c"
#endif
#define udma_u_create_jfr kvaf_legacy_udma_u_create_jfr
#define udma_u_delete_jfr kvaf_legacy_udma_u_delete_jfr
#define udma_u_delete_jfr_batch kvaf_legacy_udma_u_delete_jfr_batch
#include KVAF_UDMA_JFR_SOURCE
#undef udma_u_delete_jfr_batch
#undef udma_u_delete_jfr
#undef udma_u_create_jfr

#include "kvaf_source_matched_udma_cleanup.h"

struct kvaf_udma_jfr_state {
    struct udma_u_jfr *jfr;
    urma_context_t *owner_context;
    bool kernel_created;
    bool spin_initialized;
    bool index_bitmap_allocated;
    bool index_buffer_mapped;
    bool rq_created;
    bool sw_db_allocated;
    bool sleep_db_allocated;
    bool table_inserted;
    bool table_lock_unknown;
    bool jfc_flushed;
    bool orphan;
    struct kvaf_udma_jfr_state *next;
};

static pthread_mutex_t g_kvaf_jfr_lock = PTHREAD_MUTEX_INITIALIZER;
static struct kvaf_udma_jfr_state *g_kvaf_jfrs;
static uint32_t g_kvaf_jfr_count;
static uint32_t g_kvaf_jfr_orphan_count;

static int kvaf_unlock_jfr_table_checked(
    struct kvaf_udma_jfr_state *state,
    struct udma_u_context *context)
{
    if (pthread_rwlock_unlock(&context->jfr_table_lock) == 0) {
        return 0;
    }
    /* Never retry a lock whose ownership after unlock is unknowable. */
    state->table_lock_unknown = true;
    kvaf_udma_source_mark_cleanup_unknown();
    return -1;
}

static struct kvaf_udma_jfr_state *kvaf_find_jfr_locked(
    struct udma_u_jfr *jfr)
{
    struct kvaf_udma_jfr_state *state;

    for (state = g_kvaf_jfrs; state != NULL; state = state->next) {
        if (state->jfr == jfr) {
            return state;
        }
    }
    return NULL;
}

static void kvaf_remove_jfr_locked(struct kvaf_udma_jfr_state *state)
{
    struct kvaf_udma_jfr_state **cursor = &g_kvaf_jfrs;

    while (*cursor != NULL) {
        if (*cursor == state) {
            *cursor = state->next;
            --g_kvaf_jfr_count;
            if (state->orphan) {
                --g_kvaf_jfr_orphan_count;
            }
            return;
        }
        cursor = &(*cursor)->next;
    }
}

uint32_t kvaf_udma_source_live_jfr_count(void)
{
    uint32_t count;

    if (pthread_mutex_lock(&g_kvaf_jfr_lock) != 0) {
        return UINT32_MAX;
    }
    count = g_kvaf_jfr_count;
    (void)pthread_mutex_unlock(&g_kvaf_jfr_lock);
    return count;
}

uint32_t kvaf_udma_source_orphan_jfr_count(void)
{
    uint32_t count;

    if (pthread_mutex_lock(&g_kvaf_jfr_lock) != 0) {
        return UINT32_MAX;
    }
    count = g_kvaf_jfr_orphan_count;
    (void)pthread_mutex_unlock(&g_kvaf_jfr_lock);
    return count;
}

static int kvaf_alloc_jfr_index_checked(
    struct kvaf_udma_jfr_state *state)
{
    struct udma_u_jfr_idx_que *index = &state->jfr->idx_que;
    struct udma_u_context *context = state->jfr->rq.ctx;
    uint64_t bytes;
    uint64_t aligned_bytes;
    off_t offset;

    if (context == NULL || context->urma_ctx.dev_fd < 0 ||
        state->jfr->wqe_cnt == 0U ||
        UDMA_JFR_IDX_QUE_ENTRY_SZ == 0U ||
        (UDMA_JFR_IDX_QUE_ENTRY_SZ &
            (UDMA_JFR_IDX_QUE_ENTRY_SZ - 1U)) != 0U) {
        return -1;
    }
    index->entry_shift =
        (uint32_t)__builtin_ctz(UDMA_JFR_IDX_QUE_ENTRY_SZ);
    index->cstm = false;
    index->bitmap = udma_bitmap_alloc(
        state->jfr->wqe_cnt, &index->bitmap_cnt);
    if (index->bitmap == NULL) {
        return -1;
    }
    state->index_bitmap_allocated = true;

    bytes = (uint64_t)state->jfr->wqe_cnt << index->entry_shift;
    if (bytes == 0U || bytes > UINT32_MAX -
            (UDMA_HW_PAGE_SIZE - 1U)) {
        return -1;
    }
    aligned_bytes = (bytes + UDMA_HW_PAGE_SIZE - 1U) &
        ~((uint64_t)UDMA_HW_PAGE_SIZE - 1U);
    if (aligned_bytes == 0U || aligned_bytes > UINT32_MAX) {
        return -1;
    }
    index->buf.length = (uint32_t)aligned_bytes;
    offset = get_mmap_offset(0, context->page_size,
        UDMA_MMAP_KERNEL_BUF);
    index->buf.buf = mmap(NULL, index->buf.length,
        PROT_READ | PROT_WRITE, MAP_SHARED,
        context->urma_ctx.dev_fd, offset);
    if (index->buf.buf == MAP_FAILED) {
        index->buf.buf = NULL;
        return -1;
    }
    state->index_buffer_mapped = true;
    if (madvise(index->buf.buf, index->buf.length,
            MADV_DONTFORK) != 0) {
        /* The sidecar retains this mapping for exact cleanup retry. */
        return -1;
    }
    return 0;
}

static int kvaf_free_jfr_index_checked(
    struct kvaf_udma_jfr_state *state)
{
    struct udma_u_jfr_idx_que *index = &state->jfr->idx_que;

    if (index->cstm) {
        return -1;
    }
    if (state->index_buffer_mapped) {
        if (index->buf.buf == NULL || index->buf.length == 0U ||
            munmap(index->buf.buf, index->buf.length) != 0) {
            return -1;
        }
        index->buf.buf = NULL;
        index->buf.length = 0U;
        state->index_buffer_mapped = false;
    }
    if (state->index_bitmap_allocated) {
        if (index->bitmap == NULL) {
            return -1;
        }
        udma_bitmap_free(index->bitmap);
        index->bitmap = NULL;
        index->bitmap_cnt = 0U;
        state->index_bitmap_allocated = false;
    }
    return 0;
}

static int kvaf_remove_jfr_table_checked(
    struct kvaf_udma_jfr_state *state)
{
    struct udma_u_jfr *jfr = state->jfr;
    struct udma_u_context *context = jfr->rq.ctx;
    struct udma_u_jfr **array;
    uint32_t table_id;
    uint32_t slot;

    if (!state->table_inserted) {
        return 0;
    }
    if (state->table_lock_unknown) {
        return -1;
    }
    if (context == NULL || context->jettys_in_tbl_shift == 0U ||
        context->jettys_in_tbl == 0U) {
        return -1;
    }
    table_id = jfr->rq.idx >> context->jettys_in_tbl_shift;
    slot = jfr->rq.idx & (context->jettys_in_tbl - 1U);
    if (table_id >= UDMA_JETTY_TABLE_NUM ||
        pthread_rwlock_wrlock(&context->jfr_table_lock) != 0) {
        return -1;
    }
    array = context->jfr_table[table_id].jfr_array;
    if (context->jfr_table[table_id].refcnt <= 0 ||
        array == NULL || array[slot] != jfr) {
        (void)kvaf_unlock_jfr_table_checked(state, context);
        return -1;
    }
    array[slot] = NULL;
    --context->jfr_table[table_id].refcnt;
    if (context->jfr_table[table_id].refcnt == 0) {
        free(array);
        context->jfr_table[table_id].jfr_array = NULL;
    }
    if (kvaf_unlock_jfr_table_checked(state, context) != 0) {
        /* Never re-enter a lock whose ownership state is not provable. */
        return -1;
    }
    state->table_inserted = false;
    return 0;
}

static int kvaf_insert_jfr_table_checked(
    struct kvaf_udma_jfr_state *state)
{
    struct udma_u_jfr *jfr = state->jfr;
    struct udma_u_context *context = jfr->rq.ctx;
    struct udma_u_jfr **array;
    uint32_t table_id;
    uint32_t slot;

    if (context == NULL || context->jettys_in_tbl_shift == 0U ||
        context->jettys_in_tbl == 0U ||
        (context->jettys_in_tbl &
         (context->jettys_in_tbl - 1U)) != 0U) {
        return -1;
    }
    table_id = jfr->rq.idx >> context->jettys_in_tbl_shift;
    slot = jfr->rq.idx & (context->jettys_in_tbl - 1U);
    if (table_id >= UDMA_JETTY_TABLE_NUM ||
        pthread_rwlock_wrlock(&context->jfr_table_lock) != 0) {
        return -1;
    }
    array = context->jfr_table[table_id].jfr_array;
    if (context->jfr_table[table_id].refcnt == 0) {
        if (array != NULL) {
            (void)kvaf_unlock_jfr_table_checked(state, context);
            return -1;
        }
        array = calloc(context->jettys_in_tbl, sizeof(*array));
        if (array == NULL) {
            (void)kvaf_unlock_jfr_table_checked(state, context);
            return -1;
        }
        context->jfr_table[table_id].jfr_array = array;
    } else if (array == NULL) {
        (void)kvaf_unlock_jfr_table_checked(state, context);
        return -1;
    }
    if (array[slot] != NULL) {
        if (context->jfr_table[table_id].refcnt == 0) {
            free(array);
            context->jfr_table[table_id].jfr_array = NULL;
        }
        (void)kvaf_unlock_jfr_table_checked(state, context);
        return -1;
    }
    array[slot] = jfr;
    ++context->jfr_table[table_id].refcnt;
    state->table_inserted = true;
    if (kvaf_unlock_jfr_table_checked(state, context) != 0) {
        /* Mutation happened; retain it, but never risk re-lock deadlock. */
        return -1;
    }
    return 0;
}

static int kvaf_create_direct_jfr_command(
    urma_context_t *context, struct udma_u_jfr *jfr,
    urma_jfr_cfg_t *config, bool *kernel_created)
{
    struct udma_create_jetty_ucmd command = {0};
    struct udma_create_jfr_resp response = {0};
    urma_cmd_udrv_priv_t udata = {0};
    int status;

    *kernel_created = false;
    command.buf_addr = (uintptr_t)jfr->rq.qbuf;
    command.buf_len = jfr->rq.qbuf_size;
    command.db_addr = (uintptr_t)jfr->sw_db;
    command.jfr_sleep_buf = (uintptr_t)jfr->long_sleeptime;
    command.idx_addr = (uintptr_t)jfr->idx_que.buf.buf;
    command.idx_len = jfr->idx_que.buf.length;
    command.jetty_addr = (uintptr_t)&jfr->rq;
    command.non_pin = false;
    command.dtu_en = false;
    command.is_hugepage = jfr->rq.hugepage != NULL;
    udma_u_set_udata(&udata, &command, (uint32_t)sizeof(command),
        &response, (uint32_t)sizeof(response));
    status = urma_cmd_create_jfr(
        context, &jfr->base, config, &udata);
    if (status != 0) {
        return status;
    }
    *kernel_created = true;
    jfr->cap_flags = response.jfr_caps;
    jfr->rq.idx = jfr->base.jfr_id.id;
    return 0;
}

static int kvaf_cleanup_local_jfr_checked(
    struct kvaf_udma_jfr_state *state)
{
    struct udma_u_jfr *jfr = state->jfr;
    struct udma_u_context *context = jfr->rq.ctx;

    if (!state->orphan && !state->jfc_flushed) {
        if (jfr->base.jfr_cfg.jfc != NULL) {
            udma_u_clean_jfc(
                jfr->base.jfr_cfg.jfc, jfr->base.jfr_id.id);
        }
        state->jfc_flushed = true;
    }
    if (kvaf_remove_jfr_table_checked(state) != 0) {
        return -1;
    }
    if (state->sleep_db_allocated) {
        if (kvaf_udma_checked_free_sw_db(context,
                (uint32_t *)jfr->long_sleeptime,
                UDMA_JFR_PAYLOAD) != 0) {
            return -1;
        }
        jfr->long_sleeptime = NULL;
        state->sleep_db_allocated = false;
    }
    if (state->sw_db_allocated) {
        if (kvaf_udma_checked_free_sw_db(context, jfr->sw_db,
                UDMA_JFR_TYPE_DB) != 0) {
            return -1;
        }
        jfr->sw_db = NULL;
        state->sw_db_allocated = false;
    }
    if (state->rq_created) {
        if (kvaf_udma_checked_free_queue_buffer(&jfr->rq) != 0) {
            return -1;
        }
        state->rq_created = false;
    }
    if (kvaf_free_jfr_index_checked(state) != 0) {
        return -1;
    }
    if (state->spin_initialized) {
        if (pthread_spin_destroy(&jfr->lock) != 0) {
            return -1;
        }
        state->spin_initialized = false;
    }
    return 0;
}

int kvaf_udma_source_retry_orphan_jfrs(urma_context_t *context)
{
    struct kvaf_udma_jfr_state *state;
    struct kvaf_udma_jfr_state *next;
    int result = 0;

    if (context == NULL || pthread_mutex_lock(&g_kvaf_jfr_lock) != 0) {
        return -1;
    }
    for (state = g_kvaf_jfrs; state != NULL; state = next) {
        next = state->next;
        if (!state->orphan || state->owner_context != context) {
            continue;
        }
        if (state->kernel_created &&
            urma_cmd_delete_jfr(&state->jfr->base) != 0) {
            result = -1;
            continue;
        }
        state->kernel_created = false;
        if (kvaf_cleanup_local_jfr_checked(state) != 0) {
            result = -1;
            continue;
        }
        kvaf_remove_jfr_locked(state);
        free(state->jfr);
        free(state);
    }
    (void)pthread_mutex_unlock(&g_kvaf_jfr_lock);
    return result;
}

urma_jfr_t *kvaf_udma_create_jfr_checked(
    urma_context_t *context, urma_jfr_cfg_t *config)
{
    struct kvaf_udma_jfr_state *state;
    struct udma_u_context *udma_context;
    struct udma_u_jfr *jfr;
    uint32_t *sleep_db = NULL;
    uint32_t sge_per_wqe;
    uint32_t wqebb_count;
    int status;

    state = calloc(1, sizeof(*state));
    jfr = calloc(1, sizeof(*jfr));
    if (state == NULL || jfr == NULL || context == NULL || config == NULL ||
        pthread_mutex_lock(&g_kvaf_jfr_lock) != 0) {
        free(state);
        free(jfr);
        return NULL;
    }
    state->jfr = jfr;
    state->owner_context = context;
    state->orphan = true;
    state->next = g_kvaf_jfrs;
    g_kvaf_jfrs = state;
    ++g_kvaf_jfr_count;
    ++g_kvaf_jfr_orphan_count;

    if (config->jfc == NULL || config->jfc->urma_ctx != context ||
        config->flag.bs.non_blocking ||
        config->flag.bs.tag_matching != URMA_NO_TAG_MATCHING ||
        config->trans_mode != URMA_TM_RM ||
        udma_u_verify_jfr_param(context, config) != 0) {
        goto fail;
    }
    udma_context = to_udma_u_ctx(context);
    if (udma_context->dtu_enable || udma_context->sq_reserved ||
        udma_context->jettys_in_tbl_shift == 0U ||
        udma_context->jettys_in_tbl == 0U) {
        goto fail;
    }
    jfr->rq.ctx = udma_context;
    jfr->rq.cstm = false;
    jfr->rq.sq_reserved = false;
    jfr->rq.dtu_en = false;
    jfr->rq.is_jetty = false;
    jfr->rq.aligned_size = UDMA_HW_PAGE_SIZE;
    jfr->idx_que.cstm = false;
    jfr->swdb_cstm = false;
    udma_u_init_jfr_param(jfr, config);

    if (!jfr->lock_free) {
        if (pthread_spin_init(&jfr->lock,
                PTHREAD_PROCESS_PRIVATE) != 0) {
            goto fail;
        }
        state->spin_initialized = true;
    }
    if (kvaf_alloc_jfr_index_checked(state) != 0) {
        goto fail;
    }
    sge_per_wqe = UDMA_MIN(jfr->max_sge, udma_context->jfr_sge);
    if (sge_per_wqe == 0U ||
        jfr->wqe_cnt > UINT32_MAX / sge_per_wqe) {
        goto fail;
    }
    wqebb_count = sge_per_wqe * jfr->wqe_cnt;
    state->rq_created = true;
    if (kvaf_udma_checked_alloc_queue_buffer(&jfr->rq,
            wqebb_count, UDMA_JFR_WQEBB,
            UDMA_HW_PAGE_SIZE, true) != 0) {
        goto fail;
    }
    status = kvaf_udma_checked_alloc_sw_db(
        udma_context, UDMA_JFR_TYPE_DB, &jfr->sw_db);
    if (jfr->sw_db != NULL) {
        state->sw_db_allocated = true;
    }
    if (status != 0) {
        goto fail;
    }
    status = kvaf_udma_checked_alloc_sw_db(
        udma_context, UDMA_JFR_PAYLOAD, &sleep_db);
    jfr->long_sleeptime = (bool *)sleep_db;
    if (sleep_db != NULL) {
        state->sleep_db_allocated = true;
    }
    if (status != 0) {
        goto fail;
    }
    *jfr->long_sleeptime = false;

    status = kvaf_create_direct_jfr_command(
        context, jfr, config, &state->kernel_created);
    if (status != 0) {
        goto fail;
    }
    if (kvaf_insert_jfr_table_checked(state) != 0) {
        goto fail;
    }
    state->orphan = false;
    --g_kvaf_jfr_orphan_count;
    (void)pthread_mutex_unlock(&g_kvaf_jfr_lock);
    return &jfr->base;

fail:
    if (state->kernel_created &&
        urma_cmd_delete_jfr(&jfr->base) != 0) {
        (void)pthread_mutex_unlock(&g_kvaf_jfr_lock);
        return NULL;
    }
    state->kernel_created = false;
    if (kvaf_cleanup_local_jfr_checked(state) == 0) {
        kvaf_remove_jfr_locked(state);
        free(jfr);
        free(state);
    }
    (void)pthread_mutex_unlock(&g_kvaf_jfr_lock);
    return NULL;
}

urma_status_t kvaf_udma_delete_jfr_checked(urma_jfr_t *public_jfr)
{
    struct kvaf_udma_jfr_state *state;
    struct udma_u_jfr *jfr;

    if (public_jfr == NULL ||
        pthread_mutex_lock(&g_kvaf_jfr_lock) != 0) {
        return URMA_FAIL;
    }
    jfr = to_udma_u_jfr(public_jfr);
    state = kvaf_find_jfr_locked(jfr);
    if (state == NULL || public_jfr->jfr_cfg.flag.bs.non_blocking ||
        jfr->rq.cstm || jfr->idx_que.cstm || jfr->swdb_cstm ||
        jfr->rq.dtu_en || jfr->rq.sq_reserved) {
        (void)pthread_mutex_unlock(&g_kvaf_jfr_lock);
        return URMA_FAIL;
    }
    if (state->kernel_created) {
        if (urma_cmd_delete_jfr(public_jfr) != 0) {
            (void)pthread_mutex_unlock(&g_kvaf_jfr_lock);
            return URMA_FAIL;
        }
        state->kernel_created = false;
    }
    if (kvaf_cleanup_local_jfr_checked(state) != 0) {
        /* Kernel JFR is gone; retain all remaining local retry handles. */
        (void)pthread_mutex_unlock(&g_kvaf_jfr_lock);
        return URMA_FAIL;
    }
    kvaf_remove_jfr_locked(state);
    (void)pthread_mutex_unlock(&g_kvaf_jfr_lock);
    free(jfr);
    free(state);
    return URMA_SUCCESS;
}

urma_jfr_t *udma_u_create_jfr(
    urma_context_t *context, urma_jfr_cfg_t *config)
{
    return kvaf_udma_create_jfr_checked(context, config);
}

urma_status_t udma_u_delete_jfr(urma_jfr_t *jfr)
{
    return kvaf_udma_delete_jfr_checked(jfr);
}

urma_status_t udma_u_delete_jfr_batch(
    urma_jfr_t **jfr, int jfr_count, urma_jfr_t **bad_jfr)
{
    (void)jfr;
    (void)jfr_count;
    (void)bad_jfr;
    return URMA_ENOPERM;
}
