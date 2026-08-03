#ifndef KVAF_SOURCE_MATCHED_UDMA_CLEANUP_H
#define KVAF_SOURCE_MATCHED_UDMA_CLEANUP_H

#include <errno.h>
#include <poll.h>
#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/types.h>

#include "udma_u_buf.h"
#include "udma_u_db.h"
#include "urma_api.h"
#include "kvaf_udma_destroy_policy.h"

/*
 * Destroy ioctls report how many asynchronous events must be acknowledged
 * before their userspace objects can be released.  The upstream blocking
 * helpers wait on a condition variable forever, but the direct AICPU owner
 * has no background event thread.  Drain a bounded number on the owner
 * thread instead; a remaining backlog is resumed by the next Close call.
 *
 * The caller must keep every referenced object alive until its captured
 * reported counters equal the corresponding acked counters.
 */
#define KVAF_UDMA_ASYNC_DRAIN_BUDGET 32U

static inline bool kvaf_udma_destroy_ioctl_retry_is_proven(
    int operation_status, int operation_errno)
{
    return operation_status == EAGAIN || operation_status == EBUSY ||
        operation_errno == EAGAIN || operation_errno == EBUSY;
}

static inline bool kvaf_udma_cleanup_errno_is_retryable(int error_code)
{
    return error_code == EAGAIN || error_code == EBUSY;
}

static inline void kvaf_udma_preserve_cleanup_errno(int *saved_errno)
{
    const int error_code = errno == 0 ? EIO : errno;

    if (saved_errno == NULL) {
        return;
    }
    if (*saved_errno == 0 ||
        (kvaf_udma_cleanup_errno_is_retryable(*saved_errno) &&
         !kvaf_udma_cleanup_errno_is_retryable(error_code))) {
        *saved_errno = error_code;
    }
}

static inline int kvaf_udma_cleanup_invariant_failed(void)
{
    errno = EUCLEAN;
    kvaf_udma_source_mark_cleanup_unknown();
    return -1;
}

static inline int kvaf_udma_pthread_cleanup_failed(int status)
{
    if (status == 0) {
        return 0;
    }
    errno = status;
    /* A local mutex/condition/lock destroy has no device-side event source
     * that can make EBUSY progress.  Retrying it forever would only consume
     * the AICPU watchdog budget. */
    kvaf_udma_source_mark_cleanup_unknown();
    return -1;
}

static inline int kvaf_udma_classify_cleanup_failure(void)
{
    const int error_code = errno == 0 ? EIO : errno;

    errno = error_code;
    /* This helper is used only after local mapping/bookkeeping cleanup.  No
     * asynchronous producer owns those phases, so even EAGAIN/EBUSY is an
     * invariant failure rather than a resumable device operation. */
    kvaf_udma_source_mark_cleanup_unknown();
    return -1;
}

static inline int kvaf_udma_snapshot_event_counters_checked(
    pthread_mutex_t *mutex,
    const uint32_t *comp_events_acked,
    const uint32_t *async_events_acked,
    uint32_t *comp_events_out,
    uint32_t *async_events_out)
{
    int status;

    if (mutex == NULL || async_events_acked == NULL ||
        async_events_out == NULL ||
        ((comp_events_acked == NULL) != (comp_events_out == NULL))) {
        return kvaf_udma_cleanup_invariant_failed();
    }
    status = pthread_mutex_lock(mutex);
    if (status != 0) {
        return kvaf_udma_pthread_cleanup_failed(status);
    }
    if (comp_events_out != NULL) {
        *comp_events_out = *comp_events_acked;
    }
    *async_events_out = *async_events_acked;
    status = pthread_mutex_unlock(mutex);
    if (status != 0) {
        return kvaf_udma_pthread_cleanup_failed(status);
    }
    return 0;
}

static inline int kvaf_udma_destroy_event_sync_checked(
    pthread_mutex_t *mutex, pthread_cond_t *condition,
    bool *condition_initialized, bool *mutex_initialized)
{
    int status;

    if (mutex == NULL || condition == NULL ||
        condition_initialized == NULL || mutex_initialized == NULL) {
        return kvaf_udma_cleanup_invariant_failed();
    }
    if (*condition_initialized) {
        status = pthread_cond_destroy(condition);
        if (status != 0) {
            return kvaf_udma_pthread_cleanup_failed(status);
        }
        *condition_initialized = false;
    }
    if (*mutex_initialized) {
        status = pthread_mutex_destroy(mutex);
        if (status != 0) {
            return kvaf_udma_pthread_cleanup_failed(status);
        }
        *mutex_initialized = false;
    }
    return 0;
}

static inline int kvaf_udma_drain_async_events_bounded(
    urma_context_t *context)
{
    uint32_t drained;

    if (context == NULL || context->async_fd < 0) {
        errno = EINVAL;
        kvaf_udma_source_mark_cleanup_unknown();
        return -1;
    }
    for (drained = 0; drained < KVAF_UDMA_ASYNC_DRAIN_BUDGET;
         ++drained) {
        struct pollfd descriptor = {
            .fd = context->async_fd,
            .events = POLLIN,
        };
        urma_async_event_t event = {0};
        int poll_status;
        int poll_errno;
        enum kvaf_udma_async_poll_decision decision;

        errno = 0;
        poll_status = poll(&descriptor, 1, 0);
        poll_errno = errno;
        decision = kvaf_udma_async_poll_decide(
            poll_status, descriptor.revents, poll_errno);
        if (decision == KVAF_UDMA_ASYNC_POLL_IDLE) {
            return 0;
        }
        if (decision == KVAF_UDMA_ASYNC_POLL_RETRY_LATER) {
            errno = EAGAIN;
            return 0;
        }
        if (decision != KVAF_UDMA_ASYNC_POLL_READ_ONE) {
            errno = poll_errno == 0 ? EIO : poll_errno;
            kvaf_udma_source_mark_cleanup_unknown();
            return -1;
        }
        errno = 0;
        if (urma_get_async_event(context, &event) != URMA_SUCCESS) {
            if (errno == EAGAIN) {
                return 0;
            }
            if (errno == 0) {
                errno = EIO;
            }
            kvaf_udma_source_mark_cleanup_unknown();
            return -1;
        }
        urma_ack_async_event(&event);
    }
    return 0;
}

static inline int kvaf_udma_checked_alloc_hugepage_queue(
    struct udma_u_jetty_queue *queue)
{
    struct udma_u_hugepage_priv *mapping = NULL;
    struct udma_u_hugepage *hugepage;
    uint32_t mapping_length;
    uint32_t mapping_index;
    off_t offset;
    bool mapping_created = false;

    if (queue == NULL || queue->ctx == NULL || queue->qbuf_size == 0U ||
        queue->ctx->page_size < UDMA_HW_PAGE_SIZE ||
        queue->ctx->page_size > UDMA_HUGEPAGE_SIZE ||
        UDMA_HUGEPAGE_SIZE % queue->ctx->page_size != 0U ||
        queue->qbuf_size > UINT32_MAX - (UDMA_HUGEPAGE_SIZE - 1U)) {
        return -1;
    }
    mapping_length = (queue->qbuf_size + UDMA_HUGEPAGE_SIZE - 1U) &
        ~(UDMA_HUGEPAGE_SIZE - 1U);
    hugepage = calloc(1, sizeof(*hugepage));
    if (hugepage == NULL ||
        pthread_mutex_lock(&queue->ctx->hugepage_lock) != 0) {
        free(hugepage);
        return -1;
    }
    if (queue->ctx->hugepage_list != NULL &&
        queue->ctx->hugepage_list->left_va_len >= queue->qbuf_size) {
        mapping = queue->ctx->hugepage_list;
    } else {
        mapping = calloc(1, sizeof(*mapping));
        if (mapping == NULL) {
            goto failure;
        }
        mapping->va_len = mapping_length;
        mapping->left_va_len = mapping_length;
        mapping_index =
            (UDMA_HUGEPAGE_SIZE / queue->ctx->page_size) >>
            MAP_INDEX_SHIFT;
        offset = get_mmap_offset(mapping_index,
            queue->ctx->page_size, UDMA_MMAP_HUGEPAGE);
        mapping->va_base = mmap(NULL, mapping->va_len,
            PROT_READ | PROT_WRITE, MAP_SHARED,
            queue->ctx->urma_ctx.dev_fd, offset);
        if (mapping->va_base == MAP_FAILED) {
            free(mapping);
            goto failure;
        }
        mapping->next = queue->ctx->hugepage_list;
        if (mapping->next != NULL) {
            mapping->next->pre = mapping;
        }
        queue->ctx->hugepage_list = mapping;
        mapping_created = true;
    }
    if (mapping->refcnt == UINT32_MAX) {
        /* A newly created mapping always starts at zero. */
        (void)pthread_mutex_unlock(&queue->ctx->hugepage_lock);
        free(hugepage);
        return -1;
    }
    hugepage->va_start = (char *)mapping->va_base +
        mapping->left_va_offset;
    hugepage->va_len = queue->qbuf_size;
    hugepage->priv = mapping;
    mapping->left_va_offset += queue->qbuf_size;
    mapping->left_va_len -= queue->qbuf_size;
    ++mapping->refcnt;
    queue->hugepage = hugepage;
    queue->qbuf = hugepage->va_start;
    queue->qbuf_curr = queue->qbuf;
    queue->qbuf_end = (char *)queue->qbuf + queue->qbuf_size;
    if (mapping_created &&
        madvise(mapping->va_base, mapping->va_len, MADV_DONTFORK) != 0) {
        /* The linked mapping and queue handle make this failure retryable. */
        mapping->left_va_len = 0U;
        (void)pthread_mutex_unlock(&queue->ctx->hugepage_lock);
        return -1;
    }
    (void)pthread_mutex_unlock(&queue->ctx->hugepage_lock);
    return 0;

failure:
    (void)pthread_mutex_unlock(&queue->ctx->hugepage_lock);
    free(hugepage);
    return -1;
}

enum kvaf_udma_queue_memory_policy {
    KVAF_UDMA_QUEUE_MEMORY_AUTO = 0,
    KVAF_UDMA_QUEUE_MEMORY_KERNEL_BUFFER,
    KVAF_UDMA_QUEUE_MEMORY_USER_ANONYMOUS,
};

static inline int kvaf_udma_checked_alloc_queue_buffer_policy(
    struct udma_u_jetty_queue *queue, uint32_t max_entry_count,
    uint32_t base_block_size, uint32_t page_size, bool allocate_wrid,
    enum kvaf_udma_queue_memory_policy memory_policy)
{
    uint64_t required_bytes;
    uint64_t rounded_bytes = 1U;
    uint64_t aligned_bytes;

    if (queue == NULL || queue->ctx == NULL ||
        queue->ctx->urma_ctx.dev_fd < 0 || max_entry_count == 0U ||
        base_block_size == 0U || page_size == 0U ||
        (base_block_size & (base_block_size - 1U)) != 0U ||
        (page_size & (page_size - 1U)) != 0U ||
        (memory_policy != KVAF_UDMA_QUEUE_MEMORY_AUTO &&
         memory_policy != KVAF_UDMA_QUEUE_MEMORY_KERNEL_BUFFER &&
         memory_policy != KVAF_UDMA_QUEUE_MEMORY_USER_ANONYMOUS)) {
        return -1;
    }
    if (queue->cstm || queue->dtu_en || queue->sq_reserved) {
        return -1;
    }
    required_bytes = (uint64_t)max_entry_count * base_block_size;
    if (required_bytes == 0U || required_bytes > UINT32_MAX) {
        return -1;
    }
    while (rounded_bytes < required_bytes) {
        if (rounded_bytes > UINT32_MAX / 2U) {
            return -1;
        }
        rounded_bytes <<= 1U;
    }
    if (rounded_bytes > UINT32_MAX - (page_size - 1U)) {
        return -1;
    }
    aligned_bytes = (rounded_bytes + page_size - 1U) &
        ~((uint64_t)page_size - 1U);
    if (aligned_bytes == 0U || aligned_bytes > UINT32_MAX) {
        return -1;
    }
    queue->baseblk_shift = (uint32_t)__builtin_ctz(base_block_size);
    queue->qbuf_size = (uint32_t)aligned_bytes;
    queue->baseblk_cnt = queue->qbuf_size >> queue->baseblk_shift;
    if (queue->baseblk_cnt == 0U) {
        return -1;
    }
    queue->baseblk_mask = queue->baseblk_cnt - 1U;
    queue->aligned_size = page_size;
    if (allocate_wrid) {
        queue->wrid = calloc(queue->baseblk_cnt, sizeof(*queue->wrid));
        if (queue->wrid == NULL) {
            return -1;
        }
    }
    if (memory_policy == KVAF_UDMA_QUEUE_MEMORY_AUTO &&
        queue->ctx->hugepage_enable) {
        if (kvaf_udma_checked_alloc_hugepage_queue(queue) != 0) {
            return -1;
        }
    } else {
        const bool user_anonymous =
            memory_policy == KVAF_UDMA_QUEUE_MEMORY_USER_ANONYMOUS;
        const off_t offset = user_anonymous ? 0 : get_mmap_offset(0,
            queue->ctx->page_size, UDMA_MMAP_KERNEL_BUF);
        const int flags = user_anonymous ?
            (MAP_PRIVATE | MAP_ANONYMOUS) : MAP_SHARED;
        const int fd = user_anonymous ? -1 :
            queue->ctx->urma_ctx.dev_fd;

        queue->qbuf = mmap(NULL, queue->qbuf_size,
            PROT_READ | PROT_WRITE, flags, fd, offset);
        if (queue->qbuf == MAP_FAILED) {
            queue->qbuf = NULL;
            free(queue->wrid);
            queue->wrid = NULL;
            return -1;
        }
        queue->qbuf_curr = queue->qbuf;
        queue->qbuf_end = (char *)queue->qbuf + queue->qbuf_size;
        if (madvise(queue->qbuf, queue->qbuf_size, MADV_DONTFORK) != 0) {
            /* Caller owns a retryable mapping handle on this error. */
            return -1;
        }
    }
    return 0;
}

static inline int kvaf_udma_checked_alloc_queue_buffer(
    struct udma_u_jetty_queue *queue, uint32_t max_entry_count,
    uint32_t base_block_size, uint32_t page_size, bool allocate_wrid)
{
    return kvaf_udma_checked_alloc_queue_buffer_policy(
        queue, max_entry_count, base_block_size, page_size,
        allocate_wrid, KVAF_UDMA_QUEUE_MEMORY_AUTO);
}

static inline int kvaf_udma_checked_alloc_sw_db_policy(
    struct udma_u_context *context, enum udma_db_type type,
    uint32_t **doorbell_out,
    enum kvaf_udma_queue_memory_policy memory_policy)
{
    struct udma_u_db_page *page;
    uint32_t position;
    bool new_page = false;

    if (context == NULL || doorbell_out == NULL ||
        (uint32_t)type >= UDMA_DB_TYPE_NUM ||
        context->page_size < UDMA_DB_SIZE ||
        (context->page_size & (context->page_size - 1U)) != 0U ||
        (memory_policy != KVAF_UDMA_QUEUE_MEMORY_AUTO &&
         memory_policy != KVAF_UDMA_QUEUE_MEMORY_KERNEL_BUFFER &&
         memory_policy != KVAF_UDMA_QUEUE_MEMORY_USER_ANONYMOUS) ||
        (memory_policy == KVAF_UDMA_QUEUE_MEMORY_USER_ANONYMOUS &&
         type != UDMA_JFR_TYPE_DB)) {
        return -1;
    }
    *doorbell_out = NULL;
    if (pthread_mutex_lock(&context->db_list_mutex) != 0) {
        return -1;
    }
    /*
     * An anonymous management record DB must never silently reuse a page
     * created under another backing policy.  Conversely, its one-slot page
     * is marked full below so later AUTO data JFRs cannot reuse it.
     */
    if (memory_policy == KVAF_UDMA_QUEUE_MEMORY_USER_ANONYMOUS &&
        context->db_list[type] != NULL) {
        errno = EBUSY;
        (void)pthread_mutex_unlock(&context->db_list_mutex);
        return -1;
    }
    for (page = context->db_list[type]; page != NULL; page = page->next) {
        if (page->use_cnt < page->num_db) {
            break;
        }
    }
    if (page == NULL) {
        page = calloc(1, sizeof(*page));
        if (page == NULL) {
            goto failure;
        }
        page->num_db = context->page_size / UDMA_DB_SIZE;
        page->buf.length = context->page_size;
        page->bitmap = udma_bitmap_alloc(
            page->num_db, &page->bitmap_cnt);
        if (page->bitmap == NULL) {
            free(page);
            goto failure;
        }
        if (memory_policy == KVAF_UDMA_QUEUE_MEMORY_USER_ANONYMOUS ||
            type == UDMA_JFR_PAYLOAD) {
            page->buf.buf = mmap(NULL, page->buf.length,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        } else {
            const off_t offset = get_mmap_offset(0,
                context->page_size, UDMA_MMAP_KERNEL_BUF);

            page->buf.buf = mmap(NULL, page->buf.length,
                PROT_READ | PROT_WRITE, MAP_SHARED,
                context->urma_ctx.dev_fd, offset);
        }
        if (page->buf.buf == MAP_FAILED) {
            udma_bitmap_free(page->bitmap);
            free(page);
            goto failure;
        }
        page->prev = NULL;
        page->next = context->db_list[type];
        if (page->next != NULL) {
            page->next->prev = page;
        }
        context->db_list[type] = page;
        new_page = true;
    }
    if (udma_bitmap_use_idx(page->bitmap, page->bitmap_cnt,
            page->num_db, &position) != 0) {
        if (new_page) {
            /* Keep the newly acquired mapping reachable for orphan retry. */
            *doorbell_out = (uint32_t *)page->buf.buf;
            page->use_cnt = 1U;
        }
        (void)pthread_mutex_unlock(&context->db_list_mutex);
        return -1;
    }
    *doorbell_out = (uint32_t *)((char *)page->buf.buf +
        position * UDMA_DB_SIZE);
    **doorbell_out = 0U;
    ++page->use_cnt;
    if (memory_policy == KVAF_UDMA_QUEUE_MEMORY_USER_ANONYMOUS) {
        page->num_db = page->use_cnt;
    }
    if (new_page &&
        madvise(page->buf.buf, page->buf.length, MADV_DONTFORK) != 0) {
        /* Page remains linked and the returned slot is the cleanup handle. */
        page->num_db = page->use_cnt;
        (void)pthread_mutex_unlock(&context->db_list_mutex);
        return -1;
    }
    (void)pthread_mutex_unlock(&context->db_list_mutex);
    return 0;

failure:
    (void)pthread_mutex_unlock(&context->db_list_mutex);
    return -1;
}

static inline int kvaf_udma_checked_alloc_sw_db(
    struct udma_u_context *context, enum udma_db_type type,
    uint32_t **doorbell_out)
{
    return kvaf_udma_checked_alloc_sw_db_policy(
        context, type, doorbell_out, KVAF_UDMA_QUEUE_MEMORY_AUTO);
}

/*
 * The upstream helpers return void and clear/free their bookkeeping even if
 * munmap fails.  These private helpers change a phase only after the mapping
 * is demonstrably gone, leaving the original address available for retry.
 */
static inline int kvaf_udma_checked_free_doorbell(
    urma_context_t *public_context, struct udma_u_doorbell *doorbell)
{
    struct udma_u_context *context;
    void *mapping;

    if (public_context == NULL || doorbell == NULL) {
        return -1;
    }
    if (doorbell->addr == NULL || doorbell->addr == MAP_FAILED) {
        return 0;
    }
    context = to_udma_u_ctx(public_context);
    if (context->page_size == 0U) {
        return -1;
    }
    mapping = (void *)doorbell->addr;
    if (doorbell->type == UDMA_MMAP_JETTY_DSQE) {
        mapping = (char *)mapping -
            udma_get_dsqe_db_offset(context, doorbell);
    }
    if (munmap(mapping, (size_t)context->page_size) != 0) {
        return -1;
    }
    doorbell->addr = NULL;
    return 0;
}

static inline void kvaf_udma_unlink_hugepage(
    struct udma_u_context *context,
    struct udma_u_hugepage_priv *mapping)
{
    if (mapping->pre != NULL) {
        mapping->pre->next = mapping->next;
    } else {
        context->hugepage_list = mapping->next;
    }
    if (mapping->next != NULL) {
        mapping->next->pre = mapping->pre;
    }
}

static inline int kvaf_udma_checked_free_hugepage_queue(
    struct udma_u_jetty_queue *queue)
{
    struct udma_u_hugepage *hugepage = queue->hugepage;
    struct udma_u_hugepage_priv *mapping;

    if (hugepage == NULL || hugepage->priv == NULL || queue->ctx == NULL ||
        pthread_mutex_lock(&queue->ctx->hugepage_lock) != 0) {
        return -1;
    }
    mapping = hugepage->priv;
    if (mapping->refcnt == 0U) {
        (void)pthread_mutex_unlock(&queue->ctx->hugepage_lock);
        return -1;
    }
    if (mapping->refcnt == 1U) {
        if (mapping->va_base == NULL || mapping->va_len == 0U ||
            munmap(mapping->va_base, mapping->va_len) != 0) {
            (void)pthread_mutex_unlock(&queue->ctx->hugepage_lock);
            return -1;
        }
        kvaf_udma_unlink_hugepage(queue->ctx, mapping);
        mapping->refcnt = 0U;
        (void)pthread_mutex_unlock(&queue->ctx->hugepage_lock);
        free(mapping);
    } else {
        --mapping->refcnt;
        (void)pthread_mutex_unlock(&queue->ctx->hugepage_lock);
    }
    free(hugepage);
    queue->hugepage = NULL;
    return 0;
}

static inline int kvaf_udma_checked_free_queue_buffer(
    struct udma_u_jetty_queue *queue)
{
    if (queue == NULL) {
        return -1;
    }
    if (queue->wrid != NULL) {
        free(queue->wrid);
        queue->wrid = NULL;
    }
    if (!queue->cstm && !queue->dtu_en && !queue->sq_reserved &&
        queue->qbuf != NULL) {
        if (queue->hugepage != NULL) {
            if (kvaf_udma_checked_free_hugepage_queue(queue) != 0) {
                return -1;
            }
        } else if (munmap(queue->qbuf, queue->qbuf_size) != 0) {
            return -1;
        }
    }
    queue->qbuf = NULL;
    queue->qbuf_curr = NULL;
    queue->qbuf_end = NULL;
    return 0;
}

static inline int kvaf_udma_checked_free_sw_db(
    struct udma_u_context *context, uint32_t *doorbell,
    enum udma_db_type type)
{
    struct udma_u_db_page *page;
    uint32_t position;

    if (context == NULL || doorbell == NULL ||
        (uint32_t)type >= UDMA_DB_TYPE_NUM ||
        context->page_size == 0U ||
        (context->page_size & (context->page_size - 1U)) != 0U ||
        pthread_mutex_lock(&context->db_list_mutex) != 0) {
        return -1;
    }
    for (page = context->db_list[type]; page != NULL; page = page->next) {
        if (((uintptr_t)doorbell &
             (~((uintptr_t)context->page_size - 1U))) ==
            (uintptr_t)page->buf.buf) {
            break;
        }
    }
    if (page == NULL || page->use_cnt == 0U) {
        (void)pthread_mutex_unlock(&context->db_list_mutex);
        return -1;
    }
    position = ((uintptr_t)doorbell - (uintptr_t)page->buf.buf) /
        UDMA_DB_SIZE;
    if (page->use_cnt > 1U) {
        --page->use_cnt;
        udma_bitmap_free_idx(page->bitmap, page->bitmap_cnt, position);
        (void)pthread_mutex_unlock(&context->db_list_mutex);
        return 0;
    }
    if (page->buf.buf == NULL || page->buf.length == 0U ||
        munmap(page->buf.buf, page->buf.length) != 0) {
        (void)pthread_mutex_unlock(&context->db_list_mutex);
        return -1;
    }
    if (page->prev != NULL) {
        page->prev->next = page->next;
    } else {
        context->db_list[type] = page->next;
    }
    if (page->next != NULL) {
        page->next->prev = page->prev;
    }
    page->use_cnt = 0U;
    udma_bitmap_free(page->bitmap);
    free(page);
    (void)pthread_mutex_unlock(&context->db_list_mutex);
    return 0;
}

static inline int kvaf_udma_context_has_db_pages(
    const struct udma_u_context *context)
{
    uint32_t index;

    for (index = 0; index < UDMA_DB_TYPE_NUM; ++index) {
        if (context->db_list[index] != NULL) {
            return 1;
        }
    }
    return 0;
}

#endif
