#ifndef KVAF_SOURCE_MATCHED_UDMA_CLEANUP_H
#define KVAF_SOURCE_MATCHED_UDMA_CLEANUP_H

#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/types.h>

#include "udma_u_buf.h"
#include "udma_u_db.h"

static inline int kvaf_udma_checked_alloc_hugepage_queue(
    struct udma_u_jetty_queue *queue)
{
    struct udma_u_hugepage_priv *mapping = NULL;
    struct udma_u_hugepage *hugepage;
    uint32_t mapping_length;
    off_t offset;
    bool mapping_created = false;

    if (queue == NULL || queue->ctx == NULL || queue->qbuf_size == 0U ||
        queue->ctx->page_size < UDMA_HW_PAGE_SIZE ||
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
        offset = get_mmap_offset(
            0, queue->ctx->page_size, UDMA_MMAP_HUGEPAGE);
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

static inline int kvaf_udma_checked_alloc_queue_buffer(
    struct udma_u_jetty_queue *queue, uint32_t max_entry_count,
    uint32_t base_block_size, uint32_t page_size, bool allocate_wrid)
{
    uint64_t required_bytes;
    uint64_t rounded_bytes = 1U;
    uint64_t aligned_bytes;

    if (queue == NULL || max_entry_count == 0U ||
        base_block_size == 0U || page_size == 0U ||
        (base_block_size & (base_block_size - 1U)) != 0U ||
        (page_size & (page_size - 1U)) != 0U) {
        return -1;
    }
    if (queue->sq_reserved) {
        return 0;
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
    if (allocate_wrid) {
        queue->wrid = calloc(queue->baseblk_cnt, sizeof(*queue->wrid));
        if (queue->wrid == NULL) {
            return -1;
        }
    }
    if (queue->ctx != NULL && queue->ctx->hugepage_enable) {
        if (kvaf_udma_checked_alloc_hugepage_queue(queue) != 0) {
            return -1;
        }
    } else {
        queue->qbuf = mmap(NULL, queue->qbuf_size,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
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

static inline int kvaf_udma_checked_alloc_sw_db(
    struct udma_u_context *context, enum udma_db_type type,
    uint32_t **doorbell_out)
{
    struct udma_u_db_page *page;
    uint32_t position;
    bool new_page = false;

    if (context == NULL || doorbell_out == NULL ||
        (uint32_t)type >= UDMA_DB_TYPE_NUM ||
        context->page_size < UDMA_DB_SIZE ||
        (context->page_size & (context->page_size - 1U)) != 0U) {
        return -1;
    }
    *doorbell_out = NULL;
    if (pthread_mutex_lock(&context->db_list_mutex) != 0) {
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
        page->buf.buf = mmap(NULL, page->buf.length,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
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
    if (!queue->cstm && !queue->sq_reserved && queue->qbuf != NULL) {
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
