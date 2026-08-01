#include "kvaf_private_udma_accessor.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/*
 * This is the only translation unit allowed to include a UDMA private
 * header.  CMake builds it twice:
 *
 *   Host endpoint   : UMDK 26.06 private layout, Host execution domain
 *   AICPU endpoint  : UMDK 26.06 private layout, AICPU execution domain
 *
 * The accessor and the source-built UDMA implementation are compiled against
 * the same udma_u_common.h. ST64B is a runtime context capability in this
 * source line and TGID_VA is no longer part of the private structure layout.
 */
#include "udma_u_common.h"

#if !defined(KVAF_UDMA_ACCESSOR_HOST) && \
    !defined(KVAF_UDMA_ACCESSOR_DEVICE)
#error "Select exactly one UDMA private ABI"
#endif
#if defined(KVAF_UDMA_ACCESSOR_HOST) && \
    defined(KVAF_UDMA_ACCESSOR_DEVICE)
#error "Select exactly one UDMA private ABI"
#endif

#define KVAF_UDMA_WQEBB_SHIFT 6U
#define KVAF_UDMA_CQE_BYTES 64U
#define KVAF_UDMA_DOORBELL_OFFSET 0x80U

_Static_assert(offsetof(struct udma_u_jetty, sq) == sizeof(urma_jetty_t),
    "source-built UDMA Jetty base layout changed");
_Static_assert(offsetof(struct udma_u_jfc, cq) == sizeof(urma_jfc_t),
    "source-built UDMA JFC base layout changed");
_Static_assert(offsetof(struct udma_u_jfc, sw_db) ==
        offsetof(struct udma_u_jfc, cq) +
            sizeof(struct udma_u_jetty_queue),
    "source-built UDMA JFC queue layout changed");
_Static_assert(MAP_COMMAND_MASK == 0xf,
    "UMDK 26.06 mmap command mask changed");
_Static_assert(MAP_INDEX_MASK == 0xfffffff,
    "UMDK 26.06 mmap index mask changed");
_Static_assert(MAP_INDEX_SHIFT == 4,
    "UMDK 26.06 mmap index shift changed");
_Static_assert(UDMA_MMAP_HUGEPAGE == 0 &&
        UDMA_MMAP_JFC_PAGE == 1 &&
        UDMA_MMAP_JETTY_DSQE == 2 &&
        UDMA_MMAP_RESERVED_SQ == 3 &&
        UDMA_MMAP_KERNEL_BUF == 4,
    "UMDK 26.06 mmap command values changed");
_Static_assert(sizeof(struct udma_create_jetty_ucmd) == 72 &&
        offsetof(struct udma_create_jetty_ucmd, db_addr) == 16 &&
        offsetof(struct udma_create_jetty_ucmd, idx_addr) == 24 &&
        offsetof(struct udma_create_jetty_ucmd, jetty_addr) == 40 &&
        offsetof(struct udma_create_jetty_ucmd, jetty_type) == 52 &&
        offsetof(struct udma_create_jetty_ucmd, jfr_sleep_buf) == 56 &&
        offsetof(struct udma_create_jetty_ucmd, jfs_id) == 64,
    "UMDK 26.06 create-Jetty UAPI changed");
_Static_assert(sizeof(struct udma_create_jetty_resp) == 8,
    "UMDK 26.06 create-Jetty response changed");
_Static_assert(sizeof(struct udma_create_jfc_ucmd) == 32 &&
        offsetof(struct udma_create_jfc_ucmd, mode) == 12 &&
        offsetof(struct udma_create_jfc_ucmd, db_addr) == 16 &&
        offsetof(struct udma_create_jfc_ucmd, rsv1) == 28 &&
        sizeof(struct udma_create_jfc_resp) == 8,
    "UMDK 26.06 create-JFC UAPI changed");
_Static_assert(sizeof(struct udma_create_ctx_resp) == 64 &&
        offsetof(struct udma_create_ctx_resp, ue_id) == 4 &&
        offsetof(struct udma_create_ctx_resp, sq_reserved_va) == 32 &&
        offsetof(struct udma_create_ctx_resp, dtu_va_base) == 48,
    "UMDK 26.06 create-context UAPI changed");

static bool kvaf_is_power_of_two(uint32_t value)
{
    return value != 0 && (value & (value - 1U)) == 0;
}

static uint64_t kvaf_layout_signature(void)
{
    const uint64_t fields[] = {
        KVAF_UDMA_PRIVATE_LAYOUT_UMDK_2606,
        sizeof(struct udma_u_context),
        sizeof(struct udma_u_jetty_queue),
        sizeof(struct udma_u_jetty),
        sizeof(struct udma_u_jfc),
        offsetof(struct udma_u_context, st64b_en),
        offsetof(struct udma_u_context, jetty_table),
        offsetof(struct udma_u_jetty_queue, pi_type),
        offsetof(struct udma_u_jetty_queue, aligned_size),
        offsetof(struct udma_u_jetty, sq),
        offsetof(struct udma_u_jfc, cq),
        offsetof(struct udma_u_jfc, sw_db),
    };
    uint64_t hash = UINT64_C(1469598103934665603);
    size_t index;

    for (index = 0; index < sizeof(fields) / sizeof(fields[0]); ++index) {
        hash ^= fields[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

uint32_t kvaf_urma_provider_private_abi_v2(void)
{
#ifdef KVAF_UDMA_ACCESSOR_HOST
    return KVAF_UDMA_PRIVATE_ABI_HOST_PERFTEST_AIV_V1;
#else
    return KVAF_UDMA_PRIVATE_ABI_DEVICE_PERFTEST_AIV_V1;
#endif
}

int kvaf_udma_extract_direct_queue_info(
    void *public_jetty, void *public_send_jfc,
    struct kvaf_udma_direct_queue_info *info)
{
    struct udma_u_jetty *jetty;
    struct udma_u_jfc *jfc;
    struct udma_u_jetty_queue *sq;
    struct udma_u_jetty_queue *cq;

    if (public_jetty == NULL || public_send_jfc == NULL || info == NULL) {
        return -1;
    }
    (void)memset(info, 0, sizeof(*info));
    info->struct_size = (uint32_t)sizeof(*info);
    info->layout_version = KVAF_UDMA_PRIVATE_LAYOUT_UMDK_2606;
#ifdef KVAF_UDMA_ACCESSOR_HOST
    info->private_abi =
        KVAF_UDMA_PRIVATE_ABI_HOST_PERFTEST_AIV_V1;
#else
    info->private_abi =
        KVAF_UDMA_PRIVATE_ABI_DEVICE_PERFTEST_AIV_V1;
#endif

    jetty = to_udma_u_jetty((urma_jetty_t *)public_jetty);
    jfc = to_udma_u_jfc((urma_jfc_t *)public_send_jfc);
    sq = &jetty->sq;
    cq = &jfc->cq;

    /*
     * Fresh perftest queues have not posted a warm-up WQE.  Validate every
     * value needed by AIV before exporting an address; a layout mismatch must
     * fail before the kernel can write an arbitrary location.
     */
    if (sq->ctx == NULL || cq->ctx == NULL || sq->ctx != cq->ctx ||
        sq->ctx->dtu_enable || sq->ctx->sq_reserved ||
        sq->dtu_en || sq->sq_reserved || !sq->is_jetty ||
        sq->is_wqe_lock_buf_jetty || sq->pi_type ||
        sq->need_ring_db || sq->db_status != 0U ||
        sq->aligned_size != UDMA_HW_PAGE_SIZE ||
        sq->db.type != UDMA_MMAP_JETTY_DSQE ||
        sq->ctx->db.type != UDMA_MMAP_JFC_PAGE ||
        sq->qbuf == NULL || sq->qbuf_curr != sq->qbuf ||
        sq->qbuf_end == NULL ||
        (uintptr_t)sq->qbuf_end <= (uintptr_t)sq->qbuf ||
        sq->qbuf_size == 0 || sq->baseblk_shift != KVAF_UDMA_WQEBB_SHIFT ||
        !kvaf_is_power_of_two(sq->baseblk_cnt) ||
        sq->baseblk_mask != sq->baseblk_cnt - 1U ||
        sq->qbuf_size != sq->baseblk_cnt << sq->baseblk_shift ||
        sq->pi != 0 || sq->ci != 0 || sq->db.addr == NULL ||
        cq->dtu_en || cq->sq_reserved || cq->is_jetty ||
        cq->aligned_size != UDMA_HW_PAGE_SIZE ||
        jfc->db_cstm || jfc->ccu_flag != 0U ||
        cq->qbuf == NULL || cq->qbuf_curr != cq->qbuf ||
        cq->qbuf_end == NULL ||
        (uintptr_t)cq->qbuf_end <= (uintptr_t)cq->qbuf ||
        cq->qbuf_size == 0 || cq->baseblk_shift != KVAF_UDMA_WQEBB_SHIFT ||
        !kvaf_is_power_of_two(cq->baseblk_cnt) ||
        cq->baseblk_mask != cq->baseblk_cnt - 1U ||
        cq->qbuf_size != cq->baseblk_cnt << cq->baseblk_shift ||
        cq->pi != 0 || cq->ci != 0 || jfc->sw_db == NULL ||
        sq->ctx->cqe_size != KVAF_UDMA_CQE_BYTES) {
        return -1;
    }

    info->sq_id = sq->idx;
    info->sq_baseblock_shift = sq->baseblk_shift;
    info->sq_baseblock_count = sq->baseblk_cnt;
    info->sq_producer_index = sq->pi;
    info->sq_consumer_index = sq->ci;
    info->sq_reserved = sq->ctx->sq_reserved ? 1U : 0U;
    info->sq_base = (uint64_t)(uintptr_t)sq->qbuf;
    info->sq_bytes = sq->qbuf_size;
    info->sq_doorbell =
        (uint64_t)(uintptr_t)sq->db.addr +
        KVAF_UDMA_DOORBELL_OFFSET;
    info->cq_id = cq->idx;
    info->cq_baseblock_shift = cq->baseblk_shift;
    info->cq_baseblock_count = cq->baseblk_cnt;
    info->cq_producer_index = cq->pi;
    info->cq_consumer_index = cq->ci;
    info->cqe_size = sq->ctx->cqe_size;
    info->cq_base = (uint64_t)(uintptr_t)cq->qbuf;
    info->cq_bytes = cq->qbuf_size;
    info->cq_doorbell = (uint64_t)(uintptr_t)jfc->sw_db;
    info->layout_signature = kvaf_layout_signature();
    return 0;
}
