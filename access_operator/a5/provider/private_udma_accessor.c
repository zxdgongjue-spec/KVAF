#include "kvaf_private_udma_accessor.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/*
 * This is the only translation unit allowed to include a UDMA private
 * header.  CMake builds it twice:
 *
 *   Host endpoint   : urma_perftest AIV private ABI, Host role
 *   AICPU endpoint  : urma_perftest AIV private ABI, Device role
 *
 * The accessor and the source-built UDMA implementation are compiled against
 * the same udma_u_common.h and the same Host(ST64B)/Device(TGID_VA) role.
 * This is essential: the older top-level urma_types.h copy omits the queue
 * hmap_node and sq_reserved fields and is not the complete provider layout.
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
        sizeof(urma_jfc_t) + sizeof(struct udma_u_jetty_queue),
    "source-built UDMA JFC queue layout changed");

static bool kvaf_is_power_of_two(uint32_t value)
{
    return value != 0 && (value & (value - 1U)) == 0;
}

static uint64_t kvaf_layout_signature(void)
{
    /*
     * Do not use this value to guess an ABI at runtime.  It is provenance
     * that makes the selected compile-time layout visible in diagnostics.
     */
    return ((uint64_t)sizeof(struct udma_u_jetty_queue) << 32U) |
        ((uint64_t)offsetof(struct udma_u_jfc, sw_db) << 16U) |
        (uint64_t)offsetof(struct udma_u_jetty, sq);
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
        sq->ctx->sq_reserved || sq->sq_reserved ||
        sq->qbuf == NULL || sq->qbuf_curr != sq->qbuf ||
        sq->qbuf_end == NULL ||
        (uintptr_t)sq->qbuf_end <= (uintptr_t)sq->qbuf ||
        sq->qbuf_size == 0 || sq->baseblk_shift != KVAF_UDMA_WQEBB_SHIFT ||
        !kvaf_is_power_of_two(sq->baseblk_cnt) ||
        sq->baseblk_mask != sq->baseblk_cnt - 1U ||
        sq->qbuf_size != sq->baseblk_cnt << sq->baseblk_shift ||
        sq->pi != 0 || sq->ci != 0 || sq->db.addr == NULL ||
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
