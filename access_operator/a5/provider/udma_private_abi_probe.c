#include <stddef.h>
#include <stdio.h>

#include "udma_u_common.h"

int main(void)
{
    printf(
        "KVAF_UDMA_PRIVATE_ABI="
        "context_size:%zu,context_urma_offset:%zu,"
        "context_db_offset:%zu,context_sva_sep_offset:%zu,"
        "context_jetty_table_offset:%zu,context_jetty_lock_offset:%zu,"
        "context_jfr_table_offset:%zu,context_jfr_lock_offset:%zu,"
        "queue_size:%zu,queue_ctx_offset:%zu,queue_qbuf_offset:%zu,"
        "queue_pi_offset:%zu,queue_ci_offset:%zu,queue_idx_offset:%zu,"
        "queue_db_offset:%zu,queue_baseblk_shift_offset:%zu,"
        "queue_dwqe_offset:%zu,queue_sq_reserved_offset:%zu,"
        "jetty_size:%zu,jetty_sq_offset:%zu,"
        "jfc_size:%zu,jfc_cq_offset:%zu,jfc_sw_db_offset:%zu\n",
        sizeof(struct udma_u_context),
        offsetof(struct udma_u_context, urma_ctx),
        offsetof(struct udma_u_context, db),
        offsetof(struct udma_u_context, sva_sep_mode_en),
        offsetof(struct udma_u_context, jetty_table),
        offsetof(struct udma_u_context, jetty_table_lock),
        offsetof(struct udma_u_context, jfr_table),
        offsetof(struct udma_u_context, jfr_table_lock),
        sizeof(struct udma_u_jetty_queue),
        offsetof(struct udma_u_jetty_queue, ctx),
        offsetof(struct udma_u_jetty_queue, qbuf),
        offsetof(struct udma_u_jetty_queue, pi),
        offsetof(struct udma_u_jetty_queue, ci),
        offsetof(struct udma_u_jetty_queue, idx),
        offsetof(struct udma_u_jetty_queue, db),
        offsetof(struct udma_u_jetty_queue, baseblk_shift),
        offsetof(struct udma_u_jetty_queue, dwqe_addr),
        offsetof(struct udma_u_jetty_queue, sq_reserved),
        sizeof(struct udma_u_jetty),
        offsetof(struct udma_u_jetty, sq),
        sizeof(struct udma_u_jfc),
        offsetof(struct udma_u_jfc, cq),
        offsetof(struct udma_u_jfc, sw_db));
    return 0;
}
