#ifndef KVAF_UDMA_PUBLIC_ABI_PREINCLUDE_H
#define KVAF_UDMA_PUBLIC_ABI_PREINCLUDE_H

/*
 * The urma_perftest top-level urma_types.h contains an old, copied subset of
 * UDMA private structures after the real public ABI.  KVAF and the reference
 * perftest must use that same public ABI, while the source-built UDMA must use
 * the complete private definitions from udma_u_common.h.  Rename only the
 * copied private section while pre-including urma_provider.h.  Its include
 * guards then let udma_u_common.h reuse the public ABI without redefining the
 * stale copy under the real names.
 */
#define udma_u_node_tbl_type kvaf_shadow_udma_u_node_tbl_type
#define UDMA_U_JFR_TBL KVAF_SHADOW_UDMA_U_JFR_TBL
#define UDMA_U_JETTY_TBL KVAF_SHADOW_UDMA_U_JETTY_TBL
#define UDMA_U_TPID_TBL KVAF_SHADOW_UDMA_U_TPID_TBL
#define UDMA_U_TBL_NUM KVAF_SHADOW_UDMA_U_TBL_NUM
#define udma_db_type kvaf_shadow_udma_db_type
#define UDMA_JFR_TYPE_DB KVAF_SHADOW_UDMA_JFR_TYPE_DB
#define UDMA_JFC_TYPE_DB KVAF_SHADOW_UDMA_JFC_TYPE_DB
#define UDMA_DB_TYPE_NUM KVAF_SHADOW_UDMA_DB_TYPE_NUM
#define db_mmap_type kvaf_shadow_db_mmap_type
#define UDMA_MMAP_JFC_PAGE KVAF_SHADOW_UDMA_MMAP_JFC_PAGE
#define UDMA_MMAP_JETTY_DSQE KVAF_SHADOW_UDMA_MMAP_JETTY_DSQE
#define UDMA_MMAP_HUGEPAGE KVAF_SHADOW_UDMA_MMAP_HUGEPAGE
#define udma_u_hmap_node kvaf_shadow_udma_u_hmap_node
#define udma_u_hmap_head kvaf_shadow_udma_u_hmap_head
#define udma_u_hmap kvaf_shadow_udma_u_hmap
#define node_tbl kvaf_shadow_node_tbl
#define udma_u_doorbell kvaf_shadow_udma_u_doorbell
#define udma_u_db_page kvaf_shadow_udma_u_db_page
#define udma_u_hugepage_priv kvaf_shadow_udma_u_hugepage_priv
#define udma_u_hugepage kvaf_shadow_udma_u_hugepage
#define udma_u_context kvaf_shadow_udma_u_context
#define udma_u_jetty_queue kvaf_shadow_udma_u_jetty_queue
#define udma_u_target_jetty kvaf_shadow_udma_u_target_jetty
#define udma_u_buf kvaf_shadow_udma_u_buf
#define udma_u_jfr_idx_que kvaf_shadow_udma_u_jfr_idx_que
#define udma_u_jfr kvaf_shadow_udma_u_jfr
#define udma_u_jfs kvaf_shadow_udma_u_jfs
#define udma_u_jetty kvaf_shadow_udma_u_jetty
#define udma_u_jetty_grp kvaf_shadow_udma_u_jetty_grp
#define udma_u_jfc kvaf_shadow_udma_u_jfc
#define to_udma_u_ctx kvaf_shadow_to_udma_u_ctx
#define to_udma_u_jfr kvaf_shadow_to_udma_u_jfr
#define to_udma_u_jfs kvaf_shadow_to_udma_u_jfs
#define to_udma_u_jetty kvaf_shadow_to_udma_u_jetty
#define to_udma_u_jfc kvaf_shadow_to_udma_u_jfc

#include "urma_provider.h"

#undef to_udma_u_jfc
#undef to_udma_u_jetty
#undef to_udma_u_jfs
#undef to_udma_u_jfr
#undef to_udma_u_ctx
#undef udma_u_jfc
#undef udma_u_jetty_grp
#undef udma_u_jetty
#undef udma_u_jfs
#undef udma_u_jfr
#undef udma_u_jfr_idx_que
#undef udma_u_buf
#undef udma_u_target_jetty
#undef udma_u_jetty_queue
#undef udma_u_context
#undef udma_u_hugepage
#undef udma_u_hugepage_priv
#undef udma_u_db_page
#undef udma_u_doorbell
#undef node_tbl
#undef udma_u_hmap
#undef udma_u_hmap_head
#undef udma_u_hmap_node
#undef UDMA_MMAP_HUGEPAGE
#undef UDMA_MMAP_JETTY_DSQE
#undef UDMA_MMAP_JFC_PAGE
#undef db_mmap_type
#undef UDMA_DB_TYPE_NUM
#undef UDMA_JFC_TYPE_DB
#undef UDMA_JFR_TYPE_DB
#undef udma_db_type
#undef UDMA_U_TBL_NUM
#undef UDMA_U_TPID_TBL
#undef UDMA_U_JETTY_TBL
#undef UDMA_U_JFR_TBL
#undef udma_u_node_tbl_type

/* udma_u_common.h supplies the canonical private helpers. */
#undef container_of
#undef container_off
#undef check_types_match

#endif
