#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "urma_opcode.h"
#include "udma_abi.h"

int main(void)
{
    printf(
        "KVAF_UDMA_UAPI_ABI="
        "map_index_shift:%u,tid_shift:%u,access_guard:0x%08x,"
        "access_local_only:0x%08x,access_read:0x%08x,"
        "access_write:0x%08x,access_atomic:0x%08x,"
        "ctx_resp_size:%zu,ctx_ue_id_offset:%zu,"
        "ctx_sq_reserved_va_offset:%zu,ctx_dtu_va_base_offset:%zu,"
        "jetty_ucmd_size:%zu,jetty_buf_offset:%zu,"
        "jetty_idx_offset:%zu,jetty_addr_offset:%zu,"
        "jetty_type_offset:%zu,jetty_jfs_id_offset:%zu,"
        "jetty_resp_size:%zu,jfc_ucmd_size:%zu,jfc_resp_size:%zu\n",
        (unsigned int)MAP_INDEX_SHIFT,
        (unsigned int)UDMA_TID_SHIFT,
        (unsigned int)UDMA_SEGMENT_ACCESS_GUARD,
        (unsigned int)URMA_ACCESS_LOCAL_ONLY,
        (unsigned int)URMA_ACCESS_READ,
        (unsigned int)URMA_ACCESS_WRITE,
        (unsigned int)URMA_ACCESS_ATOMIC,
        sizeof(struct udma_create_ctx_resp),
        offsetof(struct udma_create_ctx_resp, ue_id),
        offsetof(struct udma_create_ctx_resp, sq_reserved_va),
        offsetof(struct udma_create_ctx_resp, dtu_va_base),
        sizeof(struct udma_create_jetty_ucmd),
        offsetof(struct udma_create_jetty_ucmd, buf_addr),
        offsetof(struct udma_create_jetty_ucmd, idx_addr),
        offsetof(struct udma_create_jetty_ucmd, jetty_addr),
        offsetof(struct udma_create_jetty_ucmd, jetty_type),
        offsetof(struct udma_create_jetty_ucmd, jfs_id),
        sizeof(struct udma_create_jetty_resp),
        sizeof(struct udma_create_jfc_ucmd),
        sizeof(struct udma_create_jfc_resp));
    return 0;
}
