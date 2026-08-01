#ifndef KVAF_PRIVATE_UDMA_ACCESSOR_H
#define KVAF_PRIVATE_UDMA_ACCESSOR_H

#include <stdint.h>

#include "kvaf_urma_provider_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Values extracted from the ABI-pinned UDMA provider object. Public
 * liburma query APIs do not expose userspace SQ/CQ mappings or the JFC
 * software doorbell, so this adapter is intentionally compiled against the
 * private UMDK 26.06 UDMA layout.  The source-built UDMA objects, accessor and
 * public ABI probe are compiled from one pinned source snapshot.  Host and
 * AICPU keep different execution-domain ABI values, but they no longer select
 * different C layouts through ST64B/TGID_VA.
 */
#define KVAF_UDMA_PRIVATE_LAYOUT_UMDK_2606 0x00020600U

struct kvaf_udma_direct_queue_info {
  uint32_t struct_size;
  uint32_t private_abi;
  uint32_t sq_id;
  uint32_t sq_baseblock_shift;
  uint32_t sq_baseblock_count;
  uint32_t sq_producer_index;
  uint32_t sq_consumer_index;
  uint32_t sq_reserved;
  uint64_t sq_base;
  uint64_t sq_bytes;
  uint64_t sq_doorbell;
  uint32_t cq_id;
  uint32_t cq_baseblock_shift;
  uint32_t cq_baseblock_count;
  uint32_t cq_producer_index;
  uint32_t cq_consumer_index;
  uint32_t cqe_size;
  uint32_t layout_version;
  uint64_t cq_base;
  uint64_t cq_bytes;
  uint64_t cq_doorbell;
  uint64_t layout_signature;
};

int kvaf_udma_extract_direct_queue_info(
    void *public_jetty, void *public_send_jfc,
    struct kvaf_udma_direct_queue_info *info);

uint32_t kvaf_urma_provider_private_abi_v2(void);

#ifdef __cplusplus
}
#endif

#endif
