#ifndef KVAF_ACCESS_OPERATOR_A5_HCCP_HAL_PRIVATE_ABI_H_
#define KVAF_ACCESS_OPERATOR_A5_HCCP_HAL_PRIVATE_ABI_H_

/*
 * Source-audited HCCP private payload ABI.
 *
 * This is deliberately not presented as an installed Ascend HAL API.  The
 * public ascend_hal.h defines res_map_info_in/out, the resource-type enum and
 * halResAddrMapV2/halResAddrUnmapV2, but the payload passed through
 * res_map_info_in.priv is private to HCCP.  KVAF records the exact open-source
 * snapshot from which this layout was audited so a future Host control-plane
 * adapter can fail closed when its runtime pairing has not been proved.
 *
 * Audited HCOMM revision:
 *   2591d7ff74c18fb9829dd5c0c157f97558405169
 * Source files and SHA-256:
 *   rs_ctx_inner.h  9753b9de6dbd8106677672959ffb909708dcd36aab071baa6b67c0a548c5792d
 *   rs_ub_jfc.c     548c4b1c5c079862c1706923f84b6f831d4270675982473e58ad230117c996b1
 *   rs_ub_jetty.c   9603e1c31f2505f9fd0d365f6081d09adcf1e412f3c23681784d3916f559d8c2
 *
 * The installed ascend_hal.h must be included before this header because
 * enum res_addr_type remains an official-header dependency.  No enum value or
 * public HAL declaration is copied here.
 */

#include <stddef.h>
#include <stdint.h>

#define KVAF_HCCP_PRIVATE_ABI_SCHEMA "hcomm-source-audited-v1"
#define KVAF_HCCP_PRIVATE_ABI_HCOMM_COMMIT \
  "2591d7ff74c18fb9829dd5c0c157f97558405169"
#define KVAF_HCCP_PRIVATE_ABI_RS_CTX_INNER_SHA256 \
  "9753b9de6dbd8106677672959ffb909708dcd36aab071baa6b67c0a548c5792d"
#define KVAF_HCCP_PRIVATE_ABI_RS_UB_JFC_SHA256 \
  "548c4b1c5c079862c1706923f84b6f831d4270675982473e58ad230117c996b1"
#define KVAF_HCCP_PRIVATE_ABI_RS_UB_JETTY_SHA256 \
  "9603e1c31f2505f9fd0d365f6081d09adcf1e412f3c23681784d3916f559d8c2"

struct kvaf_hccp_udma_va_info_v1 {
  enum res_addr_type res_type;
  int pid;
  uint64_t va;
  uint64_t len;
};

#if defined(__cplusplus)
#define KVAF_HCCP_PRIVATE_STATIC_ASSERT static_assert
#else
#define KVAF_HCCP_PRIVATE_STATIC_ASSERT _Static_assert
#endif

KVAF_HCCP_PRIVATE_STATIC_ASSERT(sizeof(enum res_addr_type) == 4,
                                "HCCP res_addr_type width changed");
KVAF_HCCP_PRIVATE_STATIC_ASSERT(sizeof(int) == 4,
                                "HCCP pid field width changed");
KVAF_HCCP_PRIVATE_STATIC_ASSERT(
    offsetof(struct kvaf_hccp_udma_va_info_v1, res_type) == 0,
    "HCCP private res_type offset changed");
KVAF_HCCP_PRIVATE_STATIC_ASSERT(
    offsetof(struct kvaf_hccp_udma_va_info_v1, pid) == 4,
    "HCCP private pid offset changed");
KVAF_HCCP_PRIVATE_STATIC_ASSERT(
    offsetof(struct kvaf_hccp_udma_va_info_v1, va) == 8,
    "HCCP private va offset changed");
KVAF_HCCP_PRIVATE_STATIC_ASSERT(
    offsetof(struct kvaf_hccp_udma_va_info_v1, len) == 16,
    "HCCP private len offset changed");
KVAF_HCCP_PRIVATE_STATIC_ASSERT(
    sizeof(struct kvaf_hccp_udma_va_info_v1) == 24,
    "HCCP private payload size changed");

#undef KVAF_HCCP_PRIVATE_STATIC_ASSERT

#endif  /* KVAF_ACCESS_OPERATOR_A5_HCCP_HAL_PRIVATE_ABI_H_ */
