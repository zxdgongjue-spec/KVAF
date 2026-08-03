/*
 * Compile-only probe for the installed Ascend driver HAL headers.
 *
 * The caller injects the selected, installed ascend_hal.h with -include and
 * defines exactly one KVAF_HAL_PROBE_* selector.  Deliberately do not provide
 * compatibility declarations or numeric resource-type values here: a missing
 * official declaration must make the preflight fail.  The one separately
 * included KVAF structure is a versioned, source-audited HCCP private payload;
 * it is never described as part of the public installed HAL interface.
 * This translation unit is never linked or executed.
 */

#include <stdint.h>
#include <sys/types.h>

#if defined(KVAF_HAL_PROBE_BASELINE)
static const int kvaf_probe_baseline = 0;

#elif defined(KVAF_HAL_PROBE_DRV_DEVICE_GET_INDEX_BY_PHY_ID)
_Static_assert(__builtin_types_compatible_p(
                   __typeof__(&drvDeviceGetIndexByPhyId),
                   drvError_t (*)(uint32_t, uint32_t *)),
               "drvDeviceGetIndexByPhyId signature is incompatible");

#elif defined(KVAF_HAL_PROBE_QUERY_DEVPID)
_Static_assert(__builtin_types_compatible_p(
                   __typeof__(&halQueryDevpid),
                   drvError_t (*)(struct halQueryDevpidInfo, pid_t *)),
               "halQueryDevpid signature is incompatible");
static void kvaf_probe_query_devpid_fields(void)
{
    struct halQueryDevpidInfo info = {0};

    info.hostpid = 0;
    info.devid = 0;
    info.proc_type = DEVDRV_PROCESS_CP1;
    (void)info;
}

#elif defined(KVAF_HAL_PROBE_MEM_BIND_SIBLING)
_Static_assert(__builtin_types_compatible_p(
                   __typeof__(&halMemBindSibling),
                   DVresult (*)(int, int, unsigned int, unsigned int,
                                unsigned int)),
               "halMemBindSibling signature is incompatible");
static const unsigned int kvaf_probe_sibling_flag = SVM_MEM_BIND_SP_GRP;

#elif defined(KVAF_HAL_PROBE_RES_ADDR_MAP_V2)
_Static_assert(__builtin_types_compatible_p(
                   __typeof__(&halResAddrMapV2),
                   drvError_t (*)(unsigned int, struct res_map_info_in *,
                                  struct res_map_info_out *)),
               "halResAddrMapV2 signature is incompatible");

#elif defined(KVAF_HAL_PROBE_RES_ADDR_UNMAP_V2)
_Static_assert(__builtin_types_compatible_p(
                   __typeof__(&halResAddrUnmapV2),
                   drvError_t (*)(unsigned int, struct res_map_info_in *)),
               "halResAddrUnmapV2 signature is incompatible");

#elif defined(KVAF_HAL_PROBE_RES_MAP_INFO_IN)
static void kvaf_probe_res_map_info_in_fields(void)
{
    struct res_map_info_in info = {0};

    info.res_id = 0;
    info.target_proc_type = PROCESS_CP1;
    info.res_type = RES_ADDR_TYPE_HCCP_URMA_JFC;
    info.priv_len = 0;
    info.priv = (void *)0;
    (void)info;
}

#elif defined(KVAF_HAL_PROBE_RES_MAP_INFO_OUT)
static unsigned long long kvaf_probe_res_map_info_out_fields(void)
{
    struct res_map_info_out info = {0};

    return (unsigned long long)info.va;
}

#elif defined(KVAF_HAL_PROBE_RES_TYPE_HCCP_URMA_JFC)
static const unsigned long long kvaf_probe_resource_type =
    (unsigned long long)RES_ADDR_TYPE_HCCP_URMA_JFC;

#elif defined(KVAF_HAL_PROBE_RES_TYPE_HCCP_URMA_JETTY)
static const unsigned long long kvaf_probe_resource_type =
    (unsigned long long)RES_ADDR_TYPE_HCCP_URMA_JETTY;

#elif defined(KVAF_HAL_PROBE_RES_TYPE_HCCP_URMA_DB)
static const unsigned long long kvaf_probe_resource_type =
    (unsigned long long)RES_ADDR_TYPE_HCCP_URMA_DB;

#elif defined(KVAF_HAL_PROBE_UDMA_VA_INFO_PRIVATE_ABI)
#include "kvaf/access_operator/a5/hccp_hal_private_abi.h"

/* HCCP passes this private payload through res_map_info_in.priv.  The KVAF
 * declaration is pinned to an audited HCOMM source revision; this compile
 * check proves only that its enum dependency and C layout are compatible with
 * the selected installed public header.  It does not prove runtime pairing. */
static void kvaf_probe_udma_va_info_fields(void)
{
    struct kvaf_hccp_udma_va_info_v1 info = {0};

    info.res_type = RES_ADDR_TYPE_HCCP_URMA_JFC;
    info.pid = 0;
    info.va = 0;
    info.len = 0;
    (void)info;
}

#else
#error "exactly one KVAF_HAL_PROBE_* selector is required"
#endif
