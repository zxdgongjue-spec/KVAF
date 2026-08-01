#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/ioctl.h>

#include "ummu_api.h"

int main(void)
{
    printf(
        "KVAF_UMMU_UAPI_ABI="
        "tid_info_size:%zu,tid_offset:%zu,hw_cap_offset:%zu,"
        "va_offset:%zu,size_offset:%zu,"
        "alloc_tid:0x%016" PRIx64 ",free_tid:0x%016" PRIx64 ","
        "plbi_va:0x%016" PRIx64 ",plbi_all:0x%016" PRIx64 ","
        "cap_expan:0x%016" PRIx64 ","
        "cap_double_plbi:0x%016" PRIx64 ","
        "cap_kcmd_plbi:0x%016" PRIx64 "\n",
        sizeof(struct ummu_tid_info),
        offsetof(struct ummu_tid_info, tid),
        offsetof(struct ummu_tid_info, hw_cap),
        offsetof(struct ummu_tid_info, va),
        offsetof(struct ummu_tid_info, size),
        (uint64_t)UMMU_IOCALLOC_TID,
        (uint64_t)UMMU_IOCFREE_TID,
        (uint64_t)UMMU_IOCPLBI_VA,
        (uint64_t)UMMU_IOCPLBI_ALL,
        (uint64_t)HW_CAP_EXPAN,
        (uint64_t)HW_CAP_DOUBLE_PLBI,
        (uint64_t)HW_CAP_KCMD_PLBI);
    return 0;
}
