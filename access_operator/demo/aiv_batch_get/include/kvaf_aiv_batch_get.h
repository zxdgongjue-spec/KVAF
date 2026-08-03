#ifndef KVAF_AIV_BATCH_GET_H_
#define KVAF_AIV_BATCH_GET_H_

#include <stdint.h>

#include "acl/acl.h"
#include "batch_get_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  KVAF_BATCH_GET_SUCCESS = 0,
  KVAF_BATCH_GET_INVALID_ARGUMENT = -1,
  KVAF_BATCH_GET_RUNTIME_ERROR = -2,
  KVAF_BATCH_GET_EID_ERROR = -3,
  KVAF_BATCH_GET_DIRECT_HCCP_ERROR = -4,
};

typedef struct KvafBatchGetDdrRegistration KvafBatchGetDdrRegistration;

// Creates the CPU-EID <-> device-EID Direct-HCCP route, registers Host DDR
// and output HBM, and returns device-visible USER_CTL queue entities.
int32_t KvafBatchGetRegisterLocalDdr(void *host_ddr, uint64_t host_ddr_bytes,
                                     void *output_device, uint64_t output_device_bytes,
                                     uint32_t aiv_core_count, uint32_t jettys_per_core,
                                     KvafBatchGetDdrRegistration **registration,
                                     uint64_t *channel_entity);
void KvafBatchGetUnregisterLocalDdr(KvafBatchGetDdrRegistration *registration);

int32_t KvafExecuteBatchGetLoadKernel(const char *kernel_object_path);
int32_t KvafExecuteBatchGetLaunch(const KvafExecuteBatchGetKernelArgs *arguments,
                                  aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // KVAF_AIV_BATCH_GET_H_
