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

// Creates the CPU-EID <-> device-EID Direct-HCCP transport, registers and
// imports Host DDR, and returns device-visible USER_CTL queue entities.  It
// does not take an HBM input/output buffer.
int32_t KvafBatchGetInitializeTransport(void *host_ddr, uint64_t host_ddr_bytes,
                                        uint32_t aiv_core_count, uint32_t jettys_per_core,
                                        KvafBatchGetDdrRegistration **registration,
                                        uint64_t *channel_entity);

// Ensures that one local HBM range is registered for UDMA and returns a
// device-resident KvafBatchGetLocalHbmBinding.  Registrations are cached for
// the transport lifetime.  The returned binding is passed in kernel args and
// may be different for every eager ExecuteBatchGet call.
int32_t KvafBatchGetPrepareLocalHbm(KvafBatchGetDdrRegistration *registration,
                                    void *output_device, uint64_t output_device_bytes,
                                    uint64_t *local_hbm_binding);
void KvafBatchGetDestroyTransport(KvafBatchGetDdrRegistration *registration);

int32_t KvafExecuteBatchGetLoadKernel(const char *kernel_object_path);
int32_t KvafExecuteBatchGetLaunch(const KvafExecuteBatchGetKernelArgs *arguments,
                                  aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // KVAF_AIV_BATCH_GET_H_
