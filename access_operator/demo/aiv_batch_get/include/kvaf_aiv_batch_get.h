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

typedef struct KvafBatchGetTransport KvafBatchGetTransport;
typedef struct KvafBatchGetLocalBuffer KvafBatchGetLocalBuffer;

// Transport owns only the reusable CPU-EID <-> device-EID path and Host DDR.
// It deliberately accepts no caller-owned HBM buffer. The serial baseline
// requires jettys_per_core == 1.
int32_t KvafBatchGetTransportCreate(void *host_ddr, uint64_t host_ddr_bytes,
                                    uint32_t aiv_core_count, uint32_t jettys_per_core,
                                    KvafBatchGetTransport **transport);
void KvafBatchGetTransportDestroy(KvafBatchGetTransport *transport);

// Acquire registers/caches one caller-owned HBM output range. Re-acquiring
// the exact (address, bytes) pair returns the same local-buffer handle and
// entity table while adding one reference. The returned entity is passed in
// every kernel launch that uses this output.
int32_t KvafBatchGetLocalBufferAcquire(KvafBatchGetTransport *transport,
                                       void *output_hbm, uint64_t output_hbm_bytes,
                                       KvafBatchGetLocalBuffer **local_buffer,
                                       uint64_t *device_channel_entity);

// The demo retains zero-reference registrations until TransportDestroy, so a
// later acquire can reuse them without re-registering HBM. Call only after
// all launches using this local buffer have completed.
void KvafBatchGetLocalBufferRelease(KvafBatchGetLocalBuffer *local_buffer);

int32_t KvafExecuteBatchGetLoadKernel(const char *kernel_object_path);
int32_t KvafExecuteBatchGetLaunch(const KvafExecuteBatchGetKernelArgs *arguments,
                                  aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // KVAF_AIV_BATCH_GET_H_
