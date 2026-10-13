#ifndef KVAF_AIV_BATCH_PUT_H_
#define KVAF_AIV_BATCH_PUT_H_

#include <stdint.h>

#include "acl/acl.h"
#include "batch_put_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

enum KvafBatchPutResult {
  KVAF_BATCH_PUT_SUCCESS = 0,
  KVAF_BATCH_PUT_INVALID_ARGUMENT = -1,
  KVAF_BATCH_PUT_RUNTIME_ERROR = -2,
  KVAF_BATCH_PUT_EID_ERROR = -3,
  KVAF_BATCH_PUT_DIRECT_HCCP_ERROR = -4,
  KVAF_BATCH_PUT_KERNEL_ERROR = -5,
};

typedef struct KvafBatchPutTransport KvafBatchPutTransport;
typedef struct KvafBatchPutLocalBuffer KvafBatchPutLocalBuffer;

int32_t KvafBatchPutTransportCreateWithPeerLink(
    void *host_ddr, uint64_t host_ddr_bytes, uint32_t aiv_core_count,
    uint32_t jettys_per_core, void *peer_link,
    const uint8_t *peer_local_eid, KvafBatchPutTransport **transport);
void KvafBatchPutTransportDestroy(KvafBatchPutTransport *transport);

int32_t KvafBatchPutLocalBufferAcquire(KvafBatchPutTransport *transport,
                                       void *input_device, uint64_t input_device_bytes,
                                       KvafBatchPutLocalBuffer **local_buffer);
void KvafBatchPutLocalBufferRelease(KvafBatchPutLocalBuffer *local_buffer);
int32_t KvafBatchPutLocalBufferDescribe(
    const KvafBatchPutLocalBuffer *local_buffer,
    KvafBatchPutSourceDescriptor *descriptor);

int32_t KvafExecuteBatchPutLoadKernel(const char *kernel_object_path);

// Synchronous eager execution: launch, stream synchronization, status copy
// and core-status checking all finish before this function returns.
int32_t KvafExecuteBatchPutSync(KvafBatchPutTransport *transport,
                               const KvafExecuteBatchPutKernelArgs *arguments,
                               aclrtStream stream,
                               KvafBatchPutCoreStatus *core_status,
                               uint32_t core_status_count);

int32_t KvafExecuteBatchPutAicpuSync(KvafBatchPutTransport *transport,
                                    const KvafExecuteBatchPutKernelArgs *arguments,
                                    aclrtStream stream, KvafBatchPutCoreStatus *core_status,
                                    uint32_t core_status_count, uint32_t source_count);

#ifdef __cplusplus
}
#endif

#endif  // KVAF_AIV_BATCH_PUT_H_
