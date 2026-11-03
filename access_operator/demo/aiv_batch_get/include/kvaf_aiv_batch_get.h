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

// Transport owns only the reusable peer-EID paths and remote DDR registrations.
// It deliberately accepts no caller-owned HBM buffer.
int32_t KvafBatchGetTransportCreateWithPeerLink(
    void *host_ddr, uint64_t host_ddr_bytes, uint32_t aiv_core_count,
    uint32_t jettys_per_core, void *peer_link,
    const uint8_t *peer_local_eid, KvafBatchGetTransport **transport);

// Multi-EID variant. peer_local_eids is a packed array of
// peer_local_eid_count 16-byte EIDs. Workers are assigned to EID paths in
// jetty-major round-robin order, spreading each core's primary Jetty before
// its secondary Jetty. The legacy entry point above remains a one-EID wrapper.
int32_t KvafBatchGetTransportCreateWithPeerLinkMultiEid(
    void *host_ddr, uint64_t host_ddr_bytes, uint32_t aiv_core_count,
    uint32_t jettys_per_core, void *peer_link,
    const uint8_t *peer_local_eids, uint32_t peer_local_eid_count,
    KvafBatchGetTransport **transport);
// NPU multi-MR: MR 1 is the original peer region. Additional MRs share
// contexts/queues, and must be added before READs. Failed imports stay owned
// by the transport until Close; do not retry a partial registration in place.
int32_t KvafBatchGetTransportAddRemoteMemory(KvafBatchGetTransport *transport,
    uint64_t mr_id, uint64_t base, uint64_t bytes);
int32_t KvafBatchGetLocalBufferAcquireForRemote(KvafBatchGetTransport *transport,
    uint64_t mr_id, void *output_hbm, uint64_t output_hbm_bytes,
    KvafBatchGetLocalBuffer **local_buffer, uint64_t *device_channel_entity);
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
