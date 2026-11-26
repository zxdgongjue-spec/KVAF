#ifndef KVAF_AIV_BATCH_GET_H_
#define KVAF_AIV_BATCH_GET_H_

#include <stddef.h>
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
// On creation failure, a non-null *transport owns partially initialized
// resources whose cleanup failed. Retain it and retry TransportClose.
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
// Direct Get may stripe one local EID over several CPU FE sessions, or several
// local EIDs over one CPU session. Each session binds the same logical DDR.
int32_t KvafBatchGetTransportCreateWithPeerLinks(
    void *host_ddr, uint64_t host_ddr_bytes, uint32_t aiv_core_count,
    uint32_t jettys_per_core, void *const *peer_links, uint32_t peer_link_count,
    const uint8_t *peer_local_eids, uint32_t peer_local_eid_count,
    KvafBatchGetTransport **transport);
// Optional CPU FE binding per core. Entries index peer_links in caller order.
// An empty map uses the default path distribution.
int32_t KvafBatchGetTransportCreateWithCoreFeMap(
    void *host_ddr, uint64_t host_ddr_bytes, uint32_t aiv_core_count,
    uint32_t jettys_per_core, void *const *peer_links, uint32_t peer_link_count,
    const uint8_t *peer_local_eids, uint32_t peer_local_eid_count,
    const uint32_t *core_fe_map, size_t core_fe_count, KvafBatchGetTransport **transport);
// Caller must stop submitting work and synchronize kernel streams first.
// SUCCESS destroys transport. Failure keeps its handles for Close retry; no
// further acquisitions/launches are allowed after a close attempt.
int32_t KvafBatchGetTransportClose(KvafBatchGetTransport *transport);
// Compatibility wrapper: retain on failure, with no status/retry guarantee.
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
// On error, a non-null *local_buffer retains a partially created registration.
// Keep the caller's HBM alive until LocalBufferRetire or TransportClose succeeds;
// the partial handle must not be submitted or released into the reusable cache.
int32_t KvafBatchGetLocalBufferAcquire(KvafBatchGetTransport *transport,
                                       void *output_hbm, uint64_t output_hbm_bytes,
                                       KvafBatchGetLocalBuffer **local_buffer,
                                       uint64_t *device_channel_entity);

// The transport retains zero-reference registrations until TransportDestroy, so a
// later acquire can reuse them without re-registering HBM. Call only after
// all launches using this local buffer have completed.
void KvafBatchGetLocalBufferRelease(KvafBatchGetLocalBuffer *local_buffer);
// Requires completed DMA for this reference. Only the last reference
// unregisters the range; a failed unregister can be retried.
int32_t KvafBatchGetLocalBufferRetire(KvafBatchGetLocalBuffer *local_buffer);

int32_t KvafExecuteBatchGetLoadKernel(const char *kernel_object_path);
int32_t KvafExecuteBatchGetLaunch(const KvafExecuteBatchGetKernelArgs *arguments,
                                  aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // KVAF_AIV_BATCH_GET_H_
