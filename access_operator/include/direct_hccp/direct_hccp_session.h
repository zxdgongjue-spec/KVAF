#ifndef KVAF_ACCESS_OPERATOR_DIRECT_HCCP_SESSION_H_
#define KVAF_ACCESS_OPERATOR_DIRECT_HCCP_SESSION_H_

#include <cstdint>

namespace kvaf::direct_hccp {

// Process-scoped ownership of the RA/HDC bootstrap.  A Session is shared by
// independent BatchGet and BatchPut data-plane transports that target one
// physical NPU.  It deliberately owns no RaCtx, token, MR, CQ, Jetty, or TP.
struct DirectHccpSession;

// Direct-HCCP resource setup is deliberately quiet during normal Runtime
// execution. Set KVAF_AIV_VERBOSE_DIRECT_HCCP=1 to emit the EID, context,
// token, memory-region, queue, CTP and session lifecycle diagnostics needed
// for hardware bring-up. Failures are always emitted regardless of this flag.
bool VerboseLoggingEnabled();
void VerboseLog(const char* format, ...);

// Acquires the RA/HDC service for {physical_device_id, hdc_type}. The first
// caller initializes peer-online RA and the HDC service. Device RA is either
// owned by KVAF after a successful RaInit or borrowed when HCCL already owns a
// compatible device RA. Later callers only take a reference. The caller must
// pair every success with ReleaseDirectHccpSession after all of its RaCtx
// resources are gone. On the first acquisition, reuse_device_ra asserts that
// the caller already initialized device RA: skip RaInit without probing or
// falling back to another initialization of the shared HDC state.
bool AcquireDirectHccpSession(uint32_t physical_device_id, uint32_t hdc_type,
                              DirectHccpSession **session,
                              bool reuse_device_ra = false);

// Releases a logical transport reference. The RA/HDC bootstrap remains
// process-owned after its first successful initialization: CANN has no close
// counterpart for rtOpenNetService and can retain service/TLS callbacks beyond
// an individual Runtime teardown. Callers still destroy their own RaCtx,
// token, MR, CQ, Jetty and TP resources before releasing this lease.
void ReleaseDirectHccpSession(DirectHccpSession *session);

}  // namespace kvaf::direct_hccp

#endif  // KVAF_ACCESS_OPERATOR_DIRECT_HCCP_SESSION_H_
