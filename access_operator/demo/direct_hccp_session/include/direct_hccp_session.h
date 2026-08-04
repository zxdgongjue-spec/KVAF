#ifndef KVAF_DIRECT_HCCP_SESSION_H_
#define KVAF_DIRECT_HCCP_SESSION_H_

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

// Acquires the RA/HDC service for {physical_device_id, hdc_type}.  The first
// caller performs RaInit(host), rtOpenNetService and RaInit(device); later
// callers only take a reference.  The caller must pair every success with
// ReleaseDirectHccpSession after all of its RaCtx resources are gone.
bool AcquireDirectHccpSession(uint32_t physical_device_id, uint32_t hdc_type,
                              DirectHccpSession **session);

// Releases a reference.  The last release tears down device and host RA in
// that order.  Callers keep ownership of their own data-plane resources.
void ReleaseDirectHccpSession(DirectHccpSession *session);

}  // namespace kvaf::direct_hccp

#endif  // KVAF_DIRECT_HCCP_SESSION_H_
