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

// Acquires the RA/HDC service for {physical_device_id, hdc_type}. The first
// caller initializes host RA and the HDC service. Device RA is either owned by
// KVAF after a successful RaInit or borrowed when HCCL already owns a
// compatible device RA. Later callers only take a reference. The caller must
// pair every success with ReleaseDirectHccpSession after all of its RaCtx
// resources are gone.
//
// The RA/HDC bootstrap itself is process-owned after its first successful
// initialization.  Releasing the final transport only drops the logical
// lease; it intentionally does not RaDeinit or dlclose the service DSOs.
// CANN can retain service/TLS callbacks until process exit, so unloading them
// at a Runtime boundary races those callbacks even after all KVAF CQ work has
// retired.  Per-transport RaCtx, token, MR, CQ and Jetty resources remain
// caller-owned and are still released normally.
bool AcquireDirectHccpSession(uint32_t physical_device_id, uint32_t hdc_type,
                              DirectHccpSession **session);

// Releases a logical transport reference.  A zero reference count leaves the
// initialized process-owned RA/HDC service resident for a later Runtime; it
// is reclaimed by process exit rather than a hot Runtime teardown.  Callers
// keep ownership of their own data-plane resources.
void ReleaseDirectHccpSession(DirectHccpSession *session);

}  // namespace kvaf::direct_hccp

#endif  // KVAF_DIRECT_HCCP_SESSION_H_
