#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <dlfcn.h>

namespace kvaf::direct_hccp {

// Public HCCP hccp_ctx.h JettyAttr / RaCtxQpQueryBatch ABI. Keep this small
// query separate from the opaque, product-dependent Jetty context buffer.
struct JettyDiagnosticAttr {
  std::uint32_t mask;
  std::uint32_t rx_threshold;
  std::int32_t state;
  std::uint32_t reserved[2];
};
static_assert(sizeof(JettyDiagnosticAttr) == 20U);
static_assert(alignof(JettyDiagnosticAttr) == 4U);
static_assert(offsetof(JettyDiagnosticAttr, state) == 8U);
static_assert(offsetof(JettyDiagnosticAttr, reserved) == 12U);

using QueryJettyDiagnosticFn = int (*)(void*[], JettyDiagnosticAttr[], unsigned int*);

inline const char* JettyDiagnosticStateName(std::int32_t state) {
  switch (state) {
    case 0: return "reset";
    case 1: return "ready";
    case 2: return "suspended";
    case 3: return "error";
    default: return "unknown";
  }
}

// Failure-only observation. Never load another RA image, change a queue, or
// promote diagnostic failure over the original Put error. Query each local
// Jetty separately because host and device handles belong to different RA
// contexts. The already-loaded production RA library remains caller-owned.
inline void LogBatchPutJettyState(void* ra_library, void* jetty_handle,
                                 const char* role, std::uint32_t core,
                                 std::uint32_t jetty_slot,
                                 bool peer_online = false) {
  const int original_errno = errno;
  const char* role_name = role == nullptr ? "unknown" : role;
  if (peer_online) {
    // Public RA implementation: gRaPeerCtxOps.raCtxQueryQpBatch is NULL;
    // QpQueryBatchParamCheck returns -EINVAL, converted to RDMA error 128103.
    // This is query support for the peer-online context, not a link failure.
    // The offline-HDC device context has its own supported query callback.
    std::fprintf(stderr,
                 "[KVAF_AIV] BATCH_PUT_JETTY_STATE core=%u jetty=%u role=%s "
                 "api=RaCtxQpQueryBatch query_status=unsupported_context "
                 "context=peer_online link_state=unknown api_ret=not_called\n",
                 core, jetty_slot, role_name);
    errno = original_errno;
    return;
  }
  const auto query = ra_library == nullptr ? nullptr :
      reinterpret_cast<QueryJettyDiagnosticFn>(dlsym(ra_library, "RaCtxQpQueryBatch"));
  if (jetty_handle == nullptr || query == nullptr) {
    std::fprintf(stderr,
                 "[KVAF_AIV] BATCH_PUT_JETTY_STATE core=%u jetty=%u role=%s "
                 "api=RaCtxQpQueryBatch availability=%s api_ret=not_called\n",
                 core, jetty_slot, role_name,
                 jetty_handle == nullptr ? "missing_handle" : "symbol_unavailable");
  } else {
    void* handles[] = {jetty_handle};
    JettyDiagnosticAttr attribute{};
    unsigned int count = 1U;
    const int result = query(handles, &attribute, &count);
    if (result == 0 && count == 1U) {
      std::fprintf(stderr,
                   "[KVAF_AIV] BATCH_PUT_JETTY_STATE core=%u jetty=%u role=%s "
                   "api=RaCtxQpQueryBatch availability=available api_ret=%d returned=%u "
                   "mask=0x%x state=%d state_name=%s\n",
                   core, jetty_slot, role_name, result, count, attribute.mask,
                   attribute.state, JettyDiagnosticStateName(attribute.state));
    } else {
      std::fprintf(stderr,
                   "[KVAF_AIV] BATCH_PUT_JETTY_STATE core=%u jetty=%u role=%s "
                   "api=RaCtxQpQueryBatch availability=available api_ret=%d returned=%u "
                   "state_name=unavailable\n",
                   core, jetty_slot, role_name, result, count);
    }
  }
  errno = original_errno;
}

}  // namespace kvaf::direct_hccp
