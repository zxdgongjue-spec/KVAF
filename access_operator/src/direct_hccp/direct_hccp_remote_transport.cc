#include "direct_hccp/direct_hccp_remote_transport.h"

#include <chrono>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <memory>
#include <sstream>
#include <thread>
#include <unistd.h>

#include "acl/acl.h"
#include "peer_link.h"
#include "direct_hccp/direct_hccp_session.h"

namespace kvaf::direct_hccp {
namespace {

constexpr std::uint32_t kCtpTpCount = 1U;
constexpr int kHccpAsyncAgain = 128301;
constexpr std::uint32_t kCtpQueryPollMs = 1U;
constexpr std::uint32_t kCtpQueryTimeoutMs = 10000U;
constexpr std::uint32_t kDeviceAivRqDepth = 256U;
constexpr std::uint32_t kAivWqeBytes = 64U;
constexpr std::uint32_t kAivWqebbsPerWqe = 4U;
using PeerRegion = peer::PeerRegion;
using AclrtGetPhyDevIdByLogicDevId = int32_t (*)(int32_t, int32_t*);

template <typename Function>
void RunLinkDiagnostics(const Function& function) noexcept {
  if (!LinkDiagnosticsEnabled()) return;
  const int saved_errno = errno;
  try {
    function();
  } catch (...) {
    std::fprintf(stderr,
                 "[KVAF_LINK_DIAG] pid=%ld side=npu event=diagnostic status=failed reason=cpp_exception\n",
                 static_cast<long>(getpid()));
  }
  errno = saved_errno;
}

std::string HexBytes(const std::uint8_t* bytes, std::size_t size) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string result(size * 2U, '0');
  for (std::size_t i = 0; i < size; ++i) {
    result[i * 2U] = digits[bytes[i] >> 4U];
    result[i * 2U + 1U] = digits[bytes[i] & 15U];
  }
  return result;
}

template <typename Function>
void LoadDiagnosticSymbol(void* library, const char* name, Function* function) {
  void* symbol = dlsym(library, name);
  *function = reinterpret_cast<Function>(symbol);
  Dl_info origin{};
  const bool located = symbol != nullptr && dladdr(symbol, &origin) != 0;
  std::fprintf(stderr,
               "[KVAF_LINK_DIAG] pid=%ld side=npu event=api symbol=%s status=%s dso=%s\n",
               static_cast<long>(getpid()), name, symbol == nullptr ? "unavailable" : "available",
               located && origin.dli_fname != nullptr ? origin.dli_fname : "unknown");
}

void LogJettyIdentity(const QpKey& key, const char* endpoint, void* context,
                      const char* role) {
  // CPU's packed peer key has a 28-byte wire representation; RA exports
  // the same fields with its eight-byte-aligned struct padded to 32 bytes.
  constexpr std::size_t kIdentityPrefixBytes = offsetof(JettyKeyIdentity, transport_mode) + sizeof(std::uint32_t);
  if (key.size != kIdentityPrefixBytes && key.size != sizeof(JettyKeyIdentity)) {
    std::fprintf(stderr,
                 "[KVAF_LINK_DIAG] pid=%ld side=npu event=jetty_identity role=%s context=%p "
                 "endpoint=%s source=exported_key status=unavailable key_bytes=%u expected_bytes=28_or_32\n",
                 static_cast<long>(getpid()), role, context, endpoint, key.size);
    return;
  }
  JettyKeyIdentity id{};
  std::memcpy(&id, key.value, kIdentityPrefixBytes);
  std::fprintf(stderr,
               "[KVAF_LINK_DIAG] pid=%ld side=npu event=jetty_identity role=%s context=%p "
               "endpoint=%s source=exported_key status=ok eid=%s uasid=%u jetty_id=%u transport_mode=%u\n",
               static_cast<long>(getpid()), role, context, endpoint,
               HexBytes(id.eid.raw, kEidBytes).c_str(), id.uasid, id.id, id.transport_mode);
}

void LogJettyReadback(const DirectHccpApi& api, void* context, void* jetty,
                      const char* role, const char* phase) {
  if (api.qp_query_batch == nullptr) {
    std::fprintf(stderr,
                 "[KVAF_LINK_DIAG] pid=%ld side=npu event=jetty_readback role=%s context=%p "
                 "jetty=%p status=unavailable reason=missing_symbol phase=%s\n",
                 static_cast<long>(getpid()), role, context, jetty, phase);
    return;
  }
  JettyAttr attr{};
  std::uint32_t count = 1U;
  void* handles[] = {jetty};
  const int ret = api.qp_query_batch(handles, &attr, &count);
  // JettyAttr.mask is a modify mask, not a query-validity bitmap. UMDK's
  // query path may return mask=0 while populating both state and threshold.
  std::fprintf(stderr,
               "[KVAF_LINK_DIAG] pid=%ld side=npu event=jetty_readback role=%s context=%p "
               "jetty=%p status=%s ret=%d returned_count=%u modify_mask=0x%x state_valid=%u state=%u "
               "rx_threshold_valid=%u rx_threshold=%u config_readback=not_exposed_by_ra_query "
               "phase=%s\n",
               static_cast<long>(getpid()), role, context, jetty,
               ret == 0 && count == 1U ? "ok" : "failed", ret, count, attr.mask,
               ret == 0 && count == 1U, attr.state,
               ret == 0 && count == 1U, attr.rx_threshold, phase);
}

// A timed-out HDC request can retain pointers to both fields until its
// asynchronous reply is consumed. Keep the allocation alive until process
// exit on timeout (there is no cancellation API), and stop further optional
// TP queries so a nonresponding service adds at most one 250 ms wait.
struct DiagnosticTpReadback {
  std::uint32_t bitmap{0x1ffffU};
  TpAttr attr{};
};
std::atomic<bool> diagnostic_tp_queries_disabled{false};

void LogTpReadback(const DirectHccpApi& api, void* context, std::uint64_t tp,
                   const char* role) {
  if (api.get_tp_attr_async == nullptr || api.get_async_req_result == nullptr ||
      diagnostic_tp_queries_disabled.load()) {
    std::fprintf(stderr,
                 "[KVAF_LINK_DIAG] pid=%ld side=npu event=tp_readback role=%s context=%p "
                 "local_tp=0x%llx status=unavailable reason=%s\n",
                 static_cast<long>(getpid()), role, context,
                 static_cast<unsigned long long>(tp),
                 diagnostic_tp_queries_disabled.load() ? "previous_timeout" : "missing_symbol");
    return;
  }
  auto state = std::make_unique<DiagnosticTpReadback>();
  void* request = nullptr;
  const int submitted = api.get_tp_attr_async(context, tp, &state->bitmap, &state->attr, &request);
  if (submitted != 0 || request == nullptr) {
    // A nonconforming provider returning a request on an error must not get
    // a dangling output pointer either. No further queries after this case.
    if (request != nullptr) {
      (void)state.release();
      diagnostic_tp_queries_disabled.store(true);
    }
    std::fprintf(stderr,
                 "[KVAF_LINK_DIAG] pid=%ld side=npu event=tp_readback role=%s context=%p "
                 "local_tp=0x%llx status=failed phase=submit ret=%d request=%p retained_buffer=%u\n",
                 static_cast<long>(getpid()), role, context,
                 static_cast<unsigned long long>(tp), submitted, request, request != nullptr);
    return;
  }
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
  int result = 0;
  int ret = 0;
  for (;;) {
    ret = api.get_async_req_result(request, &result);
    if (ret != kHccpAsyncAgain) break;
    if (std::chrono::steady_clock::now() >= deadline) {
      diagnostic_tp_queries_disabled.store(true);
      (void)state.release();
      std::fprintf(stderr,
                   "[KVAF_LINK_DIAG] pid=%ld side=npu event=tp_readback role=%s context=%p "
                   "local_tp=0x%llx status=timeout timeout_ms=250 request=%p "
                   "output_lifetime=process_exit further_tp_queries=disabled\n",
                   static_cast<long>(getpid()), role, context,
                   static_cast<unsigned long long>(tp), request);
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  // RaGetAsyncReqResult consumes the request on every terminal return.
  if (ret != 0 || result != 0) {
    std::fprintf(stderr,
                 "[KVAF_LINK_DIAG] pid=%ld side=npu event=tp_readback role=%s context=%p "
                 "local_tp=0x%llx status=failed phase=complete ret=%d request_ret=%d\n",
                 static_cast<long>(getpid()), role, context,
                 static_cast<unsigned long long>(tp), ret, result);
    return;
  }
  const auto& a = state->attr;
  std::ostringstream fields;
  const auto scalar = [&](unsigned bit, const char* name, unsigned value) {
    if ((state->bitmap & (1U << bit)) != 0U) fields << ' ' << name << '=' << value;
  };
  const auto bytes = [&](unsigned bit, const char* name, const std::uint8_t* value,
                         std::size_t size) {
    if ((state->bitmap & (1U << bit)) != 0U) fields << ' ' << name << '=' << HexBytes(value, size);
  };
  scalar(0, "retry_times_init", a.retry_times_init);
  scalar(1, "at", a.at);
  bytes(2, "sip", a.sip, sizeof(a.sip));
  bytes(3, "dip", a.dip, sizeof(a.dip));
  bytes(4, "smac", a.sma, sizeof(a.sma));
  bytes(5, "dmac", a.dma, sizeof(a.dma));
  scalar(6, "vlan_id", a.vlan_id);
  scalar(7, "vlan_en", a.vlan_en);
  scalar(8, "dscp", a.dscp);
  scalar(9, "at_times", a.at_times);
  scalar(10, "sl", a.sl);
  scalar(11, "ttl", a.ttl);
  scalar(12, "ack_udp_srcport", a.ack_udp_srcport);
  scalar(13, "data_udp_srcport", a.data_udp_srcport);
  scalar(14, "udp_srcport_range", a.udp_srcport_range);
  scalar(15, "spray_en", a.spray_en);
  scalar(16, "udp_global_en", a.udp_global_en);
  // Bit 17 describes the available SL set, not the selected SL (bit 10).
  // Some deployed providers return only this field, outside the request mask.
  if ((state->bitmap & (1U << 17U)) != 0U) {
    fields << " sl_bitmap=0x" << std::hex << a.sl_bitmap << std::dec;
  }
  std::fprintf(stderr,
               "[KVAF_LINK_DIAG] pid=%ld side=npu event=tp_readback role=%s context=%p "
               "local_tp=0x%llx status=ok requested_bitmap=0x1ffff returned_bitmap=0x%x%s "
               "read_outstanding=not_exposed credit=not_exposed mtu=not_exposed\n",
               static_cast<long>(getpid()), role, context,
               static_cast<unsigned long long>(tp), state->bitmap, fields.str().c_str());
}

template <typename Function>
bool LoadSymbol(void* library, const char* name, Function* function) {
  *function = reinterpret_cast<Function>(dlsym(library, name));
  if (*function != nullptr) return true;
  std::fprintf(stderr, "[KVAF_AIV] direct libhccp ABI missing symbol %s: %s\n", name, dlerror());
  return false;
}

std::uint32_t MakePeerCtpPsn(const DirectHccpApi& api,
                             std::uint32_t physical_device_id,
                             const char* role) {
  std::uint32_t psn = static_cast<std::uint32_t>(
      std::chrono::steady_clock::now().time_since_epoch().count() ^
      static_cast<std::int64_t>(getpid()));
  if (psn == 0U) psn = 1U;
  if (api.get_random != nullptr) {
    RaInfo info{};
    info.mode = kOfflineHdc;
    info.phy_id = physical_device_id;
    if (api.get_random(&info, &psn) != 0 || psn == 0U) {
      psn = static_cast<std::uint32_t>(
          std::chrono::steady_clock::now().time_since_epoch().count() ^
          static_cast<std::int64_t>(getpid()));
      if (psn == 0U) psn = 1U;
    }
  }
  std::fprintf(stderr,
               "[KVAF_AIV] DIRECT_CTP_PSN_READY role=%s psn=0x%x source=ra_secure_or_fallback\n",
               role, psn);
  return psn;
}

bool GetPeerCtp(const DirectHccpApi& api, void* context,
                const std::array<std::uint8_t, kEidBytes>& local_eid,
                const std::array<std::uint8_t, kEidBytes>& peer_eid,
                const char* role, std::uint64_t* tp_handle) {
  if (context == nullptr || role == nullptr || tp_handle == nullptr) return false;
  GetTpCfg cfg{};
  cfg.flag = CtpFlag();
  cfg.transport_mode = kReliableMessage;
  std::memcpy(cfg.local_eid.raw, local_eid.data(), local_eid.size());
  std::memcpy(cfg.peer_eid.raw, peer_eid.data(), peer_eid.size());
  TpInfo info{};
  std::uint32_t count = kCtpTpCount;
  void* async_request = nullptr;
  int ret = api.get_tp_info_async(context, &cfg, &info, &count, &async_request);
  if (ret != 0 || async_request == nullptr) {
    std::fprintf(stderr,
                 "[KVAF_AIV] DIRECT_CTP_ASYNC_SUBMIT_FAILED role=%s ret=%d count=%u request=%p\n",
                 role, ret, count, async_request);
    return false;
  }
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(kCtpQueryTimeoutMs);
  while (true) {
    int request_result = 0;
    ret = api.get_async_req_result(async_request, &request_result);
    if (ret == kHccpAsyncAgain) {
      if (std::chrono::steady_clock::now() >= deadline) {
        std::fprintf(stderr, "[KVAF_AIV] DIRECT_CTP_ASYNC_TIMEOUT role=%s timeout_ms=%u\n",
                     role, kCtpQueryTimeoutMs);
        return false;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(kCtpQueryPollMs));
      continue;
    }
    // RaGetAsyncReqResult consumes the request handle on every terminal
    // return. Never query or free it after this point.
    async_request = nullptr;
    if (ret != 0 || request_result != 0) {
      std::fprintf(stderr,
                   "[KVAF_AIV] DIRECT_CTP_ASYNC_COMPLETE_FAILED role=%s api_ret=%d request_ret=%d\n",
                   role, ret, request_result);
      return false;
    }
    break;
  }
  if (count == 0U || info.tp_handle == 0U) {
    std::fprintf(stderr,
                 "[KVAF_AIV] DIRECT_CTP_GET_FAILED role=%s mode=hdc_async count=%u tp=0x%llx\n",
                 role, count, static_cast<unsigned long long>(info.tp_handle));
    return false;
  }
  *tp_handle = info.tp_handle;
  std::fprintf(stderr,
               "[KVAF_AIV] DIRECT_CTP_READY role=%s mode=hdc_async tp_handle=0x%llx\n",
               role, static_cast<unsigned long long>(*tp_handle));
  return true;
}

}  // namespace

bool LinkDiagnosticsEnabled() {
  const char* value = std::getenv("KVAF_LINK_DIAGNOSTICS");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

void LogLinkContextDiagnostics(const DirectHccpApi& api, void* context,
                              std::uint32_t physical_device_id,
                              const std::array<std::uint8_t, kEidBytes>& eid,
                              const char* role) {
  RunLinkDiagnostics([&] {
    if (api.get_dev_base_attr == nullptr) {
      std::fprintf(stderr,
                   "[KVAF_LINK_DIAG] pid=%ld side=npu event=device_readback role=%s context=%p "
                   "physical_device=%u eid=%s status=unavailable reason=missing_symbol\n",
                   static_cast<long>(getpid()), role, context, physical_device_id,
                   HexBytes(eid.data(), eid.size()).c_str());
      return;
    }
    DevBaseAttr attr{};
    const int ret = api.get_dev_base_attr(context, &attr);
    if (ret != 0) {
      std::fprintf(stderr,
                   "[KVAF_LINK_DIAG] pid=%ld side=npu event=device_readback role=%s context=%p "
                   "physical_device=%u eid=%s status=failed ret=%d\n",
                   static_cast<long>(getpid()), role, context, physical_device_id,
                   HexBytes(eid.data(), eid.size()).c_str(), ret);
      return;
    }
    std::fprintf(stderr,
                 "[KVAF_LINK_DIAG] pid=%ld side=npu event=device_readback role=%s context=%p "
                 "physical_device=%u eid=%s status=ok source=ra_context_cached_device_attr "
                 "chip_id=%u die_id=%u func_id=%u ue_id=%u "
                 "sq_max_depth=%u rq_max_depth=%u sq_max_sge=%u rq_max_sge=%u "
                 "max_jfs_inline_len=%u max_jfs_rsge=%u max_read_size=%u max_write_size=%u "
                 "max_msg_size=%llu rm_tp_cap=0x%x rc_tp_cap=0x%x um_tp_cap=0x%x tp_feature=0x%x\n",
                 static_cast<long>(getpid()), role, context, physical_device_id,
                 HexBytes(eid.data(), eid.size()).c_str(), attr.ub.chip_id, attr.ub.die_id,
                 attr.ub.func_id, attr.ub.func_id, attr.sq_max_depth, attr.rq_max_depth,
                 attr.sq_max_sge, attr.rq_max_sge, attr.ub.max_jfs_inline_len,
                 attr.ub.max_jfs_rsge, attr.max_read_size, attr.max_write_size,
                 static_cast<unsigned long long>(attr.max_msg_size), attr.ub.rm_tp_cap,
                 attr.ub.rc_tp_cap, attr.ub.um_tp_cap, attr.ub.tp_feature);
    // HCCP RsUbGetUeInfo assigns DevBaseAttr.ub.funcId from ueInfo.ue_id.
    for (std::uint32_t priority = 0; priority < 16U; ++priority) {
      std::fprintf(stderr,
                   "[KVAF_LINK_DIAG] pid=%ld side=npu event=priority_mapping role=%s context=%p "
                   "priority=%u sl=%u tp_type=0x%x source=device_readback ctp_supported=%u\n",
                   static_cast<long>(getpid()), role, context, priority,
                   attr.ub.priority_info[priority].sl, attr.ub.priority_info[priority].tp_type,
                   (attr.ub.priority_info[priority].tp_type & 2U) != 0U);
    }
  });
}

void LogLinkCqDiagnostics(void* context, const CqInfo& info, void* handle,
                          const char* role) {
  RunLinkDiagnostics([&] {
    std::fprintf(stderr,
                 "[KVAF_LINK_DIAG] pid=%ld side=npu event=cq_mapping role=%s context=%p cq=%p "
                 "requested_mode=%u requested_depth=%u returned_jfc_id=%u returned_cq_va=0x%llx "
                 "returned_cq_buffer_va=0x%llx returned_cq_db_va=0x%llx returned_cqe_bytes=%u\n",
                 static_cast<long>(getpid()), role, context, handle, info.in.ub.mode, info.in.depth,
                 info.out.id, static_cast<unsigned long long>(info.out.va),
                 static_cast<unsigned long long>(info.out.buffer_addr),
                 static_cast<unsigned long long>(info.out.swdb_addr), info.out.cqe_size);
  });
}

void LogLinkJettyDiagnostics(const DirectHccpApi& api, void* context,
                             void* jetty, const char* role, const char* phase) {
  RunLinkDiagnostics([&] { LogJettyReadback(api, context, jetty, role, phase); });
}

bool LoadDirectHccp(DirectHccpApi* api) {
  if (api == nullptr) return false;
  for (const char* candidate : {"libra.so", "libra.so.0", "libhccp.so", "libhccp.so.0"}) {
    api->library = dlopen(candidate, RTLD_NOW | RTLD_LOCAL);
    if (api->library != nullptr) break;
  }
  if (api->library == nullptr) {
    std::fprintf(stderr, "[KVAF_AIV] dlopen direct RA ABI (libra.so) failed: %s\n", dlerror());
    return false;
  }
  const bool loaded =
      LoadSymbol(api->library, "RaCtxInit", &api->ctx_init) &&
      LoadSymbol(api->library, "RaCtxDeinit", &api->ctx_deinit) &&
      LoadSymbol(api->library, "RaCtxTokenIdAlloc", &api->token_alloc) &&
      LoadSymbol(api->library, "RaCtxTokenIdFree", &api->token_free) &&
      LoadSymbol(api->library, "RaCtxLmemRegister", &api->lmem_register) &&
      LoadSymbol(api->library, "RaCtxLmemUnregister", &api->lmem_unregister) &&
      LoadSymbol(api->library, "RaCtxRmemImport", &api->rmem_import) &&
      LoadSymbol(api->library, "RaCtxRmemUnimport", &api->rmem_unimport) &&
      LoadSymbol(api->library, "RaCtxCqCreate", &api->cq_create) &&
      LoadSymbol(api->library, "RaCtxCqDestroy", &api->cq_destroy) &&
      LoadSymbol(api->library, "RaCtxQpCreate", &api->qp_create) &&
      LoadSymbol(api->library, "RaCtxQpDestroy", &api->qp_destroy) &&
      LoadSymbol(api->library, "RaCtxQpImport", &api->qp_import) &&
      LoadSymbol(api->library, "RaCtxQpUnimport", &api->qp_unimport) &&
      LoadSymbol(api->library, "RaCtxGetTpInfoList", &api->get_tp_info) &&
      LoadSymbol(api->library, "RaGetTpInfoListAsync", &api->get_tp_info_async) &&
      LoadSymbol(api->library, "RaGetAsyncReqResult", &api->get_async_req_result);
  api->get_random = reinterpret_cast<GetRandomFn>(dlsym(api->library, "RaGetSecRandom"));
  if (!loaded) {
    dlclose(api->library);
    *api = {};
    return false;
  }
  RunLinkDiagnostics([&] {
    LoadDiagnosticSymbol(api->library, "RaGetDevBaseAttr", &api->get_dev_base_attr);
    LoadDiagnosticSymbol(api->library, "RaCtxQpQueryBatch", &api->qp_query_batch);
    LoadDiagnosticSymbol(api->library, "RaGetTpAttrAsync", &api->get_tp_attr_async);
    Dl_info origin{};
    const bool located = dladdr(reinterpret_cast<void*>(api->ctx_init), &origin) != 0;
    std::fprintf(stderr,
                 "[KVAF_LINK_DIAG] pid=%ld side=npu event=abi symbol=RaCtxInit dso=%s "
                 "dev_base_attr_bytes=%zu jetty_attr_bytes=%zu tp_attr_bytes=%zu "
                 "jetty_key_identity_bytes=%zu phase=initialization_only\n",
                 static_cast<long>(getpid()), located && origin.dli_fname != nullptr ? origin.dli_fname : "unknown",
                 sizeof(DevBaseAttr), sizeof(JettyAttr), sizeof(TpAttr), sizeof(JettyKeyIdentity));
  });
  std::fprintf(stderr, "[KVAF_AIV] DIRECT_HCCP_TRANSPORT_API_READY backend=RaCtx_no_HCOMM\n");
  return true;
}

void CloseDirectHccp(DirectHccpApi* api) {
  if (api == nullptr) return;
  // A successfully published transport shares the RA/HDC process bootstrap.
  // Keep this DSO mapped until process exit: RA can retain service/TLS
  // callbacks after an individual Runtime has torn down its queues.
  // LoadDirectHccp still dlcloses an API handle that failed before publication.
  *api = {};
}

bool GetCurrentNpuPhysicalDeviceId(std::uint32_t* physical_device_id) {
  if (physical_device_id == nullptr) return false;
  int32_t logical_device_id = -1;
  if (aclrtGetDevice(&logical_device_id) != ACL_SUCCESS || logical_device_id < 0) {
    return false;
  }
  auto get_physical = reinterpret_cast<AclrtGetPhyDevIdByLogicDevId>(
      dlsym(RTLD_DEFAULT, "aclrtGetPhyDevIdByLogicDevId"));
  if (get_physical == nullptr) {
    void* ascendcl = dlopen("libascendcl.so", RTLD_NOW | RTLD_NOLOAD);
    if (ascendcl == nullptr) ascendcl = dlopen("libascendcl.so", RTLD_NOW | RTLD_LOCAL);
    if (ascendcl != nullptr) {
      get_physical = reinterpret_cast<AclrtGetPhyDevIdByLogicDevId>(
          dlsym(ascendcl, "aclrtGetPhyDevIdByLogicDevId"));
    }
  }
  int32_t physical = -1;
  if (get_physical == nullptr || get_physical(logical_device_id, &physical) != 0 || physical < 0) {
    return false;
  }
  *physical_device_id = static_cast<std::uint32_t>(physical);
  return true;
}

void ReleaseDeviceAivTransport(const DeviceAivTransportResources& resources) {
  if (resources.api == nullptr) return;
  DirectHccpApi& api = *resources.api;
  const auto context_for_worker = [&resources](
                                      const DeviceAivWorkerResources& worker) {
    if (resources.paths != nullptr && worker.path_index < resources.path_count) {
      return resources.paths[worker.path_index].device_context;
    }
    return resources.device_context;
  };
  std::fprintf(stderr, "[KVAF_AIV] DIRECT_HCCP_TEARDOWN phase=begin\n");
  if (api.qp_unimport != nullptr) {
    std::fprintf(stderr, "[KVAF_AIV] DIRECT_HCCP_TEARDOWN phase=unimport_jettys\n");
    for (std::uint32_t index = 0U; index < resources.worker_count; ++index) {
      DeviceAivWorkerResources& worker = resources.workers[index];
      void* context = context_for_worker(worker);
      if (context != nullptr && worker.device_peer_target_jetty != nullptr) {
        (void)api.qp_unimport(context, worker.device_peer_target_jetty);
        worker.device_peer_target_jetty = nullptr;
      }
    }
  }
  if (api.qp_destroy != nullptr) {
    std::fprintf(stderr, "[KVAF_AIV] DIRECT_HCCP_TEARDOWN phase=destroy_jettys\n");
    for (std::uint32_t index = 0U; index < resources.worker_count; ++index) {
      DeviceAivWorkerResources& worker = resources.workers[index];
      if (worker.device_user_ctl_jetty.handle != nullptr) {
        (void)api.qp_destroy(worker.device_user_ctl_jetty.handle);
        worker.device_user_ctl_jetty.handle = nullptr;
      }
    }
  }
  if (api.cq_destroy != nullptr) {
    std::fprintf(stderr, "[KVAF_AIV] DIRECT_HCCP_TEARDOWN phase=destroy_completion_queues\n");
    for (std::uint32_t index = 0U; index < resources.worker_count; ++index) {
      DeviceAivWorkerResources& worker = resources.workers[index];
      void* context = context_for_worker(worker);
      if (context != nullptr && worker.device_user_ctl_cq != nullptr) {
        (void)api.cq_destroy(context, worker.device_user_ctl_cq);
        worker.device_user_ctl_cq = nullptr;
      }
    }
  }
  if (resources.paths != nullptr) {
    std::fprintf(stderr,
                 "[KVAF_AIV] DIRECT_HCCP_TEARDOWN phase=release_eid_paths count=%u\n",
                 resources.path_count);
    for (std::uint32_t reverse = resources.path_count; reverse > 0U; --reverse) {
      DeviceAivPathResources& path = resources.paths[reverse - 1U];
      if (api.rmem_unimport != nullptr && path.device_context != nullptr &&
          path.device_peer_memory_import != nullptr) {
        (void)api.rmem_unimport(path.device_context,
                                path.device_peer_memory_import);
        path.device_peer_memory_import = nullptr;
      }
      if (api.token_free != nullptr && path.device_context != nullptr &&
          path.device_transport_token_handle != nullptr) {
        (void)api.token_free(path.device_context,
                             path.device_transport_token_handle);
        path.device_transport_token_handle = nullptr;
      }
      if (api.ctx_deinit != nullptr && path.device_context != nullptr) {
        (void)api.ctx_deinit(path.device_context);
        path.device_context = nullptr;
      }
    }
  } else {
    if (api.rmem_unimport != nullptr && resources.device_context != nullptr &&
        resources.device_peer_memory_import != nullptr) {
      std::fprintf(stderr,
                   "[KVAF_AIV] DIRECT_HCCP_TEARDOWN phase=unimport_peer_memory\n");
      (void)api.rmem_unimport(resources.device_context,
                              resources.device_peer_memory_import);
    }
    if (api.token_free != nullptr && resources.device_context != nullptr &&
        resources.device_transport_token_handle != nullptr) {
      std::fprintf(stderr,
                   "[KVAF_AIV] DIRECT_HCCP_TEARDOWN phase=free_tokens\n");
      (void)api.token_free(resources.device_context,
                           resources.device_transport_token_handle);
    }
    if (api.ctx_deinit != nullptr && resources.device_context != nullptr) {
      std::fprintf(stderr,
                   "[KVAF_AIV] DIRECT_HCCP_TEARDOWN phase=deinit_contexts\n");
      (void)api.ctx_deinit(resources.device_context);
    }
  }
  if (resources.session != nullptr) {
    std::fprintf(stderr, "[KVAF_AIV] DIRECT_HCCP_TEARDOWN phase=release_session\n");
    ReleaseDirectHccpSession(resources.session);
  }
  std::fprintf(stderr, "[KVAF_AIV] DIRECT_HCCP_TEARDOWN phase=close_dynamic_libraries\n");
  CloseDirectHccp(&api);
}

bool LoadPeerMemory(PeerLink* link, std::uint64_t peer_base,
                    DirectMemory* memory,
                    std::array<std::uint8_t, kEidBytes>* peer_eid) {
  PeerRegion region{};
  if (link == nullptr || memory == nullptr || peer_eid == nullptr) return false;
  const Status link_status = link->LookupRegion(peer_base, &region);
  if (!link_status.ok() || region.length == 0U || region.token_id == 0U || region.token_value == 0U ||
      region.segment_key_bytes == 0U || region.segment_key_bytes > kMemKeyBytes) {
    const char* reason = link_status.ok() ? "invalid peer region descriptor" : link_status.message().c_str();
    std::fprintf(stderr, "[KVAF_AIV] PEER_LINK_REGION_LOOKUP_FAILED base=0x%llx reason=%s\n",
                 static_cast<unsigned long long>(peer_base), reason);
    return false;
  }
  const std::uint32_t wqe_token_id = region.token_id >> 8U;
  if (wqe_token_id == 0U) {
    std::fprintf(stderr, "[KVAF_AIV] PEER_LINK_REGION_TOKEN_INVALID raw_token_id=0x%x\n", region.token_id);
    return false;
  }
  std::fprintf(stderr, "[KVAF_AIV] PEER_LINK_DDR_TOKEN raw_token_id=0x%x wqe_token_id=0x%x\n",
               region.token_id, wqe_token_id);
  *memory = {};
  memory->registered_address = region.base;
  memory->registered_bytes = region.length;
  memory->token_id = wqe_token_id;
  memory->token_value = region.token_value;
  memory->key_bytes = region.segment_key_bytes;
  std::memcpy(memory->key, region.segment_key.data(), memory->key_bytes);
  std::memcpy(peer_eid->data(), region.eid.data(), peer_eid->size());
  return true;
}

bool ImportPeerJetty(const DirectHccpApi& api, void* context, const std::uint8_t* peer_key,
                     std::uint32_t peer_key_bytes, std::uint32_t peer_token_value,
                     std::uint64_t local_tp, std::uint64_t peer_tp, std::uint32_t local_psn,
                     std::uint32_t peer_psn, const char* role, QpImportInfoT* import_info,
                     void** target_jetty) {
  if (context == nullptr || peer_key == nullptr || peer_key_bytes == 0U ||
      peer_key_bytes > kQpKeyBytes || peer_token_value == 0U || local_tp == 0U || peer_tp == 0U ||
      local_psn == 0U || peer_psn == 0U || import_info == nullptr || target_jetty == nullptr) {
    return false;
  }
  *import_info = {};
  *target_jetty = nullptr;
  std::memcpy(import_info->in.key.value, peer_key, peer_key_bytes);
  import_info->in.key.size = peer_key_bytes;
  import_info->in.ub.mode = kJettyImportExp;
  import_info->in.ub.token_value = peer_token_value;
  import_info->in.ub.policy = kJettyGroupRoundRobin;
  import_info->in.ub.type = kTargetJetty;
  import_info->in.ub.flag = ImportFlagPlainTextToken();
  import_info->in.ub.exp_import_cfg.tp_handle = local_tp;
  import_info->in.ub.exp_import_cfg.peer_tp_handle = peer_tp;
  import_info->in.ub.exp_import_cfg.tag = 0U;
  import_info->in.ub.exp_import_cfg.tx_psn = local_psn;
  import_info->in.ub.exp_import_cfg.rx_psn = peer_psn;
  import_info->in.ub.tp_type = CtpProtocolType();
  std::fprintf(stderr,
               "[KVAF_AIV] PEER_LINK_JETTY_IMPORT_ATTEMPT role=%s local_tp=0x%llx peer_tp=0x%llx tx_psn=0x%x rx_psn=0x%x\n",
               role, static_cast<unsigned long long>(local_tp), static_cast<unsigned long long>(peer_tp),
               local_psn, peer_psn);
  const int ret = api.qp_import(context, import_info, target_jetty);
  if (ret != 0 || *target_jetty == nullptr || import_info->out.ub.tpn == 0U) {
    std::fprintf(stderr, "[KVAF_AIV] PEER_LINK_JETTY_IMPORT_FAILED role=%s ret=%d target=%p tpn=%u\n",
                 role, ret, *target_jetty, import_info->out.ub.tpn);
    return false;
  }
  std::fprintf(stderr, "[KVAF_AIV] PEER_LINK_JETTY_IMPORT_READY role=%s tpn=%u\n", role,
               import_info->out.ub.tpn);
  return true;
}

bool CreateDeviceAivJetty(const DirectHccpApi& api, void* context,
                          void* cq_handle, void* token_handle,
                          std::uint32_t token_value, std::uint32_t sq_depth,
                          const char* role, QpCreateInfo* jetty_info,
                          void** jetty_handle) {
  if (api.qp_create == nullptr || context == nullptr || cq_handle == nullptr ||
      token_handle == nullptr || token_value == 0U || sq_depth == 0U ||
      role == nullptr || jetty_info == nullptr || jetty_handle == nullptr) {
    return false;
  }
  const std::uint64_t sq_bytes = static_cast<std::uint64_t>(sq_depth) *
                                 kAivWqebbsPerWqe * kAivWqeBytes;
  if (sq_bytes > UINT32_MAX) {
    std::fprintf(stderr,
                 "[KVAF_AIV] DIRECT_AIV_DEV_USED_QUEUE_INVALID role=%s sq_depth=%u requested_sq_bytes=%llu\n",
                 role, sq_depth, static_cast<unsigned long long>(sq_bytes));
    return false;
  }

  QpCreateAttr attr{};
  attr.scq_handle = cq_handle;
  attr.rcq_handle = cq_handle;
  attr.srq_handle = cq_handle;
  attr.sq_depth = sq_depth;
  attr.rq_depth = kDeviceAivRqDepth;
  attr.transport_mode = kReliableMessage;
  attr.ub.mode = kJettyUserCtlNormal;
  attr.ub.jetty_id = 0U;
  attr.ub.flag = JettyFlagShareJfr();
  attr.ub.jfs_flag = JfsFlagErrorSuspend();
  attr.ub.token_id_handle = token_handle;
  attr.ub.token_value = token_value;
  attr.ub.priority = 2U;
  attr.ub.rnr_retry = 7U;
  attr.ub.err_timeout = 16U;
  attr.ub.ext_mode.sqebb_num = sq_depth;
  attr.ub.ext_mode.sq.buffer_size = static_cast<std::uint32_t>(sq_bytes);
  attr.ub.ext_mode.sq.buffer_va = 0U;
  attr.ub.ext_mode.cstm_flag = 0U;

  *jetty_info = {};
  *jetty_handle = nullptr;
  const int ret = api.qp_create(context, &attr, jetty_info, jetty_handle);
  if (ret != 0 || *jetty_handle == nullptr || jetty_info->key.size == 0U ||
      jetty_info->key.size > kQpKeyBytes) {
    std::fprintf(stderr,
                 "[KVAF_AIV] DIRECT_JETTY_CREATE_FAILED role=%s mode=%u sq_depth=%u sqebb_num=%u "
                 "cstm_flag=%u requested_sq_bytes=%u ret=%d handle=%p key_bytes=%u\n",
                 role, attr.ub.mode, sq_depth, attr.ub.ext_mode.sqebb_num,
                 attr.ub.ext_mode.cstm_flag, attr.ub.ext_mode.sq.buffer_size,
                 ret, *jetty_handle, jetty_info->key.size);
    return false;
  }
  if (jetty_info->ub.sq_buffer_va == 0U || jetty_info->ub.db_addr == 0U ||
      jetty_info->ub.wqebb_size != kAivWqeBytes) {
    std::fprintf(stderr,
                 "[KVAF_AIV] DIRECT_USER_CTL_QUEUE_CONTRACT_INVALID role=%s sq_va=0x%llx db_va=0x%llx "
                 "returned_wqebb_bytes=%llu expected_wqebb_bytes=%u\n",
                 role, static_cast<unsigned long long>(jetty_info->ub.sq_buffer_va),
                 static_cast<unsigned long long>(jetty_info->ub.db_addr),
                 static_cast<unsigned long long>(jetty_info->ub.wqebb_size),
                 kAivWqeBytes);
    return false;
  }
  std::fprintf(stderr,
               "[KVAF_AIV] DIRECT_JETTY_READY role=%s mode=%u cstm_flag=%u jetty_id=%u sq_va=0x%llx "
               "db_va=0x%llx returned_wqebb_bytes=%llu\n",
               role, attr.ub.mode, attr.ub.ext_mode.cstm_flag, jetty_info->ub.id,
               static_cast<unsigned long long>(jetty_info->ub.sq_buffer_va),
               static_cast<unsigned long long>(jetty_info->ub.db_addr),
               static_cast<unsigned long long>(jetty_info->ub.wqebb_size));
  std::fprintf(stderr,
               "[KVAF_AIV] DIRECT_USER_CTL_QUEUE_CONTRACT role=%s profile=aiv_dev_used "
               "sq_depth_entries=%u sqebb_num=%u requested_sq_buffer_bytes=%u "
               "returned_sq_va=0x%llx returned_wqebb_bytes=%llu db_addr_is_post_offset=1 sq_cstm=0\n",
               role, sq_depth, attr.ub.ext_mode.sqebb_num, attr.ub.ext_mode.sq.buffer_size,
               static_cast<unsigned long long>(jetty_info->ub.sq_buffer_va),
               static_cast<unsigned long long>(jetty_info->ub.wqebb_size));
  RunLinkDiagnostics([&] {
    std::fprintf(stderr,
                 "[KVAF_LINK_DIAG] pid=%ld side=npu event=jetty_mapping role=%s context=%p "
                 "jetty=%p scq=%p rcq=%p srq=%p requested_transport_mode=%u requested_mode=%u "
                 "requested_sq_depth=%u requested_rq_depth=%u requested_sqebb_num=%u "
                 "requested_sq_bytes=%u requested_priority=%u requested_rnr_retry=%u requested_err_timeout=%u "
                 "requested_jetty_flags=0x%x requested_jfs_flags=0x%x requested_cstm_flags=0x%x "
                 "returned_uasid=%u returned_jetty_id=%u returned_sq_va=0x%llx returned_sq_db_va=0x%llx "
                 "returned_wqebb_bytes=%llu\n",
                 static_cast<long>(getpid()), role, context, *jetty_handle,
                 cq_handle, cq_handle, cq_handle, attr.transport_mode, attr.ub.mode,
                 attr.sq_depth, attr.rq_depth, attr.ub.ext_mode.sqebb_num,
                 attr.ub.ext_mode.sq.buffer_size, attr.ub.priority, attr.ub.rnr_retry,
                 attr.ub.err_timeout, attr.ub.flag, attr.ub.jfs_flag, attr.ub.ext_mode.cstm_flag,
                 jetty_info->ub.uasid, jetty_info->ub.id,
                 static_cast<unsigned long long>(jetty_info->ub.sq_buffer_va),
                 static_cast<unsigned long long>(jetty_info->ub.db_addr),
                 static_cast<unsigned long long>(jetty_info->ub.wqebb_size));
    LogJettyIdentity(jetty_info->key, "local", context, role);
    LogJettyReadback(api, context, *jetty_handle, role, "after_create_before_import");
  });
  return true;
}

bool CreatePeerChannel(const PeerChannelRequest& request,
                       PeerChannel* channel) {
  if (request.link == nullptr || request.api == nullptr || request.device_context == nullptr ||
      request.peer_base == 0U || request.peer_memory == nullptr ||
      request.local_eid == nullptr || request.peer_eid == nullptr ||
      request.local_jetty_info == nullptr || request.local_jetty_info->key.size == 0U ||
      request.local_jetty_info->key.size > kQpKeyBytes || request.role == nullptr ||
      channel == nullptr) {
    return false;
  }
  *channel = {};
  std::uint64_t local_tp_handle = 0U;
  if (!GetPeerCtp(*request.api, request.device_context, *request.local_eid,
                  *request.peer_eid, request.role, &local_tp_handle)) {
    return false;
  }
  const std::uint32_t local_psn =
      MakePeerCtpPsn(*request.api, request.physical_device_id, request.role);
  std::array<std::uint8_t, kQpKeyBytes> peer_qp_key{};
  std::uint32_t peer_qp_key_bytes = 0U;
  std::uint64_t peer_tp_handle = 0U;
  std::uint32_t peer_psn = 0U;
  std::fprintf(stderr,
               "[KVAF_AIV] PEER_LINK_CHANNEL_REQUEST role=%s local_qp_key_bytes=%u\n",
               request.role, request.local_jetty_info->key.size);
  const Status status = request.link->CreateChannel(
      request.peer_base, request.local_jetty_info->key.value,
      request.local_jetty_info->key.size, request.local_eid->data(),
      request.peer_memory->token_value, local_tp_handle, local_psn, &peer_qp_key,
      &peer_qp_key_bytes, &peer_tp_handle, &peer_psn);
  if (!status.ok()) {
    std::fprintf(stderr, "[KVAF_AIV] PEER_LINK_CHANNEL_FAILED role=%s reason=%s\n",
                 request.role, status.message().c_str());
    return false;
  }
  if (!ImportPeerJetty(*request.api, request.device_context, peer_qp_key.data(),
                       peer_qp_key_bytes, request.peer_memory->token_value,
                       local_tp_handle, peer_tp_handle, local_psn, peer_psn,
                       request.role, &channel->local_import_info,
                       &channel->local_target_jetty)) {
    std::fprintf(stderr,
                 "[KVAF_AIV] PEER_LINK_CHANNEL_FAILED role=%s reason=import_peer_jetty\n",
                 request.role);
    return false;
  }
  RunLinkDiagnostics([&] {
    QpKey remote_key{};
    remote_key.size = static_cast<std::uint8_t>(peer_qp_key_bytes);
    std::memcpy(remote_key.value, peer_qp_key.data(), peer_qp_key_bytes);
    LogJettyIdentity(request.local_jetty_info->key, "local", request.device_context, request.role);
    LogJettyIdentity(remote_key, "peer", request.device_context, request.role);
    std::fprintf(stderr,
                 "[KVAF_LINK_DIAG] pid=%ld side=npu event=channel_mapping role=%s context=%p "
                 "physical_device=%u local_eid=%s peer_eid=%s local_uasid=%u local_jetty_id=%u "
                 "local_tp=0x%llx peer_tp=0x%llx returned_local_tpn=%u peer_tpn=unavailable "
                 "tx_psn=0x%x rx_psn=0x%x requested_tag=0 requested_tp_type=%u requested_import_flags=0x%x "
                 "target_jetty=%p returned_target_jetty=0x%llx peer_base=0x%llx peer_bytes=%llu\n",
                 static_cast<long>(getpid()), request.role, request.device_context,
                 request.physical_device_id,
                 HexBytes(request.local_eid->data(), request.local_eid->size()).c_str(),
                 HexBytes(request.peer_eid->data(), request.peer_eid->size()).c_str(),
                 request.local_jetty_info->ub.uasid, request.local_jetty_info->ub.id,
                 static_cast<unsigned long long>(local_tp_handle),
                 static_cast<unsigned long long>(peer_tp_handle), channel->local_import_info.out.ub.tpn,
                 local_psn, peer_psn, CtpProtocolType(), ImportFlagPlainTextToken(),
                 channel->local_target_jetty,
                 static_cast<unsigned long long>(channel->local_import_info.out.ub.target_jetty_handle),
                 static_cast<unsigned long long>(request.peer_base),
                 static_cast<unsigned long long>(request.peer_memory->registered_bytes));
    LogTpReadback(*request.api, request.device_context, local_tp_handle, request.role);
  });
  return true;
}

}  // namespace kvaf::direct_hccp
