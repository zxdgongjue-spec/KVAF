#include "direct_hccp/direct_hccp_remote_transport.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <string>
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
  memory->wqe_eid = region.has_memory_eid ? region.memory_eid : region.eid;
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
  attr.ub.priority = 15U;
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
  return true;
}

}  // namespace kvaf::direct_hccp
