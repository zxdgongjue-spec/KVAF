#include "kvaf_aiv_batch_put.h"

#include <array>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>

#include <dlfcn.h>

#include "acl/acl.h"
#include "batch_put_internal.h"
#include "direct_hccp/direct_hccp_abi.h"
#include "direct_hccp/direct_hccp_remote_transport.h"
#include "direct_hccp/direct_hccp_local_host.h"
#include "direct_hccp/direct_hccp_session.h"

extern "C" int32_t KvafExecuteBatchPutLaunchInternal(
    const KvafExecuteBatchPutLaunchArgs *arguments, aclrtStream stream);

namespace hccp = kvaf::direct_hccp;

constexpr uint64_t kPageBytes = 4096U;
constexpr uint32_t kDirectEntityMagic = 0x4b564155U;  // "KVAU"
constexpr uint32_t kDirectEntityVersion = 1U;
constexpr uint32_t kDeviceUserCtlSqDepth = 8192U;
constexpr uint32_t kWqeBytes = 64U;
constexpr uint32_t kWqebbPerWqe = 4U;
constexpr uint32_t kDeviceUserCtlCqDepth = 16384U;
constexpr uint32_t kTokenIdShift = 8U;
constexpr uint32_t kPidHdcType = 18U;
constexpr uint32_t kMaxDirectAivWorkers =
    KVAF_AIV_BATCH_PUT_MAX_AIV_CORES * KVAF_AIV_BATCH_PUT_MAX_JETTYS_PER_CORE;

using DirectHccpApi = hccp::DirectHccpApi;
using DirectMemory = hccp::DirectMemory;

struct ProcessTokenQueryInfo {
  uint64_t address;
  uint64_t bytes;
  uint32_t token_id;
  uint32_t token_value;
};
static_assert(sizeof(ProcessTokenQueryInfo) == 24U);

// HCCL queries the runtime process token for ACL-owned HBM. This is an
// opt-in comparison with the existing RA-registered token, not a replacement
// for the V9 path. CANN's QUERY_PROCESS_TOKEN command is zero.
static bool QueryAclProcessToken(void *address, uint64_t bytes,
                                 uint32_t *token_id) {
  if (token_id == nullptr || address == nullptr || bytes == 0U) return false;
  void *library = dlopen("libruntime.so", RTLD_NOW | RTLD_LOCAL);
  if (library == nullptr) {
    std::fprintf(stderr, "[KVAF_AIV] BATCH_PUT_PROCESS_TOKEN_QUERY_FAILED phase=dlopen\n");
    return false;
  }
  using QueryFn = int (*)(int, void *);
  const auto query = reinterpret_cast<QueryFn>(dlsym(library, "rtUbDevQueryInfo"));
  if (query == nullptr) {
    std::fprintf(stderr, "[KVAF_AIV] BATCH_PUT_PROCESS_TOKEN_QUERY_FAILED phase=dlsym\n");
    dlclose(library);
    return false;
  }
  ProcessTokenQueryInfo info{reinterpret_cast<uint64_t>(address), bytes, 0U, 0U};
  const int result = query(0, &info);
  dlclose(library);
  if (result != 0 || info.token_id < (1U << kTokenIdShift)) {
    std::fprintf(stderr,
                 "[KVAF_AIV] BATCH_PUT_PROCESS_TOKEN_QUERY_FAILED phase=query result=%d raw_token=0x%x\n",
                 result, info.token_id);
    return false;
  }
  *token_id = info.token_id >> kTokenIdShift;
  std::fprintf(stderr,
               "[KVAF_AIV] BATCH_PUT_PROCESS_TOKEN_READY raw_token_id=0x%x wqe_token_id=0x%x\n",
               info.token_id, *token_id);
  return true;
}

// A USER_CTL SQ/CQ is strictly single-producer/single-consumer from the AIV
// kernel's point of view. Each core/Jetty slot therefore owns one complete
// local AIV Jetty rather than sharing producer/consumer indices.
using DirectAivWorker = hccp::DeviceAivWorkerResources;

struct KvafBatchPutTransport;

struct KvafBatchPutLocalBuffer {
  KvafBatchPutTransport *transport = nullptr;
  void *token_handle = nullptr;
  DirectMemory memory{};
  uint64_t input_hbm_address = 0U;
  uint64_t input_hbm_bytes = 0U;
  void *device_entities = nullptr;
  uint64_t device_channel_entity = 0U;
  uint32_t reference_count = 0U;
};

struct KvafBatchPutTransport {
  DirectHccpApi api{};
  hccp::DirectHccpSession *session = nullptr;
  uint32_t physical_device_id = 0;
  uint32_t aiv_core_count = 0U;
  uint32_t jettys_per_core = 0U;
  uint32_t worker_count = 0U;
  void *device_context = nullptr;
  void *device_transport_token_handle = nullptr;
  uint32_t device_transport_token_value = 0U;
  std::array<uint8_t, hccp::kEidBytes> wqe_remote_eid{};
  uint64_t peer_memory_address = 0U;
  uint64_t peer_memory_bytes = 0U;
  DirectMemory peer_memory{};
  void *device_peer_memory_import = nullptr;
  std::array<DirectAivWorker, kMaxDirectAivWorkers> workers{};
  std::array<KvafBatchPutDirectUrmaEntity, kMaxDirectAivWorkers> entity_templates{};
  void *device_queue_state = nullptr;
  std::vector<std::unique_ptr<KvafBatchPutLocalBuffer>> local_buffers{};
  std::mutex mutex{};
  std::mutex launch_mutex{};
  hccp::PeerLink *peer_link{nullptr};
  std::unique_ptr<hccp::LocalHostEndpoint> local_host;
  bool retirement_started{false};
};

static std::string EidToHex(const std::array<uint8_t, hccp::kEidBytes> &eid) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string result;
  result.reserve(hccp::kEidBytes * 2U);
  for (const uint8_t byte : eid) {
    result.push_back(kHex[byte >> 4U]);
    result.push_back(kHex[byte & 0x0fU]);
  }
  return result;
}

static std::array<uint8_t, hccp::kEidBytes> ReverseEidForAivWqe(
    const std::array<uint8_t, hccp::kEidBytes> &ra_host_eid) {
  std::array<uint8_t, hccp::kEidBytes> wqe_remote_eid{};
  for (uint32_t index = 0U; index < hccp::kEidBytes; ++index) {
    wqe_remote_eid[index] = ra_host_eid[hccp::kEidBytes - 1U - index];
  }
  hccp::VerboseLog(
      "[KVAF_AIV] DIRECT_AIV_WQE_EID_ENCODING format=reverse ra_host_eid=%s wqe_remote_eid=%s\n",
      EidToHex(ra_host_eid).c_str(), EidToHex(wqe_remote_eid).c_str());
  return wqe_remote_eid;
}

static bool InitContext(const DirectHccpApi &api, int32_t mode, uint32_t physical_device_id,
                 const std::array<uint8_t, hccp::kEidBytes> &eid, const char *role, void **context) {
  hccp::CtxInitCfg cfg{};
  cfg.mode = mode;
  hccp::CtxInitAttr attr{};
  attr.phy_id = physical_device_id;
  attr.ub.eid_index = 0U;
  std::memcpy(attr.ub.eid.raw, eid.data(), eid.size());
  const int ret = api.ctx_init(&cfg, &attr, context);
  if (ret != 0 || *context == nullptr) {
    std::fprintf(stderr, "[KVAF_AIV] DIRECT_HCCP_CONTEXT_FAILED role=%s mode=%d physical=%u ret=%d context=%p\n",
                 role, mode, physical_device_id, ret, *context);
    return false;
  }
  hccp::VerboseLog(
      "[KVAF_AIV] DIRECT_HCCP_CONTEXT_READY role=%s mode=%d physical=%u context=%p\n",
      role, mode, physical_device_id, *context);
  return true;
}

static bool AllocateToken(const DirectHccpApi &api, void *context, const char *role, void **token_handle,
                   uint32_t *token_id) {
  hccp::TokenId raw{};
  const int ret = api.token_alloc(context, &raw, token_handle);
  if (ret != 0 || *token_handle == nullptr) {
    std::fprintf(stderr, "[KVAF_AIV] DIRECT_TOKEN_ALLOC_FAILED role=%s ret=%d handle=%p\n", role, ret,
                 *token_handle);
    return false;
  }
  *token_id = raw.token_id >> kTokenIdShift;
  hccp::VerboseLog("[KVAF_AIV] DIRECT_TOKEN_READY role=%s token_id=0x%x\n", role,
                   *token_id);
  return true;
}

static bool AlignRange(uint64_t address, uint64_t bytes, uint64_t *aligned_address, uint64_t *aligned_bytes) {
  if (bytes == 0U || aligned_address == nullptr || aligned_bytes == nullptr) {
    return false;
  }
  const uint64_t base = address & ~(kPageBytes - 1U);
  const uint64_t delta = address - base;
  if (bytes > std::numeric_limits<uint64_t>::max() - delta) {
    return false;
  }
  *aligned_address = base;
  *aligned_bytes = bytes + delta;
  return true;
}

static bool RegisterMemory(const DirectHccpApi &api, void *context, void *token_handle, uint32_t token_id,
                    uint32_t token_value, void *address, uint64_t bytes, const char *role, DirectMemory *memory) {
  uint64_t aligned_address = 0U;
  uint64_t aligned_bytes = 0U;
  if (memory == nullptr || address == nullptr || !AlignRange(reinterpret_cast<uint64_t>(address), bytes,
                                                              &aligned_address, &aligned_bytes)) {
    return false;
  }
  hccp::MrRegInfo info{};
  info.in.mem.addr = aligned_address;
  info.in.mem.size = aligned_bytes;
  // V9 uses non_pin=1. Only this isolated B12 reproduction accepts an
  // explicit local-HBM registration A/B; neither the source allocation nor
  // the remote DDR registration is changed.
  bool non_pin = true;
  if (std::strcmp(role, "device_hbm") == 0) {
    const char *setting = std::getenv("KVAF_B12_LOCAL_HBM_NON_PIN");
    if (setting != nullptr) {
      if (std::strcmp(setting, "0") == 0) {
        non_pin = false;
      } else if (std::strcmp(setting, "1") != 0) {
        std::fprintf(stderr,
                     "[KVAF_AIV] BATCH_PUT_LOCAL_HBM_REG_POLICY_INVALID value=%s\n",
                     setting);
        return false;
      }
    }
    std::fprintf(stderr,
                 "[KVAF_AIV] BATCH_PUT_LOCAL_HBM_REG_POLICY non_pin=%u "
                 "source=%s\n",
                 static_cast<unsigned int>(non_pin),
                 setting == nullptr ? "v9_default" : "explicit_b12_ab");
  }
  info.in.ub.flags = hccp::MemRegFlags(non_pin);
  info.in.ub.token_value = token_value;
  info.in.ub.token_id_handle = token_handle;
  const int ret = api.lmem_register(context, &info, &memory->handle);
  if (ret != 0 || memory->handle == nullptr || info.out.key.size == 0U || info.out.key.size > hccp::kMemKeyBytes) {
    std::fprintf(stderr,
                 "[KVAF_AIV] DIRECT_MEMORY_REGISTER_FAILED role=%s ret=%d handle=%p key_bytes=%u addr=0x%llx "
                 "bytes=%llu\n",
                 role, ret, memory->handle, info.out.key.size, static_cast<unsigned long long>(aligned_address),
                 static_cast<unsigned long long>(aligned_bytes));
    return false;
  }
  memory->registered_address = aligned_address;
  memory->registered_bytes = aligned_bytes;
  memory->token_id = token_id;
  memory->token_value = token_value;
  memory->key_bytes = info.out.key.size;
  std::memcpy(memory->key, info.out.key.value, memory->key_bytes);
  hccp::VerboseLog(
      "[KVAF_AIV] DIRECT_LOCAL_MR_TOKEN_CHECK role=%s allocated_token_id=0x%x "
      "registered_token_id=0x%x wqe_token_id=0x%x\n",
      role, token_id, info.out.ub.token_id, memory->token_id);
  hccp::VerboseLog(
      "[KVAF_AIV] DIRECT_MEMORY_REGISTERED role=%s token_id=0x%x host_or_hbm_va=0x%llx bytes=%llu "
      "reg_flags=0x%x non_pin=%u\n",
      role, memory->token_id, static_cast<unsigned long long>(aligned_address),
      static_cast<unsigned long long>(aligned_bytes), info.in.ub.flags,
      static_cast<unsigned int>(non_pin));
  return true;
}

static bool ImportRemoteMemory(const DirectHccpApi &api, void *context, const DirectMemory &remote, void **import_handle) {
  hccp::MrImportInfo info{};
  std::memcpy(info.in.key.value, remote.key, remote.key_bytes);
  info.in.key.size = static_cast<uint8_t>(remote.key_bytes);
  info.in.ub.flags = hccp::MemImportFlags();
  info.in.ub.mapping_addr = 0U;
  info.in.ub.token_value = remote.token_value;
  const int ret = api.rmem_import(context, &info, import_handle);
  if (ret != 0 || *import_handle == nullptr) {
    std::fprintf(stderr, "[KVAF_AIV] DIRECT_REMOTE_DDR_IMPORT_FAILED ret=%d handle=%p\n", ret, *import_handle);
    return false;
  }
  hccp::VerboseLog("[KVAF_AIV] DIRECT_REMOTE_DDR_IMPORTED target_seg=0x%llx\n",
                   static_cast<unsigned long long>(info.out.ub.target_seg_handle));
  return true;
}

static bool CreateCq(const DirectHccpApi &api, void *context, uint32_t mode, uint32_t depth, const char *role,
              hccp::CqInfo *info, void **handle) {
  *info = {};
  info->in.channel_handle = nullptr;
  info->in.depth = depth;
  info->in.ub.user_ctx = 0U;
  info->in.ub.mode = mode;
  info->in.ub.ceqn = 0U;
  info->in.ub.flag = 0U;
  const int ret = api.cq_create(context, info, handle);
  if (ret != 0 || *handle == nullptr) {
    std::fprintf(stderr, "[KVAF_AIV] DIRECT_CQ_CREATE_FAILED role=%s mode=%u depth=%u ret=%d handle=%p\n", role,
                 mode, depth, ret, *handle);
    return false;
  }
  hccp::VerboseLog(
      "[KVAF_AIV] DIRECT_CQ_READY role=%s mode=%u jfc_id=%u cq_va=0x%llx cqe_size=%u db_va=0x%llx\n",
      role, mode, info->out.id, static_cast<unsigned long long>(info->out.buffer_addr),
      info->out.cqe_size, static_cast<unsigned long long>(info->out.swdb_addr));
  return true;
}

bool ReleaseLocalBufferResources(KvafBatchPutTransport *transport, KvafBatchPutLocalBuffer *local) {
  if (transport == nullptr || local == nullptr) return false;
  if (local->memory.handle != nullptr) {
    const int ret = transport->api.lmem_unregister && transport->device_context
        ? transport->api.lmem_unregister(transport->device_context, local->memory.handle) : -1;
    if (ret != 0) {
      std::fprintf(stderr, "[KVAF_AIV] BATCH_PUT_TEARDOWN_RETAINED phase=unregister_local_hbm "
                           "core=all handle=%p ret=%d\n", local->memory.handle, ret);
      return false;
    }
    local->memory.handle = nullptr;
  }
  if (local->token_handle != nullptr) {
    const int ret = transport->api.token_free && transport->device_context
        ? transport->api.token_free(transport->device_context, local->token_handle) : -1;
    if (ret != 0) {
      std::fprintf(stderr, "[KVAF_AIV] BATCH_PUT_TEARDOWN_RETAINED phase=free_local_token "
                           "core=all handle=%p ret=%d\n", local->token_handle, ret);
      return false;
    }
    local->token_handle = nullptr;
  }
  if (local->device_entities != nullptr) {
    const aclError ret = aclrtFree(local->device_entities);
    if (ret != ACL_SUCCESS) {
      std::fprintf(stderr, "[KVAF_AIV] BATCH_PUT_TEARDOWN_RETAINED phase=free_entity_table "
                           "core=all handle=%p ret=%d\n", local->device_entities, static_cast<int>(ret));
      return false;
    }
    local->device_entities = nullptr;
    local->device_channel_entity = 0U;
  }
  return true;
}

bool ReleaseTransport(KvafBatchPutTransport *transport) {
  if (transport == nullptr) return true;
  transport->retirement_started = true;
  // Retire both reciprocal imports and all SQ/CQ objects before their MRs.
  // A failed retirement retains the transport and its caller-owned DDR.
  if (!hccp::RetireLocalAndDeviceQueues(transport->api, transport->device_context,
        transport->workers.data(), transport->worker_count, transport->local_host.get())) {
    std::fprintf(stderr, "[KVAF_AIV] BATCH_PUT_TEARDOWN_RETAINED phase=queues\n");
    return false;
  }
  for (const auto &local : transport->local_buffers) {
    if (local != nullptr && !ReleaseLocalBufferResources(transport, local.get())) return false;
  }
  transport->local_buffers.clear();
  if (transport->device_queue_state != nullptr) {
    const aclError ret = aclrtFree(transport->device_queue_state);
    if (ret != ACL_SUCCESS) {
      std::fprintf(stderr, "[KVAF_AIV] BATCH_PUT_TEARDOWN_RETAINED phase=free_queue_state "
                           "core=all handle=%p ret=%d\n", transport->device_queue_state, static_cast<int>(ret));
      return false;
    }
    transport->device_queue_state = nullptr;
  }

  if (transport->device_peer_memory_import) {
    const int ret = transport->api.rmem_unimport && transport->device_context
        ? transport->api.rmem_unimport(transport->device_context, transport->device_peer_memory_import) : -1;
    if (ret != 0) {
      std::fprintf(stderr, "[KVAF_AIV] BATCH_PUT_TEARDOWN_RETAINED phase=unimport_peer_memory "
                           "core=all handle=%p ret=%d\n", transport->device_peer_memory_import, ret);
      return false;
    }
    transport->device_peer_memory_import = nullptr;
  }
  if (!hccp::ReleaseLocalHostEndpoint(transport->api, transport->local_host.get())) return false;
  hccp::DeviceAivTransportResources resources{
      &transport->api, transport->session, transport->device_context,
      transport->device_transport_token_handle, transport->device_peer_memory_import,
      transport->workers.data(), transport->worker_count};
  const bool retired = hccp::ReleaseDeviceAivTransport(resources);
  transport->session = resources.session;
  transport->device_context = resources.device_context;
  transport->device_transport_token_handle = resources.device_transport_token_handle;
  transport->device_peer_memory_import = resources.device_peer_memory_import;
  if (!retired) return false;
  delete transport;
  return true;
}

int32_t CreateBatchPutTransport(void *host_ddr, uint64_t host_ddr_bytes,
                                uint32_t aiv_core_count, uint32_t jettys_per_core,
                                bool local_host_ddr, void *peer_link,
                                const uint8_t *peer_local_eid,
                                KvafBatchPutTransport **transport) {
  if (host_ddr == nullptr || host_ddr_bytes == 0U ||
      (!local_host_ddr && (peer_link == nullptr || peer_local_eid == nullptr)) ||
      aiv_core_count == 0U || aiv_core_count > KVAF_AIV_BATCH_PUT_MAX_AIV_CORES || jettys_per_core == 0U ||
      jettys_per_core > KVAF_AIV_BATCH_PUT_MAX_JETTYS_PER_CORE || transport == nullptr ||
      static_cast<uint64_t>(aiv_core_count) * jettys_per_core >
          static_cast<uint64_t>(KVAF_AIV_BATCH_PUT_MAX_AIV_CORES) * KVAF_AIV_BATCH_PUT_MAX_JETTYS_PER_CORE) {
    return KVAF_BATCH_PUT_INVALID_ARGUMENT;
  }
  *transport = nullptr;

  auto *state = new (std::nothrow) KvafBatchPutTransport();
  if (state == nullptr) {
    return KVAF_BATCH_PUT_RUNTIME_ERROR;
  }
  state->aiv_core_count = aiv_core_count;
  state->jettys_per_core = jettys_per_core;
  state->worker_count = aiv_core_count * jettys_per_core;
  state->peer_memory_address = reinterpret_cast<uint64_t>(host_ddr);
  state->peer_memory_bytes = host_ddr_bytes;
  state->peer_link = static_cast<hccp::PeerLink *>(peer_link);
  uint32_t physical_device_id = 0U;
  std::array<uint8_t, hccp::kEidBytes> device_eid{};
  std::array<uint8_t, hccp::kEidBytes> peer_eid{};
  int32_t result = KVAF_BATCH_PUT_DIRECT_HCCP_ERROR;
  uint32_t device_transport_token_id = 0U;
  const uint64_t queue_bytes =
      static_cast<uint64_t>(state->worker_count) * sizeof(KvafBatchPutQueueState);

  if (!hccp::GetCurrentNpuPhysicalDeviceId(&physical_device_id)) {
    std::fprintf(stderr, "[KVAF_AIV] unable to resolve physical device id from selected ACL device\n");
    result = KVAF_BATCH_PUT_EID_ERROR;
    goto failed;
  }
  state->physical_device_id = physical_device_id;
  if (!hccp::LoadDirectHccp(&state->api) ||
      !hccp::AcquireDirectHccpSession(physical_device_id, kPidHdcType, &state->session)) goto failed;
  if (local_host_ddr) {
    state->local_host = std::make_unique<hccp::LocalHostEndpoint>();
    if (!hccp::CreateLocalHostEndpoint(state->api, physical_device_id,
                                      host_ddr, host_ddr_bytes, state->local_host.get())) goto failed;
    device_eid = state->local_host->device_eid;
    peer_eid = state->local_host->host_eid;
    state->peer_memory = state->local_host->memory;
  } else {
    std::memcpy(device_eid.data(), peer_local_eid, device_eid.size());
    if (!hccp::LoadPeerMemory(state->peer_link, state->peer_memory_address,
                              &state->peer_memory, &peer_eid)) goto failed;
  }
  state->wqe_remote_eid = ReverseEidForAivWqe(peer_eid);
  if (
      !InitContext(state->api, hccp::kOfflineHdc, physical_device_id, device_eid, "device_aiv",
                   &state->device_context) ||
      !AllocateToken(state->api, state->device_context, "device_transport",
                     &state->device_transport_token_handle, &device_transport_token_id)) {
    goto failed;
  }
  state->device_transport_token_value = state->peer_memory.token_value;
  if (!ImportRemoteMemory(state->api, state->device_context, state->peer_memory,
                          &state->device_peer_memory_import)) {
    goto failed;
  }
  for (uint32_t entity_index = 0U; entity_index < state->worker_count; ++entity_index) {
    const uint32_t core = entity_index / jettys_per_core;
    const uint32_t jetty_slot = entity_index % jettys_per_core;
    DirectAivWorker &worker = state->workers[entity_index];
    const std::string suffix =
        "core=" + std::to_string(core) + "_jetty=" + std::to_string(jetty_slot);
    const std::string device_cq_role = "device_user_ctl_" + suffix;
    const std::string device_jetty_role = "device_aiv_dev_used_" + suffix;
    if (!CreateCq(state->api, state->device_context, hccp::kJfcUserCtlNormal, kDeviceUserCtlCqDepth,
                  device_cq_role.c_str(), &worker.device_cq_info, &worker.device_user_ctl_cq) ||
        !hccp::CreateDeviceAivJetty(state->api, state->device_context, worker.device_user_ctl_cq,
                           state->device_transport_token_handle, state->device_transport_token_value,
                           kDeviceUserCtlSqDepth, device_jetty_role.c_str(),
                           &worker.device_user_ctl_jetty.info, &worker.device_user_ctl_jetty.handle)) {
      goto failed;
    }

    const std::string device_ctp_role = "device_aiv_peer_link_" + suffix;
    hccp::PeerChannel channel{};
    const bool channel_ready = local_host_ddr
        ? hccp::CreateLocalHostChannel(state->api, physical_device_id, state->device_context,
              state->device_transport_token_value, worker.device_user_ctl_jetty.info,
              entity_index, state->local_host.get(), &channel)
        : hccp::CreatePeerChannel(
            {state->peer_link, &state->api, state->device_context,
             physical_device_id, state->peer_memory_address, &state->peer_memory,
             &device_eid, &peer_eid, &worker.device_user_ctl_jetty.info,
             device_ctp_role.c_str()},
            &channel);
    worker.device_peer_import_info = channel.local_import_info;
    worker.device_peer_target_jetty = channel.local_target_jetty;
    if (!channel_ready) goto failed;
  }
  for (uint32_t entity_index = 0U; entity_index < state->worker_count; ++entity_index) {
    const DirectAivWorker &worker = state->workers[entity_index];
    KvafBatchPutDirectUrmaEntity &entity = state->entity_templates[entity_index];
    entity.magic = kDirectEntityMagic;
    entity.version = kDirectEntityVersion;
    entity.remote_host_ddr_address = state->peer_memory_address;
    entity.remote_host_ddr_size = state->peer_memory_bytes;
    entity.remote_token_id = state->peer_memory.token_id;
    entity.remote_token_value = state->peer_memory.token_value;
    entity.sq_va = worker.device_user_ctl_jetty.info.ub.sq_buffer_va;
    entity.sq_db_va = worker.device_user_ctl_jetty.info.ub.db_addr;
    entity.jetty_id = worker.device_user_ctl_jetty.info.ub.id;
    entity.sq_depth = kDeviceUserCtlSqDepth * kWqebbPerWqe;
    entity.tp_id = worker.device_peer_import_info.out.ub.tpn;
    entity.wqe_size = kWqeBytes;
    std::memcpy(entity.remote_eid, state->wqe_remote_eid.data(), sizeof(entity.remote_eid));
    entity.cq_va = worker.device_cq_info.out.buffer_addr;
    entity.cq_db_va = worker.device_cq_info.out.swdb_addr;
    entity.jfc_id = worker.device_cq_info.out.id;
    entity.cq_depth = kDeviceUserCtlCqDepth;
    entity.cqe_size = worker.device_cq_info.out.cqe_size;
    if (entity.sq_va == 0U || entity.sq_db_va == 0U || entity.cq_va == 0U ||
        entity.cq_db_va == 0U || entity.cqe_size < 32U || entity.tp_id == 0U) {
      goto failed;
    }
    for (uint32_t previous = 0U; previous < entity_index; ++previous) {
      const KvafBatchPutDirectUrmaEntity &other = state->entity_templates[previous];
      if (entity.sq_va == other.sq_va || entity.sq_db_va == other.sq_db_va ||
          entity.cq_va == other.cq_va || entity.cq_db_va == other.cq_db_va ||
          entity.jetty_id == other.jetty_id) {
        goto failed;
      }
    }
  }
  if (aclrtMalloc(&state->device_queue_state, queue_bytes, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS ||
      state->device_queue_state == nullptr ||
      aclrtMemset(state->device_queue_state, queue_bytes, 0, queue_bytes) != ACL_SUCCESS) {
    std::fprintf(stderr,
                 "[KVAF_AIV] DIRECT_QUEUE_STATE_ALLOC_FAILED cores=%u jettys_per_core=%u\n",
                 aiv_core_count, jettys_per_core);
    goto failed;
  }

  *transport = state;
  return KVAF_BATCH_PUT_SUCCESS;

failed:
  // A partially registered endpoint may fail to retire. Return its ownership
  // on error so AccessOperator/Runtime keep DDR alive until shutdown succeeds.
  if (!ReleaseTransport(state)) *transport = state;
  return result;
}

extern "C" int32_t KvafBatchPutTransportCreateLocalHost(
    void *host_ddr, uint64_t host_ddr_bytes, uint32_t aiv_core_count,
    uint32_t jettys_per_core, KvafBatchPutTransport **transport) {
  return CreateBatchPutTransport(host_ddr, host_ddr_bytes, aiv_core_count,
                                   jettys_per_core, true, nullptr, nullptr, transport);
}

extern "C" int32_t KvafBatchPutTransportDestroyChecked(KvafBatchPutTransport *transport) {
  return ReleaseTransport(transport) ? KVAF_BATCH_PUT_SUCCESS : KVAF_BATCH_PUT_RUNTIME_ERROR;
}

extern "C" int32_t KvafBatchPutTransportCreateWithPeerLink(
    void *host_ddr, uint64_t host_ddr_bytes, uint32_t aiv_core_count,
    uint32_t jettys_per_core, void *peer_link, const uint8_t *peer_local_eid,
    KvafBatchPutTransport **transport) {
  if (peer_link == nullptr || peer_local_eid == nullptr) return KVAF_BATCH_PUT_INVALID_ARGUMENT;
  return CreateBatchPutTransport(host_ddr, host_ddr_bytes, aiv_core_count,
                                 jettys_per_core, false, peer_link, peer_local_eid, transport);
}

extern "C" void KvafBatchPutTransportDestroy(KvafBatchPutTransport *transport) {
  ReleaseTransport(transport);
}

extern "C" int32_t KvafBatchPutLocalBufferAcquire(KvafBatchPutTransport *transport,
                                                    void *input_device, uint64_t input_device_bytes,
                                                    KvafBatchPutLocalBuffer **local_buffer) {
  if (transport == nullptr || input_device == nullptr || input_device_bytes == 0U || local_buffer == nullptr ||
      transport->retirement_started) {
    return KVAF_BATCH_PUT_INVALID_ARGUMENT;
  }
  *local_buffer = nullptr;
  std::lock_guard<std::mutex> lock(transport->mutex);
  const uint64_t input_address = reinterpret_cast<uint64_t>(input_device);
  for (const auto &existing : transport->local_buffers) {
    if (existing->input_hbm_address == input_address &&
        existing->input_hbm_bytes == input_device_bytes) {
      ++existing->reference_count;
      *local_buffer = existing.get();
      hccp::VerboseLog(
          "[KVAF_AIV] DIRECT_LOCAL_BUFFER_CACHE_HIT input=%p bytes=%llu refs=%u\n",
          input_device, static_cast<unsigned long long>(input_device_bytes),
          existing->reference_count);
      return KVAF_BATCH_PUT_SUCCESS;
    }
  }

  auto candidate = std::make_unique<KvafBatchPutLocalBuffer>();
  candidate->transport = transport;
  candidate->input_hbm_address = input_address;
  candidate->input_hbm_bytes = input_device_bytes;
  const char *token_source = std::getenv("KVAF_B12_LOCAL_HBM_TOKEN_SOURCE");
  const bool use_process_token = token_source != nullptr &&
      std::strcmp(token_source, "process") == 0;
  if (token_source != nullptr && !use_process_token &&
      std::strcmp(token_source, "ra") != 0) {
    std::fprintf(stderr,
                 "[KVAF_AIV] BATCH_PUT_LOCAL_TOKEN_SOURCE_INVALID value=%s\n",
                 token_source);
    return KVAF_BATCH_PUT_DIRECT_HCCP_ERROR;
  }
  uint32_t process_token_id = 0U;
  if (use_process_token &&
      !QueryAclProcessToken(input_device, input_device_bytes, &process_token_id)) {
    return KVAF_BATCH_PUT_DIRECT_HCCP_ERROR;
  }
  uint32_t token_id = 0U;
  if (!AllocateToken(transport->api, transport->device_context, "device_local_hbm",
                     &candidate->token_handle, &token_id) ||
      !RegisterMemory(transport->api, transport->device_context, candidate->token_handle,
                      token_id, transport->device_transport_token_value, input_device,
                      input_device_bytes, "device_hbm", &candidate->memory)) {
    if (!ReleaseLocalBufferResources(transport, candidate.get())) {
      transport->retirement_started = true;
      transport->local_buffers.push_back(std::move(candidate));
    }
    return KVAF_BATCH_PUT_DIRECT_HCCP_ERROR;
  }

  auto entities = transport->entity_templates;
  for (uint32_t entity_index = 0U; entity_index < transport->worker_count; ++entity_index) {
    entities[entity_index].local_hbm_address = input_address;
    entities[entity_index].local_hbm_size = input_device_bytes;
    entities[entity_index].local_token_id =
        use_process_token ? process_token_id : candidate->memory.token_id;
    entities[entity_index].local_token_value = candidate->memory.token_value;
  }
  std::fprintf(stderr,
               "[KVAF_AIV] BATCH_PUT_LOCAL_TOKEN_SOURCE source=%s "
               "wqe_token_id=0x%x ra_token_id=0x%x\n",
               use_process_token ? "process" : "ra_v9_default",
               use_process_token ? process_token_id : candidate->memory.token_id,
               candidate->memory.token_id);
  const uint64_t entity_bytes =
      static_cast<uint64_t>(transport->worker_count) * sizeof(entities.front());
  if (aclrtMalloc(&candidate->device_entities, entity_bytes, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS ||
      candidate->device_entities == nullptr ||
      aclrtMemcpy(candidate->device_entities, entity_bytes, entities.data(), entity_bytes,
                  ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
    if (!ReleaseLocalBufferResources(transport, candidate.get())) {
      transport->retirement_started = true;
      transport->local_buffers.push_back(std::move(candidate));
    }
    return KVAF_BATCH_PUT_RUNTIME_ERROR;
  }
  candidate->device_channel_entity = reinterpret_cast<uint64_t>(candidate->device_entities);
  candidate->reference_count = 1U;
  *local_buffer = candidate.get();
  transport->local_buffers.push_back(std::move(candidate));
  hccp::VerboseLog(
      "[KVAF_AIV] DIRECT_LOCAL_BUFFER_REGISTERED input=%p bytes=%llu cached_mrs=%zu\n",
      input_device, static_cast<unsigned long long>(input_device_bytes),
      transport->local_buffers.size());
  return KVAF_BATCH_PUT_SUCCESS;
}

extern "C" void KvafBatchPutLocalBufferRelease(KvafBatchPutLocalBuffer *local_buffer) {
  if (local_buffer == nullptr) {
    return;
  }
  KvafBatchPutTransport *transport = local_buffer->transport;
  if (transport == nullptr) return;
  std::lock_guard<std::mutex> lock(transport->mutex);
  if (local_buffer->reference_count == 0U) {
    std::fprintf(stderr, "[KVAF_AIV] DIRECT_LOCAL_BUFFER_RELEASE_WARNING reason=reference_underflow\n");
    return;
  }
  --local_buffer->reference_count;
}

static void LogBatchPutCompletionDiagnostics(const KvafBatchPutCoreStatus *statuses,
                                            uint32_t core_count,
                                            uint32_t jettys_per_core) {
  for (uint32_t core = 0U; core < core_count; ++core) {
    const auto &status = statuses[core];
    if (status.reserved[KVAF_BATCH_PUT_CQ_DIAGNOSTIC_MAGIC_WORD] !=
        KVAF_BATCH_PUT_CQ_DIAGNOSTIC_MAGIC) {
      continue;
    }
    bool report = false;
    for (uint32_t jetty = 0U; jetty < jettys_per_core; ++jetty) {
      const uint64_t *const record = status.reserved +
          KVAF_BATCH_PUT_CQ_DIAGNOSTIC_BASE_WORD +
          jetty * KVAF_BATCH_PUT_CQ_DIAGNOSTIC_WORDS_PER_JETTY;
      report = report || record[KVAF_BATCH_PUT_CQ_DIAG_FAST_BUDGET_EXCEEDED] != 0U ||
               record[KVAF_BATCH_PUT_CQ_DIAG_RESULT] != KVAF_BATCH_PUT_CORE_SUCCESS;
    }
    if (!report) continue;
    // Print both queues of the affected core: the old aggregate SQ/CQ
    // fields refer only to jetty0 even when jetty1 is the one that failed.
    for (uint32_t jetty = 0U; jetty < jettys_per_core; ++jetty) {
      const uint64_t *const record = status.reserved +
          KVAF_BATCH_PUT_CQ_DIAGNOSTIC_BASE_WORD +
          jetty * KVAF_BATCH_PUT_CQ_DIAGNOSTIC_WORDS_PER_JETTY;
      if (record[KVAF_BATCH_PUT_CQ_DIAG_ASSIGNED_ROWS] == 0U) continue;
      const uint64_t reason = record[KVAF_BATCH_PUT_CQ_DIAG_TIMEOUT_REASON];
      const char *const reason_name =
          reason == KVAF_BATCH_PUT_CQ_TIMEOUT_DEVICE_DEADLINE ? "device_deadline" :
          reason == KVAF_BATCH_PUT_CQ_TIMEOUT_POLL_FALLBACK ? "poll_fallback" : "none";
      const uint64_t result = record[KVAF_BATCH_PUT_CQ_DIAG_RESULT];
      const bool late_success = result == KVAF_BATCH_PUT_CORE_SUCCESS &&
          record[KVAF_BATCH_PUT_CQ_DIAG_FAST_BUDGET_EXCEEDED] != 0U;
      std::fprintf(stderr,
          "[KVAF_AIV] BATCH_PUT_CQ_WAIT_DETAILS core=%u jetty=%u result=%llu "
          "rows=%llu polls=%llu elapsed_ticks=%llu slow_elapsed_ticks=%llu "
          "fast_budget_exceeded=%llu late_success=%u timeout_reason=%s "
          "raw_cq_word=0x%llx observed_owner=%llu expected_owner=%llu cqe_valid=%u "
          "cq_status=%llu cq_sub_status=%llu sq_before=%llu cq_before=%llu "
          "sq_after=%llu cq_after=%llu fast_poll_budget=20000 "
          "slow_deadline_ticks=100000000 slow_poll_limit=1000000 clock_hz=1000000000 "
          "timing_scope=eager_put_cq_not_Get_performance\n",
          core, jetty, static_cast<unsigned long long>(result),
          static_cast<unsigned long long>(record[KVAF_BATCH_PUT_CQ_DIAG_ASSIGNED_ROWS]),
          static_cast<unsigned long long>(record[KVAF_BATCH_PUT_CQ_DIAG_POLLS]),
          static_cast<unsigned long long>(record[KVAF_BATCH_PUT_CQ_DIAG_ELAPSED_TICKS]),
          static_cast<unsigned long long>(record[KVAF_BATCH_PUT_CQ_DIAG_SLOW_ELAPSED_TICKS]),
          static_cast<unsigned long long>(record[KVAF_BATCH_PUT_CQ_DIAG_FAST_BUDGET_EXCEEDED]),
          static_cast<unsigned int>(late_success), reason_name,
          static_cast<unsigned long long>(record[KVAF_BATCH_PUT_CQ_DIAG_RAW_WORD]),
          static_cast<unsigned long long>(record[KVAF_BATCH_PUT_CQ_DIAG_OBSERVED_OWNER]),
          static_cast<unsigned long long>(record[KVAF_BATCH_PUT_CQ_DIAG_EXPECTED_OWNER]),
          static_cast<unsigned int>(record[KVAF_BATCH_PUT_CQ_DIAG_OBSERVED_OWNER] !=
                                    record[KVAF_BATCH_PUT_CQ_DIAG_EXPECTED_OWNER]),
          static_cast<unsigned long long>((record[KVAF_BATCH_PUT_CQ_DIAG_RAW_WORD] >> 24U) & 0xffU),
          static_cast<unsigned long long>((record[KVAF_BATCH_PUT_CQ_DIAG_RAW_WORD] >> 16U) & 0xffU),
          static_cast<unsigned long long>(record[KVAF_BATCH_PUT_CQ_DIAG_SQ_BEFORE]),
          static_cast<unsigned long long>(record[KVAF_BATCH_PUT_CQ_DIAG_CQ_BEFORE]),
          static_cast<unsigned long long>(record[KVAF_BATCH_PUT_CQ_DIAG_SQ_AFTER]),
          static_cast<unsigned long long>(record[KVAF_BATCH_PUT_CQ_DIAG_CQ_AFTER]));
    }
  }
}

extern "C" int32_t KvafExecuteBatchPutSync(KvafBatchPutLocalBuffer *local_buffer,
                                            const KvafExecuteBatchPutKernelArgs *arguments,
                                            aclrtStream stream,
                                            KvafBatchPutCoreStatus *core_status,
                                            uint32_t core_status_count) {
  hccp::VerboseLog("[KVAF_AIV] BATCH_PUT_PHASE phase=sync_enter local=%p args=%p stream=%p\n",
                   static_cast<void *>(local_buffer), static_cast<const void *>(arguments),
                   static_cast<void *>(stream));
  if (local_buffer == nullptr || local_buffer->reference_count == 0U ||
      local_buffer->transport == nullptr || arguments == nullptr || stream == nullptr || core_status == nullptr ||
      arguments->input == 0U || arguments->input_offsets == 0U || arguments->lengths == 0U ||
      arguments->destination_addresses == 0U || arguments->core_status == 0U ||
      arguments->input_bytes == 0U || arguments->row_count == 0U ||
      arguments->aiv_core_count == 0U || arguments->aiv_core_count != core_status_count ||
      arguments->aiv_core_count != local_buffer->transport->aiv_core_count ||
      arguments->input != local_buffer->input_hbm_address ||
      arguments->input_bytes > local_buffer->input_hbm_bytes) {
    return KVAF_BATCH_PUT_INVALID_ARGUMENT;
  }

  KvafBatchPutTransport *transport = local_buffer->transport;
  std::lock_guard<std::mutex> launch_lock(transport->launch_mutex);
  hccp::VerboseLog("[KVAF_AIV] BATCH_PUT_PHASE phase=clear_status_begin\n");
  const uint64_t status_bytes = static_cast<uint64_t>(core_status_count) * sizeof(*core_status);
  const aclError clear_result =
      aclrtMemset(reinterpret_cast<void *>(arguments->core_status), status_bytes, 0, status_bytes);
  if (clear_result != ACL_SUCCESS) {
    std::fprintf(stderr, "[KVAF_AIV] DIRECT_BATCH_PUT_RUNTIME_ERROR phase=clear_status acl=%d\n",
                 static_cast<int>(clear_result));
    return KVAF_BATCH_PUT_RUNTIME_ERROR;
  }
  hccp::VerboseLog("[KVAF_AIV] BATCH_PUT_PHASE phase=launch_begin\n");
  KvafExecuteBatchPutLaunchArgs launch_args{};
  launch_args.request = *arguments;
  launch_args.channel_entity = local_buffer->device_channel_entity;
  launch_args.queue_state = reinterpret_cast<uint64_t>(transport->device_queue_state);
  launch_args.jettys_per_core = transport->jettys_per_core;
  int32_t result = KvafExecuteBatchPutLaunchInternal(&launch_args, stream);
  if (result != KVAF_BATCH_PUT_SUCCESS) {
    std::fprintf(stderr, "[KVAF_AIV] DIRECT_BATCH_PUT_RUNTIME_ERROR phase=launch result=%d\n", result);
    return KVAF_BATCH_PUT_RUNTIME_ERROR;
  }
  hccp::VerboseLog("[KVAF_AIV] BATCH_PUT_PHASE phase=synchronize_begin\n");
  const aclError synchronize_result = aclrtSynchronizeStream(stream);
  if (synchronize_result != ACL_SUCCESS) {
    std::fprintf(stderr, "[KVAF_AIV] DIRECT_BATCH_PUT_RUNTIME_ERROR phase=synchronize acl=%d\n",
                 static_cast<int>(synchronize_result));
    return KVAF_BATCH_PUT_RUNTIME_ERROR;
  }
  hccp::VerboseLog("[KVAF_AIV] BATCH_PUT_PHASE phase=copy_status_begin\n");
  const aclError status_copy_result =
      aclrtMemcpy(core_status, status_bytes, reinterpret_cast<void *>(arguments->core_status), status_bytes,
                  ACL_MEMCPY_DEVICE_TO_HOST);
  if (status_copy_result != ACL_SUCCESS) {
    std::fprintf(stderr, "[KVAF_AIV] DIRECT_BATCH_PUT_RUNTIME_ERROR phase=copy_status acl=%d\n",
                 static_cast<int>(status_copy_result));
    return KVAF_BATCH_PUT_RUNTIME_ERROR;
  }
  LogBatchPutCompletionDiagnostics(core_status, core_status_count,
                                    transport->jettys_per_core);
  hccp::VerboseLog("[KVAF_AIV] BATCH_PUT_PHASE phase=status_validate_begin\n");
  for (uint32_t core = 0U; core < core_status_count; ++core) {
    const uint64_t begin = static_cast<uint64_t>(arguments->row_count) * core / core_status_count;
    const uint64_t end = static_cast<uint64_t>(arguments->row_count) * (core + 1U) / core_status_count;
    if (core_status[core].magic != KVAF_AIV_BATCH_PUT_STATUS_MAGIC ||
        core_status[core].result != KVAF_BATCH_PUT_CORE_SUCCESS ||
        core_status[core].processed_request_count != end - begin ||
        core_status[core].failed_request_index != UINT64_MAX) {
      // The SQ address comes from HCCP QP creation, not aclrtMalloc. This
      // path has no verified ACL copy contract for that address. Keep the
      // already-copied status instead of risking a second fault while trying
      // to inspect an SQE on the failure path.
      std::fprintf(stderr,
                   "[KVAF_AIV] BATCH_PUT_VALIDATION_ERROR core=%u magic=0x%llx result=%llu "
                   "processed=%llu expected=%llu failed_request=%llu "
                   "sq_pi=%llu cq_ci=%llu cq_status=%llu cq_sub_status=%llu "
                   "last_wqe=0x%llx mr_addr=0x%llx mr_bytes=%llu "
                   "mr_token_id=0x%x\n",
                   core, static_cast<unsigned long long>(core_status[core].magic),
                   static_cast<unsigned long long>(core_status[core].result),
                   static_cast<unsigned long long>(core_status[core].processed_request_count),
                   static_cast<unsigned long long>(end - begin),
                   static_cast<unsigned long long>(core_status[core].failed_request_index),
                   static_cast<unsigned long long>(core_status[core].sq_producer_index),
                   static_cast<unsigned long long>(core_status[core].cq_consumer_index),
                   static_cast<unsigned long long>(core_status[core].cq_status),
                   static_cast<unsigned long long>(core_status[core].cq_sub_status),
                   static_cast<unsigned long long>(core_status[core].last_wqe_protocol),
                   static_cast<unsigned long long>(local_buffer->memory.registered_address),
                   static_cast<unsigned long long>(local_buffer->memory.registered_bytes),
                   local_buffer->memory.token_id);
      if (arguments->row_count == 1U && core_status_count == 1U &&
          transport->jettys_per_core == 1U) {
        const uint64_t local_address = core_status[core].reserved[0];
        const uint64_t length = core_status[core].reserved[1];
        const auto &mr = local_buffer->memory;
        const bool address_in_mr = local_address >= mr.registered_address &&
            local_address - mr.registered_address <= mr.registered_bytes &&
            length <= mr.registered_bytes -
                          (local_address - mr.registered_address);
        std::fprintf(stderr,
                     "[KVAF_AIV] BATCH_PUT_AIV_SGE_INPUT local_addr=0x%llx "
                     "length=%llu local_token_id=0x%llx local_in_mr=%u "
                     "remote_addr=0x%llx remote_token_id=0x%llx "
                     "remote_token_value=0x%llx tp_id=0x%llx "
                     "entity_local_size=%llu\n",
                     static_cast<unsigned long long>(local_address),
                     static_cast<unsigned long long>(length),
                     static_cast<unsigned long long>(core_status[core].reserved[2]),
                     static_cast<unsigned int>(address_in_mr),
                     static_cast<unsigned long long>(core_status[core].reserved[3]),
                     static_cast<unsigned long long>(core_status[core].reserved[4]),
                     static_cast<unsigned long long>(core_status[core].reserved[5]),
                     static_cast<unsigned long long>(core_status[core].reserved[6]),
                     static_cast<unsigned long long>(core_status[core].reserved[7]));
      }
      return KVAF_BATCH_PUT_KERNEL_ERROR;
    }
  }
  return KVAF_BATCH_PUT_SUCCESS;
}
