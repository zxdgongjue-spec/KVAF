#include "kvaf_aiv_batch_get.h"

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

#include "acl/acl.h"
#include "direct_hccp/direct_hccp_abi.h"
#include "direct_hccp/direct_hccp_remote_transport.h"
#include "direct_hccp/direct_hccp_local_host.h"
#include "direct_hccp/direct_hccp_session.h"

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
    KVAF_AIV_BATCH_GET_MAX_AIV_CORES *
    KVAF_AIV_BATCH_GET_MAX_JETTYS_PER_CORE;

bool LocalBufferVerboseLoggingEnabled() {
  const char *value = std::getenv("KVAF_AIV_VERBOSE_LOCAL_BUFFER");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

using DirectHccpApi = hccp::DirectHccpApi;
using DirectMemory = hccp::DirectMemory;

struct KvafBatchGetTransport;

// A local buffer has stable heap storage because it is an opaque handle held
// by the caller. Zero-reference entries stay cached until transport teardown.
struct KvafBatchGetLocalBuffer {
  KvafBatchGetTransport *transport = nullptr;
  void *token_handle = nullptr;
  DirectMemory memory{};
  uint64_t output_hbm_address = 0U;
  uint64_t output_hbm_bytes = 0U;
  uint32_t reference_count = 0U;
  void *device_entities = nullptr;
  uint64_t device_channel_entity = 0U;
};

// A USER_CTL SQ/CQ is strictly single-producer/single-consumer from the AIV
// kernel's point of view. A multi-AIV launch therefore owns one local AIV
// Jetty per core rather than sharing producer/consumer indices.
using DirectAivWorker = hccp::DeviceAivWorkerResources;

struct KvafBatchGetTransport {
  DirectHccpApi api{};
  hccp::DirectHccpSession *session = nullptr;
  uint32_t physical_device_id = 0;
  void *device_context = nullptr;
  // Queue/JETTY protection is transport-scoped.  It must not borrow a token
  // from one particular ExecuteBatchGet output allocation.
  void *device_transport_token_handle = nullptr;
  uint32_t device_transport_token_value = 0U;
  DirectMemory peer_memory{};
  void *device_peer_memory_import = nullptr;
  // Local HBM registrations are keyed by the exact caller-visible (addr,
  // bytes) pair. Heap ownership keeps opaque LocalBuffer handles stable.
  std::vector<std::unique_ptr<KvafBatchGetLocalBuffer>> local_buffers{};
  // Direct-HCCP writes private ABI outputs directly into the worker records.
  // Keep the bounded records inline with the registration, rather than in a
  // separately allocated std::vector buffer.  The old vector's allocator
  // metadata was the first heap object destroyed after the HCCP release path
  // and was where glibc reported "invalid size" on A5 teardown.
  std::array<DirectAivWorker, kMaxDirectAivWorkers> workers{};
  // Template entity records contain only transport fields. Acquire copies and
  // completes them with the selected caller-owned HBM buffer fields.
  std::array<KvafBatchGetDirectUrmaEntity, kMaxDirectAivWorkers> entity_templates{};
  uint32_t worker_count{0};
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
  std::fprintf(stderr,
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
  std::fprintf(stderr, "[KVAF_AIV] DIRECT_HCCP_CONTEXT_READY role=%s mode=%d physical=%u context=%p\n",
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
  std::fprintf(stderr, "[KVAF_AIV] DIRECT_TOKEN_READY role=%s token_id=0x%x\n", role, *token_id);
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
  info.in.ub.flags = hccp::MemRegFlags(true);
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
  std::fprintf(stderr,
               "[KVAF_AIV] DIRECT_MEMORY_REGISTERED role=%s token_id=0x%x host_or_hbm_va=0x%llx bytes=%llu\n",
               role, memory->token_id, static_cast<unsigned long long>(aligned_address),
               static_cast<unsigned long long>(aligned_bytes));
  return true;
}

bool ReleaseLocalBufferResources(KvafBatchGetTransport *transport, KvafBatchGetLocalBuffer *local) {
  if (transport == nullptr || local == nullptr) return false;
  if (local->memory.handle != nullptr) {
    const int ret = transport->api.lmem_unregister && transport->device_context
        ? transport->api.lmem_unregister(transport->device_context, local->memory.handle) : -1;
    if (ret != 0) {
      std::fprintf(stderr, "[KVAF_AIV] BATCH_GET_TEARDOWN_RETAINED phase=unregister_local_hbm "
                           "core=all handle=%p ret=%d\n", local->memory.handle, ret);
      return false;
    }
    local->memory.handle = nullptr;
  }
  if (local->token_handle != nullptr) {
    const int ret = transport->api.token_free && transport->device_context
        ? transport->api.token_free(transport->device_context, local->token_handle) : -1;
    if (ret != 0) {
      std::fprintf(stderr, "[KVAF_AIV] BATCH_GET_TEARDOWN_RETAINED phase=free_local_token "
                           "core=all handle=%p ret=%d\n", local->token_handle, ret);
      return false;
    }
    local->token_handle = nullptr;
  }
  if (local->device_entities != nullptr) {
    const aclError ret = aclrtFree(local->device_entities);
    if (ret != ACL_SUCCESS) {
      std::fprintf(stderr, "[KVAF_AIV] BATCH_GET_TEARDOWN_RETAINED phase=free_entity_table "
                           "core=all handle=%p ret=%d\n", local->device_entities, static_cast<int>(ret));
      return false;
    }
    local->device_entities = nullptr;
    local->device_channel_entity = 0U;
  }
  return true;
}

bool AcquireLocalBuffer(KvafBatchGetTransport *transport, void *output_hbm, uint64_t output_hbm_bytes,
                        KvafBatchGetLocalBuffer **local_buffer, uint64_t *device_channel_entity) {
  if (transport == nullptr || output_hbm == nullptr || output_hbm_bytes == 0U || local_buffer == nullptr ||
      device_channel_entity == nullptr || transport->device_context == nullptr || transport->worker_count == 0U ||
      transport->device_transport_token_value == 0U || transport->retirement_started) {
    return false;
  }
  const uint64_t output_address = reinterpret_cast<uint64_t>(output_hbm);
  for (const auto &existing : transport->local_buffers) {
    if (existing != nullptr && existing->output_hbm_address == output_address &&
        existing->output_hbm_bytes == output_hbm_bytes) {
      ++existing->reference_count;
      *local_buffer = existing.get();
      *device_channel_entity = existing->device_channel_entity;
      if (LocalBufferVerboseLoggingEnabled()) {
        std::fprintf(stderr,
                     "[KVAF_AIV] DIRECT_LOCAL_BUFFER_ACQUIRE_CACHE_HIT output_hbm=0x%llx bytes=%llu token_id=0x%x "
                     "refs=%u entity_table=0x%llx\n",
                     static_cast<unsigned long long>(output_address),
                     static_cast<unsigned long long>(output_hbm_bytes),
                     existing->memory.token_id, existing->reference_count,
                     static_cast<unsigned long long>(existing->device_channel_entity));
      }
      return true;
    }
  }

  auto candidate = std::make_unique<KvafBatchGetLocalBuffer>();
  candidate->transport = transport;
  candidate->output_hbm_address = output_address;
  candidate->output_hbm_bytes = output_hbm_bytes;
  uint32_t token_id = 0U;
  if (!AllocateToken(transport->api, transport->device_context, "device_local_hbm", &candidate->token_handle,
                     &token_id) ||
      !RegisterMemory(transport->api, transport->device_context, candidate->token_handle, token_id,
                      transport->device_transport_token_value, output_hbm, output_hbm_bytes, "device_hbm",
                      &candidate->memory)) {
    if (!ReleaseLocalBufferResources(transport, candidate.get())) {
      transport->retirement_started = true;
      transport->local_buffers.push_back(std::move(candidate));
    }
    return false;
  }

  std::array<KvafBatchGetDirectUrmaEntity, kMaxDirectAivWorkers> entities = transport->entity_templates;
  for (uint32_t index = 0U; index < transport->worker_count; ++index) {
    entities[index].local_hbm_address = output_address;
    entities[index].local_hbm_size = output_hbm_bytes;
    entities[index].local_token_id = candidate->memory.token_id;
    entities[index].local_token_value = candidate->memory.token_value;
  }
  const uint64_t entity_bytes = static_cast<uint64_t>(transport->worker_count) * sizeof(entities.front());
  if (entity_bytes > UINT32_MAX ||
      aclrtMalloc(&candidate->device_entities, entity_bytes, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS ||
      candidate->device_entities == nullptr ||
      aclrtMemcpy(candidate->device_entities, entity_bytes, entities.data(), entity_bytes,
                  ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
    std::fprintf(stderr,
                 "[KVAF_AIV] DIRECT_LOCAL_BUFFER_ENTITY_COPY_FAILED output_hbm=0x%llx bytes=%llu workers=%u\n",
                 static_cast<unsigned long long>(output_address), static_cast<unsigned long long>(output_hbm_bytes),
                 transport->worker_count);
    if (!ReleaseLocalBufferResources(transport, candidate.get())) {
      transport->retirement_started = true;
      transport->local_buffers.push_back(std::move(candidate));
    }
    return false;
  }
  candidate->device_channel_entity = reinterpret_cast<uint64_t>(candidate->device_entities);
  candidate->reference_count = 1U;
  *local_buffer = candidate.get();
  *device_channel_entity = candidate->device_channel_entity;
  std::fprintf(stderr,
               "[KVAF_AIV] DIRECT_LOCAL_BUFFER_ACQUIRE_NEW output_hbm=0x%llx bytes=%llu token_id=0x%x refs=1 "
               "entity_table=0x%llx transport_reused=1\n",
               static_cast<unsigned long long>(output_address), static_cast<unsigned long long>(output_hbm_bytes),
               candidate->memory.token_id, static_cast<unsigned long long>(candidate->device_channel_entity));
  transport->local_buffers.push_back(std::move(candidate));
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
  std::fprintf(stderr, "[KVAF_AIV] DIRECT_REMOTE_DDR_IMPORTED target_seg=0x%llx\n",
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
  std::fprintf(stderr,
               "[KVAF_AIV] DIRECT_CQ_READY role=%s mode=%u jfc_id=%u cq_va=0x%llx cqe_size=%u db_va=0x%llx\n",
               role, mode, info->out.id, static_cast<unsigned long long>(info->out.buffer_addr),
               info->out.cqe_size, static_cast<unsigned long long>(info->out.swdb_addr));
  return true;
}

bool ReleaseTransport(KvafBatchGetTransport *transport) {
  if (transport == nullptr) return true;
  transport->retirement_started = true;
  // Retire both reciprocal imports and all SQ/CQ objects before their MRs.
  // A failed retirement retains the transport and its caller-owned DDR.
  if (!hccp::RetireLocalAndDeviceQueues(transport->api, transport->device_context,
        transport->workers.data(), transport->worker_count, transport->local_host.get())) {
    std::fprintf(stderr, "[KVAF_AIV] BATCH_GET_TEARDOWN_RETAINED phase=queues\n");
    return false;
  }
  for (const auto &local : transport->local_buffers) {
    if (local != nullptr && !ReleaseLocalBufferResources(transport, local.get())) return false;
  }
  transport->local_buffers.clear();
  if (transport->device_peer_memory_import) {
    const int ret = transport->api.rmem_unimport && transport->device_context
        ? transport->api.rmem_unimport(transport->device_context, transport->device_peer_memory_import) : -1;
    if (ret != 0) {
      std::fprintf(stderr, "[KVAF_AIV] BATCH_GET_TEARDOWN_RETAINED phase=unimport_peer_memory "
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

int32_t CreateBatchGetTransport(void *host_ddr, uint64_t host_ddr_bytes,
                                uint32_t aiv_core_count, uint32_t jettys_per_core,
                                bool local_host_ddr, void *peer_link,
                                const uint8_t *peer_local_eid,
                                KvafBatchGetTransport **transport) {
  if (host_ddr == nullptr || host_ddr_bytes == 0U ||
      (!local_host_ddr && (peer_link == nullptr || peer_local_eid == nullptr)) ||
      aiv_core_count == 0U || aiv_core_count > KVAF_AIV_BATCH_GET_MAX_AIV_CORES || jettys_per_core == 0U ||
      jettys_per_core > KVAF_AIV_BATCH_GET_MAX_JETTYS_PER_CORE ||
      static_cast<uint64_t>(aiv_core_count) * jettys_per_core >
          static_cast<uint64_t>(KVAF_AIV_BATCH_GET_MAX_AIV_CORES) * KVAF_AIV_BATCH_GET_MAX_JETTYS_PER_CORE ||
      transport == nullptr) {
    return KVAF_BATCH_GET_INVALID_ARGUMENT;
  }
  *transport = nullptr;

  auto *state = new (std::nothrow) KvafBatchGetTransport();
  if (state == nullptr) {
    return KVAF_BATCH_GET_RUNTIME_ERROR;
  }
  uint32_t physical_device_id = 0U;
  std::array<uint8_t, hccp::kEidBytes> device_eid{};
  std::array<uint8_t, hccp::kEidBytes> peer_eid{};
  std::array<uint8_t, hccp::kEidBytes> wqe_remote_eid{};
  uint32_t device_transport_token_id = 0U;
  // Keep this before the first `goto failed`: C++ forbids jumping over even
  // a scalar initialization into the common cleanup label below.
  const uint32_t entity_count = aiv_core_count * jettys_per_core;
  int32_t result = KVAF_BATCH_GET_DIRECT_HCCP_ERROR;
  state->peer_link = static_cast<hccp::PeerLink *>(peer_link);

  if (!hccp::GetCurrentNpuPhysicalDeviceId(&physical_device_id)) {
    std::fprintf(stderr, "[KVAF_AIV] unable to resolve physical device id from selected ACL device\n");
    result = KVAF_BATCH_GET_EID_ERROR;
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
    if (!hccp::LoadPeerMemory(state->peer_link, reinterpret_cast<uint64_t>(host_ddr),
                              &state->peer_memory, &peer_eid)) goto failed;
  }
  wqe_remote_eid = ReverseEidForAivWqe(peer_eid);
  if (
      !InitContext(state->api, hccp::kOfflineHdc, physical_device_id, device_eid, "device_aiv",
                   &state->device_context) ||
      !AllocateToken(state->api, state->device_context, "device_transport", &state->device_transport_token_handle,
                     &device_transport_token_id)) {
    goto failed;
  }
  state->device_transport_token_value = state->peer_memory.token_value;
  if (!ImportRemoteMemory(state->api, state->device_context, state->peer_memory,
                          &state->device_peer_memory_import)) {
    goto failed;
  }
  state->worker_count = entity_count;
  for (uint32_t entity_index = 0U; entity_index < entity_count; ++entity_index) {
    const uint32_t core = entity_index / jettys_per_core;
    const uint32_t jetty_slot = entity_index % jettys_per_core;
    DirectAivWorker &worker = state->workers[entity_index];
    const std::string suffix = "core=" + std::to_string(core) + "_jetty=" + std::to_string(jetty_slot);
    const std::string device_cq_role = "device_user_ctl_" + suffix;
    const std::string device_jetty_role = "device_aiv_dev_used_" + suffix;
    if (!CreateCq(state->api, state->device_context, hccp::kJfcUserCtlNormal,
                  kDeviceUserCtlCqDepth, device_cq_role.c_str(), &worker.device_cq_info,
                  &worker.device_user_ctl_cq) ||
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
             physical_device_id, reinterpret_cast<uint64_t>(host_ddr),
             &state->peer_memory, &device_eid, &peer_eid,
             &worker.device_user_ctl_jetty.info, device_ctp_role.c_str()},
            &channel);
    worker.device_peer_import_info = channel.local_import_info;
    worker.device_peer_target_jetty = channel.local_target_jetty;
    if (!channel_ready) goto failed;
  }

  {
    auto &entities = state->entity_templates;
    for (uint32_t entity_index = 0U; entity_index < entity_count; ++entity_index) {
      const uint32_t core = entity_index / jettys_per_core;
      const uint32_t jetty_slot = entity_index % jettys_per_core;
      const DirectAivWorker &worker = state->workers[entity_index];
      KvafBatchGetDirectUrmaEntity &entity = entities[entity_index];
      entity.magic = kDirectEntityMagic;
      entity.version = kDirectEntityVersion;
      entity.remote_host_ddr_address = reinterpret_cast<uint64_t>(host_ddr);
      entity.remote_host_ddr_size = host_ddr_bytes;
      entity.remote_token_id = state->peer_memory.token_id;
      entity.remote_token_value = state->peer_memory.token_value;
      entity.sq_va = worker.device_user_ctl_jetty.info.ub.sq_buffer_va;
      entity.sq_db_va = worker.device_user_ctl_jetty.info.ub.db_addr;
      entity.jetty_id = worker.device_user_ctl_jetty.info.ub.id;
      // AIV indexes the USER_CTL SQ in 64-byte WQEBBs.  The HCCL AIV reference
      // exposes this context depth as sqDepth * WQE_NUM_PER_SQE, matching the
      // four-WQEBB storage allocation made in CreateJetty.
      entity.sq_depth = kDeviceUserCtlSqDepth * kWqebbPerWqe;
      entity.tp_id = worker.device_peer_import_info.out.ub.tpn;
      entity.wqe_size = kWqeBytes;
      std::memcpy(entity.remote_eid, wqe_remote_eid.data(), sizeof(entity.remote_eid));
      entity.cq_va = worker.device_cq_info.out.buffer_addr;
      entity.cq_db_va = worker.device_cq_info.out.swdb_addr;
      entity.jfc_id = worker.device_cq_info.out.id;
      entity.cq_depth = kDeviceUserCtlCqDepth;
      entity.cqe_size = worker.device_cq_info.out.cqe_size;
      if (entity.sq_va == 0U || entity.sq_db_va == 0U || entity.cq_va == 0U || entity.cq_db_va == 0U ||
          entity.cqe_size == 0U || entity.tp_id == 0U) {
        std::fprintf(stderr,
                     "[KVAF_AIV] DIRECT_AIV_ENTITY_BUILD_FAILED core=%u jetty_slot=%u sq=0x%llx cq=0x%llx tpn=%u cqe=%u\n",
                     core, jetty_slot, static_cast<unsigned long long>(entity.sq_va),
                     static_cast<unsigned long long>(entity.cq_va), entity.tp_id, entity.cqe_size);
        goto failed;
      }
      // HCCP must give every AIV block independent queue state. Do not
      // proceed if a private-ABI mismatch silently aliases two workers: that
      // would reintroduce the multi-producer PI/CI race this topology is
      // designed to eliminate. The TPN is intentionally not compared here:
      // HCCL's CTP implementation documents that it may be shared when the
      // transport path is reused by multiple imported Jettys.
      for (uint32_t previous = 0U; previous < entity_index; ++previous) {
        const KvafBatchGetDirectUrmaEntity &other = entities[previous];
        if (entity.sq_va == other.sq_va || entity.sq_db_va == other.sq_db_va || entity.cq_va == other.cq_va ||
            entity.cq_db_va == other.cq_db_va || entity.jetty_id == other.jetty_id) {
          std::fprintf(stderr,
                       "[KVAF_AIV] DIRECT_AIV_ENTITY_COLLISION core=%u previous=%u "
                       "sq=0x%llx cq=0x%llx jetty_id=%u tpn=%u\n",
                       entity_index, previous, static_cast<unsigned long long>(entity.sq_va),
                       static_cast<unsigned long long>(entity.cq_va), entity.jetty_id, entity.tp_id);
          goto failed;
        }
      }
    }
    for (uint32_t entity_index = 0U; entity_index < entity_count; ++entity_index) {
      const uint32_t core = entity_index / jettys_per_core;
      const uint32_t jetty_slot = entity_index % jettys_per_core;
      const KvafBatchGetDirectUrmaEntity &entity = entities[entity_index];
      std::fprintf(stderr,
                   "[KVAF_AIV] DIRECT_AIV_ENTITY_TEMPLATE_READY core=%u jetty_slot=%u entity_index=%u sq_depth_wqebbs=%u cq_depth=%u "
                   "architecture=peer_link_to_device_aiv_dev_used_user_ctl\n",
                   core, jetty_slot, entity_index, entity.sq_depth, entity.cq_depth);
    }
  }

  std::fprintf(stderr,
               "[KVAF_AIV] DIRECT_TRANSPORT_READY peer_memory_bytes=%llu workers=%u local_hbm=deferred_acquire\n",
               static_cast<unsigned long long>(host_ddr_bytes), entity_count);
  *transport = state;
  return KVAF_BATCH_GET_SUCCESS;

failed:
  // A partially registered endpoint may fail to retire. Return its ownership
  // on error so AccessOperator/Runtime keep DDR alive until shutdown succeeds.
  if (!ReleaseTransport(state)) *transport = state;
  return result;
}

extern "C" int32_t KvafBatchGetTransportCreateLocalHost(
    void *host_ddr, uint64_t host_ddr_bytes, uint32_t aiv_core_count,
    uint32_t jettys_per_core, KvafBatchGetTransport **transport) {
  return CreateBatchGetTransport(host_ddr, host_ddr_bytes, aiv_core_count,
                                   jettys_per_core, true, nullptr, nullptr, transport);
}

extern "C" int32_t KvafBatchGetTransportDestroyChecked(KvafBatchGetTransport *transport) {
  return ReleaseTransport(transport) ? KVAF_BATCH_GET_SUCCESS : KVAF_BATCH_GET_RUNTIME_ERROR;
}

extern "C" int32_t KvafBatchGetTransportCreateWithPeerLink(
    void *host_ddr, uint64_t host_ddr_bytes, uint32_t aiv_core_count,
    uint32_t jettys_per_core, void *peer_link, const uint8_t *peer_local_eid,
    KvafBatchGetTransport **transport) {
  if (peer_link == nullptr || peer_local_eid == nullptr) return KVAF_BATCH_GET_INVALID_ARGUMENT;
  return CreateBatchGetTransport(host_ddr, host_ddr_bytes, aiv_core_count,
                                 jettys_per_core, false, peer_link, peer_local_eid, transport);
}

extern "C" int32_t KvafBatchGetLocalBufferAcquire(KvafBatchGetTransport *transport, void *output_hbm,
                                                    uint64_t output_hbm_bytes,
                                                    KvafBatchGetLocalBuffer **local_buffer,
                                                    uint64_t *device_channel_entity) {
  if (!AcquireLocalBuffer(transport, output_hbm, output_hbm_bytes, local_buffer, device_channel_entity)) {
    return KVAF_BATCH_GET_DIRECT_HCCP_ERROR;
  }
  return KVAF_BATCH_GET_SUCCESS;
}

extern "C" void KvafBatchGetLocalBufferRelease(KvafBatchGetLocalBuffer *local_buffer) {
  if (local_buffer == nullptr || local_buffer->transport == nullptr) return;
  if (local_buffer->reference_count == 0U) {
    std::fprintf(stderr, "[KVAF_AIV] DIRECT_LOCAL_BUFFER_RELEASE_WARNING reason=reference_underflow output_hbm=0x%llx\n",
                 static_cast<unsigned long long>(local_buffer->output_hbm_address));
    return;
  }
  --local_buffer->reference_count;
  if (LocalBufferVerboseLoggingEnabled()) {
    std::fprintf(stderr,
                 "[KVAF_AIV] DIRECT_LOCAL_BUFFER_RELEASE output_hbm=0x%llx bytes=%llu refs=%u "
                 "cache_retained_until_transport_destroy=1\n",
                 static_cast<unsigned long long>(local_buffer->output_hbm_address),
                 static_cast<unsigned long long>(local_buffer->output_hbm_bytes),
                 local_buffer->reference_count);
  }
}

extern "C" void KvafBatchGetTransportDestroy(KvafBatchGetTransport *transport) {
  ReleaseTransport(transport);
}
