#include "kvaf_aiv_batch_get.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <map>
#include <mutex>
#include <new>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "direct_hccp/direct_hccp_abi.h"
#include "direct_hccp/direct_hccp_remote_transport.h"
#include "direct_hccp/direct_hccp_session.h"

namespace hccp = kvaf::direct_hccp;

constexpr uint64_t kPageBytes = 4096U;
constexpr uint32_t kDirectEntityMagic = 0x4b564155U;  // "KVAU"
constexpr uint32_t kDirectEntityVersion = KVAF_AIV_DIRECT_URMA_ENTITY_VERSION;
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

struct LocalPathRegistration {
  void *token_handle = nullptr;
  DirectMemory memory{};
};

// A local buffer has stable heap storage because it is an opaque handle held
// by the caller. Zero-reference entries stay cached until transport teardown.
struct RemoteEntityView {
  void* data{};
  bool ready{};
};
struct KvafBatchGetLocalBuffer {
  KvafBatchGetTransport *transport = nullptr;
  std::vector<LocalPathRegistration> path_registrations{};
  uint64_t output_hbm_address = 0U;
  uint64_t output_hbm_bytes = 0U;
  uint32_t reference_count = 0U;
  bool ready{false};
  void *device_entities = nullptr;
  uint64_t device_channel_entity = 0U;
  // Only descriptor storage varies by remote MR. The local registrations
  // and all SQ/CQ addresses remain shared. MR 1 uses device_entities above.
  std::map<uint64_t, RemoteEntityView> remote_entities;
};

// A USER_CTL SQ/CQ is strictly single-producer/single-consumer from the AIV
// kernel's point of view. A multi-AIV launch therefore owns one local AIV
// Jetty per core rather than sharing producer/consumer indices.
using DirectAivWorker = hccp::DeviceAivWorkerResources;

struct BatchGetPath : hccp::DeviceAivPathResources {
  kvaf::UrmaRemoteMemoryDesc peer_memory{};
  hccp::PeerLink* link{};
};

struct RemoteMemoryRegistration {
  kvaf::UrmaRemoteMemoryDesc memory{};
  uint64_t base{}, bytes{};
  std::vector<std::shared_ptr<hccp::HccpRemoteMemoryBinding>> imports;
  bool ready{};
};
struct KvafBatchGetTransport {
  DirectHccpApi api{};
  hccp::DirectHccpSession *session = nullptr;
  uint32_t physical_device_id = 0;
  // Each (local EID, remote FE) path owns its RaCtx, import and transport token.
  std::vector<BatchGetPath> paths{};
  std::map<uint64_t, RemoteMemoryRegistration> remote_memories;
  kvaf::UrmaBindingCache<hccp::HccpRemoteMemoryBinding> bindings{kvaf::MemoryDirection::Get};
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
  bool closing{false};
};

std::string EidToHex(const std::array<uint8_t, hccp::kEidBytes> &eid) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string result;
  result.reserve(hccp::kEidBytes * 2U);
  for (const uint8_t byte : eid) {
    result.push_back(kHex[byte >> 4U]);
    result.push_back(kHex[byte & 0x0fU]);
  }
  return result;
}

std::array<uint8_t, hccp::kEidBytes> ReverseEidForAivWqe(
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

bool InitContext(const DirectHccpApi &api, int32_t mode, uint32_t physical_device_id,
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

bool AllocateToken(const DirectHccpApi &api, void *context, const char *role, void **token_handle,
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

bool AlignRange(uint64_t address, uint64_t bytes, uint64_t *aligned_address, uint64_t *aligned_bytes) {
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

bool RegisterMemory(const DirectHccpApi &api, void *context, void *token_handle, uint32_t token_id,
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
  // The output registration owns the caller VA and allocated token. Check the
  // provider segment before publishing an entity with that identity, just as
  // the HBM exporter checks its remote segment. A translated local VA needs a
  // separate addressing contract; it cannot silently use the caller address.
  uint64_t segment_base=0,segment_bytes=0;
  uint32_t segment_token=0,segment_uasid=0,segment_attributes=0;
  if(memory->key_bytes==48) {
    std::memcpy(&segment_uasid,memory->key+16,4);
    std::memcpy(&segment_base,memory->key+20,8);
    std::memcpy(&segment_bytes,memory->key+32,8);
    std::memcpy(&segment_attributes,memory->key+40,4);
    std::memcpy(&segment_token,memory->key+44,4);
  }
  std::array<uint8_t,hccp::kEidBytes> segment_eid{};
  if(memory->key_bytes>=segment_eid.size())std::memcpy(segment_eid.data(),memory->key,segment_eid.size());
  const bool valid=memory->key_bytes==48 && segment_base==aligned_address &&
      segment_bytes>=aligned_bytes && segment_bytes<=UINT64_MAX-segment_base &&
      token_id && (segment_token>>kTokenIdShift)==token_id;
  std::fprintf(stderr,
      "[KVAF_AIV] DIRECT_LOCAL_HBM_SEGMENT role=%s valid=%u key_bytes=%u "
      "requested_base=0x%llx requested_bytes=%llu segment_base=0x%llx segment_bytes=%llu "
      "allocated_registration_tid=0x%x segment_raw_token_id=0x%x ra_token_id=0x%x "
      "uasid=%u attributes=0x%x registration_flags=0x%x segment_eid=%s\n",
      role,unsigned(valid),memory->key_bytes,(unsigned long long)aligned_address,
      (unsigned long long)aligned_bytes,(unsigned long long)segment_base,(unsigned long long)segment_bytes,
      token_id,segment_token,info.out.ub.token_id,segment_uasid,segment_attributes,
      info.in.ub.flags,EidToHex(segment_eid).c_str());
  if(!valid)return false;
  std::fprintf(stderr,
               "[KVAF_AIV] DIRECT_MEMORY_REGISTERED role=%s token_id=0x%x host_or_hbm_va=0x%llx bytes=%llu\n",
               role, memory->token_id, static_cast<unsigned long long>(aligned_address),
               static_cast<unsigned long long>(aligned_bytes));
  return true;
}

int32_t ReleaseLocalBufferResources(KvafBatchGetTransport *transport, KvafBatchGetLocalBuffer *local) {
  if (transport == nullptr || local == nullptr) return KVAF_BATCH_GET_SUCCESS;
  for (std::size_t reverse = local->path_registrations.size(); reverse > 0U; --reverse) {
    const std::size_t i = reverse - 1;
    if (i >= transport->paths.size()) return KVAF_BATCH_GET_DIRECT_HCCP_ERROR;
    auto& registration = local->path_registrations[i];
    auto* context = transport->paths[i].device_context;
    if (registration.memory.handle) {
      if (!transport->api.lmem_unregister || !context ||
          transport->api.lmem_unregister(context, registration.memory.handle))
        return KVAF_BATCH_GET_DIRECT_HCCP_ERROR;
      registration.memory.handle = nullptr;
    }
    if (registration.token_handle) {
      if (!transport->api.token_free || !context ||
          transport->api.token_free(context, registration.token_handle))
        return KVAF_BATCH_GET_DIRECT_HCCP_ERROR;
      registration.token_handle = nullptr;
    }
  }
  for (auto& view : local->remote_entities) if (view.second.data) {
    if (aclrtFree(view.second.data) != ACL_SUCCESS) return KVAF_BATCH_GET_RUNTIME_ERROR;
    view.second = {};
  }
  if (local->device_entities) {
    if (aclrtFree(local->device_entities) != ACL_SUCCESS) return KVAF_BATCH_GET_RUNTIME_ERROR;
    local->device_entities = nullptr;
    local->device_channel_entity = 0;
  }
  return KVAF_BATCH_GET_SUCCESS;
}

bool AcquireLocalBuffer(KvafBatchGetTransport *transport, void *output_hbm, uint64_t output_hbm_bytes,
                        KvafBatchGetLocalBuffer **local_buffer, uint64_t *device_channel_entity) {
  if (transport == nullptr || transport->closing || output_hbm == nullptr || output_hbm_bytes == 0U || local_buffer == nullptr ||
      device_channel_entity == nullptr || transport->paths.empty() || transport->worker_count == 0U) {
    return false;
  }
  *local_buffer = nullptr;
  *device_channel_entity = 0;
  const uint64_t output_address = reinterpret_cast<uint64_t>(output_hbm);
  for (const auto &existing : transport->local_buffers) {
    if (existing != nullptr && existing->output_hbm_address == output_address &&
        existing->output_hbm_bytes == output_hbm_bytes) {
      if (!existing->ready) return false;  // Registration is awaiting retirement.
      ++existing->reference_count;
      *local_buffer = existing.get();
      *device_channel_entity = existing->device_channel_entity;
      if (LocalBufferVerboseLoggingEnabled()) {
        std::fprintf(stderr,
                     "[KVAF_AIV] DIRECT_LOCAL_BUFFER_ACQUIRE_CACHE_HIT output_hbm=0x%llx bytes=%llu token_id=0x%x "
                     "refs=%u entity_table=0x%llx\n",
                     static_cast<unsigned long long>(output_address),
                     static_cast<unsigned long long>(output_hbm_bytes),
                     existing->path_registrations.front().memory.token_id,
                     existing->reference_count,
                     static_cast<unsigned long long>(existing->device_channel_entity));
      }
      return true;
    }
  }

  // Establish ownership before registration or any later allocation can fail.
  transport->local_buffers.push_back(std::make_unique<KvafBatchGetLocalBuffer>());
  auto* candidate = transport->local_buffers.back().get();
  const auto rollback = [&] {
    if (ReleaseLocalBufferResources(transport, candidate) == KVAF_BATCH_GET_SUCCESS)
      transport->local_buffers.pop_back();
    else {
      candidate->reference_count = 1U;
      *local_buffer = candidate; // Caller retains its buffer until Retire/Close.
    }
    return false;
  };
  candidate->transport = transport;
  candidate->output_hbm_address = output_address;
  candidate->output_hbm_bytes = output_hbm_bytes;
  candidate->path_registrations.resize(transport->paths.size());
  for (std::size_t path_index = 0U; path_index < transport->paths.size(); ++path_index) {
    const auto& path = transport->paths[path_index];
    LocalPathRegistration &registration = candidate->path_registrations[path_index];
    uint32_t token_id = 0U;
    // Do not allocate Host metadata after the first path registration exists.
    char token_role[64], memory_role[64];
    std::snprintf(token_role, sizeof(token_role), "device_local_hbm_eid_path=%zu", path_index);
    std::snprintf(memory_role, sizeof(memory_role), "device_hbm_eid_path=%zu", path_index);
    if (path.device_context == nullptr ||
        !AllocateToken(transport->api, path.device_context, token_role,
                       &registration.token_handle, &token_id) ||
        !RegisterMemory(transport->api, path.device_context, registration.token_handle,
                        token_id, path.peer_memory.token_value, output_hbm,
                        output_hbm_bytes, memory_role, &registration.memory)) {
      return rollback();
    }
  }

  std::array<KvafBatchGetDirectUrmaEntity, kMaxDirectAivWorkers> entities = transport->entity_templates;
  for (uint32_t index = 0U; index < transport->worker_count; ++index) {
    const std::uint32_t path_index = transport->workers[index].path_index;
    if (path_index >= candidate->path_registrations.size()) {
      return rollback();
    }
    const DirectMemory &memory = candidate->path_registrations[path_index].memory;
    entities[index].local_hbm_address = output_address;
    entities[index].local_hbm_size = output_hbm_bytes;
    entities[index].local_token_id = memory.token_id;
    entities[index].local_token_value = memory.token_value;
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
    return rollback();
  }
  candidate->device_channel_entity = reinterpret_cast<uint64_t>(candidate->device_entities);
  candidate->reference_count = 1U;
  candidate->ready = true;
  *local_buffer = candidate;
  *device_channel_entity = candidate->device_channel_entity;
  std::fprintf(stderr,
               "[KVAF_AIV] DIRECT_LOCAL_BUFFER_ACQUIRE_NEW output_hbm=0x%llx bytes=%llu token_id=0x%x refs=1 "
               "eid_paths=%zu entity_table=0x%llx transport_reused=1\n",
               static_cast<unsigned long long>(output_address), static_cast<unsigned long long>(output_hbm_bytes),
               candidate->path_registrations.front().memory.token_id,
               candidate->path_registrations.size(),
               static_cast<unsigned long long>(candidate->device_channel_entity));
  return true;
}

bool CreateCq(const DirectHccpApi &api, void *context, uint32_t mode, uint32_t depth, const char *role,
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

int32_t ReleaseTransport(KvafBatchGetTransport *transport) {
  if (transport == nullptr) return KVAF_BATCH_GET_SUCCESS;
  transport->closing = true;
  auto& api = transport->api;
  // Every handle is cleared only after its API confirms success. In
  // particular, no output registration is released while a queue still lives.
  for (uint32_t i = 0; i < transport->worker_count; ++i) {
    auto& worker = transport->workers[i];
    if (worker.device_peer_target_jetty) {
      if (worker.path_index >= transport->paths.size()) return KVAF_BATCH_GET_DIRECT_HCCP_ERROR;
      auto* context = transport->paths[worker.path_index].device_context;
      if (!api.qp_unimport || !context || api.qp_unimport(context, worker.device_peer_target_jetty))
        return KVAF_BATCH_GET_DIRECT_HCCP_ERROR;
      worker.device_peer_target_jetty = nullptr;
    }
  }
  for (uint32_t i = 0; i < transport->worker_count; ++i) {
    auto& worker = transport->workers[i];
    if (worker.device_user_ctl_jetty.handle) {
      if (!api.qp_destroy || api.qp_destroy(worker.device_user_ctl_jetty.handle))
        return KVAF_BATCH_GET_DIRECT_HCCP_ERROR;
      worker.device_user_ctl_jetty.handle = nullptr;
    }
  }
  for (uint32_t i = 0; i < transport->worker_count; ++i) {
    auto& worker = transport->workers[i];
    if (worker.device_user_ctl_cq) {
      if (worker.path_index >= transport->paths.size()) return KVAF_BATCH_GET_DIRECT_HCCP_ERROR;
      auto* context = transport->paths[worker.path_index].device_context;
      if (!api.cq_destroy || !context || api.cq_destroy(context, worker.device_user_ctl_cq))
        return KVAF_BATCH_GET_DIRECT_HCCP_ERROR;
      worker.device_user_ctl_cq = nullptr;
    }
  }
  for (auto& local : transport->local_buffers) {
    const auto released = ReleaseLocalBufferResources(transport, local.get());
    if (released != KVAF_BATCH_GET_SUCCESS) return released;
  }
  for (auto& remote : transport->remote_memories)
    for (auto& binding : remote.second.imports)
      if (!transport->bindings.Release(&binding).ok()) return KVAF_BATCH_GET_DIRECT_HCCP_ERROR;
  // Keep LocalBuffer objects valid until close succeeds; contexts can still
  // retain their opaque pointer when a later path cleanup fails.
  for (auto it = transport->paths.rbegin(); it != transport->paths.rend(); ++it) {
    auto& path = *it;
    if (!transport->bindings.Release(&path.peer_binding).ok())
      return KVAF_BATCH_GET_DIRECT_HCCP_ERROR;
    if (path.device_transport_token_handle) {
      if (!api.token_free || !path.device_context ||
          api.token_free(path.device_context, path.device_transport_token_handle))
        return KVAF_BATCH_GET_DIRECT_HCCP_ERROR;
      path.device_transport_token_handle = nullptr;
    }
    if (path.device_context) {
      if (!api.ctx_deinit || api.ctx_deinit(path.device_context)) return KVAF_BATCH_GET_DIRECT_HCCP_ERROR;
      path.device_context = nullptr;
    }
  }
  if (transport->session) {
    hccp::ReleaseDirectHccpSession(transport->session);
    transport->session = nullptr;
  }
  hccp::CloseDirectHccp(&api);
  delete transport;
  return KVAF_BATCH_GET_SUCCESS;
}

extern "C" int32_t KvafBatchGetTransportCreateWithPeerLinks(void *host_ddr, uint64_t host_ddr_bytes,
                                uint32_t aiv_core_count, uint32_t jettys_per_core,
                                void *const *peer_links, uint32_t peer_link_count,
                                const uint8_t *peer_local_eids,
                                uint32_t peer_local_eid_count,
                                KvafBatchGetTransport **transport) {
  const uint64_t entity_count_u64 =
      static_cast<uint64_t>(aiv_core_count) * jettys_per_core;
  if (host_ddr == nullptr || host_ddr_bytes == 0U || peer_links == nullptr || peer_local_eids == nullptr ||
      peer_link_count == 0U || peer_link_count > entity_count_u64 ||
      (peer_link_count > 1U && peer_local_eid_count != 1U) ||
      aiv_core_count == 0U || aiv_core_count > KVAF_AIV_BATCH_GET_MAX_AIV_CORES || jettys_per_core == 0U ||
      jettys_per_core > KVAF_AIV_BATCH_GET_MAX_JETTYS_PER_CORE ||
      entity_count_u64 >
          static_cast<uint64_t>(KVAF_AIV_BATCH_GET_MAX_AIV_CORES) * KVAF_AIV_BATCH_GET_MAX_JETTYS_PER_CORE ||
      peer_local_eid_count == 0U || peer_local_eid_count > entity_count_u64 || transport == nullptr) {
    return KVAF_BATCH_GET_INVALID_ARGUMENT;
  }
  const uint32_t entity_count = static_cast<uint32_t>(entity_count_u64);
  for (uint32_t path_index = 0U; path_index < peer_local_eid_count; ++path_index) {
    const uint8_t *eid = peer_local_eids + static_cast<size_t>(path_index) * hccp::kEidBytes;
    if (std::all_of(eid, eid + hccp::kEidBytes,
                    [](uint8_t byte) { return byte == 0U; })) {
      return KVAF_BATCH_GET_INVALID_ARGUMENT;
    }
    for (uint32_t previous = 0U; previous < path_index; ++previous) {
      const uint8_t *previous_eid =
          peer_local_eids + static_cast<size_t>(previous) * hccp::kEidBytes;
      if (std::equal(eid, eid + hccp::kEidBytes, previous_eid)) {
        return KVAF_BATCH_GET_INVALID_ARGUMENT;
      }
    }
  }
  *transport = nullptr;

  auto *state = new (std::nothrow) KvafBatchGetTransport();
  if (state == nullptr) {
    return KVAF_BATCH_GET_RUNTIME_ERROR;
  }
  uint32_t physical_device_id = 0U;
  // Keep this before the first `goto failed`: C++ forbids jumping over even
  // a scalar initialization into the common cleanup label below.
  int32_t result = KVAF_BATCH_GET_DIRECT_HCCP_ERROR;
  const uint32_t path_count = std::max(peer_link_count, peer_local_eid_count);
  state->paths.resize(path_count);

  if (!hccp::GetCurrentNpuPhysicalDeviceId(&physical_device_id)) {
    std::fprintf(stderr, "[KVAF_AIV] unable to resolve physical device id from selected ACL device\n");
    result = KVAF_BATCH_GET_EID_ERROR;
    goto failed;
  }
  state->physical_device_id = physical_device_id;
  if (!hccp::LoadDirectHccp(&state->api)) goto failed;
  if (!hccp::AcquireDirectHccpSession(physical_device_id, kPidHdcType, &state->session)) {
    goto failed;
  }
  for (uint32_t path_index = 0U; path_index < path_count; ++path_index) {
    auto& path = state->paths[path_index];
    path.link = static_cast<hccp::PeerLink*>(peer_links[peer_link_count == 1 ? 0 : path_index]);
    const auto local_index = peer_local_eid_count == 1 ? 0 : path_index;
    std::copy_n(peer_local_eids + local_index * hccp::kEidBytes, hccp::kEidBytes, path.local_eid.begin());
    if (!hccp::LoadPeerMemory(path.link, reinterpret_cast<uint64_t>(host_ddr), &path.peer_memory) ||
        path.peer_memory.registered_bytes < host_ddr_bytes) goto failed;
    uint32_t device_transport_token_id = 0U;
    const std::string role = "device_aiv_eid_path=" + std::to_string(path_index);
    std::fprintf(stderr,
                 "[KVAF_AIV] PEER_LINK_LOCAL_EID_READY physical_device=%u "
                 "eid_path=%u/%u eid=%s\n",
                 physical_device_id, path_index, path_count,
                 EidToHex(path.local_eid).c_str());
    if (!InitContext(state->api, hccp::kOfflineHdc, physical_device_id,
                     path.local_eid, role.c_str(), &path.device_context) ||
        !AllocateToken(state->api, path.device_context, role.c_str(),
                       &path.device_transport_token_handle,
                       &device_transport_token_id) ||
        !state->bindings.Acquire(
            kvaf::UrmaBindingKey::Make(kvaf::MemoryDirection::Get, path.device_context,
                path.context_generation, path.peer_memory, hccp::MemImportFlags()),
            [&](hccp::HccpRemoteMemoryBinding& binding) {
              return binding.Import(path.device_context, state->api.rmem_import,
                                    state->api.rmem_unimport, path.peer_memory);
            }, &path.peer_binding).ok()) {
      goto failed;
    }
  }
  state->worker_count = entity_count;
  for (uint32_t entity_index = 0U; entity_index < entity_count; ++entity_index) {
    const uint32_t core = entity_index / jettys_per_core;
    const uint32_t jetty_slot = entity_index % jettys_per_core;
    const uint32_t path_index = hccp::GetPathForWorker(entity_index, aiv_core_count, jettys_per_core,
                                                     peer_local_eid_count, peer_link_count);
    auto& path = state->paths[path_index];
    DirectAivWorker &worker = state->workers[entity_index];
    worker.path_index = path_index;
    const std::string suffix = "core=" + std::to_string(core) + "_jetty=" +
                               std::to_string(jetty_slot) + "_eid_path=" +
                               std::to_string(path_index);
    const std::string device_cq_role = "device_user_ctl_" + suffix;
    const std::string device_jetty_role = "device_aiv_dev_used_" + suffix;
    if (!CreateCq(state->api, path.device_context, hccp::kJfcUserCtlNormal,
                  kDeviceUserCtlCqDepth, device_cq_role.c_str(), &worker.device_cq_info,
                  &worker.device_user_ctl_cq) ||
        !hccp::CreateDeviceAivJetty(state->api, path.device_context, worker.device_user_ctl_cq,
                           path.device_transport_token_handle, path.peer_memory.token_value,
                           kDeviceUserCtlSqDepth, device_jetty_role.c_str(),
                           &worker.device_user_ctl_jetty.info, &worker.device_user_ctl_jetty.handle)) {
      goto failed;
    }
    const std::string device_ctp_role = "device_aiv_peer_link_" + suffix;
    hccp::PeerChannel channel{};
    if (!hccp::CreatePeerChannel(
            {path.link, &state->api, path.device_context,
             physical_device_id, reinterpret_cast<uint64_t>(host_ddr),
             &path.peer_memory, &path.local_eid, &path.peer_memory.eid,
             &worker.device_user_ctl_jetty.info, device_ctp_role.c_str()},
            &channel)) {
      goto failed;
    }
    worker.device_peer_import_info = channel.local_import_info;
    worker.device_peer_target_jetty = channel.local_target_jetty;
  }

  {
    auto &entities = state->entity_templates;
    for (uint32_t entity_index = 0U; entity_index < entity_count; ++entity_index) {
      const uint32_t core = entity_index / jettys_per_core;
      const uint32_t jetty_slot = entity_index % jettys_per_core;
      const uint32_t path_index = state->workers[entity_index].path_index;
      const auto& memory = state->paths[path_index].peer_memory;
      const auto wqe_remote_eid = ReverseEidForAivWqe(memory.wqe_eid);
      const DirectAivWorker &worker = state->workers[entity_index];
      KvafBatchGetDirectUrmaEntity &entity = entities[entity_index];
      entity.magic = kDirectEntityMagic;
      entity.version = kDirectEntityVersion;
      entity.remote_host_ddr_address = reinterpret_cast<uint64_t>(host_ddr);
      entity.remote_host_ddr_size = host_ddr_bytes;
      entity.registered_base = memory.registered_address;
      entity.remote_token_id = memory.wqe_token_id();
      entity.remote_token_value = memory.token_value;
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
        const bool same_path = state->workers[previous].path_index == path_index;
        if (entity.sq_va == other.sq_va || entity.sq_db_va == other.sq_db_va || entity.cq_va == other.cq_va ||
            entity.cq_db_va == other.cq_db_va || (same_path && entity.jetty_id == other.jetty_id)) {
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
      const uint32_t path_index = state->workers[entity_index].path_index;
      const KvafBatchGetDirectUrmaEntity &entity = entities[entity_index];
      std::fprintf(stderr,
                   "[KVAF_AIV] DIRECT_AIV_ENTITY_TEMPLATE_READY core=%u jetty_slot=%u "
                   "entity_index=%u eid_path=%u sq_depth_wqebbs=%u cq_depth=%u "
                   "architecture=peer_link_to_device_aiv_dev_used_user_ctl\n",
                   core, jetty_slot, entity_index, path_index, entity.sq_depth, entity.cq_depth);
    }
  }

  std::fprintf(stderr,
               "[KVAF_AIV] DIRECT_TRANSPORT_READY peer_memory_bytes=%llu workers=%u "
               "eid_paths=%u remote_fes=%u distribution=%s "
               "local_hbm=deferred_acquire\n",
               static_cast<unsigned long long>(host_ddr_bytes), entity_count,
               path_count, peer_link_count, peer_link_count > 1 ? "core_major_peer_round_robin" :
                                                                 "jetty_major_worker_round_robin");
  // All local paths of a single peer can share that peer's additional MR views.
  if (peer_link_count == 1) {
    state->remote_memories.emplace(1, RemoteMemoryRegistration{
        state->paths.front().peer_memory, reinterpret_cast<uint64_t>(host_ddr), host_ddr_bytes, {}, true});
  }
  *transport = state;
  return KVAF_BATCH_GET_SUCCESS;

failed:
  // A partial driver cleanup can fail too. Return the retained object so the
  // owner can retry Close before releasing the peer region or its libraries.
  if (ReleaseTransport(state) != KVAF_BATCH_GET_SUCCESS) *transport = state;
  return result;
}

extern "C" int32_t KvafBatchGetTransportCreateWithPeerLink(
    void *host_ddr, uint64_t host_ddr_bytes, uint32_t aiv_core_count,
    uint32_t jettys_per_core, void *peer_link, const uint8_t *peer_local_eid,
    KvafBatchGetTransport **transport) {
  if (peer_link == nullptr || peer_local_eid == nullptr) return KVAF_BATCH_GET_INVALID_ARGUMENT;
  return KvafBatchGetTransportCreateWithPeerLinks(host_ddr, host_ddr_bytes, aiv_core_count,
                                 jettys_per_core, &peer_link, 1U, peer_local_eid, 1U,
                                 transport);
}

extern "C" int32_t KvafBatchGetTransportCreateWithPeerLinkMultiEid(
    void *host_ddr, uint64_t host_ddr_bytes, uint32_t aiv_core_count,
    uint32_t jettys_per_core, void *peer_link, const uint8_t *peer_local_eids,
    uint32_t peer_local_eid_count, KvafBatchGetTransport **transport) {
  if (peer_link == nullptr || peer_local_eids == nullptr ||
      peer_local_eid_count == 0U) {
    return KVAF_BATCH_GET_INVALID_ARGUMENT;
  }
  return KvafBatchGetTransportCreateWithPeerLinks(host_ddr, host_ddr_bytes, aiv_core_count,
                                 jettys_per_core, &peer_link, 1U, peer_local_eids,
                                 peer_local_eid_count, transport);
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

extern "C" int32_t KvafBatchGetLocalBufferRetire(KvafBatchGetLocalBuffer* local) {
  if (!local || !local->transport || !local->reference_count) return KVAF_BATCH_GET_INVALID_ARGUMENT;
  if (local->reference_count > 1U) {
    --local->reference_count;
    return KVAF_BATCH_GET_SUCCESS;
  }
  local->ready = false;
  auto* transport = local->transport;
  for (std::size_t i = 0; i < local->path_registrations.size(); ++i) {
    auto& registration = local->path_registrations[i];
    auto* context = transport->paths[i].device_context;
    if (registration.memory.handle) {
      if (transport->api.lmem_unregister(context, registration.memory.handle) != 0) return KVAF_BATCH_GET_DIRECT_HCCP_ERROR;
      registration.memory.handle = nullptr;
    }
    if (registration.token_handle) {
      if (transport->api.token_free(context, registration.token_handle) != 0) return KVAF_BATCH_GET_DIRECT_HCCP_ERROR;
      registration.token_handle = nullptr;
    }
  }
  for (auto& view : local->remote_entities) if (view.second.data) {
    if (aclrtFree(view.second.data) != ACL_SUCCESS) return KVAF_BATCH_GET_RUNTIME_ERROR;
    view.second = {};
  }
  if (local->device_entities) {
    if (aclrtFree(local->device_entities) != ACL_SUCCESS) return KVAF_BATCH_GET_RUNTIME_ERROR;
    local->device_entities = nullptr;
  }
  for (auto it = transport->local_buffers.begin(); it != transport->local_buffers.end(); ++it) {
    if (it->get() == local) { transport->local_buffers.erase(it); return KVAF_BATCH_GET_SUCCESS; }
  }
  return KVAF_BATCH_GET_INVALID_ARGUMENT;
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

extern "C" int32_t KvafBatchGetTransportClose(KvafBatchGetTransport *transport) {
  return ReleaseTransport(transport);
}
extern "C" void KvafBatchGetTransportDestroy(KvafBatchGetTransport *transport) {
  (void)KvafBatchGetTransportClose(transport);
}

// Retire a registration before freeing its storage (probe cleanup or teardown).
// Every successful release nulls its handle, so partial failures are retryable.
extern "C" int32_t KvafBatchGetLocalBufferDiscard(KvafBatchGetLocalBuffer **handle) {
  if (!handle || !*handle) return KVAF_BATCH_GET_SUCCESS;
  auto* local = *handle;
  if (local->reference_count != 1) return KVAF_BATCH_GET_INVALID_ARGUMENT;
  const auto result = KvafBatchGetLocalBufferRetire(local);
  if (!result) *handle = nullptr;
  return result;
}

extern "C" int32_t KvafBatchGetTransportQuiesce(KvafBatchGetTransport *t) {
  if (!t) return KVAF_BATCH_GET_SUCCESS;
  t->closing = true;
  for (uint32_t i = 0; i < t->worker_count; ++i) {
    auto& jetty = t->workers[i].device_user_ctl_jetty;
    if (jetty.handle) {
      if (t->api.qp_destroy(jetty.handle)) return KVAF_BATCH_GET_DIRECT_HCCP_ERROR;
      jetty.handle = nullptr;
    }
  }
  return KVAF_BATCH_GET_SUCCESS;
}

extern "C" int32_t KvafBatchGetTransportAddRemoteMemory(KvafBatchGetTransport* t,
    uint64_t mr_id, uint64_t base, uint64_t bytes) {
  if (!t || t->closing || mr_id<2 || mr_id>4096 || !base || !bytes || bytes>UINT64_MAX-base ||
      !t->remote_memories.count(1) ||
      t->remote_memories.count(mr_id) || !t->local_buffers.empty()) return KVAF_BATCH_GET_INVALID_ARGUMENT;
  kvaf::UrmaRemoteMemoryDesc memory{};std::array<uint8_t,hccp::kEidBytes> route{}, original_route{};
  kvaf::UrmaRemoteMemoryDesc original{};
  if (!hccp::LoadPeerMemory(t->paths.front().link,base,&memory,&route) ||
      !hccp::LoadPeerMemory(t->paths.front().link,t->remote_memories.at(1).base,&original,&original_route) ||
      route!=original_route || memory.token_value!=t->paths.front().peer_memory.token_value ||
      memory.registered_address!=base || memory.registered_bytes!=bytes)
    return KVAF_BATCH_GET_DIRECT_HCCP_ERROR;
  // Own even a partially returned provider handle before any import call.
  auto& remote=t->remote_memories[mr_id];
  remote.memory=memory;remote.base=base;remote.bytes=bytes;remote.imports.resize(t->paths.size());
  for (size_t i=0;i<t->paths.size();++i) {
    auto& path=t->paths[i];
    const auto key=kvaf::UrmaBindingKey::Make(kvaf::MemoryDirection::Get,
        path.device_context,path.context_generation,memory,hccp::MemImportFlags());
    if (!t->bindings.Acquire(key,[&](hccp::HccpRemoteMemoryBinding& binding) {
          return binding.Import(path.device_context,t->api.rmem_import,t->api.rmem_unimport,memory);
        },&remote.imports[i]).ok()) return KVAF_BATCH_GET_DIRECT_HCCP_ERROR;
  }
  remote.ready=true;return KVAF_BATCH_GET_SUCCESS;
}

extern "C" int32_t KvafBatchGetLocalBufferAcquireForRemote(KvafBatchGetTransport* t,
    uint64_t mr_id, void* output, uint64_t bytes, KvafBatchGetLocalBuffer** out, uint64_t* entity) {
  if (!t || !out || !entity) return KVAF_BATCH_GET_INVALID_ARGUMENT;
  auto remote=t->remote_memories.find(mr_id);
  if (remote==t->remote_memories.end() || !remote->second.ready) return KVAF_BATCH_GET_INVALID_ARGUMENT;
  auto status=KvafBatchGetLocalBufferAcquire(t,output,bytes,out,entity);
  if (status || mr_id==1) return status;
  auto* local=*out;
  auto existing=local->remote_entities.find(mr_id);
  if (existing!=local->remote_entities.end() && existing->second.ready) {
    *entity=reinterpret_cast<uint64_t>(existing->second.data);return KVAF_BATCH_GET_SUCCESS;
  }
  const auto& mr=remote->second;
  std::vector<KvafBatchGetDirectUrmaEntity> entities(
      t->entity_templates.begin(),t->entity_templates.begin()+t->worker_count);
  const auto eid=ReverseEidForAivWqe(mr.memory.wqe_eid);
  for (uint32_t i=0;i<t->worker_count;++i) {
    auto& e=entities[i];const auto& local_mr=local->path_registrations[t->workers[i].path_index].memory;
    e.remote_host_ddr_address=mr.base;e.remote_host_ddr_size=mr.bytes;
    e.registered_base=mr.memory.registered_address;
    e.remote_token_id=mr.memory.wqe_token_id();e.remote_token_value=mr.memory.token_value;
    std::memcpy(e.remote_eid,eid.data(),eid.size());
    e.local_hbm_address=local->output_hbm_address;e.local_hbm_size=local->output_hbm_bytes;
    e.local_token_id=local_mr.token_id;e.local_token_value=local_mr.token_value;
  }
  // Retain incomplete allocations as well. A failed upload is never exposed
  // to the kernel; a later call may finish it, or Close frees its storage.
  auto& view=local->remote_entities[mr_id];
  const auto n=t->worker_count*sizeof(entities.front());
  if ((!view.data && aclrtMalloc(&view.data,n,ACL_MEM_MALLOC_NORMAL_ONLY)!=ACL_SUCCESS) || !view.data ||
      aclrtMemcpy(view.data,n,entities.data(),n,ACL_MEMCPY_HOST_TO_DEVICE)!=ACL_SUCCESS) {
    KvafBatchGetLocalBufferRelease(local);*out=nullptr;*entity=0;
    return KVAF_BATCH_GET_RUNTIME_ERROR;
  }
  view.ready=true;*entity=reinterpret_cast<uint64_t>(view.data);
  return KVAF_BATCH_GET_SUCCESS;
}

extern "C" int32_t KvafGatherBatchGetTransportQuiesce(KvafBatchGetTransport *t) {
  if (!t) return KVAF_BATCH_GET_SUCCESS;
  t->closing = true;
  for (uint32_t i = 0; i < t->worker_count; ++i) {
    auto &w = t->workers[i];
    auto *context = t->paths[w.path_index].device_context;
    if (w.device_peer_target_jetty) {
      if (t->api.qp_unimport(context, w.device_peer_target_jetty)) return KVAF_BATCH_GET_DIRECT_HCCP_ERROR;
      w.device_peer_target_jetty = nullptr;
    }
    if (w.device_user_ctl_jetty.handle) {
      if (t->api.qp_destroy(w.device_user_ctl_jetty.handle)) return KVAF_BATCH_GET_DIRECT_HCCP_ERROR;
      w.device_user_ctl_jetty.handle = nullptr;
    }
  }
  return KVAF_BATCH_GET_SUCCESS;
}
