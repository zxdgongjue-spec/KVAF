#include "kvaf_aiv_batch_get.h"

#include <algorithm>
#include <array>
#include <cctype>
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
#include "peer_link.h"
#include "direct_hccp/direct_hccp_abi.h"
#include "direct_hccp/direct_hccp_remote_transport.h"
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

namespace {

// Parse once per transport, before allocating any device resources. Only an
// unset or empty value selects the default; malformed lists must fail setup.
bool ParseBatchGetJettyPriorities(const char *value,
                                  std::vector<uint8_t> *priorities) {
  priorities->clear();
  if (value == nullptr || *value == '\0') {
    priorities->push_back(15U);
    return true;
  }
  const char *cursor = value;
  while (true) {
    while (std::isspace(static_cast<unsigned char>(*cursor))) ++cursor;
    if (*cursor < '0' || *cursor > '9') return false;
    uint32_t priority = 0U;
    do {
      priority = priority * 10U + static_cast<uint32_t>(*cursor - '0');
      // Check each digit so even oversized input cannot overflow.
      if (priority > 15U) return false;
      ++cursor;
    } while (*cursor >= '0' && *cursor <= '9');
    while (std::isspace(static_cast<unsigned char>(*cursor))) ++cursor;
    priorities->push_back(static_cast<uint8_t>(priority));
    if (*cursor == '\0') return true;
    if (*cursor != ',') return false;
    ++cursor;
  }
}

}  // namespace

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
struct KvafBatchGetLocalBuffer {
  KvafBatchGetTransport *transport = nullptr;
  std::vector<LocalPathRegistration> path_registrations{};
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
  // Each local/remote EID pair owns an independent RaCtx, remote-memory
  // import and transport token. Workers are striped over these paths.
  std::vector<hccp::DeviceAivPathResources> paths{};
  std::vector<DirectMemory> peer_memories;
  std::vector<std::array<uint8_t, hccp::kEidBytes>> peer_eids;
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
  std::fprintf(stderr,
               "[KVAF_AIV] DIRECT_MEMORY_REGISTERED role=%s token_id=0x%x host_or_hbm_va=0x%llx bytes=%llu\n",
               role, memory->token_id, static_cast<unsigned long long>(aligned_address),
               static_cast<unsigned long long>(aligned_bytes));
  return true;
}

void ReleaseLocalBufferResources(KvafBatchGetTransport *transport, KvafBatchGetLocalBuffer *local) {
  if (transport == nullptr || local == nullptr) return;
  if (local->device_entities != nullptr) {
    (void)aclrtFree(local->device_entities);
    local->device_entities = nullptr;
    local->device_channel_entity = 0U;
  }
  for (std::size_t reverse = local->path_registrations.size(); reverse > 0U; --reverse) {
    const std::size_t path_index = reverse - 1U;
    if (path_index >= transport->paths.size()) continue;
    hccp::DeviceAivPathResources &path = transport->paths[path_index];
    LocalPathRegistration &registration = local->path_registrations[path_index];
    if (registration.memory.handle != nullptr && transport->api.lmem_unregister != nullptr &&
        path.device_context != nullptr) {
      (void)transport->api.lmem_unregister(path.device_context, registration.memory.handle);
      registration.memory.handle = nullptr;
    }
    if (registration.token_handle != nullptr && transport->api.token_free != nullptr &&
        path.device_context != nullptr) {
      (void)transport->api.token_free(path.device_context, registration.token_handle);
      registration.token_handle = nullptr;
    }
  }
  local->path_registrations.clear();
}

bool AcquireLocalBuffer(KvafBatchGetTransport *transport, void *output_hbm, uint64_t output_hbm_bytes,
                        KvafBatchGetLocalBuffer **local_buffer, uint64_t *device_channel_entity) {
  if (transport == nullptr || output_hbm == nullptr || output_hbm_bytes == 0U || local_buffer == nullptr ||
      device_channel_entity == nullptr || transport->paths.empty() || transport->worker_count == 0U ||
      transport->peer_memories.empty()) {
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
                     existing->path_registrations.front().memory.token_id,
                     existing->reference_count,
                     static_cast<unsigned long long>(existing->device_channel_entity));
      }
      return true;
    }
  }

  auto candidate = std::make_unique<KvafBatchGetLocalBuffer>();
  candidate->transport = transport;
  candidate->output_hbm_address = output_address;
  candidate->output_hbm_bytes = output_hbm_bytes;
  candidate->path_registrations.resize(transport->paths.size());
  for (std::size_t path_index = 0U; path_index < transport->paths.size(); ++path_index) {
    hccp::DeviceAivPathResources &path = transport->paths[path_index];
    LocalPathRegistration &registration = candidate->path_registrations[path_index];
    uint32_t token_id = 0U;
    const std::string token_role = "device_local_hbm_eid_path=" + std::to_string(path_index);
    const std::string memory_role = "device_hbm_eid_path=" + std::to_string(path_index);
    if (path.device_context == nullptr ||
        !AllocateToken(transport->api, path.device_context, token_role.c_str(),
                       &registration.token_handle, &token_id) ||
        !RegisterMemory(transport->api, path.device_context, registration.token_handle,
                        token_id, transport->peer_memories[path_index].token_value, output_hbm,
                        output_hbm_bytes, memory_role.c_str(), &registration.memory)) {
      ReleaseLocalBufferResources(transport, candidate.get());
      return false;
    }
  }

  std::array<KvafBatchGetDirectUrmaEntity, kMaxDirectAivWorkers> entities = transport->entity_templates;
  for (uint32_t index = 0U; index < transport->worker_count; ++index) {
    const std::uint32_t path_index = transport->workers[index].path_index;
    if (path_index >= candidate->path_registrations.size()) {
      ReleaseLocalBufferResources(transport, candidate.get());
      return false;
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
    ReleaseLocalBufferResources(transport, candidate.get());
    return false;
  }
  candidate->device_channel_entity = reinterpret_cast<uint64_t>(candidate->device_entities);
  candidate->reference_count = 1U;
  *local_buffer = candidate.get();
  *device_channel_entity = candidate->device_channel_entity;
  std::fprintf(stderr,
               "[KVAF_AIV] DIRECT_LOCAL_BUFFER_ACQUIRE_NEW output_hbm=0x%llx bytes=%llu token_id=0x%x refs=1 "
               "eid_paths=%zu entity_table=0x%llx transport_reused=1\n",
               static_cast<unsigned long long>(output_address), static_cast<unsigned long long>(output_hbm_bytes),
               candidate->path_registrations.front().memory.token_id,
               candidate->path_registrations.size(),
               static_cast<unsigned long long>(candidate->device_channel_entity));
  transport->local_buffers.push_back(std::move(candidate));
  return true;
}

bool ImportRemoteMemory(const DirectHccpApi &api, void *context, const DirectMemory &remote, void **import_handle) {
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

void ReleaseTransport(KvafBatchGetTransport *transport) {
  if (transport == nullptr) {
    return;
  }
  if (!transport->local_buffers.empty()) {
    std::fprintf(stderr, "[KVAF_AIV] DIRECT_HCCP_TEARDOWN phase=release_local_buffers count=%zu\n",
                 transport->local_buffers.size());
    for (const auto &local : transport->local_buffers) {
      if (local != nullptr) ReleaseLocalBufferResources(transport, local.get());
    }
  }
  hccp::ReleaseDeviceAivTransport(
      {&transport->api, transport->session, nullptr, nullptr, nullptr,
       transport->workers.data(), transport->worker_count,
       transport->paths.data(),
       static_cast<uint32_t>(transport->paths.size())});
  std::fprintf(stderr,
               "[KVAF_AIV] DIRECT_HCCP_TEARDOWN phase=delete_transport "
               "worker_storage=inline workers=%u\n",
               transport->worker_count);
  delete transport;
}

int32_t CreateBatchGetTransport(void *host_ddr, uint64_t host_ddr_bytes,
                                uint32_t aiv_core_count, uint32_t jettys_per_core,
                                void *peer_link,
                                const uint8_t *peer_local_eids,
                                uint32_t peer_local_eid_count,
                                KvafBatchGetTransport **transport) {
  const uint64_t entity_count_u64 =
      static_cast<uint64_t>(aiv_core_count) * jettys_per_core;
  if (host_ddr == nullptr || host_ddr_bytes == 0U || peer_link == nullptr || peer_local_eids == nullptr ||
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

  const char *priority_env = std::getenv("KVAF_BATCH_GET_JETTY_PRIORITIES");
  std::vector<uint8_t> priorities;
  if (!ParseBatchGetJettyPriorities(priority_env, &priorities)) {
    std::fprintf(stderr,
                 "[KVAF_AIV] BATCH_GET_JETTY_PRIORITY_CONFIG_INVALID "
                 "env=KVAF_BATCH_GET_JETTY_PRIORITIES "
                 "expected=comma_separated_decimal_integers_in_0_to_15\n");
    return KVAF_BATCH_GET_INVALID_ARGUMENT;
  }
  std::string priority_list;
  for (const uint8_t priority : priorities) {
    if (!priority_list.empty()) priority_list += ',';
    priority_list += std::to_string(priority);
  }
  std::fprintf(stderr,
               "[KVAF_AIV] BATCH_GET_JETTY_PRIORITY_CONFIG source=%s "
               "distribution=core_major_round_robin priorities=%s "
               "priority_count=%zu workers=%u unused_priorities=%zu\n",
               priority_env == nullptr || *priority_env == '\0' ? "default" : "environment",
               priority_list.c_str(), priorities.size(), entity_count,
               priorities.size() > entity_count ? priorities.size() - entity_count : size_t{0});

  auto *state = new (std::nothrow) KvafBatchGetTransport();
  if (state == nullptr) {
    return KVAF_BATCH_GET_RUNTIME_ERROR;
  }
  uint32_t physical_device_id = 0U;
  // Keep this before the first `goto failed`: C++ forbids jumping over even
  // a scalar initialization into the common cleanup label below.
  int32_t result = KVAF_BATCH_GET_DIRECT_HCCP_ERROR;
  state->peer_link = static_cast<hccp::PeerLink *>(peer_link);
  const uint32_t remote_path_count = state->peer_link->RegionPathCount(
      reinterpret_cast<uint64_t>(host_ddr));
  const uint32_t path_count = std::max(remote_path_count, peer_local_eid_count);
  // Multi-peer mode uses one NPU EID and covers every peer across all workers.
  if (remote_path_count == 0U ||
      (remote_path_count > 1U &&
        (peer_local_eid_count != 1U || remote_path_count > entity_count))) {
    delete state;
    return KVAF_BATCH_GET_INVALID_ARGUMENT;
  }
  state->paths.resize(path_count);
  state->peer_memories.resize(path_count);
  state->peer_eids.resize(path_count);
  for (uint32_t path_index = 0U; path_index < path_count; ++path_index) {
    std::memcpy(state->paths[path_index].local_eid.data(),
                peer_local_eids + static_cast<size_t>(path_index % peer_local_eid_count) * hccp::kEidBytes,
                hccp::kEidBytes);
  }

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
    if (!hccp::LoadPeerMemory(state->peer_link, reinterpret_cast<uint64_t>(host_ddr),
                              &state->peer_memories[path_index], &state->peer_eids[path_index],
                              path_index % remote_path_count)) goto failed;
    hccp::DeviceAivPathResources &path = state->paths[path_index];
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
        !ImportRemoteMemory(state->api, path.device_context, state->peer_memories[path_index],
                            &path.device_peer_memory_import)) {
      goto failed;
    }
  }
  state->worker_count = entity_count;
  for (uint32_t entity_index = 0U; entity_index < entity_count; ++entity_index) {
    const uint32_t core = entity_index / jettys_per_core;
    const uint32_t jetty_slot = entity_index % jettys_per_core;
    const uint8_t priority = priorities[entity_index % priorities.size()];
    const uint32_t path_index = remote_path_count > 1U
        ? hccp::PeerEidPathIndexForWorker(core, jetty_slot, aiv_core_count,
                                        jettys_per_core, remote_path_count)
        : hccp::EidPathIndexForWorker(core, jetty_slot, aiv_core_count, path_count);
    hccp::DeviceAivPathResources &path = state->paths[path_index];
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
                           path.device_transport_token_handle, state->peer_memories[path_index].token_value,
                           kDeviceUserCtlSqDepth, device_jetty_role.c_str(),
                           &worker.device_user_ctl_jetty.info, &worker.device_user_ctl_jetty.handle,
                           priority)) {
      goto failed;
    }
    std::fprintf(stderr,
                 "[KVAF_AIV] BATCH_GET_JETTY_PRIORITY op=batch_get core=%u "
                 "jetty_slot=%u entity_index=%u eid_path=%u jetty_id=%u priority=%u\n",
                 core, jetty_slot, entity_index, path_index,
                 worker.device_user_ctl_jetty.info.ub.id, static_cast<unsigned int>(priority));
    const std::string device_ctp_role = "device_aiv_peer_link_" + suffix;
    hccp::PeerChannel channel{};
    if (!hccp::CreatePeerChannel(
            {state->peer_link, &state->api, path.device_context,
             physical_device_id, reinterpret_cast<uint64_t>(host_ddr),
             &state->peer_memories[path_index], &path.local_eid, &state->peer_eids[path_index],
             &worker.device_user_ctl_jetty.info, device_ctp_role.c_str(),
             path_index % remote_path_count},
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
      const DirectAivWorker &worker = state->workers[entity_index];
      KvafBatchGetDirectUrmaEntity &entity = entities[entity_index];
      entity.magic = kDirectEntityMagic;
      entity.version = kDirectEntityVersion;
      entity.remote_host_ddr_address = reinterpret_cast<uint64_t>(host_ddr);
      entity.remote_host_ddr_size = host_ddr_bytes;
      entity.remote_token_id = state->peer_memories[path_index].token_id;
      entity.remote_token_value = state->peer_memories[path_index].token_value;
      entity.sq_va = worker.device_user_ctl_jetty.info.ub.sq_buffer_va;
      entity.sq_db_va = worker.device_user_ctl_jetty.info.ub.db_addr;
      entity.jetty_id = worker.device_user_ctl_jetty.info.ub.id;
      entity.priority = priorities[entity_index % priorities.size()];
      entity.eid_path = path_index;
      // AIV indexes the USER_CTL SQ in 64-byte WQEBBs.  The HCCL AIV reference
      // exposes this context depth as sqDepth * WQE_NUM_PER_SQE, matching the
      // four-WQEBB storage allocation made in CreateJetty.
      entity.sq_depth = kDeviceUserCtlSqDepth * kWqebbPerWqe;
      entity.tp_id = worker.device_peer_import_info.out.ub.tpn;
      entity.wqe_size = kWqeBytes;
      const auto wqe_remote_eid = ReverseEidForAivWqe(state->peer_eids[path_index]);
      std::memcpy(entity.remote_eid, wqe_remote_eid.data(), sizeof(entity.remote_eid));
      std::fprintf(stderr, "[KVAF_AIV] BATCH_GET_PEER_ROUTE core=%u jetty_slot=%u "
                   "eid_path=%u peer_eid=%s remote_token_id=0x%x\n",
                   core, jetty_slot, path_index, EidToHex(state->peer_eids[path_index]).c_str(),
                   entity.remote_token_id);
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
               "eid_paths=%u remote_paths=%u distribution=%s "
               "local_hbm=deferred_acquire\n",
               static_cast<unsigned long long>(host_ddr_bytes), entity_count,
               path_count, remote_path_count,
               remote_path_count > 1U && jettys_per_core >= remote_path_count
                   ? "per_core_peer_round_robin" : "jetty_major_worker_round_robin");
  *transport = state;
  return KVAF_BATCH_GET_SUCCESS;

failed:
  ReleaseTransport(state);
  return result;
}

extern "C" int32_t KvafBatchGetTransportCreateWithPeerLink(
    void *host_ddr, uint64_t host_ddr_bytes, uint32_t aiv_core_count,
    uint32_t jettys_per_core, void *peer_link, const uint8_t *peer_local_eid,
    KvafBatchGetTransport **transport) {
  if (peer_link == nullptr || peer_local_eid == nullptr) return KVAF_BATCH_GET_INVALID_ARGUMENT;
  return CreateBatchGetTransport(host_ddr, host_ddr_bytes, aiv_core_count,
                                 jettys_per_core, peer_link, peer_local_eid, 1U,
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
  return CreateBatchGetTransport(host_ddr, host_ddr_bytes, aiv_core_count,
                                 jettys_per_core, peer_link, peer_local_eids,
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
