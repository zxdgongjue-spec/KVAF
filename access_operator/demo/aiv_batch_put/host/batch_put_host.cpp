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

#include "acl/acl.h"
#include "batch_put_internal.h"
#include "direct_hccp/direct_hccp_abi.h"
#include "direct_hccp/direct_hccp_remote_transport.h"
#include "direct_hccp/direct_hccp_session.h"
#if defined(KVAF_ENABLE_AICPU_URMA)
#include "host.h"
#endif

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
  void *device_entities = nullptr;
  void *device_queue_state = nullptr;
  std::vector<std::unique_ptr<KvafBatchPutLocalBuffer>> local_buffers{};
  std::mutex mutex{};
  std::mutex launch_mutex{};
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
  hccp::VerboseLog(
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
  hccp::VerboseLog(
      "[KVAF_AIV] DIRECT_HCCP_CONTEXT_READY role=%s mode=%d physical=%u context=%p\n",
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
  hccp::VerboseLog("[KVAF_AIV] DIRECT_TOKEN_READY role=%s token_id=0x%x\n", role,
                   *token_id);
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
  hccp::VerboseLog(
      "[KVAF_AIV] DIRECT_MEMORY_REGISTERED role=%s token_id=0x%x host_or_hbm_va=0x%llx bytes=%llu\n",
      role, memory->token_id, static_cast<unsigned long long>(aligned_address),
      static_cast<unsigned long long>(aligned_bytes));
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
  hccp::VerboseLog("[KVAF_AIV] DIRECT_REMOTE_DDR_IMPORTED target_seg=0x%llx\n",
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
  hccp::VerboseLog(
      "[KVAF_AIV] DIRECT_CQ_READY role=%s mode=%u jfc_id=%u cq_va=0x%llx cqe_size=%u db_va=0x%llx\n",
      role, mode, info->out.id, static_cast<unsigned long long>(info->out.buffer_addr),
      info->out.cqe_size, static_cast<unsigned long long>(info->out.swdb_addr));
  return true;
}

void ReleaseTransport(KvafBatchPutTransport *transport) {
  if (transport == nullptr) {
    return;
  }
  for (const auto &local : transport->local_buffers) {
    if (transport->api.lmem_unregister != nullptr && transport->device_context != nullptr &&
        local->memory.handle != nullptr) {
      (void)transport->api.lmem_unregister(transport->device_context, local->memory.handle);
    }
    if (transport->api.token_free != nullptr && transport->device_context != nullptr &&
        local->token_handle != nullptr) {
      (void)transport->api.token_free(transport->device_context, local->token_handle);
    }
  }
  transport->local_buffers.clear();
  if (transport->device_entities != nullptr) {
    (void)aclrtFree(transport->device_entities);
    transport->device_entities = nullptr;
  }
  if (transport->device_queue_state != nullptr) {
    (void)aclrtFree(transport->device_queue_state);
    transport->device_queue_state = nullptr;
  }
  hccp::ReleaseDeviceAivTransport({&transport->api, transport->session, transport->device_context,
                                   transport->device_transport_token_handle,
                                   transport->device_peer_memory_import, transport->workers.data(),
                                   transport->worker_count});
  delete transport;
}

int32_t CreateBatchPutTransport(void *host_ddr, uint64_t host_ddr_bytes,
                                uint32_t aiv_core_count, uint32_t jettys_per_core,
                                void *peer_link,
                                const uint8_t *peer_local_eid,
                                KvafBatchPutTransport **transport) {
  if (host_ddr == nullptr || host_ddr_bytes == 0U || peer_link == nullptr || peer_local_eid == nullptr ||
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
  const uint64_t entity_bytes =
      static_cast<uint64_t>(state->worker_count) *
      sizeof(KvafBatchPutDirectUrmaEntity);

  if (!hccp::GetCurrentNpuPhysicalDeviceId(&physical_device_id)) {
    std::fprintf(stderr, "[KVAF_AIV] unable to resolve physical device id from selected ACL device\n");
    result = KVAF_BATCH_PUT_EID_ERROR;
    goto failed;
  }
  state->physical_device_id = physical_device_id;
  std::memcpy(device_eid.data(), peer_local_eid, device_eid.size());
  hccp::VerboseLog(
      "[KVAF_AIV] PEER_LINK_LOCAL_EID_READY physical_device=%u eid=%s\n",
      physical_device_id, EidToHex(device_eid).c_str());
  if (!hccp::LoadPeerMemory(state->peer_link, state->peer_memory_address,
                            &state->peer_memory, &peer_eid)) {
    goto failed;
  }
  state->wqe_remote_eid = ReverseEidForAivWqe(peer_eid);
  if (!hccp::LoadDirectHccp(&state->api)) goto failed;
  if (!hccp::AcquireDirectHccpSession(physical_device_id, kPidHdcType, &state->session) ||
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
    if (!hccp::CreatePeerChannel(
            {state->peer_link, &state->api, state->device_context,
             physical_device_id, state->peer_memory_address, &state->peer_memory,
             &device_eid, &peer_eid, &worker.device_user_ctl_jetty.info,
             device_ctp_role.c_str()},
            &channel)) {
      goto failed;
    }
    worker.device_peer_import_info = channel.local_import_info;
    worker.device_peer_target_jetty = channel.local_target_jetty;
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
  if (aclrtMalloc(&state->device_entities, entity_bytes,
                  ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS ||
      state->device_entities == nullptr ||
      aclrtMemcpy(state->device_entities, entity_bytes,
                  state->entity_templates.data(), entity_bytes,
                  ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS ||
      aclrtMalloc(&state->device_queue_state, queue_bytes, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS ||
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
  ReleaseTransport(state);
  return result;
}

extern "C" int32_t KvafBatchPutTransportCreateWithPeerLink(
    void *host_ddr, uint64_t host_ddr_bytes, uint32_t aiv_core_count,
    uint32_t jettys_per_core, void *peer_link, const uint8_t *peer_local_eid,
    KvafBatchPutTransport **transport) {
  if (peer_link == nullptr || peer_local_eid == nullptr) return KVAF_BATCH_PUT_INVALID_ARGUMENT;
  return CreateBatchPutTransport(host_ddr, host_ddr_bytes, aiv_core_count,
                                 jettys_per_core, peer_link, peer_local_eid, transport);
}

extern "C" void KvafBatchPutTransportDestroy(KvafBatchPutTransport *transport) {
  ReleaseTransport(transport);
}

extern "C" int32_t KvafBatchPutLocalBufferAcquire(KvafBatchPutTransport *transport,
                                                    void *input_device, uint64_t input_device_bytes,
                                                    KvafBatchPutLocalBuffer **local_buffer) {
  if (transport == nullptr || input_device == nullptr || input_device_bytes == 0U || local_buffer == nullptr) {
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
  uint32_t token_id = 0U;
  if (!AllocateToken(transport->api, transport->device_context, "device_local_hbm",
                     &candidate->token_handle, &token_id) ||
      !RegisterMemory(transport->api, transport->device_context, candidate->token_handle,
                      token_id, transport->device_transport_token_value, input_device,
                      input_device_bytes, "device_hbm", &candidate->memory)) {
    if (candidate->memory.handle != nullptr && transport->api.lmem_unregister != nullptr) {
      (void)transport->api.lmem_unregister(transport->device_context, candidate->memory.handle);
    }
    if (candidate->token_handle != nullptr && transport->api.token_free != nullptr) {
      (void)transport->api.token_free(transport->device_context, candidate->token_handle);
    }
    return KVAF_BATCH_PUT_DIRECT_HCCP_ERROR;
  }

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

extern "C" int32_t KvafBatchPutLocalBufferDescribe(
    const KvafBatchPutLocalBuffer *local_buffer,
    KvafBatchPutSourceDescriptor *descriptor) {
  if (local_buffer == nullptr || local_buffer->reference_count == 0U ||
      local_buffer->transport == nullptr || descriptor == nullptr ||
      local_buffer->input_hbm_address == 0U ||
      local_buffer->memory.token_id == 0U) {
    return KVAF_BATCH_PUT_INVALID_ARGUMENT;
  }
  descriptor->base_address = local_buffer->input_hbm_address;
  descriptor->local_token_id = local_buffer->memory.token_id;
  descriptor->reserved = 0U;
  return KVAF_BATCH_PUT_SUCCESS;
}

static int32_t ExecuteBatchPutSync(KvafBatchPutTransport *transport,
                                            const KvafExecuteBatchPutKernelArgs *arguments,
                                            aclrtStream stream,
                                            KvafBatchPutCoreStatus *core_status,
                                            uint32_t core_status_count, bool aicpu,
                                            uint32_t source_count) {
  if (transport == nullptr || arguments == nullptr || stream == nullptr || core_status == nullptr ||
      arguments->source_descriptors == 0U || arguments->source_indices == 0U ||
      arguments->input_offsets == 0U || arguments->lengths == 0U ||
      arguments->destination_addresses == 0U || arguments->core_status == 0U ||
      arguments->row_count == 0U ||
      arguments->aiv_core_count == 0U || arguments->aiv_core_count != core_status_count ||
      arguments->aiv_core_count != transport->aiv_core_count ||
      transport->device_entities == nullptr) {
    return KVAF_BATCH_PUT_INVALID_ARGUMENT;
  }

  std::lock_guard<std::mutex> launch_lock(transport->launch_mutex);
  const uint64_t status_bytes = static_cast<uint64_t>(core_status_count) * sizeof(*core_status);
  const aclError clear_result =
      aclrtMemset(reinterpret_cast<void *>(arguments->core_status), status_bytes, 0, status_bytes);
  if (clear_result != ACL_SUCCESS) {
    std::fprintf(stderr, "[KVAF_AIV] DIRECT_BATCH_PUT_RUNTIME_ERROR phase=clear_status acl=%d\n",
                 static_cast<int>(clear_result));
    return KVAF_BATCH_PUT_RUNTIME_ERROR;
  }
  KvafExecuteBatchPutLaunchArgs launch_args{};
  launch_args.request = *arguments;
  launch_args.channel_entity = reinterpret_cast<uint64_t>(transport->device_entities);
  launch_args.queue_state = reinterpret_cast<uint64_t>(transport->device_queue_state);
  launch_args.jettys_per_core = transport->jettys_per_core;
  launch_args.reserved0 = source_count;
  int32_t result = KVAF_BATCH_PUT_RUNTIME_ERROR;
#if defined(KVAF_ENABLE_AICPU_URMA)
  if (aicpu) {
    result = kvaf::aicpu_urma::LaunchPut(&launch_args, stream) == 0
                 ? KVAF_BATCH_PUT_SUCCESS : KVAF_BATCH_PUT_RUNTIME_ERROR;
  } else
#endif
  {
    result = KvafExecuteBatchPutLaunchInternal(&launch_args, stream);
  }
  if (result != KVAF_BATCH_PUT_SUCCESS) {
    std::fprintf(stderr, "[KVAF_AIV] DIRECT_BATCH_PUT_RUNTIME_ERROR phase=launch result=%d\n", result);
    return KVAF_BATCH_PUT_RUNTIME_ERROR;
  }
  const aclError synchronize_result = aclrtSynchronizeStream(stream);
  if (synchronize_result != ACL_SUCCESS) {
    std::fprintf(stderr, "[KVAF_AIV] DIRECT_BATCH_PUT_RUNTIME_ERROR phase=synchronize acl=%d\n",
                 static_cast<int>(synchronize_result));
    return KVAF_BATCH_PUT_RUNTIME_ERROR;
  }
  const aclError status_copy_result =
      aclrtMemcpy(core_status, status_bytes, reinterpret_cast<void *>(arguments->core_status), status_bytes,
                  ACL_MEMCPY_DEVICE_TO_HOST);
  if (status_copy_result != ACL_SUCCESS) {
    std::fprintf(stderr, "[KVAF_AIV] DIRECT_BATCH_PUT_RUNTIME_ERROR phase=copy_status acl=%d\n",
                 static_cast<int>(status_copy_result));
    return KVAF_BATCH_PUT_RUNTIME_ERROR;
  }
  for (uint32_t core = 0U; core < core_status_count; ++core) {
    const uint64_t begin = static_cast<uint64_t>(arguments->row_count) * core / core_status_count;
    const uint64_t end = static_cast<uint64_t>(arguments->row_count) * (core + 1U) / core_status_count;
    if (core_status[core].magic != KVAF_AIV_BATCH_PUT_STATUS_MAGIC ||
        core_status[core].result != KVAF_BATCH_PUT_CORE_SUCCESS ||
        core_status[core].processed_request_count != end - begin ||
        core_status[core].failed_request_index != UINT64_MAX) {
      return KVAF_BATCH_PUT_KERNEL_ERROR;
    }
  }
  return KVAF_BATCH_PUT_SUCCESS;
}

extern "C" int32_t KvafExecuteBatchPutSync(KvafBatchPutTransport *transport,
                                             const KvafExecuteBatchPutKernelArgs *arguments,
                                             aclrtStream stream,
                                             KvafBatchPutCoreStatus *core_status,
                                             uint32_t core_status_count) {
  return ExecuteBatchPutSync(transport, arguments, stream, core_status, core_status_count, false, 0U);
}

extern "C" int32_t KvafExecuteBatchPutAicpuSync(KvafBatchPutTransport *transport,
                                                  const KvafExecuteBatchPutKernelArgs *arguments,
                                                  aclrtStream stream,
                                                  KvafBatchPutCoreStatus *core_status,
                                                  uint32_t core_status_count,
                                                  uint32_t source_count) {
#if defined(KVAF_ENABLE_AICPU_URMA)
  if (arguments == nullptr || source_count == 0U || source_count > arguments->row_count)
    return KVAF_BATCH_PUT_INVALID_ARGUMENT;
  return ExecuteBatchPutSync(transport, arguments, stream, core_status, core_status_count, true,
                             source_count);
#else
  (void)transport; (void)arguments; (void)stream; (void)core_status;
  (void)core_status_count; (void)source_count;
  return KVAF_BATCH_PUT_RUNTIME_ERROR;
#endif
}
