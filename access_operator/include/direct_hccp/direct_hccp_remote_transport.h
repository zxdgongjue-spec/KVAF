#pragma once

#include <array>
#include <cstdint>

#include "direct_hccp/direct_hccp_abi.h"

namespace kvaf::peer {
class PeerLink;
}

namespace kvaf::direct_hccp {

using PeerLink = peer::PeerLink;
struct DirectHccpSession;

// The RA ABI loader and the PeerLink adapters are shared by BatchGet and
// BatchPut.  They deliberately contain no queue, context, or Jetty ownership:
// those resources belong to the individual operator transport invocation.
struct DirectHccpApi {
  void* library{nullptr};
  CtxInitFn ctx_init{nullptr};
  CtxDeinitFn ctx_deinit{nullptr};
  TokenAllocFn token_alloc{nullptr};
  TokenFreeFn token_free{nullptr};
  LmemRegisterFn lmem_register{nullptr};
  LmemUnregisterFn lmem_unregister{nullptr};
  RmemImportFn rmem_import{nullptr};
  RmemUnimportFn rmem_unimport{nullptr};
  CqCreateFn cq_create{nullptr};
  CqDestroyFn cq_destroy{nullptr};
  QpCreateFn qp_create{nullptr};
  QpDestroyFn qp_destroy{nullptr};
  QpImportFn qp_import{nullptr};
  QpUnimportFn qp_unimport{nullptr};
  GetTpInfoFn get_tp_info{nullptr};
  GetTpInfoAsyncFn get_tp_info_async{nullptr};
  AsyncReqResultFn get_async_req_result{nullptr};
  GetRandomFn get_random{nullptr};
};

struct DirectMemory {
  void* handle{nullptr};
  std::uint64_t registered_address{0};
  std::uint64_t registered_bytes{0};
  std::uint32_t token_id{0};
  std::uint32_t token_value{0};
  std::uint8_t key[kMemKeyBytes]{};
  std::uint32_t key_bytes{0};
};

// NPU-side resources shared by the BatchGet and BatchPut transport teardown.
struct DeviceAivJetty {
  void* handle{nullptr};
  QpCreateInfo info{};
};

struct DeviceAivWorkerResources {
  // Index into the owning transport's EID-path array. Queue and imported
  // Jetty handles must always be released through that path's RaCtx.
  std::uint32_t path_index{0U};
  void* device_user_ctl_cq{nullptr};
  CqInfo device_cq_info{};
  DeviceAivJetty device_user_ctl_jetty{};
  void* device_peer_target_jetty{nullptr};
  QpImportInfoT device_peer_import_info{};
};

// One Direct-HCCP RaCtx per selected local EID. Remote DDR and transport
// tokens are context-owned, so a multi-EID transport cannot share these
// handles even when every path targets the same CPU DDR region.
struct DeviceAivPathResources {
  std::array<std::uint8_t, kEidBytes> local_eid{};
  void* device_context{nullptr};
  void* device_transport_token_handle{nullptr};
  void* device_peer_memory_import{nullptr};
};

struct DeviceAivTransportResources {
  DirectHccpApi* api{nullptr};
  DirectHccpSession* session{nullptr};
  // Legacy single-context fields are retained so BatchPut remains unchanged.
  void* device_context{nullptr};
  void* device_transport_token_handle{nullptr};
  void* device_peer_memory_import{nullptr};
  DeviceAivWorkerResources* workers{nullptr};
  std::uint32_t worker_count{0U};
  // BatchGet uses these fields when more than one local EID is configured.
  DeviceAivPathResources* paths{nullptr};
  std::uint32_t path_count{0U};
};

// Queue arrays remain core-major for the AIV ABI, but path assignment is
// jetty-major so every core's always-used primary Jetty is spread before any
// secondary Jetty. This also covers every path when path_count does not
// exceed the total worker count.
inline constexpr std::uint32_t EidPathIndexForWorker(
    std::uint32_t core, std::uint32_t jetty_slot,
    std::uint32_t core_count, std::uint32_t path_count) {
  return path_count == 0U
             ? 0U
             : (jetty_slot * core_count + core) % path_count;
}

static_assert(EidPathIndexForWorker(0U, 0U, 2U, 2U) == 0U);
static_assert(EidPathIndexForWorker(1U, 0U, 2U, 2U) == 1U);
static_assert(EidPathIndexForWorker(0U, 1U, 2U, 3U) == 2U);
static_assert(EidPathIndexForWorker(0U, 2U, 1U, 4U) == 2U);
static_assert(EidPathIndexForWorker(0U, 3U, 1U, 4U) == 3U);
static_assert(EidPathIndexForWorker(0U, 4U, 1U, 8U) == 4U);
static_assert(EidPathIndexForWorker(0U, 7U, 1U, 8U) == 7U);
static_assert(EidPathIndexForWorker(0U, 8U, 1U, 16U) == 8U);
static_assert(EidPathIndexForWorker(0U, 15U, 1U, 16U) == 15U);
static_assert(EidPathIndexForWorker(0U, 32U, 1U, 64U) == 32U);
static_assert(EidPathIndexForWorker(0U, 63U, 1U, 64U) == 63U);

// One peer channel is bound to one local AIV Jetty. Runtime has already
// selected the local EID assigned to this worker; this adapter only creates
// the transport resources required by that selected path.
struct PeerChannelRequest {
  PeerLink* link{nullptr};
  const DirectHccpApi* api{nullptr};
  void* device_context{nullptr};
  std::uint32_t physical_device_id{0};
  std::uint64_t peer_base{0};
  const DirectMemory* peer_memory{nullptr};
  const std::array<std::uint8_t, kEidBytes>* local_eid{nullptr};
  const std::array<std::uint8_t, kEidBytes>* peer_eid{nullptr};
  const QpCreateInfo* local_jetty_info{nullptr};
  const char* role{nullptr};
  std::uint32_t peer_path{0U};
};

struct PeerChannel {
  QpImportInfoT local_import_info{};
  void* local_target_jetty{nullptr};
};

bool LoadDirectHccp(DirectHccpApi* api);
void CloseDirectHccp(DirectHccpApi* api);

bool GetCurrentNpuPhysicalDeviceId(std::uint32_t* physical_device_id);
void ReleaseDeviceAivTransport(const DeviceAivTransportResources& resources);

// Converts the Link descriptor's raw UMDK token to the AIV WQE token format.
bool LoadPeerMemory(PeerLink* link, std::uint64_t peer_base,
                    DirectMemory* memory,
                    std::array<std::uint8_t, kEidBytes>* peer_eid,
                    std::uint32_t peer_path = 0U);

bool ImportPeerJetty(const DirectHccpApi& api, void* context,
                     const std::uint8_t* peer_key, std::uint32_t peer_key_bytes,
                     std::uint32_t peer_token_value, std::uint64_t local_tp,
                     std::uint64_t peer_tp, std::uint32_t local_psn,
                     std::uint32_t peer_psn, const char* role,
                     QpImportInfoT* import_info, void** target_jetty);

// Creates the one USER_CTL queue contract consumed by an AIV kernel. The
// caller owns the returned Jetty and destroys it with DirectHccpApi::qp_destroy.
bool CreateDeviceAivJetty(const DirectHccpApi& api, void* context,
                          void* cq_handle, void* token_handle,
                          std::uint32_t token_value, std::uint32_t sq_depth,
                          const char* role, QpCreateInfo* jetty_info,
                          void** jetty_handle);

// Explicit-priority variant. The original entry point retains priority 15.
bool CreateDeviceAivJetty(const DirectHccpApi& api, void* context,
                          void* cq_handle, void* token_handle,
                          std::uint32_t token_value, std::uint32_t sq_depth,
                          const char* role, QpCreateInfo* jetty_info,
                          void** jetty_handle, std::uint8_t priority);

bool CreatePeerChannel(const PeerChannelRequest& request,
                       PeerChannel* channel);

}  // namespace kvaf::direct_hccp
