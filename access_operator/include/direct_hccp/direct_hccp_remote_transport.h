#pragma once

#include <array>
#include <cstdint>
#include <string>

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
  // Optional observability symbols.  Older CANN releases do not export
  // these, so their absence must not prevent data-plane initialization.
  GetDevEidInfoNumFn get_dev_eid_info_num{nullptr};
  GetDevEidInfoListFn get_dev_eid_info_list{nullptr};
};

constexpr std::uint32_t kUnknownEidIndex = UINT32_MAX;
constexpr std::uint32_t kUnknownPhysicalPort = UINT32_MAX;

// Identity of one local EID as reported by RA and /proc/asdrv_ub/pair_info.
// `physical_port` is the active pair_info chan_id; it is intentionally kept
// separate from func_id because the latter is a hardware function identifier,
// not a port number.
struct LocalEidIdentity {
  bool found{false};
  std::uint32_t eid_index{kUnknownEidIndex};
  std::string udma_device;
  std::uint32_t physical_port{kUnknownPhysicalPort};
  std::uint32_t die_id{0U};
  std::uint32_t chip_id{0U};
  std::uint32_t func_id{0U};
  std::uint32_t dev_feature{0U};
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
  void* device_user_ctl_cq{nullptr};
  CqInfo device_cq_info{};
  DeviceAivJetty device_user_ctl_jetty{};
  void* device_peer_target_jetty{nullptr};
  QpImportInfoT device_peer_import_info{};
};

struct DeviceAivTransportResources {
  DirectHccpApi* api{nullptr};
  DirectHccpSession* session{nullptr};
  void* device_context{nullptr};
  void* device_transport_token_handle{nullptr};
  void* device_peer_memory_import{nullptr};
  DeviceAivWorkerResources* workers{nullptr};
  std::uint32_t worker_count{0U};
};

// One peer channel is bound to one local AIV Jetty. Runtime has
// already selected local_eid; this adapter only creates the transport
// resources required by that selected Link.
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
};

struct PeerChannel {
  QpImportInfoT local_import_info{};
  void* local_target_jetty{nullptr};
};

bool LoadDirectHccp(DirectHccpApi* api);
void CloseDirectHccp(DirectHccpApi* api);

bool GetCurrentNpuPhysicalDeviceId(std::uint32_t* physical_device_id);

// Resolve a raw local EID to the provider's real EID index and device name.
// The port comes from the matching active pair_info chan_id when available.
// This helper is best-effort to preserve compatibility with older CANN builds.
bool ResolveLocalEidIdentity(
    const DirectHccpApi& api, std::uint32_t physical_device_id,
    const std::array<std::uint8_t, kEidBytes>& eid,
    LocalEidIdentity* identity);
void ReleaseDeviceAivTransport(const DeviceAivTransportResources& resources);

// Converts the Link descriptor's raw UMDK token to the AIV WQE token format.
bool LoadPeerMemory(PeerLink* link, std::uint64_t peer_base,
                    DirectMemory* memory,
                    std::array<std::uint8_t, kEidBytes>* peer_eid);

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

bool CreatePeerChannel(const PeerChannelRequest& request,
                       PeerChannel* channel);

}  // namespace kvaf::direct_hccp
