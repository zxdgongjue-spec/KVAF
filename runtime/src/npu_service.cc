#include "npu_service.h"

#include <acl/acl.h>
#include <urma_api.h>
#include <dlfcn.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "direct_hccp/direct_hccp_remote_transport.h"
#include "direct_hccp/direct_hccp_session.h"
#include "peer_protocol.h"

namespace kvaf {
namespace {

using namespace peer;
using direct_hccp::DirectHccpApi;
using direct_hccp::DirectHccpSession;

constexpr std::uint32_t kHdcType = 18U;
constexpr std::uint32_t kCqDepth = 16384U;
constexpr std::uint32_t kSqDepth = 256U;
constexpr std::uint32_t kRqDepth = 256U;

bool ReadAll(int fd, void* data, std::size_t bytes) {
  auto* p = static_cast<std::uint8_t*>(data);
  for (std::size_t done = 0; done < bytes;) {
    const ssize_t n = recv(fd, p + done, bytes - done, 0);
    if (n <= 0) return false;
    done += static_cast<std::size_t>(n);
  }
  return true;
}

bool WriteAll(int fd, const void* data, std::size_t bytes) {
  const auto* p = static_cast<const std::uint8_t*>(data);
  for (std::size_t done = 0; done < bytes;) {
    const ssize_t n = send(fd, p + done, bytes - done, MSG_NOSIGNAL);
    if (n <= 0) return false;
    done += static_cast<std::size_t>(n);
  }
  return true;
}

void SendStatus(int fd, Command command, Result result) {
  StatusResponse response{};
  response.header = MakeHeader(command, sizeof(response));
  response.header.command = ResponseCommand(command);
  response.result = static_cast<std::int32_t>(result);
  (void)WriteAll(fd, &response, sizeof(response));
}

struct HbmRegion {
  std::uint64_t session_id{0U};
  void* memory{nullptr};
  std::uint64_t bytes{0U};
  void* lmem_handle{nullptr};
  std::uint32_t token_id{0U};
  std::uint32_t token_value{0U};
  std::array<std::uint8_t, kSegKeyBytes> segment_key{};
  std::uint32_t segment_key_bytes{0U};
  std::array<std::uint8_t, kQpKeyBytes> jetty_key{};
  std::array<std::uint8_t, kEidBytes> eid{};
  std::uint32_t jetty_key_bytes{0U};
  void* cq_handle{nullptr};
  void* qp_handle{nullptr};
  direct_hccp::CqInfo cq_info{};
  direct_hccp::QpCreateInfo qp_info{};
  void* peer_target_jetty{nullptr};
};

// RA exposes the NPU EID list through symbols that are not present in the
// public Direct-HCCP header. The first field is the same 16-byte EID used by
// CtxInitAttr; the remaining provider metadata is intentionally opaque.
struct RaEidInfo {
  direct_hccp::Eid eid{};
  std::uint32_t eid_index{0U};
  std::uint32_t reserved{0U};
};
struct RaInfo {
  std::int32_t mode{0};
  std::uint32_t phy_id{0};
};
using GetEidNumFn = int (*)(RaInfo, std::uint32_t*);
using GetEidListFn = int (*)(RaInfo, RaEidInfo*, std::uint32_t*);

}  // namespace

struct NpuService::Impl {
  ~Impl() { (void)Shutdown(); }

  Status Initialize(const NpuServiceOptions& options) {
    if (initialized) return Status::InvalidState("NpuService already initialized");
    if (aclrtSetDevice(options.device_id) != ACL_SUCCESS)
      return Status::Internal("aclrtSetDevice failed");
    if (!direct_hccp::LoadDirectHccp(&api))
      return Status::NotImplemented("Direct-HCCP RA symbols are unavailable");
    if (!direct_hccp::GetCurrentNpuPhysicalDeviceId(&physical_device_id))
      return Status::Internal("cannot resolve NPU physical device id");
    if (!direct_hccp::AcquireDirectHccpSession(physical_device_id, kHdcType, &session))
      return Status::Internal("failed to acquire Direct-HCCP session");

    direct_hccp::CtxInitCfg cfg{};
    cfg.mode = direct_hccp::kOfflineHdc;
    direct_hccp::CtxInitAttr attr{};
    attr.phy_id = physical_device_id;
    attr.ub.eid_index = options.eid_index;
    auto get_eid_num = reinterpret_cast<GetEidNumFn>(dlsym(api.library, "RaGetDevEidInfoNum"));
    auto get_eid_list = reinterpret_cast<GetEidListFn>(dlsym(api.library, "RaGetDevEidInfoList"));
    if (get_eid_num == nullptr || get_eid_list == nullptr)
      return Status::NotImplemented("Direct-HCCP EID discovery symbols are unavailable");
    RaInfo ra_info{static_cast<std::int32_t>(direct_hccp::kOfflineHdc), physical_device_id};
    std::uint32_t eid_count = 0U;
    if (get_eid_num(ra_info, &eid_count) != 0 || eid_count == 0U)
      return Status::NotFound("NPU service found no HDC EID candidates");
    std::vector<RaEidInfo> eid_list(eid_count);
    if (get_eid_list(ra_info, eid_list.data(), &eid_count) != 0 ||
        options.eid_index >= eid_count)
      return Status::InvalidArgument("NPU service eid-index is outside the discovered EID list");
    std::memcpy(attr.ub.eid.raw, eid_list[options.eid_index].eid.raw,
                sizeof(attr.ub.eid.raw));
    if (api.ctx_init(&cfg, &attr, &context) != 0 || context == nullptr)
      return Status::Internal("RaCtxInit failed");
    initialized = true;
    return Status::OK();
  }

  Status Shutdown() {
    for (auto& entry : regions) Release(entry.second.get());
    regions.clear();
    if (context != nullptr && api.ctx_deinit != nullptr) {
      (void)api.ctx_deinit(context);
      context = nullptr;
    }
    if (session != nullptr) {
      direct_hccp::ReleaseDirectHccpSession(session);
      session = nullptr;
    }
    if (api.library != nullptr) direct_hccp::CloseDirectHccp(&api);
    initialized = false;
    return Status::OK();
  }

  Result Allocate(const AllocateHbmRegionRequest& request,
                  AllocateHbmRegionResponse* response) {
    if (!initialized || response == nullptr || request.hbm_bytes == 0U ||
        request.slot_size == 0U || request.hbm_bytes % request.slot_size != 0U)
      return Result::kInvalidRequest;
    auto region = std::make_unique<HbmRegion>();
    region->bytes = request.hbm_bytes;
    if (aclrtMalloc(&region->memory, request.hbm_bytes, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
      return Result::kResourceError;
    direct_hccp::MrRegInfo reg{};
    reg.in.mem.addr = reinterpret_cast<std::uint64_t>(region->memory);
    reg.in.mem.size = request.hbm_bytes;
    reg.in.ub.flags = direct_hccp::MemRegFlagsDeviceMemory(true);
    reg.in.ub.token_value = 0xABCFU;
    if (api.lmem_register(context, &reg, &region->lmem_handle) != 0 ||
        region->lmem_handle == nullptr) {
      (void)aclrtFree(region->memory);
      region->memory = nullptr;
      return Result::kResourceError;
    }
    region->token_id = reg.out.ub.token_id;
    region->token_value = reg.in.ub.token_value;
    region->segment_key_bytes = sizeof(urma_seg_t);
    std::memcpy(region->segment_key.data(), reg.out.key.value, region->segment_key_bytes);
    urma_seg_t exported_segment{};
    std::memcpy(&exported_segment, reg.out.key.value, sizeof(exported_segment));
    std::memcpy(region->eid.data(), exported_segment.ubva.eid.raw, region->eid.size());

    // Export a normal UMDK-visible Jetty key. The CPU peer uses this key as
    // the target Jetty for one-sided READ/WRITE operations.
    region->cq_info.in.depth = kCqDepth;
    region->cq_info.in.ub.mode = direct_hccp::kJfcNormal;
    if (api.cq_create(context, &region->cq_info, &region->cq_handle) != 0 ||
        region->cq_handle == nullptr) {
      Release(region.get());
      return Result::kResourceError;
    }
    direct_hccp::QpCreateAttr qp_attr{};
    qp_attr.scq_handle = region->cq_handle;
    qp_attr.rcq_handle = region->cq_handle;
    qp_attr.srq_handle = region->cq_handle;
    qp_attr.sq_depth = kSqDepth;
    qp_attr.rq_depth = kRqDepth;
    qp_attr.transport_mode = direct_hccp::kReliableMessage;
    qp_attr.ub.mode = direct_hccp::kJettyStandard;
    qp_attr.ub.flag = direct_hccp::JettyFlagShareJfr();
    qp_attr.ub.jfs_flag = direct_hccp::JfsFlagErrorSuspend();
    qp_attr.ub.priority = 2U;
    qp_attr.ub.rnr_retry = 7U;
    qp_attr.ub.err_timeout = 16U;
    qp_attr.ub.token_value = region->token_value;
    if (api.qp_create(context, &qp_attr, &region->qp_info, &region->qp_handle) != 0 ||
        region->qp_handle == nullptr || region->qp_info.key.size == 0U) {
      Release(region.get());
      return Result::kResourceError;
    }
    region->jetty_key_bytes = region->qp_info.key.size;
    if (region->jetty_key_bytes > region->jetty_key.size()) {
      Release(region.get());
      return Result::kResourceError;
    }
    std::memcpy(region->jetty_key.data(), region->qp_info.key.value,
                region->jetty_key_bytes);
    region->session_id = next_session++;
    response->result = static_cast<std::int32_t>(Result::kOk);
    response->session_id = region->session_id;
    response->hbm_bytes = region->bytes;
    response->hbm_base = exported_segment.ubva.va;
    response->token_id = region->token_id;
    response->token_value = region->token_value;
    response->segment_key_bytes = region->segment_key_bytes;
    response->jetty_key_bytes = region->jetty_key_bytes;
    std::memcpy(response->segment_key, region->segment_key.data(), kSegKeyBytes);
    std::memcpy(response->jetty_key, region->jetty_key.data(), kQpKeyBytes);
    std::memcpy(response->peer_eid, exported_segment.ubva.eid.raw, kEidBytes);
    regions.emplace(region->session_id, std::move(region));
    return Result::kOk;
  }

  Result Release(std::uint64_t id) {
    auto it = regions.find(id);
    if (it == regions.end()) return Result::kNotFound;
    Release(it->second.get());
    regions.erase(it);
    return Result::kOk;
  }

  Result CreateChannel(const CreateChannelRequest& request,
                       CreateChannelResponse* response) {
    if (response == nullptr || request.session_id == 0U) return Result::kInvalidRequest;
    auto it = regions.find(request.session_id);
    if (it == regions.end()) return Result::kNotFound;
    HbmRegion& region = *it->second;
    if (region.jetty_key_bytes == 0U || request.local_qp_key_bytes == 0U)
      return Result::kResourceError;
    // Import the CPU Jetty into the NPU context so the provider can route
    // one-sided operations to the HBM owner. CTP lookup is asynchronous in
    // the HDC ABI and returns an opaque TP handle.
    std::uint64_t local_tp = 0U;
    if (api.get_tp_info_async != nullptr && api.get_async_req_result != nullptr) {
      direct_hccp::GetTpCfg cfg{};
      cfg.flag = direct_hccp::CtpFlag();
      cfg.transport_mode = direct_hccp::kReliableMessage;
      std::memcpy(cfg.local_eid.raw, region.eid.data(), region.eid.size());
      std::memcpy(cfg.peer_eid.raw, request.local_eid, kEidBytes);
      direct_hccp::TpInfo info{};
      std::uint32_t count = 1U;
      void* async_request = nullptr;
      if (api.get_tp_info_async(context, &cfg, &info, &count, &async_request) == 0 && async_request != nullptr) {
        for (int attempt = 0; attempt < 10000; ++attempt) {
          int result = 0;
          const int rc = api.get_async_req_result(async_request, &result);
          if (rc == 128301) continue;
          if (rc == 0 && result == 0 && count != 0U) local_tp = info.tp_handle;
          break;
        }
      }
    }
    if (local_tp == 0U || request.local_tp_handle == 0U) return Result::kResourceError;
    {
      direct_hccp::QpImportInfoT import_info{};
      if (!direct_hccp::ImportPeerJetty(api, context, request.local_qp_key,
                                        request.local_qp_key_bytes, region.token_value,
                                        local_tp, request.local_tp_handle, 1U,
                                        request.local_psn, "npu_service", &import_info,
                                        &region.peer_target_jetty))
        return Result::kResourceError;
    }
    response->peer_qp_key_bytes = region.jetty_key_bytes;
    std::memcpy(response->peer_qp_key, region.jetty_key.data(), kQpKeyBytes);
    response->peer_tp_handle = local_tp;
    response->peer_psn = 1U;
    return Result::kOk;
  }

  void Release(HbmRegion* region) {
    if (region == nullptr) return;
    if (region->peer_target_jetty != nullptr && api.qp_unimport != nullptr) {
      (void)api.qp_unimport(context, region->peer_target_jetty);
      region->peer_target_jetty = nullptr;
    }
    if (region->qp_handle != nullptr && api.qp_destroy != nullptr) {
      (void)api.qp_destroy(region->qp_handle);
      region->qp_handle = nullptr;
    }
    if (region->cq_handle != nullptr && api.cq_destroy != nullptr) {
      (void)api.cq_destroy(context, region->cq_handle);
      region->cq_handle = nullptr;
    }
    if (region->lmem_handle != nullptr && api.lmem_unregister != nullptr) {
      (void)api.lmem_unregister(context, region->lmem_handle);
      region->lmem_handle = nullptr;
    }
    if (region->memory != nullptr) {
      (void)aclrtFree(region->memory);
      region->memory = nullptr;
    }
  }

  void Handle(int fd) {
    MessageHeader header{};
    if (!ReadAll(fd, &header, sizeof(header)) || header.magic != kProtocolMagic ||
        header.version != kProtocolVersion) return;
    const Command command = static_cast<Command>(header.command);
    if (command == Command::kGetCandidateCount) {
      if (!IsValidHeader(header, command, sizeof(CandidateCountRequest))) {
        SendStatus(fd, command, Result::kInvalidRequest);
        return;
      }
      CandidateCountResponse response{};
      response.header = MakeHeader(command, sizeof(response));
      response.header.command = ResponseCommand(command);
      response.result = static_cast<std::int32_t>(initialized ? Result::kOk : Result::kResourceError);
      response.candidate_count = initialized ? 1U : 0U;
      (void)WriteAll(fd, &response, sizeof(response));
      return;
    }
    if (command == Command::kAllocateHbmRegion) {
      AllocateHbmRegionRequest request{};
      request.header = header;
      if (!IsValidHeader(header, command, sizeof(request)) ||
          !ReadAll(fd, reinterpret_cast<std::uint8_t*>(&request) + sizeof(header),
                   sizeof(request) - sizeof(header))) {
        SendStatus(fd, command, Result::kInvalidRequest);
        return;
      }
      AllocateHbmRegionResponse response{};
      response.header = MakeHeader(command, sizeof(response));
      response.header.command = ResponseCommand(command);
      response.result = static_cast<std::int32_t>(Allocate(request, &response));
      (void)WriteAll(fd, &response, sizeof(response));
      return;
    }
    if (command == Command::kReleaseHbmRegion) {
      ReleaseHbmRegionRequest request{};
      request.header = header;
      if (!IsValidHeader(header, command, sizeof(request)) ||
          !ReadAll(fd, reinterpret_cast<std::uint8_t*>(&request) + sizeof(header),
                   sizeof(request) - sizeof(header))) {
        SendStatus(fd, command, Result::kInvalidRequest);
        return;
      }
      SendStatus(fd, command, Release(request.session_id));
      return;
    }
    if (command == Command::kCreateChannel) {
      CreateChannelRequest request{};
      request.header = header;
      if (!IsValidHeader(header, command, sizeof(request)) ||
          !ReadAll(fd, reinterpret_cast<std::uint8_t*>(&request) + sizeof(header),
                   sizeof(request) - sizeof(header))) {
        SendStatus(fd, command, Result::kInvalidRequest);
        return;
      }
      CreateChannelResponse response{};
      response.header = MakeHeader(command, sizeof(response));
      response.header.command = ResponseCommand(command);
      response.result = static_cast<std::int32_t>(CreateChannel(request, &response));
      (void)WriteAll(fd, &response, sizeof(response));
      return;
    }
    SendStatus(fd, command, Result::kInvalidRequest);
  }

  DirectHccpApi api{};
  DirectHccpSession* session{nullptr};
  void* context{nullptr};
  std::uint32_t physical_device_id{0U};
  std::uint64_t next_session{1U};
  bool initialized{false};
  std::unordered_map<std::uint64_t, std::unique_ptr<HbmRegion>> regions;
};

NpuService::NpuService() : impl_(std::make_unique<Impl>()) {}
NpuService::~NpuService() = default;
Status NpuService::Initialize(const NpuServiceOptions& options) { return impl_->Initialize(options); }
Status NpuService::LinkTo(const LinkTarget&) { return Status::NotImplemented("NpuService is a server"); }
Status NpuService::AcceptLink(const LinkTarget&) { return initialized() ? Status::OK() : Status::NotInitialized("NpuService is not initialized"); }
Status NpuService::ReleaseLink(LinkId link_id) { return impl_->Release(link_id) == Result::kOk ? Status::OK() : Status::NotFound("NPU HBM session not found"); }
Status NpuService::ServeConnection(int fd) { if (!initialized()) return Status::NotInitialized("NpuService is not initialized"); impl_->Handle(fd); return Status::OK(); }
Status NpuService::Shutdown() { return impl_->Shutdown(); }
bool NpuService::initialized() const { return impl_ != nullptr && impl_->initialized; }
RuntimeCapabilities NpuService::capabilities() const { return RuntimeCapabilities{false, false, true, false}; }

}  // namespace kvaf
