#include "npu_service.h"

#include <acl/acl.h>
#include <urma_api.h>
#include <dlfcn.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <map>
#include <thread>
#include <unordered_map>
#include <vector>

#include "direct_hccp/direct_hccp_remote_transport.h"
#include "direct_hccp/direct_hccp_session.h"
#include "npu_target_channel.h"
#include "npu_eid_discovery.h"
#include "peer_protocol.h"

namespace kvaf {
namespace {

using namespace peer;
using direct_hccp::DirectHccpApi;
using direct_hccp::DirectHccpSession;
using npu_service_detail::TargetChannel;

constexpr std::uint32_t kHdcType = 18U;
constexpr std::uint32_t kCqDepth = 16384U;
// Each channel owns a target queue with these depths.
constexpr std::uint32_t kSqDepth = 1024U;
constexpr std::uint32_t kRqDepth = 1024U;
constexpr std::size_t kMaxPeerJettys = 16U;

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

struct DeviceState;

struct HbmRegion {
  DeviceState* owner{nullptr};
  std::uint64_t session_id{0U};
  void* memory{nullptr};
  std::uint64_t bytes{0U};
  void* lmem_handle{nullptr};
  void* token_id_handle{nullptr};
  std::uint32_t token_id{0U};
  std::uint32_t token_value{0U};
  std::array<std::uint8_t, kSegKeyBytes> segment_key{};
  std::uint32_t segment_key_bytes{0U};
  std::array<std::uint8_t, kQpKeyBytes> jetty_key{};
  std::array<std::uint8_t, kEidBytes> eid{};
  std::uint32_t jetty_key_bytes{0U};
  // Preserve the allocation response's Jetty descriptor for compatibility.
  // CreateChannel returns a separately owned target for each CPU initiator.
  std::unique_ptr<TargetChannel> allocation_target;
  std::vector<std::unique_ptr<TargetChannel>> channels;
};

struct DeviceState {
  std::uint32_t device_id{0U};
  std::uint32_t physical_device_id{0U};
  DirectHccpSession* session{nullptr};
  void* context{nullptr};
  std::array<std::uint8_t, peer::kEidBytes> local_eid{};
};


}  // namespace

struct NpuService::Impl {
  ~Impl() { (void)Shutdown(); }

  Status Initialize(const NpuServiceOptions& options) {
    if (initialized) return Status::InvalidState("NpuService already initialized");
    if (!direct_hccp::LoadDirectHccp(&api))
      return Status::NotImplemented("Direct-HCCP RA symbols are unavailable");
    hbm_huge_first = options.hbm_huge_first;
    std::fprintf(stderr, "[KVAF_NPU] HBM_ALLOCATOR mode=%s policy=%s non_pin=1\n",
                 hbm_huge_first ? "huge" : "normal",
                 hbm_huge_first ? "HUGE_FIRST" : "NORMAL_ONLY");
    initialized = true;
    if (options.has_device_id) {
      const std::vector<std::array<std::uint8_t, kEidBytes>>* candidates = nullptr;
      // Leave automatic EID selection to the client's retry loop. One unusable
      // first candidate must not prevent the service from accepting requests.
      if (GetEidCandidates(options.device_id, &candidates) != Result::kOk ||
          (options.has_eid_index && EnsureDevice(options.device_id, options.eid_index, true) == nullptr)) {
        (void)Shutdown();
        return Status::Internal("NPU service failed to initialize requested device");
      }
    }
    return Status::OK();
  }

  DeviceState* EnsureDevice(std::uint32_t device_id, std::uint32_t eid_index,
                             bool has_eid_index) {
    const std::vector<std::array<std::uint8_t, kEidBytes>>* candidates = nullptr;
    if (GetEidCandidates(device_id, &candidates) != Result::kOk) return nullptr;
    const std::uint32_t selected = has_eid_index ? eid_index : 0U;
    if (selected >= candidates->size()) return nullptr;
    const auto key = std::make_pair(device_id, selected);
    auto existing = devices.find(key);
    if (existing != devices.end()) return existing->second.get();
    auto state = std::make_unique<DeviceState>();
    state->device_id = device_id;
    state->local_eid = (*candidates)[selected];
    if (aclrtSetDevice(device_id) != ACL_SUCCESS ||
        !direct_hccp::GetCurrentNpuPhysicalDeviceId(&state->physical_device_id) ||
        !direct_hccp::AcquireDirectHccpSession(state->physical_device_id, kHdcType, &state->session)) {
      return nullptr;
    }
    direct_hccp::CtxInitCfg cfg{};
    cfg.mode = direct_hccp::kOfflineHdc;
    direct_hccp::CtxInitAttr attr{};
    attr.phy_id = state->physical_device_id;
    attr.ub.eid_index = 0U;
    std::copy(state->local_eid.begin(), state->local_eid.end(), attr.ub.eid.raw);
    const int rc = api.ctx_init(&cfg, &attr, &state->context);
    if (rc != 0 || state->context == nullptr) {
      std::fprintf(stderr, "[KVAF_NPU] CONTEXT_FAILED device=%u eid_candidate=%u rc=%d\n",
                   device_id, selected, rc);
      if (state->context != nullptr) (void)api.ctx_deinit(state->context);
      direct_hccp::ReleaseDirectHccpSession(state->session);
      return nullptr;
    }
    std::fprintf(stderr, "[KVAF_NPU] EID_SELECTED device=%u candidate=%u eid=", device_id, selected);
    for (std::uint8_t byte : state->local_eid) std::fprintf(stderr, "%02x", byte);
    std::fprintf(stderr, "\n");
    DeviceState* result = state.get();
    devices.emplace(key, std::move(state));
    return result;
  }

  Result GetEidCandidates(std::uint32_t device_id,
                         const std::vector<std::array<std::uint8_t, kEidBytes>>** output) {
    auto found = eid_candidates.find(device_id);
    if (found != eid_candidates.end()) { *output = &found->second; return Result::kOk; }
    if (aclrtSetDevice(device_id) != ACL_SUCCESS) return Result::kResourceError;
    std::uint32_t physical = 0U;
    if (!direct_hccp::GetCurrentNpuPhysicalDeviceId(&physical)) return Result::kResourceError;
    std::vector<std::array<std::uint8_t, kEidBytes>> candidates;
    std::string error;
    if (!npu_eid_detail::DiscoverDeviceEids(physical, &candidates, &error)) {
      std::fprintf(stderr, "[KVAF_NPU] EID_DISCOVERY_FAILED device=%u reason=%s\n",
                   device_id, error.c_str());
      return Result::kNotFound;
    }
    *output = &eid_candidates.emplace(device_id, std::move(candidates)).first->second;
    return Result::kOk;
  }

  void ShutdownDevice() {
    for (auto& entry : regions) Release(entry.second.get());
    regions.clear();
    for (auto& entry : devices) {
      DeviceState& state = *entry.second;
      if (state.context != nullptr && api.ctx_deinit != nullptr) {
        (void)api.ctx_deinit(state.context);
        state.context = nullptr;
      }
      if (state.session != nullptr) {
        direct_hccp::ReleaseDirectHccpSession(state.session);
        state.session = nullptr;
      }
    }
    devices.clear();
    eid_candidates.clear();
  }

  Status Shutdown() {
    ShutdownDevice();
    if (api.library != nullptr) direct_hccp::CloseDirectHccp(&api);
    initialized = false;
    return Status::OK();
  }

  Result Allocate(const AllocateHbmRegionRequest& request,
                  AllocateHbmRegionResponse* response) {
    if (!initialized || response == nullptr || request.hbm_bytes == 0U ||
        request.slot_size == 0U || request.hbm_bytes % request.slot_size != 0U)
      return Result::kInvalidRequest;
    DeviceState* device = EnsureDevice(
        request.device_id, request.eid_index, request.eid_index != UINT32_MAX);
    if (device == nullptr) return Result::kResourceError;
    auto region = std::make_unique<HbmRegion>();
    region->owner = device;
    region->bytes = request.hbm_bytes;
    const auto policy = hbm_huge_first ? ACL_MEM_MALLOC_HUGE_FIRST : ACL_MEM_MALLOC_NORMAL_ONLY;
    if (aclrtSetDevice(device->device_id) != ACL_SUCCESS ||
        aclrtMalloc(&region->memory, request.hbm_bytes, policy) != ACL_SUCCESS)
      return Result::kResourceError;
    direct_hccp::TokenId token_id{};
    if (api.token_alloc == nullptr ||
        api.token_alloc(device->context, &token_id, &region->token_id_handle) != 0 ||
        region->token_id_handle == nullptr) {
      Release(region.get());
      return Result::kResourceError;
    }
    region->token_id = token_id.token_id;
    direct_hccp::MrRegInfo reg{};
    reg.in.mem.addr = reinterpret_cast<std::uint64_t>(region->memory);
    reg.in.mem.size = request.hbm_bytes;
    // Direct HCCP registers device HBM through its non-pin device-memory path.
    reg.in.ub.flags = direct_hccp::MemRegFlags(true);
    reg.in.ub.token_value = 0xABCFU;
    reg.in.ub.token_id_handle = region->token_id_handle;
    if (api.lmem_register(device->context, &reg, &region->lmem_handle) != 0 ||
        region->lmem_handle == nullptr) {
      Release(region.get());
      return Result::kResourceError;
    }
    region->token_value = reg.in.ub.token_value;
    region->segment_key_bytes = sizeof(urma_seg_t);
    std::memcpy(region->segment_key.data(), reg.out.key.value, region->segment_key_bytes);
    urma_seg_t exported_segment{};
    std::memcpy(&exported_segment, reg.out.key.value, sizeof(exported_segment));
    std::memcpy(region->eid.data(), exported_segment.ubva.eid.raw, region->eid.size());

    // AllocateHbmRegion's protocol includes a live Jetty key. Actual channel
    // targets are created later and returned by CreateChannel.
    region->allocation_target = std::make_unique<TargetChannel>(api, device->context);
    const Status target_status = region->allocation_target->Initialize(
        region->token_id_handle, region->token_value, kSqDepth, kRqDepth, kCqDepth);
    if (!target_status.ok()) {
      std::fprintf(stderr, "[KVAF_NPU] TARGET_QUEUE_FAILED reason=%s\n",
                   target_status.message().c_str());
      Release(region.get());
      return Result::kResourceError;
    }
    std::fprintf(stderr, "[KVAF_NPU] TARGET_QUEUE sq_depth=%u rq_depth=%u\n",
                 kSqDepth, kRqDepth);
    const auto& allocation_jetty = region->allocation_target->jetty_info();
    region->jetty_key_bytes = allocation_jetty.key.size;
    if (region->jetty_key_bytes > region->jetty_key.size()) {
      Release(region.get());
      return Result::kResourceError;
    }
    std::memcpy(region->jetty_key.data(), allocation_jetty.key.value,
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
    std::memcpy(response->route_eid, device->local_eid.data(), kEidBytes);
    std::fprintf(stderr, "[KVAF_NPU] HBM_EXPORTED session=%llu base=0x%llx bytes=%llu memory_eid=",
                 static_cast<unsigned long long>(region->session_id),
                 static_cast<unsigned long long>(response->hbm_base),
                 static_cast<unsigned long long>(region->bytes));
    for (std::uint8_t byte : region->eid) std::fprintf(stderr, "%02x", byte);
    std::fprintf(stderr, " route_eid=");
    for (std::uint8_t byte : device->local_eid) std::fprintf(stderr, "%02x", byte);
    std::fprintf(stderr, "\n");
    regions.emplace(region->session_id, std::move(region));
    return Result::kOk;
  }

  Result Release(std::uint64_t id) {
    auto it = regions.find(id);
    if (it == regions.end()) return Result::kNotFound;
    std::fprintf(stderr, "[KVAF_NPU] HBM_RELEASE session=%llu base=0x%llx bytes=%llu\n",
                 static_cast<unsigned long long>(id),
                 static_cast<unsigned long long>(it->second->memory == nullptr ? 0U :
                     reinterpret_cast<std::uintptr_t>(it->second->memory)),
                 static_cast<unsigned long long>(it->second->bytes));
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
    if (region.channels.size() >= kMaxPeerJettys) {
      std::fprintf(stderr, "[KVAF_NPU] CHANNEL_FAILED stage=peer_jetty_limit limit=%zu\n",
                   kMaxPeerJettys);
      return Result::kResourceError;
    }
    DeviceState* device = region.owner;
    if (device == nullptr || device->context == nullptr) return Result::kResourceError;
    (void)aclrtSetDevice(device->device_id);
    if (region.jetty_key_bytes == 0U || request.local_qp_key_bytes == 0U) {
      std::fprintf(stderr, "[KVAF_NPU] CHANNEL_FAILED stage=descriptor session=%llu local_key_bytes=%u jetty_key_bytes=%u\n",
                   static_cast<unsigned long long>(request.session_id), request.local_qp_key_bytes,
                   region.jetty_key_bytes);
      return Result::kResourceError;
    }
    // Import the CPU Jetty into the NPU context so the provider can route
    // one-sided operations to the HBM owner. CTP lookup is asynchronous in
    // the HDC ABI and returns an opaque TP handle.
    std::uint64_t local_tp = 0U;
    if (api.get_tp_info_async != nullptr && api.get_async_req_result != nullptr) {
      direct_hccp::GetTpCfg cfg{};
      cfg.flag = direct_hccp::CtpFlag();
      cfg.transport_mode = direct_hccp::kReliableMessage;
      std::memcpy(cfg.local_eid.raw, device->local_eid.data(), device->local_eid.size());
      std::memcpy(cfg.peer_eid.raw, request.local_eid, kEidBytes);
      direct_hccp::TpInfo info{};
      std::uint32_t count = 1U;
      void* async_request = nullptr;
      const int submit_rc = api.get_tp_info_async(device->context, &cfg, &info, &count, &async_request);
      if (submit_rc == 0 && async_request != nullptr) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        for (int attempt = 0; std::chrono::steady_clock::now() < deadline; ++attempt) {
          int result = 0;
          const int rc = api.get_async_req_result(async_request, &result);
          if (rc == 128301) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
          }
          std::fprintf(stderr,
                       "[KVAF_NPU] CHANNEL_CTP_RESULT api_rc=%d result=%d count=%u tp=0x%llx local_eid=",
                       rc, result, count, static_cast<unsigned long long>(info.tp_handle));
          for (std::uint8_t byte : device->local_eid) std::fprintf(stderr, "%02x", byte);
          std::fprintf(stderr, " peer_eid=");
          for (std::uint8_t byte : request.local_eid) std::fprintf(stderr, "%02x", byte);
          std::fprintf(stderr, "\n");
          if (rc == 0 && result == 0 && count != 0U) local_tp = info.tp_handle;
          break;
        }
      } else {
        std::fprintf(stderr, "[KVAF_NPU] CHANNEL_FAILED stage=get_tp_info submit_rc=%d count=%u request=%p errno=%d\n",
                     submit_rc, count, async_request, errno);
      }
    }
    if (local_tp == 0U || request.local_tp_handle == 0U) {
      std::fprintf(stderr, "[KVAF_NPU] CHANNEL_FAILED stage=get_tp_info local_tp=0x%llx peer_tp=0x%llx\n",
                   static_cast<unsigned long long>(local_tp),
                   static_cast<unsigned long long>(request.local_tp_handle));
      return Result::kResourceError;
    }
    auto channel = std::make_unique<TargetChannel>(api, device->context);
    const Status target_status = channel->Initialize(
        region.token_id_handle, region.token_value, kSqDepth, kRqDepth, kCqDepth);
    if (!target_status.ok()) {
      std::fprintf(stderr, "[KVAF_NPU] CHANNEL_FAILED stage=create_target_jetty reason=%s\n",
                   target_status.message().c_str());
      return Result::kResourceError;
    }
    const auto& target_jetty = channel->jetty_info();
    if (target_jetty.key.size > sizeof(response->peer_qp_key)) {
      return Result::kResourceError;
    }
    direct_hccp::QpImportInfoT import_info{};
    if (!direct_hccp::ImportPeerJetty(api, device->context, request.local_qp_key,
                                      request.local_qp_key_bytes, region.token_value,
                                      local_tp, request.local_tp_handle, 1U,
                                      request.local_psn, "npu_service", &import_info,
                                      channel->peer_jetty_output())) {
      std::fprintf(stderr, "[KVAF_NPU] CHANNEL_FAILED stage=import_cpu_jetty\n");
      return Result::kResourceError;
    }
    response->peer_qp_key_bytes = target_jetty.key.size;
    std::memcpy(response->peer_qp_key, target_jetty.key.value, target_jetty.key.size);
    response->peer_tp_handle = local_tp;
    response->peer_psn = 1U;
    std::fprintf(stderr,
                 "[KVAF_NPU] CHANNEL_READY session=%llu channel_index=%zu "
                 "target_jetty_id=%u target_jfc_id=%u sq_depth=%u rq_depth=%u cq_depth=%u\n",
                 static_cast<unsigned long long>(region.session_id), region.channels.size(),
                 target_jetty.ub.id, channel->cq_info().out.id,
                 kSqDepth, kRqDepth, kCqDepth);
    region.channels.push_back(std::move(channel));
    return Result::kOk;
  }

  void Release(HbmRegion* region) {
    if (region == nullptr) return;
    DeviceState* device = region->owner;
    if (device == nullptr) return;
    (void)aclrtSetDevice(device->device_id);
    region->channels.clear();
    region->allocation_target.reset();
    if (region->lmem_handle != nullptr && api.lmem_unregister != nullptr) {
      (void)api.lmem_unregister(device->context, region->lmem_handle);
      region->lmem_handle = nullptr;
    }
    if (region->token_id_handle != nullptr && api.token_free != nullptr) {
      (void)api.token_free(device->context, region->token_id_handle);
      region->token_id_handle = nullptr;
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
    if (command == Command::kGetHbmEidCandidate) {
      HbmEidCandidateRequest request{};
      request.header = header;
      if (!IsValidHeader(header, command, sizeof(request)) ||
          !ReadAll(fd, reinterpret_cast<std::uint8_t*>(&request) + sizeof(header),
                   sizeof(request) - sizeof(header))) {
        SendStatus(fd, command, Result::kInvalidRequest);
        return;
      }
      HbmEidCandidateResponse response{};
      response.header = MakeHeader(command, sizeof(response));
      response.header.command = ResponseCommand(command);
      const std::vector<std::array<std::uint8_t, kEidBytes>>* candidates = nullptr;
      Result result = GetEidCandidates(request.device_id, &candidates);
      if (result == Result::kOk && request.candidate_index >= candidates->size()) result = Result::kNotFound;
      if (result == Result::kOk) {
        response.candidate_count = static_cast<std::uint32_t>(candidates->size());
        response.candidate_index = request.candidate_index;
        const auto& eid = (*candidates)[request.candidate_index];
        std::copy(eid.begin(), eid.end(), response.eid);
      }
      response.result = static_cast<std::int32_t>(result);
      (void)WriteAll(fd, &response, sizeof(response));
      return;
    }
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
  std::uint64_t next_session{1U};
  bool initialized{false};
  bool hbm_huge_first{true};
  std::map<std::pair<std::uint32_t, std::uint32_t>,
           std::unique_ptr<DeviceState>> devices;
  std::unordered_map<std::uint32_t, std::vector<std::array<std::uint8_t, kEidBytes>>> eid_candidates;
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
