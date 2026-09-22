#pragma once

#include "control.h"
#include "direct_hccp/direct_hccp_remote_transport.h"
#include "direct_hccp/direct_hccp_session.h"
#include "npu_eid_discovery.h"
#include "npu_target_channel.h"

#include <acl/acl.h>
#include <urma_api.h>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

namespace kvaf::aicpu_test {
inline void Acl(aclError status, const char* operation) {
  Require(status == ACL_SUCCESS, std::string(operation) + " failed: " + std::to_string(status));
}

// A test-owned target, using KVAF's RA ABI/session and TargetChannel helpers.
// No production NpuService or Runtime state is modified.
class NpuTarget {
 public:
  ~NpuTarget() {
    channels_.clear();
    if (memory_handle_) api_.lmem_unregister(context_, memory_handle_);
    if (token_) api_.token_free(context_, token_);
    if (memory_) aclrtFree(memory_);
    if (context_) api_.ctx_deinit(context_);
    if (session_) direct_hccp::ReleaseDirectHccpSession(session_);
    direct_hccp::CloseDirectHccp(&api_);
  }
  void Initialize(std::uint32_t device, const std::array<std::uint8_t, peer::kEidBytes>& eid,
                  std::uint64_t bytes) {
    namespace d = direct_hccp;
    Acl(aclrtSetDevice(device), "aclrtSetDevice");
    Require(d::LoadDirectHccp(&api_), "cannot load Direct HCCP");
    std::uint32_t physical = 0;
    Require(d::GetCurrentNpuPhysicalDeviceId(&physical) &&
            d::AcquireDirectHccpSession(physical, 18, &session_), "cannot acquire HCCP session");
    eid_ = eid;
    d::CtxInitCfg cfg{}; cfg.mode = d::kOfflineHdc;
    d::CtxInitAttr attr{}; attr.phy_id = physical; attr.ub.eid_index = 0;
    std::memcpy(attr.ub.eid.raw, eid_.data(), eid_.size());
    Require(api_.ctx_init(&cfg, &attr, &context_) == 0 && context_, "RA context creation failed");
    d::TokenId token{};
    Require(api_.token_alloc(context_, &token, &token_) == 0 && token_, "token allocation failed");
    Acl(aclrtMalloc(&memory_, bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "allocate target HBM");
    Acl(aclrtMemset(memory_, bytes, 0, bytes), "clear target HBM");
    d::MrRegInfo mr{};
    mr.in.mem.addr = reinterpret_cast<std::uint64_t>(memory_); mr.in.mem.size = bytes;
    mr.in.ub.flags = d::MemRegFlags(true); mr.in.ub.token_value = kTokenValue;
    mr.in.ub.token_id_handle = token_;
    Require(api_.lmem_register(context_, &mr, &memory_handle_) == 0 && memory_handle_,
            "register target HBM failed");
    static_assert(sizeof(urma_seg_t) <= peer::kSegKeyBytes);
    Require(mr.out.key.size >= sizeof(urma_seg_t) && mr.out.key.size <= peer::kSegKeyBytes,
            "unexpected exported segment ABI");
    urma_seg_t segment{};
    std::memcpy(&segment, mr.out.key.value, sizeof(segment));
    exported_.header = peer::MakeHeader(kConfigure, sizeof(exported_));
    exported_.header.command = peer::ResponseCommand(kConfigure);
    exported_.session_id = 1; exported_.hbm_base = segment.ubva.va;
    exported_.hbm_bytes = bytes; exported_.token_id = mr.out.ub.token_id;
    exported_.token_value = kTokenValue; exported_.segment_key_bytes = sizeof(segment);
    std::memcpy(exported_.segment_key, &segment, sizeof(segment));
    std::memcpy(exported_.peer_eid, segment.ubva.eid.raw, peer::kEidBytes);
    auto allocation_channel = NewChannel();
    const auto& key = allocation_channel->jetty_info().key;
    exported_.jetty_key_bytes = key.size;
    std::memcpy(exported_.jetty_key, key.value, key.size);
    channels_.push_back(std::move(allocation_channel));
  }
  void* memory() const { return memory_; }
  const peer::AllocateHbmRegionResponse& exported() const { return exported_; }
  void CreateChannel(int fd, const peer::MessageHeader& header) {
    Require(peer::IsValidHeader(header, peer::Command::kCreateChannel,
                               sizeof(peer::CreateChannelRequest)), "invalid CreateChannel header");
    peer::CreateChannelRequest request{}; request.header = header;
    Receive(fd, reinterpret_cast<char*>(&request) + sizeof(header), sizeof(request) - sizeof(header));
    Require(request.session_id == 1 && request.local_qp_key_bytes > 0 &&
            request.local_qp_key_bytes <= peer::kQpKeyBytes && channels_.size() <= 16,
            "invalid CreateChannel request");
    const auto tp = LookupTp(request.local_eid);
    auto channel = NewChannel();
    direct_hccp::QpImportInfoT info{};
    Require(direct_hccp::ImportPeerJetty(api_, context_, request.local_qp_key,
              request.local_qp_key_bytes, kTokenValue, tp, request.local_tp_handle,
              1, request.local_psn, "aicpu_get_test", &info, channel->peer_jetty_output()),
            "import CPU Jetty failed");
    peer::CreateChannelResponse response{};
    response.header = peer::MakeHeader(peer::Command::kCreateChannel, sizeof(response));
    response.header.command = peer::ResponseCommand(peer::Command::kCreateChannel);
    response.peer_tp_handle = tp; response.peer_psn = 1;
    const auto& key = channel->jetty_info().key;
    response.peer_qp_key_bytes = key.size;
    std::memcpy(response.peer_qp_key, key.value, key.size);
    channels_.push_back(std::move(channel));
    Send(fd, &response, sizeof(response));
  }
 private:
  std::unique_ptr<npu_service_detail::TargetChannel> NewChannel() {
    auto channel = std::make_unique<npu_service_detail::TargetChannel>(api_, context_);
    const auto status = channel->Initialize(token_, kTokenValue, 1024, 1024, 16384);
    Require(status.ok(), status.message());
    Require(channel->jetty_info().key.size <= peer::kQpKeyBytes, "oversized Jetty key");
    return channel;
  }
  std::uint64_t LookupTp(const std::uint8_t* peer_eid) {
    namespace d = direct_hccp;
    Require(api_.get_tp_info_async && api_.get_async_req_result, "async CTP lookup unavailable");
    d::GetTpCfg cfg{}; cfg.flag = d::CtpFlag(); cfg.transport_mode = d::kReliableMessage;
    std::memcpy(cfg.local_eid.raw, eid_.data(), eid_.size());
    std::memcpy(cfg.peer_eid.raw, peer_eid, eid_.size());
    d::TpInfo info{}; std::uint32_t count = 1; void* request = nullptr;
    Require(api_.get_tp_info_async(context_, &cfg, &info, &count, &request) == 0 && request,
            "submit CTP lookup failed");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < deadline) {
      int result = 0;
      const int rc = api_.get_async_req_result(request, &result);
      if (rc == 128301) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); continue; }
      Require(rc == 0 && result == 0 && count > 0 && info.tp_handle, "CTP lookup failed");
      return info.tp_handle;
    }
    throw std::runtime_error("CTP lookup timeout");
  }
  static constexpr std::uint32_t kTokenValue = 0xABCF;
  direct_hccp::DirectHccpApi api_{};
  direct_hccp::DirectHccpSession* session_{};
  void* context_{};
  void* token_{};
  void* memory_{};
  void* memory_handle_{};
  std::array<std::uint8_t, peer::kEidBytes> eid_{};
  peer::AllocateHbmRegionResponse exported_{};
  std::vector<std::unique_ptr<npu_service_detail::TargetChannel>> channels_;
};
}  // namespace kvaf::aicpu_test
