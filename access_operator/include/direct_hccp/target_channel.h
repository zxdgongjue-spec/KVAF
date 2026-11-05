#pragma once

#include <cstdint>

#include "direct_hccp/direct_hccp_remote_transport.h"
#include "status.h"

namespace kvaf::direct_hccp {

// Owns one NPU target Jetty, its CQ, and the imported CPU initiator.
// The caller keeps the context, API and region token alive until destruction.
class TargetChannel {
 public:
  TargetChannel(const direct_hccp::DirectHccpApi& api, void* context)
      : api_(api), context_(context) {}
  ~TargetChannel() { (void)Shutdown(); }
  TargetChannel(const TargetChannel&) = delete;
  TargetChannel& operator=(const TargetChannel&) = delete;

  Status Initialize(void* token_id_handle, std::uint32_t token_value,
                    std::uint32_t sq_depth, std::uint32_t rq_depth,
                    std::uint32_t cq_depth) {
    if (cq_handle_ != nullptr || jetty_handle_ != nullptr) {
      return Status::InvalidState("NPU target channel already initialized");
    }
    if (context_ == nullptr || token_id_handle == nullptr || token_value == 0U ||
        api_.cq_create == nullptr || api_.cq_destroy == nullptr ||
        api_.qp_create == nullptr || api_.qp_destroy == nullptr ||
        api_.qp_unimport == nullptr) {
      return Status::InvalidArgument("invalid NPU target channel resources");
    }
    cq_info_ = {};
    jetty_info_ = {};
    cq_info_.in.depth = cq_depth;
    cq_info_.in.ub.mode = direct_hccp::kJfcNormal;
    const int cq_rc = api_.cq_create(context_, &cq_info_, &cq_handle_);
    if (cq_rc != 0 || cq_handle_ == nullptr) {
      return Status::Internal("NPU target CQ creation failed (rc=" +
                              std::to_string(cq_rc) + ")");
    }
    direct_hccp::QpCreateAttr attr{};
    attr.scq_handle = cq_handle_;
    attr.rcq_handle = cq_handle_;
    attr.srq_handle = cq_handle_;
    attr.sq_depth = sq_depth;
    attr.rq_depth = rq_depth;
    attr.transport_mode = direct_hccp::kReliableMessage;
    attr.ub.mode = direct_hccp::kJettyStandard;
    attr.ub.flag = direct_hccp::JettyFlagShareJfr();
    attr.ub.token_id_handle = token_id_handle;
    attr.ub.jfs_flag = direct_hccp::JfsFlagErrorSuspend();
    attr.ub.priority = 2U;
    attr.ub.rnr_retry = 7U;
    attr.ub.err_timeout = 16U;
    attr.ub.token_value = token_value;
    const int qp_rc = api_.qp_create(context_, &attr, &jetty_info_, &jetty_handle_);
    if (qp_rc != 0 || jetty_handle_ == nullptr || jetty_info_.key.size == 0U ||
        jetty_info_.key.size > sizeof(jetty_info_.key.value)) {
      return Status::Internal("NPU target Jetty creation failed (rc=" +
                              std::to_string(qp_rc) + ")");
    }
    return Status::OK();
  }

  const direct_hccp::QpCreateInfo& jetty_info() const { return jetty_info_; }
  const direct_hccp::CqInfo& cq_info() const { return cq_info_; }
  // ImportPeerJetty may set a handle even when it reports failure. Retain the
  // output here so destruction also releases partially imported resources.
  void** peer_jetty_output() { return &peer_jetty_; }

  Status Shutdown() {
    if (peer_jetty_ != nullptr && api_.qp_unimport != nullptr) {
      if (api_.qp_unimport(context_, peer_jetty_)) return Status::Internal("unimport CPU target Jetty failed");
      peer_jetty_ = nullptr;
    }
    if (jetty_handle_ != nullptr && api_.qp_destroy != nullptr) {
      if (api_.qp_destroy(jetty_handle_)) return Status::Internal("destroy NPU target Jetty failed");
      jetty_handle_ = nullptr;
    }
    if (cq_handle_ != nullptr && api_.cq_destroy != nullptr) {
      if (api_.cq_destroy(context_, cq_handle_)) return Status::Internal("destroy NPU target CQ failed");
      cq_handle_ = nullptr;
    }
    return Status::OK();
  }

 private:
  const direct_hccp::DirectHccpApi& api_;
  void* context_{nullptr};
  void* cq_handle_{nullptr};
  void* jetty_handle_{nullptr};
  void* peer_jetty_{nullptr};
  direct_hccp::CqInfo cq_info_{};
  direct_hccp::QpCreateInfo jetty_info_{};
};

}  // namespace kvaf::direct_hccp
