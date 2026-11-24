#pragma once

#include <functional>
#include "access_operator.h"
#include "execution_protocol.h"

namespace kvaf {

// Owns the remote execution session and the matching local resource owner.
// Lifecycle is independent of SessionAccessOperator. The local operation is a
// borrowed adapter and must be destroyed before local resources.
class PeerExecutionResources final : public ExecutionResourceOwner {
 public:
  using Links = std::vector<std::shared_ptr<peer::PeerLink>>;
  using Factory = std::function<std::unique_ptr<AccessOperator>(const Links&)>;
  PeerExecutionResources(std::shared_ptr<peer::PeerLink> root, execution::Request config,
                         std::vector<peer::PeerCandidateInfo> candidates, Factory factory,
                         std::uint32_t capacity)
      : root_(std::move(root)), config_(config), candidates_(std::move(candidates)),
        factory_(std::move(factory)), capacity_(capacity) {}
  ~PeerExecutionResources() override {
    if (!Close().ok()) {
      (void)operation_.release();
      (void)local_.release();
      (void)new Links(std::move(links_));
    }
    operation_.reset();
  }
  Status Initialize(const RemoteRegionDesc& region) override {
    if (attempted_ || closed_) return Status::InvalidState("execution resources require recreation");
    attempted_ = true;
    // Fabric uses one session without a CPU EID. URMA opens one per selected FE.
    const std::size_t count = candidates_.empty() ? 1 : candidates_.size();
    links_.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      const auto* candidate = candidates_.empty() ? nullptr : &candidates_[i];
      links_.push_back(root_->ExecutionSession(config_, candidate));
      auto s = links_.back()->PrepareExecution();
      if (!s.ok()) return candidate ? Status::Error(s.code(), candidate->device_name + ": " + s.message()) : s;
    }
    operation_ = factory_(links_);
    if (!operation_) return Status::Internal("empty execution provider");
    local_ = operation_->TakeResources();
    return local_ ? local_->Initialize(region) : operation_->Initialize(region);
  }
  Status Warmup(const ExecutionWarmup& r) override {
    auto s = ValidateCapacity(r.request_capacity);
    if (!s.ok()) return s;
    if (closed_ || !operation_) return Status::NotInitialized("execution session not initialized");
    auto request = r;
    if (!request.request_capacity) request.request_capacity = capacity_;
    return local_ ? local_->Warmup(request) : operation_->Warmup(request);
  }
  Status Close() override {
    if (closed_) return Status::OK();
    // Describing a request may instantiate this owner without acquiring any
    // resources. In particular, closing an over-capacity request must not send
    // a remote CREATE just to close a session that was never opened.
    for (const auto& link : links_) {
      auto s = link->ResolveExecutionControl();
      if (!s.ok()) return s;
    }
    auto s = local_ ? local_->Close() : operation_ ? operation_->Shutdown() : Status::OK();
    if (!s.ok()) return s;
    while (!links_.empty()) {
      s = links_.back()->CloseExecution();
      if (!s.ok()) return s;
      links_.pop_back();
    }
    closed_ = true;
    return Status::OK();
  }
  Status ValidateCapacity(std::uint32_t count) const {
    if (!config_.put && IsWindowedGet({config_.transport, config_.executor, config_.algorithm}))
      return Status::OK();
    return count <= capacity_ ? Status::OK()
                             : Status::InvalidArgument("request exceeds execution resource capacity");
  }
  AccessOperator* operation() const { return operation_.get(); }

 private:
  std::shared_ptr<peer::PeerLink> root_;
  Links links_;
  execution::Request config_;
  std::vector<peer::PeerCandidateInfo> candidates_;
  Factory factory_;
  std::uint32_t capacity_;
  std::unique_ptr<ExecutionResourceOwner> local_;
  std::unique_ptr<AccessOperator> operation_;
  bool attempted_{false}, closed_{false};
};

}  // namespace kvaf
