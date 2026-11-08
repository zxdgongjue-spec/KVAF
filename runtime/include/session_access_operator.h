#pragma once
#include "access_operator.h"
#include "execution_protocol.h"
#include <functional>
namespace kvaf {
// Owns a CPU execution session until both the local provider and remote producer
// have acknowledged teardown. A failed initialization retains both for recovery.
class SessionAccessOperator final : public AccessOperator {
 public:
  using Factory = std::function<std::unique_ptr<AccessOperator>(std::shared_ptr<peer::PeerLink>)>;
  SessionAccessOperator(std::shared_ptr<peer::PeerLink> root, execution::Request config,
                        Factory factory, std::uint32_t capacity = 65536)
      : link_(root->ExecutionSession(config)), factory_(std::move(factory)), capacity_(capacity) {}
  ~SessionAccessOperator() override {
    if (!Shutdown().ok()) {
      (void)owner_.release();
      (void)new decltype(link_)(std::move(link_));
    }
  }
  Status Initialize(const RemoteRegionDesc& region) override {
    auto s = link_->PrepareExecution();
    if (!s.ok()) return s;
    owner_ = factory_(link_);
    return owner_ ? owner_->Initialize(region) : Status::Internal("empty execution provider");
  }
  Status Shutdown() override {
    auto resolved = link_->ResolveExecutionControl();
    if (!resolved.ok()) return resolved;
    if (owner_) {
      auto s = owner_->Shutdown();
      if (!s.ok()) return s;
      owner_.reset();
    }
    return link_->CloseExecution();
  }
  Status Warmup(const ExecutionWarmup& r) override {
    if (r.request_capacity > capacity_)
      return Status::InvalidArgument("warmup exceeds execution capacity");
    auto request = r;
    if (!request.request_capacity) request.request_capacity = capacity_;
    return owner_->Warmup(request);
  }
  Status PrepareGetGraph(const GetGraphOptions& r) override {
    if (!r.request_capacity || r.request_capacity > capacity_)
      return Status::InvalidArgument("graph capacity exceeds provider budget");
    return owner_->PrepareGetGraph(r);
  }
  Status FinishGetGraph(void* stream, bool discard) override {
    return owner_->FinishGetGraph(stream, discard);
  }
  Status DrainGet() override { return owner_ ? owner_->DrainGet() : Status::OK(); }
  bool TracksGetCompletion() const override { return true; }
  Status PreparePut(const BatchPutRequest& r, const BatchPutExecutionOptions& options) override {
    if (r.keys.size() > capacity_)
      return Status::InvalidArgument("Put exceeds this provider's request capacity");
    auto s = owner_->Warmup({static_cast<std::uint32_t>(r.keys.size()), options.stream});
    return s.ok() ? owner_->PreparePut(r, options) : s;
  }
  Status ExecuteBatchPut(const BatchPutExecRequest& r) override {
    return owner_->ExecuteBatchPut(r);
  }
  Status ExecuteBatchGet(const BatchGetExecRequest& r) override {
    if (!r.request || r.request->indices.bytes / sizeof(std::int32_t) > capacity_)
      return Status::InvalidArgument("Get exceeds this provider's request capacity");
    if (!r.execution.graph_lease) {
      auto s = owner_->Warmup(
          {static_cast<std::uint32_t>(r.request->indices.bytes / 4), r.execution.stream});
      if (!s.ok()) return s;
    }
    return owner_->ExecuteBatchGet(r);
  }

 private:
  std::shared_ptr<peer::PeerLink> link_;
  Factory factory_;
  std::unique_ptr<AccessOperator> owner_;
  std::uint32_t capacity_;
};
}  // namespace kvaf
