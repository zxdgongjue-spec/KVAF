#pragma once
#include "access_operator.h"
#include "execution_protocol.h"
#include <functional>
namespace kvaf {
// Operator facade borrows a typed resource bundle. The manager takes the bundle
// before initialization; standalone legacy callers retain it locally.
class SessionAccessOperator final : public AccessOperator {
 public:
  using Factory = std::function<std::unique_ptr<AccessOperator>(std::shared_ptr<peer::PeerLink>)>;
  SessionAccessOperator(std::shared_ptr<peer::PeerLink> root, execution::Request config,
                        Factory factory, std::uint32_t capacity = 65536)
      : standalone_(std::make_unique<Resources>(root->ExecutionSession(config))),
        resources_(standalone_.get()), factory_(std::move(factory)), capacity_(capacity) {}
  std::unique_ptr<ExecutionResourceOwner> TakeResources() override { return std::move(standalone_); }
  Status ValidateRequestCapacity(std::uint32_t count) const override {
    return count <= capacity_ ? Status::OK()
                             : Status::InvalidArgument("request exceeds execution resource capacity");
  }
  std::unique_ptr<ExecutionState> CreateExecutionState() override {
    return resources_->operation->CreateExecutionState();
  }
  Status BindExecutionState(ExecutionState* state) override {
    return resources_->operation ? resources_->operation->BindExecutionState(state)
                                : state ? Status::NotInitialized("session not initialized") : Status::OK();
  }
  Status Initialize(const RemoteRegionDesc& region) override {
    auto s = resources_->link->PrepareExecution();
    if (!s.ok()) return s;
    resources_->operation = factory_(resources_->link);
    if (!resources_->operation) return Status::Internal("empty execution provider");
    resources_->local = resources_->operation->TakeResources();
    return resources_->operation->Initialize(region);
  }
  Status Shutdown() override { return resources_->Close(); }
  Status Warmup(const ExecutionWarmup& r) override {
    if (r.request_capacity > capacity_)
      return Status::InvalidArgument("warmup exceeds execution capacity");
    auto request = r;
    if (!request.request_capacity) request.request_capacity = capacity_;
    return resources_->operation->Warmup(request);
  }
  Status PrepareGetGraph(const GetGraphOptions& r) override {
    if (!r.request_capacity || r.request_capacity > capacity_)
      return Status::InvalidArgument("graph capacity exceeds provider budget");
    return resources_->operation->PrepareGetGraph(r);
  }
  Status FinishGetGraph(void* stream, bool discard) override {
    return resources_->operation->FinishGetGraph(stream, discard);
  }
  Status DrainGet() override {
    return resources_->operation ? resources_->operation->DrainGet() : Status::OK();
  }
  Status ResolvePut(PutCompletion* result) override {
    return resources_->operation ? resources_->operation->ResolvePut(result)
                                : Status::NotInitialized("session not initialized");
  }
  Status AcknowledgePut() override { return resources_->operation->AcknowledgePut(); }
  bool TracksGetCompletion() const override { return true; }
  Status PreparePut(const BatchPutRequest& r, const BatchPutExecutionOptions& options) override {
    if (r.keys.size() > capacity_)
      return Status::InvalidArgument("Put exceeds this provider's request capacity");
    auto s = resources_->operation->Warmup({static_cast<std::uint32_t>(r.keys.size()), options.stream});
    return s.ok() ? resources_->operation->PreparePut(r, options) : s;
  }
  Status ExecuteBatchPut(const BatchPutExecRequest& r) override {
    return resources_->operation->ExecuteBatchPut(r);
  }
  Status ExecuteBatchGet(const BatchGetExecRequest& r) override {
    if (!r.request || r.request->indices.bytes / sizeof(std::int32_t) > capacity_)
      return Status::InvalidArgument("Get exceeds this provider's request capacity");
    if (!r.execution.graph_lease) {
      auto s = resources_->operation->Warmup(
          {static_cast<std::uint32_t>(r.request->indices.bytes / 4), r.execution.stream});
      if (!s.ok()) return s;
    }
    return resources_->operation->ExecuteBatchGet(r);
  }

 private:
  struct Resources final : ExecutionResourceOwner {
    explicit Resources(std::shared_ptr<peer::PeerLink> value) : link(std::move(value)) {}
    ~Resources() override {
      if (!Close().ok()) {
        (void)operation.release();
        (void)local.release();
        (void)new decltype(link)(std::move(link));
      }
      // The borrowed operation must be destroyed before its resource owner.
      operation.reset();
    }
    Status Close() override {
      if (closed) return Status::OK();
      auto s = link->ResolveExecutionControl();
      if (!s.ok()) return s;
      s = local ? local->Close() : operation ? operation->Shutdown() : Status::OK();
      if (!s.ok()) return s;
      s = link->CloseExecution();
      if (s.ok()) closed = true;
      return s;
    }
    std::shared_ptr<peer::PeerLink> link;
    std::unique_ptr<ExecutionResourceOwner> local;
    std::unique_ptr<AccessOperator> operation;
    bool closed{false};
  };
  std::unique_ptr<Resources> standalone_;
  Resources* resources_;
  Factory factory_;
  std::uint32_t capacity_;
};
}  // namespace kvaf
