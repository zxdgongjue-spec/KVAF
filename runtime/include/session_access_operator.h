#pragma once

#include "peer_execution_resources.h"

namespace kvaf {

// Session dispatch adapter. PeerExecutionResources owns all control-plane
// lifecycle; concrete providers receive their own typed views for execution.
class SessionAccessOperator final : public AccessOperator {
 public:
  using Factory = PeerExecutionResources::Factory;
  SessionAccessOperator(std::shared_ptr<peer::PeerLink> root, execution::Request config,
                        std::vector<peer::PeerCandidateInfo> candidates, Factory factory,
                        std::uint32_t capacity = 65536)
      : standalone_(std::make_unique<PeerExecutionResources>(
            std::move(root), config, std::move(candidates), std::move(factory), capacity)), resources_(standalone_.get()) {}
  std::unique_ptr<ExecutionResourceOwner> TakeResources() override { return std::move(standalone_); }
  Status ValidateRequestCapacity(std::uint32_t count) const override {
    return resources_->ValidateCapacity(count);
  }
  std::unique_ptr<ExecutionState> CreateExecutionState() override {
    return resources_->operation()->CreateExecutionState();
  }
  Status BindExecutionState(ExecutionState* state) override {
    auto* op = resources_->operation();
    return op ? op->BindExecutionState(state)
              : state ? Status::NotInitialized("session not initialized") : Status::OK();
  }
  Status Initialize(const RemoteRegionDesc& region) override { return resources_->Initialize(region); }
  Status Shutdown() override { return resources_->Close(); }
  Status Warmup(const ExecutionWarmup& r) override { return resources_->Warmup(r); }
  Status PreparePutInput(const PutInputOptions& r) override {
    auto s = ValidateRequestCapacity(r.request_capacity);
    return s.ok() ? resources_->operation()->PreparePutInput(r) : s;
  }
  Status FinishPutInput() override { return resources_->operation()->FinishPutInput(); }
  Status PrepareGetGraph(const GetGraphOptions& r) override {
    auto s = ValidateRequestCapacity(r.request_capacity);
    if (!r.request_capacity) return Status::InvalidArgument("graph capacity must be positive");
    return s.ok() ? resources_->operation()->PrepareGetGraph(r) : s;
  }
  Status FinishGetGraph(void* stream, bool discard) override {
    return resources_->operation()->FinishGetGraph(stream, discard);
  }
  bool GetGraphNeedsRecovery() const override {
    return resources_->operation()->GetGraphNeedsRecovery();
  }
  Status PrepareGetOutput(const GetOutputOptions& r) override {
    auto s = ValidateRequestCapacity(r.request_capacity);
    return s.ok() ? resources_->operation()->PrepareGetOutput(r) : s;
  }
  Status FinishGetOutput() override { return resources_->operation()->FinishGetOutput(); }
  Status SetBatchGetKernelProfiling(bool enabled, bool detailed) override {
    auto* op = resources_->operation();
    return op ? op->SetBatchGetKernelProfiling(enabled, detailed)
              : Status::NotInitialized("session not initialized");
  }
  Status CopyBatchGetKernelProfiles(std::vector<BatchGetKernelProfile>* profiles) const override {
    auto* op = resources_->operation();
    return op ? op->CopyBatchGetKernelProfiles(profiles)
              : Status::NotInitialized("session not initialized");
  }
  Status CheckBatchGetStatus(void* stream) override {
    auto* op = resources_->operation();
    return op ? op->CheckBatchGetStatus(stream)
              : Status::NotInitialized("session not initialized");
  }
  Status DrainGet() override {
    auto* op = resources_->operation();
    return op ? op->DrainGet() : Status::OK();
  }
  Status ResolvePut(PutCompletion* result) override {
    auto* op = resources_->operation();
    return op ? op->ResolvePut(result) : Status::NotInitialized("session not initialized");
  }
  Status AcknowledgePut() override { return resources_->operation()->AcknowledgePut(); }
  bool TracksGetCompletion() const override { return true; }
  Status PreparePut(const BatchPutRequest& r, const BatchPutExecutionOptions& options) override {
    return resources_->operation()->PreparePut(r, options);
  }
  Status ExecuteBatchPut(const BatchPutExecRequest& r) override {
    return resources_->operation()->ExecuteBatchPut(r);
  }
  Status ExecuteBatchGet(const BatchGetExecRequest& r) override {
    return resources_->operation()->ExecuteBatchGet(r);
  }

 private:
  std::unique_ptr<PeerExecutionResources> standalone_;
  PeerExecutionResources* resources_;
};

}  // namespace kvaf
