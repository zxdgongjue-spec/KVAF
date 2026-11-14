#pragma once
#include "execution_resources.h"

namespace kvaf::ubmem {
// Compatibility adapter. UbmemResources owns the physical resources;
// UbmemPutOperator/UbmemGetOperator execute through prepared borrowed views.
// Execute methods consume RuntimeCore-validated requests and MetaManager's
// PutAccessDesc; call RuntimeCore::BatchGet/BatchPut as the public entry points.
class AccessOperator final : public kvaf::AccessOperator {
 public:
  explicit AccessOperator(OperatorOptions options);
  ~AccessOperator() override;
  std::unique_ptr<ExecutionResourceOwner> TakeResources() override;
  std::unique_ptr<ExecutionState> CreateExecutionState() override;
  Status BindExecutionState(ExecutionState*) override;
  Status ResolvePut(PutCompletion*) override;
  Status AcknowledgePut() override;
  Status Initialize(const RemoteRegionDesc& region) override;
  Status Shutdown() override;
  Status DrainGet() override;
  Status Warmup(const ExecutionWarmup&) override;
  Status PrepareGetGraph(const GetGraphOptions&) override;
  Status FinishGetGraph(void*, bool) override;
  Status ExecuteBatchGet(const BatchGetExecRequest& request) override;
  Status ExecuteBatchPut(const BatchPutExecRequest& request) override;
  Status SetBatchGetKernelProfiling(bool enabled, bool detailed = false) override;
  Status CopyBatchGetKernelProfiles(std::vector<BatchGetKernelProfile>* profiles) const override;
 private:
  std::unique_ptr<UbmemResources> owned_resources_;
  UbmemResources* resources_;
};
}  // namespace kvaf::ubmem
