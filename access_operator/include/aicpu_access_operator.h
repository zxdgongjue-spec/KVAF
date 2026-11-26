#pragma once
#include "access_operator.h"
#include <array>
namespace kvaf {
struct UmdkAccessOperatorConfig {
  std::string kernel_json;
  std::uint32_t workers{1U};
  std::shared_ptr<PeerLink> peer_link;
  std::uint32_t token_bytes{0U};
  std::string batch_get_mode{"gather_read"};
  std::array<std::uint8_t, 16> local_eid{};
  bool enable_get{true};
  std::uint32_t sge_per_wr{0};
  bool sgl_payload_only{false};
  bool sgl_pipeline{false};
  bool sgl_push_chunked{false};
  std::uint32_t sgl_push_head_rows{1024};
  bool sgl_push_cooperative{false};
};
// Runtime adapter with independent UMDK Put/Get sessions;
// cpu_push_scatter/cpu_sgl_write own an independent Host-side direct HCCP target.
class AicpuExecutionResources;
struct AicpuInvocation;

class UmdkAccessOperator final : public AccessOperator {
 public:
  explicit UmdkAccessOperator(UmdkAccessOperatorConfig config);
  ~UmdkAccessOperator() override;
  std::unique_ptr<ExecutionResourceOwner> TakeResources() override;
  std::unique_ptr<ExecutionState> CreateExecutionState() override;
  Status BindExecutionState(ExecutionState*) override;
  Status ResolvePut(PutCompletion*) override;
  Status AcknowledgePut() override;
  Status Initialize(const RemoteRegionDesc&) override;
  Status Shutdown() override;
  Status DrainGet() override;
  Status PrepareGetGraph(const GetGraphOptions&) override;
  Status FinishGetGraph(void*, bool) override;
  bool GetGraphNeedsRecovery() const override;
  Status PrepareGetOutput(const GetOutputOptions&) override;
  Status FinishGetOutput() override;
  Status PreparePutInput(const PutInputOptions&) override;
  Status FinishPutInput() override;
  Status Warmup(const ExecutionWarmup&) override;
  Status ExecuteBatchPut(const BatchPutExecRequest&) override;
  Status ExecuteBatchGet(const BatchGetExecRequest&) override;
  Status CheckBatchGetStatus(void* stream) override;
  Status SetBatchGetKernelProfiling(bool enabled) override;
  Status CopyBatchGetKernelProfiles(std::vector<BatchGetKernelProfile>* profiles) const override;

 private:
  std::unique_ptr<AicpuExecutionResources> owned_resources_;
  AicpuExecutionResources* resources_;
};
}  // namespace kvaf
