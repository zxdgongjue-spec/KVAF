#pragma once
#include "gather_workspace.h"
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
  std::uint32_t data_jettys{1};
  std::vector<std::uint32_t> gather_candidates;
  GatherChunkConfig chunks{};
  std::uint64_t gather_ddr_bytes{0}, scatter_hbm_bytes{0};
  std::uint32_t gather_read_wave_wrs{kGatherDefaultReadWaveWrs};
  std::optional<std::uint32_t> gather_address_groups_per_post;
};
// Runtime adapter with independent UMDK Put/Get sessions;
// cpu_push_scatter owns an independent Host-side direct HCCP target.
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
