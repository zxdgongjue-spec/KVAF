#pragma once

#include <functional>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>
#include "access_operator.h"
#include "execution_context.h"

namespace kvaf {
class GetDispatchTrace;
// Policy and direction scheduling. ResourceManager owns sessions/workspaces;
// execution contexts retain typed exclusive leases until actual completion.
class ExecutionDispatcher final : public AccessOperator {
 public:
  using Factory = std::function<std::unique_ptr<AccessOperator>()>;
  explicit ExecutionDispatcher(ExecutionPolicy policy);
  ~ExecutionDispatcher() override;
  Status RegisterPut(PutExecutionConfig, Factory);
  Status RegisterGet(GetExecutionConfig, Factory);
  Status Initialize(const RemoteRegionDesc&) override;
  Status Shutdown() override;
  Status PreparePut(const BatchPutExecutionOptions&) override;
  Status PreparePut(const BatchPutRequest&, const BatchPutExecutionOptions&) override;
  Status CancelPreparedPut() override;
  Status ExecuteBatchPut(const BatchPutExecRequest&) override;
  Status ExecuteBatchGet(const BatchGetExecRequest&) override;
  Status DrainGet() override;
  bool HasPendingGet() const override;
  bool TracksGetCompletion() const override { return true; }
  Status Warmup(PutExecutionConfig, const ExecutionWarmup& = {});
  Status Warmup(GetExecutionConfig, const ExecutionWarmup& = {});
  Status BeginGetGraph(GetExecutionConfig, const GetGraphOptions&, std::uint64_t*) override;
  Status EndGetGraph(std::uint64_t, void*, bool) override;
  Status BeginGetOutput(GetExecutionConfig, const GetOutputOptions&, std::uint64_t*) override;
  Status EndGetOutput(std::uint64_t) override;
  Status BeginPutInput(PutExecutionConfig, const PutInputOptions&, std::uint64_t*) override;
  Status EndPutInput(std::uint64_t) override;
  Status Recover(PutExecutionConfig) override;
  Status Recover(GetExecutionConfig) override;
  Status ResolvePut(PutExecutionConfig, PutCompletion*) override;
  Status AcknowledgePut() override;
  Status ConfigureGetProfiling(std::optional<GetExecutionConfig>, bool enabled, bool detailed);
  Status CopyGetProfiles(std::optional<GetExecutionConfig>, std::vector<BatchGetKernelProfile>*,
                         std::uint64_t graph_lease = 0);
  Status CheckGetStatus(std::optional<GetExecutionConfig>, void* stream, std::uint64_t graph_lease = 0);
  Status CopyGetProfileDetails(GetExecutionConfig, bool drain_pending,
                               const std::function<Status(const AccessOperator&)>&);
  const ExecutionPolicy& policy() const { return policy_; }
 private:
  PutExecutionContext* FindPutInstanceLocked(PutExecutionConfig) const;
  Status PreparePutInstanceLocked(PutExecutionConfig, const BatchPutRequest*, const ExecutionWarmup&,
                                  PutExecutionContext**, PutDispatchTiming* = nullptr);
  Status PreparePutLocked(PutExecutionConfig, const BatchPutRequest&, const BatchPutExecutionOptions&);
  Status ClosePutInstanceLocked(PutExecutionContext*);
  struct GetPreparationTiming {
    std::uint64_t context_ns{}, describe_ns{}, acquire_ns{}, bind_ns{}, warmup_ns{};
  };
  GetExecutionContext* FindGetInstanceLocked(GetExecutionConfig) const;
  Status PrepareGetInstanceLocked(GetExecutionConfig, const BatchGetRequest*, const ExecutionWarmup&,
                                  GetExecutionContext**, GetPreparationTiming* = nullptr);
  Status CloseGetInstanceLocked(GetExecutionContext*);
  bool HasGraphNodesLocked(const GetExecutionContext*) const;
  Status DrainGetLocked();
  Status FindGraphNode(GetExecutionContext*, std::uint64_t, GetGraphNode**);
  Status CheckGetProfileReadableLocked(GetExecutionContext*, bool drain_pending);
  Status FinishOutputLocked(GetExecutionContext&);
  Status ValidateInputLocked(const BatchPutRequest&, const BatchPutExecutionOptions&,
                             PutExecutionConfig) const;
  ExecutionPolicy policy_;
  Status policy_status_;
  RemoteRegionDesc region_{};
  bool ready_{false};
  mutable std::shared_mutex lifecycle_;
  std::mutex put_mutex_;
  mutable std::mutex get_mutex_;
  ExecutionResourceManager resources_;
  std::unique_ptr<GetDispatchTrace> get_trace_;
  std::vector<std::unique_ptr<PutExecutionContext>> put_instances_;
  PutExecutionContext* pending_put_{nullptr};
  PutExecutionContext* prepared_put_{nullptr};
  // Borrowed only between Prepare and Execute/Cancel, never through recovery.
  const BatchPutRequest* prepared_request_{nullptr};
  void* prepared_stream_{nullptr};
  std::uint64_t input_id_{0}, next_input_id_{1};
  PutInputOptions input_options_;
  PutExecutionContext* input_context_{nullptr};
  // A failed Begin may still own a partially registered input until shutdown.
  PutExecutionContext* failed_input_context_{nullptr};
  std::vector<std::unique_ptr<GetExecutionContext>> get_instances_;
  GetExecutionContext* pending_get_{nullptr};
  std::uint64_t next_graph_id_{1};
  std::unordered_map<std::uint64_t, GetGraphNode> graph_nodes_;
  std::uint64_t next_output_id_{1};
};
}  // namespace kvaf
