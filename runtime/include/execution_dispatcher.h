#pragma once

#include <functional>
#include <mutex>
#include <shared_mutex>
#include <vector>
#include "access_operator.h"
#include "execution_context.h"

namespace kvaf {
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
  Status ExecuteBatchPut(const BatchPutExecRequest&) override;
  Status ExecuteBatchGet(const BatchGetExecRequest&) override;
  Status DrainGet() override;
  bool TracksGetCompletion() const override { return true; }
  Status Warmup(PutExecutionConfig, const ExecutionWarmup& = {});
  Status Warmup(GetExecutionConfig, const ExecutionWarmup& = {});
  Status BeginGetGraph(GetExecutionConfig, const GetGraphOptions&, std::uint64_t*) override;
  Status EndGetGraph(std::uint64_t, void*, bool) override;
  Status BeginGetOutput(GetExecutionConfig, const GetOutputOptions&, std::uint64_t*) override;
  Status EndGetOutput(std::uint64_t) override;
  Status Recover(PutExecutionConfig) override;
  Status Recover(GetExecutionConfig);
  Status ResolvePut(PutExecutionConfig, PutCompletion*) override;
  Status AcknowledgePut() override;
  const ExecutionPolicy& policy() const { return policy_; }
 private:
  Status DrainGetLocked();
  Status FinishOutputLocked();
  ExecutionPolicy policy_;
  Status policy_status_;
  RemoteRegionDesc region_{};
  bool ready_{false};
  mutable std::shared_mutex lifecycle_;
  std::mutex put_mutex_, get_mutex_;
  ExecutionResourceManager resources_;
  std::unique_ptr<PutExecutionContext> pending_put_;
  std::unique_ptr<GetExecutionContext> pending_get_;
  std::uint64_t graph_id_{0}, next_graph_id_{1};
  std::unique_ptr<GetExecutionContext> graph_context_;
  bool graph_captured_{false};
  std::uint64_t output_id_{0}, next_output_id_{1};
  GetOutputOptions output_options_;
  std::unique_ptr<GetExecutionContext> output_context_;
};
}  // namespace kvaf
