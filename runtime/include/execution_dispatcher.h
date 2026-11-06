#pragma once

#include <functional>
#include <mutex>
#include <shared_mutex>
#include <vector>
#include "access_operator.h"

namespace kvaf {
// Transitional provider adapter: each factory creates one direction's owner.
// Providers must keep all mutable queues/workspaces inside that instance. Never
// return two wrappers around one mutable session. A future typed resource view
// can replace this adapter without changing per-call execution configuration.
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
  Status ExecuteBatchPut(const BatchPutExecRequest&) override;
  Status ExecuteBatchGet(const BatchGetExecRequest&) override;
  Status DrainGet() override;
  bool TracksGetCompletion() const override { return true; }
  Status Warmup(PutExecutionConfig);
  Status Warmup(GetExecutionConfig);
  const ExecutionPolicy& policy() const { return policy_; }
 private:
  template<class Config> struct Resource {
    Config config;
    Factory factory;
    std::unique_ptr<AccessOperator> owner;
    // A failed creation is not retried implicitly. Its owner is retained until
    // Shutdown confirms cleanup (including remote producers) has completed.
    Status failure;
    bool ready{false};
  };
  template<class Config> Status Acquire(std::vector<Resource<Config>>&, Config, AccessOperator**);
  Status DrainGetLocked();
  ExecutionPolicy policy_;
  Status policy_status_;
  RemoteRegionDesc region_{};
  bool ready_{false};
  mutable std::shared_mutex lifecycle_;
  std::mutex put_mutex_, get_mutex_;
  std::vector<Resource<PutExecutionConfig>> puts_;
  std::vector<Resource<GetExecutionConfig>> gets_;
  AccessOperator* pending_get_{nullptr};
  bool put_failed_{false};
};
}  // namespace kvaf
