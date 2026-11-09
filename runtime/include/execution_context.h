#pragma once

#include "execution_resource_manager.h"
#include "meta_manager.h"

namespace kvaf {
// One context owns the exclusive direction lease and copied request metadata
// until device completion AND (for Put) the metadata transaction decision.
// Buffer storage remains governed by the public C++/Python ownership contract.
struct PutExecutionContext {
  PutResourceLease lease;
  PutExecutionConfig config;
  BatchPutRequest request;
  PutAccessDesc destination;
  BatchPutExecutionOptions options;
  std::unique_ptr<ExecutionState> state;
  PutCompletion completion;
  bool launched{false};
  bool bound{false};
  Status Bind() {
    state = lease.view().CreateState();
    auto s = lease.view().Bind(state.get());
    bound = s.ok();
    return s;
  }
  ~PutExecutionContext() { if (bound) (void)lease.view().Bind(nullptr); }
  BatchPutExecRequest invocation() const { return {&request, &destination, options}; }
};
struct GetExecutionContext {
  GetResourceLease lease;
  GetExecutionConfig config;
  BatchGetRequest request;
  AccessTableView table;
  BatchGetExecutionOptions options;
  std::unique_ptr<ExecutionState> state;
  bool bound{false};
  bool cleanup_only{false};
  Status Bind() {
    state = lease.view().CreateState();
    auto s = lease.view().Bind(state.get());
    bound = s.ok();
    return s;
  }
  ~GetExecutionContext() { if (bound) (void)lease.view().Bind(nullptr); }
  BatchGetExecRequest invocation() const { return {table, &request, options}; }
};
}  // namespace kvaf
