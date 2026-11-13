#pragma once

#include "execution_resource_manager.h"
#include "meta_manager.h"

namespace kvaf {
// A Put context retains only resources and completion state through the metadata
// transaction decision. Request vectors are borrowed during Launch, never kept
// for recovery; providers stage any asynchronous metadata in their own storage.
// Buffer storage remains governed by the public C++/Python ownership contract.
struct PutExecutionContext {
  PutResourceLease lease;
  PutExecutionConfig config;
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
