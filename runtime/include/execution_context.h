#pragma once

#include "execution_resource_manager.h"
#include "meta_manager.h"

namespace kvaf {
// One resident instance per Put configuration. Only the execution lease and
// completion state last through the metadata transaction. Request vectors are
// borrowed during Launch, never kept
// for recovery; providers stage any asynchronous metadata in their own storage.
// Buffer storage remains governed by the public C++/Python ownership contract.
struct PutExecutionContext {
  PutResourceReference resources;
  PutResourceLease lease;
  PutExecutionConfig config;
  std::unique_ptr<ExecutionState> state;
  PutCompletion completion;
  bool launched{false};
  bool bound{false};
  bool cleanup_only{false};
  bool warmed{false};
  std::uint32_t warmed_request_capacity{0}, warmed_capacity{0};
  void* warmed_stream{nullptr};
  PutResourceView view() const { return resources ? resources.view() : lease.view(); }
  Status Bind() {
    state = view().CreateState();
    auto s = view().Bind(state.get());
    bound = s.ok();
    return s;
  }
  ~PutExecutionContext() { if (bound) (void)view().Bind(nullptr); }
  void ClearExecution() {
    completion = {};
    launched = false;
    lease = {};
  }
};
// One resident instance per Get configuration. The reference and state survive
// completion; lease/request/table/options belong only to current execution.
struct GetExecutionContext {
  GetResourceReference resources;
  GetResourceLease lease;
  GetExecutionConfig config;
  BatchGetRequest request;
  AccessTableView table;
  BatchGetExecutionOptions options;
  std::unique_ptr<ExecutionState> state;
  bool bound{false};
  bool cleanup_only{false};
  bool warmed{false};
  std::uint32_t warmed_request_capacity{0}, warmed_capacity{0};
  void* warmed_stream{nullptr};
  std::uint64_t output_id{0};
  GetOutputOptions output_options;
  std::vector<BatchGetKernelProfile> last_graph_profile;
  bool profiling_enabled{false};
  bool graph_recovery_required{false};
  GetResourceView view() const { return resources ? resources.view() : lease.view(); }
  Status Bind() {
    state = view().CreateState();
    auto s = view().Bind(state.get());
    bound = s.ok();
    return s;
  }
  ~GetExecutionContext() { if (bound) (void)view().Bind(nullptr); }
  void ClearRequest() {
    request = {};
    table = {};
    options = {};
    lease = {};
  }
  BatchGetExecRequest invocation() const { return {table, &request, options}; }
};

struct GetGraphNode {
  GetExecutionContext* instance{};
  GetGraphOptions options;
  std::unique_ptr<ExecutionState> state;
  bool captured{false};
  bool ending{false};
  bool domain_closed{false};
};
}  // namespace kvaf
