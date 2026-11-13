#pragma once

#include "access_operator.h"
#include "execution_view.h"
#include "acl/acl.h"
#include "kvaf_aiv_batch_get.h"
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
#include "kvaf_aiv_batch_put.h"
#endif

namespace kvaf {
class AivExecutionResources;
// Physical execution slots, owned by the resource domain and reused under its lease.
struct AivGetWorkspace {
  void* status{};
  std::size_t status_bytes{};
  void* queue_state{};
  std::size_t queue_state_bytes{};
};
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
struct AivPutWorkspace {
  void* metadata{};
  void* core_status{};
  std::size_t metadata_capacity{};
  void* host_metadata{};
  std::size_t host_metadata_capacity{};
  KvafBatchPutCoreStatus* host_core_status{};
  std::size_t host_core_status_count{};
};
#endif
// Request/graph state is separate from the communication session. The owner
// pointer is an identity check, never an owning reference.
struct AivInvocation final : ExecutionState {
  explicit AivInvocation(const AivExecutionResources* value) : owner(value) {}
  const AivExecutionResources* owner;
  aclrtStream pending_get{};
  KvafBatchGetLocalBuffer* pending_buffer{};
  bool get_issued{};
  std::uint64_t get_entity{};
  MutableBufferView get_output;
  std::size_t get_capacity{};
  KvafBatchGetLocalBuffer* output_buffer{};
  std::uint64_t output_entity{};
  GetOutputOptions output_options;
  bool output_retiring{false};
  KvafBatchGetLocalBuffer* graph_buffer{};
  std::uint64_t graph_entity{};
  GetGraphOptions graph_options;
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
  aclrtStream pending_put{};
  KvafBatchPutLocalBuffer* pending_put_buffer{};
  std::size_t put_rows{};
  ConstBufferView put_input;
  bool put_issued{};
  // Exclusive registration retained until the caller ends its input lease.
  KvafBatchPutLocalBuffer* input_buffer{};
  PutInputOptions input_options;
  bool input_retiring{false};
#endif
  PutCompletion put_completion;
};
// Backend resource objects own the registration lifecycle. Algorithms receive
// these through direction-specific borrowed views, never create/close transports.
struct AivGetSession {
  KvafBatchGetTransport* transport{};
  AivGetWorkspace workspace;
  int RetireOutput(KvafBatchGetLocalBuffer*);
  void ReleaseOutput(KvafBatchGetLocalBuffer*);
 private:
  friend class AivExecutionResources;
  int AcquireOutput(void*, std::size_t, KvafBatchGetLocalBuffer**, std::uint64_t*);
};
struct AivPutSession {
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
  KvafBatchPutTransport* transport{};
  AivPutWorkspace workspace;
  int RetireInput(KvafBatchPutLocalBuffer*);
  void ReleaseInput(KvafBatchPutLocalBuffer*);
 private:
  friend class AivExecutionResources;
  Status Reserve(std::size_t);
  int AcquireInput(void*, std::size_t, KvafBatchPutLocalBuffer**);
#endif
};
// Immutable launch options, not the resource-creation configuration.
struct AivLaunchOptions {
  std::uint32_t npu_core_count{}, jettys_per_core{}, batch_get_max_inflight_total{};
  bool managed_completion{};
};
struct UrmaAivGetView {
  ExecutionViewCommon common;
  AivLaunchOptions options;
  AivGetWorkspace& workspace;
  KvafBatchGetLocalBuffer* buffer{};
  std::uint64_t channel_entity{};
};
struct UrmaAivPutView {
  ExecutionViewCommon common;
  AivLaunchOptions options;
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
  bool device_execution{};
  AivPutWorkspace& workspace;
  KvafBatchPutLocalBuffer* buffer{};
#endif
};
class AivExecutionResources final : public ExecutionResourceOwner {
 public:
  explicit AivExecutionResources(AccessOperatorConfig);
  ~AivExecutionResources() override;
  Status Initialize(const RemoteRegionDesc&) override;
  Status Warmup(const ExecutionWarmup&) override;
  Status Close() override;
  UrmaAivGetView GetView();
  UrmaAivPutView PutView();
  Status PrepareGetResources(const BatchGetExecRequest&, AivInvocation&, UrmaAivGetView*);
  Status PreparePutResources(const BatchPutExecRequest&, AivInvocation&, UrmaAivPutView*);
  Status PrepareGraph(const GetGraphOptions&, AivInvocation&);
  Status PrepareOutput(const GetOutputOptions&, AivInvocation&);
  Status PrepareInput(const PutInputOptions&, AivInvocation&);
  std::unique_ptr<ExecutionState> CreateState() const;
  Status Bind(ExecutionState*);
  AivInvocation& invocation();
 private:
  struct State;
  std::unique_ptr<State> state_;
};
// Stateless execution algorithms; views cannot be interchanged across directions.
class UrmaAivGetOperator {
 public:
  static Status Validate(const BatchGetExecRequest&, UrmaAivGetView, AivInvocation&,
                         std::uint64_t* capture_query_ns = nullptr);
  static Status Launch(const BatchGetExecRequest&, UrmaAivGetView, AivInvocation&);
  static Status Drain(UrmaAivGetView, AivInvocation&);
  static Status PrepareGraph(const GetGraphOptions&, UrmaAivGetView, AivInvocation&);
  static Status FinishGraph(void*, bool, UrmaAivGetView, AivInvocation&);
  static Status SetProfiling(bool, UrmaAivGetView);
  static Status CopyProfiles(std::vector<BatchGetKernelProfile>*, UrmaAivGetView);
};
class UrmaAivPutOperator {
 public:
  static Status Validate(const BatchPutExecRequest&, UrmaAivPutView, AivInvocation&);
  static Status Launch(const BatchPutExecRequest&, UrmaAivPutView, AivInvocation&);
  static Status Resolve(PutCompletion*, UrmaAivPutView, AivInvocation&);
  static Status Acknowledge(UrmaAivPutView, AivInvocation&);
 private:
  static Status Execute(const BatchPutExecRequest&, UrmaAivPutView, AivInvocation&);
};
}  // namespace kvaf
