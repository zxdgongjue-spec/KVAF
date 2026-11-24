#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <limits>
#include <string>
#include <vector>

#include "requests.h"
#include "execution_resources.h"
#include "batch_get_profile.h"
#include "status.h"
#include "types.h"

namespace kvaf::peer {
class PeerLink;
}

namespace kvaf {

struct PutAccessDesc;
struct BatchGetKernelProfile;
using PeerLink = peer::PeerLink;

struct BatchPutExecRequest {
  const BatchPutRequest* request{nullptr};
  const PutAccessDesc* put_desc{nullptr};
  BatchPutExecutionOptions execution;
};

// KV Access Operator 是 Runtime 的数据面依赖。Runtime 管理远端 DDR
// 生命周期和 Meta 可见性；Operator 负责执行和完成解析，请求级入口的资源
// 所有权与单次状态分别由 ResourceManager 和 ExecutionContext 持有。
class AccessOperator {
 public:
  virtual ~AccessOperator() = default;

  virtual Status Initialize(const RemoteRegionDesc& remote_region) = 0;
  virtual Status Shutdown() = 0;

  // Hardware providers transfer ownership once to ResourceManager and retain
  // only their concrete borrowed view. Synchronous mock providers may return null.
  virtual std::unique_ptr<ExecutionResourceOwner> TakeResources() { return {}; }
  // Pure request description: no registration, allocation or device work. The
  // registry fixes the provider/ABI/profile; the invocation supplies capacity.
  virtual Status DescribeRequirements(const BatchPutRequest& request,
                                      std::uint32_t* capacity) const {
    if (!capacity || request.keys.size() > std::numeric_limits<std::uint32_t>::max())
      return Status::InvalidArgument("invalid Put request capacity");
    *capacity = static_cast<std::uint32_t>(request.keys.size());
    return ValidateRequestCapacity(*capacity);
  }
  virtual Status DescribeRequirements(const BatchGetRequest& request,
                                      std::uint32_t* capacity) const {
    if (!capacity || request.indices.bytes % sizeof(std::int32_t) ||
        request.indices.bytes / sizeof(std::int32_t) > std::numeric_limits<std::uint32_t>::max())
      return Status::InvalidArgument("invalid Get request capacity");
    *capacity = static_cast<std::uint32_t>(request.indices.bytes / sizeof(std::int32_t));
    return ValidateRequestCapacity(*capacity);
  }
  // Check the immutable resource profile before granting a request lease.
  virtual Status ValidateRequestCapacity(std::uint32_t) const { return Status::OK(); }
  virtual std::unique_ptr<ExecutionState> CreateExecutionState() { return {}; }
  // A failed bind must leave the previously bound state intact. Passing null
  // restores the default state before the invocation's state object is destroyed.
  virtual Status BindExecutionState(ExecutionState* state) {
    return state ? Status::InvalidArgument("provider does not accept execution state") : Status::OK();
  }
  // Resolve the original submitted WRITE; never launch or replay it here.
  // A confirmed result is retained until Runtime finalizes metadata and ACKs.
  virtual Status ResolvePut(PutCompletion*) {
    return Status::NotImplemented("provider cannot resolve pending Put completion");
  }
  virtual Status ResolvePut(PutExecutionConfig, PutCompletion* completion) {
    return ResolvePut(completion);
  }
  virtual Status AcknowledgePut() { return Status::OK(); }
  virtual Status Recover(PutExecutionConfig) {
    return Status::NotImplemented("resource recovery requires execution dispatcher");
  }
  virtual Status Recover(GetExecutionConfig) {
    return Status::NotImplemented("resource recovery requires execution dispatcher");
  }

  // Synchronous implementations require no completion preparation.
  virtual Status PreparePut(const BatchPutExecutionOptions&) { return Status::OK(); }
  virtual Status PreparePut(const BatchPutRequest&, const BatchPutExecutionOptions& options) {
    return PreparePut(options);
  }
  // Cancel only an unsubmitted PreparePut reservation. Keep resident resources,
  // and leave submitted/pending execution and its acknowledgement untouched.
  // Idempotent and host-only: Runtime also calls this after Execute consumed it.
  virtual Status CancelPreparedPut() { return Status::OK(); }
  virtual Status Warmup(const ExecutionWarmup&) { return Status::OK(); }
  virtual Status PreparePutInput(const PutInputOptions&) {
    return Status::NotImplemented("provider does not support reusable input bindings");
  }
  virtual Status FinishPutInput() {
    return Status::NotImplemented("provider does not support reusable input bindings");
  }
  virtual Status BeginPutInput(PutExecutionConfig, const PutInputOptions&, std::uint64_t*) {
    return Status::NotImplemented("input bindings require execution dispatcher");
  }
  virtual Status EndPutInput(std::uint64_t) {
    return Status::NotImplemented("input bindings require execution dispatcher");
  }
  virtual Status DrainGet() { return Status::OK(); }
  // True while a request or failed teardown may still own caller buffers.
  // Completion-tracking adapters must positively establish safe retirement;
  // an adapter without a query implementation conservatively retains owners.
  virtual bool HasPendingGet() const { return TracksGetCompletion(); }
  virtual Status SetBatchGetKernelProfiling(bool) {
    return Status::NotImplemented("provider does not support Get profiling");
  }
  virtual Status SetBatchGetKernelProfiling(bool enabled, bool detailed) {
    if (detailed) return Status::InvalidArgument("detailed per-row profiling requires UBMEM");
    return SetBatchGetKernelProfiling(enabled);
  }
  virtual Status CopyBatchGetKernelProfiles(std::vector<BatchGetKernelProfile>*) const {
    return Status::NotImplemented("provider does not support Get profiling");
  }
  virtual Status CheckBatchGetStatus(void*) {
    return Status::NotImplemented("explicit Get status checking requires an AICPU backend");
  }
  virtual Status PrepareGetOutput(const GetOutputOptions&) {
    return Status::NotImplemented("provider does not support reusable output bindings");
  }
  virtual Status FinishGetOutput() {
    return Status::NotImplemented("provider does not support reusable output bindings");
  }
  virtual Status BeginGetOutput(GetExecutionConfig, const GetOutputOptions&, std::uint64_t*) {
    return Status::NotImplemented("output bindings require execution dispatcher");
  }
  virtual Status EndGetOutput(std::uint64_t) {
    return Status::NotImplemented("output bindings require execution dispatcher");
  }
  virtual Status PrepareGetGraph(const GetGraphOptions&) {
    return Status::NotImplemented("provider does not support graph leases");
  }
  virtual Status FinishGetGraph(void*, bool) {
    return Status::NotImplemented("provider does not support graph leases");
  }
  // True only after fencing the graph stream and establishing that the
  // shared communication domain must be closed before releasing its pins.
  virtual bool GetGraphNeedsRecovery() const { return false; }
  virtual Status BeginGetGraph(GetExecutionConfig, const GetGraphOptions&, std::uint64_t*) {
    return Status::NotImplemented("graph leases require execution dispatcher");
  }
  virtual Status EndGetGraph(std::uint64_t, void*, bool) {
    return Status::NotImplemented("graph leases require execution dispatcher");
  }
  virtual bool TracksGetCompletion() const { return false; }
  virtual Status ExecuteBatchGet(const BatchGetExecRequest& request) = 0;
  // Request vectors and destination descriptors are borrowed only until this
  // method returns. Stage asynchronous metadata in provider-owned storage;
  // ResolvePut/AcknowledgePut/Shutdown must not dereference these descriptions.
  // The caller still keeps the input allocation alive until completion resolves.
  virtual Status ExecuteBatchPut(const BatchPutExecRequest& request) = 0;

};

// Configuration for the current NPU-backed default implementation. Transport
// 不绑定 ACL stream；每次 BatchGet 都通过 request.execution 接收调用方 stream。
struct AccessOperatorConfig {
  // BatchGet remains available whenever this kernel is configured. BatchPut
  // uses a separate AIV binary and transport because the two current algorithms do
  // not establish a shared Direct-HCCP ABI.
  std::string batch_get_kernel_object_path;
  std::string batch_put_kernel_object_path;
  // Zero selects the link defaults: local Get 2 x 2, remote Get 4 x 1, Put 2 x 2.
  std::uint32_t npu_core_count{0};
  std::uint32_t jettys_per_core{0};
  // Per-launch total READ budget; 0 selects automatic SQ/CQ capacity windows.
  std::uint32_t batch_get_max_inflight_total{2048U};
  // Zero inherits an explicit jettys_per_core (capped at 2) or the Put default.
  std::uint32_t batch_put_jettys_per_core{0};
  // Legacy single-EID input. Kept for source compatibility; peer_local_eids
  // takes precedence when it is non-empty.
  std::array<std::uint8_t, 16> peer_local_eid{};
  // Runtime may supply more than one selected local EID. BatchGet creates one
  // RaCtx per EID and stripes core/Jetty workers over those paths; BatchPut
  // continues to use peer_local_eid above.
  std::vector<std::array<std::uint8_t, 16>> peer_local_eids{};
  std::shared_ptr<PeerLink> peer_link;
  bool enable_get{true};
  // Ordered per-FE execution sessions for remote Direct Get. Empty uses peer_link.
  std::vector<std::shared_ptr<PeerLink>> peer_get_links;
  bool peer_local{true};
};

struct BatchGetKernelProfile {
  // AICPU workers use a shared monotonic nanosecond clock. AIV backends keep
  // their existing cycle fields; timestamps from different clocks are not mixed.
  std::uint64_t device_start_ns{0};
  std::uint64_t device_finish_ns{0};
  // Preserve the stages already collected by the AICPU profile buffer.
  BatchGetWorkerProfile aicpu_profile{};
  std::uint64_t kernel_start_cycle{0};
  std::uint64_t setup_done_cycle{0};
  std::uint64_t compact_done_cycle{0};
  std::uint64_t first_doorbell_cycle{0};
  std::uint64_t post_done_cycle{0};
  std::uint64_t completion_done_cycle{0};
  std::uint64_t kernel_finish_cycle{0};
  std::uint64_t active_requests{0};
  std::uint64_t total_posts{0};
  std::array<std::uint64_t, 64> jetty_completion_wait_cycles{};
  std::array<std::uint64_t, 64> jetty_completion_polls{};
  std::uint64_t submitted_group_count{0};
  std::uint64_t completed_group_count{0};
  std::uint64_t completion_count{0};
  std::uint64_t post_active_cycles{0};
  std::uint64_t drain_wait_cycles{0};
  std::uint64_t peak_submitted_not_drained{0};
  bool ubmem_profile_detail{false};
  std::uint64_t ubmem_setup_cycles{0};
  std::uint64_t ubmem_address_cycles{0};
  std::uint64_t ubmem_submit_cycles{0};
  std::uint64_t ubmem_drain_cycles{0};
};

// Compatibility adapter for the URMA AIV algorithms and resource provider.
//
// Standalone callers own an AivExecutionResources provider through this adapter.
// Managed callers transfer it to ResourceManager before initialization.
// ExecuteBatchGet obtains a LocalBuffer from the caller-owned output in the
// request. Direct-HCCP queue/status buffers remain internal to the operator;
// launch ordering is defined by the caller stream passed with each request.
// ExecuteBatchPut uses the Direct-HCCP AIV eager path when a BatchPut kernel
// path is configured; otherwise it retains the synchronous ACL D2H mock for
// local/BatchGet-only builds. Both paths consume the packed variable-size
// BatchPutRequest contract and leave transaction visibility to Runtime.
// This class is built only with KVAF_ENABLE_AIV_BATCH_GET=ON.
class AivExecutionResources;
class DefaultAccessOperator final : public AccessOperator {
 public:
  explicit DefaultAccessOperator(AccessOperatorConfig config);
  ~DefaultAccessOperator() override;
  std::unique_ptr<ExecutionResourceOwner> TakeResources() override;
  std::unique_ptr<ExecutionState> CreateExecutionState() override;
  Status BindExecutionState(ExecutionState*) override;
  Status ResolvePut(PutCompletion*) override;
  Status AcknowledgePut() override;

  DefaultAccessOperator(const DefaultAccessOperator&) = delete;
  DefaultAccessOperator& operator=(const DefaultAccessOperator&) = delete;

  Status Initialize(const RemoteRegionDesc& remote_region) override;
  Status Shutdown() override;
  Status ExecuteBatchGet(const BatchGetExecRequest& request) override;
  Status ExecuteBatchPut(const BatchPutExecRequest& request) override;

  Status DrainGet() override;
  Status Warmup(const ExecutionWarmup&) override;
  Status PrepareGetGraph(const GetGraphOptions&) override;
  Status FinishGetGraph(void*, bool) override;
  Status PrepareGetOutput(const GetOutputOptions&) override;
  Status FinishGetOutput() override;
  Status PreparePutInput(const PutInputOptions&) override;
  Status FinishPutInput() override;

  // Profiling is opt-in and uses the reserved per-core status area. Callers
  // must complete the BatchGet stream before copying the latest snapshot.
  Status SetBatchGetKernelProfiling(bool enabled) override;
  Status CopyBatchGetKernelProfiles(
      std::vector<BatchGetKernelProfile>* profiles) const override;

 private:
  // Compatibility ownership for standalone users; managed entry transfers it.
  std::unique_ptr<AivExecutionResources> owned_resources_;
  AivExecutionResources* resources_;
};

}  // namespace kvaf
