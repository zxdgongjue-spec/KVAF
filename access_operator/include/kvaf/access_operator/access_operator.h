#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kvaf/requests.h"
#include "kvaf/status.h"
#include "kvaf/types.h"

namespace kvaf {

struct PutAccessDesc;

struct BatchPutExecRequest {
  const BatchPutRequest* request{nullptr};
  const PutAccessDesc* put_desc{nullptr};
  BatchPutExecutionOptions execution;
};

// KV Access Operator 是 Runtime 的数据面依赖。Runtime 管理远端 DDR
// 生命周期和 Meta 可见性；Operator 只负责初始化链路并执行同步搬运。
class AccessOperator {
 public:
  virtual ~AccessOperator() = default;

  virtual Status Initialize(const RemoteRegionDesc& remote_region) = 0;
  virtual Status Shutdown() = 0;

  virtual Status ExecuteBatchGet(const BatchGetExecRequest& request) = 0;
  virtual Status ExecuteBatchPut(const BatchPutExecRequest& request) = 0;
};

// Configuration for the current NPU-backed default implementation. Transport
// 不绑定 ACL stream；每次 BatchGet 都通过 request.execution 接收调用方 stream。
struct AccessOperatorConfig {
  // BatchGet remains available whenever this kernel is configured. BatchPut
  // uses a separate AIV binary and transport because the two current demos do
  // not establish a shared Direct-HCCP ABI.
  std::string batch_get_kernel_object_path;
  std::string batch_put_kernel_object_path;
  std::uint32_t npu_core_count{0};
  // Direct-HCCP Jettys assigned to each AIV core.
  std::uint32_t jettys_per_core{2};
};

struct BatchGetKernelProfile {
  std::uint64_t kernel_start_cycle{0};
  std::uint64_t setup_done_cycle{0};
  std::uint64_t compact_done_cycle{0};
  std::uint64_t first_doorbell_cycle{0};
  std::uint64_t post_done_cycle{0};
  std::uint64_t completion_done_cycle{0};
  std::uint64_t kernel_finish_cycle{0};
  std::uint64_t active_requests{0};
  std::uint64_t total_posts{0};
  std::array<std::uint64_t, 2> jetty_completion_wait_cycles{};
  std::array<std::uint64_t, 2> jetty_completion_polls{};
};

// A bound output is a cold-path Direct-HCCP registration for one fixed HBM
// output range. The ID deliberately has no scheduling semantics: it only
// identifies the cached local-memory registration and its channel entity.
// It does not own the HBM allocation; the caller must keep that allocation
// alive and must not release a binding until every stream that can read its
// output has completed.
using BatchGetOutputBindingId = std::uint64_t;

// Default synchronous NPU data-plane implementation.
//
// Initialize creates reusable BatchGet and (when configured) BatchPut
// transports for Runtime-owned Host DDR.
// ExecuteBatchGet obtains a LocalBuffer from the caller-owned output in the
// request. Direct-HCCP queue/status buffers remain internal to the operator;
// launch ordering is defined by the caller stream passed with each request.
// ExecuteBatchPut uses the Direct-HCCP AIV eager path when a BatchPut kernel
// path is configured; otherwise it retains the synchronous ACL D2H mock for
// local/BatchGet-only builds. Both paths consume the packed variable-size
// BatchPutRequest contract and leave transaction visibility to Runtime.
// This class is built only with KVAF_ENABLE_AIV_BATCH_GET=ON.
class DefaultAccessOperator final : public AccessOperator {
 public:
  explicit DefaultAccessOperator(AccessOperatorConfig config);
  ~DefaultAccessOperator() override;

  DefaultAccessOperator(const DefaultAccessOperator&) = delete;
  DefaultAccessOperator& operator=(const DefaultAccessOperator&) = delete;

  Status Initialize(const RemoteRegionDesc& remote_region) override;
  Status Shutdown() override;
  Status ExecuteBatchGet(const BatchGetExecRequest& request) override;
  Status ExecuteBatchPut(const BatchPutExecRequest& request) override;

  // Split the eager Get path into a cold output-registration phase and a hot
  // launch phase. Stage-5 fixed slots bind their HBM output once, so the
  // submit lane never performs LocalBufferAcquire on a decode critical path.
  // These are deliberately not general concurrent APIs: Bind/Release must
  // run while the native lane is idle or stopped, and ExecuteBoundBatchGet is
  // reserved for that lane's single worker. A host mutex would not make the
  // underlying async Direct-HCCP queues safe for multiple workers.
  Status BindBatchGetOutput(const MutableBufferView& output,
                            BatchGetOutputBindingId* binding_id);
  Status ReleaseBatchGetOutput(BatchGetOutputBindingId binding_id);
  Status ExecuteBoundBatchGet(const BatchGetExecRequest& request,
                              BatchGetOutputBindingId binding_id);

  // Cold-bind the exact two-worker Direct-HCCP state that a fused device
  // operator needs. `control` is a caller-owned NPU uint64[16] buffer; KVAF
  // writes KvafFusedBatchGetControl into it after registering `output`.
  // The binding is intentionally exclusive with ordinary and native-lane
  // BatchGet because all paths would otherwise mutate the same SQ/CQ state.
  // The NPU Runtime binding also freezes BatchPut, BatchFree and AccessTable
  // prepare/release for this interval, so the graph keeps a stable DDR
  // payload and DeviceAccessTable view.
  // Before binding, the caller must complete any earlier ordinary Get from
  // this Runtime; the caller must wait for every fused reader before release.
  Status BindFusedBatchGetControl(const MutableBufferView& output,
                                  const MutableBufferView& control);
  Status ReleaseFusedBatchGetControl();

  // Profiling is opt-in and uses the reserved per-core status area. Callers
  // must complete the BatchGet stream before copying the latest snapshot.
  Status SetBatchGetKernelProfiling(bool enabled);
  Status CopyBatchGetKernelProfiles(
      std::vector<BatchGetKernelProfile>* profiles) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kvaf
