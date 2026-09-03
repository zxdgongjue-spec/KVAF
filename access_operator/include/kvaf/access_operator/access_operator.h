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

// One entry in a cold, layer-keyed Direct-HCCP binding table.  The caller
// owns both NPU tensors and must keep them alive until the streaming table is
// released.  `control` receives KvafStreamingBatchGetControl, a distinct
// graph-stable uint64[16] ABI with a shared queue-pool lease table. Its
// queue_count can be smaller than the Runtime's total transport queue count.
//
// This is deliberately a lifecycle table, not a hot submission API.  All
// entries are registered and published together before graph capture.  KVAF
// rejects adding, replacing, or rebinding one layer while the table is live.
struct StreamingBatchGetLayerBinding {
  std::uint32_t layer_id{0};
  MutableBufferView output;
  MutableBufferView control;
};

// One graph-stable descriptor template for a single queue of a streaming
// layer. The address/valid table may be shared by multiple queue templates;
// indices/offsets/mask/output_indices are queue-private mutable slices that
// the LI kernel fills during replay. KVAF derives capacities and the private
// output-row base from the supplied buffers and cold-bound output arena.
struct StreamingBatchGetQueueTemplate {
  std::uint32_t queue_id{0};
  MutableBufferView address_table;
  MutableBufferView valid_table;
  MutableBufferView indices;
  MutableBufferView offsets;
  MutableBufferView mask;
  MutableBufferView output_indices;
};

// Templates for one already-bound layer. Every bound layer must be listed
// exactly once and must provide exactly one template for each queue id in
// [0, queue_count). This is a cold pre-capture API; no queue template can be
// replaced while the streaming table is live.
struct StreamingBatchGetLayerTemplates {
  std::uint32_t layer_id{0};
  std::vector<StreamingBatchGetQueueTemplate> queues;
};

// Cold binding for the P9 layer-local canonical-resident fused Get route.
// `resident_row_base/count` name a fixed interval of `output` that is already
// reserved by the caller for this layer's speculative + final-recall KV rows.
// The fused operator must join every Direct-HCCP post before it returns; KVAF
// never reclaims, remaps, or shares this resident interval with another
// layer/forward.
struct LayerResidentBatchGetBinding {
  std::uint32_t layer_id{0};
  MutableBufferView output;
  MutableBufferView control;
  std::uint32_t resident_row_base{0};
  std::uint32_t resident_row_count{0};
};

// Descriptor vectors have the established Direct-HCCP dense ABI.  Unlike a
// StreamingBatchGet queue, `output_indices` are absolute canonical resident
// rows in the binding's fixed interval, not queue-private rows.
using LayerResidentBatchGetQueueTemplate = StreamingBatchGetQueueTemplate;

// P10 uses the same cold binding and dense descriptor-vector storage as the
// hardened layer-resident path.  The forward name carries the tighter caller
// contract (one fixed resident arena for the fused forward); an alias avoids
// a second raw ACL/HCCP registration implementation that could drift from the
// already fixed bind/rollback path.
using ForwardResidentBatchGetBinding = LayerResidentBatchGetBinding;
using ForwardResidentBatchGetQueueTemplate =
    LayerResidentBatchGetQueueTemplate;

// Read-only cold-path introspection for the streaming queue pool. The three
// lifecycle counters are host lifecycle counters only; they deliberately do
// not claim a race-free live count of device-held leases during graph replay.
struct StreamingBatchGetLayerSnapshot {
  std::uint32_t layer_id{0};
  std::uint64_t output_address{0};
  std::uint64_t control_address{0};
  std::uint64_t channel_entity_address{0};
  std::uint64_t descriptor_template_address{0};
  std::uint64_t descriptor_template_bytes{0};
  std::uint64_t private_output_rows{0};
  std::uint64_t private_output_stride_bytes{0};
  bool templates_configured{false};
};

struct StreamingBatchGetControlSnapshot {
  std::uint64_t magic{0};
  std::uint64_t version{0};
  std::uint64_t bind_epoch{0};
  std::uint64_t bind_count{0};
  std::uint64_t release_count{0};
  std::uint64_t template_configure_count{0};
  std::uint64_t queue_lease_address{0};
  std::uint64_t descriptor_scratch_address{0};
  std::uint64_t queue_lease_bytes{0};
  std::uint64_t descriptor_scratch_bytes{0};
  std::uint32_t queue_count{0};
  bool active{false};
  std::vector<StreamingBatchGetLayerSnapshot> layers;
};

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

  // Cold-bind Direct-HCCP state that a fused device operator needs. `control`
  // is a caller-owned NPU uint64[16] buffer; KVAF writes the stable legacy
  // KvafFusedBatchGetControl ABI into it after registering `output`.
  //
  // The legacy two-argument overload preserves the original two-queue
  // contract. New users must pass queue_count explicitly. queue_count selects
  // the first queue_count transport queues and must be in
  // [1, npu_core_count]; the current device helper still requires exactly one
  // Jetty per selected queue.
  // The binding is intentionally exclusive with ordinary and native-lane
  // BatchGet because all paths would otherwise mutate the same SQ/CQ state.
  // The NPU Runtime binding also freezes BatchPut, BatchFree and AccessTable
  // prepare/release for this interval, so the graph keeps a stable DDR
  // payload and DeviceAccessTable view.
  // Before binding, the caller must complete any earlier ordinary Get from
  // this Runtime; the caller must wait for every fused reader before release.
  Status BindFusedBatchGetControl(const MutableBufferView& output,
                                  const MutableBufferView& control);
  Status BindFusedBatchGetControl(const MutableBufferView& output,
                                  const MutableBufferView& control,
                                  std::uint32_t queue_count);
  Status ReleaseFusedBatchGetControl();

  // Cold-bind a fixed set of per-layer Direct-HCCP outputs and controls for a
  // streaming fused operator. Each layer receives its own LocalBufferAcquire
  // registration and KvafStreamingBatchGetControl uint64[16] control.
  //
  // `queue_leases` is caller-owned graph-visible NPU int32[queue_count]
  // storage.  The device LI producer atomically acquires entries in this
  // exact allocation, so it must be the same allocation that the captured
  // graph passes to the streaming LI op.  Keeping leases internal to KVAF
  // would make control.queue_leases differ from the graph resource and would
  // either silently disable the device-side path or make its lifetime
  // unverifiable.
  // `layer_id` is a caller
  // supplied stable key; it must be unique in this table.  There is no
  // per-layer hot rebind: release the whole table after the terminal reader,
  // then bind a new table at a cold lifecycle boundary.
  //
  // The table is mutually exclusive with the single-control API, ordinary
  // BatchGet, native BatchGet lanes, and metadata-mutating Runtime paths for
  // the same reason as the legacy control: all would otherwise mutate the
  // same Direct-HCCP SQ/CQ state.
  Status BindStreamingBatchGetControl(
      const std::vector<StreamingBatchGetLayerBinding>& layers,
      const MutableBufferView& queue_leases,
      std::uint32_t queue_count);
  Status ConfigureStreamingBatchGetQueueTemplates(
      const std::vector<StreamingBatchGetLayerTemplates>& layers);
  Status ReleaseStreamingBatchGetControl();

  // Compact cold-path diagnostic for remote validation. It never touches a
  // Direct-HCCP queue and is safe only while the streaming table is live.
  // The returned text identifies queue_count, shared lease/scratch addresses,
  // and each fixed layer control/output registration.
  Status DescribeStreamingBatchGetControl(std::string* diagnostic) const;
  Status GetStreamingBatchGetControlSnapshot(
      StreamingBatchGetControlSnapshot* snapshot) const;

  // P9's layer-resident route is intentionally a single layer-local control,
  // not a table that survives graph replays.  The caller must bind one fixed
  // resident interval, configure fixed descriptor vectors, then use it only
  // from the one fused layer operator that drains all CQ work before return.
  // `queue_leases` is NPU int32[queue_count] storage used only while that
  // operator is executing; it does not describe resident-slot ownership.
  Status BindLayerResidentBatchGetControl(
      const LayerResidentBatchGetBinding& binding,
      const MutableBufferView& queue_leases, std::uint32_t queue_count);
  Status ConfigureLayerResidentBatchGetQueueTemplates(
      const std::vector<LayerResidentBatchGetQueueTemplate>& queues);
  Status ReleaseLayerResidentBatchGetControl();
  Status DescribeLayerResidentBatchGetControl(std::string* diagnostic) const;

  // P10's forward-owned spelling is a compatibility façade over the same
  // hardened canonical-resident control.  It deliberately retains the one
  // binding implementation so the upstream global ABI aliases and unposted
  // queue rollback semantics apply to both entrypoints.
  Status BindForwardResidentBatchGetControl(
      const ForwardResidentBatchGetBinding& binding,
      const MutableBufferView& queue_leases, std::uint32_t queue_count);
  Status ConfigureForwardResidentBatchGetQueueTemplates(
      const std::vector<ForwardResidentBatchGetQueueTemplate>& queues);
  Status ReleaseForwardResidentBatchGetControl();
  Status DescribeForwardResidentBatchGetControl(std::string* diagnostic) const;

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
