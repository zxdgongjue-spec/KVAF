#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "requests.h"
#include "status.h"
#include "types.h"

namespace kvaf::peer {
class PeerLink;
}

namespace kvaf {

struct PutAccessDesc;
using PeerLink = peer::PeerLink;

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
  // Direct-HCCP Jettys assigned to each BatchGet AIV core.
  std::uint32_t jettys_per_core{2};
  // Per-launch total READ budget; 0 selects the original whole-batch path.
  std::uint32_t batch_get_max_inflight_total{2048U};
  // BatchPut retains its independent one-or-two-Jetty implementation. Zero
  // derives min(jettys_per_core, 2); a direct caller may explicitly select 1/2.
  std::uint32_t batch_put_jettys_per_core{0};
  // Legacy single-EID input. Kept for source compatibility; peer_local_eids
  // takes precedence when it is non-empty.
  std::array<std::uint8_t, 16> peer_local_eid{};
  // Runtime may supply more than one selected local EID. BatchGet creates one
  // RaCtx per EID and stripes core/Jetty workers over those paths; BatchPut
  // continues to use peer_local_eid above.
  std::vector<std::array<std::uint8_t, 16>> peer_local_eids{};
  std::shared_ptr<PeerLink> peer_link;
  std::string aiv_batch_get_mode{"direct_read"};
  std::uint32_t batch_get_token_bytes{0};
  // Gather uses 1..13 local SGEs. No software inflight budget in sequential modes.
  std::uint32_t aiv_gather_sgl_size{6};
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
  std::array<std::uint64_t, 64> jetty_completion_wait_cycles{};
  std::array<std::uint64_t, 64> jetty_completion_polls{};
  std::uint64_t submitted_group_count{0};
  std::uint64_t completed_group_count{0};
  std::uint64_t completion_count{0};
  std::uint64_t post_active_cycles{0};
  std::uint64_t drain_wait_cycles{0};
  std::uint64_t peak_submitted_not_drained{0};
};

// Gather timeline; CPU wait and payload READ overlap when chunk mode is active.
// Device cycles and Host nanoseconds belong to different clock domains.
struct GatherBatchGetCoreProfile {
  std::uint64_t sequence{}, kernel_start_cycle{}, input_done_cycle{};
  std::uint64_t descriptor_done_cycle{}, notify_done_cycle{}, ack_done_cycle{};
  std::uint64_t read_done_cycle{}, kernel_finish_cycle{}, requests{}, data_wrs{};
  std::uint64_t ack_queries{}, data_wqebbs{}, data_doorbells{};
  std::uint64_t data_cq_wait_cycles{}, read_build_cycles{};
  std::uint64_t first_ready_cycle{}, first_read_submit_cycle{}, first_read_ready_bytes{};
  std::uint64_t ready_queries{}, ready_wait_cycles{}, pipeline_chunk_bytes{}, pipeline_shards{};
  std::uint64_t read_vf_calls{}, ready_no_change_queries{}, ready_query_skips{};
  std::uint64_t max_ready_backlog_wrs{}, ready_idle_waits{}, multi_run_posts{};
  std::uint64_t packed_posts{}, read_plan_segments{};
};

struct GatherBatchGetProfile {
  std::uint64_t sequence{};
  // prepare includes input adaptation/output registration in the owner;
  // launch includes status initialization and launch enqueue. wait measures
  // remaining device execution and queued status/profile D2H completion.
  std::uint64_t prepare_ns{}, launch_ns{}, status_enqueue_ns{}, wait_ns{};
  std::uint64_t finish_ns{}, total_ns{};
  std::vector<GatherBatchGetCoreProfile> cores;
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

  // Profiling is opt-in and uses the reserved per-core status area. Callers
  // must complete the BatchGet stream before copying the latest snapshot.
  Status SetBatchGetKernelProfiling(bool enabled);
  Status CopyBatchGetKernelProfiles(
      std::vector<BatchGetKernelProfile>* profiles) const;
  Status SetGatherBatchGetProfiling(bool enabled);
  Status CopyGatherBatchGetProfile(GatherBatchGetProfile* profile) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kvaf
