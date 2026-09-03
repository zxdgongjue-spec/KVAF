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
  // Direct-HCCP Jettys assigned to each AIV core.
  std::uint32_t jettys_per_core{2};
  // Preserve the MTP BatchGet ABI end-to-end.  A forward-resident control
  // publishes these values verbatim to its device executor instead of
  // assuming any second descriptor layout.
  std::uint32_t batch_get_token_bytes{0};
  std::uint32_t batch_get_tokens_per_block{0};
  // Runtime supplies the selected local EID. Operator transports consume it;
  // they do not discover EIDs or make topology decisions.
  std::array<std::uint8_t, 16> peer_local_eid{};
  std::shared_ptr<PeerLink> peer_link;
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

// Cold binding for one forward-owned MTP resident payload.  `output` is
// registered exactly once with Direct-HCCP and the device executor may write
// only the caller-reserved [resident_row_base, resident_row_base +
// resident_row_count) interval.  The caller retains all tensors until the
// enclosing forward has drained its terminal CQ work and QSFA has consumed
// the fixed rows.
struct LayerResidentBatchGetBinding {
  std::uint32_t layer_id{0};
  MutableBufferView output;
  MutableBufferView control;
  std::uint32_t resident_row_base{0};
  std::uint32_t resident_row_count{0};
};

// One queue-local descriptor vector set.  `miss_count` is an NPU int32[1]
// compact-prefix counter; it is not a per-request bitmap.
struct LayerResidentBatchGetQueueTemplate {
  std::uint32_t queue_id{0};
  MutableBufferView address_table;
  MutableBufferView valid_table;
  MutableBufferView indices;
  MutableBufferView offsets;
  MutableBufferView miss_count;
  MutableBufferView output_indices;
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

  // MTP forward-resident Direct-HCCP API.  These are cold setup/teardown
  // operations only: the fused device operator owns PostOnly/CQ polling on
  // its graph streams; there is no Python/GIL hot submit path.
  Status BindLayerResidentBatchGetControl(
      const LayerResidentBatchGetBinding& binding,
      const MutableBufferView& queue_leases, std::uint32_t queue_count);
  Status ConfigureLayerResidentBatchGetQueueTemplates(
      const std::vector<LayerResidentBatchGetQueueTemplate>& queues);
  Status ReleaseLayerResidentBatchGetControl();
  Status DescribeLayerResidentBatchGetControl(std::string* diagnostic) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kvaf
