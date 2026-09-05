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

// Borrowed device resources for a caller-launched fused kernel. KVAF retains
// the registrations and queue storage until ReleaseFusedBatchGetContext.
// Queue cursors are persistent: never clear them between kernel invocations.
// The caller keeps output HBM alive and submits all uses sequentially on one
// stream. This is a host API record, not a device tensor serialization ABI.
struct FusedBatchGetContext {
  std::uint64_t lease_id{0};
  std::uint64_t device_channel_entity{0};
  std::uint64_t device_queue_state{0};
  std::uint64_t device_core_status{0};
  std::uint64_t output_address{0};
  std::uint64_t output_bytes{0};
  std::uint32_t worker_count{0};
  std::uint32_t jettys_per_core{0};
  std::uint32_t token_bytes{0};
  std::uint32_t tokens_per_block{0};
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

  Status AcquireFusedBatchGetContext(
      void* output, std::uint64_t output_bytes, std::uint32_t token_bytes,
      std::uint32_t tokens_per_block, FusedBatchGetContext* context);
  // Synchronizes the supplied stream and checks the device queue result.
  // A failed release keeps the lease and registrations alive. In particular,
  // a CQ timeout cannot be treated as proof that DMA has stopped. Shutdown
  // refuses an active lease; after a transport fault the caller must retain
  // the output allocation and stop execution until device/process teardown.
  Status ReleaseFusedBatchGetContext(const FusedBatchGetContext& context,
                                    void* stream);

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
