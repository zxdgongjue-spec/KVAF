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

struct BatchGetKernelProfileRecord {
  std::uint32_t core{0};
  std::uint64_t sequence{0};
  std::uint64_t result{0};
  std::uint64_t processed_request_count{0};
  std::uint64_t completed_batch_count{0};
  std::uint64_t sq_producer_index{0};
  std::uint64_t cq_consumer_index{0};
  std::uint64_t failed_request_index{0};
  std::uint64_t last_wqe_protocol{0};
  BatchGetKernelProfile profile;
};

struct BatchGetKernelProfileHistory {
  std::uint32_t capacity{0};
  std::vector<std::uint64_t> sample_counts;
  std::vector<std::uint64_t> record_counts;
  std::vector<BatchGetKernelProfileRecord> records;
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

  // Profiling is opt-in. `history_capacity` records are retained per AIV core
  // in a device-resident ring; `min_total_cycles` filters the ring only, not
  // the latest per-core status snapshot. Configure it before graph capture.
  // Callers must complete the BatchGet stream before copying a snapshot.
  Status SetBatchGetKernelProfiling(bool enabled,
                                    std::uint32_t history_capacity = 1,
                                    std::uint64_t min_total_cycles = 0);
  Status CopyBatchGetKernelProfiles(
      std::vector<BatchGetKernelProfile>* profiles) const;
  Status CopyBatchGetKernelProfileHistory(
      BatchGetKernelProfileHistory* history) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kvaf
