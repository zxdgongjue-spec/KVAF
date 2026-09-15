#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>

#include "requests.h"
#include "status.h"
#include "types.h"

namespace kvaf::peer {
class PeerLink;
}

namespace kvaf {

struct PutAccessDesc;
struct BatchPutHostProfile;
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

  // Host profiling is opt-in because each stage reads a host clock. Default
  // implementations keep test or alternate operators source-compatible.
  virtual Status SetBatchPutHostProfiling(bool) {
    return Status::NotImplemented("BatchPut host profiling is not available");
  }
  virtual Status CopyBatchPutHostProfile(BatchPutHostProfile*) const {
    return Status::NotImplemented("BatchPut host profiling is not available");
  }
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

struct BatchPutKernelProfile {
  std::uint64_t execute_cycles{0};
  std::uint64_t non_execute_cycles{0};
};
// Host-side timing for one successful synchronous BatchPut. The AIV kernel
// timeline is intentionally separate: these fields expose Runtime and
// host/ACL work surrounding the device kernel without changing its ABI.
struct BatchPutHostProfile {
  std::uint64_t row_count{0};
  std::uint64_t payload_bytes{0};
  std::uint64_t metadata_bytes{0};
  std::uint64_t request_validation_ns{0};
  std::uint64_t prepare_put_ns{0};
  std::uint64_t metadata_reserve_ns{0};
  std::uint64_t metadata_pack_ns{0};
  // Host enqueue time for the asynchronous pinned-host metadata upload. Its
  // completion is ordered before the kernel on the same stream.
  std::uint64_t metadata_h2d_ns{0};
  std::uint64_t local_buffer_acquire_ns{0};
  std::uint64_t execute_sync_ns{0};
  std::uint64_t local_buffer_release_ns{0};
  std::uint64_t commit_put_ns{0};
  // Runtime E2E minus the synchronous data-plane ExecuteBatchPut interval.
  std::uint64_t runtime_non_execute_ns{0};
  std::uint64_t runtime_e2e_ns{0};

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

  // BatchPut is synchronous, so its latest host status already represents a
  // completed kernel when ExecuteBatchPut returns.
  Status SetBatchPutKernelProfiling(bool enabled);
  Status CopyBatchPutKernelProfiles(
      std::vector<BatchPutKernelProfile>* profiles) const;
  Status SetBatchPutHostProfiling(bool enabled) override;
  Status CopyBatchPutHostProfile(BatchPutHostProfile* profile) const override;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kvaf
