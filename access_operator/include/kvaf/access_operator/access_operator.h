#pragma once

#include <cstdint>
#include <memory>
#include <string>

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

// DSA_ROW bypasses the legacy KeyId/MetaManager allocation path completely.
// These two address-mode requests are deliberately expressed in terms of the
// caller-owned packed NPU buffers and the DSA row-major directory.  They are
// not part of AccessOperator's virtual interface because only the CANN-backed
// DefaultAccessOperator can issue Direct-HCCP transfers.
struct DsaFullBlockPutRequest {
  // One contiguous packed HBM source.  Each offset/length/destination triplet
  // describes one full logical DSA block.  destination_addresses are absolute
  // Host-DDR UBAs allocated by the DSA_ROW arena, never KeyId lookups.
  ConstBufferView source;
  std::vector<std::uint64_t> source_offsets;
  std::vector<std::uint64_t> lengths;
  std::vector<Address> destination_addresses;
  void* stream{nullptr};
};

struct DsaDirectBatchGetRequest {
  // Device directory planes indexed by flat_index.  token_offsets contains
  // the packed-token offset inside each logical block, so the AIV kernel reads
  // `address_table[flat_index] + token_offsets[i] * output_block_bytes`.
  ConstBufferView address_table;
  ConstBufferView valid_table;
  ConstBufferView flat_indices;
  ConstBufferView token_offsets;
  ConstBufferView mask;
  ConstBufferView output_indices;
  MutableBufferView output;
  std::uint64_t output_block_bytes{0};
  // Required when token_offsets is present.  For C8 DSA this is 128.
  std::uint32_t max_token_offset{0};
  void* stream{nullptr};
  bool synchronize{true};
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
  // BatchGet Direct-HCCP Jettys assigned to each AIV core.
  std::uint32_t jettys_per_core{2};
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

  // DSA_ROW native data plane.  These methods require the Direct-HCCP AIV
  // kernels configured at construction time; they intentionally have no ACL
  // memcpy fallback because falling back would silently reintroduce a
  // host-side payload path into the production DSA/KVAF route.
  Status ExecuteDsaFullBlockPut(const DsaFullBlockPutRequest& request);
  Status ExecuteDsaDirectBatchGet(const DsaDirectBatchGetRequest& request);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kvaf
