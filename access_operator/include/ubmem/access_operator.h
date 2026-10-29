#pragma once
#include "../access_operator.h"

namespace kvaf::ubmem {
struct OperatorOptions {
  std::string kernel_object_path;
  std::uint32_t device_id{0};
  std::uint32_t cores{2};
  std::shared_ptr<peer::PeerLink> peer_link;
};

// Experimental mapped-memory implementation of the existing KVAF contract.
// Execute methods consume RuntimeCore-validated requests and MetaManager's
// PutAccessDesc; call RuntimeCore::BatchGet/BatchPut as the public entry points.
class AccessOperator final : public kvaf::AccessOperator {
 public:
  explicit AccessOperator(OperatorOptions options);
  ~AccessOperator() override;
  Status Initialize(const RemoteRegionDesc& region) override;
  Status Shutdown() override;
  Status ExecuteBatchGet(const BatchGetExecRequest& request) override;
  Status ExecuteBatchPut(const BatchPutExecRequest& request) override;
  // Test runner control experiment: same Put kernel, with local HBM/DDR as the
  // destination. metadata is device uint64[3 * rows]: source offsets, lengths,
  // absolute CPU destination addresses. Buffers/ranges are prepared by
  // tests/ubmem/npu_test.cc; only mapped_base is replaced with local memory.
  Status ProbeLocalCopy(const ConstBufferView& input, const MutableBufferView& output,
                       const ConstBufferView& metadata, void* stream);
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace kvaf::ubmem
