#pragma once
#include "../access_operator.h"

namespace kvaf::ubmem {
struct OperatorOptions {
  std::string batch_get_kernel_object_path;
  std::string batch_put_kernel_object_path;
  std::uint32_t device_id{0};
  std::uint32_t cores{2};
  std::shared_ptr<peer::PeerLink> peer_link;
};

// Mapped-memory backend of the existing KVAF contract.
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
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace kvaf::ubmem
