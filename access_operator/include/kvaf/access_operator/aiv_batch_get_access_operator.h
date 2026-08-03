#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "kvaf/access_operator/access_operator.h"

namespace kvaf {

// AIV adapter configuration. The caller creates the ACL context and stream;
// this adapter only loads the kernel and owns the Direct-HCCP registration.
struct AivBatchGetOperatorConfig {
  std::string kernel_object_path;
  void* stream{nullptr};
  std::uint32_t aiv_core_count{0};
};

// All buffers below are caller-owned Device buffers. BindBatchGetOutput must
// run before a graph capture that will use this output address. Runtime never
// allocates, frees, or re-registers these buffers.
struct AivBatchGetOutputBinding {
  MutableBufferView output;
  MutableBufferView core_status;
  MutableBufferView queue_state;
};

// Optional CANN implementation of AccessOperator backed by the merged AIV
// ExecuteBatchGet demo. This target is built only when
// KVAF_ENABLE_AIV_BATCH_GET=ON on a CANN host.
class AivBatchGetAccessOperator final : public AccessOperator {
 public:
  explicit AivBatchGetAccessOperator(AivBatchGetOperatorConfig config);
  ~AivBatchGetAccessOperator() override;

  AivBatchGetAccessOperator(const AivBatchGetAccessOperator&) = delete;
  AivBatchGetAccessOperator& operator=(const AivBatchGetAccessOperator&) =
      delete;

  Status Initialize(const RemoteRegionDesc& remote_region) override;
  Status Shutdown() override;
  Status ExecuteBatchGet(const BatchGetExecRequest& request) override;
  Status ExecuteBatchPut(const BatchPutExecRequest& request) override;

  Status BindBatchGetOutput(const AivBatchGetOutputBinding& binding);
  Status UnbindBatchGetOutput();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kvaf
