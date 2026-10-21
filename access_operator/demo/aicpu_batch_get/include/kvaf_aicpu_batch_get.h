#pragma once
#include <acl/acl.h>

#include <memory>

#include "access_operator.h"
#include "aicpu_batch_get/include/batch_get_abi.h"
namespace kvaf::aicpu_urma {
class UmdkSession;
// Owns gather_read workspace and execution; borrows the Get/Put session.
class BatchGetTransport {
 public:
  BatchGetTransport(UmdkSession&, std::uint32_t token_bytes);
  ~BatchGetTransport();
  Status Initialize(const RemoteRegionDesc&);
  Status Execute(const BatchGetExecRequest&);
  Status Drain();
  // Release after the shared session has unregistered its HBM buffers.
  Status Shutdown();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
int32_t LaunchBatchGet(const UmdkBatchGetArgs*, aclrtStream);
}  // namespace kvaf::aicpu_urma
