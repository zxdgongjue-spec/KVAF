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
  BatchGetTransport(UmdkSession&, std::uint32_t token_bytes, bool direct_addresses = false);
  ~BatchGetTransport();
  Status Initialize(const RemoteRegionDesc&, bool slot_sized_staging = false);
  Status Execute(const BatchGetExecRequest&);
  Status ExecuteRanges(const BatchGetRangesExecRequest&);
  Status Drain(bool check_status = true);
  Status Check(aclrtStream);
  Status CheckRanges(aclrtStream);
  Status SetKernelProfiling(bool);
  Status CopyKernelProfiles(std::vector<BatchGetKernelProfile>*) const;
  // Release after the shared session has unregistered its HBM buffers.
  Status Shutdown();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
int32_t LaunchBatchGet(const UmdkBatchGetArgs*, aclrtStream);
int32_t LaunchBatchGetRanges(const UmdkBatchGetRangesArgs*, aclrtStream);
}  // namespace kvaf::aicpu_urma
namespace kvaf::cpu_push {
int32_t LaunchCpuPushScatter(const KernelArgs*, aclrtStream);
}  // namespace kvaf::cpu_push
