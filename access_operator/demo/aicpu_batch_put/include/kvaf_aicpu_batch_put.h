#pragma once
#include <acl/acl.h>

#include <vector>

#include "access_operator.h"
#include "aicpu_batch_put/include/batch_put_abi.h"
namespace kvaf::aicpu_urma {
class UmdkSession;
class BatchPutTransport {
 public:
  explicit BatchPutTransport(UmdkSession& session) : session_(session) {}
  Status Execute(const RemoteRegionDesc&, const BatchPutExecRequest&);
  Status Shutdown();

 private:
  UmdkSession& session_;
  void* transfers_{nullptr};
  std::size_t capacity_{0};
  std::vector<UmdkTransfer> host_transfers_;
};
int32_t LaunchBatchPut(const BatchPutKernelArgs*, aclrtStream);
}  // namespace kvaf::aicpu_urma
