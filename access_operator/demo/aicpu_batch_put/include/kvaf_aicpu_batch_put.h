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
  Status Prepare(const RemoteRegionDesc&, const BatchPutRequest&);
  Status ExecutePrepared(const RemoteRegionDesc&, const BatchPutExecRequest&);
  Status Shutdown();
  Status Reserve(std::uint32_t capacity);
  std::unique_ptr<ExecutionState> CreateExecutionState();
  Status BindExecutionState(ExecutionState*);

 private:
  UmdkSession& session_;
  void* transfers_{nullptr};
  std::size_t capacity_{0};
  std::vector<UmdkTransfer> host_transfers_;
  struct Preparation final : ExecutionState {
    explicit Preparation(const BatchPutTransport* value) : owner(value) {}
    const BatchPutTransport* owner;
    const BatchPutRequest* request{nullptr};
    std::uint64_t region{0};
    const void* input{nullptr};
    std::size_t bytes{0}, rows{0};
  };
  Preparation legacy_preparation_{this};
  Preparation* preparation_{&legacy_preparation_};
};
int32_t LaunchBatchPut(const BatchPutKernelArgs*, aclrtStream);
}  // namespace kvaf::aicpu_urma
