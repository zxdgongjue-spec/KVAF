#pragma once
#include <acl/acl.h>

#include <vector>

#include "access_operator.h"
#include "aicpu_urma/launch_view.h"
#include "put/urma_aicpu/include/batch_put_abi.h"
namespace kvaf::aicpu_urma {
class UmdkSession;
class BatchPutTransport {
 public:
  explicit BatchPutTransport(UmdkSession& session) : session_(session) {}
  Status Execute(const RemoteRegionDesc&, const BatchPutExecRequest&);
  Status Prepare(const RemoteRegionDesc&, const BatchPutRequest&, PutProviderTiming* timing = nullptr);
  // The owner has registered the bound input and reserved descriptor capacity.
  Status PrepareBound(const RemoteRegionDesc&, const BatchPutRequest&, const PutInputOptions&);
  Status ExecutePrepared(const RemoteRegionDesc&, const BatchPutExecRequest&);
  PutLaunchView View();
  static Status LaunchPrepared(const BatchPutExecRequest&, const PutLaunchView&, ExecutionState*);
  Status Shutdown();
  Status Reserve(std::uint32_t capacity);
  std::unique_ptr<ExecutionState> CreateExecutionState();
  Status BindExecutionState(ExecutionState*);

 private:
  Status RecordPreparation(const RemoteRegionDesc&, const BatchPutRequest&);
  UmdkSession& session_;
  void* transfers_{nullptr};
  std::size_t capacity_{0};
  std::vector<UmdkTransfer> host_transfers_;
  using Preparation = PutPreparation;
  ExecutionViewState views_;
  Preparation default_preparation_{this};
  Preparation* preparation_{&default_preparation_};
};
int32_t LaunchBatchPut(const BatchPutKernelArgs*, aclrtStream);
}  // namespace kvaf::aicpu_urma
