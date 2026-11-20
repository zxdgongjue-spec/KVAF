#pragma once
#include <acl/acl.h>

#include <memory>

#include "access_operator.h"
#include "aicpu_urma/launch_view.h"
#include "get/common/aicpu/include/batch_get_abi.h"
namespace kvaf::aicpu_urma {
class UmdkSession;
// Owns gather_read workspace and execution; borrows the Get/Put session.
class BatchGetTransport {
 public:
  BatchGetTransport(UmdkSession&, std::uint32_t token_bytes, bool direct_addresses = false);
  ~BatchGetTransport();
  Status Initialize(const RemoteRegionDesc&);
  Status Execute(const BatchGetExecRequest&);
  Status Prepare(const BatchGetExecRequest&, std::optional<bool> expected_capture = std::nullopt);
  Status ExecutePrepared(const BatchGetExecRequest&);
  GatherLaunchView View() const;
  static Status LaunchPrepared(const BatchGetExecRequest&, const GatherLaunchView&, ExecutionState*);
  Status Drain(bool check_status = true, GetProviderTiming* = nullptr);
  Status Check(aclrtStream);
  Status PrepareGraph(const GetGraphOptions&);
  Status FinishGraph(void* stream, bool discard);
  Status SetKernelProfiling(bool);
  Status CopyKernelProfiles(std::vector<BatchGetKernelProfile>*) const;
  std::unique_ptr<ExecutionState> CreateExecutionState();
  Status BindExecutionState(ExecutionState*);
  Status Warmup();
  // Release after the shared session has unregistered its HBM buffers.
  Status Shutdown();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
int32_t LaunchBatchGet(const UmdkBatchGetArgs*, aclrtStream);
}  // namespace kvaf::aicpu_urma
namespace kvaf::cpu_push {
int32_t LaunchCpuPushScatter(const KernelArgs*, aclrtStream);
}  // namespace kvaf::cpu_push
