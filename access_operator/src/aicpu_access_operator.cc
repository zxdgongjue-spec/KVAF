#include "aicpu_access_operator.h"
#include "aicpu_execution_resources.h"
#include "aicpu_batch_get/include/kvaf_aicpu_batch_get.h"
#include "aicpu_batch_get/include/cpu_push_transport.h"
#include "aicpu_batch_put/include/kvaf_aicpu_batch_put.h"

namespace kvaf {
Status UrmaAicpuPutOperator::Launch(const BatchPutExecRequest& request,
                                   const UrmaAicpuPutView& view, AicpuInvocation& call) {
  auto valid = view.common.Validate(&call, ExecutionDirection::Put);
  if (!valid.ok()) return valid;
  auto result = aicpu_urma::BatchPutTransport::LaunchPrepared(request, view.launch, call.put.get());
  if (result.ok()) call.put_completion = {PutOutcome::Succeeded, result};
  else if (!call.write.attempted) call.put_completion = {PutOutcome::Failed, result};
  else call.put_completion.result = result;
  return result;
}
Status UrmaAicpuGatherOperator::Launch(const BatchGetExecRequest& request,
                                      const UrmaAicpuGatherView& view, AicpuInvocation& call) {
  auto valid = view.common.Validate(&call, ExecutionDirection::Get);
  if (!valid.ok()) return valid;
  return aicpu_urma::BatchGetTransport::LaunchPrepared(request, view.launch, call.get.get());
}
Status UrmaAicpuScatterOperator::Launch(const BatchGetExecRequest& request,
                                       const UrmaAicpuScatterView& view, AicpuInvocation& call) {
  auto valid = view.common.Validate(&call, ExecutionDirection::Get);
  if (!valid.ok()) return valid;
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
  return cpu_push::CpuPushTransport::LaunchPrepared(request, view.launch, call.push.get());
#else
  return Status::NotImplemented("AICPU Scatter was not built");
#endif
}

// Public compatibility adapter: resources are independently owned after
// TakeResources. No session, queue, MR or workspace is a member of this adapter.
UmdkAccessOperator::UmdkAccessOperator(UmdkAccessOperatorConfig config)
    : owned_resources_(std::make_unique<AicpuExecutionResources>(std::move(config))),
      resources_(owned_resources_.get()) {}
UmdkAccessOperator::~UmdkAccessOperator() = default;
std::unique_ptr<ExecutionResourceOwner> UmdkAccessOperator::TakeResources() {
  return std::move(owned_resources_);
}
std::unique_ptr<ExecutionState> UmdkAccessOperator::CreateExecutionState() {
  return resources_->CreateInvocation();
}
Status UmdkAccessOperator::BindExecutionState(ExecutionState* state) { return resources_->Bind(state); }
Status UmdkAccessOperator::Initialize(const RemoteRegionDesc& region) { return resources_->Initialize(region); }
Status UmdkAccessOperator::Warmup(const ExecutionWarmup& request) { return resources_->Warmup(request); }
Status UmdkAccessOperator::Shutdown() { return resources_->Close(); }
Status UmdkAccessOperator::ResolvePut(PutCompletion* result) { return resources_->ResolvePut(result); }
Status UmdkAccessOperator::AcknowledgePut() { return resources_->AcknowledgePut(); }
Status UmdkAccessOperator::DrainGet() { return resources_->DrainGet(); }
Status UmdkAccessOperator::PrepareGetOutput(const GetOutputOptions& r) { return resources_->PrepareGetOutput(r); }
Status UmdkAccessOperator::FinishGetOutput() { return resources_->FinishGetOutput(); }
Status UmdkAccessOperator::ExecuteBatchPut(const BatchPutExecRequest& request) {
  std::lock_guard<std::mutex> lock(resources_->put_mutex());
  auto& call = resources_->invocation();
  call.put_completion = {};
  call.write = {};
  auto result = resources_->PreparePutResources(request);
  if (!result.ok()) { call.put_completion = {PutOutcome::Failed, result}; return result; }
  result = UrmaAicpuPutOperator::Launch(request, resources_->PutView(), call);
  if (result.ok() && resources_->managed_completion()) result = resources_->ReleasePutBuffers();
  return result;
}
Status UmdkAccessOperator::ExecuteBatchGet(const BatchGetExecRequest& request) {
  std::lock_guard<std::mutex> lock(resources_->get_mutex());
  auto ready = resources_->PrepareGetResources(request);
  if (!ready.ok()) return ready;
  auto result = resources_->scatter()
      ? UrmaAicpuScatterOperator::Launch(request, resources_->ScatterView(), resources_->invocation())
      : UrmaAicpuGatherOperator::Launch(request, resources_->GatherView(), resources_->invocation());
  if (result.ok() && request.execution.synchronize && resources_->managed_completion())
    result = resources_->ReleaseGetBuffers();
  return result;
}
}  // namespace kvaf
