#include "aicpu_access_operator.h"
#include "aicpu_execution_resources.h"
#include "aicpu_batch_get/include/kvaf_aicpu_batch_get.h"
#include "aicpu_batch_get/include/cpu_push_transport.h"
#include "aicpu_batch_put/include/kvaf_aicpu_batch_put.h"
#include "get_host_timing.h"
#include "put_host_timing.h"

namespace kvaf {
Status UrmaAicpuPutOperator::Launch(const BatchPutExecRequest& request,
                                   const UrmaAicpuPutView& view, AicpuInvocation& call) {
  auto valid = view.common.Validate(&call, ExecutionDirection::Put);
  if (!valid.ok()) return valid;
  if (call.input_retiring)
    return Status::Busy("cannot launch with a partially retired Put input binding");
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
Status UmdkAccessOperator::PreparePutInput(const PutInputOptions& r) { return resources_->PreparePutInput(r); }
Status UmdkAccessOperator::FinishPutInput() { return resources_->FinishPutInput(); }
Status UmdkAccessOperator::ExecuteBatchPut(const BatchPutExecRequest& request) {
  auto* timing = request.execution.host_timing;
  if (timing) { timing->transport = "urma"; timing->executor = "aicpu"; }
  std::lock_guard<std::mutex> lock(resources_->put_mutex());
  PutHostTimingScope prepare_timing(timing, &PutProviderTiming::prepare_ns);
  auto& call = resources_->invocation();
  if (call.input_retiring || (call.input_bound && call.write.attempted))
    return Status::Busy("finish pending Put/retirement before reusing its input binding");
  call.put_completion = {};
  call.write = {};
  auto result = resources_->PreparePutResources(request);
  prepare_timing.Finish();
  if (!result.ok()) { call.put_completion = {PutOutcome::Failed, result}; return result; }
  result = UrmaAicpuPutOperator::Launch(request, resources_->PutView(), call);
  if (result.ok() && resources_->managed_completion() && !call.input_bound) {
    PutHostTimingScope cleanup_timing(timing, &PutProviderTiming::cleanup_ns);
    result = resources_->ReleasePutBuffers();
  }
  return result;
}
Status UmdkAccessOperator::ExecuteBatchGet(const BatchGetExecRequest& request) {
  std::lock_guard<std::mutex> lock(resources_->get_mutex());
  const auto prepare_started = GetHostTimingStart(request.host_timing);
  auto ready = resources_->PrepareGetResources(request);
  GetHostTimingRecord(request.host_timing, &GetProviderTiming::prepare_ns, prepare_started);
  if (!ready.ok()) return ready;
  auto result = resources_->scatter()
      ? UrmaAicpuScatterOperator::Launch(request, resources_->ScatterView(), resources_->invocation())
      : UrmaAicpuGatherOperator::Launch(request, resources_->GatherView(), resources_->invocation());
  if (result.ok() && request.execution.synchronize && resources_->managed_completion()) {
    const auto cleanup_started = GetHostTimingStart(request.host_timing);
    result = resources_->ReleaseGetBuffers();
    GetHostTimingRecord(request.host_timing, &GetProviderTiming::cleanup_ns, cleanup_started);
  }
  return result;
}
Status UmdkAccessOperator::CheckBatchGetStatus(void* stream) {
  return resources_->CheckBatchGetStatus(stream);
}
Status UmdkAccessOperator::SetBatchGetKernelProfiling(bool enabled) {
  return resources_->SetBatchGetKernelProfiling(enabled);
}
Status UmdkAccessOperator::CopyBatchGetKernelProfiles(std::vector<BatchGetKernelProfile>* profiles) const {
  return resources_->CopyBatchGetKernelProfiles(profiles);
}
}  // namespace kvaf
