#include "gather_execution.h"
#include "scatter_execution.h"

namespace kvaf {
namespace {
template <class Invocation, class Resources>
Status BindInvocation(Resources& resources, ExecutionState* state) {
  auto* invocation = state ? dynamic_cast<Invocation*>(state) : &resources.fallback;
  if (!invocation || invocation->owner != &resources)
    return Status::InvalidArgument("Get invocation belongs to another resource owner");
  // A temporary graph binding keeps the eager invocation alive for drain.
  // Replacing that eager invocation would lose ownership of pending work.
  if (!invocation->graph_node && invocation != resources.eager && resources.eager->HasPendingResources())
    return Status::Busy("cannot replace a Get invocation with pending work");
  resources.views.Bind(invocation);
  resources.active = invocation;
  if (!invocation->graph_node) resources.eager = invocation;
  return Status::OK();
}
}  // namespace

GatherBatchGet::GatherBatchGet(AccessOperatorConfig config)
    : standalone_(std::make_unique<GatherResources>(std::move(config))), resources_(standalone_.get()) {}

GatherBatchGet::~GatherBatchGet() {
  if (standalone_ && !standalone_->Close().ok()) (void)standalone_.release();
}

std::unique_ptr<ExecutionResourceOwner> GatherBatchGet::TakeResources() { return std::move(standalone_); }
Status GatherBatchGet::Initialize(const RemoteRegionDesc& region) { return resources_->Initialize(region); }
Status GatherBatchGet::Shutdown() { return resources_->Close(); }

std::unique_ptr<ExecutionState> GatherBatchGet::CreateExecutionState() {
  return std::make_unique<GatherInvocation>(resources_);
}

Status GatherBatchGet::BindExecutionState(ExecutionState* state) {
  return BindInvocation<GatherInvocation>(*resources_, state);
}

Status GatherBatchGet::ExecuteBatchGet(const BatchGetExecRequest& exec) {
  auto& resources = *resources_;
  std::lock_guard<std::mutex> lock(resources.mutex);
  return GatherHostOperator::Execute(exec, resources.View(), *resources.active, *resources.eager);
}

Status GatherBatchGet::DrainGet() {
  auto& resources = *resources_;
  std::lock_guard<std::mutex> lock(resources.mutex);
  return GatherHostOperator::Drain(resources.View(), *resources.active, *resources.eager);
}

Status GatherBatchGet::PrepareGetGraph(const GetGraphOptions& options) {
  auto& resources = *resources_;
  std::lock_guard<std::mutex> lock(resources.mutex);
  auto status = GatherHostOperator::PrepareGraph(options, resources.View(), *resources.active, *resources.eager);
  if (status.ok()) resources.views.Prepare(ExecutionDirection::Get);
  return status;
}

Status GatherBatchGet::CheckBatchGetStatus(void* stream) {
  auto& resources = *resources_;
  std::lock_guard<std::mutex> lock(resources.mutex);
  return GatherHostOperator::CheckGraph(stream, resources.View(), *resources.active);
}

Status GatherBatchGet::FinishGetGraph(void* stream, bool discard) {
  auto& resources = *resources_;
  std::lock_guard<std::mutex> lock(resources.mutex);
  auto status = GatherHostOperator::FinishGraph(stream, discard, resources.View(), *resources.active);
  if (status.ok()) resources.views.Prepare(ExecutionDirection::Get);
  return status;
}

bool GatherBatchGet::GetGraphNeedsRecovery() const {
  return resources_->active->graph_node && resources_->active->recovery_fenced;
}

Status GatherBatchGet::SetBatchGetKernelProfiling(bool enabled) {
  std::lock_guard<std::mutex> lock(resources_->mutex);
  return resources_->SetProfiling(enabled);
}

Status GatherBatchGet::CopyGatherBatchGetProfile(GatherBatchGetProfile* profile) const {
  auto& resources = *resources_;
  std::lock_guard<std::mutex> lock(resources.mutex);
  return GatherHostOperator::CopyProfile(profile, resources.View(), *resources.active, *resources.eager);
}

ScatterBatchGet::ScatterBatchGet(AccessOperatorConfig config)
    : standalone_(std::make_unique<ScatterResources>(std::move(config))), resources_(standalone_.get()) {}

ScatterBatchGet::~ScatterBatchGet() {
  if (standalone_ && !standalone_->Close().ok()) (void)standalone_.release();
}

std::unique_ptr<ExecutionResourceOwner> ScatterBatchGet::TakeResources() { return std::move(standalone_); }
Status ScatterBatchGet::Initialize(const RemoteRegionDesc& region) { return resources_->Initialize(region); }
Status ScatterBatchGet::Shutdown() { return resources_->Close(); }

std::unique_ptr<ExecutionState> ScatterBatchGet::CreateExecutionState() {
  return std::make_unique<ScatterInvocation>(resources_);
}

Status ScatterBatchGet::BindExecutionState(ExecutionState* state) {
  return BindInvocation<ScatterInvocation>(*resources_, state);
}

Status ScatterBatchGet::ExecuteBatchGet(const BatchGetExecRequest& exec) {
  auto* timing = exec.host_timing;
  BatchGetProfileScope total(timing ? &timing->total_ns : nullptr);
  auto& resources = *resources_;
  std::lock_guard<std::mutex> lock(resources.mutex);
  return ScatterHostOperator::Execute(exec, resources.View(), *resources.active, *resources.eager);
}

Status ScatterBatchGet::DrainGet() {
  auto& resources = *resources_;
  std::lock_guard<std::mutex> lock(resources.mutex);
  return ScatterHostOperator::Drain(resources.View(), *resources.active, *resources.eager);
}

Status ScatterBatchGet::PrepareGetGraph(const GetGraphOptions& options) {
  auto& resources = *resources_;
  std::lock_guard<std::mutex> lock(resources.mutex);
  auto status = ScatterHostOperator::PrepareGraph(options, resources.View(), *resources.active, *resources.eager);
  if (status.ok()) resources.views.Prepare(ExecutionDirection::Get);
  return status;
}

Status ScatterBatchGet::CheckBatchGetStatus(void* stream) {
  auto& resources = *resources_;
  std::lock_guard<std::mutex> lock(resources.mutex);
  return ScatterHostOperator::CheckGraph(stream, resources.View(), *resources.active);
}

Status ScatterBatchGet::FinishGetGraph(void* stream, bool discard) {
  auto& resources = *resources_;
  std::lock_guard<std::mutex> lock(resources.mutex);
  auto status = ScatterHostOperator::FinishGraph(stream, discard, resources.View(), *resources.active);
  if (status.ok()) resources.views.Prepare(ExecutionDirection::Get);
  return status;
}

bool ScatterBatchGet::GetGraphNeedsRecovery() const {
  return resources_->active->graph_node && resources_->active->recovery_fenced;
}

Status ScatterBatchGet::SetBatchGetKernelProfiling(bool enabled) {
  std::lock_guard<std::mutex> lock(resources_->mutex);
  return resources_->SetProfiling(enabled);
}

Status ScatterBatchGet::CopyScatterProfiles(std::vector<ScatterBatchGetCoreProfile>* profile) const {
  auto& resources = *resources_;
  std::lock_guard<std::mutex> lock(resources.mutex);
  return ScatterHostOperator::CopyProfiles(profile, resources.View(), *resources.active, *resources.eager);
}

std::unique_ptr<SequentialBatchGet> CreateSequentialBatchGet(GetAlgorithm algorithm, AccessOperatorConfig config) {
  if (algorithm == GetAlgorithm::Scatter)
    return std::make_unique<ScatterBatchGet>(std::move(config));
  if (algorithm == GetAlgorithm::Gather)
    return std::make_unique<GatherBatchGet>(std::move(config));
  return nullptr;
}
}  // namespace kvaf
