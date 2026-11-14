#include <algorithm>
#include <utility>

#include "aicpu_batch_get/include/kvaf_aicpu_batch_get.h"
#include "batch_get_args.h"
#include "aicpu_batch_get_capture.h"
#include "aicpu_batch_get_profile.h"
#include "cpu_gather.h"
#include "aicpu_urma/session.h"
#include "get_host_timing.h"
namespace kvaf::aicpu_urma {
namespace {
Status InvalidArgument(std::string s) { return Status::InvalidArgument(std::move(s)); }
Status Internal(std::string s) { return Status::Internal(std::move(s)); }
using kvaf::detail::BatchGetExecutionState;
using kvaf::detail::BuildBatchGetKernelArgs;
using kvaf::detail::PointerValue;

}  // namespace
struct BatchGetTransport::Impl {
  Impl(UmdkSession& shared, std::uint32_t token, bool direct)
      : session(shared), token_bytes(token), direct_addresses(direct) {}
  UmdkSession& session;
  std::uint32_t token_bytes;
  bool direct_addresses;
  RemoteRegionDesc region;
  BatchGetExecutionState execution;
  void* workspace{nullptr};
  void* control{nullptr};
  void* status_host{nullptr};
  void* profile_device{nullptr};
  void* profile_host{nullptr};
  using Invocation = AicpuGetInvocation;
  ExecutionViewState views;
  Invocation legacy{this};
  Invocation* invocation{&legacy};
  bool profiling{false};
  bool initialized{false}, cpu_enabled{false};
  Status Drain(const char* stage = "drain", bool check_status = true);
};
static Status DrainGather(const GatherLaunchView& view, AicpuGetInvocation& call, const char* stage,
                           bool completion_enqueued = false, bool check_status = true) {
  const auto stream = call.pending ? call.pending : call.captured_stream;
  if (!stream) return check_status ? call.failure : Status::OK();
  bool capturing = false;
  auto allowed = kvaf::detail::BatchGetCapture(stream, true, false, &capturing);
  if (!allowed.ok()) return allowed;
  if (check_status && call.failure.ok() && call.submitted && !completion_enqueued &&
      aclrtMemcpyAsync(view.completion.host, view.completion.bytes, view.completion.device,
                       view.completion.bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream) != ACL_SUCCESS)
    return Internal("GatherRead status enqueue failed; resources retained");
  const auto result = aclrtSynchronizeStream(stream);
  if (result != ACL_SUCCESS) {
    const char* detail = aclGetRecentErrMsg();
    return Internal("GatherRead stream synchronization failed: acl=" + std::to_string(result) +
                    " stage=" + stage + " workers=" + std::to_string(view.options.workers) +
                    " source_mode=" + (view.direct_addresses ? "direct" : "table") +
                    "; buffers retained; detail=" + (detail ? detail : "unavailable"));
  }
  call.pending = nullptr;
  if (!check_status || !call.submitted) return Status::OK();
  if (!call.failure.ok()) return call.failure;
  const auto* reports = static_cast<const GatherReadStatus*>(view.completion.host);
  const auto sequence = reports[0].sequence;
  // Preserve the actual worker error before reporting stale slots from a partial launch.
  for (unsigned worker = 0; worker < view.options.workers; ++worker)
    if (reports[worker].magic == kGatherStatusMagic && reports[worker].result)
      return call.failure = Internal("GatherRead worker " + std::to_string(worker) +
          " failed: sequence=" + std::to_string(reports[worker].sequence) +
          " phase=" + std::to_string(reports[worker].phase) +
          " result=" + std::to_string(reports[worker].result));
  for (unsigned worker = 0; worker < view.options.workers; ++worker)
    if (reports[worker].magic != kGatherStatusMagic || !sequence || reports[worker].sequence != sequence)
      return call.failure = Internal("GatherRead worker status is stale or incomplete");
  call.sequence = sequence;
  return Status::OK();
}
BatchGetTransport::BatchGetTransport(UmdkSession& session, std::uint32_t token_bytes,
                                     bool direct_addresses)
    : impl_(std::make_unique<Impl>(session, token_bytes, direct_addresses)) {}
BatchGetTransport::~BatchGetTransport() = default;
std::unique_ptr<ExecutionState> BatchGetTransport::CreateExecutionState() {
  return std::make_unique<Impl::Invocation>(impl_.get());
}
Status BatchGetTransport::BindExecutionState(ExecutionState* state) {
  auto* typed = state ? dynamic_cast<Impl::Invocation*>(state) : &impl_->legacy;
  if (!typed || typed->owner != impl_.get()) return Status::InvalidArgument("Get invocation belongs to another resource");
  impl_->invocation = typed;
  impl_->views.Bind(typed);
  return Status::OK();
}
GatherLaunchView BatchGetTransport::View() const {
  return {impl_->views.View(ExecutionDirection::Get),
      {impl_->session.session(), impl_->session.owner_pid()},
      {impl_->session.config().workers, impl_->token_bytes}, impl_->direct_addresses,
      impl_->workspace, {impl_->execution.status, impl_->status_host, impl_->execution.status_bytes},
      {impl_->profile_device, impl_->profile_host, impl_->session.config().workers * sizeof(BatchGetWorkerProfile)},
      impl_->control, impl_->profiling};
}
Status BatchGetTransport::Impl::Drain(const char* stage, bool check_status) {
  GatherLaunchView view;
  view.options.workers = session.config().workers; view.direct_addresses = direct_addresses;
  view.completion = {execution.status, status_host, execution.status_bytes};
  return DrainGather(view, *invocation, stage, false, check_status);
}
Status BatchGetTransport::Drain(bool check_status) { return impl_->Drain("drain", check_status); }
Status BatchGetTransport::Check(aclrtStream stream) {
  auto& call = *impl_->invocation;
  if (!impl_->initialized || !call.submitted)
    return Status::InvalidState("no AICPU BatchGet has been submitted");
  if (stream != call.last_stream)
    return InvalidArgument("check_batch_get_status requires the BatchGet execution stream");
  call.pending = stream;
  return impl_->Drain("check");
}
Status BatchGetTransport::SetKernelProfiling(bool enabled) {
  if (!impl_->initialized) return Status::NotInitialized("GatherRead is not initialized");
  auto ready = impl_->Drain();
  if (!ready.ok()) return ready;
  if (enabled) {
    ready = kvaf::detail::AllocateAicpuProfiles(&impl_->profile_device, &impl_->profile_host,
        impl_->session.config().workers * sizeof(BatchGetWorkerProfile));
    if (!ready.ok()) return ready;
  }
  impl_->profiling = enabled;
  impl_->invocation->profile_submitted = false;
  return Status::OK();
}
Status BatchGetTransport::CopyKernelProfiles(std::vector<BatchGetKernelProfile>* profiles) const {
  const auto& call = *impl_->invocation;
  if (!impl_->profiling || !call.profile_submitted)
    return Status::InvalidState("no profiled GatherRead launch");
  bool capturing = false;
  auto ready = kvaf::detail::BatchGetCapture(call.last_stream, true, false, &capturing);
  if (!ready.ok()) return ready;
  return kvaf::detail::CopyAicpuProfiles(impl_->profile_device, impl_->profile_host,
      impl_->session.config().workers, profiles);
}
Status BatchGetTransport::Initialize(const RemoteRegionDesc& region) {
  if (impl_->initialized) return Status::InvalidState("UMDK operator already initialized");
  auto& state = impl_->execution;
  state.status_bytes = impl_->session.config().workers * sizeof(GatherReadStatus);
  if (aclrtMalloc(&state.status, state.status_bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
      aclrtMalloc(&impl_->workspace, GatherWorkspaceBytes(impl_->session.config().workers),
                  ACL_MEM_MALLOC_HUGE_FIRST) !=
          ACL_SUCCESS ||
      aclrtMallocHost(&impl_->status_host, state.status_bytes) != ACL_SUCCESS ||
      aclrtMalloc(&impl_->control, sizeof(GatherExecutionControl), ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS ||
      aclrtMemset(impl_->control, sizeof(GatherExecutionControl), 0, sizeof(GatherExecutionControl)) != ACL_SUCCESS ||
      aclrtMemset(state.status, state.status_bytes, 0, state.status_bytes) != ACL_SUCCESS) {
    (void)Shutdown();
    return Internal("failed to allocate UMDK execution buffers");
  }
  peer::PeerRegion registered;
  auto found = impl_->session.config().peer_link->LookupRegion(region.base, &registered);
  if (!found.ok()) {
    (void)Shutdown();
    return found;
  }
  if (!impl_->token_bytes ||
      registered.length < region.length +
          GatherRemoteWorkspaceBytes(impl_->token_bytes, impl_->session.config().workers)) {
    (void)Shutdown();
    return InvalidArgument("UMDK BatchGet requires reserved remote gather workspace");
  }
  impl_->region = region;
  impl_->initialized = true;
  impl_->views.Bind(impl_->invocation);
  impl_->views.Initialize(region);
  return Status::OK();
}
Status BatchGetTransport::Shutdown() {
  impl_->views.Close();
  auto drained = impl_->Drain("shutdown", false);
  if (!drained.ok()) return drained;
  if (impl_->workspace) (void)aclrtFree(impl_->workspace);
  if (impl_->control) (void)aclrtFree(impl_->control);
  if (impl_->status_host) (void)aclrtFreeHost(impl_->status_host);
  if (impl_->profile_device) (void)aclrtFree(impl_->profile_device);
  if (impl_->profile_host) (void)aclrtFreeHost(impl_->profile_host);
  if (impl_->execution.status) (void)aclrtFree(impl_->execution.status);
  impl_->workspace = impl_->status_host = nullptr;
  impl_->profile_device = impl_->profile_host = nullptr;
  impl_->execution = {};
  impl_->control = nullptr;
  *impl_->invocation = Impl::Invocation{impl_.get()};
  impl_->profiling = false;
  impl_->initialized = impl_->cpu_enabled = false;
  return Status::OK();
}
Status BatchGetTransport::Warmup() {
  if (!impl_->initialized) return Status::NotInitialized("Gather not initialized");
  const auto workers = impl_->session.config().workers;
  auto s =
      impl_->session.Ensure(impl_->region.base, impl_->workspace, GatherWorkspaceBytes(workers));
  if (!s.ok()) return s;
  if (!impl_->cpu_enabled) {
    const auto staging = (impl_->region.base + impl_->region.length + 63U) & ~std::uint64_t{63U};
    const auto controls = staging + GatherStagingBytes(impl_->token_bytes, workers) +
                          workers * GatherLaneDescriptorBytes(workers);
    s = impl_->session.config().peer_link->EnableGatherLanes(impl_->region.base, controls, workers);
    if (s.ok()) impl_->cpu_enabled = true;
  }
  return s;
}
Status BatchGetTransport::Prepare(const BatchGetExecRequest& exec) {
  impl_->views.Prepare();
  auto& call = *impl_->invocation;
  call.prepared = nullptr;
  if (exec.execution.profile && !exec.execution.synchronize)
    return InvalidArgument("BatchGet profiling requires synchronize=true");
  call.prepare_started = exec.execution.profile ? BatchGetProfileNowNs() : 0;
  if (exec.execution.profile) *exec.execution.profile = {};
  if (!impl_->initialized || !exec.execution.stream)
    return InvalidArgument("UMDK BatchGet needs initialization and stream");
  if (!call.failure.ok()) return call.failure;
  const auto stream = static_cast<aclrtStream>(exec.execution.stream);
  bool capturing = false;
  auto allowed = kvaf::detail::BatchGetCapture(stream, exec.execution.synchronize,
                                              exec.execution.profile != nullptr, &capturing);
  if (!allowed.ok()) return allowed;
  if (call.captured_stream && call.captured_stream != stream)
    return InvalidArgument("captured AICPU transport must use its capture stream");
  if (capturing) {
    if (call.pending && call.pending != stream)
      return Status::InvalidState("drain previous AICPU work outside capture on its stream");
  } else {
    auto drained = impl_->Drain("previous_request");
    if (!drained.ok()) return drained;
  }
  UmdkBatchGetArgs checked{};
  auto valid = BuildBatchGetKernelArgs(exec, &checked.request);
  if (!valid.ok()) return valid;
  if (checked.request.request_capacity > kGatherMaxRequests || checked.request.token_bytes != impl_->token_bytes)
    return InvalidArgument("UMDK BatchGet exceeds requests or configured token size");
  if (impl_->direct_addresses && (exec.request->access_table.opaque != 0U ||
      checked.request.table_capacity < checked.request.request_capacity))
    return InvalidArgument("invalid direct address request");
  const auto profile_bytes = impl_->session.config().workers * sizeof(BatchGetWorkerProfile);
  if (exec.execution.profile || impl_->profiling) {
    if (capturing && (!impl_->profile_device || !impl_->profile_host))
      return Status::InvalidState("enable kernel profiling before capture");
    auto allocated = kvaf::detail::AllocateAicpuProfiles(&impl_->profile_device, &impl_->profile_host, profile_bytes);
    if (!allocated.ok()) return allocated;
  }
  auto ready = impl_->session.Ensure(impl_->region.base, impl_->workspace,
      GatherWorkspaceBytes(impl_->session.config().workers), capturing);
  if (!ready.ok()) return ready;
  ready = impl_->session.Ensure(impl_->region.base, exec.request->output.data,
                                exec.request->output.bytes, capturing);
  if (!ready.ok()) return ready;
  if (!impl_->cpu_enabled) {
    if (capturing) return Status::InvalidState("CPU gather lanes are not prepared; warm up outside capture");
    ready = Warmup();
    if (!ready.ok()) return ready;
  }
  call.prepared_capturing = capturing;
  call.prepared = exec.request;
  call.prepared_stream = exec.execution.stream;
  call.prepared_profile = exec.execution.profile;
  return Status::OK();
}
Status BatchGetTransport::Execute(const BatchGetExecRequest& exec) {
  auto ready = Prepare(exec);
  return ready.ok() ? ExecutePrepared(exec) : ready;
}
Status BatchGetTransport::ExecutePrepared(const BatchGetExecRequest& exec) {
  return LaunchPrepared(exec, View(), impl_->invocation);
}
Status BatchGetTransport::LaunchPrepared(const BatchGetExecRequest& exec,
                                         const GatherLaunchView& view, ExecutionState* state) {
  const auto submit_started = GetHostTimingStart(exec.host_timing);
  auto* invocation = dynamic_cast<AicpuGetInvocation*>(state);
  auto valid_view = view.common.Validate(invocation, ExecutionDirection::Get);
  if (!valid_view.ok()) return valid_view;
  auto* profile = exec.execution.profile;
  if (profile && !exec.execution.synchronize)
    return InvalidArgument("BatchGet profiling requires synchronize=true");
  auto& call = *invocation;
  if (call.prepared != exec.request || call.prepared_stream != exec.execution.stream ||
      call.prepared_profile != profile || !call.prepared)
    return Status::InvalidState("Gather resources were not prepared for this request");
  call.prepared = nullptr;
  const auto begin = call.prepare_started;
  UmdkBatchGetArgs launch{};
  auto valid = BuildBatchGetKernelArgs(exec, &launch.request);
  if (!valid.ok()) return valid;
  launch.request.canonical_base = view.common.region().canonical_base;
  launch.request.logical_bytes = view.common.region().logical_bytes;
  if (launch.request.request_capacity > kGatherMaxRequests ||
      launch.request.token_bytes != view.options.token_bytes)
    return InvalidArgument("UMDK BatchGet exceeds 65536 requests or configured token size");
  if (view.direct_addresses) {
    if (exec.request->access_table.opaque != 0U)
      return InvalidArgument("gather_read_direct requires the addresses Tensor overload");
    if (launch.request.table_capacity < launch.request.request_capacity)
      return InvalidArgument("direct addresses must cover the request capacity");
    launch.request.indices = launch.request.offsets = 0U;
  }
  const auto workers = view.options.workers;
  const auto profile_bytes = workers * sizeof(BatchGetWorkerProfile);
  const auto staging = (view.common.region().canonical_base + view.common.region().logical_bytes + 63U) & ~std::uint64_t{63U};
  const auto descriptors = staging + GatherStagingBytes(view.options.token_bytes, workers);
  const auto controls = descriptors + workers * GatherLaneDescriptorBytes(workers);
  auto& args = launch.gather;
  args.execution_control = PointerValue(view.control);
  args.context = {view.context.session, view.context.owner_pid};
  args.core_status = PointerValue(view.completion.device);
  args.scratch = PointerValue(view.workspace);
  args.source_descriptor = args.scratch + kGatherDescriptorOffset;
  args.output = PointerValue(exec.request->output.data);
  args.output_extent = exec.request->output.bytes;
  args.remote_staging = staging;
  args.remote_descriptor = descriptors;
  args.remote_notify = controls;
  args.remote_ack = args.remote_notify + 8U;
  args.remote_command = args.remote_notify + 16U;
  args.worker_count = view.options.workers;
  args.profile = (profile || view.profiling) ? PointerValue(view.profile.device) : 0U;
  const auto stream = static_cast<aclrtStream>(exec.execution.stream);
  call.pending = stream;
  call.last_stream = stream;
  if (call.prepared_capturing) call.captured_stream = stream;
  auto stamp = profile ? BatchGetProfileNowNs() : 0U;
  if (profile) profile->prepare_ns = stamp - begin;
  if (LaunchBatchGet(&launch, stream))
    return call.failure = Internal("UMDK BatchGet launch failed; shutdown required");
  call.submitted = true;
  call.profile_submitted = args.profile != 0;
  GetHostTimingRecord(exec.host_timing, &GetProviderTiming::submit_ns, submit_started);
  const auto completion_started = GetHostTimingStart(exec.host_timing);
  if (profile) {
    const auto now = BatchGetProfileNowNs();
    profile->launch_ns = now - stamp;
    stamp = now;
  }
  if (!exec.execution.synchronize) return Status::OK();
  if (aclrtMemcpyAsync(view.completion.host, view.completion.bytes, view.completion.device,
                       view.completion.bytes, ACL_MEMCPY_DEVICE_TO_HOST,
                       stream) != ACL_SUCCESS)
    return Internal("BatchGet status enqueue failed");
  if (profile) {
    if (aclrtMemcpyAsync(view.profile.host, profile_bytes, view.profile.device,
                         profile_bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream) != ACL_SUCCESS)
      return Internal("BatchGet profile enqueue failed");
    const auto now = BatchGetProfileNowNs();
    profile->status_enqueue_ns = now - stamp;
    stamp = now;
  }
  auto completed = DrainGather(view, call, "completion", true);
  if (profile) {
    const auto now = BatchGetProfileNowNs();
    profile->wait_ns = now - stamp;
    stamp = now;
  }
  if (!completed.ok()) return completed;
  if (profile) {
    profile->worker_count = workers;
    std::copy_n(static_cast<const BatchGetWorkerProfile*>(view.profile.host), workers,
                profile->workers.begin());
    const auto end = BatchGetProfileNowNs();
    profile->finish_ns = end - stamp;
    profile->total_ns = end - begin;
  }
  GetHostTimingRecord(exec.host_timing, &GetProviderTiming::completion_ns, completion_started);
  return Status::OK();
}
}  // namespace kvaf::aicpu_urma
