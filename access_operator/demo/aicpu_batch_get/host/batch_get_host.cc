#include <algorithm>
#include <utility>

#include "aicpu_batch_get/include/kvaf_aicpu_batch_get.h"
#include "batch_get_args.h"
#include "aicpu_batch_get_capture.h"
#include "aicpu_batch_get_profile.h"
#include "cpu_gather.h"
#include "aicpu_urma/session.h"
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
  std::uint64_t lane_staging_bytes{0U};
  BatchGetExecutionState execution;
  void* workspace{nullptr};
  void* control{nullptr};
  void* status_host{nullptr};
  void* profile_device{nullptr};
  void* profile_host{nullptr};
  aclrtStream pending{nullptr};
  aclrtStream captured_stream{nullptr};
  aclrtStream last_stream{nullptr};
  bool submitted{false};
  bool last_was_range{false};
  Status range_status;
  bool profiling{false}, profile_submitted{false};
  Status failure;
  bool initialized{false}, cpu_enabled{false};
  Status PrepareGather(MutableBufferView output, bool capturing, GatherReadArgs& args) {
    const auto workers = session.config().workers;
    auto ready = session.Ensure(region.base, workspace, GatherWorkspaceBytes(workers), capturing);
    if (!ready.ok()) return ready;
    ready = session.Ensure(region.base, output.data, output.bytes, capturing);
    if (!ready.ok()) return ready;
    const auto staging = (region.base + region.length + 63U) & ~std::uint64_t{63U};
    const auto descriptors = staging + workers * lane_staging_bytes;
    const auto controls = descriptors + workers * GatherLaneDescriptorBytes(workers);
    if (!cpu_enabled) {
      if (capturing)
        return Status::InvalidState("CPU gather lanes are not prepared; warm up outside capture");
      ready = session.config().peer_link->EnableGatherLanes(region.base, controls, workers);
      if (!ready.ok()) return ready;
      cpu_enabled = true;
    }
    args.execution_control = PointerValue(control);
    args.context = {session.session(), session.owner_pid()};
    args.core_status = PointerValue(execution.status);
    args.scratch = PointerValue(workspace);
    args.source_descriptor = args.scratch + kGatherDescriptorOffset;
    args.output = PointerValue(output.data);
    args.output_extent = output.bytes;
    args.remote_staging = staging;
    args.remote_descriptor = descriptors;
    args.remote_notify = controls;
    args.remote_ack = controls + 8U;
    args.remote_command = controls + 16U;
    args.worker_count = workers;
    return Status::OK();
  }
  Status Drain(const char* stage = "drain", bool check_status = true) {
    const auto stream = pending ? pending : captured_stream;
    if (!stream) return check_status ? failure : Status::OK();
    bool capturing = false;
    auto allowed = kvaf::detail::BatchGetCapture(stream, true, false, &capturing);
    if (!allowed.ok()) return allowed;
    if (check_status && failure.ok() && submitted && aclrtMemcpyAsync(status_host, execution.status_bytes, execution.status,
          execution.status_bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream) != ACL_SUCCESS)
      return Internal("GatherRead status enqueue failed; resources retained");
    const auto result = aclrtSynchronizeStream(stream);
    if (result != ACL_SUCCESS) {
      const char* detail = aclGetRecentErrMsg();
      return Internal("GatherRead stream synchronization failed: acl=" + std::to_string(result) +
                      " stage=" + stage +
                      " workers=" + std::to_string(session.config().workers) +
                      " source_mode=" + (direct_addresses ? "direct" : "table") +
                      "; buffers retained; detail=" + (detail ? detail : "unavailable"));
    }
    pending = nullptr;
    if (check_status && !failure.ok()) return failure;
    if (!check_status || !submitted) return Status::OK();
    auto checked = ValidateStatus();
    if (last_was_range) range_status = checked;
    return checked;
  }
  Status ValidateStatus() {
    const auto* statuses = static_cast<const GatherReadStatus*>(status_host);
    const auto sequence = statuses[0].sequence;
    const auto recoverable = [&](const GatherReadStatus& status) {
      return last_was_range && IsGatherRangeArgumentError(status.result);
    };
    // Report an actual worker failure before a partial launch's stale slots.
    for (unsigned worker = 0; worker < session.config().workers; ++worker)
      if (statuses[worker].magic == kGatherStatusMagic && statuses[worker].result &&
          !recoverable(statuses[worker])) {
        failure = Internal("GatherRead worker " + std::to_string(worker) +
            " failed: sequence=" + std::to_string(statuses[worker].sequence) +
            " phase=" + std::to_string(statuses[worker].phase) +
            " result=" + std::to_string(statuses[worker].result));
        return failure;
      }
    for (unsigned worker = 0; worker < session.config().workers; ++worker)
      if (statuses[worker].magic != kGatherStatusMagic || !sequence ||
          statuses[worker].sequence != sequence)
        return failure = Internal("GatherRead worker " + std::to_string(worker) +
            " failed: sequence=" + std::to_string(statuses[worker].sequence) +
            " phase=" + std::to_string(statuses[worker].phase) +
            " result=" + std::to_string(statuses[worker].result));
    for (unsigned worker = 0; last_was_range && worker < session.config().workers; ++worker) {
      const auto& status = statuses[worker];
      if (status.result == kGatherStagingExceeded)
        return InvalidArgument("GatherRead range staging capacity exceeded: worker=" +
            std::to_string(worker) + " request=" + std::to_string(status.failed_request) +
            " required_at_least=" + std::to_string(status.required_bytes) +
            " capacity=" + std::to_string(std::min<std::uint64_t>(lane_staging_bytes, UINT32_MAX)));
      if (status.result == kInvalidGatherRequest)
        return InvalidArgument("invalid GatherRead range request: worker=" + std::to_string(worker) +
            (status.failed_request == UINT64_MAX ? " (invalid request_count or layout)" :
             " request=" + std::to_string(status.failed_request)));
    }
    return Status::OK();
  }
};
BatchGetTransport::BatchGetTransport(UmdkSession& session, std::uint32_t token_bytes,
                                     bool direct_addresses)
    : impl_(std::make_unique<Impl>(session, token_bytes, direct_addresses)) {}
BatchGetTransport::~BatchGetTransport() = default;
Status BatchGetTransport::Drain(bool check_status) { return impl_->Drain("drain", check_status); }
Status BatchGetTransport::Check(aclrtStream stream) {
  if (!impl_->initialized || !impl_->submitted || impl_->last_was_range)
    return Status::InvalidState("no AICPU BatchGet has been submitted");
  if (stream != impl_->last_stream)
    return InvalidArgument("check_batch_get_status requires the BatchGet execution stream");
  impl_->pending = stream;
  return impl_->Drain("check");
}
Status BatchGetTransport::CheckRanges(aclrtStream stream) {
  if (!impl_->initialized || !impl_->last_was_range)
    return Status::InvalidState("no current AICPU range Get result");
  if (stream != impl_->last_stream)
    return InvalidArgument("check_batch_get_ranges_status requires the range Get execution stream");
  if (!impl_->pending) return impl_->failure.ok() ? impl_->range_status : impl_->failure;
  return impl_->Drain("check_ranges");
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
  impl_->profile_submitted = false;
  return Status::OK();
}
Status BatchGetTransport::CopyKernelProfiles(std::vector<BatchGetKernelProfile>* profiles) const {
  if (!impl_->profiling || !impl_->profile_submitted)
    return Status::InvalidState("no profiled GatherRead launch");
  bool capturing = false;
  auto ready = kvaf::detail::BatchGetCapture(impl_->last_stream, true, false, &capturing);
  if (!ready.ok()) return ready;
  return kvaf::detail::CopyAicpuProfiles(impl_->profile_device, impl_->profile_host,
      impl_->session.config().workers, profiles);
}
Status BatchGetTransport::Initialize(const RemoteRegionDesc& region, bool slot_sized_staging) {
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
  const auto workers = impl_->session.config().workers;
  const auto min_lane_bytes = slot_sized_staging ? region.slot_size : 0U;
  const auto workspace_bytes = GatherRemoteWorkspaceBytes(impl_->token_bytes, workers, min_lane_bytes);
  if (!impl_->token_bytes || registered.base > UINT64_MAX - registered.length ||
      registered.length < region.length || workspace_bytes > registered.length - region.length) {
    (void)Shutdown();
    return InvalidArgument("UMDK BatchGet requires reserved remote gather workspace");
  }
  impl_->lane_staging_bytes = GatherLaneStagingBytes(impl_->token_bytes, workers, min_lane_bytes);
  impl_->region = region;
  impl_->initialized = true;
  return Status::OK();
}
Status BatchGetTransport::Shutdown() {
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
  impl_->pending = impl_->captured_stream = impl_->last_stream = nullptr;
  impl_->submitted = false;
  impl_->last_was_range = false;
  impl_->range_status = Status::OK();
  impl_->profiling = impl_->profile_submitted = false;
  impl_->failure = Status::OK();
  impl_->initialized = impl_->cpu_enabled = false;
  return Status::OK();
}
Status BatchGetTransport::Execute(const BatchGetExecRequest& exec) {
  auto* profile = exec.execution.profile;
  if (profile && !exec.execution.synchronize)
    return InvalidArgument("BatchGet profiling requires synchronize=true");
  if (profile) *profile = {};
  const auto begin = profile ? BatchGetProfileNowNs() : 0U;
  if (!impl_->initialized || !exec.execution.stream)
    return InvalidArgument("UMDK BatchGet needs initialization and stream");
  if (!impl_->failure.ok()) return impl_->failure;
  const auto stream = static_cast<aclrtStream>(exec.execution.stream);
  bool capturing = false;
  auto capture = kvaf::detail::BatchGetCapture(stream, exec.execution.synchronize, profile != nullptr, &capturing);
  if (!capture.ok()) return capture;
  if (impl_->captured_stream && impl_->captured_stream != stream)
    return InvalidArgument("captured AICPU transport must use its capture stream");
  if (capturing) {
    if (impl_->last_was_range && impl_->pending)
      return Status::InvalidState("check pending AICPU range Get before capture");
    if (impl_->pending && impl_->pending != stream)
      return Status::InvalidState("drain previous AICPU work outside capture on its stream");
  } else {
    auto drained = impl_->Drain("previous_request");
    if (!drained.ok()) return drained;
  }
  UmdkBatchGetArgs launch{};
  auto valid = BuildBatchGetKernelArgs(exec, &launch.request);
  if (!valid.ok()) return valid;
  if (launch.request.request_capacity > kGatherMaxRequests ||
      launch.request.token_bytes != impl_->token_bytes)
    return InvalidArgument("UMDK BatchGet exceeds 65536 requests or configured token size");
  if (impl_->direct_addresses) {
    if (exec.request->access_table.opaque != 0U)
      return InvalidArgument("gather_read_direct requires the addresses Tensor overload");
    if (launch.request.table_capacity < launch.request.request_capacity)
      return InvalidArgument("direct addresses must cover the request capacity");
    launch.request.indices = launch.request.offsets = 0U;
  }
  const auto workers = impl_->session.config().workers;
  const auto profile_bytes = workers * sizeof(BatchGetWorkerProfile);
  if (profile || impl_->profiling) {
    auto allocated = kvaf::detail::AllocateAicpuProfiles(&impl_->profile_device, &impl_->profile_host, profile_bytes);
    if (!allocated.ok()) return allocated;
  }
  auto& args = launch.gather;
  auto ready = impl_->PrepareGather(exec.request->output, capturing, args);
  if (!ready.ok()) return ready;
  args.profile = (profile || impl_->profiling) ? PointerValue(impl_->profile_device) : 0U;
  impl_->pending = stream;
  impl_->last_stream = stream;
  impl_->last_was_range = false;
  if (capturing) impl_->captured_stream = stream;
  auto stamp = profile ? BatchGetProfileNowNs() : 0U;
  if (profile) profile->prepare_ns = stamp - begin;
  if (LaunchBatchGet(&launch, stream))
    return impl_->failure = Internal("UMDK BatchGet launch failed; shutdown required");
  impl_->submitted = true;
  impl_->profile_submitted = args.profile != 0;
  if (profile) {
    const auto now = BatchGetProfileNowNs();
    profile->launch_ns = now - stamp;
    stamp = now;
  }
  if (!exec.execution.synchronize) return Status::OK();
  if (aclrtMemcpyAsync(impl_->status_host, impl_->execution.status_bytes, impl_->execution.status,
                       impl_->execution.status_bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream) != ACL_SUCCESS)
    return Internal("BatchGet status enqueue failed");
  if (profile) {
    if (aclrtMemcpyAsync(impl_->profile_host, profile_bytes, impl_->profile_device,
                         profile_bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream) != ACL_SUCCESS)
      return Internal("BatchGet profile enqueue failed");
    const auto now = BatchGetProfileNowNs();
    profile->status_enqueue_ns = now - stamp;
    stamp = now;
  }
  auto completed = impl_->Drain("completion", false);
  if (profile) {
    const auto now = BatchGetProfileNowNs();
    profile->wait_ns = now - stamp;
    stamp = now;
  }
  if (!completed.ok()) return completed;
  auto checked = impl_->ValidateStatus();
  if (!checked.ok()) return checked;
  if (profile) {
    profile->worker_count = workers;
    std::copy_n(static_cast<const BatchGetWorkerProfile*>(impl_->profile_host), workers,
                profile->workers.begin());
    const auto end = BatchGetProfileNowNs();
    profile->finish_ns = end - stamp;
    profile->total_ns = end - begin;
  }
  return Status::OK();
}

Status BatchGetTransport::ExecuteRanges(const BatchGetRangesExecRequest& exec) {
  if (!impl_->initialized) return Status::NotInitialized("GatherRead is not initialized");
  if (!exec.request || !exec.execution.stream || exec.execution.profile)
    return InvalidArgument("GatherRead range Get requires a stream and does not support profiling");
  const auto& r = *exec.request;
  const auto& table = r.device_access_table;
  const auto requests = r.indices.bytes / sizeof(std::int32_t);
  const auto entries = table.addresses.bytes / sizeof(Address);
  if (r.indices.device != DeviceKind::kNpu || !requests || requests > kGatherMaxRequests ||
      !entries || entries > UINT32_MAX)
    return InvalidArgument("GatherRead range Get requires an NPU table and 1..65536 request slots");
  const auto stream = static_cast<aclrtStream>(exec.execution.stream);
  bool capturing = false;
  auto ready = kvaf::detail::BatchGetCapture(stream, false, false, &capturing);
  if (!ready.ok()) return ready;
  if (capturing || impl_->captured_stream)
    return Status::NotImplemented("GatherRead range Get requires eager execution on an uncaptured Runtime");
  ready = impl_->Drain("previous_request");
  if (!ready.ok()) return ready;

  UmdkBatchGetRangesArgs launch{};
  launch.request = {PointerValue(table.addresses.data), PointerValue(table.valid.data),
      PointerValue(table.lengths.data), PointerValue(r.indices.data), PointerValue(r.source_offsets.data),
      PointerValue(r.lengths.data), PointerValue(r.request_count.data), PointerValue(r.output_offsets.data),
      static_cast<std::uint32_t>(entries), static_cast<std::uint32_t>(requests), r.read_chunk_bytes};
  launch.kv_base = impl_->region.base;
  launch.kv_bytes = impl_->region.length;
  launch.lane_staging_bytes = impl_->lane_staging_bytes;
  ready = impl_->PrepareGather(r.output, false, launch.gather);
  if (!ready.ok()) return ready;

  // Track the status reset and launch together. Neither buffer is reusable
  // until the stream has drained, including when enqueue fails.
  impl_->pending = impl_->last_stream = stream;
  impl_->last_was_range = true;
  impl_->submitted = impl_->profile_submitted = false;
  impl_->range_status = Status::OK();
  auto& state = impl_->execution;
  if (aclrtMemsetAsync(state.status, state.status_bytes, 0, state.status_bytes, stream) != ACL_SUCCESS ||
      LaunchBatchGetRanges(&launch, stream) != 0)
    return impl_->failure = Internal("UMDK range Get enqueue failed; shutdown required");
  impl_->submitted = true;
  return exec.execution.synchronize ? impl_->Drain("range_completion") : Status::OK();
}
}  // namespace kvaf::aicpu_urma
