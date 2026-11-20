#include <algorithm>
#include <array>
#include <string>
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
  Impl(UmdkSession& shared, std::uint32_t token, bool direct, std::uint64_t offset)
      : session(shared), token_bytes(token), direct_addresses(direct), staging_offset(offset) {}
  UmdkSession& session;
  std::uint32_t token_bytes;
  bool direct_addresses;
  // Bytes into the reserved remote workspace where this width's staging starts.
  std::uint64_t staging_offset;
  RemoteRegionDesc region;
  BatchGetExecutionState execution;
  void* workspace{nullptr};
  void* control{nullptr};
  void* status_host{nullptr};
  void* profile_device{nullptr};
  void* profile_host{nullptr};
  // One AICPU session serves both the main and the prefetch stream, so every
  // stream that runs BatchGet owns a slot and a private status window.
  static constexpr std::size_t kMaxStreams = 4;
  struct StreamSlot {
    aclrtStream stream{nullptr};
    bool submitted{false};
  };
  std::array<StreamSlot, kMaxStreams> slots{};
  bool profiling{false}, profile_submitted{false};
  Status failure;
  bool initialized{false}, cpu_enabled{false};

  StreamSlot* FindSlot(aclrtStream stream) {
    for (auto& slot : slots)
      if (slot.stream == stream) return &slot;
    return nullptr;
  }
  StreamSlot* ClaimSlot(aclrtStream stream) {
    if (auto* existing = FindSlot(stream)) return existing;
    for (auto& slot : slots)
      if (slot.stream == nullptr) {
        slot.stream = stream;
        slot.submitted = false;
        return &slot;
      }
    return nullptr;
  }
  std::size_t SlotIndex(const StreamSlot& slot) const {
    return static_cast<std::size_t>(&slot - slots.data());
  }
  void* StatusWindow(std::size_t index) const {
    return static_cast<std::uint8_t*>(execution.status) + index * execution.status_bytes;
  }
  void* StatusHostWindow(std::size_t index) const {
    return static_cast<std::uint8_t*>(status_host) + index * execution.status_bytes;
  }
  aclrtStream ActiveStream() const {
    for (const auto& slot : slots)
      if (slot.stream != nullptr) return slot.stream;
    return nullptr;
  }
  // The AICPU worker keeps its phase counter in an HBM status window, and the
  // execution control block keeps the global claim counters. A stream error
  // discards the queued device-to-host copy, so read both again with a plain
  // synchronous copy: that copy does not depend on the broken stream and it is
  // the only record of where inside the gather the worker stopped.
  std::string DescribeStatus(std::size_t index) {
    std::string text = "; AICPU";
    GatherExecutionControl counters{};
    if (aclrtMemcpy(&counters, sizeof(counters), control, sizeof(counters),
                    ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS)
      text += " claimed=" + std::to_string(counters.claimed) +
              " failed=" + std::to_string(counters.failed);
    else
      text += " counters=unreadable";
    if (aclrtMemcpy(StatusHostWindow(index), execution.status_bytes, StatusWindow(index),
                    execution.status_bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
      return text + " status=unreadable;" + LastLaunchText();
    const auto* statuses = static_cast<const GatherReadStatus*>(StatusHostWindow(index));
    for (unsigned worker = 0; worker < session.config().workers; ++worker) {
      const auto& state = statuses[worker];
      text += " w" + std::to_string(worker) + "{magic=" + std::to_string(state.magic) +
              " seq=" + std::to_string(state.sequence) +
              " phase=" + std::to_string(state.phase) +
              " result=" + std::to_string(state.result) +
              " ack=" + std::to_string(state.ack_seen) +
              " cqe=" + std::to_string(state.cqe_word) + "}";
    }
    // phase 1=write source list 2=notify cpu 3=wait cpu ack 4=read staging
    // 5=wait ready prefix; result uses GatherReadError.
    return text + ";" + LastLaunchText();
  }
  // Identifies the launch that owned the stream when the fault was reported.
  // DrainSlot runs before the next launch is built, so this still describes the
  // kernel that actually failed even when the error surfaces one submit later.
  struct LastLaunch {
    std::string number, bytes, capacity, token_bytes, spans, workers, output, extent, sequence,
        notify, ack, command, staging, descriptor;
  };
  std::string LastLaunchText() const {
    return " #" + last_launch.number + " bytes=" + last_launch.bytes +
           " cap=" + last_launch.capacity + " token_bytes=" + last_launch.token_bytes +
           " spans=" + last_launch.spans + " workers=" + last_launch.workers +
           " output=" + last_launch.output + " extent=" + last_launch.extent +
           " sid=" + last_launch.sequence + " notify=" + last_launch.notify +
           " ack=" + last_launch.ack + " cmd=" + last_launch.command +
           " staging=" + last_launch.staging + " desc=" + last_launch.descriptor;
  }
  LastLaunch last_launch;
  // One-based launch number, so a crash report can be matched to a dump.
  unsigned launch_count{0};
  Status DrainSlot(StreamSlot& slot, const char* stage, bool check_status) {
    if (!slot.stream) return check_status ? failure : Status::OK();
    bool capturing = false;
    auto allowed = kvaf::detail::BatchGetCapture(slot.stream, true, false, &capturing);
    if (!allowed.ok()) return allowed;
    const std::size_t index = SlotIndex(slot);
    if (check_status && failure.ok() && slot.submitted &&
        aclrtMemcpyAsync(StatusHostWindow(index), execution.status_bytes, StatusWindow(index),
                         execution.status_bytes, ACL_MEMCPY_DEVICE_TO_HOST, slot.stream) != ACL_SUCCESS)
      return Internal("GatherRead status enqueue failed; resources retained");
    const auto result = aclrtSynchronizeStream(slot.stream);
    if (result != ACL_SUCCESS) {
      // aclGetRecentErrMsg stays valid only until the next runtime call, so copy
      // it before the status read below replaces it.
      const char* detail = aclGetRecentErrMsg();
      const std::string reason = detail ? detail : "unavailable";
      return Internal("GatherRead stream synchronization failed: acl=" + std::to_string(result) +
                      " stage=" + stage +
                      " workers=" + std::to_string(session.config().workers) +
                      " source_mode=" + (direct_addresses ? "direct" : "table") +
                      "; buffers retained; detail=" + reason + DescribeStatus(index));
    }
    if (check_status && !failure.ok()) return failure;
    if (check_status && slot.submitted) return ValidateStatus(index);
    return Status::OK();
  }
  Status Drain(const char* stage = "drain", bool check_status = true) {
    for (auto& slot : slots) {
      const Status status = DrainSlot(slot, stage, check_status);
      if (!status.ok()) return status;
    }
    return Status::OK();
  }
  Status ValidateStatus(std::size_t index) {
    const auto* statuses = static_cast<const GatherReadStatus*>(StatusHostWindow(index));
    const auto sequence = statuses[0].sequence;
    // Report an actual worker failure before a partial launch's stale slots.
    for (unsigned worker = 0; worker < session.config().workers; ++worker)
      if (statuses[worker].magic == kGatherStatusMagic && statuses[worker].result) {
        failure = Internal("GatherRead worker " + std::to_string(worker) +
            " failed: sequence=" + std::to_string(statuses[worker].sequence) +
            " phase=" + std::to_string(statuses[worker].phase) +
            " result=" + std::to_string(statuses[worker].result));
        return failure;
      }
    for (unsigned worker = 0; worker < session.config().workers; ++worker)
      if (statuses[worker].magic != kGatherStatusMagic || !sequence ||
          statuses[worker].sequence != sequence || statuses[worker].result)
        return failure = Internal("GatherRead worker " + std::to_string(worker) +
            " failed: sequence=" + std::to_string(statuses[worker].sequence) +
            " phase=" + std::to_string(statuses[worker].phase) +
            " result=" + std::to_string(statuses[worker].result));
    return Status::OK();
  }
};
BatchGetTransport::BatchGetTransport(UmdkSession& session, std::uint32_t token_bytes,
                                     bool direct_addresses, std::uint64_t staging_offset)
    : impl_(std::make_unique<Impl>(session, token_bytes, direct_addresses, staging_offset)) {}
BatchGetTransport::~BatchGetTransport() = default;
Status BatchGetTransport::Drain(bool check_status) { return impl_->Drain("drain", check_status); }
Status BatchGetTransport::Check(aclrtStream stream) {
  if (!impl_->initialized) return Status::NotInitialized("GatherRead is not initialized");
  auto* slot = impl_->FindSlot(stream);
  if (slot == nullptr || !slot->submitted)
    return Status::InvalidState("no AICPU BatchGet has been submitted on this stream");
  return impl_->DrainSlot(*slot, "check", true);
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
  auto ready = kvaf::detail::BatchGetCapture(impl_->ActiveStream(), true, false, &capturing);
  if (!ready.ok()) return ready;
  return kvaf::detail::CopyAicpuProfiles(impl_->profile_device, impl_->profile_host,
      impl_->session.config().workers, profiles);
}
Status BatchGetTransport::Initialize(const RemoteRegionDesc& region) {
  if (impl_->initialized) return Status::InvalidState("UMDK operator already initialized");
  auto& state = impl_->execution;
  state.status_bytes = impl_->session.config().workers * sizeof(GatherReadStatus);
  // One status window per concurrent stream; kernels write their worker state there.
  const auto status_total = Impl::kMaxStreams * state.status_bytes;
  if (aclrtMalloc(&state.status, status_total, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
      aclrtMalloc(&impl_->workspace, GatherWorkspaceBytes(impl_->session.config().workers),
                  ACL_MEM_MALLOC_HUGE_FIRST) !=
          ACL_SUCCESS ||
      aclrtMallocHost(&impl_->status_host, status_total) != ACL_SUCCESS ||
      aclrtMalloc(&impl_->control, sizeof(GatherExecutionControl), ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS ||
      aclrtMemset(impl_->control, sizeof(GatherExecutionControl), 0, sizeof(GatherExecutionControl)) != ACL_SUCCESS ||
      aclrtMemset(state.status, status_total, 0, status_total) != ACL_SUCCESS) {
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
      registered.length < region.length + impl_->staging_offset +
          GatherRemoteWorkspaceBytes(impl_->token_bytes, impl_->session.config().workers)) {
    (void)Shutdown();
    return InvalidArgument("UMDK BatchGet requires reserved remote gather workspace");
  }
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
  for (auto& slot : impl_->slots) slot = Impl::StreamSlot{};
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
  auto* slot = impl_->ClaimSlot(stream);
  if (slot == nullptr)
    return InvalidArgument("AICPU BatchGet exceeds the concurrent stream limit");
  if (!capturing) {
    // Eager launches on the same stream must not overlap.
    auto drained = impl_->DrainSlot(*slot, "previous_request", false);
    if (!drained.ok()) return drained;
  }
  const std::size_t slot_index = impl_->SlotIndex(*slot);
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
  auto ensure = [&](void* buffer, std::size_t bytes) {
    return impl_->session.Ensure(impl_->region.base, buffer, bytes, capturing);
  };
  const auto workers = impl_->session.config().workers;
  const auto profile_bytes = workers * sizeof(BatchGetWorkerProfile);
  if (profile || impl_->profiling) {
    auto allocated = kvaf::detail::AllocateAicpuProfiles(&impl_->profile_device, &impl_->profile_host, profile_bytes);
    if (!allocated.ok()) return allocated;
  }
  auto ready = ensure(impl_->workspace, GatherWorkspaceBytes(workers));
  if (!ready.ok()) return ready;
  ready = ensure(exec.request->output.data, exec.request->output.bytes);
  if (!ready.ok()) return ready;
  const auto staging = ((impl_->region.base + impl_->region.length + 63U) & ~std::uint64_t{63U}) +
                       impl_->staging_offset;
  const auto descriptors = staging + GatherStagingBytes(impl_->token_bytes, workers);
  const auto controls = descriptors + workers * GatherLaneDescriptorBytes(workers);
  if (!impl_->cpu_enabled) {
    if (capturing)
      return Status::InvalidState("CPU gather lanes are not prepared; warm up outside capture");
    ready = impl_->session.config().peer_link->EnableGatherLanes(impl_->region.base, controls, workers);
    if (!ready.ok()) return ready;
    impl_->cpu_enabled = true;
  }
  auto& args = launch.gather;
  args.execution_control = PointerValue(impl_->control);
  args.context = {impl_->session.session(), impl_->session.owner_pid()};
  args.core_status = PointerValue(impl_->StatusWindow(slot_index));
  args.scratch = PointerValue(impl_->workspace);
  args.source_descriptor = args.scratch + kGatherDescriptorOffset;
  args.output = PointerValue(exec.request->output.data);
  args.output_extent = exec.request->output.bytes;
  args.remote_staging = staging;
  args.remote_descriptor = descriptors;
  args.remote_notify = controls;
  args.remote_ack = args.remote_notify + 8U;
  args.remote_command = args.remote_notify + 16U;
  args.worker_count = impl_->session.config().workers;
  args.profile = (profile || impl_->profiling) ? PointerValue(impl_->profile_device) : 0U;
  slot->submitted = false;
  impl_->last_launch = {std::to_string(++impl_->launch_count),
                        std::to_string(launch.gather.bytes),
                        std::to_string(launch.request.request_capacity),
                        std::to_string(launch.request.token_bytes),
                        std::to_string(launch.gather.source_span_count),
                        std::to_string(launch.gather.worker_count),
                        std::to_string(launch.gather.output),
                        std::to_string(launch.gather.output_extent),
                        std::to_string(launch.gather.sequence),
                        std::to_string(launch.gather.remote_notify),
                        std::to_string(launch.gather.remote_ack),
                        std::to_string(launch.gather.remote_command),
                        std::to_string(launch.gather.remote_staging),
                        std::to_string(launch.gather.remote_descriptor)};
  auto stamp = profile ? BatchGetProfileNowNs() : 0U;
  if (profile) profile->prepare_ns = stamp - begin;
  if (LaunchBatchGet(&launch, stream))
    return impl_->failure = Internal("UMDK BatchGet launch failed; shutdown required");
  // A captured launch is recorded, not executed. Marking it submitted would force
  // the next eager Drain to copy back a status window that the kernel never wrote
  // (all zeros), and Put drains every width before it reuses the session.
  slot->submitted = !capturing;
  impl_->profile_submitted = args.profile != 0;
  if (profile) {
    const auto now = BatchGetProfileNowNs();
    profile->launch_ns = now - stamp;
    stamp = now;
  }
  if (!exec.execution.synchronize) return Status::OK();
  if (aclrtMemcpyAsync(impl_->StatusHostWindow(slot_index), impl_->execution.status_bytes,
                       impl_->StatusWindow(slot_index),
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
  auto completed = impl_->DrainSlot(*slot, "completion", false);
  if (profile) {
    const auto now = BatchGetProfileNowNs();
    profile->wait_ns = now - stamp;
    stamp = now;
  }
  if (!completed.ok()) return completed;
  auto checked = impl_->ValidateStatus(slot_index);
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
}  // namespace kvaf::aicpu_urma
