#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
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

bool ArgsDumpEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("KVAF_ARGS_DUMP");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }();
  return enabled;
}

// Print the exact byte block handed to the AICPU launcher. CANN rejects the
// captured func call with a bare "execute kernel param invalid" and no field
// name, so the eager and captured dumps must be diffed to see what changed.
void DumpLaunchArgs(const UmdkBatchGetArgs& launch, aclrtStream stream, bool capturing,
                    std::size_t slot_index) {
  if (!ArgsDumpEnabled()) return;
  static std::atomic<unsigned long long> counter{0};
  const unsigned long long seq = ++counter;
  const auto& r = launch.request;
  const auto& g = launch.gather;
  std::fprintf(stderr,
               "[KVAF_ARGS] seq=%llu capturing=%d stream=%p slot=%zu bytes=%zu blocks=%u "
               "off{token_bytes=%zu gather=%zu}\n",
               seq, capturing ? 1 : 0, static_cast<void*>(stream), slot_index, sizeof(launch),
               g.worker_count, offsetof(UmdkBatchGetArgs, request.token_bytes),
               offsetof(UmdkBatchGetArgs, gather));
  std::fprintf(stderr,
               "[KVAF_ARGS] request table=%llu indices=%llu offsets=%llu miss=%llu out_idx=%llu "
               "table_cap=%u req_cap=%u token_bytes=%u\n",
               static_cast<unsigned long long>(r.address_table),
               static_cast<unsigned long long>(r.indices),
               static_cast<unsigned long long>(r.offsets),
               static_cast<unsigned long long>(r.miss_count),
               static_cast<unsigned long long>(r.output_indices), r.table_capacity,
               r.request_capacity, r.token_bytes);
  std::fprintf(stderr,
               "[KVAF_ARGS] gather session=%llu pid=%llu core_status=%llu output=%llu extent=%llu "
               "scratch=%llu spans_ptr=%llu exec_ctl=%llu\n",
               static_cast<unsigned long long>(g.context.session),
               static_cast<unsigned long long>(g.context.owner_pid),
               static_cast<unsigned long long>(g.core_status),
               static_cast<unsigned long long>(g.output),
               static_cast<unsigned long long>(g.output_extent),
               static_cast<unsigned long long>(g.scratch),
               static_cast<unsigned long long>(g.source_descriptor),
               static_cast<unsigned long long>(g.execution_control));
  std::fprintf(stderr,
               "[KVAF_ARGS] gather staging=%llu desc=%llu cmd=%llu notify=%llu ack=%llu "
               "bytes=%u seq=%llu spans=%u sgl=%u workers=%u wqes=%u widx=%u chunk=%u "
               "profile=%llu\n",
               static_cast<unsigned long long>(g.remote_staging),
               static_cast<unsigned long long>(g.remote_descriptor),
               static_cast<unsigned long long>(g.remote_command),
               static_cast<unsigned long long>(g.remote_notify),
               static_cast<unsigned long long>(g.remote_ack), g.bytes, g.source_span_count,
               static_cast<unsigned long long>(g.sequence), g.sgl_size, g.worker_count,
               g.sq_batch_wqes, g.worker_index, g.pipeline_chunk_bytes,
               static_cast<unsigned long long>(g.profile));
  const auto* raw = reinterpret_cast<const unsigned char*>(&launch);
  std::fprintf(stderr, "[KVAF_ARGS] hex=");
  for (std::size_t i = 0; i < sizeof(launch); ++i) std::fprintf(stderr, "%02x", raw[i]);
  std::fprintf(stderr, "\n");
  std::fflush(stderr);
}

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
      const char* detail = aclGetRecentErrMsg();
      return Internal("GatherRead stream synchronization failed: acl=" + std::to_string(result) +
                      " stage=" + stage +
                      " workers=" + std::to_string(session.config().workers) +
                      " source_mode=" + (direct_addresses ? "direct" : "table") +
                      "; buffers retained; detail=" + (detail ? detail : "unavailable"));
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
  auto stamp = profile ? BatchGetProfileNowNs() : 0U;
  if (profile) profile->prepare_ns = stamp - begin;
  DumpLaunchArgs(launch, stream, capturing, slot_index);
  if (LaunchBatchGet(&launch, stream))
    return impl_->failure = Internal("UMDK BatchGet launch failed; shutdown required");
  slot->submitted = true;
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
