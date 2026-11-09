#include <algorithm>
#include <utility>

#include "aicpu_batch_get/include/kvaf_aicpu_batch_get.h"
#include "batch_get_args.h"
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
  BatchGetExecutionState execution;
  void* workspace{nullptr};
  void* status_host{nullptr};
  void* profile_device{nullptr};
  void* profile_host{nullptr};
  struct Invocation final : ExecutionState {
    explicit Invocation(const Impl* value) : owner(value) {}
    const Impl* owner;
    aclrtStream pending{};
    std::uint64_t sequence{};
    const BatchGetRequest* prepared{};
    void* prepared_stream{};
    BatchGetProfile* prepared_profile{};
    std::uint64_t prepare_started{};
  };
  Invocation legacy{this};
  Invocation* invocation{&legacy};
  std::uint64_t sequence{0U};
  bool initialized{false}, cpu_enabled{false};
  Status Drain(const char* stage = "drain") {
    if (!invocation->pending) return Status::OK();
    const auto result = aclrtSynchronizeStream(invocation->pending);
    if (result != ACL_SUCCESS) {
      const char* detail = aclGetRecentErrMsg();
      return Internal("GatherRead stream synchronization failed: acl=" + std::to_string(result) +
                      " stage=" + stage + " sequence=" + std::to_string(sequence) +
                      " workers=" + std::to_string(session.config().workers) +
                      " source_mode=" + (direct_addresses ? "direct" : "table") +
                      "; buffers retained; detail=" + (detail ? detail : "unavailable"));
    }
    if (aclrtMemcpy(status_host, execution.status_bytes, execution.status,
                    execution.status_bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
      return Internal("GatherRead completion status unavailable; buffers retained");
    const auto* reports = static_cast<const GatherReadStatus*>(status_host);
    for (unsigned worker = 0; worker < session.config().workers; ++worker)
      if (reports[worker].magic != kGatherStatusMagic || reports[worker].sequence != invocation->sequence ||
          reports[worker].result != 0)
        return Internal("GatherRead worker failed; session retained for recovery");
    invocation->pending = nullptr;
    return Status::OK();
  }
};
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
  return Status::OK();
}
Status BatchGetTransport::Drain() { return impl_->Drain(); }
Status BatchGetTransport::Initialize(const RemoteRegionDesc& region) {
  if (impl_->initialized) return Status::InvalidState("UMDK operator already initialized");
  auto& state = impl_->execution;
  state.status_bytes = impl_->session.config().workers * sizeof(GatherReadStatus);
  if (aclrtMalloc(&state.status, state.status_bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
      aclrtMalloc(&impl_->workspace, GatherWorkspaceBytes(impl_->session.config().workers),
                  ACL_MEM_MALLOC_HUGE_FIRST) !=
          ACL_SUCCESS ||
      aclrtMallocHost(&impl_->status_host, state.status_bytes) != ACL_SUCCESS) {
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
  return Status::OK();
}
Status BatchGetTransport::Shutdown() {
  auto drained = impl_->Drain("shutdown");
  if (!drained.ok()) return drained;
  if (impl_->workspace) (void)aclrtFree(impl_->workspace);
  if (impl_->status_host) (void)aclrtFreeHost(impl_->status_host);
  if (impl_->profile_device) (void)aclrtFree(impl_->profile_device);
  if (impl_->profile_host) (void)aclrtFreeHost(impl_->profile_host);
  if (impl_->execution.status) (void)aclrtFree(impl_->execution.status);
  impl_->workspace = impl_->status_host = nullptr;
  impl_->profile_device = impl_->profile_host = nullptr;
  impl_->execution = {};
  impl_->sequence = 0;
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
  auto& call = *impl_->invocation;
  call.prepared = nullptr;
  if (exec.execution.profile && !exec.execution.synchronize)
    return InvalidArgument("BatchGet profiling requires synchronize=true");
  call.prepare_started = exec.execution.profile ? BatchGetProfileNowNs() : 0;
  if (exec.execution.profile) *exec.execution.profile = {};
  auto drained = impl_->Drain("previous_request");
  if (!drained.ok()) return drained;
  if (!impl_->initialized || !exec.execution.stream)
    return InvalidArgument("UMDK BatchGet needs initialization and stream");
  UmdkBatchGetArgs checked{};
  auto valid = BuildBatchGetKernelArgs(exec, &checked.request);
  if (!valid.ok()) return valid;
  if (checked.request.request_capacity > kGatherMaxRequests || checked.request.token_bytes != impl_->token_bytes)
    return InvalidArgument("UMDK BatchGet exceeds requests or configured token size");
  if (impl_->direct_addresses && (exec.request->access_table.opaque != 0U ||
      checked.request.table_capacity < checked.request.request_capacity))
    return InvalidArgument("invalid direct address request");
  const auto profile_bytes = impl_->session.config().workers * sizeof(BatchGetWorkerProfile);
  if (exec.execution.profile &&
      ((!impl_->profile_device && aclrtMalloc(&impl_->profile_device, profile_bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) ||
       (!impl_->profile_host && aclrtMallocHost(&impl_->profile_host, profile_bytes) != ACL_SUCCESS)))
    return Internal("failed to allocate BatchGet profiling buffers");
  auto ready = Warmup();
  if (!ready.ok()) return ready;
  ready = impl_->session.Ensure(impl_->region.base, exec.request->output.data, exec.request->output.bytes);
  if (!ready.ok()) return ready;
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
  auto* profile = exec.execution.profile;
  if (profile && !exec.execution.synchronize)
    return InvalidArgument("BatchGet profiling requires synchronize=true");
  auto& call = *impl_->invocation;
  if (call.prepared != exec.request || call.prepared_stream != exec.execution.stream ||
      call.prepared_profile != profile || !call.prepared)
    return Status::InvalidState("Gather resources were not prepared for this request");
  call.prepared = nullptr;
  const auto begin = call.prepare_started;
  UmdkBatchGetArgs launch{};
  auto valid = BuildBatchGetKernelArgs(exec, &launch.request);
  if (!valid.ok()) return valid;
  launch.request.canonical_base = impl_->region.base;
  launch.request.logical_bytes = impl_->region.length;
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
  const auto staging = (impl_->region.base + impl_->region.length + 63U) & ~std::uint64_t{63U};
  const auto descriptors = staging + GatherStagingBytes(impl_->token_bytes, workers);
  const auto controls = descriptors + workers * GatherLaneDescriptorBytes(workers);
  if (impl_->sequence == peer::kGatherErrorBit - 1U)
    return Status::InvalidState("BatchGet sequence exhausted");
  auto& args = launch.gather;
  args.sequence = ++impl_->sequence;
  args.context = {impl_->session.session(), impl_->session.owner_pid()};
  args.core_status = PointerValue(impl_->execution.status);
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
  args.profile = profile ? PointerValue(impl_->profile_device) : 0U;
  const auto stream = static_cast<aclrtStream>(exec.execution.stream);
  impl_->invocation->pending = stream;
  impl_->invocation->sequence = impl_->sequence;
  auto stamp = profile ? BatchGetProfileNowNs() : 0U;
  if (profile) profile->prepare_ns = stamp - begin;
  if (LaunchBatchGet(&launch, stream)) return Internal("UMDK BatchGet launch failed");
  if (profile) {
    const auto now = BatchGetProfileNowNs();
    profile->launch_ns = now - stamp;
    stamp = now;
  }
  if (!exec.execution.synchronize) return Status::OK();
  if (aclrtMemcpyAsync(impl_->status_host, impl_->execution.status_bytes, impl_->execution.status,
                       impl_->execution.status_bytes, ACL_MEMCPY_DEVICE_TO_HOST,
                       stream) != ACL_SUCCESS)
    return Internal("BatchGet status enqueue failed");
  if (profile) {
    if (aclrtMemcpyAsync(impl_->profile_host, profile_bytes, impl_->profile_device,
                         profile_bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream) != ACL_SUCCESS)
      return Internal("BatchGet profile enqueue failed");
    const auto now = BatchGetProfileNowNs();
    profile->status_enqueue_ns = now - stamp;
    stamp = now;
  }
  auto completed = impl_->Drain("completion");
  if (profile) {
    const auto now = BatchGetProfileNowNs();
    profile->wait_ns = now - stamp;
    stamp = now;
  }
  if (!completed.ok()) return completed;
  const auto* statuses = static_cast<const GatherReadStatus*>(impl_->status_host);
  for (unsigned worker = 0; worker < args.worker_count; ++worker)
    if (statuses[worker].magic != kGatherStatusMagic ||
        statuses[worker].sequence != args.sequence || statuses[worker].result)
      return Internal("BatchGet worker " + std::to_string(worker) +
                      " failed: phase=" + std::to_string(statuses[worker].phase) +
                      " result=" + std::to_string(statuses[worker].result));
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
