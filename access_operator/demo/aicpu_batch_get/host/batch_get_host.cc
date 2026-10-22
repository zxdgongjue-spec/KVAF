#include <algorithm>
#include <utility>

#include "aicpu_batch_get/include/kvaf_aicpu_batch_get.h"
#include "batch_get_args.h"
#include "cpu_gather.h"
#include "umdk/session.h"
namespace kvaf::aicpu_urma {
namespace {
Status InvalidArgument(std::string s) { return Status::InvalidArgument(std::move(s)); }
Status Internal(std::string s) { return Status::Internal(std::move(s)); }
using kvaf::detail::BatchGetExecutionState;
using kvaf::detail::BuildBatchGetKernelArgs;
using kvaf::detail::PointerValue;

}  // namespace
struct BatchGetTransport::Impl {
  Impl(UmdkSession& shared, std::uint32_t token) : session(shared), token_bytes(token) {}
  UmdkSession& session;
  std::uint32_t token_bytes;
  RemoteRegionDesc region;
  BatchGetExecutionState execution;
  void* workspace{nullptr};
  void* status_host{nullptr};
  void* profile_device{nullptr};
  void* profile_host{nullptr};
  aclrtStream pending{nullptr};
  std::uint64_t sequence{0U};
  bool initialized{false}, cpu_enabled{false};
  Status Drain() {
    if (pending && aclrtSynchronizeStream(pending) != ACL_SUCCESS)
      return Internal("GatherRead stream still has pending work; buffers retained");
    pending = nullptr;
    return Status::OK();
  }
};
BatchGetTransport::BatchGetTransport(UmdkSession& session, std::uint32_t token_bytes)
    : impl_(std::make_unique<Impl>(session, token_bytes)) {}
BatchGetTransport::~BatchGetTransport() = default;
Status BatchGetTransport::Drain() { return impl_->Drain(); }
Status BatchGetTransport::Initialize(const RemoteRegionDesc& region) {
  if (impl_->initialized) return Status::InvalidState("UMDK operator already initialized");
  auto& state = impl_->execution;
  state.status_bytes = impl_->session.config().workers * sizeof(GatherReadStatus);
  state.queue_state_bytes = impl_->session.config().workers * sizeof(GatherQueueState);
  if (aclrtMalloc(&state.status, state.status_bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
      aclrtMalloc(&state.queue_state, state.queue_state_bytes, ACL_MEM_MALLOC_HUGE_FIRST) !=
          ACL_SUCCESS ||
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
  auto drained = impl_->Drain();
  if (!drained.ok()) return drained;
  if (impl_->workspace) (void)aclrtFree(impl_->workspace);
  if (impl_->status_host) (void)aclrtFreeHost(impl_->status_host);
  if (impl_->profile_device) (void)aclrtFree(impl_->profile_device);
  if (impl_->profile_host) (void)aclrtFreeHost(impl_->profile_host);
  if (impl_->execution.status) (void)aclrtFree(impl_->execution.status);
  if (impl_->execution.queue_state) (void)aclrtFree(impl_->execution.queue_state);
  impl_->workspace = impl_->status_host = nullptr;
  impl_->profile_device = impl_->profile_host = nullptr;
  impl_->execution = {};
  impl_->sequence = 0;
  impl_->initialized = impl_->cpu_enabled = false;
  return Status::OK();
}
Status BatchGetTransport::Execute(const BatchGetExecRequest& exec) {
  auto* profile = exec.execution.profile;
  if (profile && !exec.execution.synchronize)
    return InvalidArgument("BatchGet profiling requires synchronize=true");
  if (profile) *profile = {};
  const auto begin = profile ? BatchGetProfileNowNs() : 0U;
  auto drained = impl_->Drain();
  if (!drained.ok()) return drained;
  if (!impl_->initialized || !exec.execution.stream)
    return InvalidArgument("UMDK BatchGet needs initialization and stream");
  UmdkBatchGetArgs launch{};
  auto valid = BuildBatchGetKernelArgs(exec, 1U, 1U, impl_->execution, &launch.request,
                                       sizeof(GatherReadStatus), sizeof(GatherQueueState));
  if (!valid.ok()) return valid;
  if (launch.request.request_capacity > kGatherMaxRequests ||
      launch.request.token_bytes != impl_->token_bytes)
    return InvalidArgument("UMDK BatchGet exceeds 65536 requests or configured token size");
  auto ensure = [&](void* buffer, std::size_t bytes) {
    return impl_->session.Ensure(impl_->region.base, buffer, bytes);
  };
  const auto workers = impl_->session.config().workers;
  const auto profile_bytes = workers * sizeof(BatchGetWorkerProfile);
  if (profile) {
    if ((!impl_->profile_device &&
         aclrtMalloc(&impl_->profile_device, profile_bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) ||
        (!impl_->profile_host &&
         aclrtMallocHost(&impl_->profile_host, profile_bytes) != ACL_SUCCESS))
      return Internal("failed to allocate BatchGet profiling buffers");
  }
  auto ready = ensure(impl_->workspace, GatherWorkspaceBytes(workers));
  if (!ready.ok()) return ready;
  ready = ensure(exec.request->output.data, exec.request->output.bytes);
  if (!ready.ok()) return ready;
  const auto staging = (impl_->region.base + impl_->region.length + 63U) & ~std::uint64_t{63U};
  const auto descriptors = staging + GatherStagingBytes(impl_->token_bytes, workers);
  const auto controls = descriptors + workers * GatherLaneDescriptorBytes(workers);
  if (!impl_->cpu_enabled) {
    ready = impl_->session.config().peer_link->EnableGatherLanes(impl_->region.base, controls, workers);
    if (!ready.ok()) return ready;
    impl_->cpu_enabled = true;
  }
  if (impl_->sequence == peer::kGatherErrorBit - 1U)
    return Status::InvalidState("BatchGet sequence exhausted");
  auto& args = launch.gather;
  args.sequence = ++impl_->sequence;
  args.context = {impl_->session.session(), impl_->session.owner_pid()};
  args.queue_state = PointerValue(impl_->execution.queue_state);
  args.core_status = PointerValue(impl_->execution.status);
  args.scratch = PointerValue(impl_->workspace);
  args.source_descriptor = args.scratch + kGatherDescriptorOffset;
  args.output_spans = args.scratch + GatherTargetsOffset(workers);
  args.output = PointerValue(exec.request->output.data);
  args.output_extent = exec.request->output.bytes;
  args.remote_staging = staging;
  args.remote_descriptor = descriptors;
  args.remote_data_bytes = impl_->region.length;
  args.remote_notify = controls;
  args.remote_ack = args.remote_notify + 8U;
  args.remote_command = args.remote_notify + 16U;
  args.worker_count = impl_->session.config().workers;
  args.profile = profile ? PointerValue(impl_->profile_device) : 0U;
  const auto stream = static_cast<aclrtStream>(exec.execution.stream);
  impl_->pending = stream;
  auto stamp = profile ? BatchGetProfileNowNs() : 0U;
  if (profile) profile->prepare_ns = stamp - begin;
  if (aclrtMemsetAsync(impl_->execution.queue_state, impl_->execution.queue_state_bytes, 0,
                       impl_->execution.queue_state_bytes, stream) != ACL_SUCCESS)
    return Internal("BatchGet worker counter reset failed");
  if (profile) {
    const auto now = BatchGetProfileNowNs();
    profile->reset_enqueue_ns = now - stamp;
    stamp = now;
  }
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
  auto completed = impl_->Drain();
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
