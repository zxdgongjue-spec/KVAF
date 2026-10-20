#include <algorithm>
#include <mutex>
#include <utility>

#include "aicpu_gather_access_operator.h"
#include "aicpu_urma/umdk_session.h"
#include "batch_get_args.h"
#include "cpu_gather.h"
#include "meta_manager.h"
namespace kvaf {
namespace {
using namespace aicpu_urma;
Status InvalidArgument(std::string s) { return Status::InvalidArgument(std::move(s)); }
Status Internal(std::string s) { return Status::Internal(std::move(s)); }
using detail::BatchGetExecutionState;
using detail::BuildBatchGetKernelArgs;
using detail::PointerValue;

}  // namespace
struct UmdkAccessOperator::Impl {
  explicit Impl(UmdkAccessOperatorConfig value) : config(std::move(value)) {}
  UmdkAccessOperatorConfig config;
  UmdkSession session;
  RemoteRegionDesc region;
  BatchGetExecutionState execution;
  void* workspace{nullptr};
  std::uint64_t remote_bytes{0};
  void* status_host{nullptr};
  aclrtStream pending{nullptr};
  std::uint64_t sequence{0U};
  bool initialized{false}, cpu_enabled{false};
  std::mutex mutex;
  Status Drain() {
    if (pending && aclrtSynchronizeStream(pending) != ACL_SUCCESS)
      return Internal("GatherRead stream still has pending work; buffers retained");
    pending = nullptr;
    return Status::OK();
  }
};
UmdkAccessOperator::UmdkAccessOperator(UmdkAccessOperatorConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}
UmdkAccessOperator::~UmdkAccessOperator() { (void)Shutdown(); }
Status UmdkAccessOperator::Initialize(const RemoteRegionDesc& region) {
  if (impl_->initialized) return Status::InvalidState("UMDK operator already initialized");
  if (!impl_->config.peer_link || impl_->config.kernel_json.empty() || !impl_->config.workers ||
      impl_->config.workers > 8U || !Load(impl_->config.kernel_json.c_str()))
    return InvalidArgument("UMDK operator requires PeerLink, kernel JSON and 1..8 workers");
  auto& state = impl_->execution;
  state.status_bytes = impl_->config.workers * sizeof(GatherReadStatus);
  state.queue_state_bytes = impl_->config.workers * sizeof(GatherQueueState);
  if (aclrtMalloc(&state.status, state.status_bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
      aclrtMalloc(&state.queue_state, state.queue_state_bytes, ACL_MEM_MALLOC_HUGE_FIRST) !=
          ACL_SUCCESS ||
      aclrtMalloc(&impl_->workspace, kGatherWorkspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST) !=
          ACL_SUCCESS ||
      aclrtMallocHost(&impl_->status_host, state.status_bytes) != ACL_SUCCESS) {
    (void)Shutdown();
    return Internal("failed to allocate UMDK execution buffers");
  }
  peer::PeerRegion registered;
  auto found = impl_->config.peer_link->LookupRegion(region.base, &registered);
  if (!found.ok()) {
    (void)Shutdown();
    return found;
  }
  if (!impl_->config.token_bytes ||
      registered.length < region.length + GatherRemoteWorkspaceBytes(impl_->config.token_bytes)) {
    (void)Shutdown();
    return InvalidArgument("UMDK BatchGet requires reserved remote gather workspace");
  }
  impl_->remote_bytes = registered.length;
  impl_->region = region;
  impl_->initialized = true;
  return Status::OK();
}
Status UmdkAccessOperator::Shutdown() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  auto drained = impl_->Drain();
  if (!drained.ok()) return drained;
  auto closed = impl_->session.Shutdown();
  if (!closed.ok()) return closed;
  if (impl_->workspace) (void)aclrtFree(impl_->workspace);
  if (impl_->status_host) (void)aclrtFreeHost(impl_->status_host);
  if (impl_->execution.status) (void)aclrtFree(impl_->execution.status);
  if (impl_->execution.queue_state) (void)aclrtFree(impl_->execution.queue_state);
  impl_->workspace = impl_->status_host = nullptr;
  impl_->execution = {};
  impl_->sequence = 0;
  impl_->initialized = impl_->cpu_enabled = false;
  return Status::OK();
}
Status UmdkAccessOperator::ExecuteBatchPut(const BatchPutExecRequest& exec) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  auto drained = impl_->Drain();
  if (!drained.ok()) return drained;
  if (!impl_->initialized || !exec.request || !exec.put_desc || !exec.execution.stream)
    return InvalidArgument(
        "UMDK BatchPut requires initialized operator, input, destination and stream");
  const auto& request = *exec.request;
  if (!request.input.valid() || request.input.device != DeviceKind::kNpu || request.keys.empty() ||
      request.keys.size() != request.input_offsets.size() ||
      request.keys.size() != request.lengths.size() ||
      request.keys.size() != exec.put_desc->entries.size())
    return InvalidArgument("invalid BatchPut layout");
  auto ready = impl_->session.Ensure(
      *impl_->config.peer_link, impl_->region.base, impl_->config.device, impl_->config.eid_index,
      const_cast<void*>(request.input.data), request.input.bytes, impl_->config.workers);
  if (!ready.ok()) return ready;
  std::vector<UmdkTransfer> transfers;
  transfers.reserve(request.keys.size());
  for (std::size_t index = 0; index < request.keys.size(); ++index) {
    const auto offset = request.input_offsets[index], length = request.lengths[index];
    const auto& destination = exec.put_desc->entries[index];
    if (!length || offset > request.input.bytes || length > request.input.bytes - offset ||
        !destination.address || destination.reserved_length != length)
      return InvalidArgument("invalid BatchPut span");
    UmdkTransfer transfer{destination.address,
                          reinterpret_cast<std::uint64_t>(request.input.data) + offset, length};
    if (!transfers.empty() && transfers.back().local + transfers.back().bytes == transfer.local &&
        transfers.back().remote + transfers.back().bytes == transfer.remote)
      transfers.back().bytes += length;
    else
      transfers.push_back(transfer);
  }
  return impl_->session.Write(transfers, static_cast<aclrtStream>(exec.execution.stream));
}
Status UmdkAccessOperator::ExecuteBatchGet(const BatchGetExecRequest& exec) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  auto drained = impl_->Drain();
  if (!drained.ok()) return drained;
  if (!impl_->initialized || !exec.execution.stream)
    return InvalidArgument("UMDK BatchGet needs initialization and stream");
  UmdkBatchGetArgs launch{};
  auto valid = BuildBatchGetKernelArgs(exec, 1U, 1U, impl_->execution, &launch.request,
                                       sizeof(GatherReadStatus), sizeof(GatherQueueState));
  if (!valid.ok()) return valid;
  if (launch.request.request_capacity > kGatherMaxRequests ||
      launch.request.token_bytes != impl_->config.token_bytes)
    return InvalidArgument("UMDK BatchGet exceeds 65536 requests or configured token size");
  auto ensure = [&](void* buffer, std::size_t bytes) {
    return impl_->session.Ensure(*impl_->config.peer_link, impl_->region.base, impl_->config.device,
                                 impl_->config.eid_index, buffer, bytes, impl_->config.workers);
  };
  auto ready = ensure(impl_->workspace, kGatherWorkspaceBytes);
  if (!ready.ok()) return ready;
  ready = ensure(exec.request->output.data, exec.request->output.bytes);
  if (!ready.ok()) return ready;
  if (!impl_->cpu_enabled) {
    ready = impl_->config.peer_link->EnableDirectGather(impl_->region.base);
    if (!ready.ok()) return ready;
    impl_->cpu_enabled = true;
  }
  if (impl_->sequence == peer::kGatherErrorBit - 1U)
    return Status::InvalidState("BatchGet sequence exhausted");
  auto& args = launch.gather;
  args.sequence = ++impl_->sequence;
  args.umdk_session = impl_->session.session();
  args.umdk_owner_pid = impl_->session.owner_pid();
  args.queue_state = PointerValue(impl_->execution.queue_state);
  args.core_status = PointerValue(impl_->execution.status);
  args.scratch = PointerValue(impl_->workspace);
  args.source_descriptor = args.scratch + kGatherDescriptorOffset;
  args.output_spans = args.scratch + kGatherTargetsOffset;
  args.output = PointerValue(exec.request->output.data);
  args.output_extent = exec.request->output.bytes;
  args.remote_staging = (impl_->region.base + impl_->region.length + 63U) & ~std::uint64_t{63U};
  args.remote_descriptor = args.remote_staging + GatherStagingBytes(impl_->config.token_bytes);
  args.remote_notify = impl_->region.base + peer::GatherControlOffset(impl_->remote_bytes);
  args.remote_ack = args.remote_notify + 8U;
  args.remote_command = args.remote_notify + 16U;
  args.worker_count = impl_->config.workers;
  const auto stream = static_cast<aclrtStream>(exec.execution.stream);
  impl_->pending = stream;
  if (LaunchUmdkGet(&launch, stream)) return Internal("UMDK BatchGet launch failed");
  if (!exec.execution.synchronize) return Status::OK();
  if (aclrtMemcpyAsync(impl_->status_host, impl_->execution.status_bytes, impl_->execution.status,
                       impl_->execution.status_bytes, ACL_MEMCPY_DEVICE_TO_HOST,
                       stream) != ACL_SUCCESS)
    return Internal("BatchGet status enqueue failed");
  auto completed = impl_->Drain();
  if (!completed.ok()) return completed;
  const auto* statuses = static_cast<const GatherReadStatus*>(impl_->status_host);
  for (unsigned worker = 0; worker < args.worker_count; ++worker)
    if (statuses[worker].magic != kGatherStatusMagic ||
        statuses[worker].sequence != args.sequence || statuses[worker].result)
      return Internal("BatchGet worker " + std::to_string(worker) +
                      " failed: phase=" + std::to_string(statuses[worker].phase) +
                      " result=" + std::to_string(statuses[worker].result));
  return Status::OK();
}
}  // namespace kvaf
