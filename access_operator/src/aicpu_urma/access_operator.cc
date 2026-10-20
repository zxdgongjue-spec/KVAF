#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <mutex>
#include <utility>

#include "aicpu_gather_access_operator.h"
#include "aicpu_urma/umdk_session.h"
#include "cpu_gather.h"
#include "host_plan.h"
#include "meta_manager.h"
namespace kvaf {
namespace {
using namespace aicpu_urma;
Status InvalidArgument(std::string s) { return Status::InvalidArgument(std::move(s)); }
Status Internal(std::string s) { return Status::Internal(std::move(s)); }
std::uint64_t Elapsed(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                                              start)
      .count();
}
struct BatchGetExecutionState {
  void* status{nullptr};
  std::size_t status_bytes{0};
  void* queue_state{nullptr};
  std::size_t queue_state_bytes{0};
};
std::uint64_t PointerValue(const void* pointer) {
  return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pointer));
}

bool FitsU32(std::uint64_t value) { return value <= std::numeric_limits<std::uint32_t>::max(); }

Status BuildBatchGetKernelArgs(const BatchGetExecRequest& exec, std::uint32_t npu_core_count,
                               std::uint32_t jettys_per_core,
                               const BatchGetExecutionState& execution,
                               KvafExecuteBatchGetKernelArgs* arguments) {
  if (exec.request == nullptr || arguments == nullptr) {
    return InvalidArgument("BatchGet request and arguments must not be null");
  }
  const BatchGetRequest& request = *exec.request;
  if (!request.indices.has_elements<std::int32_t>() ||
      !request.offsets.has_elements<std::int32_t>() ||
      !request.output_indices.has_elements<std::int32_t>() || !request.output.valid() ||
      request.output_block_bytes == 0 || request.tokens_per_block == 0 ||
      request.output.bytes % request.output_block_bytes != 0 ||
      request.indices.device != request.offsets.device ||
      request.indices.device != request.output_indices.device ||
      request.indices.device != request.output.device ||
      request.indices.device != DeviceKind::kNpu ||
      request.output_indices.device != DeviceKind::kNpu || request.mask.data == nullptr ||
      request.mask.device != DeviceKind::kNpu || request.output.device != DeviceKind::kNpu ||
      request.device_access_table.addresses.device != DeviceKind::kNpu ||
      request.device_access_table.valid.device != DeviceKind::kNpu ||
      request.device_access_table.addresses.bytes == 0 ||
      request.device_access_table.addresses.bytes % sizeof(Address) != 0 ||
      request.device_access_table.valid.bytes !=
          request.device_access_table.addresses.bytes / sizeof(Address) ||
      request.indices.bytes % sizeof(std::int32_t) != 0 ||
      request.offsets.bytes != request.indices.bytes ||
      request.output_indices.bytes != request.indices.bytes ||
      request.mask.bytes != request.indices.bytes / sizeof(std::int32_t) ||
      execution.status == nullptr ||
      execution.status_bytes <
          static_cast<std::size_t>(npu_core_count) * sizeof(KvafBatchGetCoreStatus) ||
      execution.queue_state == nullptr ||
      execution.queue_state_bytes < static_cast<std::size_t>(npu_core_count) * jettys_per_core *
                                        sizeof(KvafBatchGetQueueState)) {
    return InvalidArgument("BatchGet request does not satisfy the NPU buffer contract");
  }

  const std::size_t request_count = request.indices.bytes / sizeof(std::int32_t);
  const std::size_t output_rows = request.output.bytes / request.output_block_bytes;
  const std::size_t device_table_entries =
      request.device_access_table.addresses.bytes / sizeof(Address);
  if (!FitsU32(device_table_entries) || !FitsU32(request_count) || !FitsU32(output_rows) ||
      !FitsU32(request.output_block_bytes) || !FitsU32(request.tokens_per_block)) {
    return InvalidArgument("BatchGet dimensions exceed the device ABI");
  }
  *arguments = KvafExecuteBatchGetKernelArgs{};
  arguments->address_table = PointerValue(request.device_access_table.addresses.data);
  arguments->valid_table = PointerValue(request.device_access_table.valid.data);
  arguments->indices = PointerValue(request.indices.data);
  arguments->offsets = PointerValue(request.offsets.data);
  arguments->mask = PointerValue(request.mask.data);
  arguments->output_indices = PointerValue(request.output_indices.data);
  arguments->output = PointerValue(request.output.data);
  arguments->core_status = PointerValue(execution.status);
  arguments->queue_state = PointerValue(execution.queue_state);
  arguments->table_capacity = static_cast<std::uint32_t>(device_table_entries);
  arguments->request_capacity = static_cast<std::uint32_t>(request_count);
  arguments->output_rows = static_cast<std::uint32_t>(output_rows);
  arguments->token_bytes = static_cast<std::uint32_t>(request.output_block_bytes);
  arguments->tokens_per_block = static_cast<std::uint32_t>(request.tokens_per_block);
  arguments->block_count = npu_core_count;
  arguments->jettys_per_core = jettys_per_core;
  return Status::OK();
}

}  // namespace
struct UmdkAccessOperator::Impl {
  explicit Impl(UmdkAccessOperatorConfig value) : config(std::move(value)) {}
  UmdkAccessOperatorConfig config;
  UmdkSession session;
  RemoteRegionDesc region;
  BatchGetExecutionState execution;
  void* metadata{nullptr};
  std::size_t metadata_capacity{0};
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
  state.status_bytes =
      impl_->config.workers * std::max(sizeof(GatherReadStatus), sizeof(KvafBatchGetCoreStatus));
  state.queue_state_bytes = impl_->config.workers * sizeof(GatherQueueState);
  if (aclrtMalloc(&state.status, state.status_bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
      aclrtMalloc(&state.queue_state, state.queue_state_bytes, ACL_MEM_MALLOC_HUGE_FIRST) !=
          ACL_SUCCESS ||
      aclrtMallocHost(&impl_->status_host, state.status_bytes) != ACL_SUCCESS) {
    (void)Shutdown();
    return Internal("failed to allocate UMDK execution buffers");
  }
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
  if (impl_->metadata) (void)aclrtFreeHost(impl_->metadata);
  if (impl_->status_host) (void)aclrtFreeHost(impl_->status_host);
  if (impl_->execution.status) (void)aclrtFree(impl_->execution.status);
  if (impl_->execution.queue_state) (void)aclrtFree(impl_->execution.queue_state);
  impl_->metadata = impl_->status_host = nullptr;
  impl_->metadata_capacity = 0;
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
      const_cast<void*>(request.input.data), request.input.bytes, 1U, 1U, impl_->config.workers);
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
  KvafExecuteBatchGetKernelArgs args{};
  auto valid = BuildBatchGetKernelArgs(exec, 1U, 1U, impl_->execution, &args);
  if (!valid.ok()) return valid;
  auto ready = impl_->session.Ensure(
      *impl_->config.peer_link, impl_->region.base, impl_->config.device, impl_->config.eid_index,
      exec.request->output.data, exec.request->output.bytes, 1U, 1U, impl_->config.workers);
  if (!ready.ok()) return ready;
  const UmdkBatchGetArgs launch{args, impl_->session.session(), impl_->session.owner_pid()};
  const auto stream = static_cast<aclrtStream>(exec.execution.stream);
  if (LaunchUmdkGet(&launch, stream)) return Internal("UMDK BatchGet launch failed");
  if (exec.execution.synchronize) {
    KvafBatchGetCoreStatus completed{};
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS ||
        aclrtMemcpy(&completed, sizeof(completed), impl_->execution.status, sizeof(completed),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
        completed.magic != KVAF_AIV_BATCH_GET_STATUS_MAGIC || completed.result)
      return Internal("UMDK BatchGet completion failed");
  }
  return Status::OK();
}
Status UmdkAccessOperator::ExecuteGatherRead(const CpuGatherReadRequest& request) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  auto drained = impl_->Drain();
  if (!drained.ok()) return drained;
  if (request.timing) *request.timing = {};
  auto* timing = request.timing;
  auto start = std::chrono::steady_clock::now();
  const auto dimensions = ComputeGatherDimensions(request);
  const auto total = dimensions.bytes;
  const auto extent = dimensions.extent;
  const GatherOutputSpan contiguous{0U, static_cast<std::uint32_t>(total), 0U};
  const auto* targets = request.output_spans.empty() ? &contiguous : request.output_spans.data();
  const auto target_count = request.output_spans.empty() ? 1U : request.output_spans.size();
  if (timing) timing->layout_compute_ns = Elapsed(start);
  start = std::chrono::steady_clock::now();
  if (!impl_->cpu_enabled) {
    auto enabled = impl_->config.peer_link->EnableDirectGather(impl_->region.base);
    if (!enabled.ok()) return enabled;
    impl_->cpu_enabled = true;
  }
  if (impl_->sequence == peer::kGatherErrorBit - 1U)
    return Status::InvalidState("GatherRead sequence exhausted");
  const auto sequence = ++impl_->sequence;
  if (timing) timing->configure_ns = Elapsed(start);
  start = std::chrono::steady_clock::now();
  const auto source_bytes = peer::GatherDescriptorBytes(request.spans.size());
  const auto target_offset = GatherPackedTargetOffset(source_bytes);
  const auto metadata_bytes = target_offset + target_count * sizeof(GatherOutputSpan);
  const auto scratch_offset = (extent + 63U) & ~std::uint64_t{63U};
  const auto source_offset = scratch_offset + GatherScratchBytes(request.pipeline_chunk_bytes);
  if (metadata_bytes > impl_->metadata_capacity) {
    void* allocation = nullptr;
    if (aclrtMallocHost(&allocation, metadata_bytes) != ACL_SUCCESS)
      return Internal("failed to allocate GatherRead metadata");
    if (impl_->metadata) (void)aclrtFreeHost(impl_->metadata);
    impl_->metadata = allocation;
    impl_->metadata_capacity = metadata_bytes;
  }
  auto* metadata = static_cast<std::uint8_t*>(impl_->metadata);
  const peer::GatherDescriptorHeader descriptor{
      peer::kGatherDescriptorMagic,      sequence,
      request.staging_address,           static_cast<std::uint32_t>(request.spans.size()),
      static_cast<std::uint32_t>(total), request.pipeline_chunk_bytes};
  std::memcpy(metadata, &descriptor, sizeof(descriptor));
  std::memcpy(metadata + sizeof(descriptor), request.spans.data(),
              request.spans.size() * sizeof(peer::GatherSpan));
  std::memset(metadata + source_bytes, 0, target_offset - source_bytes);
  std::memcpy(metadata + target_offset, targets, target_count * sizeof(GatherOutputSpan));
  if (timing) timing->metadata_pack_ns = Elapsed(start);
  start = std::chrono::steady_clock::now();
  auto ready = impl_->session.Ensure(
      *impl_->config.peer_link, impl_->region.base, impl_->config.device, impl_->config.eid_index,
      request.output.data, request.output.bytes, total, target_count, impl_->config.workers);
  if (!ready.ok()) return ready;
  if (timing) {
    timing->register_ns = Elapsed(start);
  }
  GatherReadArgs args{};
  args.umdk_session = impl_->session.session();
  args.umdk_owner_pid = impl_->session.owner_pid();
  args.queue_state = PointerValue(impl_->execution.queue_state);
  args.core_status = PointerValue(impl_->execution.status);
  args.output = PointerValue(request.output.data);
  args.output_extent = extent;
  args.source_descriptor = args.output + source_offset;
  args.output_spans = args.source_descriptor + target_offset;
  args.remote_staging = request.staging_address;
  args.remote_descriptor = request.source_descriptor_address;
  args.remote_notify = impl_->region.base + peer::GatherControlOffset(impl_->region.length);
  args.remote_ack = args.remote_notify + 8U;
  args.remote_command = args.remote_notify + 16U;
  args.sequence = sequence;
  args.bytes = total;
  args.source_span_count = request.spans.size();
  args.output_span_count = target_count;
  args.worker_count = impl_->config.workers;
  args.sgl_size = request.sgl_size;
  args.sq_batch_wqes = request.sq_batch_wqes;
  args.pipeline_chunk_bytes = request.pipeline_chunk_bytes;
  args.profile_detail = request.profile_detail;
  const auto stream = static_cast<aclrtStream>(request.stream);
  impl_->pending = stream;
  start = std::chrono::steady_clock::now();
  if (aclrtMemcpyAsync(reinterpret_cast<void*>(args.source_descriptor), metadata_bytes, metadata,
                       metadata_bytes, ACL_MEMCPY_HOST_TO_DEVICE, stream) != ACL_SUCCESS)
    return Internal("GatherRead metadata enqueue failed");
  if (timing) {
    timing->metadata_enqueue_ns = Elapsed(start);
    timing->metadata_upload_async = true;
  }
  start = std::chrono::steady_clock::now();
  if (LaunchGatherRead(&args, stream)) return Internal("GatherRead launch failed");
  if (timing) timing->launch_ns = Elapsed(start);
  start = std::chrono::steady_clock::now();
  if (aclrtMemcpyAsync(impl_->status_host, impl_->execution.status_bytes, impl_->execution.status,
                       impl_->execution.status_bytes, ACL_MEMCPY_DEVICE_TO_HOST,
                       stream) != ACL_SUCCESS)
    return Internal("GatherRead status enqueue failed");
  if (timing) {
    timing->status_enqueue_ns = Elapsed(start);
    timing->status_readback_async = true;
  }
  start = std::chrono::steady_clock::now();
  auto completed = impl_->Drain();
  if (timing) timing->sync_ns = Elapsed(start);
  if (!completed.ok()) return completed;
  start = std::chrono::steady_clock::now();
  const auto* statuses = static_cast<const GatherReadStatus*>(impl_->status_host);
  std::uint64_t begin = UINT64_MAX, end = 0U, latest_begin = 0U;
  for (unsigned worker = 0; worker < args.worker_count; ++worker) {
    const auto& status = statuses[worker];
    if (status.magic != kGatherStatusMagic || status.sequence != sequence || status.result ||
        !status.scatter_begin_ns || status.scatter_end_ns < status.scatter_begin_ns)
      return Internal("GatherRead worker " + std::to_string(worker) + " failed: phase=" +
                      std::to_string(status.phase) + " result=" + std::to_string(status.result));
    begin = std::min(begin, status.scatter_begin_ns);
    end = std::max(end, status.scatter_end_ns);
    latest_begin = std::max(latest_begin, status.scatter_begin_ns);
    if (timing) {
      timing->scatter_plan_ns = std::max(timing->scatter_plan_ns, status.scatter_plan_ns);
      timing->scatter_post_ns = std::max(timing->scatter_post_ns, status.scatter_post_ns);
      timing->scatter_cq_ns = std::max(timing->scatter_cq_ns, status.scatter_cq_ns);
      timing->ready_wait_ns = std::max(timing->ready_wait_ns, status.ready_wait_ns);
      timing->scatter_worker_ns =
          std::max(timing->scatter_worker_ns, status.scatter_end_ns - status.scatter_begin_ns);
      timing->read_wqes += status.read_wqes;
      timing->provider_post_calls += status.provider_post_calls;
      timing->read_tasks += status.read_tasks;
      timing->workers[worker] = {status.scatter_begin_ns,
                                 status.scatter_end_ns,
                                 status.scatter_end_ns - status.scatter_begin_ns,
                                 status.read_wqes,
                                 status.provider_post_calls,
                                 status.read_tasks,
                                 status.scatter_plan_ns,
                                 status.scatter_post_ns,
                                 status.scatter_cq_ns,
                                 status.thread_cpu_ns,
                                 status.thread_cpu_available,
                                 status.start_cpu,
                                 status.end_cpu,
                                 status.cq_poll_calls,
                                 status.read_bytes};
    }
  }
  if (statuses[0].ack_seen != sequence) return Internal("GatherRead CPU ACK missing");
  if (timing) {
    for (unsigned worker = 0; worker < args.worker_count; ++worker) {
      timing->workers[worker].start_offset_ns -= begin;
      timing->workers[worker].end_offset_ns -= begin;
    }
    const auto& status = statuses[0];
    timing->notify_ns = status.notify_ns;
    timing->source_write_ns = status.source_write_ns;
    timing->source_write_post_ns = status.source_write_post_ns;
    timing->source_write_cq_ns = status.source_write_cq_ns;
    timing->notify_post_ns = status.notify_post_ns;
    timing->notify_cq_ns = status.notify_cq_ns;
    timing->ack_poll_ns = status.ack_poll_ns;
    timing->ack_started_ns = status.ack_started_ns;
    timing->ack_post_ns = status.ack_post_ns;
    timing->ack_cq_ns = status.ack_cq_ns;
    timing->ack_cache_ns = status.ack_cache_ns;
    timing->ack_polls = status.ack_polls;
    timing->output_layout_ns = status.output_layout_ns;
    timing->cpu_source_cache_ns = status.cpu.source_cache_ns;
    timing->cpu_copy_ns = status.cpu.copy_ns;
    timing->cpu_dispatch_ns = status.cpu.dispatch_ns;
    timing->cpu_notify_spin_ns = status.cpu.notify_spin_ns;
    timing->cpu_staging_flush_ns = status.cpu.staging_flush_ns;
    timing->scatter_total_ns = end - begin;
    timing->scatter_start_skew_ns = latest_begin - begin;
    timing->aicpu_cores = args.worker_count;
    timing->profile_detail = request.profile_detail;
    timing->pipeline_chunk_bytes = request.pipeline_chunk_bytes;
    timing->status_decode_ns = Elapsed(start);
  }
  return Status::OK();
}
}  // namespace kvaf
