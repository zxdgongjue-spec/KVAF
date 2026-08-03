#include "kvaf/access_operator/access_operator.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include "acl/acl.h"
#include "acl/acl_mdl.h"
#include "batch_get_abi.h"
#include "kvaf_aiv_batch_get.h"
#include "kvaf/meta_manager/meta_manager.h"

namespace kvaf {
namespace {

Status InvalidArgument(std::string message) {
  return Status::InvalidArgument(std::move(message));
}

Status InvalidState(std::string message) {
  return Status::InvalidState(std::move(message));
}

Status Internal(std::string message) {
  return Status::Internal(std::move(message));
}

std::uint64_t PointerValue(const void* pointer) {
  return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pointer));
}

bool FitsU32(std::uint64_t value) {
  return value <= std::numeric_limits<std::uint32_t>::max();
}

Status BuildBatchGetKernelArgs(const BatchGetExecRequest& exec,
                               std::uint32_t npu_core_count,
                               KvafExecuteBatchGetKernelArgs* arguments) {
  if (exec.request == nullptr || arguments == nullptr) {
    return InvalidArgument("BatchGet request and arguments must not be null");
  }
  const BatchGetRequest& request = *exec.request;
  const BatchGetExecutionBuffers& execution = request.execution_buffers;
  if (!request.indices.has_elements<std::int32_t>() ||
      !request.output_indices.has_elements<std::int32_t>() ||
      !request.output.valid() || request.output_block_bytes == 0 ||
      request.output.bytes % request.output_block_bytes != 0 ||
      request.indices.device != request.output_indices.device ||
      request.indices.device != request.output.device ||
      request.indices.device != DeviceKind::kNpu ||
      request.output_indices.device != DeviceKind::kNpu ||
      request.mask.data == nullptr || request.mask.device != DeviceKind::kNpu ||
      request.output.device != DeviceKind::kNpu ||
      request.device_access_table.addresses.device != DeviceKind::kNpu ||
      request.device_access_table.valid.device != DeviceKind::kNpu ||
      request.device_access_table.addresses.bytes == 0 ||
      request.device_access_table.addresses.bytes % sizeof(Address) != 0 ||
      request.device_access_table.valid.bytes !=
          request.device_access_table.addresses.bytes / sizeof(Address) ||
      execution.status.data == nullptr || execution.status.device != DeviceKind::kNpu ||
      execution.queue_state.data == nullptr ||
          execution.queue_state.device != DeviceKind::kNpu ||
      request.output_block_bytes == 0 || exec.host_access_table.size == 0 ||
      request.indices.bytes % sizeof(std::int32_t) != 0 ||
      request.output_indices.bytes != request.indices.bytes ||
      request.mask.bytes != request.indices.bytes / sizeof(std::int32_t) ||
      execution.status.bytes < static_cast<std::size_t>(npu_core_count) *
                                   sizeof(KvafBatchGetCoreStatus) ||
      execution.queue_state.bytes < static_cast<std::size_t>(npu_core_count) *
                                        sizeof(KvafBatchGetQueueState)) {
    return InvalidArgument("BatchGet request does not satisfy the NPU buffer contract");
  }

  const std::size_t request_count = request.indices.bytes / sizeof(std::int32_t);
  const std::size_t output_rows = request.output.bytes / request.output_block_bytes;
  const std::size_t device_table_entries =
      request.device_access_table.addresses.bytes / sizeof(Address);
  if (!FitsU32(device_table_entries) || !FitsU32(request_count) ||
      !FitsU32(output_rows) || !FitsU32(request.output_block_bytes)) {
    return InvalidArgument("BatchGet dimensions exceed the device ABI");
  }
  for (std::size_t index = 0; index < exec.host_access_table.size; ++index) {
    if (exec.host_access_table.valid[index] != 0 &&
        exec.host_access_table.lengths[index] != request.output_block_bytes) {
      return InvalidArgument("NPU BatchGet supports fixed-size valid blocks only");
    }
  }

  *arguments = KvafExecuteBatchGetKernelArgs{};
  arguments->address_table = PointerValue(request.device_access_table.addresses.data);
  arguments->valid_table = PointerValue(request.device_access_table.valid.data);
  arguments->indices = PointerValue(request.indices.data);
  arguments->mask = PointerValue(request.mask.data);
  arguments->output_indices = PointerValue(request.output_indices.data);
  arguments->output = PointerValue(request.output.data);
  arguments->core_status = PointerValue(execution.status.data);
  arguments->queue_state = PointerValue(execution.queue_state.data);
  arguments->table_capacity = static_cast<std::uint32_t>(device_table_entries);
  arguments->request_capacity = static_cast<std::uint32_t>(request_count);
  arguments->output_rows = static_cast<std::uint32_t>(output_rows);
  arguments->block_bytes = static_cast<std::uint32_t>(request.output_block_bytes);
  arguments->block_count = npu_core_count;
  return Status::OK();
}

}  // namespace

struct CapturedBatchGetGraph::Impl {
  aclmdlRI graph_model{nullptr};
  aclrtStream stream{nullptr};
  KvafBatchGetLocalBuffer* local_buffer{nullptr};
  void* replay_epoch{nullptr};
  std::uint8_t next_epoch{0};
};

CapturedBatchGetGraph::~CapturedBatchGetGraph() {
  if (impl_ == nullptr) {
    return;
  }
  if (impl_->stream != nullptr) {
    (void)aclrtSynchronizeStream(impl_->stream);
  }
  if (impl_->graph_model != nullptr) {
    (void)aclmdlRIDestroy(impl_->graph_model);
  }
  if (impl_->replay_epoch != nullptr) {
    (void)aclrtFree(impl_->replay_epoch);
  }
  if (impl_->local_buffer != nullptr) {
    KvafBatchGetLocalBufferRelease(impl_->local_buffer);
  }
}

Status CapturedBatchGetGraph::Replay() {
  if (impl_ == nullptr || impl_->graph_model == nullptr ||
      impl_->stream == nullptr || impl_->replay_epoch == nullptr) {
    return Status::InvalidState("BatchGet graph has been released");
  }
  ++impl_->next_epoch;
  // The underlying demo has observed that a completely naked graph replay can
  // stall. Put a tiny, independent producer operation on the same private
  // stream before every replay; the caller's fixed tensors are not modified.
  if (aclrtMemcpyAsync(impl_->replay_epoch, sizeof(impl_->next_epoch),
                       &impl_->next_epoch, sizeof(impl_->next_epoch),
                       ACL_MEMCPY_HOST_TO_DEVICE, impl_->stream) != ACL_SUCCESS) {
    return Internal("failed to enqueue graph replay producer update");
  }
  if (aclmdlRIExecuteAsync(impl_->graph_model, impl_->stream) != ACL_SUCCESS) {
    return Internal("failed to enqueue captured BatchGet graph");
  }
  if (aclrtSynchronizeStream(impl_->stream) != ACL_SUCCESS) {
    return Internal("captured BatchGet graph stream synchronization failed");
  }
  return Status::OK();
}

struct DefaultAccessOperator::Impl {
  explicit Impl(AccessOperatorConfig init_config) : config(std::move(init_config)) {}

  AccessOperatorConfig config;
  RemoteRegionDesc remote_region;
  KvafBatchGetTransport* transport{nullptr};
  bool initialized{false};
};

DefaultAccessOperator::DefaultAccessOperator(AccessOperatorConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

DefaultAccessOperator::~DefaultAccessOperator() { (void)Shutdown(); }

Status DefaultAccessOperator::Initialize(const RemoteRegionDesc& remote_region) {
  if (impl_->initialized) {
    return InvalidState("AccessOperator has already been initialized");
  }
  if (remote_region.base == 0 || remote_region.length == 0 ||
      impl_->config.kernel_object_path.empty() || impl_->config.stream == nullptr ||
      impl_->config.npu_core_count == 0 ||
      impl_->config.npu_core_count > KVAF_AIV_BATCH_GET_MAX_AIV_CORES) {
    return InvalidArgument("invalid AccessOperator initialization config");
  }
  if (KvafExecuteBatchGetLoadKernel(impl_->config.kernel_object_path.c_str()) !=
      KVAF_BATCH_GET_SUCCESS) {
    return Internal("failed to load BatchGet kernel");
  }

  KvafBatchGetTransport* transport = nullptr;
  const int32_t result = KvafBatchGetTransportCreate(
      reinterpret_cast<void*>(remote_region.base), remote_region.length,
      impl_->config.npu_core_count, 1U, &transport);
  if (result != KVAF_BATCH_GET_SUCCESS || transport == nullptr) {
    return Internal("failed to create reusable BatchGet transport");
  }
  impl_->remote_region = remote_region;
  impl_->transport = transport;
  impl_->initialized = true;
  return Status::OK();
}

Status DefaultAccessOperator::Shutdown() {
  if (!impl_->initialized) {
    return Status::OK();
  }

  Status result = Status::OK();
  // BatchGet is synchronous. Synchronize once more before destroying cached
  // local HBM registrations so a caller cannot tear down an in-flight WQE.
  if (aclrtSynchronizeStream(static_cast<aclrtStream>(impl_->config.stream)) != ACL_SUCCESS) {
    result = Internal("BatchGet stream synchronization failed during shutdown");
  }
  if (impl_->transport != nullptr) {
    KvafBatchGetTransportDestroy(impl_->transport);
  }
  impl_->transport = nullptr;
  impl_->remote_region = RemoteRegionDesc{};
  impl_->initialized = false;
  return result;
}

Status DefaultAccessOperator::ExecuteBatchGet(const BatchGetExecRequest& exec) {
  if (!impl_->initialized || impl_->transport == nullptr) {
    return Status::Error(StatusCode::kNotInitialized,
                         "AccessOperator has not been initialized");
  }
  KvafExecuteBatchGetKernelArgs arguments{};
  Status status = BuildBatchGetKernelArgs(exec, impl_->config.npu_core_count,
                                          &arguments);
  if (!status.ok()) {
    return status;
  }

  KvafBatchGetLocalBuffer* local_buffer = nullptr;
  std::uint64_t channel_entity = 0;
  if (KvafBatchGetLocalBufferAcquire(impl_->transport,
                                     reinterpret_cast<void*>(arguments.output),
                                     exec.request->output.bytes, &local_buffer,
                                     &channel_entity) != KVAF_BATCH_GET_SUCCESS ||
      local_buffer == nullptr || channel_entity == 0) {
    return Internal("failed to acquire caller output for BatchGet");
  }
  arguments.channel_entity = channel_entity;

  const int32_t launch_result = KvafExecuteBatchGetLaunch(
      &arguments, static_cast<aclrtStream>(impl_->config.stream));
  if (launch_result != KVAF_BATCH_GET_SUCCESS) {
    KvafBatchGetLocalBufferRelease(local_buffer);
    return Internal("failed to launch BatchGet");
  }
  const aclError sync_result =
      aclrtSynchronizeStream(static_cast<aclrtStream>(impl_->config.stream));
  KvafBatchGetLocalBufferRelease(local_buffer);
  if (sync_result != ACL_SUCCESS) {
    return Internal("BatchGet stream synchronization failed");
  }
  return Status::OK();
}

Status DefaultAccessOperator::CaptureBatchGetGraph(
    const BatchGetExecRequest& exec,
    std::unique_ptr<CapturedBatchGetGraph>* graph) {
  if (!impl_->initialized || impl_->transport == nullptr) {
    return Status::NotInitialized("AccessOperator has not been initialized");
  }
  if (graph == nullptr) {
    return InvalidArgument("BatchGet graph result must not be null");
  }
  if (*graph != nullptr) {
    return InvalidState("BatchGet graph result must be empty");
  }

  KvafExecuteBatchGetKernelArgs arguments{};
  Status status = BuildBatchGetKernelArgs(exec, impl_->config.npu_core_count,
                                          &arguments);
  if (!status.ok()) {
    return status;
  }

  KvafBatchGetLocalBuffer* local_buffer = nullptr;
  std::uint64_t channel_entity = 0;
  if (KvafBatchGetLocalBufferAcquire(impl_->transport,
                                     reinterpret_cast<void*>(arguments.output),
                                     exec.request->output.bytes, &local_buffer,
                                     &channel_entity) != KVAF_BATCH_GET_SUCCESS ||
      local_buffer == nullptr || channel_entity == 0) {
    return Internal("failed to acquire caller output for BatchGet graph");
  }
  arguments.channel_entity = channel_entity;

  auto captured = std::unique_ptr<CapturedBatchGetGraph>(
      new CapturedBatchGetGraph());
  captured->impl_ = std::make_unique<CapturedBatchGetGraph::Impl>();
  captured->impl_->stream = static_cast<aclrtStream>(impl_->config.stream);
  captured->impl_->local_buffer = local_buffer;
  if (aclrtMalloc(&captured->impl_->replay_epoch, 1U,
                  ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS ||
      captured->impl_->replay_epoch == nullptr) {
    return Internal("failed to allocate graph replay producer buffer");
  }

  if (aclmdlRICaptureBegin(captured->impl_->stream,
                           ACL_MODEL_RI_CAPTURE_MODE_RELAXED) != ACL_SUCCESS) {
    return Internal("failed to begin BatchGet graph capture");
  }
  if (KvafExecuteBatchGetLaunch(&arguments, captured->impl_->stream) !=
      KVAF_BATCH_GET_SUCCESS) {
    (void)aclmdlRICaptureEnd(captured->impl_->stream,
                             &captured->impl_->graph_model);
    return Internal("failed to capture BatchGet launch");
  }
  if (aclmdlRICaptureEnd(captured->impl_->stream,
                          &captured->impl_->graph_model) != ACL_SUCCESS ||
      captured->impl_->graph_model == nullptr) {
    return Internal("failed to finish BatchGet graph capture");
  }
  *graph = std::move(captured);
  return Status::OK();
}

Status DefaultAccessOperator::ExecuteBatchPut(const BatchPutExecRequest& exec) {
  if (!impl_->initialized) {
    return Status::NotInitialized("AccessOperator has not been initialized");
  }
  if (exec.request == nullptr || exec.put_desc == nullptr ||
      exec.request->keys.empty() ||
      exec.request->keys.size() != exec.request->blocks.size() ||
      exec.request->keys.size() != exec.put_desc->entries.size()) {
    return InvalidArgument("BatchPut request does not satisfy the NPU mock contract");
  }

  for (std::size_t index = 0; index < exec.request->blocks.size(); ++index) {
    const ConstBufferView& source = exec.request->blocks[index];
    const PutAllocation& destination = exec.put_desc->entries[index];
    if (source.data == nullptr || source.bytes == 0 ||
        source.device != DeviceKind::kNpu || destination.address == 0 ||
        destination.reserved_length != source.bytes) {
      return InvalidArgument("BatchPut source and destination blocks do not match");
    }
    const aclError result = aclrtMemcpy(
        reinterpret_cast<void*>(destination.address), destination.reserved_length,
        source.data, source.bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (result != ACL_SUCCESS) {
      return Internal("ACL device-to-host mock BatchPut copy failed");
    }
  }
  return Status::OK();
}

}  // namespace kvaf
