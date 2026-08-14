#include "kvaf/access_operator/access_operator.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "acl/acl.h"
#include "batch_get_abi.h"
#include "kvaf_aiv_batch_get.h"
#include "kvaf/meta_manager/meta_manager.h"

#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
#include "batch_put_abi.h"
#include "kvaf_aiv_batch_put.h"
#endif

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

struct BatchGetExecutionState {
  void* status{nullptr};
  std::size_t status_bytes{0};
  void* queue_state{nullptr};
  std::size_t queue_state_bytes{0};
};

#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
// BatchPut request metadata is transient from the caller's perspective, but
// the device staging allocations are owned by the Operator and reused across
// synchronous eager calls. This keeps Runtime's packed variable-length
// interface while avoiding four ACL allocations for every BatchPut.
struct BatchPutExecutionState {
  void* metadata{nullptr};
  void* core_status{nullptr};
  std::size_t metadata_capacity{0};
  std::vector<std::uint64_t> host_metadata;
  std::vector<KvafBatchPutCoreStatus> host_core_status;
};

void ReleaseBatchPutExecutionState(BatchPutExecutionState* execution) {
  if (execution == nullptr) {
    return;
  }
  if (execution->core_status != nullptr) {
    (void)aclrtFree(execution->core_status);
  }
  if (execution->metadata != nullptr) {
    (void)aclrtFree(execution->metadata);
  }
  *execution = BatchPutExecutionState{};
}

Status InitializeBatchPutExecutionState(std::uint32_t npu_core_count,
                                        BatchPutExecutionState* execution) {
  if (execution == nullptr || npu_core_count == 0) {
    return InvalidArgument("invalid BatchPut execution state");
  }
  BatchPutExecutionState state;
  state.host_core_status.resize(npu_core_count);
  const std::size_t status_bytes =
      static_cast<std::size_t>(npu_core_count) * sizeof(KvafBatchPutCoreStatus);
  if (aclrtMalloc(&state.core_status, status_bytes,
                  ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
      aclrtMemset(state.core_status, status_bytes, 0, status_bytes) != ACL_SUCCESS) {
    ReleaseBatchPutExecutionState(&state);
    return Internal("failed to allocate internal BatchPut execution state");
  }
  *execution = std::move(state);
  return Status::OK();
}

Status EnsureBatchPutMetadataCapacity(std::size_t rows,
                                      BatchPutExecutionState* execution) {
  if (execution == nullptr || rows == 0 ||
      rows > std::numeric_limits<std::size_t>::max() /
                 (3U * sizeof(std::uint64_t))) {
    return InvalidArgument("BatchPut metadata dimensions are invalid");
  }
  execution->host_metadata.resize(rows * 3U);
  if (rows <= execution->metadata_capacity) {
    return Status::OK();
  }

  const std::size_t bytes = rows * 3U * sizeof(std::uint64_t);
  void* metadata = nullptr;
  if (aclrtMalloc(&metadata, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
    return Internal("failed to grow BatchPut device metadata buffers");
  }

  if (execution->metadata != nullptr) {
    (void)aclrtFree(execution->metadata);
  }
  execution->metadata = metadata;
  execution->metadata_capacity = rows;
  return Status::OK();
}

void LogBatchPutDeviceStatus(const BatchPutExecutionState& execution,
                             std::uint32_t npu_core_count,
                             int32_t result) {
  std::fprintf(stderr, "[KVAF_AIV] BATCH_PUT_EXECUTION_ERROR result=%d\n",
               static_cast<int>(result));
  for (std::uint32_t core = 0; core < npu_core_count; ++core) {
    const KvafBatchPutCoreStatus& status = execution.host_core_status[core];
    std::fprintf(stderr,
                 "[KVAF_AIV] BATCH_PUT_DEVICE_ERROR core=%u magic=0x%llx "
                 "result=%llu processed=%llu failed_request=%llu cq_status=%llu "
                 "cq_sub_status=%llu\n",
                 core, static_cast<unsigned long long>(status.magic),
                 static_cast<unsigned long long>(status.result),
                 static_cast<unsigned long long>(status.processed_request_count),
                 static_cast<unsigned long long>(status.failed_request_index),
                 static_cast<unsigned long long>(status.cq_status),
                 static_cast<unsigned long long>(status.cq_sub_status));
  }
}
#endif

std::uint64_t PointerValue(const void* pointer) {
  return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pointer));
}

bool FitsU32(std::uint64_t value) {
  return value <= std::numeric_limits<std::uint32_t>::max();
}

Status BuildBatchGetKernelArgs(const BatchGetExecRequest& exec,
                               std::uint32_t npu_core_count,
                               std::uint32_t jettys_per_core,
                               const BatchGetExecutionState& execution,
                               KvafExecuteBatchGetKernelArgs* arguments) {
  if (exec.request == nullptr || arguments == nullptr) {
    return InvalidArgument("BatchGet request and arguments must not be null");
  }
  const BatchGetRequest& request = *exec.request;
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
      request.output_block_bytes == 0 || exec.host_access_table.size == 0 ||
      request.indices.bytes % sizeof(std::int32_t) != 0 ||
      request.output_indices.bytes != request.indices.bytes ||
      request.mask.bytes != request.indices.bytes / sizeof(std::int32_t) ||
      execution.status == nullptr ||
      execution.status_bytes < static_cast<std::size_t>(npu_core_count) *
                                   sizeof(KvafBatchGetCoreStatus) ||
      execution.queue_state == nullptr ||
      execution.queue_state_bytes <
          static_cast<std::size_t>(npu_core_count) * jettys_per_core *
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
  *arguments = KvafExecuteBatchGetKernelArgs{};
  arguments->address_table = PointerValue(request.device_access_table.addresses.data);
  arguments->valid_table = PointerValue(request.device_access_table.valid.data);
  arguments->indices = PointerValue(request.indices.data);
  arguments->mask = PointerValue(request.mask.data);
  arguments->output_indices = PointerValue(request.output_indices.data);
  arguments->output = PointerValue(request.output.data);
  arguments->core_status = PointerValue(execution.status);
  arguments->queue_state = PointerValue(execution.queue_state);
  arguments->table_capacity = static_cast<std::uint32_t>(device_table_entries);
  arguments->request_capacity = static_cast<std::uint32_t>(request_count);
  arguments->output_rows = static_cast<std::uint32_t>(output_rows);
  arguments->block_bytes = static_cast<std::uint32_t>(request.output_block_bytes);
  arguments->block_count = npu_core_count;
  arguments->jettys_per_core = jettys_per_core;
  return Status::OK();
}

// Build the same Direct-HCCP launch ABI without involving a legacy
// AccessTable/KeyId.  The DSA row directory is already a device-resident
// address table and `flat_indices` are its deterministic row-major indices.
Status BuildDsaDirectBatchGetKernelArgs(
    const DsaDirectBatchGetRequest& request, std::uint32_t npu_core_count,
    std::uint32_t jettys_per_core, const BatchGetExecutionState& execution,
    KvafExecuteBatchGetKernelArgs* arguments) {
  if (arguments == nullptr || !request.flat_indices.has_elements<std::int32_t>() ||
      !request.token_offsets.has_elements<std::int32_t>() ||
      !request.output_indices.has_elements<std::int32_t>() ||
      request.mask.data == nullptr || !request.output.valid() ||
      request.output_block_bytes == 0 ||
      request.max_token_offset == 0 ||
      request.output.bytes % request.output_block_bytes != 0 ||
      request.flat_indices.device != DeviceKind::kNpu ||
      request.token_offsets.device != DeviceKind::kNpu ||
      request.mask.device != DeviceKind::kNpu ||
      request.output_indices.device != DeviceKind::kNpu ||
      request.output.device != DeviceKind::kNpu ||
      request.address_table.device != DeviceKind::kNpu ||
      request.valid_table.device != DeviceKind::kNpu ||
      request.address_table.data == nullptr || request.valid_table.data == nullptr ||
      request.address_table.bytes == 0 ||
      request.address_table.bytes % sizeof(Address) != 0 ||
      request.valid_table.bytes != request.address_table.bytes / sizeof(Address) ||
      request.flat_indices.bytes % sizeof(std::int32_t) != 0 ||
      request.token_offsets.bytes != request.flat_indices.bytes ||
      request.output_indices.bytes != request.flat_indices.bytes ||
      request.mask.bytes != request.flat_indices.bytes / sizeof(std::int32_t) ||
      execution.status == nullptr ||
      execution.status_bytes < static_cast<std::size_t>(npu_core_count) *
                                   sizeof(KvafBatchGetCoreStatus) ||
      execution.queue_state == nullptr ||
      execution.queue_state_bytes <
          static_cast<std::size_t>(npu_core_count) * jettys_per_core *
              sizeof(KvafBatchGetQueueState)) {
    return InvalidArgument(
        "DSA_ROW DirectGet request does not satisfy the NPU buffer contract");
  }

  const std::size_t request_count =
      request.flat_indices.bytes / sizeof(std::int32_t);
  const std::size_t output_rows =
      request.output.bytes / request.output_block_bytes;
  const std::size_t directory_cells =
      request.address_table.bytes / sizeof(Address);
  if (!FitsU32(directory_cells) || !FitsU32(request_count) ||
      !FitsU32(output_rows) || !FitsU32(request.output_block_bytes)) {
    return InvalidArgument("DSA_ROW DirectGet dimensions exceed the device ABI");
  }

  *arguments = KvafExecuteBatchGetKernelArgs{};
  arguments->address_table = PointerValue(request.address_table.data);
  arguments->valid_table = PointerValue(request.valid_table.data);
  arguments->indices = PointerValue(request.flat_indices.data);
  arguments->token_offsets = PointerValue(request.token_offsets.data);
  arguments->mask = PointerValue(request.mask.data);
  arguments->output_indices = PointerValue(request.output_indices.data);
  arguments->output = PointerValue(request.output.data);
  arguments->core_status = PointerValue(execution.status);
  arguments->queue_state = PointerValue(execution.queue_state);
  arguments->table_capacity = static_cast<std::uint32_t>(directory_cells);
  arguments->request_capacity = static_cast<std::uint32_t>(request_count);
  arguments->output_rows = static_cast<std::uint32_t>(output_rows);
  arguments->block_bytes =
      static_cast<std::uint32_t>(request.output_block_bytes);
  arguments->block_count = npu_core_count;
  arguments->jettys_per_core = jettys_per_core;
  arguments->max_token_offset = request.max_token_offset;
  return Status::OK();
}

// AIV status lives in device memory.  Copying it to the host after every
// Decode-layer DirectGet adds a hidden D2H synchronization and defeats the
// payload path's latency goal.  Keep verification opt-in for bring-up and
// fault triage; when requested it is a strict Status, never a log-only hint.
bool IsEnabledByEnvironment(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

Status CheckBatchGetDeviceStatus(const BatchGetExecutionState& execution,
                                 std::uint32_t npu_core_count) {
  std::vector<KvafBatchGetCoreStatus> statuses(npu_core_count);
  const aclError copy_result = aclrtMemcpy(
      statuses.data(), statuses.size() * sizeof(statuses.front()),
      execution.status, execution.status_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
  if (copy_result != ACL_SUCCESS) {
    return Internal("failed to copy AIV BatchGet device status: aclrtMemcpy=" +
                    std::to_string(static_cast<int>(copy_result)));
  }
  for (std::uint32_t core = 0; core < npu_core_count; ++core) {
    const KvafBatchGetCoreStatus& status = statuses[core];
    if (status.magic != KVAF_AIV_BATCH_GET_STATUS_MAGIC || status.result != 0U) {
      return Internal(
          "AIV BatchGet device failure: core=" + std::to_string(core) +
          " magic=" + std::to_string(status.magic) +
          " result=" + std::to_string(status.result) +
          " processed=" + std::to_string(status.processed_request_count) +
          " failed_request=" + std::to_string(status.failed_request_index));
    }
  }
  return Status::OK();
}

}  // namespace

struct DefaultAccessOperator::Impl {
  explicit Impl(AccessOperatorConfig init_config) : config(std::move(init_config)) {}

  AccessOperatorConfig config;
  RemoteRegionDesc remote_region;
  KvafBatchGetTransport* transport{nullptr};
  BatchGetExecutionState execution;
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
  KvafBatchPutTransport* batch_put_transport{nullptr};
  BatchPutExecutionState batch_put_execution;
#endif
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
      impl_->config.batch_get_kernel_object_path.empty() ||
      impl_->config.npu_core_count == 0 ||
      impl_->config.npu_core_count > KVAF_AIV_BATCH_GET_MAX_AIV_CORES ||
      impl_->config.jettys_per_core == 0 ||
      impl_->config.jettys_per_core >
          KVAF_AIV_BATCH_GET_MAX_JETTYS_PER_CORE) {
    return InvalidArgument("invalid AccessOperator initialization config");
  }
  if (KvafExecuteBatchGetLoadKernel(
          impl_->config.batch_get_kernel_object_path.c_str()) !=
      KVAF_BATCH_GET_SUCCESS) {
    return Internal("failed to load BatchGet kernel");
  }

  KvafBatchGetTransport* transport = nullptr;
  const int32_t result = KvafBatchGetTransportCreate(
      reinterpret_cast<void*>(remote_region.base), remote_region.length,
      impl_->config.npu_core_count, impl_->config.jettys_per_core, &transport);
  if (result != KVAF_BATCH_GET_SUCCESS || transport == nullptr) {
    return Internal("failed to create reusable BatchGet transport");
  }

  BatchGetExecutionState execution;
  execution.status_bytes = static_cast<std::size_t>(impl_->config.npu_core_count) *
                           sizeof(KvafBatchGetCoreStatus);
  execution.queue_state_bytes =
      static_cast<std::size_t>(impl_->config.npu_core_count) *
      impl_->config.jettys_per_core *
      sizeof(KvafBatchGetQueueState);
  if (aclrtMalloc(&execution.status, execution.status_bytes,
                  ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
      aclrtMalloc(&execution.queue_state, execution.queue_state_bytes,
                  ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
      aclrtMemset(execution.status, execution.status_bytes, 0,
                  execution.status_bytes) != ACL_SUCCESS ||
      aclrtMemset(execution.queue_state, execution.queue_state_bytes, 0,
                  execution.queue_state_bytes) != ACL_SUCCESS) {
    if (execution.queue_state != nullptr) (void)aclrtFree(execution.queue_state);
    if (execution.status != nullptr) (void)aclrtFree(execution.status);
    KvafBatchGetTransportDestroy(transport);
    return Internal("failed to allocate internal BatchGet execution state");
  }
  impl_->remote_region = remote_region;
  impl_->transport = transport;
  impl_->execution = execution;
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
  if (!impl_->config.batch_put_kernel_object_path.empty()) {
    if (impl_->config.npu_core_count > KVAF_AIV_BATCH_PUT_MAX_AIV_CORES ||
        KvafExecuteBatchPutLoadKernel(
            impl_->config.batch_put_kernel_object_path.c_str()) !=
            KVAF_BATCH_PUT_SUCCESS) {
      (void)aclrtFree(impl_->execution.queue_state);
      (void)aclrtFree(impl_->execution.status);
      KvafBatchGetTransportDestroy(impl_->transport);
      impl_->transport = nullptr;
      impl_->execution = BatchGetExecutionState{};
      return Internal("failed to load BatchPut kernel");
    }

    KvafBatchPutTransport* batch_put_transport = nullptr;
    const int32_t batch_put_result = KvafBatchPutTransportCreate(
        reinterpret_cast<void*>(remote_region.base), remote_region.length,
        impl_->config.npu_core_count, &batch_put_transport);
    if (batch_put_result != KVAF_BATCH_PUT_SUCCESS ||
        batch_put_transport == nullptr) {
      (void)aclrtFree(impl_->execution.queue_state);
      (void)aclrtFree(impl_->execution.status);
      KvafBatchGetTransportDestroy(impl_->transport);
      impl_->transport = nullptr;
      impl_->execution = BatchGetExecutionState{};
      return Internal("failed to create reusable BatchPut transport");
    }
    BatchPutExecutionState batch_put_execution;
    Status batch_put_status = InitializeBatchPutExecutionState(
        impl_->config.npu_core_count, &batch_put_execution);
    if (!batch_put_status.ok()) {
      KvafBatchPutTransportDestroy(batch_put_transport);
      (void)aclrtFree(impl_->execution.queue_state);
      (void)aclrtFree(impl_->execution.status);
      KvafBatchGetTransportDestroy(impl_->transport);
      impl_->transport = nullptr;
      impl_->execution = BatchGetExecutionState{};
      return batch_put_status;
    }
    impl_->batch_put_transport = batch_put_transport;
    impl_->batch_put_execution = std::move(batch_put_execution);
    std::fprintf(stderr,
                 "[KVAF_AIV] RUNTIME_BATCH_PUT_TRANSPORT_READY cores=%u "
                 "metadata_staging=reusable\n",
                 impl_->config.npu_core_count);
  }
#endif
  impl_->initialized = true;
  return Status::OK();
}

Status DefaultAccessOperator::Shutdown() {
  if (!impl_->initialized) {
    return Status::OK();
  }

  // Callers must complete all submitted streams before Runtime::Shutdown,
  // including the default enqueue-only Python BatchGet path.
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
  ReleaseBatchPutExecutionState(&impl_->batch_put_execution);
  if (impl_->batch_put_transport != nullptr) {
    KvafBatchPutTransportDestroy(impl_->batch_put_transport);
  }
  impl_->batch_put_transport = nullptr;
#endif
  if (impl_->transport != nullptr) {
    KvafBatchGetTransportDestroy(impl_->transport);
  }
  if (impl_->execution.queue_state != nullptr) {
    (void)aclrtFree(impl_->execution.queue_state);
  }
  if (impl_->execution.status != nullptr) {
    (void)aclrtFree(impl_->execution.status);
  }
  impl_->transport = nullptr;
  impl_->execution = BatchGetExecutionState{};
  impl_->remote_region = RemoteRegionDesc{};
  impl_->initialized = false;
  return Status::OK();
}

Status DefaultAccessOperator::ExecuteBatchGet(const BatchGetExecRequest& exec) {
  if (!impl_->initialized || impl_->transport == nullptr) {
    return Status::Error(StatusCode::kNotInitialized,
                         "AccessOperator has not been initialized");
  }
  if (exec.execution.stream == nullptr) {
    return InvalidArgument("NPU BatchGet requires a caller-owned ACL stream");
  }
  KvafExecuteBatchGetKernelArgs arguments{};
  Status status = BuildBatchGetKernelArgs(
      exec, impl_->config.npu_core_count, impl_->config.jettys_per_core,
      impl_->execution, &arguments);
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

  const aclrtStream stream = static_cast<aclrtStream>(exec.execution.stream);
  const int32_t launch_result = KvafExecuteBatchGetLaunch(&arguments, stream);
  if (launch_result != KVAF_BATCH_GET_SUCCESS) {
    KvafBatchGetLocalBufferRelease(local_buffer);
    return Internal("failed to launch BatchGet");
  }
  if (exec.execution.synchronize) {
    const aclError sync_result = aclrtSynchronizeStream(stream);
    KvafBatchGetLocalBufferRelease(local_buffer);
    if (sync_result != ACL_SUCCESS) {
      return Internal("BatchGet stream synchronization failed");
    }
    if (IsEnabledByEnvironment("KVAF_VERIFY_AIV_BATCH_GET_STATUS")) {
      status = CheckBatchGetDeviceStatus(impl_->execution,
                                         impl_->config.npu_core_count);
      if (!status.ok()) return status;
    }
  } else {
    // The demo transport caches zero-reference registrations until shutdown.
    // Releasing after an externally captured launch keeps KVAF graph-agnostic;
    // the caller owns stream completion and tensor lifetime.
    KvafBatchGetLocalBufferRelease(local_buffer);
  }
  return Status::OK();
}

Status DefaultAccessOperator::ExecuteDsaDirectBatchGet(
    const DsaDirectBatchGetRequest& request) {
  if (!impl_->initialized || impl_->transport == nullptr) {
    return Status::Error(StatusCode::kNotInitialized,
                         "AccessOperator has not been initialized");
  }
  if (request.stream == nullptr) {
    return InvalidArgument(
        "DSA_ROW DirectGet requires a caller-owned ACL stream");
  }
  KvafExecuteBatchGetKernelArgs arguments{};
  Status status = BuildDsaDirectBatchGetKernelArgs(
      request, impl_->config.npu_core_count, impl_->config.jettys_per_core,
      impl_->execution, &arguments);
  if (!status.ok()) {
    return status;
  }

  KvafBatchGetLocalBuffer* local_buffer = nullptr;
  std::uint64_t channel_entity = 0;
  if (KvafBatchGetLocalBufferAcquire(
          impl_->transport, reinterpret_cast<void*>(arguments.output),
          request.output.bytes, &local_buffer,
          &channel_entity) != KVAF_BATCH_GET_SUCCESS ||
      local_buffer == nullptr || channel_entity == 0) {
    return Internal("failed to acquire caller output for DSA_ROW DirectGet");
  }
  arguments.channel_entity = channel_entity;

  const aclrtStream stream = static_cast<aclrtStream>(request.stream);
  const int32_t launch_result = KvafExecuteBatchGetLaunch(&arguments, stream);
  if (launch_result != KVAF_BATCH_GET_SUCCESS) {
    KvafBatchGetLocalBufferRelease(local_buffer);
    return Internal("failed to launch DSA_ROW DirectGet");
  }
  if (request.synchronize) {
    const aclError sync_result = aclrtSynchronizeStream(stream);
    KvafBatchGetLocalBufferRelease(local_buffer);
    if (sync_result != ACL_SUCCESS) {
      return Internal("DSA_ROW DirectGet stream synchronization failed");
    }
    // Production DSA decode never D2H-copies this diagnostic buffer.  Set the
    // opt-in flag only while debugging a transport/kernel issue; any device
    // status failure is then propagated to Python as a hard error.
    if (IsEnabledByEnvironment("KVAF_DSA_VERIFY_AIV_GET_STATUS")) {
      status = CheckBatchGetDeviceStatus(impl_->execution,
                                         impl_->config.npu_core_count);
      if (!status.ok()) return status;
    }
  } else {
    KvafBatchGetLocalBufferRelease(local_buffer);
  }
  return Status::OK();
}

Status DefaultAccessOperator::ExecuteBatchPut(const BatchPutExecRequest& exec) {
  if (!impl_->initialized) {
    return Status::NotInitialized("AccessOperator has not been initialized");
  }
  if (exec.request == nullptr || exec.put_desc == nullptr ||
      exec.request->keys.empty() || exec.request->input.data == nullptr ||
      exec.request->input.bytes == 0 ||
      exec.request->keys.size() != exec.request->input_offsets.size() ||
      exec.request->keys.size() != exec.request->lengths.size() ||
      exec.request->keys.size() != exec.put_desc->entries.size()) {
    return InvalidArgument("BatchPut request does not satisfy the NPU mock contract");
  }

  const auto* input = static_cast<const std::uint8_t*>(exec.request->input.data);
  for (std::size_t index = 0; index < exec.request->keys.size(); ++index) {
    const std::uint64_t offset = exec.request->input_offsets[index];
    const std::uint64_t length = exec.request->lengths[index];
    const PutAllocation& destination = exec.put_desc->entries[index];
    if (length == 0 || offset > exec.request->input.bytes ||
        length > exec.request->input.bytes - offset ||
        exec.request->input.device != DeviceKind::kNpu ||
        destination.address == 0 || destination.reserved_length != length) {
      return InvalidArgument("BatchPut source and destination blocks do not match");
    }
  }

#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
  if (impl_->batch_put_transport != nullptr) {
    if (exec.execution.stream == nullptr) {
      return InvalidArgument("AIV BatchPut requires a caller-owned ACL stream");
    }
    BatchPutExecutionState& execution = impl_->batch_put_execution;
    const std::size_t rows = exec.request->keys.size();
    if (!FitsU32(rows)) {
      return InvalidArgument("BatchPut row count exceeds the device ABI");
    }
    Status status = EnsureBatchPutMetadataCapacity(rows, &execution);
    if (!status.ok()) {
      return status;
    }
    std::uint64_t* const host_offsets = execution.host_metadata.data();
    std::uint64_t* const host_lengths = host_offsets + rows;
    std::uint64_t* const host_destinations = host_lengths + rows;
    std::copy(exec.request->input_offsets.begin(),
              exec.request->input_offsets.end(), host_offsets);
    std::copy(exec.request->lengths.begin(), exec.request->lengths.end(),
              host_lengths);
    for (std::size_t index = 0; index < rows; ++index) {
      host_destinations[index] = exec.put_desc->entries[index].address;
    }

    const std::size_t metadata_bytes = rows * 3U * sizeof(std::uint64_t);
    if (aclrtMemcpy(execution.metadata, metadata_bytes,
                    execution.host_metadata.data(), metadata_bytes,
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
      return Internal("failed to upload BatchPut metadata to NPU");
    }

    KvafBatchPutLocalBuffer* local_buffer = nullptr;
    if (KvafBatchPutLocalBufferAcquire(
            impl_->batch_put_transport, const_cast<void*>(exec.request->input.data),
            exec.request->input.bytes, &local_buffer) != KVAF_BATCH_PUT_SUCCESS ||
        local_buffer == nullptr) {
      return Internal("failed to acquire caller input for BatchPut");
    }

    KvafExecuteBatchPutKernelArgs arguments{};
    arguments.input = PointerValue(exec.request->input.data);
    arguments.input_offsets = PointerValue(execution.metadata);
    arguments.lengths = PointerValue(
        static_cast<const std::uint8_t*>(execution.metadata) +
        rows * sizeof(std::uint64_t));
    arguments.destination_addresses = PointerValue(
        static_cast<const std::uint8_t*>(execution.metadata) +
        rows * 2U * sizeof(std::uint64_t));
    arguments.core_status = PointerValue(execution.core_status);
    arguments.input_bytes = exec.request->input.bytes;
    arguments.row_count = static_cast<std::uint32_t>(rows);
    arguments.aiv_core_count = impl_->config.npu_core_count;

    for (KvafBatchPutCoreStatus& core_status : execution.host_core_status) {
      core_status = KvafBatchPutCoreStatus{};
    }
    const int32_t result = KvafExecuteBatchPutSync(
        local_buffer, &arguments,
        static_cast<aclrtStream>(exec.execution.stream),
        execution.host_core_status.data(),
        static_cast<std::uint32_t>(execution.host_core_status.size()));
    KvafBatchPutLocalBufferRelease(local_buffer);
    if (result != KVAF_BATCH_PUT_SUCCESS) {
      LogBatchPutDeviceStatus(execution, impl_->config.npu_core_count, result);
      return Internal("AIV Direct-HCCP BatchPut execution failed");
    }
    return Status::OK();
  }
#endif

  // This is still only a temporary ACL D2H BatchPut mock, but it must not
  // turn a contiguous caller batch into one ACL API call per KV. Preserve the
  // packed variable-length contract by coalescing only adjacent source and
  // destination ranges; non-contiguous ranges remain separate copies.
  std::size_t first = 0;
  while (first < exec.request->keys.size()) {
    const PutAllocation& first_destination = exec.put_desc->entries[first];
    const std::size_t first_offset =
        static_cast<std::size_t>(exec.request->input_offsets[first]);
    std::size_t bytes = static_cast<std::size_t>(exec.request->lengths[first]);
    std::size_t next = first + 1;
    while (next < exec.request->keys.size()) {
      const PutAllocation& previous_destination = exec.put_desc->entries[next - 1];
      const PutAllocation& next_destination = exec.put_desc->entries[next];
      const std::size_t next_offset =
          static_cast<std::size_t>(exec.request->input_offsets[next]);
      const std::size_t next_bytes =
          static_cast<std::size_t>(exec.request->lengths[next]);
      const bool contiguous_input = next_offset == first_offset + bytes;
      const bool contiguous_destination =
          next_destination.address >= previous_destination.address &&
          next_destination.address - previous_destination.address ==
              previous_destination.reserved_length;
      if (!contiguous_input || !contiguous_destination) {
        break;
      }
      bytes += next_bytes;
      ++next;
    }

    const aclError result = aclrtMemcpy(
        reinterpret_cast<void*>(first_destination.address), bytes,
        input + first_offset, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (result != ACL_SUCCESS) {
      return Internal("ACL device-to-host mock BatchPut copy failed");
    }
    first = next;
  }
  return Status::OK();
}

Status DefaultAccessOperator::ExecuteDsaFullBlockPut(
    const DsaFullBlockPutRequest& request) {
  if (!impl_->initialized) {
    return Status::NotInitialized("AccessOperator has not been initialized");
  }
  if (request.source.device != DeviceKind::kNpu || !request.source.valid() ||
      request.source_offsets.empty() || request.stream == nullptr ||
      request.source_offsets.size() != request.lengths.size() ||
      request.source_offsets.size() != request.destination_addresses.size()) {
    return InvalidArgument(
        "DSA_ROW full-block Put does not satisfy the NPU source contract");
  }
  for (std::size_t index = 0; index < request.source_offsets.size(); ++index) {
    const std::uint64_t offset = request.source_offsets[index];
    const std::uint64_t length = request.lengths[index];
    if (length == 0 || request.destination_addresses[index] == 0 ||
        offset > request.source.bytes || length > request.source.bytes - offset) {
      return InvalidArgument(
          "DSA_ROW full-block Put source or destination is outside its arena");
    }
  }

#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
  if (impl_->batch_put_transport == nullptr) {
    return Status::NotImplemented(
        "DSA_ROW requires an AIV Direct-HCCP BatchPut kernel; configure "
        "batch_put_kernel_object_path");
  }
  const std::size_t rows = request.source_offsets.size();
  if (!FitsU32(rows)) {
    return InvalidArgument("DSA_ROW full-block Put row count exceeds the device ABI");
  }
  BatchPutExecutionState& execution = impl_->batch_put_execution;
  Status status = EnsureBatchPutMetadataCapacity(rows, &execution);
  if (!status.ok()) {
    return status;
  }
  std::uint64_t* const host_offsets = execution.host_metadata.data();
  std::uint64_t* const host_lengths = host_offsets + rows;
  std::uint64_t* const host_destinations = host_lengths + rows;
  std::copy(request.source_offsets.begin(), request.source_offsets.end(),
            host_offsets);
  std::copy(request.lengths.begin(), request.lengths.end(), host_lengths);
  std::copy(request.destination_addresses.begin(),
            request.destination_addresses.end(), host_destinations);

  const std::size_t metadata_bytes = rows * 3U * sizeof(std::uint64_t);
  if (aclrtMemcpy(execution.metadata, metadata_bytes,
                  execution.host_metadata.data(), metadata_bytes,
                  ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
    return Internal("failed to upload DSA_ROW full-block Put metadata");
  }

  KvafBatchPutLocalBuffer* local_buffer = nullptr;
  if (KvafBatchPutLocalBufferAcquire(
          impl_->batch_put_transport, const_cast<void*>(request.source.data),
          request.source.bytes, &local_buffer) != KVAF_BATCH_PUT_SUCCESS ||
      local_buffer == nullptr) {
    return Internal("failed to acquire DSA_ROW packed HBM source for BatchPut");
  }

  KvafExecuteBatchPutKernelArgs arguments{};
  arguments.input = PointerValue(request.source.data);
  arguments.input_offsets = PointerValue(execution.metadata);
  arguments.lengths = PointerValue(
      static_cast<const std::uint8_t*>(execution.metadata) +
      rows * sizeof(std::uint64_t));
  arguments.destination_addresses = PointerValue(
      static_cast<const std::uint8_t*>(execution.metadata) +
      rows * 2U * sizeof(std::uint64_t));
  arguments.core_status = PointerValue(execution.core_status);
  arguments.input_bytes = request.source.bytes;
  arguments.row_count = static_cast<std::uint32_t>(rows);
  arguments.aiv_core_count = impl_->config.npu_core_count;
  for (KvafBatchPutCoreStatus& core_status : execution.host_core_status) {
    core_status = KvafBatchPutCoreStatus{};
  }
  const int32_t result = KvafExecuteBatchPutSync(
      local_buffer, &arguments, static_cast<aclrtStream>(request.stream),
      execution.host_core_status.data(),
      static_cast<std::uint32_t>(execution.host_core_status.size()));
  KvafBatchPutLocalBufferRelease(local_buffer);
  if (result != KVAF_BATCH_PUT_SUCCESS) {
    LogBatchPutDeviceStatus(execution, impl_->config.npu_core_count, result);
    return Internal("AIV Direct-HCCP DSA_ROW full-block Put failed");
  }
  return Status::OK();
#else
  (void)request;
  return Status::NotImplemented(
      "DSA_ROW requires KVAF_ENABLE_AIV_BATCH_PUT=ON");
#endif
}

}  // namespace kvaf
