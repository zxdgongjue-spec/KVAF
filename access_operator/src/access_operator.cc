#include "access_operator.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "acl/acl.h"
#include "batch_get_abi.h"
#include "direct_hccp/direct_hccp_session.h"
#include "kvaf_aiv_batch_get.h"
#include "meta_manager.h"

#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
#include "batch_put_abi.h"
#include "kvaf_aiv_batch_put.h"
#endif

namespace kvaf {
namespace {

using HostClock = std::chrono::steady_clock;

std::uint64_t ElapsedNs(HostClock::time_point begin,
                        HostClock::time_point end) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
          .count());
}

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
  void* host_metadata{nullptr};
  std::size_t host_metadata_capacity{0};
  KvafBatchPutCoreStatus* host_core_status{nullptr};
  std::size_t host_core_status_count{0};
  bool profiling_enabled{false};
  bool host_profiling_enabled{false};
  BatchPutHostProfile last_host_profile{};
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
  if (execution->host_core_status != nullptr) {
    (void)aclrtFreeHost(execution->host_core_status);
  }
  if (execution->host_metadata != nullptr) {
    (void)aclrtFreeHost(execution->host_metadata);
  }
  *execution = BatchPutExecutionState{};
}

Status InitializeBatchPutExecutionState(std::uint32_t npu_core_count,
                                        BatchPutExecutionState* execution) {
  if (execution == nullptr || npu_core_count == 0) {
    return InvalidArgument("invalid BatchPut execution state");
  }
  BatchPutExecutionState state;
  const std::size_t status_bytes =
      static_cast<std::size_t>(npu_core_count) * sizeof(KvafBatchPutCoreStatus);
  void* host_core_status = nullptr;
  if (aclrtMallocHost(&host_core_status, status_bytes) != ACL_SUCCESS ||
      host_core_status == nullptr) {
    return Internal("failed to allocate pinned BatchPut core status buffer");
  }
  state.host_core_status =
      static_cast<KvafBatchPutCoreStatus*>(host_core_status);
  state.host_core_status_count = npu_core_count;
  std::memset(state.host_core_status, 0, status_bytes);
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
  const std::size_t bytes = rows * 3U * sizeof(std::uint64_t);
  if (rows > execution->host_metadata_capacity) {
    void* host_metadata = nullptr;
    if (aclrtMallocHost(&host_metadata, bytes) != ACL_SUCCESS ||
        host_metadata == nullptr) {
      return Internal("failed to grow pinned BatchPut host metadata buffer");
    }
    if (execution->host_metadata != nullptr) {
      (void)aclrtFreeHost(execution->host_metadata);
    }
    execution->host_metadata = host_metadata;
    execution->host_metadata_capacity = rows;
  }
  if (rows <= execution->metadata_capacity) {
    return Status::OK();
  }

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
      !request.offsets.has_elements<std::int32_t>() ||
      !request.output_indices.has_elements<std::int32_t>() ||
      !request.output.valid() ||
      request.output_block_bytes == 0 ||
      request.tokens_per_block == 0 ||
      request.output.bytes % request.output_block_bytes != 0 ||
      request.indices.device != request.offsets.device ||
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
      request.indices.bytes % sizeof(std::int32_t) != 0 ||
      request.offsets.bytes != request.indices.bytes ||
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
      !FitsU32(output_rows) || !FitsU32(request.output_block_bytes) ||
      !FitsU32(request.tokens_per_block)) {
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
  arguments->token_bytes =
      static_cast<std::uint32_t>(request.output_block_bytes);
  arguments->tokens_per_block =
      static_cast<std::uint32_t>(request.tokens_per_block);
  arguments->block_count = npu_core_count;
  arguments->jettys_per_core = jettys_per_core;
  return Status::OK();
}

void LogBatchGetDeviceStatus(const BatchGetExecutionState& execution,
                             std::uint32_t npu_core_count) {
  std::vector<KvafBatchGetCoreStatus> statuses(npu_core_count);
  const aclError copy_result = aclrtMemcpy(
      statuses.data(), statuses.size() * sizeof(statuses.front()),
      execution.status, execution.status_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
  if (copy_result != ACL_SUCCESS) {
    std::fprintf(stderr,
                 "[KVAF_AIV] BATCH_GET_STATUS_LOG_ERROR aclrtMemcpy=%d\n",
                 static_cast<int>(copy_result));
    return;
  }
  for (std::uint32_t core = 0; core < npu_core_count; ++core) {
    const KvafBatchGetCoreStatus& status = statuses[core];
    if (status.magic != KVAF_AIV_BATCH_GET_STATUS_MAGIC || status.result != 0U) {
      // Keep device execution faults in the operator diagnostic channel rather
      // than turning the kernel status buffer into a Python BatchGet result.
      // Deployments can route this marker through their CANN/plog collection.
      std::fprintf(stderr,
                   "[KVAF_AIV] BATCH_GET_DEVICE_ERROR core=%u magic=0x%llx "
                   "result=%llu processed=%llu failed_request=%llu\n",
                   core, static_cast<unsigned long long>(status.magic),
                   static_cast<unsigned long long>(status.result),
                   static_cast<unsigned long long>(status.processed_request_count),
                   static_cast<unsigned long long>(status.failed_request_index));
    }
  }
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
      impl_->config.jettys_per_core > KVAF_AIV_BATCH_GET_MAX_JETTYS_PER_CORE) {
    return InvalidArgument("invalid AccessOperator initialization config");
  }
  if (impl_->config.peer_link == nullptr) {
    return InvalidState("AccessOperator requires an established Runtime Link");
  }
  if (std::all_of(impl_->config.peer_local_eid.begin(),
                  impl_->config.peer_local_eid.end(),
                  [](std::uint8_t value) { return value == 0U; })) {
    return InvalidArgument("Runtime Link requires a Runtime-selected local EID");
  }
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
  if (!impl_->config.batch_put_kernel_object_path.empty() &&
      (impl_->config.npu_core_count > KVAF_AIV_BATCH_PUT_MAX_AIV_CORES ||
       impl_->config.jettys_per_core >
           KVAF_AIV_BATCH_PUT_MAX_JETTYS_PER_CORE)) {
    return InvalidArgument("invalid BatchPut transport configuration");
  }
#endif
  if (KvafExecuteBatchGetLoadKernel(
          impl_->config.batch_get_kernel_object_path.c_str()) !=
      KVAF_BATCH_GET_SUCCESS) {
    return Internal("failed to load BatchGet kernel");
  }

  KvafBatchGetTransport* transport = nullptr;
  const int32_t result = KvafBatchGetTransportCreateWithPeerLink(
      reinterpret_cast<void*>(remote_region.base), remote_region.length,
      impl_->config.npu_core_count, impl_->config.jettys_per_core,
      impl_->config.peer_link.get(), impl_->config.peer_local_eid.data(), &transport);
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
    const int32_t batch_put_result = KvafBatchPutTransportCreateWithPeerLink(
        reinterpret_cast<void*>(remote_region.base), remote_region.length,
        impl_->config.npu_core_count, impl_->config.jettys_per_core,
        impl_->config.peer_link.get(), impl_->config.peer_local_eid.data(),
        &batch_put_transport);
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
    direct_hccp::VerboseLog(
        "[KVAF_AIV] RUNTIME_BATCH_PUT_TRANSPORT_READY cores=%u "
        "jettys_per_core=%u metadata_staging=reusable\n",
        impl_->config.npu_core_count, impl_->config.jettys_per_core);
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
    LogBatchGetDeviceStatus(impl_->execution, impl_->config.npu_core_count);
  } else {
    // The demo transport caches zero-reference registrations until shutdown.
    // Releasing after an externally captured launch keeps KVAF graph-agnostic;
    // the caller owns stream completion and tensor lifetime.
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
    const bool host_profiling = execution.host_profiling_enabled;
    BatchPutHostProfile host_profile{};
    if (host_profiling) {
      host_profile.row_count = rows;
      host_profile.payload_bytes = exec.request->input.bytes;
    }
    if (!FitsU32(rows)) {
      return InvalidArgument("BatchPut row count exceeds the device ABI");
    }
    const HostClock::time_point metadata_reserve_begin =
        host_profiling ? HostClock::now() : HostClock::time_point{};
    Status status = EnsureBatchPutMetadataCapacity(rows, &execution);
    if (host_profiling) {
      host_profile.metadata_reserve_ns =
          ElapsedNs(metadata_reserve_begin, HostClock::now());
    }
    if (!status.ok()) {
      return status;
    }
    const HostClock::time_point metadata_pack_begin =
        host_profiling ? HostClock::now() : HostClock::time_point{};
    std::uint64_t* const host_offsets =
        static_cast<std::uint64_t*>(execution.host_metadata);
    std::uint64_t* const host_lengths = host_offsets + rows;
    std::uint64_t* const host_destinations = host_lengths + rows;
    std::copy(exec.request->input_offsets.begin(),
              exec.request->input_offsets.end(), host_offsets);
    std::copy(exec.request->lengths.begin(), exec.request->lengths.end(),
              host_lengths);
    for (std::size_t index = 0; index < rows; ++index) {
      host_destinations[index] = exec.put_desc->entries[index].address;
    }
    if (host_profiling) {
      host_profile.metadata_pack_ns =
          ElapsedNs(metadata_pack_begin, HostClock::now());
    }

    const std::size_t metadata_bytes = rows * 3U * sizeof(std::uint64_t);
    if (host_profiling) {
      host_profile.metadata_bytes = metadata_bytes;
    }

    KvafBatchPutLocalBuffer* local_buffer = nullptr;
    const HostClock::time_point local_buffer_acquire_begin =
        host_profiling ? HostClock::now() : HostClock::time_point{};
    if (KvafBatchPutLocalBufferAcquire(
            impl_->batch_put_transport, const_cast<void*>(exec.request->input.data),
            exec.request->input.bytes, &local_buffer) != KVAF_BATCH_PUT_SUCCESS ||
        local_buffer == nullptr) {
      return Internal("failed to acquire caller input for BatchPut");
    }
    if (host_profiling) {
      host_profile.local_buffer_acquire_ns =
          ElapsedNs(local_buffer_acquire_begin, HostClock::now());
    }

    const aclrtStream stream = static_cast<aclrtStream>(exec.execution.stream);
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

    std::memset(execution.host_core_status, 0,
                execution.host_core_status_count *
                    sizeof(KvafBatchPutCoreStatus));
    for (std::size_t core = 0; core < execution.host_core_status_count;
         ++core) {
      KvafBatchPutCoreStatus& core_status = execution.host_core_status[core];
      if (execution.profiling_enabled) {
        core_status.profile.magic = KVAF_AIV_BATCH_PUT_PROFILE_MAGIC;
      }
    }
    const HostClock::time_point metadata_h2d_begin =
        host_profiling ? HostClock::now() : HostClock::time_point{};
    if (aclrtMemcpyAsync(execution.metadata, metadata_bytes,
                         execution.host_metadata, metadata_bytes,
                         ACL_MEMCPY_HOST_TO_DEVICE, stream) != ACL_SUCCESS) {
      KvafBatchPutLocalBufferRelease(local_buffer);
      return Internal("failed to enqueue BatchPut metadata upload to NPU");
    }
    if (host_profiling) {
      host_profile.metadata_h2d_ns =
          ElapsedNs(metadata_h2d_begin, HostClock::now());
    }

    const HostClock::time_point execute_sync_begin =
        host_profiling ? HostClock::now() : HostClock::time_point{};
    const int32_t result = KvafExecuteBatchPutSync(
        local_buffer, &arguments, stream, execution.host_core_status,
        static_cast<std::uint32_t>(execution.host_core_status_count));
    if (host_profiling) {
      host_profile.execute_sync_ns =
          ElapsedNs(execute_sync_begin, HostClock::now());
    }
    if (result != KVAF_BATCH_PUT_SUCCESS) {
      // The metadata upload may have been accepted even when launch or status
      // setup subsequently failed. Do not reuse its pinned source buffer until
      // the stream has stopped reading it.
      (void)aclrtSynchronizeStream(stream);
    }
    const HostClock::time_point local_buffer_release_begin =
        host_profiling ? HostClock::now() : HostClock::time_point{};
    KvafBatchPutLocalBufferRelease(local_buffer);
    if (host_profiling) {
      host_profile.local_buffer_release_ns =
          ElapsedNs(local_buffer_release_begin, HostClock::now());
    }
    if (result != KVAF_BATCH_PUT_SUCCESS) {
      LogBatchPutDeviceStatus(execution, impl_->config.npu_core_count, result);
      return Internal("AIV Direct-HCCP BatchPut execution failed");
    }
    if (host_profiling) {
      execution.last_host_profile = host_profile;
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

Status DefaultAccessOperator::SetBatchPutKernelProfiling(bool enabled) {
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
  if (!impl_->initialized || impl_->batch_put_transport == nullptr) {
    return Status::Error(StatusCode::kNotInitialized,
                         "BatchPut transport has not been initialized");
  }
  impl_->batch_put_execution.profiling_enabled = enabled;
  return Status::OK();
#else
  (void)enabled;
  return InvalidState("BatchPut kernel profiling is not available in this build");
#endif
}

Status DefaultAccessOperator::SetBatchPutHostProfiling(bool enabled) {
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
  if (!impl_->initialized || impl_->batch_put_transport == nullptr) {
    return Status::Error(StatusCode::kNotInitialized,
                         "BatchPut transport has not been initialized");
  }
  impl_->batch_put_execution.host_profiling_enabled = enabled;
  impl_->batch_put_execution.last_host_profile = BatchPutHostProfile{};
  return Status::OK();
#else
  (void)enabled;
  return InvalidState("BatchPut host profiling is not available in this build");
#endif
}

Status DefaultAccessOperator::CopyBatchPutHostProfile(
    BatchPutHostProfile* profile) const {
  if (profile == nullptr) {
    return InvalidArgument("BatchPut host profile output must not be null");
  }
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
  if (!impl_->initialized || impl_->batch_put_transport == nullptr) {
    return Status::Error(StatusCode::kNotInitialized,
                         "BatchPut transport has not been initialized");
  }
  if (!impl_->batch_put_execution.host_profiling_enabled) {
    return InvalidState("BatchPut host profiling is not enabled");
  }
  *profile = impl_->batch_put_execution.last_host_profile;
  return Status::OK();
#else
  return InvalidState("BatchPut host profiling is not available in this build");
#endif
}

Status DefaultAccessOperator::CopyBatchPutKernelProfiles(
    std::vector<BatchPutKernelProfile>* profiles) const {
  if (profiles == nullptr) {
    return InvalidArgument("BatchPut kernel profile output must not be null");
  }
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
  if (!impl_->initialized || impl_->batch_put_transport == nullptr) {
    return Status::Error(StatusCode::kNotInitialized,
                         "BatchPut transport has not been initialized");
  }
  const BatchPutExecutionState& execution = impl_->batch_put_execution;
  const KvafBatchPutCoreStatus* statuses = execution.host_core_status;
  const std::size_t status_count = execution.host_core_status_count;
  if (statuses == nullptr || status_count == 0U) {
    return Internal("BatchPut host core status buffer is not initialized");
  }
  profiles->clear();
  profiles->reserve(status_count);
  for (std::size_t core = 0; core < status_count; ++core) {
    const KvafBatchPutCoreStatus& status = statuses[core];
    const KvafBatchPutCoreProfile& source = status.profile;
    if (status.magic != KVAF_AIV_BATCH_PUT_STATUS_MAGIC ||
        status.result != KVAF_BATCH_PUT_CORE_SUCCESS ||
        source.magic != KVAF_AIV_BATCH_PUT_PROFILE_MAGIC) {
      profiles->clear();
      return Internal("BatchPut kernel returned an invalid profile summary");
    }
    BatchPutKernelProfile destination;
    destination.execute_cycles = source.execute_cycles;
    destination.non_execute_cycles = source.non_execute_cycles;
    profiles->push_back(destination);
  }
  return Status::OK();
#else
  profiles->clear();
  return InvalidState("BatchPut kernel profiling is not available in this build");
#endif
}

}  // namespace kvaf
