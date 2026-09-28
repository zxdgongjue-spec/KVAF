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
#include "kvaf_aiv_batch_get.h"
#include "meta_manager.h"
#if defined(KVAF_ENABLE_AICPU_URMA)
#include "host.h"
#endif

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
  std::size_t metadata_capacity_bytes{0};
  std::vector<std::uint64_t> host_metadata;
  std::vector<KvafBatchPutCoreStatus> host_core_status;
  std::vector<KvafBatchPutLocalBuffer*> local_buffers;
};

struct BatchPutMetadataLayout {
  std::size_t source_descriptors{0};
  std::size_t input_offsets{0};
  std::size_t lengths{0};
  std::size_t destinations{0};
  std::size_t source_indices{0};
  std::size_t bytes{0};
};

std::size_t AlignToU64(std::size_t bytes) {
  return (bytes + sizeof(std::uint64_t) - 1U) &
         ~(sizeof(std::uint64_t) - 1U);
}

BatchPutMetadataLayout MakeBatchPutMetadataLayout(std::size_t sources,
                                                   std::size_t rows) {
  BatchPutMetadataLayout layout;
  layout.source_descriptors = 0;
  layout.input_offsets =
      layout.source_descriptors +
      sources * sizeof(KvafBatchPutSourceDescriptor);
  layout.lengths = layout.input_offsets + rows * sizeof(std::uint64_t);
  layout.destinations = layout.lengths + rows * sizeof(std::uint64_t);
  layout.source_indices =
      layout.destinations + rows * sizeof(std::uint64_t);
  layout.bytes = layout.source_indices + rows * sizeof(std::uint32_t);
  return layout;
}

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

Status EnsureBatchPutMetadataCapacity(std::size_t sources, std::size_t rows,
                                      BatchPutExecutionState* execution) {
  if (execution == nullptr || sources == 0 || rows == 0 || sources > rows ||
      rows > (std::numeric_limits<std::size_t>::max() - 7U) / 44U) {
    return InvalidArgument("BatchPut metadata dimensions are invalid");
  }
  const BatchPutMetadataLayout layout =
      MakeBatchPutMetadataLayout(sources, rows);
  execution->host_metadata.resize(AlignToU64(layout.bytes) /
                                  sizeof(std::uint64_t));
  if (layout.bytes <= execution->metadata_capacity_bytes) {
    return Status::OK();
  }

  void* metadata = nullptr;
  if (aclrtMalloc(&metadata, layout.bytes, ACL_MEM_MALLOC_HUGE_FIRST) !=
      ACL_SUCCESS) {
    return Internal("failed to grow BatchPut device metadata buffers");
  }

  if (execution->metadata != nullptr) {
    (void)aclrtFree(execution->metadata);
  }
  execution->metadata = metadata;
  execution->metadata_capacity_bytes = layout.bytes;
  return Status::OK();
}

void LogBatchPutDeviceStatus(const BatchPutExecutionState& execution,
                             std::uint32_t npu_core_count,
                             int32_t result, bool aicpu) {
  std::fprintf(stderr, "[KVAF_AIV] BATCH_PUT_EXECUTION_ERROR result=%d\n",
               static_cast<int>(result));
  if (aicpu && result == KVAF_BATCH_PUT_RUNTIME_ERROR) {
    std::fprintf(stderr, "[KVAF_AIV] BATCH_PUT_DEVICE_STATUS_UNAVAILABLE reason=runtime_error\n");
    return;
  }
  for (std::uint32_t core = 0; core < npu_core_count; ++core) {
    const KvafBatchPutCoreStatus& status = execution.host_core_status[core];
    if (aicpu) {
      std::fprintf(stderr,
                   "[KVAF_AIV] BATCH_PUT_DEVICE_ERROR core=%u magic=0x%llx "
                   "result=%llu processed=%llu failed_request=%llu cq_status=%llu "
                   "cq_sub_status=%llu cqe_word=0x%llx sq_pi=%llu cq_ci=%llu last_wqe=0x%llx\n",
                   core, static_cast<unsigned long long>(status.magic),
                   static_cast<unsigned long long>(status.result),
                   static_cast<unsigned long long>(status.processed_request_count),
                   static_cast<unsigned long long>(status.failed_request_index),
                   static_cast<unsigned long long>(status.cq_status),
                   static_cast<unsigned long long>(status.cq_sub_status),
                   static_cast<unsigned long long>(status.reserved[0]),
                   static_cast<unsigned long long>(status.sq_producer_index),
                   static_cast<unsigned long long>(status.cq_consumer_index),
                   static_cast<unsigned long long>(status.last_wqe_protocol));
    } else {
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
#if defined(KVAF_ENABLE_AICPU_URMA)
  void* gather_output_spans{nullptr};
  std::size_t gather_output_spans_capacity{0U};
#endif
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
      (impl_->config.batch_get_kernel_object_path.empty() &&
       impl_->config.aicpu_urma_kernel_json_path.empty()) ||
      impl_->config.npu_core_count == 0 ||
      impl_->config.npu_core_count > KVAF_AIV_BATCH_GET_MAX_AIV_CORES ||
      impl_->config.jettys_per_core == 0 ||
      impl_->config.jettys_per_core > KVAF_AIV_BATCH_GET_MAX_JETTYS_PER_CORE) {
    return InvalidArgument("invalid AccessOperator initialization config");
  }
  if (impl_->config.peer_link == nullptr) {
    return InvalidState("AccessOperator requires an established Runtime Link");
  }
#if defined(KVAF_ENABLE_AICPU_URMA)
  if (!impl_->config.aicpu_urma_kernel_json_path.empty() &&
      (impl_->config.npu_core_count != 1U || impl_->config.jettys_per_core != 1U ||
       !aicpu_urma::Load(impl_->config.aicpu_urma_kernel_json_path.c_str()))) {
    return Internal("AICPU URMA requires one worker and a loadable kernel JSON");
  }
#else
  if (!impl_->config.aicpu_urma_kernel_json_path.empty()) {
    return InvalidArgument("AICPU URMA was not enabled at build time");
  }
#endif
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
  if (impl_->config.aicpu_urma_kernel_json_path.empty() &&
      KvafExecuteBatchGetLoadKernel(
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
  if (!impl_->config.batch_put_kernel_object_path.empty() ||
      !impl_->config.aicpu_urma_kernel_json_path.empty()) {
    if (impl_->config.npu_core_count > KVAF_AIV_BATCH_PUT_MAX_AIV_CORES ||
        (impl_->config.aicpu_urma_kernel_json_path.empty() &&
         KvafExecuteBatchPutLoadKernel(
            impl_->config.batch_put_kernel_object_path.c_str()) !=
            KVAF_BATCH_PUT_SUCCESS)) {
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
    std::fprintf(stderr,
                 "[KVAF_AIV] RUNTIME_BATCH_PUT_TRANSPORT_READY cores=%u "
                 "jettys_per_core=%u "
                 "metadata_staging=reusable\n",
                 impl_->config.npu_core_count,
                 impl_->config.jettys_per_core);
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
#if defined(KVAF_ENABLE_AICPU_URMA)
  if (impl_->gather_output_spans != nullptr) {
    (void)aclrtFree(impl_->gather_output_spans);
    impl_->gather_output_spans = nullptr;
    impl_->gather_output_spans_capacity = 0U;
  }
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
  int32_t launch_result = KVAF_BATCH_GET_RUNTIME_ERROR;
#if defined(KVAF_ENABLE_AICPU_URMA)
  if (!impl_->config.aicpu_urma_kernel_json_path.empty()) {
    launch_result = aicpu_urma::LaunchGet(&arguments, stream) == 0
                        ? KVAF_BATCH_GET_SUCCESS : KVAF_BATCH_GET_RUNTIME_ERROR;
  } else
#endif
  {
    launch_result = KvafExecuteBatchGetLaunch(&arguments, stream);
  }
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
#if defined(KVAF_ENABLE_AICPU_URMA)
    if (!impl_->config.aicpu_urma_kernel_json_path.empty()) {
      KvafBatchGetCoreStatus device_status{};
      if (aclrtMemcpy(&device_status, sizeof(device_status), impl_->execution.status,
                      sizeof(device_status), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
          device_status.magic != KVAF_AIV_BATCH_GET_STATUS_MAGIC || device_status.result != 0U) {
        return Internal("AICPU URMA BatchGet device completion failed");
      }
    }
#endif
  } else {
    // The demo transport caches zero-reference registrations until shutdown.
    // Releasing after an externally captured launch keeps KVAF graph-agnostic;
    // the caller owns stream completion and tensor lifetime.
    KvafBatchGetLocalBufferRelease(local_buffer);
  }
  return Status::OK();
}

Status DefaultAccessOperator::ExecuteCpuGatherRead(const CpuGatherReadExecRequest& request) {
#if defined(KVAF_ENABLE_AICPU_URMA)
  const auto elapsed_ns = [](std::chrono::steady_clock::time_point start) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count());
  };
  if (!impl_->initialized || impl_->transport == nullptr ||
      impl_->config.aicpu_urma_kernel_json_path.empty())
    return InvalidState("AICPU URMA GatherRead transport is unavailable");
  if (request.output.device != DeviceKind::kNpu || !request.output.valid() ||
      request.stream == nullptr || request.bytes == 0U ||
      request.output_spans == nullptr || request.output_span_count == 0U ||
      request.sgl_size == 0U ||
      request.output_extent > UINT64_MAX - 63U)
    return InvalidArgument("GatherRead output requires a valid SGL and 64-byte HBM scratch");
  const std::uint64_t scratch_offset =
      (request.output_extent + 63U) & ~std::uint64_t{63U};
  if (scratch_offset > request.output.bytes ||
      request.output.bytes - scratch_offset < aicpu_urma::kGatherScratchBytes)
    return InvalidArgument("GatherRead output requires 64-byte HBM scratch");
  const auto descriptor_start = std::chrono::steady_clock::now();
  const std::size_t descriptor_bytes =
      static_cast<std::size_t>(request.output_span_count) * sizeof(GatherOutputSpan);
  if (descriptor_bytes > impl_->gather_output_spans_capacity) {
    void* descriptors = nullptr;
    if (aclrtMalloc(&descriptors, descriptor_bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)
      return Internal("failed to allocate GatherRead HBM SGL descriptors");
    if (impl_->gather_output_spans != nullptr) (void)aclrtFree(impl_->gather_output_spans);
    impl_->gather_output_spans = descriptors;
    impl_->gather_output_spans_capacity = descriptor_bytes;
  }
  if (aclrtMemcpy(impl_->gather_output_spans, impl_->gather_output_spans_capacity,
                  request.output_spans, descriptor_bytes,
                  ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS)
    return Internal("failed to upload GatherRead HBM SGL descriptors");
  if (request.timing != nullptr)
    request.timing->descriptor_upload_ns = elapsed_ns(descriptor_start);
  KvafBatchGetLocalBuffer* local_buffer = nullptr;
  std::uint64_t channel_entity = 0U;
  const auto register_start = std::chrono::steady_clock::now();
  if (KvafBatchGetLocalBufferAcquire(impl_->transport, request.output.data,
                                     request.output.bytes, &local_buffer,
                                     &channel_entity) != KVAF_BATCH_GET_SUCCESS ||
      local_buffer == nullptr || channel_entity == 0U)
    return Internal("failed to register GatherRead HBM output");
  if (request.timing != nullptr) request.timing->register_ns = elapsed_ns(register_start);
  aicpu_urma::GatherReadArgs arguments{};
  arguments.channel_entity = channel_entity;
  arguments.queue_state = reinterpret_cast<std::uint64_t>(impl_->execution.queue_state);
  arguments.core_status = reinterpret_cast<std::uint64_t>(impl_->execution.status);
  arguments.output = reinterpret_cast<std::uint64_t>(request.output.data);
  arguments.output_spans = reinterpret_cast<std::uint64_t>(impl_->gather_output_spans);
  arguments.output_span_count = request.output_span_count;
  arguments.sgl_size = request.sgl_size;
  arguments.output_extent = request.output_extent;
  arguments.remote_staging = request.remote_staging;
  arguments.remote_command = request.remote_command;
  arguments.remote_notify = request.remote_notify;
  arguments.remote_ack = request.remote_ack;
  arguments.sequence = request.sequence;
  arguments.bytes = request.bytes;
  const aclrtStream stream = static_cast<aclrtStream>(request.stream);
  const auto launch_start = std::chrono::steady_clock::now();
  const int32_t launch = aicpu_urma::LaunchGatherRead(&arguments, stream);
  if (request.timing != nullptr) request.timing->launch_ns = elapsed_ns(launch_start);
  if (launch != 0) {
    KvafBatchGetLocalBufferRelease(local_buffer);
    return Internal("AICPU URMA GatherRead launch failed");
  }
  aclError sync = ACL_SUCCESS;
  const auto sync_start = std::chrono::steady_clock::now();
  sync = aclrtSynchronizeStream(stream);
  if (request.timing != nullptr) request.timing->sync_ns = elapsed_ns(sync_start);
  const auto release_start = std::chrono::steady_clock::now();
  KvafBatchGetLocalBufferRelease(local_buffer);
  if (request.timing != nullptr) request.timing->release_ns = elapsed_ns(release_start);
  aicpu_urma::GatherReadStatus device_status{};
  const auto copy_start = std::chrono::steady_clock::now();
  const aclError copy = aclrtMemcpy(&device_status, sizeof(device_status),
                                   impl_->execution.status, sizeof(device_status),
                                   ACL_MEMCPY_DEVICE_TO_HOST);
  if (request.timing != nullptr) request.timing->status_copy_ns = elapsed_ns(copy_start);
  if (request.timing != nullptr && copy == ACL_SUCCESS &&
      device_status.magic == aicpu_urma::kGatherStatusMagic &&
      device_status.sequence == request.sequence) {
    request.timing->notify_ns = device_status.notify_ns;
    request.timing->notify_post_ns = device_status.notify_post_ns;
    request.timing->notify_cq_ns = device_status.notify_cq_ns;
    request.timing->ack_poll_ns = device_status.ack_poll_ns;
    request.timing->ack_post_ns = device_status.ack_post_ns;
    request.timing->ack_cq_ns = device_status.ack_cq_ns;
    request.timing->ack_cache_ns = device_status.ack_cache_ns;
    request.timing->scatter_cache_ns = device_status.scatter_cache_ns;
    request.timing->scatter_plan_ns = device_status.scatter_plan_ns;
    request.timing->scatter_post_ns = device_status.scatter_post_ns;
    request.timing->scatter_cq_ns = device_status.scatter_cq_ns;
    request.timing->cpu_dispatch_ns = device_status.cpu.dispatch_ns;
    request.timing->cpu_notify_spin_ns = device_status.cpu.notify_spin_ns;
    request.timing->cpu_source_cache_ns = device_status.cpu.source_cache_ns;
    request.timing->cpu_copy_ns = device_status.cpu.copy_ns;
    request.timing->cpu_staging_flush_ns = device_status.cpu.staging_flush_ns;
    request.timing->ack_polls = device_status.ack_polls;
    request.timing->read_wqes = device_status.read_wqes;
  }
  if (copy == ACL_SUCCESS && device_status.magic == aicpu_urma::kGatherStatusMagic &&
      device_status.sequence == request.sequence &&
      (sync != ACL_SUCCESS || device_status.result != 0U)) {
    std::fprintf(stderr, "[KVAF_AICPU_URMA] GATHER_READ_ERROR acl=%d result=%llu phase=%llu ack=%llu "
                         "sq=%llu cq=%llu cqe_word=0x%llx sequence=%llu spans=%u sgl_size=%u\n",
                 static_cast<int>(sync),
                 static_cast<unsigned long long>(device_status.result),
                 static_cast<unsigned long long>(device_status.phase),
                 static_cast<unsigned long long>(device_status.ack_seen),
                 static_cast<unsigned long long>(device_status.sq_producer_index),
                 static_cast<unsigned long long>(device_status.cq_consumer_index),
                 static_cast<unsigned long long>(device_status.cqe_word),
                 static_cast<unsigned long long>(device_status.sequence),
                 request.output_span_count, request.sgl_size);
  }
  if (sync != ACL_SUCCESS) return Internal("AICPU URMA GatherRead stream synchronization failed");
  if (copy != ACL_SUCCESS || device_status.magic != aicpu_urma::kGatherStatusMagic ||
      device_status.sequence != request.sequence || device_status.result != 0U)
    return Internal("AICPU URMA GatherRead device completion failed");
  return Status::OK();
#else
  (void)request;
  return Status::NotImplemented("AICPU URMA is disabled");
#endif
}

Status DefaultAccessOperator::ExecuteBatchPut(const BatchPutExecRequest& exec) {
  if (!impl_->initialized) {
    return Status::NotInitialized("AccessOperator has not been initialized");
  }
  if (exec.plan == nullptr || exec.put_desc == nullptr ||
      exec.plan->flat_keys.empty() || exec.plan->sources.empty() ||
      exec.plan->flat_keys.size() != exec.plan->source_indices.size() ||
      exec.plan->flat_keys.size() != exec.plan->offsets.size() ||
      exec.plan->flat_keys.size() != exec.plan->lengths.size() ||
      exec.plan->flat_keys.size() != exec.put_desc->entries.size()) {
    return InvalidArgument("BatchPut request does not satisfy the NPU contract");
  }

  for (std::size_t index = 0; index < exec.plan->flat_keys.size(); ++index) {
    const std::uint32_t source_index = exec.plan->source_indices[index];
    if (source_index >= exec.plan->sources.size()) {
      return InvalidArgument("BatchPut source index is invalid");
    }
    const ConstBufferView& source = exec.plan->sources[source_index];
    const std::uint64_t offset = exec.plan->offsets[index];
    const std::uint64_t length = exec.plan->lengths[index];
    const PutAllocation& destination = exec.put_desc->entries[index];
    if (source.data == nullptr || source.bytes == 0 || length == 0 ||
        offset > source.bytes || length > source.bytes - offset ||
        source.device != DeviceKind::kNpu ||
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
    const std::size_t rows = exec.plan->flat_keys.size();
    const std::size_t sources = exec.plan->sources.size();
    if (!FitsU32(rows) || !FitsU32(sources)) {
      return InvalidArgument("BatchPut row count exceeds the device ABI");
    }
    Status status = EnsureBatchPutMetadataCapacity(sources, rows, &execution);
    if (!status.ok()) {
      return status;
    }
    const BatchPutMetadataLayout layout =
        MakeBatchPutMetadataLayout(sources, rows);
    auto* const host_metadata = reinterpret_cast<std::uint8_t*>(
        execution.host_metadata.data());
    std::memcpy(host_metadata + layout.input_offsets,
                exec.plan->offsets.data(), rows * sizeof(std::uint64_t));
    std::memcpy(host_metadata + layout.lengths, exec.plan->lengths.data(),
                rows * sizeof(std::uint64_t));
    auto* const host_destinations = reinterpret_cast<std::uint64_t*>(
        host_metadata + layout.destinations);
    for (std::size_t index = 0; index < rows; ++index) {
      host_destinations[index] = exec.put_desc->entries[index].address;
    }

    std::vector<KvafBatchPutLocalBuffer*>& local_buffers =
        execution.local_buffers;
    local_buffers.clear();
    local_buffers.reserve(sources);
    for (std::size_t source_index = 0; source_index < sources; ++source_index) {
      const ConstBufferView& source = exec.plan->sources[source_index];
      KvafBatchPutLocalBuffer* local_buffer = nullptr;
      KvafBatchPutSourceDescriptor descriptor{};
      if (KvafBatchPutLocalBufferAcquire(
              impl_->batch_put_transport, const_cast<void*>(source.data),
              source.bytes, &local_buffer) != KVAF_BATCH_PUT_SUCCESS ||
          local_buffer == nullptr ||
          KvafBatchPutLocalBufferDescribe(local_buffer, &descriptor) !=
              KVAF_BATCH_PUT_SUCCESS) {
        if (local_buffer != nullptr) {
          KvafBatchPutLocalBufferRelease(local_buffer);
        }
        for (KvafBatchPutLocalBuffer* acquired : local_buffers) {
          KvafBatchPutLocalBufferRelease(acquired);
        }
        local_buffers.clear();
        return Internal("failed to acquire caller inputs for BatchPut");
      }
      std::memcpy(host_metadata + layout.source_descriptors +
                      source_index * sizeof(descriptor),
                  &descriptor, sizeof(descriptor));
      local_buffers.push_back(local_buffer);
    }
    std::memcpy(host_metadata + layout.source_indices,
                exec.plan->source_indices.data(),
                rows * sizeof(std::uint32_t));

    if (aclrtMemcpy(execution.metadata, layout.bytes,
                    execution.host_metadata.data(), layout.bytes,
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
      for (KvafBatchPutLocalBuffer* local_buffer : local_buffers) {
        KvafBatchPutLocalBufferRelease(local_buffer);
      }
      local_buffers.clear();
      return Internal("failed to upload BatchPut metadata to NPU");
    }

    KvafExecuteBatchPutKernelArgs arguments{};
    const auto* const device_metadata =
        static_cast<const std::uint8_t*>(execution.metadata);
    arguments.source_descriptors = PointerValue(
        device_metadata + layout.source_descriptors);
    arguments.source_indices = PointerValue(
        device_metadata + layout.source_indices);
    arguments.input_offsets = PointerValue(
        device_metadata + layout.input_offsets);
    arguments.lengths = PointerValue(device_metadata + layout.lengths);
    arguments.destination_addresses = PointerValue(
        device_metadata + layout.destinations);
    arguments.core_status = PointerValue(execution.core_status);
    arguments.row_count = static_cast<std::uint32_t>(rows);
    arguments.aiv_core_count = impl_->config.npu_core_count;

    for (KvafBatchPutCoreStatus& core_status : execution.host_core_status) {
      core_status = KvafBatchPutCoreStatus{};
    }
    int32_t result = KVAF_BATCH_PUT_RUNTIME_ERROR;
#if defined(KVAF_ENABLE_AICPU_URMA)
    if (!impl_->config.aicpu_urma_kernel_json_path.empty()) {
      result = KvafExecuteBatchPutAicpuSync(
          impl_->batch_put_transport, &arguments,
          static_cast<aclrtStream>(exec.execution.stream),
          execution.host_core_status.data(),
          static_cast<std::uint32_t>(execution.host_core_status.size()),
          static_cast<std::uint32_t>(sources));
    } else
#endif
    {
      result = KvafExecuteBatchPutSync(
        impl_->batch_put_transport, &arguments,
        static_cast<aclrtStream>(exec.execution.stream),
        execution.host_core_status.data(),
        static_cast<std::uint32_t>(execution.host_core_status.size()));
    }
    for (KvafBatchPutLocalBuffer* local_buffer : local_buffers) {
      KvafBatchPutLocalBufferRelease(local_buffer);
    }
    local_buffers.clear();
    if (result != KVAF_BATCH_PUT_SUCCESS) {
      const bool aicpu = !impl_->config.aicpu_urma_kernel_json_path.empty();
      LogBatchPutDeviceStatus(execution, impl_->config.npu_core_count, result, aicpu);
      return Internal(aicpu ? "Direct-HCCP BatchPut execution failed"
                            : "AIV Direct-HCCP BatchPut execution failed");
    }
    return Status::OK();
  }
#endif

  // This is still only a temporary ACL D2H BatchPut mock, but it must not
  // turn a contiguous caller batch into one ACL API call per KV. Preserve the
  // packed variable-length contract by coalescing only adjacent source and
  // destination ranges; non-contiguous ranges remain separate copies.
  std::size_t first = 0;
  while (first < exec.plan->flat_keys.size()) {
    const std::uint32_t source_index = exec.plan->source_indices[first];
    const ConstBufferView& source = exec.plan->sources[source_index];
    const auto* input = static_cast<const std::uint8_t*>(source.data);
    const PutAllocation& first_destination = exec.put_desc->entries[first];
    const std::size_t first_offset =
        static_cast<std::size_t>(exec.plan->offsets[first]);
    std::size_t bytes = static_cast<std::size_t>(exec.plan->lengths[first]);
    std::size_t next = first + 1;
    while (next < exec.plan->flat_keys.size()) {
      const PutAllocation& previous_destination = exec.put_desc->entries[next - 1];
      const PutAllocation& next_destination = exec.put_desc->entries[next];
      const std::size_t next_offset =
          static_cast<std::size_t>(exec.plan->offsets[next]);
      const std::size_t next_bytes =
          static_cast<std::size_t>(exec.plan->lengths[next]);
      const bool contiguous_input =
          exec.plan->source_indices[next] == source_index &&
          next_offset == first_offset + bytes;
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

}  // namespace kvaf
