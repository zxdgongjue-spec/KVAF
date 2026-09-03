#include "access_operator.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "acl/acl.h"
#include "batch_get_abi.h"
#include "kvaf_aiv_batch_get.h"
#include "meta_manager.h"

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

bool BufferRangesOverlap(const MutableBufferView& left,
                         const MutableBufferView& right) {
  if (!left.valid() || !right.valid()) {
    return false;
  }
  const std::uint64_t left_begin = PointerValue(left.data);
  const std::uint64_t right_begin = PointerValue(right.data);
  if (left.bytes > std::numeric_limits<std::uint64_t>::max() - left_begin ||
      right.bytes > std::numeric_limits<std::uint64_t>::max() - right_begin) {
    return true;
  }
  const std::uint64_t left_end = left_begin + left.bytes;
  const std::uint64_t right_end = right_begin + right.bytes;
  return left_begin < right_end && right_begin < left_end;
}

bool SameBufferRange(const MutableBufferView& left,
                     const MutableBufferView& right) {
  return left.data == right.data && left.bytes == right.bytes &&
         left.device == right.device;
}

Status ValidateLayerResidentQueueCount(const AccessOperatorConfig& config,
                                       std::uint32_t queue_count) {
  if (queue_count == 0U ||
      queue_count > KVAF_AIV_BATCH_GET_MAX_AIV_CORES ||
      queue_count > config.npu_core_count) {
    return InvalidArgument("layer-resident queue_count must select existing AIV queues");
  }
  // One resident executor queue maps to one established USER_CTL SQ/CQ pair.
  // The current 112-byte BatchGet helper has no per-wave Jetty selector.
  if (config.jettys_per_core != 1U) {
    return InvalidArgument(
        "layer-resident BatchGet requires jettys_per_core=1");
  }
  if (config.batch_get_token_bytes == 0U ||
      config.batch_get_tokens_per_block == 0U) {
    return InvalidArgument(
        "layer-resident BatchGet requires token_bytes and tokens_per_block");
  }
  return Status::OK();
}

Status ValidateLayerResidentBatchGetBinding(
    const LayerResidentBatchGetBinding& binding,
    const AccessOperatorConfig& config) {
  if (!binding.output.valid() || binding.output.device != DeviceKind::kNpu ||
      binding.output.bytes % config.batch_get_token_bytes != 0U) {
    return InvalidArgument(
        "layer-resident payload must be an NPU buffer of whole token rows");
  }
  const std::uint64_t output_rows =
      binding.output.bytes / config.batch_get_token_bytes;
  if (output_rows > std::numeric_limits<std::uint32_t>::max() ||
      binding.resident_row_count == 0U ||
      binding.resident_row_base >= output_rows ||
      binding.resident_row_count > output_rows - binding.resident_row_base) {
    return InvalidArgument(
        "layer-resident fixed slot interval is outside the payload");
  }
  if (!binding.control.valid() || binding.control.device != DeviceKind::kNpu ||
      binding.control.bytes != sizeof(KvafLayerResidentBatchGetControl)) {
    return InvalidArgument(
        "layer-resident control must be NPU torch.int64[16]");
  }
  if (BufferRangesOverlap(binding.output, binding.control)) {
    return InvalidArgument(
        "layer-resident payload and control must not alias");
  }
  return Status::OK();
}

Status BuildLayerResidentBatchGetQueueTemplate(
    const LayerResidentBatchGetQueueTemplate& source,
    const AccessOperatorConfig& config, std::uint32_t queue_count,
    KvafLayerResidentBatchGetDescriptorScratch* destination,
    const std::string& label) {
  if (destination == nullptr || source.queue_id >= queue_count) {
    return InvalidArgument(label + " has an invalid queue id");
  }
  const MutableBufferView* const buffers[] = {
      &source.address_table, &source.valid_table, &source.indices,
      &source.offsets, &source.miss_count, &source.output_indices};
  for (const MutableBufferView* buffer : buffers) {
    if (buffer == nullptr || !buffer->valid() ||
        buffer->device != DeviceKind::kNpu) {
      return InvalidArgument(label + " requires non-empty NPU descriptor buffers");
    }
  }
  if (source.address_table.bytes % sizeof(std::uint64_t) != 0U ||
      source.valid_table.bytes !=
          source.address_table.bytes / sizeof(std::uint64_t) ||
      source.indices.bytes % sizeof(std::int32_t) != 0U ||
      source.offsets.bytes != source.indices.bytes ||
      source.output_indices.bytes != source.indices.bytes ||
      source.miss_count.bytes != sizeof(std::int32_t)) {
    return InvalidArgument(label + " violates the MTP 112-byte BatchGet ABI");
  }
  const std::uint64_t table_capacity =
      source.address_table.bytes / sizeof(std::uint64_t);
  const std::uint64_t request_capacity =
      source.indices.bytes / sizeof(std::int32_t);
  if (!FitsU32(table_capacity) || !FitsU32(request_capacity) ||
      table_capacity == 0U || request_capacity == 0U) {
    return InvalidArgument(label + " capacity exceeds the device ABI");
  }
  constexpr std::size_t kBufferCount = sizeof(buffers) / sizeof(buffers[0]);
  for (std::size_t left = 0U; left < kBufferCount; ++left) {
    for (std::size_t right = left + 1U; right < kBufferCount; ++right) {
      if (BufferRangesOverlap(*buffers[left], *buffers[right])) {
        return InvalidArgument(label + " descriptor buffers must not alias");
      }
    }
  }
  *destination = KvafLayerResidentBatchGetDescriptorScratch{};
  destination->address_table = PointerValue(source.address_table.data);
  destination->valid_table = PointerValue(source.valid_table.data);
  destination->indices = PointerValue(source.indices.data);
  destination->offsets = PointerValue(source.offsets.data);
  destination->miss_count = PointerValue(source.miss_count.data);
  destination->output_indices = PointerValue(source.output_indices.data);
  destination->table_capacity = static_cast<std::uint32_t>(table_capacity);
  destination->request_capacity = static_cast<std::uint32_t>(request_capacity);
  destination->token_bytes = config.batch_get_token_bytes;
  destination->tokens_per_block = config.batch_get_tokens_per_block;
  return Status::OK();
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
      !request.miss_count.has_elements<std::int32_t>() ||
      request.miss_count.bytes != sizeof(std::int32_t) ||
      request.miss_count.device != DeviceKind::kNpu ||
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
  arguments->miss_count = PointerValue(request.miss_count.data);
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

  // A cold registration owned by exactly one enclosing forward/layer graph.
  // Unlike the historical eager BatchGet path, this object deliberately keeps
  // its output LocalBufferAcquire registration alive while the device-side
  // executor owns PostOnly/CQ progress.  It has no host hot-submit method.
  struct LayerResidentBatchGetControlBinding {
    std::uint32_t layer_id{0};
    void* output{nullptr};
    std::size_t output_bytes{0};
    void* control{nullptr};
    std::size_t control_bytes{0};
    KvafBatchGetLocalBuffer* local_buffer{nullptr};
    std::uint64_t channel_entity{0};
    void* queue_leases{nullptr};
    std::size_t queue_lease_bytes{0};
    void* descriptor_templates{nullptr};
    std::size_t descriptor_template_bytes{0};
    std::uint32_t queue_count{0};
    std::uint32_t resident_row_base{0};
    std::uint32_t resident_row_count{0};
    bool templates_configured{false};
    bool active{false};
  };

  AccessOperatorConfig config;
  RemoteRegionDesc remote_region;
  KvafBatchGetTransport* transport{nullptr};
  BatchGetExecutionState execution;
  LayerResidentBatchGetControlBinding layer_resident_control;
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

  // A forward-resident control holds the LocalBufferAcquire registration used
  // by a device-side PostOnly/CQ executor.  Tearing down the transport under
  // it would turn an incomplete forward into a use-after-free, so this is a
  // deliberately fail-closed lifecycle boundary.  The graph must drain then
  // call ReleaseLayerResidentBatchGetControl first.
  if (impl_->layer_resident_control.active) {
    return Status::Busy(
        "layer-resident BatchGet control is still bound; drain and release it "
        "before Runtime shutdown");
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
  if (impl_->layer_resident_control.active) {
    return Status::Busy(
        "ordinary BatchGet is forbidden while layer-resident control owns "
        "the Direct-HCCP SQ/CQ state");
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

Status DefaultAccessOperator::BindLayerResidentBatchGetControl(
    const LayerResidentBatchGetBinding& binding,
    const MutableBufferView& queue_leases, std::uint32_t queue_count) {
  if (!impl_->initialized || impl_->transport == nullptr) {
    return Status::NotInitialized("AccessOperator has not been initialized");
  }
  if (impl_->layer_resident_control.active) {
    return Status::AlreadyExists(
        "a layer-resident BatchGet control is already bound; release it "
        "after the current forward terminal");
  }

  Status status = ValidateLayerResidentQueueCount(impl_->config, queue_count);
  if (!status.ok()) {
    return status;
  }
  status = ValidateLayerResidentBatchGetBinding(binding, impl_->config);
  if (!status.ok()) {
    return status;
  }
  if (impl_->execution.status == nullptr ||
      impl_->execution.queue_state == nullptr) {
    return InvalidState("layer-resident BatchGet execution state is absent");
  }

  const std::size_t queue_lease_bytes =
      static_cast<std::size_t>(queue_count) * sizeof(std::int32_t);
  if (!queue_leases.valid() || queue_leases.device != DeviceKind::kNpu ||
      queue_leases.bytes != queue_lease_bytes ||
      BufferRangesOverlap(queue_leases, binding.output) ||
      BufferRangesOverlap(queue_leases, binding.control)) {
    return InvalidArgument(
        "layer-resident queue_leases must be a disjoint NPU int32[queue_count]");
  }

  // This is deliberately a cold graph-construction fence.  PyTorch's tensor
  // initialization can be queued on a different stream from the raw ACL
  // publish below; without the fence, a late zero-fill could erase the
  // control/lease state after this function returned.
  if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
    return Internal(
        "failed to synchronize device before binding layer-resident BatchGet");
  }

  const std::size_t template_bytes =
      static_cast<std::size_t>(queue_count) *
      sizeof(KvafLayerResidentBatchGetDescriptorScratch);
  void* descriptor_templates = nullptr;
  if (aclrtMemset(queue_leases.data, queue_lease_bytes, 0,
                  queue_lease_bytes) != ACL_SUCCESS ||
      aclrtMalloc(&descriptor_templates, template_bytes,
                  ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
      descriptor_templates == nullptr ||
      aclrtMemset(descriptor_templates, template_bytes, 0,
                  template_bytes) != ACL_SUCCESS) {
    if (descriptor_templates != nullptr) {
      (void)aclrtFree(descriptor_templates);
    }
    return Internal(
        "failed to allocate layer-resident descriptor templates");
  }

  KvafBatchGetLocalBuffer* local_buffer = nullptr;
  std::uint64_t channel_entity = 0U;
  if (KvafBatchGetLocalBufferAcquire(impl_->transport, binding.output.data,
                                     binding.output.bytes, &local_buffer,
                                     &channel_entity) !=
          KVAF_BATCH_GET_SUCCESS ||
      local_buffer == nullptr || channel_entity == 0U) {
    if (local_buffer != nullptr) {
      KvafBatchGetLocalBufferRelease(local_buffer);
    }
    (void)aclrtFree(descriptor_templates);
    return Internal("failed to register layer-resident BatchGet payload");
  }

  KvafLayerResidentBatchGetControl host_control{};
  host_control.magic = KVAF_LAYER_RESIDENT_BATCH_GET_CONTROL_MAGIC;
  host_control.version = KVAF_LAYER_RESIDENT_BATCH_GET_CONTROL_VERSION;
  host_control.channel_entity = channel_entity;
  host_control.queue_state = PointerValue(impl_->execution.queue_state);
  host_control.core_status = PointerValue(impl_->execution.status);
  host_control.output = PointerValue(binding.output.data);
  host_control.queue_leases = PointerValue(queue_leases.data);
  host_control.descriptor_templates = PointerValue(descriptor_templates);
  host_control.queue_count = queue_count;
  host_control.output_rows =
      static_cast<std::uint64_t>(binding.output.bytes /
                                 impl_->config.batch_get_token_bytes);
  host_control.resident_row_base = binding.resident_row_base;
  host_control.resident_row_count = binding.resident_row_count;
  host_control.jettys_per_core = impl_->config.jettys_per_core;
  host_control.token_bytes = impl_->config.batch_get_token_bytes;
  host_control.tokens_per_block = impl_->config.batch_get_tokens_per_block;
  host_control.descriptor_template_count = queue_count;
  if (aclrtMemcpy(binding.control.data, binding.control.bytes, &host_control,
                  sizeof(host_control), ACL_MEMCPY_HOST_TO_DEVICE) !=
          ACL_SUCCESS ||
      aclrtSynchronizeDevice() != ACL_SUCCESS) {
    KvafBatchGetLocalBufferRelease(local_buffer);
    (void)aclrtFree(descriptor_templates);
    return Internal("failed to publish layer-resident BatchGet control");
  }

  impl_->layer_resident_control = Impl::LayerResidentBatchGetControlBinding{
      binding.layer_id,
      binding.output.data,
      binding.output.bytes,
      binding.control.data,
      binding.control.bytes,
      local_buffer,
      channel_entity,
      queue_leases.data,
      queue_lease_bytes,
      descriptor_templates,
      template_bytes,
      queue_count,
      binding.resident_row_base,
      binding.resident_row_count,
      false,
      true};
  return Status::OK();
}

Status DefaultAccessOperator::ConfigureLayerResidentBatchGetQueueTemplates(
    const std::vector<LayerResidentBatchGetQueueTemplate>& queues) {
  if (!impl_->initialized || impl_->transport == nullptr) {
    return Status::NotInitialized("AccessOperator has not been initialized");
  }
  Impl::LayerResidentBatchGetControlBinding& binding =
      impl_->layer_resident_control;
  if (!binding.active) {
    return Status::NotFound(
        "layer-resident BatchGet control must be bound before templates");
  }
  if (binding.templates_configured) {
    return InvalidState(
        "layer-resident BatchGet templates are immutable after publication; "
        "release before a cold rebind");
  }
  if (queues.size() != binding.queue_count) {
    return InvalidArgument(
        "layer-resident templates must contain exactly queue_count entries");
  }

  struct BufferReference {
    const MutableBufferView* view{nullptr};
    std::uint32_t queue_id{0};
    std::uint32_t role{0};
  };
  constexpr std::uint32_t kAddressTableRole = 0U;
  constexpr std::uint32_t kValidTableRole = 1U;
  std::vector<KvafLayerResidentBatchGetDescriptorScratch> templates(
      binding.queue_count);
  std::vector<bool> seen(binding.queue_count, false);
  std::vector<BufferReference> references;
  references.reserve(queues.size() * 6U);
  const MutableBufferView control_view{binding.control, binding.control_bytes,
                                       DeviceKind::kNpu};
  const MutableBufferView output_view{binding.output, binding.output_bytes,
                                      DeviceKind::kNpu};
  const MutableBufferView lease_view{binding.queue_leases,
                                     binding.queue_lease_bytes,
                                     DeviceKind::kNpu};
  const MutableBufferView template_view{binding.descriptor_templates,
                                        binding.descriptor_template_bytes,
                                        DeviceKind::kNpu};

  for (const LayerResidentBatchGetQueueTemplate& queue : queues) {
    if (queue.queue_id >= binding.queue_count || seen[queue.queue_id]) {
      return InvalidArgument(
          "layer-resident templates must contain each queue_id exactly once");
    }
    seen[queue.queue_id] = true;
    const MutableBufferView* const buffers[] = {
        &queue.address_table, &queue.valid_table, &queue.indices,
        &queue.offsets, &queue.miss_count, &queue.output_indices};
    for (std::uint32_t role = 0U;
         role < sizeof(buffers) / sizeof(buffers[0]); ++role) {
      const MutableBufferView* const buffer = buffers[role];
      if (BufferRangesOverlap(*buffer, control_view) ||
          BufferRangesOverlap(*buffer, output_view) ||
          BufferRangesOverlap(*buffer, lease_view) ||
          BufferRangesOverlap(*buffer, template_view)) {
        return InvalidArgument(
            "layer-resident descriptor buffer aliases control, resident "
            "payload, lease table, or descriptor templates");
      }
      references.push_back(BufferReference{buffer, queue.queue_id, role});
    }
    Status status = BuildLayerResidentBatchGetQueueTemplate(
        queue, impl_->config, binding.queue_count,
        &templates[queue.queue_id],
        "layer-resident descriptor template queue_id=" +
            std::to_string(queue.queue_id));
    if (!status.ok()) {
      return status;
    }
  }
  for (bool queue_seen : seen) {
    if (!queue_seen) {
      return InvalidArgument(
          "layer-resident templates omit a queue_id");
    }
  }

  // address_table/valid_table are read-only address metadata and may be
  // identically shared by all queues.  Every candidate vector, including the
  // MTP compact-prefix `miss_count` word, is queue-local mutable state.
  for (std::size_t left = 0U; left < references.size(); ++left) {
    for (std::size_t right = left + 1U; right < references.size(); ++right) {
      const BufferReference& a = references[left];
      const BufferReference& b = references[right];
      if (!BufferRangesOverlap(*a.view, *b.view)) {
        continue;
      }
      const bool same_shareable_role =
          (a.role == kAddressTableRole && b.role == kAddressTableRole) ||
          (a.role == kValidTableRole && b.role == kValidTableRole);
      if (same_shareable_role && SameBufferRange(*a.view, *b.view)) {
        continue;
      }
      return InvalidArgument(
          "layer-resident descriptor template buffers overlap: queue_id=" +
          std::to_string(a.queue_id) + " conflicts with queue_id=" +
          std::to_string(b.queue_id));
    }
  }

  // The template publication is the cold producer boundary.  Device code
  // must copy one immutable 64-byte record, fill only its per-queue candidate
  // vectors, then issue its GM/DDR publish barrier before the HCCP doorbell.
  if (aclrtSynchronizeDevice() != ACL_SUCCESS ||
      aclrtMemcpy(binding.descriptor_templates,
                  binding.descriptor_template_bytes, templates.data(),
                  templates.size() * sizeof(templates.front()),
                  ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS ||
      aclrtSynchronizeDevice() != ACL_SUCCESS) {
    return Internal("failed to publish layer-resident descriptor templates");
  }
  binding.templates_configured = true;
  return Status::OK();
}

Status DefaultAccessOperator::ReleaseLayerResidentBatchGetControl() {
  if (!impl_->initialized || impl_->transport == nullptr) {
    return Status::NotInitialized("AccessOperator has not been initialized");
  }
  Impl::LayerResidentBatchGetControlBinding& binding =
      impl_->layer_resident_control;
  if (!binding.active) {
    return Status::NotFound("layer-resident BatchGet control is not bound");
  }

  // This is a cold terminal assertion, not a hidden host-side completion
  // mechanism: the enclosing forward must have returned only after all
  // device CQ work reached terminal state.  A nonzero lease fails closed and
  // keeps the LocalBufferAcquire registration alive.
  if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
    return Internal("failed to synchronize before layer-resident teardown");
  }
  std::vector<std::int32_t> owners(binding.queue_count, 0);
  if (aclrtMemcpy(owners.data(), binding.queue_lease_bytes,
                  binding.queue_leases, binding.queue_lease_bytes,
                  ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
    return Internal("failed to read layer-resident queue leases");
  }
  for (std::uint32_t queue_id = 0U; queue_id < binding.queue_count;
       ++queue_id) {
    if (owners[queue_id] != 0) {
      return Status::Busy(
          "layer-resident Direct-HCCP queue remains owned; terminal CQ drain "
          "must finish before release");
    }
  }

  if (binding.descriptor_templates != nullptr) {
    (void)aclrtFree(binding.descriptor_templates);
  }
  KvafBatchGetLocalBufferRelease(binding.local_buffer);
  binding = Impl::LayerResidentBatchGetControlBinding{};
  return Status::OK();
}

Status DefaultAccessOperator::DescribeLayerResidentBatchGetControl(
    std::string* diagnostic) const {
  if (diagnostic == nullptr) {
    return InvalidArgument("layer-resident BatchGet diagnostic output is null");
  }
  if (!impl_->initialized || !impl_->layer_resident_control.active) {
    return Status::NotFound("layer-resident BatchGet control is not bound");
  }
  const Impl::LayerResidentBatchGetControlBinding& binding =
      impl_->layer_resident_control;
  std::ostringstream message;
  message << "KVAF_MTP_LAYER_RESIDENT_CONTROL layer=" << binding.layer_id
          << " queue_count=" << binding.queue_count
          << " resident_row_base=" << binding.resident_row_base
          << " resident_row_count=" << binding.resident_row_count
          << " token_bytes=" << impl_->config.batch_get_token_bytes
          << " tokens_per_block=" << impl_->config.batch_get_tokens_per_block
          << " output=0x" << std::hex << PointerValue(binding.output)
          << " control=0x" << PointerValue(binding.control)
          << " leases=0x" << PointerValue(binding.queue_leases)
          << " templates=0x" << PointerValue(binding.descriptor_templates)
          << std::dec << " templates_configured="
          << (binding.templates_configured ? 1 : 0);
  *diagnostic = message.str();
  return Status::OK();
}

Status DefaultAccessOperator::SetBatchGetKernelProfiling(bool enabled) {
  if (!impl_->initialized || impl_->execution.status == nullptr) {
    return Status::Error(StatusCode::kNotInitialized,
                         "AccessOperator has not been initialized");
  }
  std::vector<KvafBatchGetCoreStatus> statuses(
      impl_->config.npu_core_count);
  if (enabled) {
    for (KvafBatchGetCoreStatus& status : statuses) {
      status.profile.magic = KVAF_AIV_BATCH_GET_PROFILE_MAGIC;
    }
  }
  if (aclrtMemcpy(impl_->execution.status, impl_->execution.status_bytes,
                  statuses.data(), statuses.size() * sizeof(statuses.front()),
                  ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
    return Internal("failed to configure BatchGet kernel profiling");
  }
  return Status::OK();
}

Status DefaultAccessOperator::CopyBatchGetKernelProfiles(
    std::vector<BatchGetKernelProfile>* profiles) const {
  if (profiles == nullptr) {
    return InvalidArgument("BatchGet kernel profile output must not be null");
  }
  if (!impl_->initialized || impl_->execution.status == nullptr) {
    return Status::Error(StatusCode::kNotInitialized,
                         "AccessOperator has not been initialized");
  }
  std::vector<KvafBatchGetCoreStatus> statuses(
      impl_->config.npu_core_count);
  if (aclrtMemcpy(statuses.data(), statuses.size() * sizeof(statuses.front()),
                  impl_->execution.status, impl_->execution.status_bytes,
                  ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
    return Internal("failed to copy BatchGet kernel profiles");
  }
  profiles->clear();
  profiles->reserve(statuses.size());
  for (const KvafBatchGetCoreStatus& status : statuses) {
    const KvafBatchGetCoreProfile& source = status.profile;
    const bool invalid_doorbell =
        (source.first_doorbell_cycle != 0U &&
         (source.first_doorbell_cycle < source.compact_done_cycle ||
          source.first_doorbell_cycle > source.post_done_cycle)) ||
        (status.result == 0U && source.active_requests != 0U &&
         source.first_doorbell_cycle == 0U) ||
        (source.active_requests == 0U &&
         source.first_doorbell_cycle != 0U);
    if (status.magic != KVAF_AIV_BATCH_GET_STATUS_MAGIC ||
        status.result != 0U ||
        source.magic != KVAF_AIV_BATCH_GET_PROFILE_MAGIC ||
        source.kernel_start_cycle > source.setup_done_cycle ||
        source.setup_done_cycle > source.compact_done_cycle ||
        source.compact_done_cycle > source.post_done_cycle ||
        source.post_done_cycle > source.completion_done_cycle ||
        source.completion_done_cycle > source.kernel_finish_cycle ||
        invalid_doorbell) {
      profiles->clear();
      return Internal("BatchGet kernel returned an invalid profile timeline");
    }
    BatchGetKernelProfile destination;
    destination.kernel_start_cycle = source.kernel_start_cycle;
    destination.setup_done_cycle = source.setup_done_cycle;
    destination.compact_done_cycle = source.compact_done_cycle;
    destination.first_doorbell_cycle = source.first_doorbell_cycle;
    destination.post_done_cycle = source.post_done_cycle;
    destination.completion_done_cycle = source.completion_done_cycle;
    destination.kernel_finish_cycle = source.kernel_finish_cycle;
    destination.active_requests = source.active_requests;
    destination.total_posts = source.total_posts;
    for (std::size_t jetty = 0;
         jetty < destination.jetty_completion_wait_cycles.size(); ++jetty) {
      destination.jetty_completion_wait_cycles[jetty] =
          source.jetty_completion_wait_cycles[jetty];
      destination.jetty_completion_polls[jetty] =
          source.jetty_completion_polls[jetty];
    }
    profiles->push_back(destination);
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

}  // namespace kvaf
