#include "kvaf/access_operator/access_operator.h"

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

Status ValidateFusedBatchGetQueueCount(const AccessOperatorConfig& config,
                                       std::uint32_t queue_count) {
  if (queue_count == 0U ||
      queue_count > KVAF_AIV_BATCH_GET_MAX_AIV_CORES) {
    return InvalidArgument(
        "fused BatchGet queue_count must be in [1, "
        + std::to_string(KVAF_AIV_BATCH_GET_MAX_AIV_CORES) + "]");
  }
  if (queue_count > config.npu_core_count) {
    return InvalidArgument(
        "fused BatchGet queue_count=" + std::to_string(queue_count) +
        " exceeds Runtime npu_core_count=" +
        std::to_string(config.npu_core_count));
  }
  // The device-side Begin/TryRetire/Finish primitive treats one selected
  // worker as one AIV-owned USER_CTL SQ/CQ pair.  It does not yet carry a
  // per-worker Jetty selector, so a fused control must keep this dimension
  // fixed at one even when an ordinary Runtime was configured differently.
  if (config.jettys_per_core != 1U) {
    return InvalidArgument(
        "fused BatchGet control requires jettys_per_core=1; "
        "queue_count selects one Direct-HCCP queue per worker");
  }
  return Status::OK();
}

Status ValidateFusedBatchGetBuffers(const MutableBufferView& output,
                                    const MutableBufferView& control,
                                    const char* binding_name) {
  if (!output.valid() || output.device != DeviceKind::kNpu ||
      output.bytes % kBatchGetTransferBytes != 0U) {
    return InvalidArgument(std::string(binding_name) +
                           " output must be an NPU buffer with 656-byte rows");
  }
  if (output.bytes / kBatchGetTransferBytes >
      std::numeric_limits<std::uint32_t>::max()) {
    return InvalidArgument(std::string(binding_name) +
                           " output row count exceeds the device uint32 ABI");
  }
  if (!control.valid() || control.device != DeviceKind::kNpu ||
      control.bytes != sizeof(KvafFusedBatchGetControl)) {
    return InvalidArgument(std::string(binding_name) +
                           " control must be an NPU uint64[16] buffer");
  }
  if (output.data == control.data) {
    return InvalidArgument(std::string(binding_name) +
                           " output and control must not alias");
  }
  return Status::OK();
}

Status ValidateStreamingBatchGetBuffers(const MutableBufferView& output,
                                        const MutableBufferView& control,
                                        const char* binding_name) {
  if (!output.valid() || output.device != DeviceKind::kNpu ||
      output.bytes % kBatchGetTransferBytes != 0U) {
    return InvalidArgument(std::string(binding_name) +
                           " output must be an NPU buffer with 656-byte rows");
  }
  if (output.bytes / kBatchGetTransferBytes >
      std::numeric_limits<std::uint32_t>::max()) {
    return InvalidArgument(std::string(binding_name) +
                           " output row count exceeds the device uint32 ABI");
  }
  if (!control.valid() || control.device != DeviceKind::kNpu ||
      control.bytes != sizeof(KvafStreamingBatchGetControl)) {
    return InvalidArgument(std::string(binding_name) +
                           " control must be an NPU uint64[16] buffer");
  }
  if (output.data == control.data) {
    return InvalidArgument(std::string(binding_name) +
                           " output and control must not alias");
  }
  return Status::OK();
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
    // Treat an unrepresentable range as overlapping so the caller fails
    // closed rather than accepting an ambiguous HBM address span.
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

Status BuildStreamingBatchGetQueueTemplate(
    const StreamingBatchGetQueueTemplate& source, std::uint32_t queue_count,
    std::uint64_t private_output_rows,
    KvafStreamingBatchGetDescriptorScratch* destination,
    const std::string& label) {
  if (destination == nullptr || source.queue_id >= queue_count ||
      private_output_rows == 0U) {
    return InvalidArgument(label + " has an invalid queue id or private arena");
  }
  const MutableBufferView* const buffers[] = {
      &source.address_table, &source.valid_table, &source.indices,
      &source.offsets, &source.mask, &source.output_indices};
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
      source.mask.bytes != source.indices.bytes / sizeof(std::int32_t)) {
    return InvalidArgument(label + " descriptor buffers do not satisfy the "
                           "address/valid and request-vector ABI");
  }
  const std::uint64_t table_capacity =
      source.address_table.bytes / sizeof(std::uint64_t);
  const std::uint64_t request_capacity =
      source.indices.bytes / sizeof(std::int32_t);
  if (!FitsU32(table_capacity) || !FitsU32(request_capacity) ||
      table_capacity == 0U || request_capacity == 0U ||
      request_capacity > KVAF_AIV_BATCH_GET_DENSE_REQUEST_CAPACITY) {
    return InvalidArgument(label + " capacity exceeds the streaming dense "
                           "Direct-HCCP ABI");
  }
  constexpr std::size_t kDescriptorBufferCount =
      sizeof(buffers) / sizeof(buffers[0]);
  for (std::size_t left = 0; left < kDescriptorBufferCount; ++left) {
    for (std::size_t right = left + 1U; right < kDescriptorBufferCount;
         ++right) {
      if (BufferRangesOverlap(*buffers[left], *buffers[right])) {
        return InvalidArgument(label + " descriptor buffers must not alias");
      }
    }
  }
  const std::uint64_t private_base =
      static_cast<std::uint64_t>(source.queue_id) * private_output_rows;
  if (!FitsU32(private_base)) {
    return InvalidArgument(label + " private output-row base exceeds uint32");
  }
  *destination = KvafStreamingBatchGetDescriptorScratch{};
  destination->address_table = PointerValue(source.address_table.data);
  destination->valid_table = PointerValue(source.valid_table.data);
  destination->indices = PointerValue(source.indices.data);
  destination->offsets = PointerValue(source.offsets.data);
  destination->mask = PointerValue(source.mask.data);
  destination->output_indices = PointerValue(source.output_indices.data);
  destination->table_capacity = static_cast<std::uint32_t>(table_capacity);
  destination->request_capacity = static_cast<std::uint32_t>(request_capacity);
  destination->active_request_count = 0U;
  destination->private_output_row_base =
      static_cast<std::uint32_t>(private_base);
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
      request.output_block_bytes != kBatchGetTransferBytes ||
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
      request.output_block_bytes == 0 ||
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
      !FitsU32(output_rows) || !FitsU32(request.output_block_bytes)) {
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
  arguments->block_bytes = KVAF_AIV_BATCH_GET_WQE_DATA_BYTES;
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

  struct BoundBatchGetOutput {
    BatchGetOutputBindingId id{0};
    void* output{nullptr};
    std::size_t output_bytes{0};
    KvafBatchGetLocalBuffer* local_buffer{nullptr};
    std::uint64_t channel_entity{0};
  };

  // One fused invocation retains a cold LocalBufferAcquire registration and
  // exposes only its device-resident ABI through a caller-owned
  // KvafFusedBatchGetControl. It deliberately has no hot host submit method:
  // the fused AIV helper owns SQ PI, CQ CI, doorbells, and completion polling.
  struct FusedBatchGetControlBinding {
    void* output{nullptr};
    std::size_t output_bytes{0};
    void* control{nullptr};
    std::size_t control_bytes{0};
    KvafBatchGetLocalBuffer* local_buffer{nullptr};
    std::uint64_t channel_entity{0};
    std::uint32_t queue_count{0};
    bool active{false};
  };

  struct StreamingBatchGetLayerControlBinding {
    std::uint32_t layer_id{0};
    void* output{nullptr};
    std::size_t output_bytes{0};
    void* control{nullptr};
    std::size_t control_bytes{0};
    KvafBatchGetLocalBuffer* local_buffer{nullptr};
    std::uint64_t channel_entity{0};
    void* descriptor_templates{nullptr};
    std::size_t descriptor_template_bytes{0};
    std::uint64_t private_output_rows{0};
    std::uint64_t private_output_stride_bytes{0};
    bool templates_configured{false};
  };

  // The streaming table is a single cold lifecycle object.  It permits a
  // captured layer graph to select a pre-published control by its stable
  // layer id, but it does not permit a single entry to be replaced while any
  // graph may still read the table.
  struct StreamingBatchGetControlBinding {
    std::vector<StreamingBatchGetLayerControlBinding> layers;
    void* queue_leases{nullptr};
    std::size_t queue_lease_bytes{0};
    void* descriptor_scratch{nullptr};
    std::size_t descriptor_scratch_bytes{0};
    std::uint32_t queue_count{0};
    std::uint64_t bind_epoch{0};
    bool active{false};
  };

  AccessOperatorConfig config;
  RemoteRegionDesc remote_region;
  KvafBatchGetTransport* transport{nullptr};
  BatchGetExecutionState execution;
  std::vector<BoundBatchGetOutput> bound_outputs;
  BatchGetOutputBindingId next_bound_output_id{1};
  FusedBatchGetControlBinding fused_control;
  StreamingBatchGetControlBinding streaming_control;
  std::uint64_t streaming_bind_count{0};
  std::uint64_t streaming_release_count{0};
  std::uint64_t streaming_template_configure_count{0};
  std::uint64_t next_streaming_bind_epoch{1};
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
  KvafBatchPutTransport* batch_put_transport{nullptr};
  BatchPutExecutionState batch_put_execution;
#endif
  bool initialized{false};

  bool HasDeviceFusedBatchGetControl() const {
    return fused_control.active || streaming_control.active;
  }
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
        impl_->config.npu_core_count,
        impl_->config.jettys_per_core, &batch_put_transport);
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

  // A fused or streaming Get owns Direct-HCCP local-HBM registrations that
  // can still be read by graph-replayed AIV code. Unlike an ordinary host
  // submission there is no host completion object to prove that the final
  // reader has retired. Fail closed at this cold lifecycle boundary.
  if (impl_->HasDeviceFusedBatchGetControl()) {
    return Status::Busy(
        "Shutdown is forbidden while fused or streaming BatchGet control is "
        "bound; wait for the final fused reader and release the control first");
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
  // Bound outputs deliberately retain their LocalBufferAcquire reference so
  // the decode hot path does not touch registration state.  Runtime shutdown
  // is only legal after all caller streams have completed, so releasing here
  // is safe and keeps transport ownership balanced.
  for (const Impl::BoundBatchGetOutput& binding : impl_->bound_outputs) {
    KvafBatchGetLocalBufferRelease(binding.local_buffer);
  }
  impl_->bound_outputs.clear();
  impl_->next_bound_output_id = 1;
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
  if (impl_->HasDeviceFusedBatchGetControl()) {
    return Status::Error(
        StatusCode::kBusy,
        "ordinary BatchGet is forbidden while fused or streaming BatchGet "
        "control owns "
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

Status DefaultAccessOperator::BindBatchGetOutput(
    const MutableBufferView& output, BatchGetOutputBindingId* binding_id) {
  if (binding_id == nullptr) {
    return InvalidArgument("BatchGet output binding id must not be null");
  }
  if (!impl_->initialized || impl_->transport == nullptr) {
    return Status::Error(StatusCode::kNotInitialized,
                         "AccessOperator has not been initialized");
  }
  if (impl_->HasDeviceFusedBatchGetControl()) {
    return Status::Error(
        StatusCode::kBusy,
        "native BatchGet output binding is forbidden while fused or streaming "
        "BatchGet control owns the Direct-HCCP SQ/CQ state");
  }
  if (!output.valid() || output.device != DeviceKind::kNpu ||
      output.bytes % kBatchGetTransferBytes != 0) {
    return InvalidArgument(
        "BatchGet bound output must be an NPU buffer with 656-byte rows");
  }
  for (const Impl::BoundBatchGetOutput& binding : impl_->bound_outputs) {
    if (binding.output == output.data && binding.output_bytes == output.bytes) {
      return Status::Error(
          StatusCode::kAlreadyExists,
          "BatchGet output is already bound; one lane owns one fixed output");
    }
  }

  KvafBatchGetLocalBuffer* local_buffer = nullptr;
  std::uint64_t channel_entity = 0;
  if (KvafBatchGetLocalBufferAcquire(impl_->transport, output.data, output.bytes,
                                     &local_buffer, &channel_entity) !=
          KVAF_BATCH_GET_SUCCESS ||
      local_buffer == nullptr || channel_entity == 0) {
    if (local_buffer != nullptr) {
      KvafBatchGetLocalBufferRelease(local_buffer);
    }
    return Internal("failed to bind caller output for BatchGet");
  }
  if (impl_->next_bound_output_id == 0) {
    KvafBatchGetLocalBufferRelease(local_buffer);
    return Internal("BatchGet output binding id overflow");
  }
  const BatchGetOutputBindingId id = impl_->next_bound_output_id++;
  impl_->bound_outputs.push_back(
      Impl::BoundBatchGetOutput{id, output.data, output.bytes, local_buffer,
                                 channel_entity});
  *binding_id = id;
  return Status::OK();
}

Status DefaultAccessOperator::ReleaseBatchGetOutput(
    BatchGetOutputBindingId binding_id) {
  if (!impl_->initialized || impl_->transport == nullptr) {
    return Status::Error(StatusCode::kNotInitialized,
                         "AccessOperator has not been initialized");
  }
  const auto it = std::find_if(
      impl_->bound_outputs.begin(), impl_->bound_outputs.end(),
      [binding_id](const Impl::BoundBatchGetOutput& binding) {
        return binding.id == binding_id;
      });
  if (it == impl_->bound_outputs.end()) {
    return Status::NotFound("BatchGet output binding was not found");
  }
  KvafBatchGetLocalBufferRelease(it->local_buffer);
  impl_->bound_outputs.erase(it);
  return Status::OK();
}

Status DefaultAccessOperator::ExecuteBoundBatchGet(
    const BatchGetExecRequest& exec, BatchGetOutputBindingId binding_id) {
  if (!impl_->initialized || impl_->transport == nullptr) {
    return Status::Error(StatusCode::kNotInitialized,
                         "AccessOperator has not been initialized");
  }
  if (impl_->HasDeviceFusedBatchGetControl()) {
    return Status::Error(
        StatusCode::kBusy,
        "bound BatchGet is forbidden while fused or streaming BatchGet "
        "control owns the Direct-HCCP SQ/CQ state");
  }
  if (exec.execution.stream == nullptr) {
    return InvalidArgument("NPU BatchGet requires a caller-owned ACL stream");
  }
  const auto it = std::find_if(
      impl_->bound_outputs.begin(), impl_->bound_outputs.end(),
      [binding_id](const Impl::BoundBatchGetOutput& binding) {
        return binding.id == binding_id;
      });
  if (it == impl_->bound_outputs.end()) {
    return Status::NotFound("BatchGet output binding was not found");
  }
  if (exec.request == nullptr || exec.request->output.data != it->output ||
      exec.request->output.bytes != it->output_bytes) {
    return InvalidArgument(
        "bound BatchGet request output does not match its fixed HBM binding");
  }

  KvafExecuteBatchGetKernelArgs arguments{};
  Status status = BuildBatchGetKernelArgs(
      exec, impl_->config.npu_core_count, impl_->config.jettys_per_core,
      impl_->execution, &arguments);
  if (!status.ok()) {
    return status;
  }
  arguments.channel_entity = it->channel_entity;

  const aclrtStream stream = static_cast<aclrtStream>(exec.execution.stream);
  if (KvafExecuteBatchGetLaunch(&arguments, stream) !=
      KVAF_BATCH_GET_SUCCESS) {
    return Internal("failed to launch bound BatchGet");
  }
  if (exec.execution.synchronize) {
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
      return Internal("bound BatchGet stream synchronization failed");
    }
    LogBatchGetDeviceStatus(impl_->execution, impl_->config.npu_core_count);
  }
  return Status::OK();
}

Status DefaultAccessOperator::BindFusedBatchGetControl(
    const MutableBufferView& output, const MutableBufferView& control) {
  // Preserve the original public call shape and two-worker behavior. New
  // callers opt into an explicit queue count through the overload below.
  return BindFusedBatchGetControl(output, control, 2U);
}

Status DefaultAccessOperator::BindFusedBatchGetControl(
    const MutableBufferView& output, const MutableBufferView& control,
    std::uint32_t queue_count) {
  if (!impl_->initialized || impl_->transport == nullptr) {
    return Status::Error(StatusCode::kNotInitialized,
                         "AccessOperator has not been initialized");
  }
  if (impl_->HasDeviceFusedBatchGetControl()) {
    return Status::Error(StatusCode::kAlreadyExists,
                         "a fused or streaming BatchGet control is already bound");
  }
  // Existing enqueue-only BatchGet calls intentionally do not retain a host
  // completion object.  A cold caller must therefore drain any earlier
  // ordinary Get before claiming this state; once active, the checks above
  // prevent new ordinary/native submissions from racing the selected queues.
  Status status = ValidateFusedBatchGetQueueCount(impl_->config, queue_count);
  if (!status.ok()) {
    return status;
  }
  if (!impl_->bound_outputs.empty()) {
    return Status::Error(
        StatusCode::kBusy,
        "fused BatchGet control requires no active native BatchGet output "
        "binding");
  }
  status = ValidateFusedBatchGetBuffers(output, control, "fused BatchGet");
  if (!status.ok()) {
    return status;
  }
  if (output.data == control.data || impl_->execution.status == nullptr ||
      impl_->execution.queue_state == nullptr) {
    return InvalidArgument("invalid fused BatchGet control/output state");
  }

  // `control` is usually created by torch.zeros on the caller's current
  // stream.  That fill is asynchronous.  The raw ACL H2D copy below is not
  // ordered with the PyTorch stream, so without this cold-boundary drain the
  // earlier zero-fill can complete after the copy and erase the whole control
  // block.  This API is deliberately outside graph replay: pay the device
  // fence once here, then publish a stable control block for every replay.
  if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
    return Internal(
        "failed to synchronize device before binding fused BatchGet control");
  }

  KvafBatchGetLocalBuffer* local_buffer = nullptr;
  std::uint64_t channel_entity = 0U;
  if (KvafBatchGetLocalBufferAcquire(impl_->transport, output.data,
                                     output.bytes, &local_buffer,
                                     &channel_entity) !=
          KVAF_BATCH_GET_SUCCESS ||
      local_buffer == nullptr || channel_entity == 0U) {
    if (local_buffer != nullptr) {
      KvafBatchGetLocalBufferRelease(local_buffer);
    }
    return Internal("failed to bind fused BatchGet output");
  }

  KvafFusedBatchGetControl host_control{};
  host_control.magic = KVAF_FUSED_BATCH_GET_CONTROL_MAGIC;
  host_control.version = KVAF_FUSED_BATCH_GET_CONTROL_VERSION;
  host_control.channel_entity = channel_entity;
  host_control.queue_state = PointerValue(impl_->execution.queue_state);
  host_control.core_status = PointerValue(impl_->execution.status);
  host_control.output = PointerValue(output.data);
  host_control.output_rows =
      static_cast<std::uint64_t>(output.bytes / kBatchGetTransferBytes);
  host_control.block_count = queue_count;
  host_control.jettys_per_core = impl_->config.jettys_per_core;
  host_control.block_bytes = kBatchGetTransferBytes;
  if (aclrtMemcpy(control.data, control.bytes, &host_control,
                  sizeof(host_control), ACL_MEMCPY_HOST_TO_DEVICE) !=
      ACL_SUCCESS) {
    KvafBatchGetLocalBufferRelease(local_buffer);
    return Internal("failed to write fused BatchGet device control");
  }

  // Keep publication atomic from the caller's perspective: do not mark the
  // binding active until the complete 128-byte control block is visible in
  // HBM.  This is still a one-time cold lifecycle cost.
  if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
    KvafBatchGetLocalBufferRelease(local_buffer);
    return Internal(
        "failed to synchronize device after binding fused BatchGet control");
  }

  impl_->fused_control = Impl::FusedBatchGetControlBinding{
      output.data, output.bytes, control.data, control.bytes, local_buffer,
      channel_entity, queue_count, true};
  return Status::OK();
}

Status DefaultAccessOperator::ReleaseFusedBatchGetControl() {
  if (!impl_->initialized || impl_->transport == nullptr) {
    return Status::Error(StatusCode::kNotInitialized,
                         "AccessOperator has not been initialized");
  }
  if (!impl_->fused_control.active) {
    return Status::NotFound("fused BatchGet control is not bound");
  }
  // The caller must already have observed the terminal reader.  This method
  // never synchronizes a stream because it is a cold lifecycle boundary, not
  // a hidden host-side completion path.
  KvafBatchGetLocalBufferRelease(impl_->fused_control.local_buffer);
  impl_->fused_control = Impl::FusedBatchGetControlBinding{};
  return Status::OK();
}

Status DefaultAccessOperator::BindStreamingBatchGetControl(
    const std::vector<StreamingBatchGetLayerBinding>& layers,
    const MutableBufferView& queue_leases,
    std::uint32_t queue_count) {
  if (!impl_->initialized || impl_->transport == nullptr) {
    return Status::Error(StatusCode::kNotInitialized,
                         "AccessOperator has not been initialized");
  }
  if (impl_->HasDeviceFusedBatchGetControl()) {
    return Status::Error(
        StatusCode::kAlreadyExists,
        "a fused or streaming BatchGet control is already bound; "
        "streaming bindings may only be replaced after releasing the whole "
        "table at a cold lifecycle boundary");
  }
  if (layers.empty()) {
    return InvalidArgument(
        "streaming BatchGet control requires at least one layer binding");
  }
  Status status = ValidateFusedBatchGetQueueCount(impl_->config, queue_count);
  if (!status.ok()) {
    return status;
  }
  if (!impl_->bound_outputs.empty()) {
    return Status::Error(
        StatusCode::kBusy,
        "streaming BatchGet control requires no active native BatchGet output "
        "binding");
  }
  if (impl_->execution.status == nullptr || impl_->execution.queue_state == nullptr) {
    return InvalidArgument("invalid streaming BatchGet execution state");
  }

  // Validate the entire table before creating one LocalBufferAcquire
  // registration. This makes a malformed later entry fail without changing
  // the existing Runtime state.
  for (std::size_t index = 0; index < layers.size(); ++index) {
    const StreamingBatchGetLayerBinding& layer = layers[index];
    const std::string label =
        "streaming BatchGet layer " + std::to_string(layer.layer_id);
    status = ValidateStreamingBatchGetBuffers(layer.output, layer.control,
                                              label.c_str());
    if (!status.ok()) {
      return status;
    }
    const std::uint64_t output_rows =
        static_cast<std::uint64_t>(layer.output.bytes / kBatchGetTransferBytes);
    if (output_rows < queue_count || output_rows % queue_count != 0U) {
      return InvalidArgument(
          "streaming BatchGet layer_id=" + std::to_string(layer.layer_id) +
          " output rows=" + std::to_string(output_rows) +
          " must split into queue_count=" + std::to_string(queue_count) +
          " non-empty private output regions");
    }
    for (std::size_t previous = 0; previous < index; ++previous) {
      const StreamingBatchGetLayerBinding& earlier = layers[previous];
      if (earlier.layer_id == layer.layer_id) {
        return InvalidArgument(
            "streaming BatchGet layer_id=" + std::to_string(layer.layer_id) +
            " appears more than once; hot per-layer rebind is unsupported");
      }
      if (earlier.output.data == layer.output.data &&
          earlier.output.bytes == layer.output.bytes) {
        return InvalidArgument(
            "streaming BatchGet layer_id=" + std::to_string(layer.layer_id) +
            " reuses another layer's output registration; provide one fixed "
            "output range per layer");
      }
      if (earlier.control.data == layer.control.data) {
        return InvalidArgument(
            "streaming BatchGet layer_id=" + std::to_string(layer.layer_id) +
            " reuses another layer's control buffer");
      }
      if (earlier.output.data == layer.control.data ||
          earlier.control.data == layer.output.data) {
        return InvalidArgument(
            "streaming BatchGet output and control buffers must not cross-alias "
            "between layers");
      }
    }
  }

  // As with the single-control path, all caller tensor initialization must be
  // visible before raw ACL H2D publication starts. This is intentionally a
  // cold bind, never a graph-replay synchronization.
  if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
    return Internal(
        "failed to synchronize device before binding streaming BatchGet control");
  }

  const std::size_t queue_lease_bytes =
      static_cast<std::size_t>(queue_count) *
      KVAF_STREAMING_BATCH_GET_QUEUE_LEASE_BYTES;
  if (queue_leases.device != DeviceKind::kNpu || queue_leases.data == nullptr ||
      queue_leases.bytes != queue_lease_bytes) {
    return InvalidArgument(
        "streaming BatchGet queue_leases must be one graph-visible NPU "
        "int32 entry per queue");
  }
  for (const StreamingBatchGetLayerBinding& layer : layers) {
    if (BufferRangesOverlap(queue_leases, layer.output) ||
        BufferRangesOverlap(queue_leases, layer.control)) {
      return InvalidArgument(
          "streaming BatchGet queue_leases must not alias a layer output or control");
    }
  }
  const std::size_t descriptor_scratch_bytes =
      static_cast<std::size_t>(queue_count) *
      sizeof(KvafStreamingBatchGetDescriptorScratch);
  void* descriptor_scratch = nullptr;
  if (aclrtMemset(queue_leases.data, queue_lease_bytes, 0,
                  queue_lease_bytes) != ACL_SUCCESS ||
      aclrtMalloc(&descriptor_scratch, descriptor_scratch_bytes,
                  ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
      descriptor_scratch == nullptr ||
      aclrtMemset(descriptor_scratch, descriptor_scratch_bytes, 0,
                  descriptor_scratch_bytes) != ACL_SUCCESS) {
    if (descriptor_scratch != nullptr) {
      (void)aclrtFree(descriptor_scratch);
    }
    return Internal(
        "failed to allocate streaming BatchGet queue-pool lease or descriptor table");
  }

  std::vector<Impl::StreamingBatchGetLayerControlBinding> pending;
  pending.reserve(layers.size());
  const auto release_pending = [&pending, &descriptor_scratch]() {
    for (const Impl::StreamingBatchGetLayerControlBinding& bound : pending) {
      if (bound.descriptor_templates != nullptr) {
        (void)aclrtFree(bound.descriptor_templates);
      }
      KvafBatchGetLocalBufferRelease(bound.local_buffer);
    }
    pending.clear();
    if (descriptor_scratch != nullptr) {
      (void)aclrtFree(descriptor_scratch);
      descriptor_scratch = nullptr;
    }
  };
  for (const StreamingBatchGetLayerBinding& layer : layers) {
    KvafBatchGetLocalBuffer* local_buffer = nullptr;
    std::uint64_t channel_entity = 0U;
    if (KvafBatchGetLocalBufferAcquire(impl_->transport, layer.output.data,
                                       layer.output.bytes, &local_buffer,
                                       &channel_entity) !=
            KVAF_BATCH_GET_SUCCESS ||
        local_buffer == nullptr || channel_entity == 0U) {
      if (local_buffer != nullptr) {
        KvafBatchGetLocalBufferRelease(local_buffer);
      }
      release_pending();
      return Internal(
          "failed to register streaming BatchGet output for layer_id=" +
          std::to_string(layer.layer_id));
    }

    // Templates are layer-private and graph-stable. The shared scratch table
    // below holds only a leased queue's active wave, whereas this allocation
    // preserves each layer's fixed descriptor pointers/capacities across
    // earlier/later layer replays.
    void* descriptor_templates = nullptr;
    if (aclrtMalloc(&descriptor_templates, descriptor_scratch_bytes,
                    ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        descriptor_templates == nullptr ||
        aclrtMemset(descriptor_templates, descriptor_scratch_bytes, 0,
                    descriptor_scratch_bytes) != ACL_SUCCESS) {
      if (descriptor_templates != nullptr) {
        (void)aclrtFree(descriptor_templates);
      }
      KvafBatchGetLocalBufferRelease(local_buffer);
      release_pending();
      return Internal(
          "failed to allocate streaming descriptor templates for layer_id=" +
          std::to_string(layer.layer_id));
    }

    KvafStreamingBatchGetControl host_control{};
    host_control.magic = KVAF_STREAMING_BATCH_GET_CONTROL_MAGIC;
    host_control.version = KVAF_STREAMING_BATCH_GET_CONTROL_VERSION;
    host_control.channel_entity = channel_entity;
    host_control.queue_state = PointerValue(impl_->execution.queue_state);
    host_control.core_status = PointerValue(impl_->execution.status);
    host_control.output = PointerValue(layer.output.data);
    host_control.queue_leases = PointerValue(queue_leases.data);
    host_control.descriptor_scratch = PointerValue(descriptor_scratch);
    host_control.queue_count = queue_count;
    // `output` must stay the complete LocalBufferAcquire base because the
    // direct device helper validates entity.local_hbm_address == args.output.
    // Per-queue privacy is represented by final absolute output_indices in
    // non-overlapping row intervals, not by changing args.output.
    host_control.output_rows = static_cast<std::uint64_t>(
        layer.output.bytes / kBatchGetTransferBytes);
    host_control.private_output_stride_bytes =
        static_cast<std::uint64_t>(layer.output.bytes / queue_count);
    host_control.output_offset_bytes = 0U;
    host_control.jettys_per_core = impl_->config.jettys_per_core;
    host_control.block_bytes = kBatchGetTransferBytes;
    host_control.descriptor_templates = PointerValue(descriptor_templates);
    host_control.descriptor_template_count = queue_count;
    if (aclrtMemcpy(layer.control.data, layer.control.bytes, &host_control,
                    sizeof(host_control), ACL_MEMCPY_HOST_TO_DEVICE) !=
        ACL_SUCCESS) {
      (void)aclrtFree(descriptor_templates);
      KvafBatchGetLocalBufferRelease(local_buffer);
      release_pending();
      return Internal(
          "failed to write streaming BatchGet control for layer_id=" +
          std::to_string(layer.layer_id));
    }
    pending.push_back(Impl::StreamingBatchGetLayerControlBinding{
        layer.layer_id, layer.output.data, layer.output.bytes, layer.control.data,
        layer.control.bytes, local_buffer, channel_entity, descriptor_templates,
        descriptor_scratch_bytes, host_control.output_rows,
        host_control.private_output_stride_bytes, false});
  }

  // Publish the table atomically from the caller's perspective: no layer is
  // marked active until every control and the shared int32 queue-owner table
  // has reached HBM.
  if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
    release_pending();
    return Internal(
        "failed to synchronize device after binding streaming BatchGet control");
  }
  impl_->streaming_control.layers = std::move(pending);
  // queue_leases is caller owned and kept alive by NpuRuntime.  Do not free
  // it in ReleaseStreamingBatchGetControl; the caller owns its graph tensor.
  impl_->streaming_control.queue_leases = queue_leases.data;
  impl_->streaming_control.queue_lease_bytes = queue_lease_bytes;
  impl_->streaming_control.descriptor_scratch = descriptor_scratch;
  impl_->streaming_control.descriptor_scratch_bytes = descriptor_scratch_bytes;
  impl_->streaming_control.queue_count = queue_count;
  impl_->streaming_control.bind_epoch = impl_->next_streaming_bind_epoch++;
  impl_->streaming_control.active = true;
  ++impl_->streaming_bind_count;
  return Status::OK();
}

Status DefaultAccessOperator::ConfigureStreamingBatchGetQueueTemplates(
    const std::vector<StreamingBatchGetLayerTemplates>& layers) {
  if (!impl_->initialized || impl_->transport == nullptr) {
    return Status::Error(StatusCode::kNotInitialized,
                         "AccessOperator has not been initialized");
  }
  if (!impl_->streaming_control.active) {
    return Status::NotFound(
        "streaming BatchGet control table must be bound before templates");
  }
  Impl::StreamingBatchGetControlBinding& table = impl_->streaming_control;
  if (layers.size() != table.layers.size()) {
    return InvalidArgument(
        "streaming descriptor templates must list every bound layer exactly once");
  }
  for (const Impl::StreamingBatchGetLayerControlBinding& bound : table.layers) {
    if (bound.templates_configured) {
      return InvalidState(
          "streaming descriptor templates are already configured; release the "
          "whole table before a cold rebind");
    }
  }

  struct ResolvedLayerTemplates {
    Impl::StreamingBatchGetLayerControlBinding* binding{nullptr};
    std::vector<KvafStreamingBatchGetDescriptorScratch> templates;
  };
  struct BufferReference {
    const MutableBufferView* view{nullptr};
    std::uint32_t layer_id{0};
    std::uint32_t queue_id{0};
    std::uint32_t role{0};
  };
  constexpr std::uint32_t kAddressTableRole = 0U;
  constexpr std::uint32_t kValidTableRole = 1U;

  std::vector<ResolvedLayerTemplates> resolved;
  std::vector<BufferReference> references;
  resolved.reserve(layers.size());
  references.reserve(layers.size() * table.queue_count * 6U);
  for (const StreamingBatchGetLayerTemplates& supplied : layers) {
    auto binding = std::find_if(
        table.layers.begin(), table.layers.end(),
        [&supplied](const Impl::StreamingBatchGetLayerControlBinding& candidate) {
          return candidate.layer_id == supplied.layer_id;
        });
    if (binding == table.layers.end()) {
      return InvalidArgument(
          "streaming descriptor templates name an unbound layer_id=" +
          std::to_string(supplied.layer_id));
    }
    const bool duplicate_layer = std::any_of(
        resolved.begin(), resolved.end(), [&binding](const ResolvedLayerTemplates& item) {
          return item.binding == &*binding;
        });
    if (duplicate_layer) {
      return InvalidArgument(
          "streaming descriptor templates repeat layer_id=" +
          std::to_string(supplied.layer_id));
    }
    if (supplied.queues.size() != table.queue_count) {
      return InvalidArgument(
          "streaming descriptor templates for layer_id=" +
          std::to_string(supplied.layer_id) + " must contain queue_count=" +
          std::to_string(table.queue_count) + " entries");
    }

    ResolvedLayerTemplates item;
    item.binding = &*binding;
    item.templates.resize(table.queue_count);
    std::vector<bool> seen_queue(table.queue_count, false);
    for (const StreamingBatchGetQueueTemplate& queue : supplied.queues) {
      if (queue.queue_id >= table.queue_count || seen_queue[queue.queue_id]) {
        return InvalidArgument(
            "streaming descriptor templates for layer_id=" +
            std::to_string(supplied.layer_id) +
            " must contain each queue_id exactly once");
      }
      seen_queue[queue.queue_id] = true;
      const std::string label =
          "streaming descriptor template layer_id=" +
          std::to_string(supplied.layer_id) + " queue_id=" +
          std::to_string(queue.queue_id);
      // Descriptor vectors are LI-owned mutable scratch, never a view onto a
      // registered private output arena, a graph-stable control block, or
      // KVAF's shared lease/active-descriptor tables. Rejecting all of those
      // aliases here prevents a malformed cold template from turning a later
      // device-only replay into a persistent mapping/output corruption.
      const MutableBufferView* const queue_buffers[] = {
          &queue.address_table, &queue.valid_table, &queue.indices,
          &queue.offsets, &queue.mask, &queue.output_indices};
      const MutableBufferView shared_lease{
          table.queue_leases, table.queue_lease_bytes, DeviceKind::kNpu};
      const MutableBufferView shared_scratch{
          table.descriptor_scratch, table.descriptor_scratch_bytes,
          DeviceKind::kNpu};
      for (const MutableBufferView* queue_buffer : queue_buffers) {
        if (BufferRangesOverlap(*queue_buffer, shared_lease) ||
            BufferRangesOverlap(*queue_buffer, shared_scratch)) {
          return InvalidArgument(label +
                                 " descriptor buffer aliases shared "
                                 "queue-pool state");
        }
        for (const Impl::StreamingBatchGetLayerControlBinding& reserved :
             table.layers) {
          const MutableBufferView reserved_output{
              reserved.output, reserved.output_bytes, DeviceKind::kNpu};
          const MutableBufferView reserved_control{
              reserved.control, reserved.control_bytes, DeviceKind::kNpu};
          const MutableBufferView reserved_templates{
              reserved.descriptor_templates, reserved.descriptor_template_bytes,
              DeviceKind::kNpu};
          if (BufferRangesOverlap(*queue_buffer, reserved_output) ||
              BufferRangesOverlap(*queue_buffer, reserved_control) ||
              BufferRangesOverlap(*queue_buffer, reserved_templates)) {
            return InvalidArgument(label +
                                   " descriptor buffer aliases a reserved "
                                   "streaming output/control/template range");
          }
        }
      }
      Status status = BuildStreamingBatchGetQueueTemplate(
          queue, table.queue_count, binding->private_output_rows,
          &item.templates[queue.queue_id], label);
      if (!status.ok()) {
        return status;
      }
      for (std::uint32_t role = 0U;
           role < sizeof(queue_buffers) / sizeof(queue_buffers[0]); ++role) {
        references.push_back(BufferReference{queue_buffers[role],
                                             supplied.layer_id, queue.queue_id,
                                             role});
      }
    }
    resolved.push_back(std::move(item));
  }
  if (resolved.size() != table.layers.size()) {
    return InvalidArgument(
        "streaming descriptor templates omit at least one bound layer");
  }

  // Address/valid tables may intentionally be shared identically by multiple
  // queues. Every mutable descriptor vector must be disjoint across every
  // queue/layer, and a partial overlapping static table is rejected too.
  for (std::size_t left = 0; left < references.size(); ++left) {
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
          "streaming descriptor template buffers overlap: layer_id=" +
          std::to_string(a.layer_id) + " queue_id=" +
          std::to_string(a.queue_id) + " conflicts with layer_id=" +
          std::to_string(b.layer_id) + " queue_id=" +
          std::to_string(b.queue_id));
    }
  }

  // This is a cold pre-capture publication. It deliberately synchronizes
  // only here: replaying LI never asks the host to rewrite a descriptor.
  if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
    return Internal(
        "failed to synchronize device before configuring streaming templates");
  }
  for (const ResolvedLayerTemplates& item : resolved) {
    if (item.binding == nullptr || item.binding->descriptor_templates == nullptr ||
        item.binding->descriptor_template_bytes !=
            item.templates.size() *
                sizeof(KvafStreamingBatchGetDescriptorScratch) ||
        aclrtMemcpy(item.binding->descriptor_templates,
                    item.binding->descriptor_template_bytes,
                    item.templates.data(),
                    item.templates.size() *
                        sizeof(KvafStreamingBatchGetDescriptorScratch),
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
      return Internal("failed to publish a streaming descriptor template table");
    }
  }
  if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
    return Internal(
        "failed to synchronize device after configuring streaming templates");
  }
  for (const ResolvedLayerTemplates& item : resolved) {
    item.binding->templates_configured = true;
  }
  ++impl_->streaming_template_configure_count;
  return Status::OK();
}

Status DefaultAccessOperator::ReleaseStreamingBatchGetControl() {
  if (!impl_->initialized || impl_->transport == nullptr) {
    return Status::Error(StatusCode::kNotInitialized,
                         "AccessOperator has not been initialized");
  }
  if (!impl_->streaming_control.active) {
    return Status::NotFound("streaming BatchGet control table is not bound");
  }
  // The caller must already have observed the terminal reader for every
  // layer. This remains a cold lifecycle operation and intentionally does
  // not synchronize a stream behind the caller's back.
  for (const Impl::StreamingBatchGetLayerControlBinding& bound :
       impl_->streaming_control.layers) {
    if (bound.descriptor_templates != nullptr) {
      (void)aclrtFree(bound.descriptor_templates);
    }
    KvafBatchGetLocalBufferRelease(bound.local_buffer);
  }
  if (impl_->streaming_control.descriptor_scratch != nullptr) {
    (void)aclrtFree(impl_->streaming_control.descriptor_scratch);
  }
  impl_->streaming_control = Impl::StreamingBatchGetControlBinding{};
  ++impl_->streaming_release_count;
  return Status::OK();
}

Status DefaultAccessOperator::DescribeStreamingBatchGetControl(
    std::string* diagnostic) const {
  if (diagnostic == nullptr) {
    return InvalidArgument("streaming BatchGet diagnostic output is null");
  }
  if (!impl_->initialized || impl_->transport == nullptr) {
    return Status::Error(StatusCode::kNotInitialized,
                         "AccessOperator has not been initialized");
  }
  if (!impl_->streaming_control.active) {
    return Status::NotFound("streaming BatchGet control table is not bound");
  }
  const Impl::StreamingBatchGetControlBinding& table =
      impl_->streaming_control;
  std::ostringstream message;
  message << "KVAF_STREAMING_CONTROL epoch=" << table.bind_epoch
          << " bind_count=" << impl_->streaming_bind_count
          << " release_count=" << impl_->streaming_release_count
          << " template_configure_count="
          << impl_->streaming_template_configure_count
          << " queue_count=" << table.queue_count
          << " layers=" << table.layers.size() << " queue_leases=0x"
          << std::hex << PointerValue(table.queue_leases)
          << " descriptor_scratch=0x" << PointerValue(table.descriptor_scratch)
          << std::dec << " lease_bytes=" << table.queue_lease_bytes
          << " scratch_bytes=" << table.descriptor_scratch_bytes;
  for (const Impl::StreamingBatchGetLayerControlBinding& layer : table.layers) {
    message << " | layer=" << layer.layer_id << " control=0x" << std::hex
            << PointerValue(layer.control) << " output=0x"
            << PointerValue(layer.output) << " entity=0x"
            << layer.channel_entity << std::dec
            << " templates=0x" << std::hex
            << PointerValue(layer.descriptor_templates) << std::dec
            << " template_bytes=" << layer.descriptor_template_bytes
            << " templates_configured="
            << (layer.templates_configured ? 1 : 0)
            << " private_rows=" << layer.private_output_rows
            << " private_stride=" << layer.private_output_stride_bytes;
  }
  *diagnostic = message.str();
  return Status::OK();
}

Status DefaultAccessOperator::GetStreamingBatchGetControlSnapshot(
    StreamingBatchGetControlSnapshot* snapshot) const {
  if (snapshot == nullptr) {
    return InvalidArgument("streaming BatchGet snapshot output is null");
  }
  if (!impl_->initialized || impl_->transport == nullptr) {
    return Status::Error(StatusCode::kNotInitialized,
                         "AccessOperator has not been initialized");
  }
  StreamingBatchGetControlSnapshot result;
  result.magic = KVAF_STREAMING_BATCH_GET_CONTROL_MAGIC;
  result.version = KVAF_STREAMING_BATCH_GET_CONTROL_VERSION;
  result.bind_count = impl_->streaming_bind_count;
  result.release_count = impl_->streaming_release_count;
  result.template_configure_count = impl_->streaming_template_configure_count;
  const Impl::StreamingBatchGetControlBinding& table =
      impl_->streaming_control;
  result.active = table.active;
  if (table.active) {
    result.bind_epoch = table.bind_epoch;
    result.queue_count = table.queue_count;
    result.queue_lease_address = PointerValue(table.queue_leases);
    result.descriptor_scratch_address = PointerValue(table.descriptor_scratch);
    result.queue_lease_bytes = table.queue_lease_bytes;
    result.descriptor_scratch_bytes = table.descriptor_scratch_bytes;
    result.layers.reserve(table.layers.size());
    for (const Impl::StreamingBatchGetLayerControlBinding& layer :
         table.layers) {
      result.layers.push_back(StreamingBatchGetLayerSnapshot{
          layer.layer_id, PointerValue(layer.output), PointerValue(layer.control),
          layer.channel_entity, PointerValue(layer.descriptor_templates),
          layer.descriptor_template_bytes, layer.private_output_rows,
          layer.private_output_stride_bytes, layer.templates_configured});
    }
  }
  *snapshot = std::move(result);
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
  // NpuRuntime rejects BatchPut before it mutates Runtime metadata. Keep the
  // same protection here as a second boundary for direct C++ users: an AIV
  // fused or streaming Get reads this transport's remote payload and fixed
  // output view.
  if (impl_->HasDeviceFusedBatchGetControl()) {
    return Status::Busy(
        "BatchPut is forbidden while fused or streaming BatchGet control "
        "owns the Direct-HCCP read state");
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
