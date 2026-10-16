#include "access_operator.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
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

void LogBatchGetDeviceStatus(const BatchGetExecutionState& execution,
                             std::uint32_t npu_core_count,
                             bool include_successful = false);

const char* BatchGetCqWaitReason(std::uint32_t flags);

Status CollectBatchGetCqDiagnostics(const BatchGetExecutionState& execution,
                                   std::uint32_t workers,
                                   std::uint32_t jettys_per_core,
                                   std::vector<BatchGetCqDiagnostic>* diagnostics) {
  std::vector<KvafBatchGetQueueState> queues(
      static_cast<std::size_t>(workers) * jettys_per_core);
  if (aclrtMemcpy(queues.data(), queues.size() * sizeof(queues.front()),
                  execution.queue_state, execution.queue_state_bytes,
                  ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
    return Internal("failed to inspect BatchGet queue completion state");
  }
  diagnostics->clear();
  for (std::size_t index = 0; index < queues.size(); ++index) {
    const auto& queue = queues[index];
    const auto trace_header = static_cast<std::uint32_t>(queue.reserved[2] >> 32U);
    const bool trace_valid =
        (trace_header & KVAF_BATCH_GET_CQ_TRACE_TAG_MASK) == KVAF_BATCH_GET_CQ_TRACE_TAG;
    if (queue.last_error == 0U && !trace_valid) continue;
    BatchGetCqDiagnostic item;
    item.core = static_cast<std::uint32_t>(index / jettys_per_core);
    item.jetty = static_cast<std::uint32_t>(index % jettys_per_core);
    item.last_error = queue.last_error;
    item.sq_producer_index = queue.sq_producer_index;
    item.cq_consumer_index = queue.cq_consumer_index;
    item.completed_batch_count = queue.completed_batch_count;
    item.trace_valid = trace_valid;
    if (trace_valid) {
      item.raw_cq_word = static_cast<std::uint32_t>(queue.reserved[0]);
      item.expected_owner = static_cast<std::uint32_t>((queue.reserved[0] >> 32U) & 1U);
      item.slow_elapsed_ticks = queue.reserved[1];
      item.polls = static_cast<std::uint32_t>(queue.reserved[2]);
      item.flags = trace_header & ~KVAF_BATCH_GET_CQ_TRACE_TAG_MASK;
      item.fast_budget_exceeded = (item.flags & KVAF_BATCH_GET_CQ_TRACE_SLOW_ENTERED) != 0U;
      item.late_success = (item.flags & KVAF_BATCH_GET_CQ_TRACE_LATE_SUCCESS) != 0U;
      item.bypass_probe = (item.flags & KVAF_BATCH_GET_CQ_TRACE_BYPASS_PROBE) != 0U;
      item.reason = BatchGetCqWaitReason(item.flags);
    }
    diagnostics->push_back(item);
  }
  return Status::OK();
}

const char* BatchGetCqWaitReason(std::uint32_t flags) {
  if ((flags & KVAF_BATCH_GET_CQ_TRACE_ERROR_CQE) != 0U) return "error_cqe";
  if ((flags & KVAF_BATCH_GET_CQ_TRACE_DEADLINE) != 0U) return "device_deadline";
  if ((flags & KVAF_BATCH_GET_CQ_TRACE_POLL_LIMIT) != 0U) return "slow_poll_limit";
  if ((flags & KVAF_BATCH_GET_CQ_TRACE_LATE_SUCCESS) != 0U) return "late_success";
  return "unclassified";
}

void LogBatchGetCqDiagnostic(const BatchGetCqDiagnostic& item) {
  if (!item.trace_valid) return;
  std::fprintf(stderr,
               "[KVAF_AIV] BATCH_GET_CQ_WAIT_DETAILS core=%u jetty=%u "
               "last_error=%llu polls=%u slow_elapsed_ticks=%llu "
               "reason=%s raw_cq_word=0x%x observed_owner=%u expected_owner=%u "
               "cq_status=%u cq_sub_status=%u fast_budget_exceeded=%u late_success=%u "
               "bypass_probe=%u sq_after=%llu cq_after=%llu "
               "fast_poll_budget=20000 slow_deadline_ticks=100000000 clock_hz=1000000000 "
               "trace_scope=most_recent_slow_or_error_wait timing_scope=Get_CQ\n",
               item.core, item.jetty, static_cast<unsigned long long>(item.last_error),
               item.polls, static_cast<unsigned long long>(item.slow_elapsed_ticks),
               BatchGetCqWaitReason(item.flags), item.raw_cq_word,
               (item.raw_cq_word >> 2U) & 1U, item.expected_owner,
               (item.raw_cq_word >> 24U) & 0xffU, (item.raw_cq_word >> 16U) & 0xffU,
               static_cast<unsigned>((item.flags & KVAF_BATCH_GET_CQ_TRACE_SLOW_ENTERED) != 0U),
               static_cast<unsigned>((item.flags & KVAF_BATCH_GET_CQ_TRACE_LATE_SUCCESS) != 0U),
               static_cast<unsigned>((item.flags & KVAF_BATCH_GET_CQ_TRACE_BYPASS_PROBE) != 0U),
               static_cast<unsigned long long>(item.sq_producer_index),
               static_cast<unsigned long long>(item.cq_consumer_index));
}

Status CheckBatchGetQueueHealth(const BatchGetExecutionState& execution,
                               std::uint32_t workers,
                               std::uint32_t jettys_per_core) {
  std::vector<BatchGetCqDiagnostic> diagnostics;
  const Status read = CollectBatchGetCqDiagnostics(execution, workers, jettys_per_core, &diagnostics);
  if (!read.ok()) return read;
  bool failed = false;
  for (const auto& queue : diagnostics) {
    LogBatchGetCqDiagnostic(queue);
    if (queue.last_error != 0U) {
      failed = true;
      std::fprintf(stderr,
                   "[KVAF_AIV] BATCH_GET_QUEUE_ERROR core=%u jetty=%u "
                   "last_error=%llu sq_pi=%llu cq_ci=%llu completed_batches=%llu\n",
                   queue.core, queue.jetty,
                   static_cast<unsigned long long>(queue.last_error),
                   static_cast<unsigned long long>(queue.sq_producer_index),
                   static_cast<unsigned long long>(queue.cq_consumer_index),
                   static_cast<unsigned long long>(queue.completed_batch_count));
    }
  }
  if (failed) {
    // A failed queue may have a partial or stale core status. Keep both views
    // for diagnosis without clearing poison or releasing uncertain DMA state.
    LogBatchGetDeviceStatus(execution, workers, true);
    return Internal("BatchGet queue has a transport error; keep its output "
                    "and registrations alive until device/process teardown");
  }
  return Status::OK();
}

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
  if (result != KVAF_BATCH_PUT_KERNEL_ERROR) {
    // Sync/launch/copy failure returned before a valid D2H snapshot. The
    // caller's zeroed host buffer is not evidence about any device core.
    std::fprintf(stderr,
                 "[KVAF_AIV] BATCH_PUT_DEVICE_STATUS_UNAVAILABLE result=%d "
                 "status_snapshot=unavailable device_status_read=0 "
                 "host_buffer_is_not_device_snapshot=1\n", static_cast<int>(result));
    return;
  }
  for (std::uint32_t core = 0; core < npu_core_count; ++core) {
    const KvafBatchPutCoreStatus& status = execution.host_core_status[core];
    std::fprintf(stderr,
                 "[KVAF_AIV] BATCH_PUT_DEVICE_ERROR core=%u magic=0x%llx "
                 "result=%llu processed=%llu failed_request=%llu cq_status=%llu "
                 "cq_sub_status=%llu completed_batches=%llu sq_pi=%llu cq_ci=%llu "
                 "last_wqe=0x%llx\n",
                 core, static_cast<unsigned long long>(status.magic),
                 static_cast<unsigned long long>(status.result),
                 static_cast<unsigned long long>(status.processed_request_count),
                 static_cast<unsigned long long>(status.failed_request_index),
                 static_cast<unsigned long long>(status.cq_status),
                 static_cast<unsigned long long>(status.cq_sub_status),
                 static_cast<unsigned long long>(status.completed_batch_count),
                 static_cast<unsigned long long>(status.sq_producer_index),
                 static_cast<unsigned long long>(status.cq_consumer_index),
                 static_cast<unsigned long long>(status.last_wqe_protocol));
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
                             std::uint32_t npu_core_count,
                             bool include_successful) {
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
    const bool failed = status.magic != KVAF_AIV_BATCH_GET_STATUS_MAGIC || status.result != 0U;
    if (include_successful || failed) {
      // Keep device execution faults in the operator diagnostic channel rather
      // than turning the kernel status buffer into a Python BatchGet result.
      // Deployments can route this marker through their CANN/plog collection.
      std::fprintf(stderr,
                   "[KVAF_AIV] %s core=%u magic=0x%llx "
                   "result=%llu processed=%llu completed_batches=%llu "
                   "sq_pi=%llu cq_ci=%llu failed_request=%llu last_wqe=0x%llx\n",
                   failed ? "BATCH_GET_DEVICE_ERROR" : "BATCH_GET_DEVICE_SNAPSHOT",
                   core, static_cast<unsigned long long>(status.magic),
                   static_cast<unsigned long long>(status.result),
                   static_cast<unsigned long long>(status.processed_request_count),
                   static_cast<unsigned long long>(status.completed_batch_count),
                   static_cast<unsigned long long>(status.sq_producer_index),
                   static_cast<unsigned long long>(status.cq_consumer_index),
                   static_cast<unsigned long long>(status.failed_request_index),
                   static_cast<unsigned long long>(status.last_wqe_protocol));
      if (status.profile.magic == KVAF_AIV_BATCH_GET_PROFILE_MAGIC) {
        // Keep failed timelines without passing them through the success-only
        // profile API. These timings belong to each Jetty, while core_result
        // is the aggregate result of the owner group. The host cannot infer
        // the device clock frequency or an independent Jetty result here.
        for (std::uint32_t jetty = 0;
             jetty < KVAF_AIV_BATCH_GET_MAX_JETTYS_PER_CORE; ++jetty) {
          std::fprintf(stderr,
                       "[KVAF_AIV] BATCH_GET_CQ_PROFILE core=%u jetty=%u "
                       "core_result=%llu polls=%llu wait_ticks=%llu "
                       "active_requests=%llu total_posts=%llu clock_hz=unconfirmed\n",
                       core, jetty, static_cast<unsigned long long>(status.result),
                       static_cast<unsigned long long>(status.profile.jetty_completion_polls[jetty]),
                       static_cast<unsigned long long>(status.profile.jetty_completion_wait_cycles[jetty]),
                       static_cast<unsigned long long>(status.profile.active_requests),
                       static_cast<unsigned long long>(status.profile.total_posts));
        }
      }
    }
  }
}

}  // namespace

struct DefaultAccessOperator::Impl {
  explicit Impl(AccessOperatorConfig init_config) : config(std::move(init_config)) {
    if (config.batch_put_npu_core_count == 0U) {
      config.batch_put_npu_core_count = config.npu_core_count;
    }
  }

  AccessOperatorConfig config;
  RemoteRegionDesc remote_region;
  KvafBatchGetTransport* transport{nullptr};
  BatchGetExecutionState execution;
  std::mutex batch_get_mutex;
  FusedBatchGetContext fused_context;
  KvafBatchGetLocalBuffer* fused_local_buffer{nullptr};
  bool fused_completion_uncertain{false};
  std::uint64_t next_fused_lease_id{1};
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
  KvafBatchPutTransport* batch_put_transport{nullptr};
  BatchPutExecutionState batch_put_execution;
#endif
  bool initialized{false};
  bool initialization_failed{false};
};

DefaultAccessOperator::DefaultAccessOperator(AccessOperatorConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

DefaultAccessOperator::~DefaultAccessOperator() {
  if (!Shutdown().ok()) {
    // A borrowed device context may still be referenced by a running kernel
    // or late DMA. Leak the owning state rather than deregister live memory.
    std::fprintf(stderr, "[KVAF_AIV] FUSED_CONTEXT_RETAINED_ON_DESTRUCTION "
                         "reason=active_or_failed_lease\n");
    (void)impl_.release();
  }
}

Status DefaultAccessOperator::Initialize(const RemoteRegionDesc& remote_region) {
  std::lock_guard<std::mutex> lock(impl_->batch_get_mutex);
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
  if (!impl_->config.local_host_ddr && impl_->config.peer_link == nullptr) {
    return InvalidState("AccessOperator requires an established Runtime Link");
  }
  if (!impl_->config.local_host_ddr && std::all_of(impl_->config.peer_local_eid.begin(),
                  impl_->config.peer_local_eid.end(),
                  [](std::uint8_t value) { return value == 0U; })) {
    return InvalidArgument("Runtime Link requires a Runtime-selected local EID");
  }
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
  if (!impl_->config.batch_put_kernel_object_path.empty() &&
      (impl_->config.batch_put_npu_core_count == 0U ||
       impl_->config.batch_put_npu_core_count > KVAF_AIV_BATCH_PUT_MAX_AIV_CORES ||
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
  const int32_t result = impl_->config.local_host_ddr
      ? KvafBatchGetTransportCreateLocalHost(
            reinterpret_cast<void*>(remote_region.base), remote_region.length,
            impl_->config.npu_core_count, impl_->config.jettys_per_core, &transport)
      : KvafBatchGetTransportCreateWithPeerLink(
            reinterpret_cast<void*>(remote_region.base), remote_region.length,
            impl_->config.npu_core_count, impl_->config.jettys_per_core,
            impl_->config.peer_link.get(), impl_->config.peer_local_eid.data(), &transport);
  if (transport != nullptr) {
    impl_->transport = transport;
    impl_->remote_region = remote_region;
    // Publish ownership before a later setup step can fail. Runtime rolls
    // back through checked Shutdown and retains DDR if retirement fails.
    impl_->initialized = true;
    impl_->initialization_failed = true;
  }
  if (result != KVAF_BATCH_GET_SUCCESS || transport == nullptr) {
    return Internal("failed to create reusable BatchGet transport");
  }

  BatchGetExecutionState& execution = impl_->execution;
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
    return Internal("failed to allocate internal BatchGet execution state");
  }
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
  if (!impl_->config.batch_put_kernel_object_path.empty()) {
    if (KvafExecuteBatchPutLoadKernel(
            impl_->config.batch_put_kernel_object_path.c_str()) !=
            KVAF_BATCH_PUT_SUCCESS) {
      return Internal("failed to load BatchPut kernel");
    }

    KvafBatchPutTransport* batch_put_transport = nullptr;
    const int32_t batch_put_result = impl_->config.local_host_ddr
        ? KvafBatchPutTransportCreateLocalHost(
              reinterpret_cast<void*>(remote_region.base), remote_region.length,
              impl_->config.batch_put_npu_core_count, impl_->config.jettys_per_core, &batch_put_transport)
        : KvafBatchPutTransportCreateWithPeerLink(
              reinterpret_cast<void*>(remote_region.base), remote_region.length,
              impl_->config.batch_put_npu_core_count, impl_->config.jettys_per_core,
              impl_->config.peer_link.get(), impl_->config.peer_local_eid.data(), &batch_put_transport);
    impl_->batch_put_transport = batch_put_transport;
    if (batch_put_result != KVAF_BATCH_PUT_SUCCESS ||
        batch_put_transport == nullptr) {
      return Internal("failed to create reusable BatchPut transport");
    }
    BatchPutExecutionState batch_put_execution;
    Status batch_put_status = InitializeBatchPutExecutionState(
        impl_->config.batch_put_npu_core_count, &batch_put_execution);
    if (!batch_put_status.ok()) {
      return batch_put_status;
    }
    impl_->batch_put_transport = batch_put_transport;
    impl_->batch_put_execution = std::move(batch_put_execution);
    std::fprintf(stderr,
                 "[KVAF_AIV] RUNTIME_BATCH_PUT_TRANSPORT_READY cores=%u "
                 "jettys_per_core=%u "
                 "metadata_staging=reusable\n",
                 impl_->config.batch_put_npu_core_count,
                 impl_->config.jettys_per_core);
  }
#endif
  impl_->initialized = true;
  impl_->initialization_failed = false;
  return Status::OK();
}

Status DefaultAccessOperator::Shutdown() {
  std::lock_guard<std::mutex> lock(impl_->batch_get_mutex);
  if (!impl_->initialized) {
    return Status::OK();
  }
  if (impl_->fused_context.lease_id != 0U) {
    return Status::Busy("release the fused BatchGet context before shutdown; "
                        "a failed release requires device/process teardown");
  }

  // Callers must complete all submitted streams before Runtime::Shutdown,
  // including the default enqueue-only Python BatchGet path.
  // A failed retirement can leave only part of the transport graph alive.
  // Reject new launches until checked Shutdown has released it completely.
  impl_->initialization_failed = true;
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
  if (impl_->batch_put_transport != nullptr &&
      KvafBatchPutTransportDestroyChecked(impl_->batch_put_transport) != KVAF_BATCH_PUT_SUCCESS) {
    return Internal("BatchPut transport retirement failed; retain Runtime and DDR");
  }
  ReleaseBatchPutExecutionState(&impl_->batch_put_execution);
  impl_->batch_put_transport = nullptr;
#endif
  if (impl_->transport != nullptr &&
      KvafBatchGetTransportDestroyChecked(impl_->transport) != KVAF_BATCH_GET_SUCCESS) {
    return Internal("BatchGet transport retirement failed; retain Runtime and DDR");
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
  impl_->initialization_failed = false;
  return Status::OK();
}

Status DefaultAccessOperator::ExecuteBatchGet(const BatchGetExecRequest& exec) {
  std::lock_guard<std::mutex> lock(impl_->batch_get_mutex);
  if (!impl_->initialized || impl_->initialization_failed || impl_->transport == nullptr) {
    return Status::Error(StatusCode::kNotInitialized,
                         "AccessOperator has not been initialized");
  }
  if (impl_->fused_context.lease_id != 0U) {
    return Status::Busy("BatchGet queues are leased to a fused kernel");
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

Status DefaultAccessOperator::AcquireFusedBatchGetContext(
    void* output, std::uint64_t output_bytes, std::uint32_t token_bytes,
    std::uint32_t tokens_per_block, FusedBatchGetContext* context) {
  std::lock_guard<std::mutex> lock(impl_->batch_get_mutex);
  if (!impl_->initialized || impl_->initialization_failed || impl_->transport == nullptr) {
    return Status::NotInitialized("AccessOperator has not been initialized");
  }
  if (impl_->fused_context.lease_id != 0U) {
    return Status::Busy("BatchGet queues already have a fused lease");
  }
  if (context == nullptr || output == nullptr || output_bytes == 0U ||
      token_bytes == 0U || tokens_per_block == 0U ||
      output_bytes % token_bytes != 0U ||
      output_bytes > std::numeric_limits<std::uint64_t>::max() - PointerValue(output) ||
      impl_->next_fused_lease_id == 0U) {
    return InvalidArgument("invalid fused BatchGet output or context");
  }
  // Setup-only boundary: any previous enqueue-only ordinary BatchGet must
  // finish before its persistent queues are handed to the external kernel.
  if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
    return Internal("failed to drain device before acquiring fused BatchGet queues");
  }
  Status status = CheckBatchGetQueueHealth(
      impl_->execution, impl_->config.npu_core_count,
      impl_->config.jettys_per_core);
  if (!status.ok()) return status;

  KvafBatchGetLocalBuffer* local_buffer = nullptr;
  std::uint64_t entity = 0U;
  if (KvafBatchGetLocalBufferAcquire(impl_->transport, output, output_bytes,
                                     &local_buffer, &entity) !=
          KVAF_BATCH_GET_SUCCESS || local_buffer == nullptr || entity == 0U) {
    if (local_buffer != nullptr) KvafBatchGetLocalBufferRelease(local_buffer);
    return Internal("failed to register fused BatchGet output");
  }
  impl_->fused_context = FusedBatchGetContext{
      impl_->next_fused_lease_id++, entity,
      PointerValue(impl_->execution.queue_state),
      PointerValue(impl_->execution.status), PointerValue(output), output_bytes,
      impl_->config.npu_core_count, impl_->config.jettys_per_core,
      token_bytes, tokens_per_block};
  impl_->fused_local_buffer = local_buffer;
  *context = impl_->fused_context;
  return Status::OK();
}

Status DefaultAccessOperator::ReleaseFusedBatchGetContext(
    const FusedBatchGetContext& context, void* stream) {
  std::lock_guard<std::mutex> lock(impl_->batch_get_mutex);
  if (impl_->fused_context.lease_id == 0U ||
      context.lease_id != impl_->fused_context.lease_id ||
      context.device_channel_entity != impl_->fused_context.device_channel_entity ||
      context.device_queue_state != impl_->fused_context.device_queue_state ||
      context.output_address != impl_->fused_context.output_address ||
      context.output_bytes != impl_->fused_context.output_bytes) {
    return InvalidArgument("fused BatchGet context does not own the current lease");
  }
  if (stream == nullptr) return InvalidArgument("fused release requires its launch stream");
  if (impl_->fused_completion_uncertain) {
    return Internal("fused BatchGet context is quarantined after a failed completion check");
  }
  if (aclrtSynchronizeStream(static_cast<aclrtStream>(stream)) != ACL_SUCCESS) {
    impl_->fused_completion_uncertain = true;
    return Internal("fused BatchGet stream did not complete; retain output and context");
  }
  Status status = CheckBatchGetQueueHealth(
      impl_->execution, impl_->config.npu_core_count,
      impl_->config.jettys_per_core);
  if (!status.ok()) {
    impl_->fused_completion_uncertain = true;
    return status;
  }
  KvafBatchGetLocalBufferRelease(impl_->fused_local_buffer);
  impl_->fused_local_buffer = nullptr;
  impl_->fused_context = FusedBatchGetContext{};
  return Status::OK();
}

Status DefaultAccessOperator::SetBatchGetKernelProfiling(bool enabled) {
  std::lock_guard<std::mutex> lock(impl_->batch_get_mutex);
  if (!impl_->initialized || impl_->initialization_failed || impl_->execution.status == nullptr) {
    return Status::Error(StatusCode::kNotInitialized,
                         "AccessOperator has not been initialized");
  }
  if (impl_->fused_context.lease_id != 0U) {
    return Status::Busy("cannot reset BatchGet status while a fused lease is active");
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

Status DefaultAccessOperator::CopyBatchGetCqDiagnostics(
    std::vector<BatchGetCqDiagnostic>* diagnostics) const {
  std::lock_guard<std::mutex> lock(impl_->batch_get_mutex);
  if (diagnostics == nullptr) {
    return InvalidArgument("BatchGet CQ diagnostic output must not be null");
  }
  if (!impl_->initialized || impl_->initialization_failed || impl_->execution.queue_state == nullptr) {
    return Status::NotInitialized("AccessOperator has not been initialized");
  }
  // The caller owns stream synchronization. In particular this read does not
  // release or reacquire a fused lease and does not run inside a timed graph.
  return CollectBatchGetCqDiagnostics(impl_->execution, impl_->config.npu_core_count,
                                     impl_->config.jettys_per_core, diagnostics);
}

Status DefaultAccessOperator::CopyBatchGetKernelProfiles(
    std::vector<BatchGetKernelProfile>* profiles) const {
  if (profiles == nullptr) {
    return InvalidArgument("BatchGet kernel profile output must not be null");
  }
  if (!impl_->initialized || impl_->initialization_failed || impl_->execution.status == nullptr) {
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
  if (!impl_->initialized || impl_->initialization_failed) {
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

  if (exec.request->input.device == DeviceKind::kCpu) {
    std::lock_guard<std::mutex> lock(impl_->batch_get_mutex);
    if (!impl_->config.local_host_ddr || impl_->remote_region.device != DeviceKind::kCpu) {
      return InvalidArgument("CPU seed Put requires this Runtime's local Host DDR");
    }
    if (impl_->fused_context.lease_id != 0U) {
      return Status::Busy("cannot seed Host DDR while BatchGet queues have a fused lease");
    }
    const auto& region = impl_->remote_region;
    const auto& request = *exec.request;
    if (region.base == 0U || region.length == 0U ||
        region.base > std::numeric_limits<std::uint64_t>::max() - region.length ||
        PointerValue(request.input.data) > std::numeric_limits<std::uint64_t>::max() - request.input.bytes) {
      return InvalidArgument("CPU seed Put source or owned DDR range is invalid");
    }
    // Validate every row before touching any destination. Runtime commits
    // metadata only after this synchronous copy and its release fence.
    for (std::size_t index = 0; index < request.keys.size(); ++index) {
      const auto offset = request.input_offsets[index];
      const auto length = request.lengths[index];
      const auto& destination = exec.put_desc->entries[index];
      if (length == 0U || offset > request.input.bytes || length > request.input.bytes - offset ||
          destination.key != request.keys[index] || destination.reserved_length != length ||
          destination.address < region.base || destination.address - region.base > region.length ||
          length > region.length - (destination.address - region.base)) {
        return InvalidArgument("CPU seed Put source/destination escapes its validated blocks or owned DDR");
      }
    }
    const auto* source = static_cast<const std::uint8_t*>(request.input.data);
    for (std::size_t index = 0; index < request.keys.size(); ++index) {
      std::memmove(reinterpret_cast<void*>(exec.put_desc->entries[index].address),
                   source + request.input_offsets[index], request.lengths[index]);
    }
    std::atomic_thread_fence(std::memory_order_release);
    return Status::OK();
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
    arguments.aiv_core_count = impl_->config.batch_put_npu_core_count;

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
      LogBatchPutDeviceStatus(execution, impl_->config.batch_put_npu_core_count, result);
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
