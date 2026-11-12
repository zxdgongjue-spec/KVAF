#include "aiv_execution_resources.h"

#include <algorithm>
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
#include "get_host_timing.h"

#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
#include "batch_put_abi.h"
#include "direct_hccp/direct_hccp_session.h"
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

#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
void LogBatchPutDeviceStatus(const AivPutWorkspace& execution,
                             std::uint32_t npu_core_count,
                             int32_t result) {
  std::fprintf(stderr, "[KVAF_AIV] BATCH_PUT_EXECUTION_ERROR result=%d\n",
               static_cast<int>(result));
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
                               std::uint32_t max_inflight_total,
                               const AivGetWorkspace& execution,
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
  arguments->batch_get_max_inflight_total = max_inflight_total;
  return Status::OK();
}

void LogBatchGetDeviceStatus(const AivGetWorkspace& execution,
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

DefaultAccessOperator::DefaultAccessOperator(AccessOperatorConfig config)
    : owned_resources_(std::make_unique<AivExecutionResources>(std::move(config))),
      resources_(owned_resources_.get()) {}
DefaultAccessOperator::~DefaultAccessOperator() = default;
std::unique_ptr<ExecutionResourceOwner> DefaultAccessOperator::TakeResources() {
  return std::move(owned_resources_);
}
std::unique_ptr<ExecutionState> DefaultAccessOperator::CreateExecutionState() { return resources_->CreateState(); }
Status DefaultAccessOperator::BindExecutionState(ExecutionState* state) { return resources_->Bind(state); }
Status DefaultAccessOperator::Initialize(const RemoteRegionDesc& region) { return resources_->Initialize(region); }
Status DefaultAccessOperator::Shutdown() { return resources_->Close(); }
Status DefaultAccessOperator::Warmup(const ExecutionWarmup& request) { return resources_->Warmup(request); }
Status DefaultAccessOperator::PrepareGetGraph(const GetGraphOptions& options) {
  auto& call = resources_->invocation();
  auto status = UrmaAivGetOperator::PrepareGraph(options, resources_->GetView(), call);
  return status.ok() ? resources_->PrepareGraph(options, call) : status;
}
Status DefaultAccessOperator::FinishGetGraph(void* stream, bool discard) {
  return UrmaAivGetOperator::FinishGraph(stream, discard, resources_->GetView(), resources_->invocation());
}
Status DefaultAccessOperator::ExecuteBatchGet(const BatchGetExecRequest& request) {
  const auto prepare_started = GetHostTimingStart(request.host_timing);
  auto& call = resources_->invocation();
  auto view = resources_->GetView();
  if (view.options.managed_completion && !call.graph_buffer) {
    auto status = UrmaAivGetOperator::Drain(view, call);
    if (!status.ok()) return status;
  }
  auto status = UrmaAivGetOperator::Validate(request, view, call);
  if (status.ok()) status = resources_->PrepareGetResources(request, call, &view);
  GetHostTimingRecord(request.host_timing, &GetProviderTiming::prepare_ns, prepare_started);
  return status.ok() ? UrmaAivGetOperator::Launch(request, view, call) : status;
}
Status DefaultAccessOperator::DrainGet() {
  return UrmaAivGetOperator::Drain(resources_->GetView(), resources_->invocation());
}
Status DefaultAccessOperator::PrepareGetOutput(const GetOutputOptions& options) {
  aclmdlRICaptureStatus capture = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
  aclmdlRI graph = nullptr;
  if (!options.stream || aclmdlRICaptureGetInfo(static_cast<aclrtStream>(options.stream), &capture, &graph) != ACL_SUCCESS)
    return InvalidArgument("output binding requires a valid stream");
  if (capture != ACL_MODEL_RI_CAPTURE_STATUS_NONE)
    return Status::InvalidState("bind Get output before graph capture");
  auto s = DrainGet();
  if (s.ok()) s = resources_->Warmup({options.request_capacity, options.stream});
  return s.ok() ? resources_->PrepareOutput(options, resources_->invocation()) : s;
}
Status DefaultAccessOperator::FinishGetOutput() {
  auto s = DrainGet();
  if (!s.ok()) return s;
  auto& call = resources_->invocation();
  if (!call.output_buffer) return Status::OK();
  // Retire can unregister the MR before a later token/entity release fails.
  // That partially retired binding must never be used by another Get.
  call.output_retiring = true;
  if (KvafBatchGetLocalBufferRetire(call.output_buffer) != KVAF_BATCH_GET_SUCCESS)
    return Internal("cannot retire bound Get output; binding retained");
  call.output_buffer = nullptr;
  call.output_entity = 0;
  call.output_options = {};
  call.output_retiring = false;
  return Status::OK();
}
Status DefaultAccessOperator::SetBatchGetKernelProfiling(bool enabled) {
  return UrmaAivGetOperator::SetProfiling(enabled, resources_->GetView());
}
Status DefaultAccessOperator::CopyBatchGetKernelProfiles(std::vector<BatchGetKernelProfile>* profiles) const {
  return UrmaAivGetOperator::CopyProfiles(profiles, resources_->GetView());
}
Status DefaultAccessOperator::ExecuteBatchPut(const BatchPutExecRequest& request) {
  auto& call = resources_->invocation();
  auto view = resources_->PutView();
  auto status = UrmaAivPutOperator::Validate(request, view, call);
  if (!status.ok()) {
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
    if (!call.pending_put)
#endif
      call.put_completion = {PutOutcome::Failed, status};
    return status;
  }
  status = resources_->PreparePutResources(request, call, &view);
  return status.ok() ? UrmaAivPutOperator::Launch(request, view, call) : status;
}
Status DefaultAccessOperator::ResolvePut(PutCompletion* out) {
  return UrmaAivPutOperator::Resolve(out, resources_->PutView(), resources_->invocation());
}
Status DefaultAccessOperator::AcknowledgePut() {
  return UrmaAivPutOperator::Acknowledge(resources_->PutView(), resources_->invocation());
}

Status UrmaAivGetOperator::PrepareGraph(const GetGraphOptions& options, UrmaAivGetView view, AivInvocation& invocation) {
  if (!view.common.Validate(&invocation, ExecutionDirection::Get).ok())
    return InvalidArgument("AIV invocation does not match resource view");
  if (!view.common.ready() || invocation.graph_buffer)
    return Status::InvalidState("AIV graph provider is unavailable or leased");
  if (!options.stream || !options.output.valid() || options.output.device != DeviceKind::kNpu ||
      !options.request_capacity)
    return Status::InvalidArgument("graph preparation requires output, capacity and stream");
  aclmdlRICaptureStatus capture = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
  aclmdlRI graph = nullptr;
  if (aclmdlRICaptureGetInfo(static_cast<aclrtStream>(options.stream), &capture, &graph) !=
      ACL_SUCCESS)
    return Internal("cannot query graph capture state");
  if (capture != ACL_MODEL_RI_CAPTURE_STATUS_NONE)
    return Status::InvalidState("prepare graph resources before capture");
  auto s = Drain(view, invocation);
  if (!s.ok()) return s;
  return Status::OK();
}
Status UrmaAivGetOperator::FinishGraph(void* stream, bool discard, UrmaAivGetView view, AivInvocation& invocation) {
  if (!view.common.Validate(&invocation, ExecutionDirection::Get).ok())
    return InvalidArgument("AIV invocation does not match resource view");
  if (!invocation.graph_buffer) return Status::NotFound("AIV graph lease not found");
  if (!stream) return InvalidArgument("graph release requires its last replay stream");
  aclmdlRICaptureStatus capture = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
  aclmdlRI graph = nullptr;
  if (aclmdlRICaptureGetInfo(static_cast<aclrtStream>(stream), &capture, &graph) != ACL_SUCCESS)
    return Internal("cannot query graph capture state");
  if (capture != ACL_MODEL_RI_CAPTURE_STATUS_NONE)
    return Status::InvalidState("finish capture before releasing graph");
  if (aclrtSynchronizeStream(static_cast<aclrtStream>(stream)) != ACL_SUCCESS)
    return Status::CompletionUnknown("graph replay completion unresolved");
  if (invocation.graph_recorded && !discard) {
    std::vector<KvafBatchGetCoreStatus> reports(view.options.npu_core_count);
    if (aclrtMemcpy(reports.data(), view.workspace.status_bytes, view.workspace.status,
                    view.workspace.status_bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
      return Internal("cannot collect graph completion");
    for (const auto& r : reports)
      if (r.magic != KVAF_AIV_BATCH_GET_STATUS_MAGIC || r.result)
        return Internal(
            "graph result failed or graph was never replayed; inspect then release with "
            "discard_result=True");
  }
  if (KvafBatchGetLocalBufferRetire(invocation.graph_buffer) != KVAF_BATCH_GET_SUCCESS)
    return Internal("cannot retire graph registration");
  invocation.graph_buffer = nullptr;
  invocation.graph_entity = 0;
  invocation.graph_options = {};
  invocation.graph_recorded = false;
  return Status::OK();
}

Status UrmaAivGetOperator::Validate(const BatchGetExecRequest& exec, UrmaAivGetView view, AivInvocation& invocation) {
  if (!view.common.Validate(&invocation, ExecutionDirection::Get).ok())
    return InvalidArgument("AIV invocation does not match resource view");
  if (!view.common.ready()) {
    return Status::Error(StatusCode::kNotInitialized,
                         "AccessOperator has not been initialized");
  }
  if (exec.execution.stream == nullptr) {
    return InvalidArgument("NPU BatchGet requires a caller-owned ACL stream");
  }
  const bool graph_lease = invocation.graph_buffer != nullptr;
  if (graph_lease) {
    if (!exec.request || exec.request->output.data != invocation.graph_options.output.data ||
        exec.request->output.bytes != invocation.graph_options.output.bytes ||
        exec.execution.stream != invocation.graph_options.stream ||
        exec.request->indices.bytes / sizeof(std::int32_t) >
            invocation.graph_options.request_capacity ||
        exec.execution.synchronize)
      return InvalidArgument("captured request differs from prepared graph lease");
    aclmdlRICaptureStatus capture = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    aclmdlRI graph = nullptr;
    if (aclmdlRICaptureGetInfo(static_cast<aclrtStream>(exec.execution.stream), &capture, &graph) !=
        ACL_SUCCESS)
      return Internal("cannot query graph capture state");
    if (capture != ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE)
      return Status::InvalidState("graph lease submissions require active capture");
  }
  if (view.options.managed_completion && !graph_lease) {
    aclmdlRICaptureStatus capture = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    aclmdlRI graph = nullptr;
    if (aclmdlRICaptureGetInfo(static_cast<aclrtStream>(exec.execution.stream), &capture, &graph) != ACL_SUCCESS)
      return Internal("cannot query graph capture state");
    if (capture == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE)
      return Status::InvalidState("call begin_get_graph before capturing a request-level Get");
  }
  KvafExecuteBatchGetKernelArgs arguments{};
  Status status = BuildBatchGetKernelArgs(
      exec, view.options.npu_core_count, view.options.jettys_per_core,
      view.options.batch_get_max_inflight_total, view.workspace, &arguments);
  if (!status.ok()) {
    return status;
  }

  return Status::OK();
}

Status UrmaAivGetOperator::Launch(const BatchGetExecRequest& exec, UrmaAivGetView view, AivInvocation& invocation) {
  const auto submit_started = GetHostTimingStart(exec.host_timing);
  auto status = Validate(exec, view, invocation);
  if (!status.ok()) return status;
  const bool graph_lease = invocation.graph_buffer != nullptr;
  if (!graph_lease && invocation.get_issued)
    return Status::Busy("Get invocation is still in flight");
  KvafExecuteBatchGetKernelArgs arguments{};
  status = BuildBatchGetKernelArgs(exec, view.options.npu_core_count,
      view.options.jettys_per_core, view.options.batch_get_max_inflight_total,
      view.workspace, &arguments);
  if (!status.ok()) return status;
  KvafBatchGetLocalBuffer* local_buffer = view.buffer;
  const std::uint64_t channel_entity = view.channel_entity;
  const aclrtStream stream = static_cast<aclrtStream>(exec.execution.stream);
  if (!local_buffer || !channel_entity ||
      local_buffer != (graph_lease ? invocation.graph_buffer : invocation.pending_buffer) ||
      (!graph_lease && (exec.request->output.data != invocation.get_output.data ||
                       exec.request->output.bytes != invocation.get_output.bytes ||
                       exec.execution.stream != invocation.pending_get ||
                       exec.request->indices.bytes / sizeof(std::int32_t) > invocation.get_capacity)))
    return InvalidState("AIV Get registration must be prepared before Launch");
  arguments.channel_entity = channel_entity;

  if (view.options.managed_completion) {
    if (!graph_lease) {
      invocation.pending_get = stream;
      invocation.pending_buffer = local_buffer;
      invocation.get_issued = false;
    }
    if (aclrtMemsetAsync(view.workspace.status, view.workspace.status_bytes, 0xFF,
                         view.workspace.status_bytes, stream) != ACL_SUCCESS)
      return Status::CompletionUnknown("failed to initialize Get completion status");
  }
  if (graph_lease) invocation.graph_recorded = true;
  else invocation.get_issued = true;
  const int32_t launch_result = KvafExecuteBatchGetLaunch(&arguments, stream);
  if (launch_result != KVAF_BATCH_GET_SUCCESS) {
    if (!view.options.managed_completion && !graph_lease)
      { KvafBatchGetLocalBufferRelease(local_buffer); invocation.pending_buffer = nullptr;
        invocation.pending_get = nullptr; invocation.get_entity = 0; }
    return Internal("failed to launch BatchGet");
  }
  GetHostTimingRecord(exec.host_timing, &GetProviderTiming::submit_ns, submit_started);
  const auto completion_started = GetHostTimingStart(exec.host_timing);
  if (graph_lease) return Status::OK();
  if (view.options.managed_completion) {
    auto completed = exec.execution.synchronize ? Drain(view, invocation) : Status::OK();
    GetHostTimingRecord(exec.host_timing, &GetProviderTiming::completion_ns, completion_started);
    return completed;
  }
  if (exec.execution.synchronize) {
    const aclError sync_result = aclrtSynchronizeStream(stream);
    KvafBatchGetLocalBufferRelease(local_buffer);
    invocation.pending_buffer = nullptr; invocation.pending_get = nullptr; invocation.get_entity = 0;
    if (sync_result != ACL_SUCCESS) {
      return Internal("BatchGet stream synchronization failed");
    }
    LogBatchGetDeviceStatus(view.workspace, view.options.npu_core_count);
  } else {
    // The demo transport caches zero-reference registrations until shutdown.
    // Releasing after an externally captured launch keeps KVAF graph-agnostic;
    // the caller owns stream completion and tensor lifetime.
    KvafBatchGetLocalBufferRelease(local_buffer);
    invocation.pending_buffer = nullptr; invocation.pending_get = nullptr; invocation.get_entity = 0;
  }
  GetHostTimingRecord(exec.host_timing, &GetProviderTiming::completion_ns, completion_started);
  return Status::OK();
}

Status UrmaAivGetOperator::Drain(UrmaAivGetView view, AivInvocation& invocation) {
  if (!view.common.Validate(&invocation, ExecutionDirection::Get).ok())
    return InvalidArgument("AIV invocation does not match resource view");
  if (!view.common.ready()) return Status::CompletionUnknown("AIV transport is closing; retry Shutdown");
  if (!invocation.pending_get) return Status::OK();
  if (aclrtSynchronizeStream(invocation.pending_get) != ACL_SUCCESS)
    return Status::CompletionUnknown("BatchGet stream completion failed; resources retained");
  if (invocation.get_issued) {
    std::vector<KvafBatchGetCoreStatus> reports(view.options.npu_core_count);
    if (aclrtMemcpy(reports.data(), view.workspace.status_bytes, view.workspace.status,
                    view.workspace.status_bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
      return Internal("failed to collect BatchGet completion status");
    for (const auto& r : reports)
      if (r.magic != KVAF_AIV_BATCH_GET_STATUS_MAGIC || r.result != 0)
        return Internal("BatchGet kernel reported an execution error; session retained");
  }
  if (invocation.pending_buffer != invocation.output_buffer &&
      KvafBatchGetLocalBufferRetire(invocation.pending_buffer) != KVAF_BATCH_GET_SUCCESS)
    return Internal("failed to retire Get registration; resources retained");
  invocation.pending_buffer = nullptr;
  invocation.pending_get = nullptr;
  invocation.get_issued = false;
  invocation.get_entity = 0;
  return Status::OK();
}

Status UrmaAivGetOperator::SetProfiling(bool enabled, UrmaAivGetView view) {
  if (!view.common.ready() || view.workspace.status == nullptr) {
    return Status::Error(StatusCode::kNotInitialized,
                         "AccessOperator has not been initialized");
  }
  std::vector<KvafBatchGetCoreStatus> statuses(
      view.options.npu_core_count);
  if (enabled) {
    for (KvafBatchGetCoreStatus& status : statuses) {
      status.profile.magic = KVAF_AIV_BATCH_GET_PROFILE_MAGIC;
    }
  }
  if (aclrtMemcpy(view.workspace.status, view.workspace.status_bytes,
                  statuses.data(), statuses.size() * sizeof(statuses.front()),
                  ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
    return Internal("failed to configure BatchGet kernel profiling");
  }
  return Status::OK();
}

Status UrmaAivGetOperator::CopyProfiles(
    std::vector<BatchGetKernelProfile>* profiles, UrmaAivGetView view) {
  if (profiles == nullptr) {
    return InvalidArgument("BatchGet kernel profile output must not be null");
  }
  if (!view.common.ready() || view.workspace.status == nullptr) {
    return Status::Error(StatusCode::kNotInitialized,
                         "AccessOperator has not been initialized");
  }
  std::vector<KvafBatchGetCoreStatus> statuses(
      view.options.npu_core_count);
  if (aclrtMemcpy(statuses.data(), statuses.size() * sizeof(statuses.front()),
                  view.workspace.status, view.workspace.status_bytes,
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
    destination.submitted_group_count = source.submitted_group_count;
    destination.completed_group_count = source.completed_group_count;
    destination.completion_count = source.completion_count;
    destination.post_active_cycles = source.post_active_cycles;
    destination.drain_wait_cycles = source.drain_wait_cycles;
    destination.peak_submitted_not_drained = source.peak_submitted_not_drained;
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

Status UrmaAivPutOperator::Launch(const BatchPutExecRequest& exec, UrmaAivPutView view, AivInvocation& invocation) {
  if (!view.common.Validate(&invocation, ExecutionDirection::Put).ok())
    return InvalidArgument("AIV invocation does not match resource view");
  auto valid = Validate(exec, view, invocation);
  if (!valid.ok()) return valid;
  auto& call = invocation;
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
  if (call.put_issued && call.pending_put)
    return Status::Busy("Put invocation is still in flight");
#endif
  call.put_completion = {};
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
  call.put_issued = false;
#endif
  auto result = Execute(exec, view, invocation);
  if (result.ok()) call.put_completion = {PutOutcome::Succeeded, result};
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
  else if (!call.pending_put)
    call.put_completion = {PutOutcome::Failed, result};
  else call.put_completion.result = result;
#else
  else call.put_completion = {PutOutcome::Failed, result};
#endif
  return result;
}
Status UrmaAivPutOperator::Resolve(PutCompletion* out, UrmaAivPutView view, AivInvocation& invocation) {
  if (!view.common.Validate(&invocation, ExecutionDirection::Put).ok())
    return InvalidArgument("AIV invocation does not match resource view");
  if (!out) return InvalidArgument("null Put completion output");
  if (!view.common.ready()) return Status::CompletionUnknown("AIV transport is closing; retry Shutdown");
  auto& call = invocation;
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
  if (call.pending_put) {
    if (aclrtSynchronizeStream(call.pending_put) != ACL_SUCCESS)
      return Status::CompletionUnknown("Put stream still pending");
    if (call.put_issued) {
      auto& workspace = view.workspace;
      const auto bytes = workspace.host_core_status_count * sizeof(KvafBatchPutCoreStatus);
      if (aclrtMemcpy(workspace.host_core_status, bytes, workspace.core_status, bytes,
                      ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
        return Status::CompletionUnknown("Put completion report unavailable");
      for (std::size_t i = 0; i < workspace.host_core_status_count; ++i) {
        const auto& report = workspace.host_core_status[i];
        const auto expected = call.put_rows * (i + 1) / workspace.host_core_status_count -
                              call.put_rows * i / workspace.host_core_status_count;
        if (report.magic != KVAF_AIV_BATCH_PUT_STATUS_MAGIC || report.result != 0 ||
            report.processed_request_count != expected || report.failed_request_index != UINT64_MAX)
          return Status::CompletionUnknown("Put reports do not prove all remote WRs completed");
      }
    }
    if (KvafBatchPutLocalBufferRetire(call.pending_put_buffer) != KVAF_BATCH_PUT_SUCCESS)
      return Status::CompletionUnknown("Put registration retirement still pending");
    call.pending_put_buffer = nullptr;
    call.pending_put = nullptr;
    call.put_completion.outcome = call.put_issued ? PutOutcome::Succeeded : PutOutcome::Failed;
    if (call.put_issued) call.put_completion.result = Status::OK();
  }
#endif
  *out = call.put_completion;
  return Status::OK();
}
Status UrmaAivPutOperator::Acknowledge(UrmaAivPutView view, AivInvocation& invocation) {
  if (!view.common.Validate(&invocation, ExecutionDirection::Put).ok())
    return InvalidArgument("AIV invocation does not match resource view");
  if (invocation.put_completion.outcome == PutOutcome::Unknown)
    return Status::CompletionUnknown("cannot acknowledge unresolved Put");
  invocation.put_completion = {};
  return Status::OK();
}

Status UrmaAivPutOperator::Validate(const BatchPutExecRequest& exec, UrmaAivPutView view, AivInvocation& invocation) {
  if (!view.common.Validate(&invocation, ExecutionDirection::Put).ok())
    return InvalidArgument("AIV invocation does not match resource view");
  if (!view.common.ready()) {
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

  return Status::OK();
}

Status UrmaAivPutOperator::Execute(const BatchPutExecRequest& exec, UrmaAivPutView view, AivInvocation& invocation) {
  auto valid = Validate(exec, view, invocation);
  if (!valid.ok()) return valid;
  const auto* input = static_cast<const std::uint8_t*>(exec.request->input.data);
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
  if (view.device_execution) {
    if (exec.execution.stream == nullptr) {
      return InvalidArgument("AIV BatchPut requires a caller-owned ACL stream");
    }
    AivPutWorkspace& execution = view.workspace;
    const std::size_t rows = exec.request->keys.size();
    invocation.put_rows = rows;
    if (!FitsU32(rows)) {
      return InvalidArgument("BatchPut row count exceeds the device ABI");
    }
    if (rows > execution.metadata_capacity || rows > execution.host_metadata_capacity)
      return InvalidState("AIV Put workspace must be reserved before Launch");
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

    const std::size_t metadata_bytes = rows * 3U * sizeof(std::uint64_t);

    const aclrtStream stream = static_cast<aclrtStream>(exec.execution.stream);
    KvafBatchPutLocalBuffer* local_buffer = view.buffer;
    if (!local_buffer || local_buffer != invocation.pending_put_buffer ||
        exec.request->input.data != invocation.put_input.data ||
        exec.request->input.bytes != invocation.put_input.bytes ||
        exec.execution.stream != invocation.pending_put)
      return InvalidState("AIV Put registration must be prepared before Launch");

    if (view.options.managed_completion) {
      invocation.pending_put = stream;
      invocation.pending_put_buffer = local_buffer;
    }
    const auto status_bytes = execution.host_core_status_count * sizeof(KvafBatchPutCoreStatus);
    if (aclrtMemset(execution.core_status, status_bytes, 0, status_bytes) != ACL_SUCCESS) {
      if (!view.options.managed_completion) {
        KvafBatchPutLocalBufferRelease(local_buffer);
        invocation.pending_put = nullptr; invocation.pending_put_buffer = nullptr;
      }
      return Internal("failed to reset Put completion status");
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
    arguments.aiv_core_count = view.options.npu_core_count;

    std::memset(execution.host_core_status, 0,
                execution.host_core_status_count *
                    sizeof(KvafBatchPutCoreStatus));
    if (aclrtMemcpyAsync(execution.metadata, metadata_bytes,
                         execution.host_metadata, metadata_bytes,
                         ACL_MEMCPY_HOST_TO_DEVICE, stream) != ACL_SUCCESS) {
      if (!view.options.managed_completion) {
        KvafBatchPutLocalBufferRelease(local_buffer);
        invocation.pending_put = nullptr; invocation.pending_put_buffer = nullptr;
      }
      return Status::CompletionUnknown("failed to enqueue BatchPut metadata upload to NPU");
    }

    if (view.options.managed_completion) {
      invocation.pending_put = stream;
      invocation.pending_put_buffer = local_buffer;
    }
    invocation.put_issued = true;
    const int32_t result = KvafExecuteBatchPutSync(
        local_buffer, &arguments, stream, execution.host_core_status,
        static_cast<std::uint32_t>(execution.host_core_status_count));
    if (result != KVAF_BATCH_PUT_SUCCESS) {
      // The metadata upload may have been accepted even when launch or status
      // setup subsequently failed. Do not reuse its pinned source buffer until
      // the stream has stopped reading it.
      (void)aclrtSynchronizeStream(stream);
    }
    if (view.options.managed_completion && result != KVAF_BATCH_PUT_SUCCESS)
      return Status::CompletionUnknown("AIV BatchPut execution failed; session retained for shutdown");
    if (view.options.managed_completion) {
      if (KvafBatchPutLocalBufferRetire(local_buffer) != KVAF_BATCH_PUT_SUCCESS)
        return Status::CompletionUnknown("failed to retire Put registration; resources retained");
    } else KvafBatchPutLocalBufferRelease(local_buffer);
    invocation.pending_put = nullptr;
    invocation.pending_put_buffer = nullptr;
    if (result != KVAF_BATCH_PUT_SUCCESS) {
      LogBatchPutDeviceStatus(execution, view.options.npu_core_count, result);
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
