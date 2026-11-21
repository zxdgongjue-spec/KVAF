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
#include "put_host_timing.h"

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
Status ReleaseAivPutWorkspace(AivPutWorkspace* execution) {
  if (!execution) return Status::OK();
  if (execution->core_status) {
    if (aclrtFree(execution->core_status) != ACL_SUCCESS)
      return Status::CompletionUnknown("Put status release failed; allocation retained");
    execution->core_status = nullptr;
  }
  if (execution->metadata) {
    if (aclrtFree(execution->metadata) != ACL_SUCCESS)
      return Status::CompletionUnknown("Put metadata release failed; allocation retained");
    execution->metadata = nullptr;
  }
  if (execution->host_core_status) {
    if (aclrtFreeHost(execution->host_core_status) != ACL_SUCCESS)
      return Status::CompletionUnknown("Put host status release failed; allocation retained");
    execution->host_core_status = nullptr;
  }
  if (execution->host_metadata) {
    if (aclrtFreeHost(execution->host_metadata) != ACL_SUCCESS)
      return Status::CompletionUnknown("Put host metadata release failed; allocation retained");
    execution->host_metadata = nullptr;
  }
  *execution = AivPutWorkspace{};
  return Status::OK();
}

Status InitializeAivPutWorkspace(std::uint32_t npu_core_count,
                                        AivPutWorkspace* execution) {
  if (execution == nullptr || npu_core_count == 0) {
    return InvalidArgument("invalid BatchPut execution state");
  }
  auto& state = *execution;
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
    return Internal("failed to allocate internal BatchPut execution state");
  }
  return Status::OK();
}

Status EnsureBatchPutMetadataCapacity(std::size_t rows,
                                      AivPutWorkspace* execution) {
  if (execution == nullptr || rows == 0 ||
      rows > std::numeric_limits<std::size_t>::max() /
                 (3U * sizeof(std::uint64_t))) {
    return InvalidArgument("BatchPut metadata dimensions are invalid");
  }
  const std::size_t bytes = rows * 3U * sizeof(std::uint64_t);
  if (rows > execution->host_metadata_capacity) {
    if (execution->host_metadata) {
      if (aclrtFreeHost(execution->host_metadata) != ACL_SUCCESS)
        return Status::CompletionUnknown("old Put metadata release failed; allocation retained");
      execution->host_metadata = nullptr;
      execution->host_metadata_capacity = 0;
    }
    if (aclrtMallocHost(&execution->host_metadata, bytes) != ACL_SUCCESS ||
        !execution->host_metadata)
      return Internal("failed to grow pinned BatchPut host metadata buffer");
    execution->host_metadata_capacity = rows;
  }
  if (rows > execution->metadata_capacity) {
    if (execution->metadata) {
      if (aclrtFree(execution->metadata) != ACL_SUCCESS)
        return Status::CompletionUnknown("old Put device metadata release failed; allocation retained");
      execution->metadata = nullptr;
      execution->metadata_capacity = 0;
    }
    if (aclrtMalloc(&execution->metadata, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        !execution->metadata)
      return Internal("failed to grow BatchPut device metadata buffers");
    execution->metadata_capacity = rows;
  }
  return Status::OK();
}

#endif
}  // namespace
struct AivExecutionResources::State {
  State(AccessOperatorConfig value, const AivExecutionResources* owner)
      : config(std::move(value)), default_invocation(owner), active(&default_invocation) { views.Bind(active); }
  ExecutionViewState views;
  AccessOperatorConfig config;
  RemoteRegionDesc remote_region;
  AivGetSession get;
  AivPutSession put;
  bool initialized{false}, closing{false};
  // Keeps initialization and Close valid before binding and after unbinding.
  AivInvocation default_invocation;
  AivInvocation* active;
};
AivExecutionResources::AivExecutionResources(AccessOperatorConfig config)
    : state_(std::make_unique<State>(std::move(config), this)) {}
AivExecutionResources::~AivExecutionResources() {
  // Retain partial transport and workspace when teardown cannot prove safety.
  if (!Close().ok()) (void)state_.release();
}
UrmaAivGetView AivExecutionResources::GetView() {
  const auto& call = *state_->active;
  const auto& c = state_->config;
  auto workspace = state_->get.workspace;
  if (call.graph_report.device) workspace.status = call.graph_report.device;
  return {state_->views.View(ExecutionDirection::Get),
          {c.npu_core_count, c.jettys_per_core, c.batch_get_max_inflight_total},
          workspace,
          call.graph_buffer ? call.graph_buffer : call.pending_buffer,
          call.graph_buffer ? call.graph_entity : call.get_entity};
}
UrmaAivPutView AivExecutionResources::PutView() {
  const auto& c = state_->config;
  return {state_->views.View(ExecutionDirection::Put),
          {c.npu_core_count, c.jettys_per_core, c.batch_get_max_inflight_total}
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
          , state_->put.transport != nullptr, state_->put.workspace, state_->active->pending_put_buffer
#endif
  };
}
std::unique_ptr<ExecutionState> AivExecutionResources::CreateState() const {
  return std::make_unique<AivInvocation>(this);
}
Status AivExecutionResources::Bind(ExecutionState* state) {
  auto* typed = state ? dynamic_cast<AivInvocation*>(state) : &state_->default_invocation;
  if (!typed || typed->owner != this)
    return Status::InvalidArgument("AIV execution state belongs to another resource");
  const auto& active = *state_->active;
  if (typed != state_->active && !active.graph_node &&
      (active.pending_get || active.pending_buffer || active.graph_buffer
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
      || active.pending_put || active.pending_put_buffer || active.input_buffer
#endif
      )) return Status::Busy("cannot replace an AIV invocation with pending resources");
  state_->views.Bind(typed);
  state_->active = typed;
  return Status::OK();
}
AivInvocation& AivExecutionResources::invocation() { return *state_->active; }
int AivGetSession::AcquireOutput(void* buffer, std::size_t bytes,
    KvafBatchGetLocalBuffer** out, std::uint64_t* entity) {
  return KvafBatchGetLocalBufferAcquire(transport, buffer, bytes, out, entity);
}
int AivGetSession::RetireOutput(KvafBatchGetLocalBuffer* buffer) {
  return KvafBatchGetLocalBufferRetire(buffer);
}
void AivGetSession::ReleaseOutput(KvafBatchGetLocalBuffer* buffer) {
  KvafBatchGetLocalBufferRelease(buffer);
}
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
Status AivPutSession::Reserve(std::size_t rows) {
  return EnsureBatchPutMetadataCapacity(rows, &workspace);
}
int AivPutSession::AcquireInput(void* buffer, std::size_t bytes, KvafBatchPutLocalBuffer** out) {
  return KvafBatchPutLocalBufferAcquire(transport, buffer, bytes, out);
}
int AivPutSession::RetireInput(KvafBatchPutLocalBuffer* buffer) {
  return KvafBatchPutLocalBufferRetire(buffer);
}
void AivPutSession::ReleaseInput(KvafBatchPutLocalBuffer* buffer) {
  KvafBatchPutLocalBufferRelease(buffer);
}
#endif
Status AivExecutionResources::PrepareGetResources(const BatchGetExecRequest& exec,
    AivInvocation& call, UrmaAivGetView* out) {
  if (!out || !out->common.Validate(&call, ExecutionDirection::Get).ok() || call.owner != this || state_->active != &call)
    return InvalidArgument("Get preparation requires the bound AIV invocation");
  if (!state_->initialized || !state_->get.transport)
    return Status::NotInitialized("AIV Get resource not initialized");
  if (!exec.request || !exec.request->output.valid() || !exec.execution.stream)
    return InvalidArgument("Get preparation requires output and stream");
  if (call.graph_buffer) {
    state_->views.Prepare(ExecutionDirection::Get);
    out->common = state_->views.View(ExecutionDirection::Get);
    out->buffer = call.graph_buffer; out->channel_entity = call.graph_entity;
    return Status::OK();
  }
  if (call.pending_buffer || call.pending_get)
    return Status::Busy("drain the preceding Get before preparing resources");
  if (call.output_retiring)
    return Status::InvalidState("output registration is retiring; retry end_get_output or shutdown");
  state_->views.Prepare(ExecutionDirection::Get);
  KvafBatchGetLocalBuffer* buffer = nullptr;
  std::uint64_t entity = 0;
  int result = KVAF_BATCH_GET_SUCCESS;
  if (call.output_buffer) {
    if (exec.request->output.data != call.output_options.output.data ||
        exec.request->output.bytes != call.output_options.output.bytes ||
        exec.request->indices.bytes / sizeof(std::int32_t) > call.output_options.request_capacity)
      return InvalidArgument("request differs from bound Get output/capacity");
    buffer = call.output_buffer;
    entity = call.output_entity;
  } else {
    result = state_->get.AcquireOutput(exec.request->output.data,
        exec.request->output.bytes, &buffer, &entity);
  }
  if (buffer) {
    call.pending_get = static_cast<aclrtStream>(exec.execution.stream);
    call.pending_buffer = buffer;
    call.get_entity = entity;
    call.get_output = exec.request->output;
    call.get_capacity = exec.request->indices.bytes / sizeof(std::int32_t);
    call.get_issued = false;
  }
  if (result != KVAF_BATCH_GET_SUCCESS || !buffer || !entity)
    return Internal("failed to acquire caller output for BatchGet");
  out->common = state_->views.View(ExecutionDirection::Get);
  out->buffer = call.pending_buffer;
  out->channel_entity = call.get_entity;
  return Status::OK();
}
Status AivExecutionResources::PrepareGraph(const GetGraphOptions& options, AivInvocation& call) {
  if (call.owner != this || state_->active != &call)
    return InvalidArgument("graph preparation requires the bound AIV invocation");
  if (call.graph_buffer) return Status::Busy("graph binding already exists");
  if (call.graph_node) {
    if (aclrtSynchronizeStream(static_cast<aclrtStream>(options.stream)) != ACL_SUCCESS)
      return Status::CompletionUnknown("Get graph stream is still pending");
    const auto& shared = state_->get.workspace;
    auto s = call.graph_report.Allocate(shared.status_bytes, 0, false);
    if (!s.ok()) return s;
    if (shared.profiling) {
      auto view = GetView();
      s = UrmaAivGetOperator::SetProfiling(true, view);
      if (!s.ok()) return s;
    }
  }
  BatchGetRequest request;
  request.output = options.output;
  BatchGetExecRequest exec;
  exec.request = &request;
  exec.execution.stream = options.stream;
  auto view = GetView();
  auto status = PrepareGetResources(exec, call, &view);
  if (!status.ok()) {
    if (!call.pending_buffer) return status;
    auto retired = UrmaAivGetOperator::Drain(view, call);
    return retired.ok() ? status : Status::CompletionUnknown(retired.message());
  }
  call.graph_buffer = call.pending_buffer;
  call.graph_entity = call.get_entity;
  call.graph_options = options;
  call.pending_buffer = nullptr;
  call.pending_get = nullptr;
  call.get_entity = 0;
  return Status::OK();
}
Status AivExecutionResources::SetGetProfiling(bool enabled) {
  auto s = UrmaAivGetOperator::SetProfiling(enabled, GetView());
  if (s.ok()) state_->get.workspace.profiling = enabled;
  return s;
}
Status AivExecutionResources::PrepareOutput(const GetOutputOptions& options, AivInvocation& call) {
  if (call.output_buffer || call.graph_buffer || !options.request_capacity || !options.stream ||
      !options.output.valid() || options.output.device != DeviceKind::kNpu)
    return InvalidArgument("invalid or already bound Get output");
  BatchGetRequest request;
  request.output = options.output;
  BatchGetExecRequest exec;
  exec.request = &request;
  exec.execution.stream = options.stream;
  auto view = GetView();
  auto s = PrepareGetResources(exec, call, &view);
  if (!s.ok()) return s;
  call.output_buffer = call.pending_buffer;
  call.output_entity = call.get_entity;
  call.output_options = options;
  call.pending_buffer = nullptr;
  call.pending_get = nullptr;
  call.get_entity = 0;
  return Status::OK();
}
Status AivExecutionResources::PrepareInput(const PutInputOptions& options, AivInvocation& call) {
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
  if (call.owner != this || state_->active != &call)
    return InvalidArgument("input binding requires the bound AIV invocation");
  if (!state_->initialized || !state_->put.transport)
    return Status::NotInitialized("AIV Put resource not initialized");
  if (call.input_buffer || call.pending_put || call.pending_put_buffer)
    return Status::Busy("resolve the preceding Put or finish its input binding first");
  if (!options.input.valid() || options.input.device != DeviceKind::kNpu ||
      !options.stream || !options.request_capacity)
    return InvalidArgument("input binding requires NPU input, stream and capacity");
  auto status = Warmup({options.request_capacity, options.stream});
  if (!status.ok()) return status;
  KvafBatchPutLocalBuffer* buffer = nullptr;
  const auto result = state_->put.AcquireInput(const_cast<void*>(options.input.data),
                                              options.input.bytes, &buffer);
  if (result != KVAF_BATCH_PUT_SUCCESS || !buffer) {
    // Acquire can leave a partial registration. Keep the caller's memory and
    // handle alive for Resolve/Close; never publish a partial input binding.
    call.pending_put = buffer ? static_cast<aclrtStream>(options.stream) : nullptr;
    call.pending_put_buffer = buffer;
    call.put_issued = false;
    auto failure = Internal("failed to register bound Put input");
    call.put_completion = {buffer ? PutOutcome::Unknown : PutOutcome::Failed, failure};
    return failure;
  }
  call.input_buffer = buffer;
  call.input_options = options;
  call.input_retiring = false;
  return Status::OK();
#else
  (void)options; (void)call;
  return Status::NotImplemented("AIV Put input binding is not built");
#endif
}
Status AivExecutionResources::PreparePutResources(const BatchPutExecRequest& exec,
    AivInvocation& call, UrmaAivPutView* out) {
  if (!out || !out->common.Validate(&call, ExecutionDirection::Put).ok() || call.owner != this || state_->active != &call)
    return InvalidArgument("Put preparation requires the bound AIV invocation");
  if (!state_->initialized) return Status::NotInitialized("AIV Put resource not initialized");
  if (!exec.request || !exec.request->input.valid() || exec.request->keys.empty() ||
      exec.request->keys.size() > UINT32_MAX)
    return InvalidArgument("invalid Put resource capacity or input");
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
  if (call.pending_put || call.pending_put_buffer)
    return Status::Busy("resolve the preceding Put before preparing resources");
  if (call.input_retiring)
    return Status::Busy("retry ending the partially retired Put input binding");
  if (call.input_buffer &&
      (exec.request->input.data != call.input_options.input.data ||
       exec.request->input.bytes != call.input_options.input.bytes ||
       exec.request->input.device != call.input_options.input.device ||
       exec.request->keys.size() > call.input_options.request_capacity))
    return InvalidArgument("Put input or request capacity does not match its binding");
#endif
  state_->views.Prepare(ExecutionDirection::Put);
  call.put_completion = {};
  auto status = Status::OK();
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
  if (!call.input_buffer)
#endif
    status = Warmup({static_cast<std::uint32_t>(exec.request->keys.size()), exec.execution.stream});
  if (!status.ok()) { call.put_completion = {PutOutcome::Failed, status}; return status; }
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
  call.put_rows = exec.request->keys.size();
  call.put_input = exec.request->input;
  call.put_issued = false;
  if (state_->put.transport) {
    if (!exec.execution.stream) return InvalidArgument("AIV Put requires a stream");
    KvafBatchPutLocalBuffer* buffer = call.input_buffer;
    int32_t result = KVAF_BATCH_PUT_SUCCESS;
    if (!buffer) {
      PutHostTimingScope registration(exec.execution.host_timing, &PutProviderTiming::registration_ns);
      result = state_->put.AcquireInput(const_cast<void*>(exec.request->input.data),
                                       exec.request->input.bytes, &buffer);
    }
    if (buffer) {
      call.pending_put = static_cast<aclrtStream>(exec.execution.stream);
      call.pending_put_buffer = buffer;
    }
    if (result != KVAF_BATCH_PUT_SUCCESS || !buffer) {
      auto failure = Internal("failed to acquire caller input for BatchPut");
      call.put_completion = {buffer ? PutOutcome::Unknown : PutOutcome::Failed, failure};
      return failure;
    }
  }
#endif
  out->common = state_->views.View(ExecutionDirection::Put);
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
  out->buffer = call.pending_put_buffer;
#endif
  return Status::OK();
}

Status AivExecutionResources::Initialize(const RemoteRegionDesc& remote_region) {
  if (state_->initialized || state_->closing || state_->get.transport
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
      || state_->put.transport
#endif
  ) {
    return InvalidState("AccessOperator has already been initialized");
  }
  if (remote_region.base == 0 || remote_region.length == 0 ||
      (state_->config.enable_get && state_->config.batch_get_kernel_object_path.empty()) ||
      state_->config.npu_core_count == 0 ||
      state_->config.npu_core_count > KVAF_AIV_BATCH_GET_MAX_AIV_CORES ||
      state_->config.jettys_per_core == 0 ||
      state_->config.jettys_per_core > KVAF_AIV_BATCH_GET_MAX_JETTYS_PER_CORE ||
      (state_->config.enable_get && state_->config.batch_get_max_inflight_total != 0U &&
       state_->config.batch_get_max_inflight_total < 2U * state_->config.npu_core_count)) {
    return InvalidArgument("invalid AccessOperator initialization config");
  }
  if (state_->config.peer_link == nullptr) {
    return InvalidState("AccessOperator requires an established Runtime Link");
  }
  const std::size_t worker_count =
      static_cast<std::size_t>(state_->config.npu_core_count) *
      state_->config.jettys_per_core;
  if (state_->config.peer_local_eids.empty() &&
      std::any_of(state_->config.peer_local_eid.begin(),
                  state_->config.peer_local_eid.end(),
                  [](std::uint8_t value) { return value != 0U; })) {
    state_->config.peer_local_eids.push_back(state_->config.peer_local_eid);
  }
  if (state_->config.peer_local_eids.empty() ||
      state_->config.peer_local_eids.size() > worker_count ||
      std::any_of(state_->config.peer_local_eids.begin(),
                  state_->config.peer_local_eids.end(),
                  [](const auto& eid) {
                    return std::all_of(eid.begin(), eid.end(),
                                       [](std::uint8_t value) { return value == 0U; });
                  })) {
    return InvalidArgument("Runtime Link requires valid Runtime-selected local EIDs");
  }
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
  const std::uint32_t batch_put_jettys_per_core =
      state_->config.batch_put_jettys_per_core == 0U
          ? std::min(
                state_->config.jettys_per_core,
                static_cast<std::uint32_t>(
                    KVAF_AIV_BATCH_PUT_MAX_JETTYS_PER_CORE))
          : state_->config.batch_put_jettys_per_core;
  if (!state_->config.batch_put_kernel_object_path.empty() &&
      (state_->config.npu_core_count > KVAF_AIV_BATCH_PUT_MAX_AIV_CORES ||
       batch_put_jettys_per_core >
           KVAF_AIV_BATCH_PUT_MAX_JETTYS_PER_CORE)) {
    return InvalidArgument("invalid BatchPut transport configuration");
  }
#endif
  if (state_->config.enable_get) {
    if (KvafExecuteBatchGetLoadKernel(
            state_->config.batch_get_kernel_object_path.c_str()) !=
        KVAF_BATCH_GET_SUCCESS) {
      return Internal("failed to load BatchGet kernel");
    }

    KvafBatchGetTransport* transport = nullptr;
    static_assert(sizeof(state_->config.peer_local_eids.front()) == 16U,
                  "NPU EID storage must remain packed");
    const auto* peer_local_eids = reinterpret_cast<const std::uint8_t*>(
        state_->config.peer_local_eids.data());
    const auto peer_local_eid_count =
        static_cast<std::uint32_t>(state_->config.peer_local_eids.size());
    std::vector<void*> peer_links;
    for (const auto& link : state_->config.peer_get_links) peer_links.push_back(link.get());
    if (peer_links.empty()) peer_links.push_back(state_->config.peer_link.get());
    const int32_t result = KvafBatchGetTransportCreateWithPeerLinks(
        reinterpret_cast<void*>(remote_region.base), remote_region.length,
        state_->config.npu_core_count, state_->config.jettys_per_core,
        peer_links.data(), static_cast<uint32_t>(peer_links.size()), peer_local_eids, peer_local_eid_count,
        &transport);
    state_->get.transport = transport;
    state_->remote_region = remote_region;
    if (result != KVAF_BATCH_GET_SUCCESS || transport == nullptr) {
      (void)Close();
      return Internal("failed to create reusable BatchGet transport");
    }

    AivGetWorkspace execution;
    execution.status_bytes = static_cast<std::size_t>(state_->config.npu_core_count) *
                             sizeof(KvafBatchGetCoreStatus);
    execution.queue_state_bytes =
        static_cast<std::size_t>(state_->config.npu_core_count) *
        state_->config.jettys_per_core *
        sizeof(KvafBatchGetQueueState);
    if (aclrtMalloc(&execution.status, execution.status_bytes,
                    ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMalloc(&execution.queue_state, execution.queue_state_bytes,
                    ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMemset(execution.status, execution.status_bytes, 0,
                    execution.status_bytes) != ACL_SUCCESS ||
        aclrtMemset(execution.queue_state, execution.queue_state_bytes, 0,
                    execution.queue_state_bytes) != ACL_SUCCESS) {
      state_->get.workspace = execution;
      (void)Close();
      return Internal("failed to allocate internal BatchGet execution state");
  }
  state_->remote_region = remote_region;
  state_->get.transport = transport;
  state_->get.workspace = execution;
  }
  state_->remote_region = remote_region;
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
  if (!state_->config.batch_put_kernel_object_path.empty()) {
    if (state_->config.npu_core_count > KVAF_AIV_BATCH_PUT_MAX_AIV_CORES ||
        KvafExecuteBatchPutLoadKernel(
            state_->config.batch_put_kernel_object_path.c_str()) !=
            KVAF_BATCH_PUT_SUCCESS) {
      (void)Close();
      return Internal("failed to load BatchPut kernel");
    }

    KvafBatchPutTransport* batch_put_transport = nullptr;
    const int32_t batch_put_result = KvafBatchPutTransportCreateWithPeerLink(
        reinterpret_cast<void*>(remote_region.base), remote_region.length,
        state_->config.npu_core_count, batch_put_jettys_per_core,
        state_->config.peer_link.get(), state_->config.peer_local_eid.data(),
        &batch_put_transport);
    state_->put.transport = batch_put_transport;
    if (batch_put_result != KVAF_BATCH_PUT_SUCCESS ||
        batch_put_transport == nullptr) {
      (void)Close();
      return Internal("failed to create reusable BatchPut transport");
    }
    Status batch_put_status = InitializeAivPutWorkspace(
        state_->config.npu_core_count, &state_->put.workspace);
    if (!batch_put_status.ok()) {
      (void)Close();
      return batch_put_status;
    }
    state_->put.transport = batch_put_transport;
    std::fprintf(stderr,
                 "[KVAF_AIV] RUNTIME_BATCH_PUT_TRANSPORT_READY cores=%u "
                 "jettys_per_core=%u "
                 "metadata_staging=reusable\n",
                 state_->config.npu_core_count,
                 batch_put_jettys_per_core);
  }
#endif
  state_->initialized = true;
  state_->views.Initialize(remote_region);
  return Status::OK();
}

Status AivExecutionResources::Close() {
  if (state_->active->graph_buffer) return Status::Busy("release graph lease before provider shutdown");
  state_->initialized = false;
  state_->views.Close();
  if (state_->active->pending_get &&
      aclrtSynchronizeStream(state_->active->pending_get) != ACL_SUCCESS)
    return Status::CompletionUnknown("Get stream still pending; resources retained");

  // Callers must complete all submitted streams before Runtime::Shutdown,
  // including the default enqueue-only Python BatchGet path.
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
  if (state_->active->pending_put) {
    if (aclrtSynchronizeStream(state_->active->pending_put) != ACL_SUCCESS)
      return Status::CompletionUnknown("Put stream still pending; resources retained");
  }
  state_->closing = true;
  // Kernel completion does not retire remote WRs after CQ errors. Close queues
  // before touching their buffers, preserving partial teardown for retry.
  if (state_->put.transport != nullptr) {
    if (KvafBatchPutTransportClose(state_->put.transport) != KVAF_BATCH_PUT_SUCCESS)
      return Status::CompletionUnknown("Put transport teardown failed; resources retained");
  }
  state_->put.transport = nullptr;
  state_->active->pending_put_buffer = nullptr;
  state_->active->pending_put = nullptr;
  state_->active->input_buffer = nullptr;
  state_->active->input_options = {};
  state_->active->input_retiring = false;
  auto released = ReleaseAivPutWorkspace(&state_->put.workspace);
  if (!released.ok()) return released;
#endif
  state_->closing = true;
  if (state_->get.transport != nullptr) {
    if (KvafBatchGetTransportClose(state_->get.transport) != KVAF_BATCH_GET_SUCCESS)
      return Status::CompletionUnknown("Get transport teardown failed; resources retained");
  }
  state_->get.transport = nullptr;
  state_->active->pending_buffer = nullptr;
  state_->active->pending_get = nullptr;
  state_->active->get_issued = false;
  state_->active->output_buffer = nullptr;
  state_->active->output_entity = 0;
  state_->active->output_options = {};
  state_->active->output_retiring = false;
  if (state_->get.workspace.queue_state != nullptr) {
    if (aclrtFree(state_->get.workspace.queue_state) != ACL_SUCCESS)
      return Status::CompletionUnknown("Get queue state release failed; allocation retained");
    state_->get.workspace.queue_state = nullptr;
  }
  if (state_->get.workspace.status != nullptr) {
    if (aclrtFree(state_->get.workspace.status) != ACL_SUCCESS)
      return Status::CompletionUnknown("Get status release failed; allocation retained");
    state_->get.workspace.status = nullptr;
  }
  state_->get.transport = nullptr;
  state_->get.workspace = AivGetWorkspace{};
  state_->remote_region = RemoteRegionDesc{};
  state_->initialized = false;
  state_->closing = false;
  return Status::OK();
}

Status AivExecutionResources::Warmup(const ExecutionWarmup& request) {
  (void)request;
  if (!state_->initialized) return Status::NotInitialized("AIV provider not initialized");
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
  if (state_->put.transport && request.request_capacity) {
    if (state_->active->pending_put &&
        request.request_capacity > state_->put.workspace.metadata_capacity)
      return Status::Busy("cannot resize workspace used by a pending Put");
    state_->views.Prepare(ExecutionDirection::Put);
    return state_->put.Reserve(request.request_capacity);
  }
#endif
  return Status::OK();
}


}  // namespace kvaf
