#include "aicpu_execution_resources.h"

#include <mutex>
#include "aicpu_batch_get_capture.h"

#include "get/common/aicpu/include/kvaf_aicpu_batch_get.h"
#include "get/scatter/urma_aicpu/include/cpu_push_transport.h"
#include "put/urma_aicpu/include/kvaf_aicpu_batch_put.h"
#include "aicpu_urma/session.h"
#include "aicpu_urma/kernel_loader.h"
namespace kvaf {
struct AicpuExecutionResources::State {
  State(AicpuExecutionResources* owner, UmdkAccessOperatorConfig config)
      : default_invocation(owner), valid_mode(config.batch_get_mode == "gather_read" ||
                   config.batch_get_mode == "cpu_push_scatter" || (config.batch_get_mode == "cpu_sgl_write" || config.batch_get_mode == "cpu_sgl_push")),
        get_session({config.kernel_json, config.workers,
                 config.peer_link, config.local_eid}),
        put_session({config.kernel_json, config.workers, config.peer_link, config.local_eid,
                     aicpu_urma::UmdkQueueProfile::Put}),
        get(get_session, config.token_bytes),
        put(put_session), enable_get(config.enable_get) {
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
    if (config.batch_get_mode == "cpu_push_scatter" || (config.batch_get_mode == "cpu_sgl_write" || config.batch_get_mode == "cpu_sgl_push"))
      push = std::make_unique<cpu_push::CpuPushTransport>(cpu_push::Config{
          config.peer_link, config.kernel_json, config.local_eid, config.token_bytes, config.workers,
          (config.batch_get_mode == "cpu_sgl_write" || config.batch_get_mode == "cpu_sgl_push"), config.sge_per_wr, config.sgl_payload_only, false,
          config.sgl_pipeline, config.batch_get_mode == "cpu_sgl_push"});
#else
    valid_mode = config.batch_get_mode == "gather_read";
#endif
    default_invocation.put = put.CreateExecutionState();
    default_invocation.get = get.CreateExecutionState();
    (void)put.BindExecutionState(default_invocation.put.get());
    (void)get.BindExecutionState(default_invocation.get.get());
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
    if (push) { default_invocation.push = push->CreateExecutionState(); (void)push->BindExecutionState(default_invocation.push.get()); }
#endif
  }
  ExecutionViewState views;
  // Keeps initialization and Close valid before binding and after unbinding.
  AicpuInvocation default_invocation;
  AicpuInvocation* invocation{&default_invocation};
  bool valid_mode;
  aicpu_urma::UmdkSession get_session, put_session;
  aicpu_urma::BatchGetTransport get;
  aicpu_urma::BatchPutTransport put;
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
  std::unique_ptr<cpu_push::CpuPushTransport> push;
  Status DrainGet() { return push ? push->Drain() : get.Drain(); }
#else
  Status DrainGet() { return get.Drain(); }
#endif
  bool enable_get;
  RemoteRegionDesc region;
  bool initialized{false};
  void* warmup_buffer{nullptr};
  std::mutex get_mutex, put_mutex;
};

AicpuExecutionResources::AicpuExecutionResources(UmdkAccessOperatorConfig config)
    : state_(std::make_unique<State>(this, std::move(config))) { state_->views.Bind(&state_->default_invocation); }
AicpuExecutionResources::~AicpuExecutionResources() {
  // An unresolved DMA/failed release keeps every handle and backing buffer.
  if (!Close().ok()) (void)state_.release();
}
std::unique_ptr<ExecutionState> AicpuExecutionResources::CreateInvocation() {
  auto call = std::make_unique<AicpuInvocation>(this);
  call->put = state_->put.CreateExecutionState();
  call->get = state_->get.CreateExecutionState();
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
  if (state_->push) call->push = state_->push->CreateExecutionState();
#endif
  return call;
}
Status AicpuExecutionResources::Bind(ExecutionState* state) {
  auto* call = state ? dynamic_cast<AicpuInvocation*>(state) : &state_->default_invocation;
  if (!call || call->owner != this)
    return Status::InvalidArgument("AICPU invocation belongs to another resource");
  if (state_->invocation != call &&
      (state_->invocation->input_bound ||
       (state_->invocation->graph_active && !state_->invocation->graph_node)))
    return Status::Busy("release input binding before replacing invocation");
  auto bound = state_->put.BindExecutionState(call->put.get());
  if (!bound.ok()) return bound;
  // The dispatcher marks graph nodes after CreateInvocation(). Forward the
  // flag at bind time so inner transports use node-owned completion/profile.
  call->get->graph_node = call->graph_node;
  bound = state_->get.BindExecutionState(call->get.get());
  if (!bound.ok()) {
    (void)state_->put.BindExecutionState(state_->invocation->put.get());
    return bound;
  }
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
  if (state_->push) {
    call->push->graph_node = call->graph_node;
    bound = state_->push->BindExecutionState(call->push.get());
    if (!bound.ok()) {
      (void)state_->get.BindExecutionState(state_->invocation->get.get());
      (void)state_->put.BindExecutionState(state_->invocation->put.get());
      return bound;
    }
  }
#endif
  state_->invocation = call;
  state_->views.Bind(call);
  state_->put_session.BindWriteState(&call->write);
  return Status::OK();
}
AicpuInvocation& AicpuExecutionResources::invocation() { return *state_->invocation; }
std::mutex& AicpuExecutionResources::put_mutex() { return state_->put_mutex; }
std::mutex& AicpuExecutionResources::get_mutex() { return state_->get_mutex; }
bool AicpuExecutionResources::scatter() const {
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
  return bool(state_->push);
#else
  return false;
#endif
}
Status AicpuExecutionResources::PreparePutResources(const BatchPutExecRequest& request) {
  state_->views.Prepare(ExecutionDirection::Put);
  if (!state_->initialized) return Status::NotInitialized("AICPU resources not initialized");
  if (!request.request || !request.put_desc || !request.execution.stream)
    return Status::InvalidArgument("AICPU Put needs input, destination and stream");
  auto& call = *state_->invocation;
  if (call.input_retiring)
    return Status::Busy("retry ending the partially retired Put input binding");
  if (call.input_bound) {
    if (request.execution.stream != call.put_input.stream)
      return Status::InvalidArgument("request differs from bound Put stream");
    return state_->put.PrepareBound(state_->region, *request.request, call.put_input);
  }
  auto loaded = state_->put_session.Load();
  if (!loaded.ok()) return loaded;
  state_->put_session.BindWriteState(&state_->invocation->write);
  return state_->put.Prepare(state_->region, *request.request, request.execution.host_timing);
}
Status AicpuExecutionResources::PrepareGetResources(const BatchGetExecRequest& request) {
  state_->views.Prepare(ExecutionDirection::Get);
  if (!state_->initialized || !state_->enable_get)
    return Status::NotInitialized("AICPU Get resources not initialized");
  if (!request.request) return Status::InvalidArgument("missing Get request");
  auto& call = *state_->invocation;
  if (call.graph_active) {
    const auto& options = call.graph_options;
    if (call.graph_retiring) return Status::Busy("graph retirement is still pending");
    if (call.graph_recorded) return Status::NotImplemented("one Get node per graph lease is supported");
    if (request.execution.stream != options.stream || request.execution.synchronize ||
        request.execution.profile || request.request->output.data != options.output.data ||
        request.request->output.bytes != options.output.bytes ||
        request.request->indices.bytes / sizeof(std::int32_t) > options.request_capacity)
      return Status::InvalidArgument("captured request differs from prepared graph lease");
  }
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
  if (state_->push) {
    if (call.output_bound && (request.request->output.data != call.get_output.data ||
                             request.request->output.bytes != call.get_output.bytes))
      return Status::InvalidArgument("request differs from bound CPU SGL output");
    return state_->push->Prepare(request, call.graph_active);
  }
#endif
  if (state_->invocation->output_bound &&
      (request.request->output.data != state_->invocation->get_output.data ||
       request.request->output.bytes != state_->invocation->get_output.bytes))
    return Status::InvalidArgument("request differs from bound Get output");
  call.get_output = request.request->output;
  auto ready = state_->get.Prepare(request, call.graph_active);
  if (ready.ok() && !call.graph_active && !call.output_bound && !call.output_pinned) {
    ready = state_->get_session.PinBuffer(state_->get.View().remote_workspace.base,
                                         call.get_output.data, call.get_output.bytes);
    if (ready.ok()) call.output_pinned = true;
  }
  return ready;
}
Status AicpuExecutionResources::ReleasePutBuffers() {
  // Ordinary calls retire temporary input; explicit bindings own it until End.
  if (state_->invocation->input_bound) return Status::OK();
  return state_->put_session.ReleaseBuffers();
}
Status AicpuExecutionResources::PreparePutInput(const PutInputOptions& options) {
  std::lock_guard<std::mutex> lock(state_->put_mutex);
  auto& call = *state_->invocation;
  if (!state_->initialized || call.input_bound || call.write.attempted ||
      !options.stream || !options.request_capacity ||
      !options.input.valid() || options.input.device != DeviceKind::kNpu)
    return Status::InvalidArgument("invalid AICPU Put input binding");
  aclmdlRICaptureStatus capture = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
  aclmdlRI graph = nullptr;
  if (aclmdlRICaptureGetInfo(static_cast<aclrtStream>(options.stream), &capture, &graph) != ACL_SUCCESS)
    return Status::Internal("cannot query graph capture state");
  if (capture != ACL_MODEL_RI_CAPTURE_STATUS_NONE)
    return Status::InvalidState("bind Put input before graph capture");
  auto s = state_->put_session.Load();
  if (s.ok()) s = state_->put.Reserve(options.request_capacity);
  if (s.ok()) s = state_->put_session.Ensure(state_->region.base,
      const_cast<void*>(options.input.data), options.input.bytes);
  if (!s.ok()) return s;
  state_->put_session.BindWriteState(&call.write);
  call.put_input = options;
  call.input_bound = true;
  call.input_retiring = false;
  return Status::OK();
}
Status AicpuExecutionResources::FinishPutInput() {
  std::lock_guard<std::mutex> lock(state_->put_mutex);
  auto& call = *state_->invocation;
  if (!call.input_bound) return Status::OK();
  // Even a completed WRITE keeps its binding until metadata finalization/ACK.
  if (call.write.attempted)
    return Status::Busy("acknowledge the preceding Put before ending its input binding");
  call.input_retiring = true;
  state_->views.Prepare(ExecutionDirection::Put);
  auto s = state_->put_session.ReleaseBuffer(const_cast<void*>(call.put_input.input.data),
                                            call.put_input.input.bytes);
  if (s.ok()) {
    call.put_input = {};
    call.input_bound = call.input_retiring = false;
  }
  return s;
}
Status AicpuExecutionResources::ReleaseGetBuffers() {
  auto& call = *state_->invocation;
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
  if (state_->push) {
    if (call.output_bound || call.graph_active) return Status::OK();
    return state_->push->ReleaseOutput();
  }
#endif
  if (call.output_bound || call.graph_active || !call.get_output.valid()) return Status::OK();
  auto s = call.output_pinned
      ? state_->get_session.UnpinBuffer(call.get_output.data, call.get_output.bytes)
      : state_->get_session.ReleaseBuffer(call.get_output.data, call.get_output.bytes);
  if (s.ok()) { call.get_output = {}; call.output_pinned = false; }
  return s;
}
Status AicpuExecutionResources::PrepareGetGraph(const GetGraphOptions& options) {
  std::lock_guard<std::mutex> lock(state_->get_mutex);
  auto& call = *state_->invocation;
  if (!state_->initialized || !state_->enable_get)
    return Status::NotInitialized("AICPU Get resources not initialized");
  if (call.graph_active || call.output_bound) return Status::Busy("Get resources already leased");
  Status ready;
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
  if (state_->push) ready = state_->push->PrepareGraph(options);
  else
#endif
    ready = state_->get.PrepareGraph(options);
  if (!ready.ok()) return ready;
  call.graph_options = options;
  call.get_output = options.output;
  call.graph_active = true;
  call.graph_recorded = call.graph_retiring = call.graph_close_required = false;
  return Status::OK();
}
Status AicpuExecutionResources::FinishGetGraph(void* stream, bool discard) {
  std::unique_lock<std::mutex> lock(state_->get_mutex);
  auto& call = *state_->invocation;
  if (!call.graph_active) return Status::NotFound("AICPU graph lease not found");
  if (!stream || stream != call.graph_options.stream)
    return Status::InvalidArgument("graph release requires its prepared execution stream");
  bool capturing = false;
  auto allowed = detail::BatchGetCapture(static_cast<aclrtStream>(stream), true, false, &capturing);
  if (!allowed.ok()) return allowed;
  if (call.graph_retiring && discard && !call.graph_close_required) {
    // Output unregister can fail permanently after device completion is known.
    // An explicit discard must still be able to retire the complete transport,
    // rather than leave an unrecoverable graph lease blocking Shutdown/Recover.
    if (aclrtSynchronizeStream(static_cast<aclrtStream>(stream)) != ACL_SUCCESS)
      return Status::CompletionUnknown("graph discard completion unknown; resources retained");
    call.graph_close_required = true;
  }
  if (!call.graph_retiring) {
    Status ready;
    auto* low = dynamic_cast<aicpu_urma::AicpuGetInvocation*>(call.get.get());
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
    if (state_->push) {
      low = dynamic_cast<aicpu_urma::AicpuGetInvocation*>(call.push.get());
      ready = state_->push->FinishGraph(stream, discard);
    } else
#endif
      ready = state_->get.FinishGraph(stream, discard);
    if (!ready.ok()) {
      if (!low || low->failure.ok() || ready.code() == StatusCode::kCompletionUnknown) return ready;
      if (!discard && !call.graph_node) return ready;
      // Discard cannot make a failed DMA transport safe. Retire its complete
      // session (Scatter sends STOP first), retaining the lease on any failure.
      call.graph_close_required = true;
      if (call.graph_node) {
        call.graph_retiring = true;
        return ready;
      }
    }
    call.graph_retiring = true;
  }
  if (call.graph_close_required) {
    if (call.graph_node)
      return Status::InvalidState("Get graph communication domain requires recovery");
    lock.unlock();
    return Close();
  }
  if (!scatter()) {
    auto retired = state_->get_session.UnpinBuffer(call.get_output.data, call.get_output.bytes);
    if (!retired.ok()) {
      // FinishGraph already fenced this node. A failed registration control
      // operation can make individual unpins permanently unavailable; retain
      // the output and let the dispatcher close the whole shared session once
      // every graph using it has stopped.
      call.graph_close_required = true;
      return retired;
    }
  }
  call.graph_active = call.graph_recorded = call.graph_retiring = false;
  call.graph_options = {};
  call.get_output = {};
  return Status::OK();
}
bool AicpuExecutionResources::GetGraphNeedsRecovery() const {
  return state_->invocation->graph_close_required;
}
Status AicpuExecutionResources::PrepareGetOutput(const GetOutputOptions& options) {
  std::lock_guard<std::mutex> lock(state_->get_mutex);
  auto& call = *state_->invocation;
  bool supports_output = !scatter();
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
  if (state_->push) supports_output = state_->push->direct_output();
#endif
  if (!state_->initialized || !state_->enable_get || !supports_output || call.output_bound || call.graph_active ||
      !options.stream || !options.request_capacity ||
      !options.output.valid() || options.output.device != DeviceKind::kNpu)
    return Status::InvalidArgument("invalid AICPU output binding");
  aclmdlRICaptureStatus capture = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
  aclmdlRI graph = nullptr;
  if (aclmdlRICaptureGetInfo(static_cast<aclrtStream>(options.stream), &capture, &graph) != ACL_SUCCESS)
    return Status::Internal("cannot query graph capture state");
  if (capture != ACL_MODEL_RI_CAPTURE_STATUS_NONE)
    return Status::InvalidState("bind Get output before graph capture");
  auto s = state_->DrainGet();
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
  if (state_->push) {
    if (s.ok()) s = state_->push->RegisterOutput(options.output);
    if (s.ok()) { call.get_output = options.output; call.output_bound = true; }
    return s;
  }
#endif
  if (s.ok()) s = state_->get.Warmup();
  if (!s.ok()) return s;
  call.get_output = options.output;
  s = state_->get_session.PinBuffer(state_->get.View().remote_workspace.base,
                                    options.output.data, options.output.bytes);
  if (s.ok()) call.output_bound = call.output_pinned = true;
  return s;
}
Status AicpuExecutionResources::FinishGetOutput() {
  std::lock_guard<std::mutex> lock(state_->get_mutex);
  if (state_->invocation->graph_active) return Status::Busy("Get output belongs to a graph lease");
  auto s = state_->DrainGet();
  if (!s.ok()) return s;
  auto& call = *state_->invocation;
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
  if (state_->push) {
    s = state_->push->ReleaseOutput();
    if (s.ok()) { call.get_output = {}; call.output_bound = false; }
    return s;
  }
#endif
  s = state_->get_session.UnpinBuffer(call.get_output.data, call.get_output.bytes);
  if (s.ok()) { call.get_output = {}; call.output_bound = call.output_pinned = false; }
  return s;
}
UrmaAicpuPutView AicpuExecutionResources::PutView() {
  return {state_->views.View(ExecutionDirection::Put), state_->put.View()};
}
UrmaAicpuGatherView AicpuExecutionResources::GatherView() {
  return {state_->views.View(ExecutionDirection::Get), state_->get.View()};
}
UrmaAicpuScatterView AicpuExecutionResources::ScatterView() {
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
  if (state_->push) return {state_->views.View(ExecutionDirection::Get), state_->push->View()};
#endif
  return {};
}

Status AicpuExecutionResources::Initialize(const RemoteRegionDesc& region) {
  if (!state_->valid_mode) return Status::InvalidArgument("unknown AICPU BatchGet mode");
  if (state_->initialized) return Status::InvalidState("AICPU resources already initialized");
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
  auto loaded = state_->enable_get && state_->push
      ? aicpu_urma::PrepareUmdkKernel(state_->get_session.config().kernel_json,
                                    state_->push->direct_output() ? aicpu_urma::UmdkKernelKind::CpuSglWrite :
                                    aicpu_urma::UmdkKernelKind::Scatter)
      : state_->enable_get ? state_->get_session.Load() : state_->put_session.Load();
#else
  auto loaded = state_->enable_get ? state_->get_session.Load() : state_->put_session.Load();
#endif
  if (!loaded.ok()) return loaded;
  if (!state_->enable_get) { state_->region = region; state_->initialized = true; state_->views.Initialize(region); return Status::OK(); }
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
  auto ready = state_->push ? state_->push->Initialize(region) : state_->get.Initialize(region);
#else
  auto ready = state_->get.Initialize(region);
#endif
  if (!ready.ok()) return ready;
  state_->region = region;
  state_->initialized = true;
  state_->views.Initialize(region);
  return Status::OK();
}
Status AicpuExecutionResources::Warmup(const ExecutionWarmup& request) {
  if (!state_->initialized) return Status::NotInitialized("AICPU provider not initialized");
  if (state_->enable_get) {
    std::lock_guard<std::mutex> lock(state_->get_mutex);
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
    if (state_->push) return Status::OK();
#endif
    return state_->get.Warmup();
  }
  std::lock_guard<std::mutex> lock(state_->put_mutex);
  if (!state_->warmup_buffer &&
      aclrtMalloc(&state_->warmup_buffer, 64, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
    return Status::OutOfMemory("cannot allocate session setup buffer");
  auto s = state_->put_session.EnsureSession(state_->region.base, state_->warmup_buffer, 64);
  return s.ok() ? state_->put.Reserve(request.request_capacity ? request.request_capacity : 65536)
                : s;
}
Status AicpuExecutionResources::Close() {
  std::scoped_lock lock(state_->get_mutex, state_->put_mutex);
  if (state_->invocation->graph_active && !state_->invocation->graph_retiring)
    return Status::Busy("release Get graph before closing AICPU resources");
  state_->views.Close();
  // Push shutdown must stop the remote DMA producer even after a kernel failure.
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
  if (state_->push) {
    auto stopped = state_->push->Shutdown();
    if (!stopped.ok()) return stopped;
  }
#endif
  // Reported protocol errors do not prevent synchronized resource retirement.
  auto drained = state_->get.Drain(false);
  if (!drained.ok()) return drained;
  auto closed = state_->get_session.Shutdown();
  if (!closed.ok()) return closed;
  closed = state_->put_session.Shutdown();
  if (!closed.ok()) return closed;
  auto released = state_->get.Shutdown();
  if (!released.ok()) return released;
  auto put_released = state_->put.Shutdown();
  if (!put_released.ok()) return put_released;
  if (state_->warmup_buffer) {
    if (aclrtFree(state_->warmup_buffer) != ACL_SUCCESS)
      return Status::Internal("cannot free session setup buffer");
    state_->warmup_buffer = nullptr;
  }
  state_->initialized = false;
  state_->invocation->get_output = {};
  state_->invocation->output_bound = state_->invocation->output_pinned = false;
  state_->invocation->graph_active = state_->invocation->graph_recorded = false;
  state_->invocation->graph_retiring = state_->invocation->graph_close_required = false;
  state_->invocation->graph_options = {};
  state_->invocation->put_input = {};
  state_->invocation->input_bound = state_->invocation->input_retiring = false;
  state_->invocation->put_completion = {};
  state_->invocation->write = {};
  return Status::OK();
}
Status AicpuExecutionResources::ResolvePut(PutCompletion* out) {
  if (!out) return Status::InvalidArgument("null Put completion output");
  std::lock_guard<std::mutex> lock(state_->put_mutex);
  auto& call = *state_->invocation;
  if (call.put_completion.outcome == PutOutcome::Unknown) {
    auto status = state_->put_session.ResolveWrite(&call.put_completion);
    if (!status.ok()) return status;
  }
  *out = call.put_completion;
  return Status::OK();
}
Status AicpuExecutionResources::AcknowledgePut() {
  std::lock_guard<std::mutex> lock(state_->put_mutex);
  auto& call = *state_->invocation;
  if (call.put_completion.outcome == PutOutcome::Unknown)
    return Status::CompletionUnknown("cannot acknowledge unresolved AICPU Put");
  auto released = ReleasePutBuffers();
  if (!released.ok()) return released;
  call.put_completion = {};
  call.write = {};
  return Status::OK();
}
Status AicpuExecutionResources::DrainGet() {
  std::lock_guard<std::mutex> lock(state_->get_mutex);
  auto status = state_->DrainGet();
  if (status.ok()) status = ReleaseGetBuffers();
  return status;
}
Status AicpuExecutionResources::CheckBatchGetStatus(void* stream) {
  std::lock_guard<std::mutex> lock(state_->get_mutex);
  if (!state_->initialized) return Status::NotInitialized("AICPU resources are not initialized");
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
  if (state_->push) return state_->push->Check(stream);
#endif
  return state_->get.Check(static_cast<aclrtStream>(stream));
}
Status AicpuExecutionResources::SetBatchGetKernelProfiling(bool enabled) {
  std::lock_guard<std::mutex> lock(state_->get_mutex);
  if (state_->invocation->graph_recorded || state_->invocation->graph_retiring)
    return Status::Busy("configure profiling before recording the graph");
  if (!state_->initialized) return Status::NotInitialized("AICPU resources are not initialized");
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
  if (state_->push) return state_->push->SetKernelProfiling(enabled);
#endif
  return state_->get.SetKernelProfiling(enabled);
}
Status AicpuExecutionResources::CopyBatchGetKernelProfiles(std::vector<BatchGetKernelProfile>* profiles) const {
  std::lock_guard<std::mutex> lock(state_->get_mutex);
  if (!state_->initialized) return Status::NotInitialized("AICPU resources are not initialized");
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
  if (state_->push) return state_->push->CopyKernelProfiles(profiles);
#endif
  return state_->get.CopyKernelProfiles(profiles);
}
}  // namespace kvaf
