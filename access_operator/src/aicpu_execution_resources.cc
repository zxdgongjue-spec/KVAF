#include "aicpu_execution_resources.h"

#include <mutex>

#include "aicpu_batch_get/include/kvaf_aicpu_batch_get.h"
#include "aicpu_batch_get/include/cpu_push_transport.h"
#include "aicpu_batch_put/include/kvaf_aicpu_batch_put.h"
#include "aicpu_urma/session.h"
#include "aicpu_urma/kernel_loader.h"
namespace kvaf {
struct AicpuExecutionResources::State {
  State(AicpuExecutionResources* owner, UmdkAccessOperatorConfig config)
      : legacy(owner), valid_mode(config.batch_get_mode == "gather_read" ||
                   config.batch_get_mode == "cpu_push_scatter"),
        get_session({config.kernel_json, config.workers,
                 config.peer_link, config.local_eid}),
        put_session({config.kernel_json, config.workers, config.peer_link, config.local_eid}),
        get(get_session, config.token_bytes),
        put(put_session), enable_get(config.enable_get), managed_completion(config.managed_completion) {
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
    if (config.batch_get_mode == "cpu_push_scatter")
      push = std::make_unique<cpu_push::CpuPushTransport>(cpu_push::Config{
          config.peer_link, config.kernel_json, config.local_eid, config.token_bytes, config.workers});
#else
    valid_mode = config.batch_get_mode == "gather_read";
#endif
    legacy.put = put.CreateExecutionState();
    legacy.get = get.CreateExecutionState();
    (void)put.BindExecutionState(legacy.put.get());
    (void)get.BindExecutionState(legacy.get.get());
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
    if (push) { legacy.push = push->CreateExecutionState(); (void)push->BindExecutionState(legacy.push.get()); }
#endif
  }
  ExecutionViewState views;
  AicpuInvocation legacy;
  AicpuInvocation* invocation{&legacy};
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
  bool managed_completion;
  RemoteRegionDesc region;
  bool initialized{false};
  void* warmup_buffer{nullptr};
  std::mutex get_mutex, put_mutex;
};

AicpuExecutionResources::AicpuExecutionResources(UmdkAccessOperatorConfig config)
    : state_(std::make_unique<State>(this, std::move(config))) { state_->views.Bind(&state_->legacy); }
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
  auto* call = state ? dynamic_cast<AicpuInvocation*>(state) : &state_->legacy;
  if (!call || call->owner != this)
    return Status::InvalidArgument("AICPU invocation belongs to another resource");
  if (state_->invocation != call && state_->invocation->output_bound)
    return Status::Busy("release Get output binding before replacing invocation");
  auto bound = state_->put.BindExecutionState(call->put.get());
  if (!bound.ok()) return bound;
  bound = state_->get.BindExecutionState(call->get.get());
  if (!bound.ok()) {
    (void)state_->put.BindExecutionState(state_->invocation->put.get());
    return bound;
  }
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
  if (state_->push) {
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
bool AicpuExecutionResources::managed_completion() const { return state_->managed_completion; }
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
  if (state_->managed_completion && request.execution.stream) {
    aclmdlRICaptureStatus capture = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    aclmdlRI graph = nullptr;
    if (aclmdlRICaptureGetInfo(static_cast<aclrtStream>(request.execution.stream), &capture, &graph) != ACL_SUCCESS)
      return Status::Internal("cannot query graph capture state");
    if (capture == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE)
      return Status::NotImplemented("per-call AICPU graph leases are not implemented");
  }
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
  if (state_->push) return state_->push->Prepare(request);
#endif
  if (state_->invocation->output_bound &&
      (request.request->output.data != state_->invocation->get_output.data ||
       request.request->output.bytes != state_->invocation->get_output.bytes))
    return Status::InvalidArgument("request differs from bound Get output");
  state_->invocation->get_output = request.request->output;
  return state_->get.Prepare(request);
}
Status AicpuExecutionResources::ReleasePutBuffers() { return state_->put_session.ReleaseBuffers(); }
Status AicpuExecutionResources::ReleaseGetBuffers() {
  auto& call = *state_->invocation;
  if (call.output_bound || !call.get_output.valid()) return Status::OK();
  auto s = state_->get_session.ReleaseBuffer(call.get_output.data, call.get_output.bytes);
  if (s.ok()) call.get_output = {};
  return s;
}
Status AicpuExecutionResources::PrepareGetOutput(const GetOutputOptions& options) {
  std::lock_guard<std::mutex> lock(state_->get_mutex);
  auto& call = *state_->invocation;
  if (!state_->initialized || !state_->enable_get || scatter() || call.output_bound ||
      !options.stream || !options.request_capacity ||
      !options.output.valid() || options.output.device != DeviceKind::kNpu)
    return Status::InvalidArgument("invalid Gather output binding");
  aclmdlRICaptureStatus capture = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
  aclmdlRI graph = nullptr;
  if (aclmdlRICaptureGetInfo(static_cast<aclrtStream>(options.stream), &capture, &graph) != ACL_SUCCESS)
    return Status::Internal("cannot query graph capture state");
  if (capture != ACL_MODEL_RI_CAPTURE_STATUS_NONE)
    return Status::InvalidState("bind Get output before graph capture");
  auto s = state_->DrainGet();
  if (s.ok()) s = state_->get.Warmup();
  if (!s.ok()) return s;
  call.get_output = options.output;
  s = state_->get_session.Ensure(state_->region.base, options.output.data, options.output.bytes);
  if (s.ok()) call.output_bound = true;
  return s;
}
Status AicpuExecutionResources::FinishGetOutput() {
  std::lock_guard<std::mutex> lock(state_->get_mutex);
  auto s = state_->DrainGet();
  if (!s.ok()) return s;
  auto& call = *state_->invocation;
  s = state_->get_session.ReleaseBuffer(call.get_output.data, call.get_output.bytes);
  if (s.ok()) { call.get_output = {}; call.output_bound = false; }
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
  if (state_->push) {
    auto checked = aicpu_urma::PrepareUmdkKernel(state_->get_session.config().kernel_json, true);
    if (!checked.ok()) return checked;
  }
#endif
  auto loaded = state_->enable_get ? state_->get_session.Load() : state_->put_session.Load();
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
  state_->views.Close();
  // Push shutdown must stop the remote DMA producer even after a kernel failure.
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
  if (state_->push) {
    auto stopped = state_->push->Shutdown();
    if (!stopped.ok()) return stopped;
  }
#endif
  auto drained = state_->DrainGet();
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
  state_->invocation->output_bound = false;
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
  auto released = state_->put_session.ReleaseBuffers();
  if (!released.ok()) return released;
  call.put_completion = {};
  call.write = {};
  return Status::OK();
}
Status AicpuExecutionResources::DrainGet() {
  std::lock_guard<std::mutex> lock(state_->get_mutex);
  auto status = state_->DrainGet();
  if (status.ok() && state_->managed_completion) status = ReleaseGetBuffers();
  return status;
}
}  // namespace kvaf
