#include "get/scatter/urma_aicpu/include/cpu_push_transport.h"
#include "get/common/aicpu/include/kvaf_aicpu_batch_get.h"
#include "aicpu_urma/kernel_loader.h"
#include "batch_get_args.h"
#include "aicpu_batch_get_capture.h"
#include "aicpu_batch_get_profile.h"
#include "direct_hccp/direct_hccp_session.h"
#include "direct_hccp/target_channel.h"
#include "peer_link.h"
#include "get_host_timing.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <utility>
#include <urma_api.h>

namespace kvaf::cpu_push {
namespace d = direct_hccp;
namespace {
Status Error(const char* operation) { return Status::Internal(std::string("CPU push: ") + operation); }
}  // namespace
struct CpuPushTransport::Impl {
  explicit Impl(Config c) : config(std::move(c)) {}
  Config config;
  RemoteRegionDesc source;
  d::DirectHccpApi api;
  d::DirectHccpSession* session{};
  void* context{};
  void* token{};
  void* memory_handle{};
  void* memory{};
  void* coordination{};
  void* status_device{};
  void* status_host{};
  void* profile_device{};
  void* profile_host{};
  std::array<std::unique_ptr<d::TargetChannel>, kJettyCount> channels;
  Layout layout;
  using Invocation = aicpu_urma::AicpuGetInvocation;
  ExecutionViewState views;
  Invocation default_invocation{this};
  Invocation* invocation{&default_invocation};
  bool peer_open{}, initialized{}, profiling{};
  ExecutionHealth health;

  Status Drain();
};
static Status DrainScatter(const aicpu_urma::ScatterLaunchView& view, aicpu_urma::AicpuGetInvocation& call) {
  const auto stream = call.pending ? call.pending : call.captured_stream;
  if (!stream) return view.health->failed ? (call.failure.ok() ?
      Status::InvalidState("CPU push failed; shutdown required") : call.failure) : Status::OK();
  bool capturing = false;
  auto allowed = detail::BatchGetCapture(stream, true, false, &capturing);
  if (!allowed.ok()) return allowed;
  const auto bytes = view.options.workers * sizeof(WorkerStatus);
  if (!view.health->failed && aclrtMemcpyAsync(view.completion.host, bytes, view.completion.device, bytes,
                       ACL_MEMCPY_DEVICE_TO_HOST, stream) != ACL_SUCCESS) {
    view.health->failed = true;
    return call.failure = Error("cannot enqueue execution status; resources retained");
  }
  if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    view.health->failed = true;
    return call.failure = Error("stream failed; resources retained");
  }
  call.pending = nullptr;
  if (view.health->failed) return call.failure.ok() ?
      Status::InvalidState("CPU push failed; shutdown required") : call.failure;
  const auto* statuses = static_cast<const WorkerStatus*>(view.completion.host);
  const auto sequence = statuses[0].sequence;
  if (call.graph_active && std::all_of(statuses, statuses + view.options.workers,
      [&](const WorkerStatus& status) { return status.sequence <= call.graph_start_sequence; }))
    return Status::InvalidState("Scatter graph has not replayed; replay or discard it");
  // Report the first real failure before checking agreement between workers.
  for (std::uint32_t w = 0; w < view.options.workers; ++w)
    if (statuses[w].magic == kMagic && statuses[w].sequence &&
        statuses[w].result != static_cast<std::uint64_t>(Result::kOk)) {
      view.health->failed = true;
      return call.failure = Status::Internal("CPU push worker " + std::to_string(w) + " result=" +
          std::to_string(statuses[w].result) + " sequence=" + std::to_string(statuses[w].sequence));
    }
  for (std::uint32_t w = 0; w < view.options.workers; ++w)
    if (statuses[w].magic != kMagic || !sequence || statuses[w].sequence != sequence ||
        statuses[w].result != static_cast<std::uint64_t>(Result::kOk)) {
      view.health->failed = true;
      return call.failure = Status::Internal("CPU push worker " + std::to_string(w) + " result=" +
          std::to_string(statuses[w].result) + " sequence=" + std::to_string(statuses[w].sequence));
    }
  return Status::OK();
}
CpuPushTransport::CpuPushTransport(Config config) : impl_(std::make_unique<Impl>(std::move(config))) {}
CpuPushTransport::~CpuPushTransport() {
  if (!Shutdown().ok()) (void)impl_.release();
}
std::unique_ptr<ExecutionState> CpuPushTransport::CreateExecutionState() {
  return std::make_unique<Impl::Invocation>(impl_.get());
}
Status CpuPushTransport::BindExecutionState(ExecutionState* state) {
  auto* typed = state ? dynamic_cast<Impl::Invocation*>(state) : &impl_->default_invocation;
  if (!typed || typed->owner != impl_.get()) return Status::InvalidArgument("Get invocation belongs to another resource");
  impl_->invocation = typed;
  impl_->views.Bind(typed);
  return Status::OK();
}
aicpu_urma::ScatterLaunchView CpuPushTransport::View() const {
  auto& s = *impl_;
  const auto* status = s.invocation->graph_completion.get();
  const auto* profile = s.invocation->graph_profile.get();
  return {s.views.View(ExecutionDirection::Get), {s.config.workers, s.config.token_bytes},
      s.memory, s.coordination,
      {status ? status->device : s.status_device, status ? status->host : s.status_host,
       s.config.workers * sizeof(WorkerStatus)},
      {profile ? profile->device : s.profile_device, profile ? profile->host : s.profile_host,
       s.config.workers * sizeof(BatchGetWorkerProfile)}, &s.health, s.profiling};
}
Status CpuPushTransport::Impl::Drain() {
  aicpu_urma::ScatterLaunchView view;
  view.options.workers = config.workers; view.health = &health;
  const auto* report = invocation->graph_completion.get();
  view.completion = {report ? report->device : status_device,
                     report ? report->host : status_host, config.workers * sizeof(WorkerStatus)};
  return DrainScatter(view, *invocation);
}
Status CpuPushTransport::Drain() { return impl_->Drain(); }
Status CpuPushTransport::PrepareGraph(const GetGraphOptions& options) {
  auto& s = *impl_;
  auto& call = *s.invocation;
  if (!s.initialized) return Status::NotInitialized("CPU push is not initialized");
  if (!options.stream || !options.output.data || !options.output.bytes ||
      options.output.device != DeviceKind::kNpu || !options.request_capacity ||
      options.request_capacity > kMaxRequests)
    return Status::InvalidArgument("invalid Scatter graph options");
  if (call.graph_active || call.captured_stream)
    return Status::Busy("Scatter capture already owns this execution state");
  bool capturing = false;
  auto ready = detail::BatchGetCapture(static_cast<aclrtStream>(options.stream), true, false, &capturing);
  if (!ready.ok()) return ready;
  ready = s.Drain();
  if (!ready.ok()) return ready;
  if (call.graph_node &&
      aclrtSynchronizeStream(static_cast<aclrtStream>(options.stream)) != ACL_SUCCESS)
    return Status::CompletionUnknown("Scatter graph preparation could not fence its stream");
  Coordination coordination{};
  if (aclrtMemcpy(&coordination, sizeof(coordination), s.coordination, sizeof(coordination),
                  ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
    return Error("cannot read Scatter graph ticket baseline");
  if (coordination.result.value || coordination.claimed.value % s.config.workers)
    return Status::InvalidState("Scatter ticket state is incomplete; shutdown required");
  if (call.graph_node) {
    call.graph_completion = std::make_unique<GetGraphReport>();
    ready = call.graph_completion->Allocate(s.config.workers * sizeof(WorkerStatus));
    if (ready.ok() && s.profiling) {
      call.graph_profile = std::make_unique<GetGraphReport>();
      ready = call.graph_profile->Allocate(s.config.workers * sizeof(BatchGetWorkerProfile));
    }
    if (!ready.ok()) return ready;
  }
  call.graph_start_sequence = coordination.claimed.value / s.config.workers;
  call.graph_stream = static_cast<aclrtStream>(options.stream);
  call.graph_active = true;
  call.submitted = call.profile_submitted = call.profile_complete = false;
  call.last_stream = nullptr;
  return Status::OK();
}
Status CpuPushTransport::FinishGraph(void* stream, bool discard) {
  auto& s = *impl_;
  auto& call = *s.invocation;
  if (!call.graph_active || stream != call.graph_stream)
    return Status::InvalidArgument("Scatter graph stream or lease does not match");
  bool capturing = false;
  auto ready = detail::BatchGetCapture(call.graph_stream, true, false, &capturing);
  if (!ready.ok()) return ready;
  // A failed fence is retryable. It does not prove that the graph or remote
  // producer has completed, so the caller must retain the execution lease.
  if (aclrtSynchronizeStream(call.graph_stream) != ACL_SUCCESS)
    return Status::CompletionUnknown("Scatter graph stream is still pending; lease retained");
  if (s.health.failed)
    return call.failure.ok() ? (call.failure = Status::InvalidState("Scatter graph failed; shutdown required")) : call.failure;
  Coordination coordination{};
  const auto completion = View().completion;
  if (aclrtMemcpy(&coordination, sizeof(coordination), s.coordination, sizeof(coordination),
                  ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
      aclrtMemcpy(completion.host, completion.bytes, completion.device, completion.bytes,
                  ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
    return Status::CompletionUnknown("Scatter graph status unavailable; lease retained");
  const auto fail = [&](const std::string& reason) {
    s.health.failed = true;
    return call.failure = Status::Internal(reason);
  };
  if (coordination.result.value)
    return fail("Scatter graph device result=" + std::to_string(coordination.result.value));
  const auto sequence = coordination.claimed.value / s.config.workers;
  if (coordination.claimed.value % s.config.workers || sequence < call.graph_start_sequence)
    return fail("Scatter graph ticket sequence is inconsistent");
  const auto* statuses = static_cast<const WorkerStatus*>(completion.host);
  const bool replayed = call.graph_node ? std::any_of(statuses, statuses + s.config.workers,
      [&](const auto& status) { return status.sequence > call.graph_start_sequence; })
      : sequence > call.graph_start_sequence;
  if (!replayed) {
    if (!discard) return Status::InvalidState("Scatter graph has not replayed; replay or discard it");
  } else {
    const auto node_sequence = call.graph_node ? statuses[0].sequence : sequence;
    if (node_sequence <= call.graph_start_sequence || node_sequence > sequence)
      return fail("Scatter graph status is inconsistent with shared tickets");
    for (std::uint32_t worker = 0; worker < s.config.workers; ++worker)
      if (statuses[worker].magic != kMagic || statuses[worker].sequence != node_sequence ||
          statuses[worker].result != static_cast<std::uint64_t>(Result::kOk))
        return fail("Scatter graph worker " + std::to_string(worker) + " result=" +
            std::to_string(statuses[worker].result) + " sequence=" +
            std::to_string(statuses[worker].sequence));
  }
  // Graph destruction/discard is the caller's responsibility. These buffers
  // and cumulative device tickets remain resident for the next execution.
  call.graph_active = false;
  call.graph_stream = call.captured_stream = call.pending = call.last_stream = nullptr;
  call.prepared = nullptr;
  call.prepared_stream = nullptr;
  call.prepared_profile = nullptr;
  call.profile_complete = call.profile_submitted && replayed;
  call.profile_submitted = call.profile_complete;
  call.prepared_capturing = call.submitted = false;
  return Status::OK();
}
Status CpuPushTransport::Check(void* stream) {
  auto& s = *impl_;
  if (!s.initialized || !s.invocation->last_stream)
    return Status::InvalidState("no AICPU BatchGet has been submitted");
  if (stream != s.invocation->last_stream)
    return Status::InvalidArgument("check_batch_get_status requires the BatchGet execution stream");
  s.invocation->pending = static_cast<aclrtStream>(stream);
  return s.Drain();
}
Status CpuPushTransport::SetKernelProfiling(bool enabled) {
  auto& s = *impl_;
  if (!s.initialized) return Status::NotInitialized("CPU push is not initialized");
  auto ready = s.Drain();
  if (!ready.ok()) return ready;
  if (enabled) {
    ready = detail::AllocateAicpuProfiles(&s.profile_device, &s.profile_host,
        s.config.workers * sizeof(BatchGetWorkerProfile));
    if (!ready.ok()) return ready;
  }
  s.profiling = enabled;
  s.invocation->profile_submitted = s.invocation->profile_complete = false;
  return Status::OK();
}
Status CpuPushTransport::CopyKernelProfiles(std::vector<BatchGetKernelProfile>* profiles) const {
  const auto& s = *impl_;
  if (!s.profiling || !s.invocation->profile_submitted)
    return Status::InvalidState("no profiled CPU push launch");
  if (!s.invocation->profile_complete) {
    bool capturing = false;
    auto ready = detail::BatchGetCapture(s.invocation->last_stream, true, false, &capturing);
    if (!ready.ok()) return ready;
  }
  const auto profile = View().profile;
  return detail::CopyAicpuProfiles(profile.device, profile.host, s.config.workers, profiles);
}

Status CpuPushTransport::Initialize(const RemoteRegionDesc& source) {
  auto& s = *impl_;
  if (s.context || s.memory || s.initialized) return Status::InvalidState("CPU push already initialized");
  if (!s.config.peer_link || !MakeLayout(s.config.token_bytes, s.config.workers, &s.layout) ||
      std::all_of(s.config.local_eid.begin(), s.config.local_eid.end(), [](auto b) { return b == 0; }) ||
      !aicpu_urma::LoadUmdkKernel(s.config.kernel_json.c_str()))
    return Status::InvalidArgument("CPU push requires kernel JSON, PeerLink, local EID and valid shape");
  s.source = source;
  peer::PeerRegion region;
  auto result = s.config.peer_link->LookupRegion(source.base, &region);
  if (!result.ok()) return result;
  std::uint32_t physical = 0;
  if (!d::LoadDirectHccp(&s.api) || !d::GetCurrentNpuPhysicalDeviceId(&physical) ||
      !d::AcquireDirectHccpSession(physical, 18, &s.session)) return Error("acquire direct HCCP session");
  d::CtxInitCfg cfg{};
  cfg.mode = d::kOfflineHdc;
  d::CtxInitAttr attr{};
  attr.phy_id = physical;
  std::copy(s.config.local_eid.begin(), s.config.local_eid.end(), attr.ub.eid.raw);
  if (s.api.ctx_init(&cfg, &attr, &s.context) || !s.context) return Error("create RA context");
  d::TokenId token{};
  if (s.api.token_alloc(s.context, &token, &s.token) || !s.token) return Error("allocate RA token");
  const auto status_bytes = s.config.workers * sizeof(WorkerStatus);
  if (aclrtMalloc(&s.memory, s.layout.bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
      aclrtMemset(s.memory, s.layout.bytes, 0, s.layout.staging) != ACL_SUCCESS ||
      aclrtMalloc(&s.coordination, sizeof(Coordination), ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS ||
      aclrtMalloc(&s.status_device, status_bytes, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS ||
      aclrtMallocHost(&s.status_host, status_bytes) != ACL_SUCCESS ||
      aclrtMemset(s.coordination, sizeof(Coordination), 0, sizeof(Coordination)) != ACL_SUCCESS ||
      aclrtMemset(s.status_device, status_bytes, 0, status_bytes) != ACL_SUCCESS)
    return Error("allocate workspace");
  // Match the probe's HBM registration contract; the token is session-local.
  const auto token_value = region.token_value;
  d::MrRegInfo mr{};
  mr.in.mem.addr = reinterpret_cast<std::uint64_t>(s.memory);
  mr.in.mem.size = s.layout.bytes;
  mr.in.ub.flags = d::MemRegFlags(true);
  mr.in.ub.token_value = token_value;
  mr.in.ub.token_id_handle = s.token;
  if (s.api.lmem_register(s.context, &mr, &s.memory_handle) || !s.memory_handle)
    return Error("register HBM workspace");
  static_assert(sizeof(urma_seg_t) <= peer::kSegKeyBytes);
  if (mr.out.key.size < sizeof(urma_seg_t) || mr.out.key.size > peer::kSegKeyBytes)
    return Error("unexpected HBM segment ABI");
  urma_seg_t segment{};
  std::memcpy(&segment, mr.out.key.value, sizeof(segment));
  peer::OpenCpuPushGetRequest open{};
  open.source_bytes = source.length;
  open.hbm_base = segment.ubva.va;
  open.hbm_bytes = s.layout.bytes;
  open.npu_psn = 1;
  open.token_value = token_value;
  open.segment_bytes = sizeof(segment);
  open.token_bytes = s.config.token_bytes;
  open.workers = s.config.workers;
  std::copy(s.config.local_eid.begin(), s.config.local_eid.end(), open.eid);
  std::memcpy(open.segment, mr.out.key.value, open.segment_bytes);
  std::uint64_t npu_tp = 0;
  if (!d::GetPeerTp(s.api, s.context, s.config.local_eid, region.eid, "cpu_push_get", &npu_tp))
    return Error("query NPU CTP");
  open.npu_tp = npu_tp;
  for (std::uint32_t q = 0; q < kJettyCount; ++q) {
    s.channels[q] = std::make_unique<d::TargetChannel>(s.api, s.context);
    result = s.channels[q]->Initialize(s.token, token_value, 1024, 1024, 16384);
    if (!result.ok()) return result;
    const auto& key = s.channels[q]->jetty_info().key;
    open.channels[q].bytes = key.size;
    std::memcpy(open.channels[q].key, key.value, key.size);
  }
  peer::OpenCpuPushGetResponse reply{};
  // Even a failed/lost response may have created CPU resources. STOP is idempotent.
  s.peer_open = true;
  result = s.config.peer_link->OpenCpuPushGet(source.base, open, &reply);
  if (!result.ok()) return result;
  for (std::uint32_t q = 0; q < kJettyCount; ++q) {
    d::QpImportInfoT info{};
    if (!d::ImportPeerJetty(s.api, s.context, reply.channels[q].key, reply.channels[q].bytes,
        token_value, open.npu_tp, reply.cpu_tp, open.npu_psn, reply.cpu_psn, "cpu_push_get",
        &info, s.channels[q]->peer_jetty_output())) return Error("import CPU Jetty");
  }
  result = s.config.peer_link->StartCpuPushGet(source.base);
  if (!result.ok()) return result;
  s.initialized = true;
  s.views.Bind(s.invocation);
  s.views.Initialize(s.source);
  std::fprintf(stderr, "[KVAF_AICPU] CPU_PUSH_REFERENCE commit=%s protocol=3 "
      "copy=per_row_memcpy_sp barrier=dsb_sy device_optimization=O3 "
      "workspace_policy=huge_first\n", kReferenceCommit);
  return Status::OK();
}

Status CpuPushTransport::Prepare(const BatchGetExecRequest& exec, std::optional<bool> expected_capture) {
  impl_->views.Prepare();
  auto& s = *impl_;
  auto& call = *s.invocation;
  call.prepared = nullptr;
  if (exec.execution.profile && !exec.execution.synchronize)
    return Status::InvalidArgument("profiling requires synchronize=true");
  call.prepare_started = exec.execution.profile ? BatchGetProfileNowNs() : 0;
  if (exec.execution.profile) *exec.execution.profile = {};
  if (!s.initialized || !exec.execution.stream)
    return Status::InvalidState("CPU push needs initialization and stream");
  if (s.health.failed) return Status::InvalidState("CPU push failed; shutdown required");
  const auto stream = static_cast<aclrtStream>(exec.execution.stream);
  call.profile_complete = false;
  bool capturing = false;
  auto ready = detail::BatchGetCapture(stream, exec.execution.synchronize,
                                       exec.execution.profile != nullptr, &capturing, expected_capture);
  if (!ready.ok()) return ready;
  if (call.captured_stream && call.captured_stream != stream)
    return Status::InvalidArgument("captured AICPU transport must use its capture stream");
  if (capturing) {
    if (call.pending && call.pending != stream)
      return Status::InvalidState("drain previous AICPU work outside capture on its stream");
  } else {
    ready = s.Drain();
    if (!ready.ok()) return ready;
  }
  KernelArgs checked{};
  ready = detail::BuildBatchGetKernelArgs(exec, &checked.request);
  if (!ready.ok()) return ready;
  if (checked.request.request_capacity > kMaxRequests || checked.request.token_bytes != s.config.token_bytes)
    return Status::InvalidArgument("CPU push exceeds request capacity or configured token size");
  const auto bytes = s.config.workers * sizeof(BatchGetWorkerProfile);
  if (exec.execution.profile || s.profiling) {
    if (capturing && (!s.profile_device || !s.profile_host))
      return Status::InvalidState("enable kernel profiling before capture");
    ready = detail::AllocateAicpuProfiles(&s.profile_device, &s.profile_host, bytes);
    if (!ready.ok()) return ready;
  }
  call.prepared_capturing = capturing;
  call.prepared = exec.request;
  call.prepared_stream = exec.execution.stream;
  call.prepared_profile = exec.execution.profile;
  return Status::OK();
}
Status CpuPushTransport::Execute(const BatchGetExecRequest& exec) {
  auto ready = Prepare(exec);
  return ready.ok() ? ExecutePrepared(exec) : ready;
}
Status CpuPushTransport::ExecutePrepared(const BatchGetExecRequest& exec) {
  return LaunchPrepared(exec, View(), impl_->invocation);
}
Status CpuPushTransport::LaunchPrepared(const BatchGetExecRequest& exec,
                                        const aicpu_urma::ScatterLaunchView& view, ExecutionState* state) {
  const auto submit_started = GetHostTimingStart(exec.host_timing);
  auto* invocation = dynamic_cast<aicpu_urma::AicpuGetInvocation*>(state);
  auto valid_view = view.common.Validate(invocation, ExecutionDirection::Get);
  if (!valid_view.ok()) return valid_view;
  auto* profile = exec.execution.profile;
  if (profile && !exec.execution.synchronize) return Status::InvalidArgument("profiling requires synchronize=true");
  auto& call = *invocation;
  if (!call.prepared || call.prepared != exec.request || call.prepared_stream != exec.execution.stream ||
      call.prepared_profile != profile)
    return Status::InvalidState("Scatter resources were not prepared for this request");
  call.prepared = nullptr;
  const auto begin = call.prepare_started;
  Status result;

  KernelArgs args{};
  result = detail::BuildBatchGetKernelArgs(exec, &args.request);
  args.request.canonical_base = view.common.region().canonical_base;
  args.request.logical_bytes = view.common.region().logical_bytes;
  if (!result.ok()) return result;
  if (args.request.request_capacity > kMaxRequests || args.request.token_bytes != view.options.token_bytes)
    return Status::InvalidArgument("CPU push exceeds request capacity or configured token size");
  const auto profile_bytes = view.options.workers * sizeof(BatchGetWorkerProfile);
  args.workspace = detail::PointerValue(view.workspace);
  args.output = detail::PointerValue(exec.request->output.data);
  args.output_bytes = exec.request->output.bytes;
  args.coordination = detail::PointerValue(view.coordination);
  args.status = detail::PointerValue(view.completion.device);
  args.profile = (profile || view.profiling) ? detail::PointerValue(view.profile.device) : 0;
  args.workers = view.options.workers;
  const auto stream = static_cast<aclrtStream>(exec.execution.stream);
  call.pending = call.last_stream = stream;
  if (call.prepared_capturing) call.captured_stream = stream;
  auto stamp = profile ? BatchGetProfileNowNs() : 0;
  if (profile) profile->prepare_ns = stamp - begin;
  if (LaunchCpuPushScatter(&args, stream)) {
    view.health->failed = true;
    return call.failure = Error("launch kernel");
  }
  call.submitted = true;
  call.profile_submitted = args.profile != 0;
  GetHostTimingRecord(exec.host_timing, &GetProviderTiming::submit_ns, submit_started);
  const auto completion_started = GetHostTimingStart(exec.host_timing);
  if (profile) { const auto now = BatchGetProfileNowNs(); profile->launch_ns = now - stamp; stamp = now; }
  if (!exec.execution.synchronize) return Status::OK();
  if (profile && aclrtMemcpyAsync(view.profile.host, profile_bytes, view.profile.device, profile_bytes,
                                 ACL_MEMCPY_DEVICE_TO_HOST, stream) != ACL_SUCCESS) {
    view.health->failed = true; return call.failure = Error("enqueue profiling result");
  }
  if (profile) { const auto now = BatchGetProfileNowNs(); profile->status_enqueue_ns = now - stamp; stamp = now; }
  result = DrainScatter(view, call);
  if (!result.ok()) return result;
  if (profile) {
    const auto now = BatchGetProfileNowNs(); profile->wait_ns = now - stamp; stamp = now;
    profile->worker_count = view.options.workers;
    std::copy_n(static_cast<const BatchGetWorkerProfile*>(view.profile.host), view.options.workers, profile->workers.begin());
    const auto end = BatchGetProfileNowNs(); profile->finish_ns = end - stamp; profile->total_ns = end - begin;
  }
  GetHostTimingRecord(exec.host_timing, &GetProviderTiming::completion_ns, completion_started);
  return Status::OK();
}

Status CpuPushTransport::Shutdown() {
  impl_->views.Close();
  auto& s = *impl_;
  // Completed eager work no longer depends on its caller-owned stream.
  if (const auto stream = s.invocation->pending ? s.invocation->pending : s.invocation->captured_stream) {
    bool capturing = false;
    auto allowed = detail::BatchGetCapture(stream, true, false, &capturing);
    if (!allowed.ok()) return allowed;
  }
  // A healthy enqueued request still needs the CPU producer. Drain it before
  // STOP; after an execution failure, STOP must run even if draining failed.
  const auto drained = s.health.failed ? Status::OK() : s.Drain();
  // Stop the remote initiator before releasing any HBM, even after kernel failure.
  if (s.peer_open) {
    auto stopped = s.config.peer_link->StopCpuPushGet(s.source.base);
    if (!stopped.ok()) return stopped;
    s.peer_open = false;
  }
  const auto remaining = s.invocation->pending ? s.invocation->pending : s.invocation->captured_stream;
  if (remaining && aclrtSynchronizeStream(remaining) != ACL_SUCCESS)
    return Error("stream still failed/pending; HBM retained");
  s.invocation->pending = nullptr;
  for (auto& channel : s.channels) if (channel) {
    auto closed = channel->Shutdown();
    if (!closed.ok()) return closed;
    channel.reset();
  }
  if (s.memory_handle) {
    if (s.api.lmem_unregister(s.context, s.memory_handle)) return Error("unregister HBM");
    s.memory_handle = nullptr;
  }
  if (s.token) { if (s.api.token_free(s.context, s.token)) return Error("free RA token"); s.token = nullptr; }
  if (s.context) { if (s.api.ctx_deinit(s.context)) return Error("destroy RA context"); s.context = nullptr; }
  if (s.session) { d::ReleaseDirectHccpSession(s.session); s.session = nullptr; }
  d::CloseDirectHccp(&s.api);
  for (auto buffer : {&s.memory, &s.coordination, &s.status_device, &s.profile_device}) {
    if (*buffer && aclrtFree(*buffer) != ACL_SUCCESS) return Error("free device buffer");
    *buffer = nullptr;
  }
  for (auto buffer : {&s.status_host, &s.profile_host}) {
    if (*buffer && aclrtFreeHost(*buffer) != ACL_SUCCESS) return Error("free Host buffer");
    *buffer = nullptr;
  }
  s.invocation->last_stream = s.invocation->captured_stream = s.invocation->graph_stream = nullptr;
  s.invocation->graph_active = false;
  s.initialized = s.health.failed = false;
  s.profiling = s.invocation->profile_submitted = s.invocation->profile_complete =
      s.invocation->submitted = false;
  s.invocation->failure = Status::OK();
  return drained;
}
}  // namespace kvaf::cpu_push
