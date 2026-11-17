#include "aicpu_batch_get/include/cpu_push_transport.h"
#include "aicpu_batch_get/include/kvaf_aicpu_batch_get.h"
#include "aicpu_urma/kernel_loader.h"
#include "batch_get_args.h"
#include "aicpu_batch_get_capture.h"
#include "aicpu_batch_get_profile.h"
#include "direct_hccp/direct_hccp_session.h"
#include "direct_hccp/target_channel.h"
#include "peer_link.h"

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
  aclrtStream pending{};
  aclrtStream captured_stream{}, last_stream{};
  bool peer_open{}, initialized{}, failed{};
  bool profiling{}, profile_submitted{};
  Status failure;

  Status Drain() {
    const auto stream = pending ? pending : captured_stream;
    if (!stream) return failed ? failure : Status::OK();
    bool capturing = false;
    auto allowed = detail::BatchGetCapture(stream, true, false, &capturing);
    if (!allowed.ok()) return allowed;
    const auto bytes = config.workers * sizeof(WorkerStatus);
    if (!failed && aclrtMemcpyAsync(status_host, bytes, status_device, bytes,
                         ACL_MEMCPY_DEVICE_TO_HOST, stream) != ACL_SUCCESS) {
      failed = true;
      return failure = Error("cannot enqueue execution status; resources retained");
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
      failed = true;
      return failure = Error("stream failed; resources retained");
    }
    pending = nullptr;
    if (failed) return failure.ok() ? Status::InvalidState("CPU push failed; shutdown required") : failure;
    const auto* statuses = static_cast<const WorkerStatus*>(status_host);
    const auto sequence = statuses[0].sequence;
    for (std::uint32_t w = 0; w < config.workers; ++w)
      if (statuses[w].magic == kMagic && statuses[w].result != std::uint64_t(Result::kOk) &&
          statuses[w].sequence) {
        failed = true;
        return failure = Status::Internal("CPU push worker " + std::to_string(w) + " result=" +
            std::to_string(statuses[w].result) + " sequence=" + std::to_string(statuses[w].sequence));
      }
    for (std::uint32_t w = 0; w < config.workers; ++w)
      if (statuses[w].magic != kMagic || !sequence || statuses[w].sequence != sequence ||
          statuses[w].result != static_cast<std::uint64_t>(Result::kOk)) {
        failed = true;
        return failure = Status::Internal("CPU push worker " + std::to_string(w) + " result=" +
                                std::to_string(statuses[w].result) + " sequence=" +
                                std::to_string(statuses[w].sequence));
      }
    return Status::OK();
  }
};
CpuPushTransport::CpuPushTransport(Config config) : impl_(std::make_unique<Impl>(std::move(config))) {}
CpuPushTransport::~CpuPushTransport() {
  if (!Shutdown().ok()) (void)impl_.release();
}
Status CpuPushTransport::Drain() { return impl_->Drain(); }
Status CpuPushTransport::Check(void* stream) {
  auto& s = *impl_;
  if (!s.initialized || !s.last_stream)
    return Status::InvalidState("no AICPU BatchGet has been submitted");
  if (stream != s.last_stream)
    return Status::InvalidArgument("check_batch_get_status requires the BatchGet execution stream");
  s.pending = static_cast<aclrtStream>(stream);
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
  s.profile_submitted = false;
  return Status::OK();
}
Status CpuPushTransport::CopyKernelProfiles(std::vector<BatchGetKernelProfile>* profiles) const {
  const auto& s = *impl_;
  if (!s.profiling || !s.profile_submitted)
    return Status::InvalidState("no profiled CPU push launch");
  bool capturing = false;
  auto ready = detail::BatchGetCapture(s.last_stream, true, false, &capturing);
  if (!ready.ok()) return ready;
  return detail::CopyAicpuProfiles(s.profile_device, s.profile_host, s.config.workers, profiles);
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
  std::fprintf(stderr, "[KVAF_AICPU] CPU_PUSH_REFERENCE commit=%s protocol=3 "
      "copy=per_row_memcpy_sp barrier=dsb_sy device_optimization=O3 "
      "workspace_policy=huge_first\n", kReferenceCommit);
  return Status::OK();
}

Status CpuPushTransport::Execute(const BatchGetExecRequest& exec) {
  auto& s = *impl_;
  auto* profile = exec.execution.profile;
  if (profile && !exec.execution.synchronize) return Status::InvalidArgument("profiling requires synchronize=true");
  if (profile) *profile = {};
  const auto begin = profile ? BatchGetProfileNowNs() : 0;
  if (!s.initialized || !exec.execution.stream) return Status::InvalidState("CPU push needs initialization and stream");
  if (s.failed) return Status::InvalidState("CPU push failed; shutdown required");
  const auto stream = static_cast<aclrtStream>(exec.execution.stream);
  bool capturing = false;
  auto result = detail::BatchGetCapture(stream, exec.execution.synchronize, profile != nullptr, &capturing);
  if (!result.ok()) return result;
  if (s.captured_stream && s.captured_stream != stream)
    return Status::InvalidArgument("captured AICPU transport must use its capture stream");
  if (capturing) {
    if (s.pending && s.pending != stream)
      return Status::InvalidState("drain previous AICPU work outside capture on its stream");
  } else {
    result = s.Drain();
    if (!result.ok()) return result;
  }
  KernelArgs args{};
  result = detail::BuildBatchGetKernelArgs(exec, &args.request);
  if (!result.ok()) return result;
  if (args.request.request_capacity > kMaxRequests || args.request.token_bytes != s.config.token_bytes)
    return Status::InvalidArgument("CPU push exceeds request capacity or configured token size");
  const auto profile_bytes = s.config.workers * sizeof(BatchGetWorkerProfile);
  if (profile || s.profiling) {
    auto allocated = detail::AllocateAicpuProfiles(&s.profile_device, &s.profile_host, profile_bytes);
    if (!allocated.ok()) return allocated;
  }
  args.workspace = detail::PointerValue(s.memory);
  args.output = detail::PointerValue(exec.request->output.data);
  args.output_bytes = exec.request->output.bytes;
  args.coordination = detail::PointerValue(s.coordination);
  args.status = detail::PointerValue(s.status_device);
  args.profile = (profile || s.profiling) ? detail::PointerValue(s.profile_device) : 0;
  args.workers = s.config.workers;
  s.pending = stream;
  s.last_stream = stream;
  if (capturing) s.captured_stream = stream;
  auto stamp = profile ? BatchGetProfileNowNs() : 0;
  if (profile) profile->prepare_ns = stamp - begin;
  if (LaunchCpuPushScatter(&args, stream)) { s.failed = true; return Error("launch kernel"); }
  s.profile_submitted = args.profile != 0;
  if (profile) { const auto now = BatchGetProfileNowNs(); profile->launch_ns = now - stamp; stamp = now; }
  if (!exec.execution.synchronize) return Status::OK();
  if (profile && aclrtMemcpyAsync(s.profile_host, profile_bytes, s.profile_device, profile_bytes,
                                 ACL_MEMCPY_DEVICE_TO_HOST, stream) != ACL_SUCCESS) {
    s.failed = true; return Error("enqueue profiling result");
  }
  if (profile) { const auto now = BatchGetProfileNowNs(); profile->status_enqueue_ns = now - stamp; stamp = now; }
  result = s.Drain();
  if (!result.ok()) return result;
  if (profile) {
    const auto now = BatchGetProfileNowNs(); profile->wait_ns = now - stamp; stamp = now;
    profile->worker_count = s.config.workers;
    std::copy_n(static_cast<const BatchGetWorkerProfile*>(s.profile_host), s.config.workers, profile->workers.begin());
    const auto end = BatchGetProfileNowNs(); profile->finish_ns = end - stamp; profile->total_ns = end - begin;
  }
  return Status::OK();
}

Status CpuPushTransport::Shutdown() {
  auto& s = *impl_;
  // Completed eager work no longer depends on its caller-owned stream.
  if (const auto stream = s.pending ? s.pending : s.captured_stream) {
    bool capturing = false;
    auto allowed = detail::BatchGetCapture(stream, true, false, &capturing);
    if (!allowed.ok()) return allowed;
  }
  // A healthy enqueued request still needs the CPU producer. Drain it before
  // STOP; after an execution failure, STOP must run even if draining failed.
  const auto drained = s.failed ? Status::OK() : s.Drain();
  // Stop the remote initiator before releasing any HBM, even after kernel failure.
  if (s.peer_open) {
    auto stopped = s.config.peer_link->StopCpuPushGet(s.source.base);
    if (!stopped.ok()) return stopped;
    s.peer_open = false;
  }
  const auto remaining = s.pending ? s.pending : s.captured_stream;
  if (remaining && aclrtSynchronizeStream(remaining) != ACL_SUCCESS)
    return Error("stream still failed/pending; HBM retained");
  s.pending = nullptr;
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
  s.last_stream = s.captured_stream = nullptr;
  s.initialized = s.failed = false;
  s.profiling = s.profile_submitted = false;
  s.failure = Status::OK();
  return drained;
}
}  // namespace kvaf::cpu_push
