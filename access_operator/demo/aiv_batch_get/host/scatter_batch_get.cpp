#include "scatter_batch_get.h"
#include "sequential_batch_get_common.h"
#include "kvaf_aiv_batch_get.h"
#include "cpu_push_get_layout.h"
#include "direct_hccp/direct_hccp_session.h"
#include "direct_hccp/target_channel.h"
#include "peer_link.h"
#include <acl/acl.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <mutex>
#include <utility>

namespace kvaf {
namespace d = direct_hccp;
using namespace sequential_detail;
namespace {
Status Error(const char* what) { return Status::Internal(std::string("AIV sequential BatchGet: ") + what); }
// The exported segment is the public 48-byte urma_seg_t wire ABI; no liburma is
// loaded on the NPU Host or AIV. ubva.va is packed at byte 20, len at byte 32.
constexpr uint32_t kSegmentBytes = 48;
}

class ScatterBatchGet final : public SequentialBatchGet {
 public:
  explicit ScatterBatchGet(AccessOperatorConfig config);
  ~ScatterBatchGet() override;
  Status Initialize(const RemoteRegionDesc& region) override;
  Status Execute(const BatchGetExecRequest& exec) override;
  Status Drain() override;
  Status Shutdown() override;
  Status SetScatterProfiling(bool enabled) override;
  Status CopyScatterProfiles(std::vector<ScatterBatchGetCoreProfile>* profiles) override;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct ScatterBatchGet::Impl {
  explicit Impl(AccessOperatorConfig c) : config(std::move(c)) {}
  AccessOperatorConfig config;
  std::mutex mutex;
  RemoteRegionDesc region{};
  KvafScatterBatchGetArgs args{};
  void* workspace{};
  void* status_device{};
  void* status_host{};
  d::DirectHccpApi api{};
  d::DirectHccpSession* session{};
  void* context{};
  void* token{};
  void* memory_handle{};
  std::array<std::unique_ptr<d::TargetChannel>, 4> channels;
  aclrtStream pending{};
  uint64_t sequence{};
  bool push_open{}, initialized{}, failed{}, profile_ready{};
  uint64_t StatusBytes() const { return config.npu_core_count * sizeof(KvafSequentialStatus); }
  uint64_t ProfileBytes() const { return config.npu_core_count * sizeof(ScatterBatchGetCoreProfile); }

  Status AclFailure(const char* phase, aclError code) const {
    return Status::DeviceError("AIV sequential BatchGet: mode=" + config.aiv_batch_get_mode +
        " sequence=" + std::to_string(sequence) + " phase=" + phase +
        " acl=" + std::to_string(code) +
        "; resources retained");
  }

  Status Drain() {
    if (failed) return Status::InvalidState("sequential BatchGet failed; shutdown/reinitialize required");
    if (!pending) return Status::OK();
    const aclError copied = aclrtMemcpyAsync(status_host, StatusBytes(), status_device,
                                            StatusBytes(), ACL_MEMCPY_DEVICE_TO_HOST, pending);
    if (copied != ACL_SUCCESS) {
      failed = true; return AclFailure("copy_status", copied);
    }
    const aclError synchronized = aclrtSynchronizeStream(pending);
    if (synchronized != ACL_SUCCESS) {
      failed = true; return AclFailure("synchronize", synchronized);
    }
    pending = nullptr;
    const auto* statuses = static_cast<const KvafSequentialStatus*>(status_host);
    for (uint32_t i = 0; i < config.npu_core_count; ++i) {
      const auto& s = statuses[i];
      if (s.magic != KVAF_SEQ_STATUS_MAGIC || s.sequence != sequence ||
          s.result != KVAF_SEQ_OK || s.phase != KVAF_SEQ_COMPLETE) {
        failed = true;
        return Status::Internal("AIV sequential BatchGet worker=" + std::to_string(i) +
            " sequence=" + std::to_string(sequence) + " observed_sequence=" +
            std::to_string(s.sequence) + " phase=" + std::to_string(s.phase) +
            " result=" + std::to_string(s.result));
      }
    }
    return Status::OK();
  }

  Status InitializePush(const peer::PeerRegion& peer) {
    cpu_push::Layout layout;
    if (!cpu_push::MakeLayout(config.batch_get_token_bytes, config.npu_core_count, &layout))
      return Status::InvalidArgument("invalid scatter workspace shape");
    args.descriptor_offset = layout.descriptor; args.staging_offset = layout.staging;
    // Last private line is an AIV-owned build failure marker, outside CPU layout.
    args.workspace_bytes = layout.bytes + 64;
    if (aclrtMalloc(&workspace, args.workspace_bytes, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS ||
        aclrtMemset(workspace, args.workspace_bytes, 0, args.workspace_bytes) != ACL_SUCCESS)
      return Error("allocate scatter workspace");
    uint32_t physical = 0;
    if (!d::LoadDirectHccp(&api) || !d::GetCurrentNpuPhysicalDeviceId(&physical) ||
        !d::AcquireDirectHccpSession(physical, 18, &session)) return Error("acquire HCCP session");
    d::CtxInitCfg cfg{}; cfg.mode = d::kOfflineHdc;
    d::CtxInitAttr attr{}; attr.phy_id = physical;
    std::copy(config.peer_local_eid.begin(), config.peer_local_eid.end(), attr.ub.eid.raw);
    if (api.ctx_init(&cfg, &attr, &context) || !context) return Error("create HCCP context");
    d::TokenId id{};
    if (api.token_alloc(context, &id, &token) || !token) return Error("allocate HCCP token");
    d::MrRegInfo mr{};
    mr.in.mem.addr = Ptr(workspace); mr.in.mem.size = args.workspace_bytes;
    mr.in.ub.flags = d::MemRegFlags(true); mr.in.ub.token_value = peer.token_value;
    mr.in.ub.token_id_handle = token;
    if (api.lmem_register(context, &mr, &memory_handle) || !memory_handle ||
        mr.out.key.size < kSegmentBytes || mr.out.key.size > peer::kSegKeyBytes)
      return Error("register scatter HBM segment");
    peer::OpenCpuPushGetRequest open{};
    open.source_bytes = region.length; open.hbm_bytes = args.workspace_bytes;
    std::memcpy(&open.hbm_base, mr.out.key.value + 20, sizeof(open.hbm_base));
    uint64_t exported_bytes = 0;
    std::memcpy(&exported_bytes, mr.out.key.value + 32, sizeof(exported_bytes));
    if (exported_bytes < args.workspace_bytes || !open.hbm_base)
      return Error("invalid exported HBM segment ABI");
    open.npu_psn = 1; open.token_value = peer.token_value; open.segment_bytes = kSegmentBytes;
    open.token_bytes = config.batch_get_token_bytes; open.workers = config.npu_core_count;
    std::copy(config.peer_local_eid.begin(), config.peer_local_eid.end(), open.eid);
    std::memcpy(open.segment, mr.out.key.value, kSegmentBytes);
    // The wire request is packed; do not pass an unaligned field as uint64_t*.
    uint64_t npu_tp = 0;
    if (!d::GetPeerTp(api, context, config.peer_local_eid, peer.eid, "aiv_scatter", &npu_tp))
      return Error("query scatter TP");
    open.npu_tp = npu_tp;
    for (uint32_t q = 0; q < 4; ++q) {
      channels[q] = std::make_unique<d::TargetChannel>(api, context);
      auto result = channels[q]->Initialize(token, peer.token_value, 1024, 1024, 16384);
      if (!result.ok()) return result;
      const auto& key = channels[q]->jetty_info().key;
      open.channels[q].bytes = key.size;
      std::memcpy(open.channels[q].key, key.value, key.size);
    }
    peer::OpenCpuPushGetResponse response{};
    push_open = true;  // A lost response can still leave a live remote session.
    auto result = config.peer_link->OpenCpuPushGet(region.base, open, &response);
    if (!result.ok()) return result;
    for (uint32_t q = 0; q < 4; ++q) {
      d::QpImportInfoT info{};
      if (!d::ImportPeerJetty(api, context, response.channels[q].key, response.channels[q].bytes,
          peer.token_value, open.npu_tp, response.cpu_tp, open.npu_psn, response.cpu_psn,
          "aiv_scatter", &info, channels[q]->peer_jetty_output())) return Error("import CPU Jetty");
    }
    return config.peer_link->StartCpuPushGet(region.base);
  }

  Status Shutdown() {
    // A normal close must let the CPU finish serving the pending request first.
    // On execution errors, keep closing the transports before releasing memory.
    const auto execution_status = failed ? Status::OK() : Drain();
    failed = true;  // A partially closed transport cannot accept another Execute.
    if (pending) {
      const aclError synchronized = aclrtSynchronizeStream(pending);
      if (synchronized != ACL_SUCCESS) return AclFailure("shutdown_synchronize", synchronized);
    }
    pending = nullptr;
    if (push_open) {
      auto result = config.peer_link->StopCpuPushGet(region.base);
      if (!result.ok()) return result;
      push_open = false;
    }
    for (auto& channel : channels) if (channel) {
      auto result = channel->Shutdown();
      if (!result.ok()) return result;
      channel.reset();
    }
    if (memory_handle) {
      if (api.lmem_unregister(context, memory_handle)) return Error("unregister scatter HBM");
      memory_handle = nullptr;
    }
    if (token) { if (api.token_free(context, token)) return Error("release token"); token = nullptr; }
    if (context) { if (api.ctx_deinit(context)) return Error("release context"); context = nullptr; }
    if (session) { d::ReleaseDirectHccpSession(session); session = nullptr; }
    d::CloseDirectHccp(&api);
    for (auto p : {&workspace, &status_device}) {
      if (*p && aclrtFree(*p) != ACL_SUCCESS) return Error("free device memory");
      *p = nullptr;
    }
    if (status_host && aclrtFreeHost(status_host) != ACL_SUCCESS) return Error("free status memory");
    status_host = nullptr;
    initialized = failed = profile_ready = false; sequence = 0; args = {};
    return execution_status;
  }
};

ScatterBatchGet::ScatterBatchGet(AccessOperatorConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}
ScatterBatchGet::~ScatterBatchGet() {
  if (!Shutdown().ok()) (void)impl_.release();
}
Status ScatterBatchGet::Initialize(const RemoteRegionDesc& region) {
  auto& s = *impl_;
  std::lock_guard<std::mutex> lock(s.mutex);
  if (s.initialized || s.failed || s.workspace || s.status_device || s.status_host)
    return Status::InvalidState("sequential transport already initialized; shutdown before retry");
  if (!s.config.peer_link || !s.config.npu_core_count || s.config.npu_core_count > KVAF_SEQ_MAX_WORKERS ||
      !s.config.batch_get_token_bytes || uint64_t(s.config.batch_get_token_bytes) * KVAF_SEQ_MAX_REQUESTS > UINT32_MAX ||
      s.config.aiv_batch_get_mode != "cpu_push_scatter" ||
      s.config.peer_local_eids.size() > 1)
    return Status::InvalidArgument("invalid sequential BatchGet configuration");
  s.region = region;
  if (s.config.peer_local_eids.size() == 1) s.config.peer_local_eid = s.config.peer_local_eids.front();
  peer::PeerRegion peer;
  auto result = s.config.peer_link->LookupRegion(region.base, &peer);
  if (!result.ok()) return result;
  if (!region.base || !region.length || region.base > UINT64_MAX - peer.length ||
      peer.length < region.length)
    return Status::InvalidArgument("invalid sequential remote region");
  if (KvafScatterBatchGetLoadKernel(s.config.batch_get_kernel_object_path.c_str()))
    return Error("load sequential kernel (rebuild Host and AIV binary together)");
  if (aclrtMalloc(&s.status_device, s.StatusBytes() + s.ProfileBytes(), ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS ||
      aclrtMallocHost(&s.status_host, s.StatusBytes()) != ACL_SUCCESS)
    return Error("allocate statuses");
  s.args.max_polls = 100000000U;
  s.args.source_base = region.base; s.args.source_bytes = region.length;
  result = s.InitializePush(peer);
  if (!result.ok()) return result;
  s.args.workspace = Ptr(s.workspace);
  s.initialized = true;
  return Status::OK();
}
Status ScatterBatchGet::Drain() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->Drain();
}
Status ScatterBatchGet::Execute(const BatchGetExecRequest& exec) {
  auto& s = *impl_;
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!s.initialized) return Status::NotInitialized("sequential BatchGet is not initialized");
  if (!exec.execution.stream) return Status::InvalidArgument("BatchGet needs a stream");
  aclmdlRICaptureStatus capture = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
  aclmdlRI graph = nullptr;
  if (aclmdlRICaptureGetInfo(static_cast<aclrtStream>(exec.execution.stream), &capture, &graph) != ACL_SUCCESS)
    return Error("query stream capture state");
  if (capture != ACL_MODEL_RI_CAPTURE_STATUS_NONE)
    return Status::NotImplemented("sequential BatchGet supports eager execution only");
  auto result = s.Drain();
  if (!result.ok()) return result;
  auto args = s.args;
  result = Input(exec, s.config, &args.request);
  if (!result.ok()) return result;
  if (s.sequence == UINT32_MAX) return Status::InvalidState("BatchGet sequence exhausted; rebuild Runtime");
  args.request.core_status = Ptr(s.status_device); args.request.queue_state = 0;
  args.sequence = ++s.sequence;
  s.profile_ready = false;
  s.pending = static_cast<aclrtStream>(exec.execution.stream);
  if (aclrtMemsetAsync(s.status_device, s.StatusBytes(), 0, s.StatusBytes(), s.pending) != ACL_SUCCESS ||
      KvafScatterBatchGetLaunch(&args, s.pending)) {
    s.failed = true; return Error("enqueue failed; resources retained");
  }
  s.profile_ready = args.reserved != 0;
  return exec.execution.synchronize ? s.Drain() : Status::OK();
}
Status ScatterBatchGet::SetScatterProfiling(bool enabled) {
  auto& s = *impl_;
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!s.initialized) return Status::NotInitialized("scatter is not initialized");
  auto result = s.Drain();
  if (!result.ok()) return result;
  s.args.reserved = enabled ? kScatterProfileFlag : 0U;
  s.profile_ready = false;
  return Status::OK();
}
Status ScatterBatchGet::CopyScatterProfiles(std::vector<ScatterBatchGetCoreProfile>* profiles) {
  auto& s = *impl_;
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!profiles) return Status::InvalidArgument("scatter profile output is null");
  if (!s.initialized || !s.args.reserved || !s.profile_ready)
    return Status::InvalidState("enable scatter profiling and complete a BatchGet first");
  auto result = s.Drain();
  if (!result.ok()) return result;
  profiles->resize(s.config.npu_core_count);
  const auto* source = static_cast<const char*>(s.status_device) + s.StatusBytes();
  const auto copied = aclrtMemcpy(profiles->data(), s.ProfileBytes(), source,
                                 s.ProfileBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
  if (copied != ACL_SUCCESS) { s.failed = true; return s.AclFailure("copy_profile", copied); }
  uint64_t requests = 0;
  for (uint32_t core = 0; core < profiles->size(); ++core) {
    const auto& p = (*profiles)[core];
    if (p.magic != kScatterProfileMagic || p.sequence != s.sequence ||
        !p.start_cycle || p.inputs_ready_cycle < p.start_cycle ||
        p.done_cycle < p.inputs_ready_cycle || p.finish_cycle < p.done_cycle ||
        p.bytes != p.requests * s.config.batch_get_token_bytes ||
        (!core && p.notify_cycle && (p.notify_cycle < p.inputs_ready_cycle || p.notify_cycle > p.done_cycle)))
      return Error("invalid scatter profile; rebuild Host and AIV kernel together");
    requests += p.requests;
  }
  if (requests && profiles->front().cpu_magic != cpu_push::ProfileToken(s.sequence))
    return Error("CPU push profile report missing; rebuild and restart the CPU service");
  return Status::OK();
}
Status ScatterBatchGet::Shutdown() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->Shutdown();
}
std::unique_ptr<SequentialBatchGet> CreateScatterBatchGet(AccessOperatorConfig config) {
  return std::make_unique<ScatterBatchGet>(std::move(config));
}
}  // namespace kvaf
