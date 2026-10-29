#include "sequential_batch_get.h"
#include "kvaf_aiv_batch_get.h"
#include "cpu_push_get_layout.h"
#include "cpu_gather.h"
#include "gather_workspace.h"
#include "direct_hccp/direct_hccp_session.h"
#include "direct_hccp/target_channel.h"
#include "peer_link.h"
#include <acl/acl.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <mutex>
#include <utility>

namespace kvaf {
namespace d = direct_hccp;
namespace {
Status Error(const char* what) { return Status::Internal(std::string("AIV sequential BatchGet: ") + what); }
uint64_t Ptr(const void* p) { return reinterpret_cast<uint64_t>(p); }
// The exported segment is the public 48-byte urma_seg_t wire ABI; no liburma is
// loaded on the NPU Host or AIV. ubva.va is packed at byte 20, len at byte 32.
constexpr uint32_t kSegmentBytes = 48;
static_assert(sizeof(KvafSequentialGatherHeader) == sizeof(peer::GatherDescriptorHeader));
static_assert(offsetof(KvafSequentialGatherHeader, ready_bytes) ==
              offsetof(peer::GatherDescriptorHeader, ready_bytes));
static_assert(sizeof(KvafSequentialGatherSpan) == sizeof(peer::GatherSpan));

Status Input(const BatchGetExecRequest& exec, const AccessOperatorConfig& config,
             KvafExecuteBatchGetKernelArgs* a) {
  if (!exec.request || !exec.execution.stream) return Status::InvalidArgument("BatchGet needs request and stream");
  const auto& r = *exec.request;
  const auto& table = r.device_access_table;
  const uint64_t count = r.indices.bytes / 4;
  const uint64_t entries = table.addresses.bytes / 8;
  if (!r.indices.has_elements<int32_t>() || !r.offsets.has_elements<int32_t>() ||
      !r.output_indices.has_elements<int32_t>() || !r.miss_count.has_elements<int32_t>() ||
      r.miss_count.bytes != 4 || r.indices.bytes != r.offsets.bytes ||
      r.indices.bytes != r.output_indices.bytes || !r.output.valid() ||
      !table.addresses.has_elements<uint64_t>() || !table.valid.valid() || table.valid.bytes != entries ||
      r.indices.device != DeviceKind::kNpu || r.offsets.device != DeviceKind::kNpu ||
      r.output_indices.device != DeviceKind::kNpu || r.miss_count.device != DeviceKind::kNpu ||
      r.output.device != DeviceKind::kNpu || table.addresses.device != DeviceKind::kNpu ||
      table.valid.device != DeviceKind::kNpu || count > KVAF_SEQ_MAX_REQUESTS || entries > UINT32_MAX ||
      !r.output_block_bytes || r.output_block_bytes != config.batch_get_token_bytes ||
      !r.tokens_per_block || r.tokens_per_block > UINT32_MAX ||
      r.output.bytes % r.output_block_bytes || r.output.bytes / r.output_block_bytes > UINT32_MAX ||
      Ptr(r.output.data) > UINT64_MAX - r.output.bytes)
    return Status::InvalidArgument("invalid sequential BatchGet Tensor layout/capacity");
  *a = {};
  a->address_table = Ptr(table.addresses.data); a->valid_table = Ptr(table.valid.data);
  a->indices = Ptr(r.indices.data); a->offsets = Ptr(r.offsets.data);
  a->miss_count = Ptr(r.miss_count.data); a->output_indices = Ptr(r.output_indices.data);
  a->output = Ptr(r.output.data); a->table_capacity = entries; a->request_capacity = count;
  a->output_rows = r.output.bytes / r.output_block_bytes;
  a->token_bytes = r.output_block_bytes; a->tokens_per_block = r.tokens_per_block;
  a->block_count = config.npu_core_count; a->jettys_per_core = 2;
  // Intentionally never copy the direct-read inflight budget into this ABI.
  return Status::OK();
}
}

struct SequentialBatchGet::Impl {
  explicit Impl(AccessOperatorConfig c) : config(std::move(c)) {}
  AccessOperatorConfig config;
  std::mutex mutex;
  RemoteRegionDesc region{};
  KvafSequentialBatchGetArgs args{};
  void* workspace{};
  void* status_device{};
  void* status_host{};
  void* queues{};
  KvafBatchGetTransport* gather_transport{};
  KvafBatchGetLocalBuffer* workspace_buffer{};
  KvafBatchGetLocalBuffer* output_buffer{};
  d::DirectHccpApi api{};
  d::DirectHccpSession* session{};
  void* context{};
  void* token{};
  void* memory_handle{};
  std::array<std::unique_ptr<d::TargetChannel>, 4> channels;
  aclrtStream pending{};
  uint64_t sequence{};
  bool push_open{}, gather_enabled{}, initialized{}, failed{};
  bool Push() const { return config.aiv_batch_get_mode == "cpu_push_scatter"; }
  uint64_t StatusBytes() const { return config.npu_core_count * sizeof(KvafSequentialStatus); }

  Status AclFailure(const char* phase, aclError code) const {
    return Status::DeviceError("AIV sequential BatchGet: mode=" + config.aiv_batch_get_mode +
        " sequence=" + std::to_string(sequence) + " phase=" + phase +
        " acl=" + std::to_string(code) + "; resources retained");
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
      if (s.magic != KVAF_SEQ_STATUS_MAGIC || s.sequence != sequence || s.result != KVAF_SEQ_OK ||
          s.phase != KVAF_SEQ_COMPLETE) {
        failed = true;
        return Status::Internal("AIV sequential BatchGet worker=" + std::to_string(i) +
            " sequence=" + std::to_string(sequence) + " observed_sequence=" +
            std::to_string(s.sequence) + " phase=" + std::to_string(s.phase) +
            " result=" + std::to_string(s.result));
      }
    }
    if (output_buffer) KvafBatchGetLocalBufferRelease(output_buffer);
    output_buffer = nullptr;
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
    if (!d::GetPeerTp(api, context, config.peer_local_eid, peer.eid, "aiv_scatter", &open.npu_tp))
      return Error("query scatter TP");
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

  Status InitializeGather(const peer::PeerRegion& peer) {
    const auto workers = config.npu_core_count;
    args.lane_descriptor_bytes = GatherLaneDescriptorBytes(workers);
    args.lane_staging_bytes = GatherLaneStagingBytes(config.batch_get_token_bytes, workers);
    args.descriptor_offset = workers * 64ULL;
    args.workspace_bytes = args.descriptor_offset + workers * args.lane_descriptor_bytes;
    if (region.length > UINT64_MAX - 63 || region.base > UINT64_MAX - region.length - 63)
      return Status::InvalidArgument("gather address overflow");
    args.remote_staging = cpu_push::Align64(region.base + region.length);
    args.remote_descriptor = args.remote_staging + workers * args.lane_staging_bytes;
    args.remote_control = args.remote_descriptor + workers * args.lane_descriptor_bytes;
    if (peer.length < region.length || peer.length - region.length <
        GatherRemoteWorkspaceBytes(config.batch_get_token_bytes, workers))
      return Status::InvalidArgument("remote DDR lacks reserved gather workspace");
    if (aclrtMalloc(&workspace, args.workspace_bytes, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS ||
        aclrtMemset(workspace, args.workspace_bytes, 0, args.workspace_bytes) != ACL_SUCCESS ||
        aclrtMalloc(&queues, workers * 2 * sizeof(KvafBatchGetQueueState), ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS ||
        aclrtMemset(queues, workers * 2 * sizeof(KvafBatchGetQueueState), 0,
                    workers * 2 * sizeof(KvafBatchGetQueueState)) != ACL_SUCCESS)
      return Error("allocate gather workspace");
    if (KvafBatchGetTransportCreateWithPeerLink(reinterpret_cast<void*>(region.base), peer.length,
        workers, 2, config.peer_link.get(), config.peer_local_eid.data(), &gather_transport))
      return Error("create gather control/data queues");
    if (KvafBatchGetLocalBufferAcquire(gather_transport, workspace, args.workspace_bytes,
        &workspace_buffer, &args.workspace_entity)) return Error("register gather workspace");
    gather_enabled = true;  // STOP also handles an uncertain enable response.
    return config.peer_link->EnableGatherLanes(region.base, args.remote_control, workers);
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
    if (gather_transport && KvafBatchGetTransportQuiesce(gather_transport))
      return Error("cannot quiesce gather DMA; resources retained");
    if (gather_enabled) {
      auto result = config.peer_link->StopGatherLanes(region.base);
      if (!result.ok()) return result;
      gather_enabled = false;
    }
    if (output_buffer) KvafBatchGetLocalBufferRelease(output_buffer);
    if (workspace_buffer) KvafBatchGetLocalBufferRelease(workspace_buffer);
    output_buffer = workspace_buffer = nullptr;
    if (gather_transport) KvafBatchGetTransportDestroy(gather_transport);
    gather_transport = nullptr;
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
    for (auto p : {&workspace, &queues, &status_device}) {
      if (*p && aclrtFree(*p) != ACL_SUCCESS) return Error("free device memory");
      *p = nullptr;
    }
    if (status_host && aclrtFreeHost(status_host) != ACL_SUCCESS) return Error("free status memory");
    status_host = nullptr;
    initialized = failed = false; sequence = 0; args = {};
    return execution_status;
  }
};

SequentialBatchGet::SequentialBatchGet(AccessOperatorConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}
SequentialBatchGet::~SequentialBatchGet() {
  if (!Shutdown().ok()) (void)impl_.release();
}
Status SequentialBatchGet::Initialize(const RemoteRegionDesc& region) {
  auto& s = *impl_;
  std::lock_guard<std::mutex> lock(s.mutex);
  if (s.initialized || s.failed || s.workspace || s.status_device || s.status_host)
    return Status::InvalidState("sequential transport already initialized; shutdown before retry");
  if (!s.config.peer_link || !s.config.npu_core_count || s.config.npu_core_count > KVAF_SEQ_MAX_WORKERS ||
      !s.config.batch_get_token_bytes || uint64_t(s.config.batch_get_token_bytes) * KVAF_SEQ_MAX_REQUESTS > UINT32_MAX ||
      !s.config.aiv_gather_sgl_size || s.config.aiv_gather_sgl_size > KVAF_SEQ_MAX_SGES ||
      (s.config.aiv_batch_get_mode != "cpu_push_scatter" && s.config.aiv_batch_get_mode != "gather_read") ||
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
  if (KvafSequentialBatchGetLoadKernel(s.config.batch_get_kernel_object_path.c_str()))
    return Error("load sequential kernel (rebuild Host and AIV binary together)");
  if (aclrtMalloc(&s.status_device, s.StatusBytes(), ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS ||
      aclrtMallocHost(&s.status_host, s.StatusBytes()) != ACL_SUCCESS)
    return Error("allocate statuses");
  s.args.mode = s.Push() ? KVAF_SEQ_SCATTER : KVAF_SEQ_GATHER;
  s.args.sgl_size = s.config.aiv_gather_sgl_size;
  s.args.max_polls = 100000000U;
  s.args.source_base = region.base; s.args.source_bytes = region.length;
  result = s.Push() ? s.InitializePush(peer) : s.InitializeGather(peer);
  if (!result.ok()) return result;
  s.args.workspace = Ptr(s.workspace);
  s.initialized = true;
  return Status::OK();
}
Status SequentialBatchGet::Drain() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->Drain();
}
Status SequentialBatchGet::Execute(const BatchGetExecRequest& exec) {
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
  if (!s.Push() && KvafBatchGetLocalBufferAcquire(s.gather_transport, exec.request->output.data,
      exec.request->output.bytes, &s.output_buffer, &args.request.channel_entity))
    return Error("register output");
  args.request.core_status = Ptr(s.status_device); args.request.queue_state = Ptr(s.queues);
  args.sequence = ++s.sequence;
  s.pending = static_cast<aclrtStream>(exec.execution.stream);
  if (aclrtMemsetAsync(s.status_device, s.StatusBytes(), 0, s.StatusBytes(), s.pending) != ACL_SUCCESS ||
      KvafSequentialBatchGetLaunch(&args, s.pending)) {
    s.failed = true; return Error("enqueue failed; resources retained");
  }
  return exec.execution.synchronize ? s.Drain() : Status::OK();
}
Status SequentialBatchGet::Shutdown() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->Shutdown();
}
}  // namespace kvaf
