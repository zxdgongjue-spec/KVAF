#include "scatter_execution.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace kvaf {
namespace d = direct_hccp;
using namespace sequential_detail;
namespace {
// The exported segment is the public 48-byte urma_seg_t wire ABI; no liburma is
// loaded on the NPU Host or AIV. ubva.va is packed at byte 20, len at byte 32.
constexpr uint32_t kSegmentBytes = 48;
}  // namespace

Status ScatterBatchGet::Resources::InitializePush(const peer::PeerRegion& peer, bool cpu_gather) {
  cpu_push::Layout layout;
  if (config.npu_core_count > cpu_push::kMaxAivWorkers ||
      !cpu_push::MakeLayout(config.batch_get_token_bytes, config.npu_core_count, &layout,
                            kDefaultGetWindow, payload_jettys))
    return Status::InvalidArgument("invalid scatter workspace shape");
  args.descriptor_offset = layout.descriptor; args.staging_offset = layout.staging;
  // Queue context follows the shared layout and precedes the private
  // build-ready/failure lines.
  args.workspace_bytes = layout.bytes + sizeof(KvafScatterMailboxContext) +
                         uint64_t(config.npu_core_count + 1) * 64;
  // Keep the original Scatter policy for both payload sources. This HBM
  // region is the persistent CPU WRITE target, including the SGL path.
  if (aclrtMalloc(&workspace, args.workspace_bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
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
  peer::OpenAivCpuPushJettyRequest request{};
  request.payload_jettys = payload_jettys;
  auto& open = request.base;
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
  for (uint32_t q = 0; q < payload_jettys; ++q) {
    channels[q] = std::make_unique<d::TargetChannel>(api, context);
    // Queue 0 is bidirectional: CPU payload target and AIV mailbox initiator.
    auto result = q == 0 ? channels[q]->InitializeAiv(token, peer.token_value, 1024, 16384)
                         : channels[q]->Initialize(token, peer.token_value, 1024, 1024, 16384);
    if (!result.ok()) return result;
    const auto& key = channels[q]->jetty_info().key;
    auto& channel = request.Channel(q);
    channel.bytes = key.size;
    std::memcpy(channel.key, key.value, key.size);
  }
  peer::OpenAivCpuPushJettyResponse reply{};
  const auto& response = reply.base;
  push_open = true;  // A lost response can still leave a live remote session.
  auto result = config.peer_link->OpenAivCpuPushGet(region.base, request, &reply, cpu_gather);
  if (!result.ok()) return result;
  if (response.mailbox_segment_bytes != kSegmentBytes || !response.mailbox_base ||
      response.mailbox_bytes < layout.staging || !(response.mailbox_token_id >> 8) ||
      response.max_write_bytes < 64 || response.max_write_bytes > 65536)
    return Error("invalid CPU mailbox setup (rebuild CPU service and NPU runtime together)");
  d::MrImportInfo imported{};
  imported.in.key.size = kSegmentBytes;
  std::memcpy(imported.in.key.value, response.mailbox_segment, kSegmentBytes);
  imported.in.ub.flags = ((1U << 1U) | (1U << 2U)) << 1U; // READ/WRITE, no mapping/cache.
  imported.in.ub.token_value = peer.token_value;
  if (api.rmem_import(context, &imported, &mailbox_import) || !mailbox_import)
    return Error("import CPU request mailbox");
  KvafScatterMailboxContext mailbox{};
  mailbox.max_write_bytes = response.max_write_bytes;
  auto& e = mailbox.entity;
  e.magic = 0x4b564155U; e.version = 1;
  e.remote_host_ddr_address = response.mailbox_base;
  e.remote_host_ddr_size = response.mailbox_bytes;
  e.remote_token_id = response.mailbox_token_id >> 8;
  e.remote_token_value = peer.token_value;
  e.local_hbm_address = Ptr(workspace); e.local_hbm_size = args.workspace_bytes;
  e.local_token_id = id.token_id; e.local_token_value = peer.token_value;
  for (uint32_t q = 0; q < payload_jettys; ++q) {
    d::QpImportInfoT info{};
    const auto& channel = reply.Channel(q);
    if (!d::ImportPeerJetty(api, context, channel.key, channel.bytes,
        peer.token_value, open.npu_tp, response.cpu_tp, open.npu_psn, response.cpu_psn,
        "aiv_scatter", &info, channels[q]->peer_jetty_output())) return Error("import CPU Jetty");
    if (!q) e.tp_id = info.out.ub.tpn;
  }
  const auto& sq = channels[0]->jetty_info().ub;
  const auto& cq = channels[0]->cq_info().out;
  e.sq_va = sq.sq_buffer_va; e.sq_db_va = sq.db_addr; e.jetty_id = sq.id;
  e.sq_depth = 1024 * 4; e.wqe_size = 64;
  e.cq_va = cq.buffer_addr; e.cq_db_va = cq.swdb_addr; e.jfc_id = cq.id;
  e.cq_depth = 16384; e.cqe_size = cq.cqe_size;
  // Same EID encoding as the validated device-initiated BatchPut path.
  for (uint32_t i = 0; i < sizeof(e.remote_eid); ++i)
    e.remote_eid[i] = peer.eid[sizeof(e.remote_eid) - 1 - i];
  if (!e.sq_va || !e.sq_db_va || !e.cq_va || !e.cq_db_va || !e.tp_id ||
      !e.local_token_id || e.cqe_size < 32) return Error("invalid mailbox queue ABI");
  mailbox_context = Ptr(workspace) + layout.bytes;
  PrepareScatterMailboxTemplates(&mailbox);
  if (aclrtMemcpy(reinterpret_cast<void*>(mailbox_context), sizeof(mailbox), &mailbox,
                  sizeof(mailbox), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS)
    return Error("initialize AIV mailbox context");
  return config.peer_link->StartAivCpuPushGet(region.base);
}

Status ScatterBatchGet::Resources::Close() {
  std::lock_guard<std::mutex> lock(mutex);
  if (graph.leases)
    return Status::InvalidState("release Scatter graph leases before shutdown");
  // A normal close must let the CPU finish serving the pending request first.
  // On execution errors, keep closing the transports before releasing memory.
  const auto execution_status = failed ? Status::OK() : Drain();
  const bool execution_failed = failed || !execution_status.ok();
  failed = true;  // A partially closed transport cannot accept another Execute.
  if (eager->pending) {
    const aclError synchronized = aclrtSynchronizeStream(eager->pending);
    if (synchronized != ACL_SUCCESS) return AclFailure("shutdown_synchronize", synchronized);
  }
  eager->pending = nullptr;
  // A timed-out control WRITE can still reference the CPU mailbox. Retire
  // its local SQ before StopCpuPushGet unregisters/frees that remote memory.
  if (execution_failed && channels[0]) {
    auto result = channels[0]->Shutdown();
    if (!result.ok()) return result;
    channels[0].reset();
  }
  if (push_open) {
    auto result = config.peer_link->StopAivCpuPushGet(region.base);
    if (!result.ok()) return result;
    push_open = false;
  }
  for (auto& channel : channels) if (channel) {
    auto result = channel->Shutdown();
    if (!result.ok()) return result;
    channel.reset();
  }
  if (mailbox_import) {
    if (api.rmem_unimport(context, mailbox_import)) return Error("unimport CPU mailbox");
    mailbox_import = nullptr;
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
  initialized = failed = profile_ready = false; sequence = mailbox_context = 0; args = {};
  return execution_status;
}

ScatterBatchGet::ScatterBatchGet(AccessOperatorConfig config)
    : standalone_(std::make_unique<Resources>(std::move(config))), resources_(standalone_.get()) {}
ScatterBatchGet::~ScatterBatchGet() {
  if (standalone_ && !standalone_->Close().ok()) (void)standalone_.release();
}
std::unique_ptr<ExecutionResourceOwner> ScatterBatchGet::TakeResources() {
  return std::move(standalone_);
}
Status ScatterBatchGet::Initialize(const RemoteRegionDesc& region) {
  return resources_->Initialize(region);
}
Status ScatterBatchGet::Resources::Initialize(const RemoteRegionDesc& region) {
  auto& s = *this;
  std::lock_guard<std::mutex> lock(s.mutex);
  if (s.initialized || s.failed || s.workspace || s.status_device || s.status_host)
    return Status::InvalidState("sequential transport already initialized; shutdown before retry");
  if (!s.config.peer_link || !s.config.npu_core_count || s.config.npu_core_count > KVAF_SEQ_MAX_WORKERS ||
      !s.config.batch_get_token_bytes || uint64_t(s.config.batch_get_token_bytes) * KVAF_SEQ_MAX_REQUESTS > UINT32_MAX ||
      s.config.peer_local_eids.size() > 1)
    return Status::InvalidArgument("invalid sequential BatchGet configuration");
  const char* pipeline_option = std::getenv("KVAF_SCATTER_PIPELINE");
  if (pipeline_option && std::strcmp(pipeline_option, "0") && std::strcmp(pipeline_option, "1"))
    return Status::InvalidArgument("KVAF_SCATTER_PIPELINE must be 0 or 1");
  const bool pipeline = !pipeline_option || pipeline_option[0] == '1';
  const char* cursor_option = std::getenv("KVAF_SCATTER_JETTY_CURSORS");
  if (cursor_option && std::strcmp(cursor_option, "0") && std::strcmp(cursor_option, "1"))
    return Status::InvalidArgument("KVAF_SCATTER_JETTY_CURSORS must be 0 or 1");
  const bool jetty_cursors = cursor_option && cursor_option[0] == '1';
  if (jetty_cursors && !pipeline)
    return Status::InvalidArgument("KVAF_SCATTER_JETTY_CURSORS=1 requires KVAF_SCATTER_PIPELINE=1");
  const char* overlap_option = std::getenv("KVAF_SCATTER_OVERLAP_HEAD_CQ");
  if (overlap_option && std::strcmp(overlap_option, "0") && std::strcmp(overlap_option, "1"))
    return Status::InvalidArgument("KVAF_SCATTER_OVERLAP_HEAD_CQ must be 0 or 1");
  const bool overlap_head_cq = overlap_option && overlap_option[0] == '1';
  const char* cpu_gather_option = std::getenv("KVAF_SCATTER_CPU_GATHER");
  if (cpu_gather_option && std::strcmp(cpu_gather_option, "0") && std::strcmp(cpu_gather_option, "1"))
    return Status::InvalidArgument("KVAF_SCATTER_CPU_GATHER must be 0 or 1");
  const bool cpu_gather = cpu_gather_option && cpu_gather_option[0] == '1';
  s.payload_jettys = cpu_push::kJettyCount;
  s.payload_jettys_per_core = s.config.aiv_scatter_jettys_per_core;
  const auto* value = std::getenv("KVAF_SCATTER_JETTYS_PER_CORE");
  if (!s.payload_jettys_per_core && value) {
    char* end = nullptr;
    const auto per_core = std::strtoul(value, &end, 10);
    if (value[0] < '1' || value[0] > '9' || *end || !per_core || per_core > cpu_push::kMaxJettyCount)
      return Status::InvalidArgument("KVAF_SCATTER_JETTYS_PER_CORE must be in 1..32");
    s.payload_jettys_per_core = per_core;
  }
  if (s.payload_jettys_per_core) {
    if (s.payload_jettys_per_core > cpu_push::kMaxJettyCount / s.config.npu_core_count)
      return Status::InvalidArgument("Scatter AIV core count times Jettys per core must be at most 32");
    s.payload_jettys = s.config.npu_core_count * s.payload_jettys_per_core;
  }
  s.args.reserved = kScatterOptionsV1 | (pipeline ? kScatterPipeline : 0U) |
      (jetty_cursors ? kScatterJettyCursors : 0U) |
      (overlap_head_cq ? kScatterOverlapHeadCq : 0U) |
      (s.payload_jettys != cpu_push::kJettyCount ? kScatterConfiguredJettys : 0U);
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
  // Clear recycled HBM once: first-launch stale bytes must not look like
  // sequence 1. Later requests are distinguished by their increasing sequence.
  if (aclrtMemset(s.status_device, s.StatusBytes() + s.ProfileBytes(), 0,
                 s.StatusBytes() + s.ProfileBytes()) != ACL_SUCCESS)
    return Error("initialize statuses");
  s.args.max_polls = 100000000U;
  s.args.source_base = region.base; s.args.source_bytes = region.length;
  result = s.InitializePush(peer, cpu_gather);
  if (!result.ok()) return result;
  s.args.workspace = Ptr(s.workspace);
  s.initialized = true;
  std::fprintf(stderr, "[KVAF_AIV] SCATTER_OPTIONS pipeline=%u cpu_gather=%u payload_jettys=%u\n",
               unsigned(pipeline), unsigned(cpu_gather), s.payload_jettys);
  return Status::OK();
}
Status ScatterBatchGet::DrainGet() {
  std::lock_guard<std::mutex> lock(resources_->mutex);
  return resources_->Drain();
}
std::unique_ptr<ExecutionState> ScatterBatchGet::CreateExecutionState() {
  return std::make_unique<Invocation>(resources_);
}
Status ScatterBatchGet::BindExecutionState(ExecutionState* state) {
  auto* invocation = dynamic_cast<Invocation*>(state);
  if (state && (!invocation || invocation->owner != resources_))
    return Status::InvalidArgument("Get invocation belongs to another resource owner");
  node_ = invocation && invocation->graph_node ? invocation : nullptr;
  if (!node_) resources_->eager = invocation ? invocation : &resources_->fallback;
  return Status::OK();
}
Status ScatterBatchGet::PrepareGetGraph(const GetGraphOptions& options) {
  auto& s = *resources_;
  std::lock_guard<std::mutex> lock(s.mutex);
  const auto status = s.CheckExecutionState();
  if (!status.ok()) return status;
  const auto stream = static_cast<aclrtStream>(options.stream);
  if (!node_ || node_->prepared || !stream || s.eager->pending || s.graph.id || s.graph.retiring ||
      (s.graph.stream && s.graph.stream != stream) || s.graph.leases == UINT32_MAX)
    return Status::InvalidState("prepare all Scatter graph leases on one idle stream before capture");
  auto capture = s.RequireOutsideCapture(stream);
  if (!capture.ok()) return capture;
  s.graph.stream = stream;
  ++s.graph.leases;
  node_->prepared = true;
  return Status::OK();
}
Status ScatterBatchGet::CheckBatchGetStatus(void* stream) {
  auto& s = *resources_;
  std::lock_guard<std::mutex> lock(s.mutex);
  const auto status = s.CheckExecutionState();
  if (!status.ok()) return status;
  if (!node_ || !node_->captured || s.graph.stream != stream || s.graph.retiring)
    return Status::InvalidState("check Scatter replay using a captured graph lease and its stream");
  auto capture = s.RequireOutsideCapture(s.graph.stream);
  return capture.ok() ? s.ReadGraphStatus(false) : capture;
}
Status ScatterBatchGet::FinishGetGraph(void* stream, bool discard) {
  auto& s = *resources_;
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!node_ || !node_->prepared || s.graph.stream != stream)
    return Status::InvalidState("finish Scatter using its prepared graph lease and stream");
  if (s.failed && !discard)
    return Status::InvalidState("destroy and discard the failed Scatter graph");
  if (!s.graph.retiring) {
    auto status = s.RequireOutsideCapture(s.graph.stream);
    if (!status.ok()) return status;
    if (s.graph.nodes && !s.failed) {
      status = s.ReadGraphStatus(true);
      if (!status.ok() && !discard) return status;
    }
    if (s.failed) {
      const auto result = aclrtSynchronizeStream(s.graph.stream);
      if (result != ACL_SUCCESS) return s.AclFailure("external_graph_synchronize", result);
    }
    s.graph.retiring = true;
  }
  node_->prepared = false;
  node_->recovery_fenced = s.failed;
  if (--s.graph.leases == 0) {
    s.graph = {};
    s.profile_ready = false;
  }
  return Status::OK();
}
bool ScatterBatchGet::GetGraphNeedsRecovery() const {
  return node_ && node_->recovery_fenced;
}
Status ScatterBatchGet::ExecuteBatchGet(const BatchGetExecRequest& exec) {
  return Launch(exec, {*resources_, *resources_->eager, node_});
}
Status ScatterBatchGet::SetScatterProfiling(bool enabled) {
  auto& s = *resources_;
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!s.initialized) return Status::NotInitialized("scatter is not initialized");
  auto result = s.Drain();
  if (!result.ok()) return result;
  if (s.graph.stream && enabled != ScatterProfileEnabled(s.args.reserved))
    return Status::InvalidState("detach the Scatter graph before changing profiling");
  s.args.reserved = ScatterWithProfiling(s.args.reserved, enabled);
  s.profile_ready = false;
  return Status::OK();
}
Status ScatterBatchGet::CopyScatterProfiles(std::vector<ScatterBatchGetCoreProfile>* profiles) const {
  auto& s = *resources_;
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!profiles) return Status::InvalidArgument("scatter profile output is null");
  if (!s.initialized || !ScatterProfileEnabled(s.args.reserved) || !s.profile_ready)
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
        p.pipeline != ScatterOptionEnabled(s.args.reserved, kScatterPipeline) ||
        (p.requests && (!p.scatter_cycles || !p.scatter_chunks ||
            p.scatter_start_cycle < p.inputs_ready_cycle ||
            p.scatter_finish_cycle > p.finish_cycle || p.scatter_finish_cycle <= p.scatter_start_cycle ||
            p.scatter_cycles > p.scatter_finish_cycle - p.scatter_start_cycle)) ||
        p.bytes != p.requests * s.config.batch_get_token_bytes ||
        (!core && p.notify_cycle && (p.notify_cycle < p.inputs_ready_cycle || p.notify_cycle > p.done_cycle)))
      return Error("invalid scatter profile; rebuild Host and AIV kernel together");
    requests += p.requests;
  }
  if (requests && profiles->front().cpu_magic != cpu_push::ProfileToken(s.sequence))
    return Error("CPU push profile report missing; rebuild and restart the CPU service");
  return Status::OK();
}
Status ScatterBatchGet::Shutdown() { return resources_->Close(); }
std::unique_ptr<SequentialBatchGet> CreateScatterBatchGet(AccessOperatorConfig config) {
  return std::make_unique<ScatterBatchGet>(std::move(config));
}
}  // namespace kvaf
