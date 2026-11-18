#include "scatter_batch_get.h"
#include "sequential_batch_get_common.h"
#include "kvaf_aiv_batch_get.h"
#include "cpu_push_get_layout.h"
#include "scatter_host_profile.h"
#include "scatter_mailbox_templates.h"
#include "direct_hccp/direct_hccp_session.h"
#include "direct_hccp/target_channel.h"
#include "peer_link.h"
#include <acl/acl.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <utility>

namespace kvaf {
namespace d = direct_hccp;
using namespace sequential_detail;
namespace {
Status Error(const char* what) { return Status::Internal(std::string("AIV sequential BatchGet: ") + what); }

void FillTrustedScatterInput(const BatchGetRequest& r, const AccessOperatorConfig& config,
                             KvafExecuteBatchGetKernelArgs* a) {
  const auto& table = r.device_access_table;
  *a = {};
  a->address_table = Ptr(table.addresses.data); a->valid_table = Ptr(table.valid.data);
  a->indices = Ptr(r.indices.data); a->offsets = Ptr(r.offsets.data);
  a->miss_count = Ptr(r.miss_count.data); a->output_indices = Ptr(r.output_indices.data);
  a->output = Ptr(r.output.data); a->table_capacity = table.addresses.bytes / 8;
  a->request_capacity = r.indices.bytes / 4;
  a->output_rows = r.output.bytes / config.batch_get_token_bytes;
  a->token_bytes = config.batch_get_token_bytes; a->tokens_per_block = r.tokens_per_block;
  a->block_count = config.npu_core_count; a->jettys_per_core = 2;
}

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
  Status ReplayScatterGraph(void* stream, bool synchronize) override;
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
  void* mailbox_import{};
  uint64_t mailbox_context{};
  std::array<std::unique_ptr<d::TargetChannel>, cpu_push::kMaxJettyCount> channels;
  uint32_t payload_jettys{cpu_push::kJettyCount}, payload_jettys_per_core{};
  aclrtStream pending{};
  aclmdlRI graph{};
  aclrtStream graph_stream{}, capture_pending{};
  aclrtStream external_graph_stream{};
  aclmdlRI external_graph{};
  uint32_t external_graph_calls{};
  uint64_t sequence{};
  bool push_open{}, initialized{}, failed{}, profile_ready{}, graph_mailbox{};
  uint64_t StatusBytes() const { return config.npu_core_count * sizeof(KvafSequentialStatus); }
  uint64_t ProfileBytes() const { return config.npu_core_count * sizeof(ScatterBatchGetCoreProfile); }

  Status CheckExecutionState() const {
    if (!initialized) return Status::NotInitialized("sequential BatchGet is not initialized");
    if (failed) return Status::InvalidState("sequential BatchGet failed; shutdown/reinitialize required");
    return Status::OK();
  }

  Status AclFailure(const char* phase, aclError code) const {
    return Status::DeviceError("AIV sequential BatchGet: mode=" + config.aiv_batch_get_mode +
        " sequence=" + std::to_string(sequence) + " phase=" + phase +
        " acl=" + std::to_string(code) +
        "; resources retained");
  }

  Status Drain(bool allow_initial_status = false) {
    if (failed) return Status::InvalidState("sequential BatchGet failed; shutdown/reinitialize required");
    if (!pending) return Status::OK();
    scatter_host::Timer copy_timer(scatter_host::kStatusCopy);
    const aclError copied = aclrtMemcpyAsync(status_host, StatusBytes(), status_device,
                                            StatusBytes(), ACL_MEMCPY_DEVICE_TO_HOST, pending);
    copy_timer.Stop();
    if (copied != ACL_SUCCESS) {
      failed = true; return AclFailure("copy_status", copied);
    }
    scatter_host::Timer sync_timer(scatter_host::kStreamSync);
    const aclError synchronized = aclrtSynchronizeStream(pending);
    sync_timer.Stop();
    if (synchronized != ACL_SUCCESS) {
      failed = true; return AclFailure("synchronize", synchronized);
    }
    pending = nullptr;
    scatter_host::Timer validation_timer(scatter_host::kStatusValidate);
    const auto* statuses = static_cast<const KvafSequentialStatus*>(status_host);
    for (uint32_t i = 0; i < config.npu_core_count; ++i) {
      const auto& s = statuses[i];
      if (allow_initial_status && sequence == 0 && s.sequence == 0) continue;
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

  Status QueryCapture(aclrtStream stream, bool* capturing, aclmdlRI* captured = nullptr) const {
    aclmdlRICaptureStatus capture = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    aclmdlRI active = nullptr;
    scatter_host::Timer timer(scatter_host::kCaptureQuery);
    const auto result = aclmdlRICaptureGetInfo(stream, &capture, &active);
    if (result != ACL_SUCCESS) return Error("query stream capture state");
    *capturing = capture != ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    if (captured) *captured = active;
    return Status::OK();
  }

  Status CaptureGraph(const KvafScatterBatchGetArgs& args, aclrtStream stream) {
    const auto begin = aclmdlRICaptureBegin(stream, ACL_MODEL_RI_CAPTURE_MODE_RELAXED);
    if (begin != ACL_SUCCESS) return AclFailure("graph_capture_begin", begin);
    capture_pending = stream;
    const auto launched = KvafScatterBatchGetLaunch(&args, stream);
    const auto end = aclmdlRICaptureEnd(stream, &graph);
    if (end != ACL_SUCCESS) {
      failed = true;
      return AclFailure("graph_capture_end", end);
    }
    capture_pending = nullptr;
    if (launched || !graph) {
      failed = true;
      return Error("capture Scatter launch");
    }
    graph_stream = stream;
    profile_ready = false;
    return Status::OK();
  }

  Status CheckExternalReplay(aclrtStream stream) {
    if (!external_graph_stream || stream != external_graph_stream)
      return Status::InvalidState("caller-managed Scatter graph requires its capture stream");
    bool capturing = false;
    auto status = QueryCapture(external_graph_stream, &capturing);
    if (!status.ok()) return status;
    if (capturing) return Status::InvalidState("check the Scatter result after capture and replay");
    if (external_graph_calls > UINT32_MAX - sequence)
      return Status::InvalidState("BatchGet sequence exhausted; rebuild Runtime");
    sequence += external_graph_calls;
    pending = external_graph_stream;
    status = Drain();
    profile_ready = status.ok() && ScatterProfileEnabled(args.reserved);
    return status;
  }

  Status DetachExternalGraph() {
    if (!external_graph_stream) return Status::InvalidState("no caller-managed Scatter graph attached");
    bool capturing = false;
    auto status = QueryCapture(external_graph_stream, &capturing);
    if (!status.ok()) return status;
    if (capturing) return Status::InvalidState("end and destroy the caller's graph before detaching Scatter");
    if (failed) {
      const auto result = aclrtSynchronizeStream(external_graph_stream);
      if (result != ACL_SUCCESS) return AclFailure("external_graph_synchronize", result);
      pending = nullptr;
    } else {
      // Verify the last checked sequence, including a capture that never ran.
      // An unreported replay must not leave the next eager launch out of sync.
      pending = external_graph_stream;
      status = Drain(/*allow_initial_status=*/true);
      if (!status.ok()) return status;
    }
    external_graph_stream = nullptr;
    external_graph = nullptr;
    external_graph_calls = 0;
    profile_ready = false;
    return Status::OK();
  }

  Status ReplayGraph(aclrtStream stream, bool synchronize) {
    if (!graph || stream != graph_stream) return Status::InvalidState("Scatter graph requires its capture stream");
    if (sequence == UINT32_MAX) return Status::InvalidState("BatchGet sequence exhausted; rebuild Runtime");
    pending = stream;
    scatter_host::Timer launch_timer(scatter_host::kLaunch);
    const auto replayed = aclmdlRIExecuteAsync(graph, stream);
    launch_timer.Stop();
    if (replayed != ACL_SUCCESS) {
      failed = true;
      return AclFailure("graph_replay", replayed);
    }
    // The AIV advances its own sequence. Host keeps the expected value for
    // status validation when this stream is drained, including empty replays.
    ++sequence;
    profile_ready = ScatterProfileEnabled(args.reserved);
    return synchronize ? Drain() : Status::OK();
  }

  Status DestroyGraph() {
    if (capture_pending) {
      aclmdlRICaptureStatus capture = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
      aclmdlRI active = nullptr;
      const auto queried = aclmdlRICaptureGetInfo(capture_pending, &capture, &active);
      if (queried != ACL_SUCCESS) return AclFailure("graph_capture_query", queried);
      if (capture != ACL_MODEL_RI_CAPTURE_STATUS_NONE) {
        aclmdlRI captured = nullptr;
        const auto end = aclmdlRICaptureEnd(capture_pending, &captured);
        if (end != ACL_SUCCESS) return AclFailure("graph_capture_end", end);
        graph = captured;
      }
      capture_pending = nullptr;
    }
    if (graph) {
      const auto destroyed = aclmdlRIDestroy(graph);
      if (destroyed != ACL_SUCCESS) return AclFailure("graph_destroy", destroyed);
      graph = nullptr;
    }
    graph_stream = nullptr;
    return Status::OK();
  }

  Status InitializePush(const peer::PeerRegion& peer, bool cpu_gather) {
    cpu_push::Layout layout;
    if (!cpu_push::MakeLayout(config.batch_get_token_bytes, config.npu_core_count, &layout, payload_jettys))
      return Status::InvalidArgument("invalid scatter workspace shape");
    args.descriptor_offset = layout.descriptor; args.staging_offset = layout.staging;
    // Queue context follows the shared layout and precedes the private
    // build-ready/failure lines.
    args.workspace_bytes = layout.bytes + sizeof(KvafScatterMailboxContext) +
                           uint64_t(config.npu_core_count + 1) * 64;
    // Prefer huge pages for the persistent CPU WRITE target and scatter staging;
    // ACL may fall back to normal pages when huge pages are unavailable.
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

  Status Shutdown() {
    if (external_graph_stream)
      return Status::InvalidState("destroy and detach the caller's Scatter graph before shutdown");
    // A normal close must let the CPU finish serving the pending request first.
    // On execution errors, keep closing the transports before releasing memory.
    const auto execution_status = failed ? Status::OK() : Drain();
    const bool execution_failed = failed || !execution_status.ok();
    failed = true;  // A partially closed transport cannot accept another Execute.
    if (pending) {
      const aclError synchronized = aclrtSynchronizeStream(pending);
      if (synchronized != ACL_SUCCESS) return AclFailure("shutdown_synchronize", synchronized);
    }
    pending = nullptr;
    auto graph_status = DestroyGraph();
    if (!graph_status.ok()) return graph_status;
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
  const char* ready_refresh_option = std::getenv("KVAF_SCATTER_READY_REFRESH");
  if (ready_refresh_option && std::strcmp(ready_refresh_option, "0") && std::strcmp(ready_refresh_option, "1"))
    return Status::InvalidArgument("KVAF_SCATTER_READY_REFRESH must be 0 or 1");
  const bool ready_refresh = ready_refresh_option && ready_refresh_option[0] == '1';
  if (ready_refresh && !jetty_cursors)
    return Status::InvalidArgument("KVAF_SCATTER_READY_REFRESH=1 requires KVAF_SCATTER_JETTY_CURSORS=1");
  const char* overlap_option = std::getenv("KVAF_SCATTER_OVERLAP_HEAD_CQ");
  if (overlap_option && std::strcmp(overlap_option, "0") && std::strcmp(overlap_option, "1"))
    return Status::InvalidArgument("KVAF_SCATTER_OVERLAP_HEAD_CQ must be 0 or 1");
  const bool overlap_head_cq = overlap_option && overlap_option[0] == '1';
  const char* graph_mailbox_option = std::getenv("KVAF_SCATTER_GRAPH_MAILBOX");
  if (graph_mailbox_option && std::strcmp(graph_mailbox_option, "0") && std::strcmp(graph_mailbox_option, "1"))
    return Status::InvalidArgument("KVAF_SCATTER_GRAPH_MAILBOX must be 0 or 1");
  s.graph_mailbox = graph_mailbox_option && graph_mailbox_option[0] == '1';
  const char* cpu_gather_option = std::getenv("KVAF_SCATTER_CPU_GATHER");
  if (cpu_gather_option && std::strcmp(cpu_gather_option, "0") && std::strcmp(cpu_gather_option, "1"))
    return Status::InvalidArgument("KVAF_SCATTER_CPU_GATHER must be 0 or 1");
  const bool cpu_gather = cpu_gather_option && cpu_gather_option[0] == '1';
  s.payload_jettys = cpu_push::kJettyCount;
  s.payload_jettys_per_core = 0;
  if (const auto* value = std::getenv("KVAF_SCATTER_JETTYS_PER_CORE")) {
    char* end = nullptr;
    const auto per_core = std::strtoul(value, &end, 10);
    if (value[0] < '1' || value[0] > '9' || *end || !per_core || per_core > cpu_push::kMaxJettyCount)
      return Status::InvalidArgument("KVAF_SCATTER_JETTYS_PER_CORE must be in 1..32");
    if (s.config.npu_core_count * per_core > cpu_push::kMaxJettyCount)
      return Status::InvalidArgument("Scatter AIV core count times Jettys per core must be at most 32");
    s.payload_jettys_per_core = per_core;
    s.payload_jettys = s.config.npu_core_count * per_core;
  }
  s.args.reserved = kScatterOptionsV1 | (pipeline ? kScatterPipeline : 0U) |
      (jetty_cursors ? kScatterJettyCursors : 0U) |
      (ready_refresh ? kScatterReadyRefresh : 0U) |
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
  std::fprintf(stderr, "[KVAF_AIV] SCATTER_OPTIONS pipeline=%u cpu_gather=%u payload_jettys=%u ready_refresh=%u graph_mailbox=%u\n",
               unsigned(pipeline), unsigned(cpu_gather), s.payload_jettys, unsigned(ready_refresh), unsigned(s.graph_mailbox));
  return Status::OK();
}
Status ScatterBatchGet::Drain() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->Drain();
}
Status ScatterBatchGet::ReplayScatterGraph(void* stream, bool synchronize) {
  scatter_host::Timer operator_timer(scatter_host::kOperator);
  auto& s = *impl_;
  scatter_host::Timer mutex_timer(scatter_host::kMutexWait);
  std::lock_guard<std::mutex> lock(s.mutex);
  mutex_timer.Stop();
  auto state = s.CheckExecutionState();
  if (!state.ok()) return state;
  if (!stream) return Status::InvalidArgument("BatchGet needs a stream");
  return s.ReplayGraph(static_cast<aclrtStream>(stream), synchronize);
}

Status ScatterBatchGet::Execute(const BatchGetExecRequest& exec) {
  const auto graph_action = exec.execution.scatter_graph;
  if (graph_action == ScatterGraphAction::kReplay)
    return ReplayScatterGraph(exec.execution.stream, exec.execution.synchronize);
  scatter_host::Timer operator_timer(scatter_host::kOperator);
  auto& s = *impl_;
  scatter_host::Timer mutex_timer(scatter_host::kMutexWait);
  std::lock_guard<std::mutex> lock(s.mutex);
  mutex_timer.Stop();
  if (graph_action == ScatterGraphAction::kDetach) return s.DetachExternalGraph();
  auto state = s.CheckExecutionState();
  if (!state.ok()) return state;
  if (graph_action == ScatterGraphAction::kCheck)
    return s.CheckExternalReplay(static_cast<aclrtStream>(exec.execution.stream));
  if (s.external_graph_stream && graph_action != ScatterGraphAction::kAttach)
    return Status::InvalidState("detach the caller's Scatter graph before another BatchGet");
  if (graph_action == ScatterGraphAction::kReset) {
    auto status = s.Drain();
    return status.ok() ? s.DestroyGraph() : status;
  }
  if (!exec.execution.stream) return Status::InvalidArgument("BatchGet needs a stream");
  const auto stream = static_cast<aclrtStream>(exec.execution.stream);
  if (s.graph) return Status::InvalidState("reset the prepared Scatter graph before another BatchGet");
  const bool capture_graph = graph_action == ScatterGraphAction::kCapture;
  const bool attach_graph = graph_action == ScatterGraphAction::kAttach;
  aclmdlRI external_capture = nullptr;
  if (capture_graph || attach_graph || !exec.execution.trusted_npu_request) {
    bool capturing = false;
    const auto status = s.QueryCapture(stream, &capturing, &external_capture);
    if (!status.ok()) return status;
    if (attach_graph) {
      if (!capturing || s.pending || exec.execution.synchronize)
        return Status::InvalidState("attach Scatter during active capture after draining previous work");
      if (!external_capture || (s.external_graph_stream &&
          (stream != s.external_graph_stream || external_capture != s.external_graph)))
        return Status::InvalidState("attach Scatter nodes to the same caller-managed graph and stream");
      if (s.external_graph_calls >= UINT32_MAX - s.sequence)
        return Status::InvalidState("BatchGet sequence exhausted; rebuild Runtime");
    } else if (capturing) {
      return Status::NotImplemented("use capture_into_graph for caller-managed capture");
    }
  }
  scatter_host::Timer prior_timer(scatter_host::kPriorDrain);
  auto result = s.Drain();
  prior_timer.Stop();
  if (!result.ok()) return result;
  auto args = s.args;
  if (exec.execution.trusted_npu_request) args.reserved |= kScatterTrustedInputs;
  scatter_host::Timer input_timer(scatter_host::kInput);
  if (exec.execution.trusted_npu_request) FillTrustedScatterInput(*exec.request, s.config, &args.request);
  else result = Input(exec, s.config, &args.request);
  input_timer.Stop();
  if (!result.ok()) return result;
  if (s.payload_jettys_per_core) args.request.jettys_per_core = s.payload_jettys_per_core;
  if (s.sequence == UINT32_MAX) return Status::InvalidState("BatchGet sequence exhausted; rebuild Runtime");
  args.request.core_status = Ptr(s.status_device);
  args.request.channel_entity = s.mailbox_context;
  args.request.queue_state = s.mailbox_context + offsetof(KvafScatterMailboxContext, queue);
  if (capture_graph || attach_graph) {
    args.sequence = 0;
    args.reserved |= kScatterDeviceSequence;
    if (s.graph_mailbox) args.reserved |= kScatterGraphMailbox;
    if (capture_graph) return s.CaptureGraph(args, stream);
    if (KvafScatterBatchGetLaunch(&args, stream)) {
      s.failed = true;
      return Error("attach Scatter node to caller's graph");
    }
    s.external_graph_stream = stream;
    s.external_graph = external_capture;
    ++s.external_graph_calls;
    s.profile_ready = false;
    return Status::OK();
  }
  args.sequence = ++s.sequence;
  s.profile_ready = false;
  s.pending = stream;
  // Every launched core overwrites its complete status. Drain checks this
  // request's monotonically increasing sequence, so an old status cannot pass.
  // Avoid submitting a separate device memset task before every kernel.
  scatter_host::Timer launch_timer(scatter_host::kLaunch);
  const auto launch_result = KvafScatterBatchGetLaunch(&args, s.pending);
  launch_timer.Stop();
  if (launch_result) {
    s.failed = true; return Error("enqueue failed; resources retained");
  }
  s.profile_ready = ScatterProfileEnabled(args.reserved);
  return exec.execution.synchronize ? s.Drain() : Status::OK();
}
Status ScatterBatchGet::SetScatterProfiling(bool enabled) {
  auto& s = *impl_;
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!s.initialized) return Status::NotInitialized("scatter is not initialized");
  auto result = s.Drain();
  if (!result.ok()) return result;
  if ((s.graph || s.external_graph_stream) && enabled != ScatterProfileEnabled(s.args.reserved))
    return Status::InvalidState("reset the prepared Scatter graph before changing profiling");
  s.args.reserved = ScatterWithProfiling(s.args.reserved, enabled);
  s.profile_ready = false;
  return Status::OK();
}
Status ScatterBatchGet::CopyScatterProfiles(std::vector<ScatterBatchGetCoreProfile>* profiles) {
  auto& s = *impl_;
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
Status ScatterBatchGet::Shutdown() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->Shutdown();
}
std::unique_ptr<SequentialBatchGet> CreateScatterBatchGet(AccessOperatorConfig config) {
  return std::make_unique<ScatterBatchGet>(std::move(config));
}
}  // namespace kvaf
