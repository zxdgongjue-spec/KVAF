#include "gather_execution.h"

#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace kvaf {
using namespace sequential_detail;
namespace {
Status ReadProbe(uint32_t* probe) {
  *probe = KVAF_GATHER_PROBE_NONE;
  const char* value = std::getenv("KVAF_AIV_GATHER_PROBE");
  if (!value || !*value) return Status::OK();
  for (uint32_t p = KVAF_GATHER_PROBE_ACK; p <= KVAF_GATHER_PROBE_FIRST_READ; ++p)
    if (std::strcmp(value, ProbeName(p)) == 0) { *probe = p; return Status::OK(); }
  return Status::InvalidArgument("KVAF_AIV_GATHER_PROBE must be ack/wqe/sq/first_read or empty");
}
}  // namespace

Status GatherBatchGet::Resources::InitializeGather() {
  const auto workers = config.npu_core_count;
  args.lane_descriptor_bytes = GatherLaneDescriptorBytes(workers);
  args.lane_staging_bytes = GatherLaneStagingBytes(config.batch_get_token_bytes, workers);
  args.descriptor_offset = workers * uint64_t(kGatherScratchBytes);
  args.workspace_bytes = args.descriptor_offset + workers * args.lane_descriptor_bytes;
  peer::PeerRegion remote_workspace;
  const auto remote_bytes = (GatherRemoteWorkspaceBytes(config.batch_get_token_bytes, workers) + 63U) &
                            ~uint64_t{63};
  auto status = config.peer_link->PrepareGatherWorkspace(region.base, remote_bytes, &remote_workspace);
  if (!status.ok()) return status;
  args.remote_staging = remote_workspace.base;
  args.remote_descriptor = args.remote_staging + workers * args.lane_staging_bytes;
  args.remote_control = args.remote_descriptor + workers * args.lane_descriptor_bytes;
  // Prefer huge pages; allow normal-page fallback, including small queue states.
  if (aclrtMalloc(&workspace, args.workspace_bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
      aclrtMemset(workspace, args.workspace_bytes, 0, args.workspace_bytes) != ACL_SUCCESS ||
      aclrtMalloc(&queues, workers * 2 * sizeof(KvafBatchGetQueueState), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
      aclrtMemset(queues, workers * 2 * sizeof(KvafBatchGetQueueState), 0,
                  workers * 2 * sizeof(KvafBatchGetQueueState)) != ACL_SUCCESS)
    return Error("allocate gather workspace");
  if (KvafBatchGetTransportCreateWithPeerLink(reinterpret_cast<void*>(remote_workspace.base), remote_workspace.length,
      workers, 2, config.peer_link.get(), config.peer_local_eid.data(), &gather_transport))
    return Error("create gather control/data queues");
  if (KvafBatchGetLocalBufferAcquire(gather_transport, workspace, args.workspace_bytes,
      &workspace_buffer, &args.workspace_entity)) return Error("register gather workspace");
  gather_enabled = true;  // STOP also handles an uncertain enable response.
  uint32_t capabilities = 0;
  status = config.peer_link->EnableGatherLanesWithCapabilities(
      region.base, args.remote_control, workers, capabilities, true);
  if (status.ok() && args.grouped_addresses && !(capabilities & peer::kGatherLanesChunkedPullGroups))
    return Error("CPU service does not support grouped AIV Gather");
  return status;
}

Status GatherBatchGet::Resources::Close() {
  std::lock_guard<std::mutex> lock(mutex);
  if (live_graph) return Status::Busy("release the Gather graph lease before shutdown");
  // Finish pending CPU work; on errors quiesce transports before freeing memory.
  const auto execution_status = failed ? Status::OK() : Drain();
  failed = true;  // A partially closed transport cannot accept another Execute.
  if (eager->pending) {
    const aclError synchronized = aclrtSynchronizeStream(eager->pending);
    if (synchronized != ACL_SUCCESS) return AclFailure("shutdown_synchronize", synchronized);
  }
  eager->pending = nullptr;
  // Stop CPU copy workers before releasing descriptor/staging resources.
  if (gather_enabled) {
    auto result = config.peer_link->StopGatherLanes(region.base);
    if (!result.ok()) return result;
    gather_enabled = false;
  }
  if (gather_transport && KvafGatherBatchGetTransportQuiesce(gather_transport))
    return Error("cannot quiesce gather DMA; resources retained");
  if (eager->output_buffer) KvafBatchGetLocalBufferRelease(eager->output_buffer);
  if (retained_graph_output) KvafBatchGetLocalBufferRelease(retained_graph_output);
  retained_graph_output = nullptr;
  if (workspace_buffer) KvafBatchGetLocalBufferRelease(workspace_buffer);
  eager->output_buffer = workspace_buffer = nullptr;
  if (gather_transport) KvafBatchGetTransportDestroy(gather_transport);
  gather_transport = nullptr;
  for (auto p : {&workspace, &queues, &status_device, &profile_device}) {
    if (*p && aclrtFree(*p) != ACL_SUCCESS) return Error("free device memory");
    *p = nullptr;
  }
  if (status_host && aclrtFreeHost(status_host) != ACL_SUCCESS) return Error("free status memory");
  status_host = nullptr;
  if (profile_host && aclrtFreeHost(profile_host) != ACL_SUCCESS)
    return Error("free profile memory");
  profile_host = nullptr;
  profiling = eager->pending_profile = eager->profile_valid = false;
  check_async_status = eager->pending_status_check = false;
  eager->latest_profile = {};
  initialized = failed = false; sequence = 0; args = {};
  return execution_status;
}

Status GatherBatchGet::Resources::CaptureProfile() {
  const auto* profiles = static_cast<const KvafGatherBatchGetCoreProfile*>(profile_host);
  const auto* statuses = static_cast<const KvafSequentialStatus*>(status_host);
  eager->latest_profile.cores.clear();
  eager->latest_profile.cores.reserve(config.npu_core_count);
  for (uint32_t i = 0; i < config.npu_core_count; ++i) {
    const auto& p = profiles[i];
    const auto payload_bytes = p.requests * config.batch_get_token_bytes;
    const auto chunk_bytes = !p.requests ? 0 : args.grouped_addresses ?
        std::min<uint64_t>(p.requests, KVAF_GATHER_GROUP_WRS * args.sgl_size) * config.batch_get_token_bytes :
        args.cooperative_chunks ? GatherCooperativeChunkBytes(
            payload_bytes, args.cooperative_chunks, args.first_chunk_divisor) : args.pipeline_chunk_bytes;
    const char* reason = nullptr;
    if (p.magic != KVAF_GATHER_PROFILE_MAGIC) reason = "magic";
    else if (p.sequence != sequence) reason = "sequence";
    else if (p.result != KVAF_SEQ_OK) reason = "result";
    else if (p.valid_mask != KVAF_GATHER_PROFILE_ALL_EVENTS) reason = "valid_mask";
    else if (p.kernel_start_cycle > p.input_done_cycle) reason = "timestamp_start_input";
    else if (p.input_done_cycle > p.descriptor_done_cycle) reason = "timestamp_input_descriptor";
    else if (!args.grouped_addresses && !(eager->head_start && !args.probe && p.requests > kGatherHeadRows) &&
             p.descriptor_done_cycle > p.notify_done_cycle) reason = "timestamp_descriptor_notify";
    else if (p.notify_done_cycle > p.ack_done_cycle) reason = "timestamp_notify_ack";
    else if (!p.pipeline_chunk_bytes && p.ack_done_cycle > p.read_done_cycle) reason = "timestamp_ack_read";
    else if (p.ack_done_cycle > p.kernel_finish_cycle) reason = "timestamp_ack_finish";
    else if (p.notify_done_cycle > p.read_done_cycle) reason = "timestamp_notify_read";
    else if (p.read_done_cycle > p.kernel_finish_cycle) reason = "timestamp_read_finish";
    else if (p.requests != statuses[i].requests) reason = "requests";
    else if (p.data_wrs != statuses[i].data_wrs) reason = "data_wrs";
    else if (p.pipeline_chunk_bytes != chunk_bytes) reason = "pipeline_chunk_bytes";
    else if (p.prepare_ahead != (p.requests ? args.prepare_ahead : 0)) reason = "prepare_ahead";
    else if (p.cooperative_chunks != (p.requests ? args.cooperative_chunks : 0)) reason = "cooperative_chunks";
    else if (p.pipeline_chunk_bytes &&
        (p.first_ready_cycle < p.notify_done_cycle || p.first_ready_cycle > p.first_read_submit_cycle ||
         p.first_read_submit_cycle > p.read_done_cycle || !p.first_read_ready_bytes ||
         p.first_read_ready_bytes > p.requests * config.batch_get_token_bytes))
      reason = "timestamp_ready_submit_read";
    if (reason) {
      // Preserve the failed record before Shutdown; never return a partial snapshot.
      std::ostringstream detail;
      detail << "AIV sequential BatchGet: invalid gather profile worker=" << i
             << " sequence=" << sequence << " reason=" << reason
             << " magic=0x" << std::hex << p.magic
             << " expected_magic=0x" << KVAF_GATHER_PROFILE_MAGIC << std::dec
             << " observed_sequence=" << p.sequence << " result=" << p.result
             << " expected_result=" << KVAF_SEQ_OK
             << " valid_mask=0x" << std::hex << p.valid_mask
             << " expected_mask=0x" << KVAF_GATHER_PROFILE_ALL_EVENTS << std::dec
             << " cycles=[start:" << p.kernel_start_cycle << ",input:" << p.input_done_cycle
             << ",descriptor:" << p.descriptor_done_cycle << ",notify:" << p.notify_done_cycle
             << ",ack:" << p.ack_done_cycle << ",read:" << p.read_done_cycle
             << ",finish:" << p.kernel_finish_cycle << "]"
             << " requests=" << p.requests << " status_requests=" << statuses[i].requests
             << " data_wrs=" << p.data_wrs << " status_data_wrs=" << statuses[i].data_wrs
             << " ack_queries=" << p.ack_queries << " data_wqebbs=" << p.data_wqebbs
             << " data_doorbells=" << p.data_doorbells
             << " data_cq_wait_cycles=" << p.data_cq_wait_cycles
             << " read_build_cycles=" << p.read_build_cycles
             << " pipeline_chunk_bytes=" << p.pipeline_chunk_bytes
             << " first_ready_cycle=" << p.first_ready_cycle
             << " first_read_submit_cycle=" << p.first_read_submit_cycle
             << " first_read_ready_bytes=" << p.first_read_ready_bytes
             << " ready_queries=" << p.ready_queries << " ready_wait_cycles=" << p.ready_wait_cycles
             << " prepare_ahead=" << p.prepare_ahead << " cooperative_chunks=" << p.cooperative_chunks
             << " prepare_ahead_wrs=" << p.prepare_ahead_wrs
             << " prepare_ahead_cycles=" << p.prepare_ahead_cycles
             << " args_bytes=" << sizeof(KvafGatherBatchGetArgs)
             << " record_bytes=" << sizeof(p);
      return Status::Internal(detail.str());
    }
    GatherBatchGetCoreProfile out;
    out.sequence = p.sequence;
    out.kernel_start_cycle = p.kernel_start_cycle;
    out.input_done_cycle = p.input_done_cycle;
    out.descriptor_done_cycle = p.descriptor_done_cycle;
    out.notify_done_cycle = p.notify_done_cycle;
    out.ack_done_cycle = p.ack_done_cycle;
    out.read_done_cycle = p.read_done_cycle;
    out.kernel_finish_cycle = p.kernel_finish_cycle;
    out.requests = p.requests;
    out.data_wrs = p.data_wrs;
    out.ack_queries = p.ack_queries;
    out.data_wqebbs = p.data_wqebbs;
    out.data_doorbells = p.data_doorbells;
    out.data_cq_wait_cycles = p.data_cq_wait_cycles;
    out.read_build_cycles = p.read_build_cycles;
    out.first_ready_cycle = p.first_ready_cycle;
    out.first_read_submit_cycle = p.first_read_submit_cycle;
    out.first_read_ready_bytes = p.first_read_ready_bytes;
    out.ready_queries = p.ready_queries;
    out.ready_wait_cycles = p.ready_wait_cycles;
    out.pipeline_chunk_bytes = p.pipeline_chunk_bytes;
    out.read_vf_calls = p.read_vf_calls;
    out.prepare_ahead = p.prepare_ahead;
    out.cooperative_chunks = p.cooperative_chunks;
    out.prepare_ahead_wrs = p.prepare_ahead_wrs;
    out.prepare_ahead_cycles = p.prepare_ahead_cycles;
    eager->latest_profile.cores.push_back(out);
  }
  return Status::OK();
}

GatherBatchGet::GatherBatchGet(AccessOperatorConfig config)
    : standalone_(std::make_unique<Resources>(std::move(config))), resources_(standalone_.get()) {}
GatherBatchGet::~GatherBatchGet() {
  if (standalone_ && !standalone_->Close().ok()) (void)standalone_.release();
}
std::unique_ptr<ExecutionResourceOwner> GatherBatchGet::TakeResources() {
  return std::move(standalone_);
}
Status GatherBatchGet::Initialize(const RemoteRegionDesc& region) {
  return resources_->Initialize(region);
}
Status GatherBatchGet::Resources::Initialize(const RemoteRegionDesc& region) {
  auto& s = *this;
  std::lock_guard<std::mutex> lock(s.mutex);
  if (s.initialized || s.failed || s.workspace || s.status_device || s.status_host)
    return Status::InvalidState("sequential transport already initialized; shutdown before retry");
  if (!s.config.peer_link || !s.config.npu_core_count || s.config.npu_core_count > KVAF_SEQ_MAX_WORKERS ||
      !s.config.batch_get_token_bytes || uint64_t(s.config.batch_get_token_bytes) * KVAF_SEQ_MAX_REQUESTS > UINT32_MAX ||
      !s.config.aiv_gather_sgl_size || s.config.aiv_gather_sgl_size > KVAF_SEQ_MAX_SGES ||
      s.config.peer_local_eids.size() > 1)
    return Status::InvalidArgument("invalid sequential BatchGet configuration");
  for (const char* name : {"KVAF_AIV_GATHER_SHARD_PIPELINE", "KVAF_AIV_GATHER_PACKED_POSTS",
                           "KVAF_AIV_GATHER_PROGRESS_MODE"}) {
    const char* value = std::getenv(name);
    if (value && *value)
      return Status::InvalidArgument(std::string(name) +
          " has been removed; unset it and use PREPARE_AHEAD / COOPERATIVE_CHUNKS");
  }
  auto result = ReadProbe(&s.args.probe);
  if (!result.ok()) return result;
  const bool fast_read = uint64_t(s.config.batch_get_token_bytes) * s.config.aiv_gather_sgl_size <= 65536;
  const char* grouped = std::getenv("KVAF_AIV_GATHER_GROUPED");
  if (grouped && *grouped && std::strcmp(grouped, "0") && std::strcmp(grouped, "1"))
    return Status::InvalidArgument("KVAF_AIV_GATHER_GROUPED must be 0 or 1");
  s.args.grouped_addresses = grouped && std::strcmp(grouped, "1") == 0;
  if (s.args.grouped_addresses && (s.args.probe || !fast_read))
    return Status::InvalidArgument("grouped gather requires no probe and token_bytes * SGL <= 65536");
  if (s.args.grouped_addresses) {
    s.args.prepare_ahead = 1;
  } else {
    s.args.pipeline_chunk_bytes = kGatherChunkBytes;
    const char* chunk = std::getenv("KVAF_AIV_GATHER_CHUNK_BYTES");
    if (chunk && *chunk) {
      const auto parsed = std::from_chars(chunk, chunk + std::strlen(chunk), s.args.pipeline_chunk_bytes);
      if (parsed.ec != std::errc{} || *parsed.ptr != '\0' ||
          (s.args.pipeline_chunk_bytes &&
           (s.args.pipeline_chunk_bytes < 64 || s.args.pipeline_chunk_bytes % 64)))
        return Status::InvalidArgument("KVAF_AIV_GATHER_CHUNK_BYTES must be 0 or a uint32 multiple of 64");
    }
    // Probes and split-token reads retain whole-batch ACK semantics.
    if (s.args.probe || !fast_read)
      s.args.pipeline_chunk_bytes = 0;
    const char* cooperative = std::getenv("KVAF_AIV_GATHER_COOPERATIVE_CHUNKS");
    if (cooperative && *cooperative) {
      const auto parsed = std::from_chars(cooperative, cooperative + std::strlen(cooperative), s.args.cooperative_chunks);
      if (parsed.ec != std::errc{} || *parsed.ptr != '\0')
        return Status::InvalidArgument("KVAF_AIV_GATHER_COOPERATIVE_CHUNKS must be an unsigned integer");
    }
    const char* first = std::getenv("KVAF_AIV_GATHER_FIRST_CHUNK_DIVISOR");
    if (first && *first) {
      const auto parsed = std::from_chars(first, first + std::strlen(first), s.args.first_chunk_divisor);
      if (parsed.ec != std::errc{} || *parsed.ptr != '\0' || s.args.first_chunk_divisor == 1)
        return Status::InvalidArgument("KVAF_AIV_GATHER_FIRST_CHUNK_DIVISOR must be 0 or a uint32 >=2");
    }
    if (!GatherCooperativeLayoutValid(s.args.cooperative_chunks, s.args.first_chunk_divisor))
      return Status::InvalidArgument("COOPERATIVE_CHUNKS must be 0/2/4/8/16/32 in equal mode, "
                                    "or 1..32 with FIRST_CHUNK_DIVISOR>=2");
    if (s.args.cooperative_chunks && (!s.args.pipeline_chunk_bytes || s.args.probe || !fast_read))
      return Status::InvalidArgument("cooperative gather requires nonzero chunk mode, no probe, token_bytes * SGL <= 65536");
    const char* prepare = std::getenv("KVAF_AIV_GATHER_PREPARE_AHEAD");
    if (prepare && *prepare && std::strcmp(prepare, "0") && std::strcmp(prepare, "1"))
      return Status::InvalidArgument("KVAF_AIV_GATHER_PREPARE_AHEAD must be 0 or 1");
    s.args.prepare_ahead = prepare && *prepare ? std::strcmp(prepare, "1") == 0 : bool(s.args.cooperative_chunks);
    if (s.args.cooperative_chunks && !s.args.prepare_ahead)
      return Status::InvalidArgument("cooperative gather requires KVAF_AIV_GATHER_PREPARE_AHEAD=1");
    if (s.args.prepare_ahead && (s.args.probe || !fast_read))
      return Status::InvalidArgument("gather prepare-ahead requires whole-batch/prefix, no probe, token_bytes * SGL <= 65536");
  }
  const char* coalesce = std::getenv("KVAF_AIV_GATHER_COALESCE_SGES");
  if (coalesce && *coalesce && std::strcmp(coalesce, "0") && std::strcmp(coalesce, "1"))
    return Status::InvalidArgument("KVAF_AIV_GATHER_COALESCE_SGES must be 0 or 1");
  s.args.coalesce_sges = coalesce && std::strcmp(coalesce, "1") == 0;
  if (s.args.coalesce_sges && !fast_read)
    return Status::InvalidArgument("coalesced gather requires token_bytes * SGL <= 65536");
  if (s.args.probe > KVAF_GATHER_PROBE_ACK && !fast_read)
    return Status::InvalidArgument("gather WQE/SQ probes require token_bytes * SGL <= 65536");
  s.region = region;
  if (s.config.peer_local_eids.size() == 1) s.config.peer_local_eid = s.config.peer_local_eids.front();
  peer::PeerRegion peer;
  result = s.config.peer_link->LookupRegion(region.base, &peer);
  if (!result.ok()) return result;
  if (!region.base || !region.length || region.base > UINT64_MAX - peer.length ||
      peer.length < region.length)
    return Status::InvalidArgument("invalid sequential remote region");
  if (KvafGatherBatchGetLoadKernel(s.config.batch_get_kernel_object_path.c_str()))
    return Error("load sequential kernel (rebuild Host and AIV binary together)");
  if (aclrtMalloc(&s.status_device, s.StatusBytes(), ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS ||
      aclrtMallocHost(&s.status_host, s.StatusBytes()) != ACL_SUCCESS)
    return Error("allocate statuses");
  s.args.sgl_size = s.config.aiv_gather_sgl_size;
  const char* check_status = std::getenv("KVAF_AIV_GATHER_CHECK_STATUS");
  s.check_async_status = check_status && std::strcmp(check_status, "1") == 0;
  s.args.max_polls = 100000000U;
  s.args.source_base = region.base; s.args.source_bytes = region.length;
  result = s.InitializeGather();
  if (!result.ok()) return result;
  s.args.workspace = Ptr(s.workspace);
  s.initialized = true;
  std::fprintf(stderr, "[KVAF_AIV] GATHER_OPTIONS pipeline=%s scheduler=%s chunk_bytes=%u prepare_ahead=%u cooperative_chunks=%u first_chunk_divisor=%u group_wrs=%u\n",
      s.args.grouped_addresses ? "grouped" : s.args.first_chunk_divisor ? "bootstrap_prefix" : s.args.cooperative_chunks ? "cooperative_prefix" : s.args.pipeline_chunk_bytes ? "prefix" : "whole_batch",
      s.args.prepare_ahead ? "prepared" : "baseline",
      s.args.pipeline_chunk_bytes, s.args.prepare_ahead, s.args.cooperative_chunks, s.args.first_chunk_divisor,
      s.args.grouped_addresses ? KVAF_GATHER_GROUP_WRS : 0U);
  return Status::OK();
}
Status GatherBatchGet::DrainGet() {
  std::lock_guard<std::mutex> lock(resources_->mutex);
  return resources_->Drain();
}
std::unique_ptr<ExecutionState> GatherBatchGet::CreateExecutionState() {
  return std::make_unique<Invocation>(resources_);
}
Status GatherBatchGet::BindExecutionState(ExecutionState* state) {
  auto* invocation = dynamic_cast<Invocation*>(state);
  if (state && (!invocation || invocation->owner != resources_))
    return Status::InvalidArgument("Get invocation belongs to another resource owner");
  node_ = invocation && invocation->graph_node ? invocation : nullptr;
  if (!node_) resources_->eager = invocation ? invocation : &resources_->fallback;
  return Status::OK();
}
Status GatherBatchGet::PrepareGetGraph(const GetGraphOptions& options) {
  auto& s = *resources_;
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!s.initialized || s.failed || !node_ || s.live_graph || s.eager->pending ||
      s.profiling || s.args.probe || s.sequence == UINT32_MAX)
    return Status::InvalidState("Gather requires one graph lease on an idle provider without profiling or probes");
  if (!options.stream || !options.output.valid() || options.output.device != DeviceKind::kNpu ||
      !options.request_capacity || options.request_capacity > KVAF_SEQ_MAX_REQUESTS)
    return Status::InvalidArgument("invalid Gather graph stream, output or capacity");
  const auto stream = static_cast<aclrtStream>(options.stream);
  bool capturing = false;
  auto status = s.QueryCapture(stream, &capturing);
  if (!status.ok()) return status;
  if (capturing) return Status::InvalidState("prepare the Gather graph lease before capture");
  auto* lanes = static_cast<KvafSequentialStatus*>(s.status_host);
  for (uint32_t i = 0; i < s.config.npu_core_count; ++i)
    lanes[i] = {KVAF_SEQ_STATUS_MAGIC, s.sequence, KVAF_SEQ_OK, KVAF_SEQ_COMPLETE, 0, 0, 0, 0};
  const auto copied = aclrtMemcpy(s.status_device, s.StatusBytes(), s.status_host,
                                 s.StatusBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
  if (copied != ACL_SUCCESS) return s.AclFailure("initialize_graph_status", copied);
  uint64_t entity = 0;
  if (KvafBatchGetLocalBufferAcquire(s.gather_transport, options.output.data, options.output.bytes,
                                   &node_->output_buffer, &entity)) return Error("register graph output");
  node_->graph = {stream, s.sequence, entity, false};
  s.live_graph = node_;
  s.eager->profile_valid = false;
  return Status::OK();
}
Status GatherBatchGet::CheckBatchGetStatus(void* stream) {
  auto& s = *resources_;
  std::lock_guard<std::mutex> lock(s.mutex);
  if (s.live_graph != node_ || !node_ || !node_->graph.captured || node_->graph.stream != stream || s.failed)
    return Status::InvalidState("check Gather replay using its captured graph lease and stream");
  bool capturing = false;
  auto status = s.QueryCapture(node_->graph.stream, &capturing);
  if (!status.ok()) return status;
  if (capturing) return Status::InvalidState("finish Gather graph capture before checking replay");
  return s.ReadGraphStatus(*node_, true);
}
Status GatherBatchGet::FinishGetGraph(void* stream, bool discard) {
  auto& s = *resources_;
  std::lock_guard<std::mutex> lock(s.mutex);
  if (s.live_graph != node_ || !node_ || !node_->graph.stream || node_->graph.stream != stream)
    return Status::InvalidState("finish Gather using its prepared graph lease and stream");
  if (s.failed && !discard) return Status::InvalidState("destroy and discard the failed Gather graph");
  bool capturing = false;
  auto status = s.QueryCapture(node_->graph.stream, &capturing);
  if (!status.ok()) return status;
  if (capturing) return Status::InvalidState("finish Gather graph capture before releasing its lease");
  if (node_->graph.captured && !s.failed) {
    status = s.ReadGraphStatus(*node_, false);
    if (!status.ok() && !discard) return status;
  }
  if (s.failed) {
    const auto synchronized = aclrtSynchronizeStream(node_->graph.stream);
    if (synchronized != ACL_SUCCESS) return s.AclFailure("discard_graph_synchronize", synchronized);
    node_->recovery_fenced = true;
    // Failed DMA may still reference the registration after this node is unbound.
    s.retained_graph_output = std::exchange(node_->output_buffer, nullptr);
  } else {
    KvafBatchGetLocalBufferRelease(node_->output_buffer);
    node_->output_buffer = nullptr;
  }
  node_->graph = {};
  s.live_graph = nullptr;
  return Status::OK();
}
bool GatherBatchGet::GetGraphNeedsRecovery() const {
  return node_ && node_->recovery_fenced;
}
Status GatherBatchGet::SetGatherBatchGetProfiling(bool enabled) {
  auto& s = *resources_;
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!s.initialized) return Status::NotInitialized("gather transport is not initialized");
  if (!enabled) { s.profiling = false; return Status::OK(); }
  if (s.live_graph) return Status::InvalidState("release the Gather graph lease before profiling");
  if (s.args.probe) return Status::InvalidState("gather prefix probes cannot collect performance profiles");
  auto result = s.Drain();
  if (!result.ok()) return result;
  if (!s.profile_device &&
      aclrtMalloc(&s.profile_device, s.ProfileBytes(), ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
    return Error("allocate gather device profile");
  if (!s.profile_host && aclrtMallocHost(&s.profile_host, s.ProfileBytes()) != ACL_SUCCESS)
    return Error("allocate gather Host profile");
  s.profiling = true;
  s.eager->profile_valid = false;
  return Status::OK();
}
Status GatherBatchGet::CopyGatherBatchGetProfile(GatherBatchGetProfile* profile) const {
  auto& s = *resources_;
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!profile) return Status::InvalidArgument("gather profile output must not be null");
  *profile = {};
  if (!s.initialized) return Status::NotInitialized("gather transport is not initialized");
  if (s.failed || !s.eager->profile_valid)
    return Status::InvalidState("no successful synchronous gather profile is available");
  *profile = s.eager->latest_profile;
  return Status::OK();
}
Status GatherBatchGet::ExecuteBatchGet(const BatchGetExecRequest& exec) {
  return Launch(exec, {*resources_, *resources_->eager, node_});
}
Status GatherBatchGet::Shutdown() { return resources_->Close(); }
std::unique_ptr<SequentialBatchGet> CreateGatherBatchGet(AccessOperatorConfig config) {
  return std::make_unique<GatherBatchGet>(std::move(config));
}
}  // namespace kvaf
