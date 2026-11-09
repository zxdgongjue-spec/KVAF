#include "gather_batch_get.h"
#include "sequential_batch_get_common.h"
#include "kvaf_aiv_batch_get.h"
#include "cpu_gather.h"
#include "gather_workspace.h"
#include "batch_get_profile.h"
#include "gather_call_profile.h"
#include "peer_link.h"
#include <acl/acl.h>
#include <algorithm>
#include <array>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <sstream>
#include <utility>

namespace kvaf {
using namespace sequential_detail;
namespace {
Status Error(const char* what) { return Status::Internal(std::string("AIV sequential BatchGet: ") + what); }
const char* ProbeName(uint32_t probe) {
  switch (probe) {
    case KVAF_GATHER_PROBE_ACK: return "ack";
    case KVAF_GATHER_PROBE_WQE: return "wqe";
    case KVAF_GATHER_PROBE_SQ: return "sq";
    case KVAF_GATHER_PROBE_FIRST_READ: return "first_read";
    default: return "none";
  }
}
Status ReadProbe(uint32_t* probe) {
  *probe = KVAF_GATHER_PROBE_NONE;
  const char* value = std::getenv("KVAF_AIV_GATHER_PROBE");
  if (!value || !*value) return Status::OK();
  for (uint32_t p = KVAF_GATHER_PROBE_ACK; p <= KVAF_GATHER_PROBE_FIRST_READ; ++p)
    if (std::strcmp(value, ProbeName(p)) == 0) { *probe = p; return Status::OK(); }
  return Status::InvalidArgument("KVAF_AIV_GATHER_PROBE must be ack/wqe/sq/first_read or empty");
}
static_assert(sizeof(KvafSequentialGatherHeader) == sizeof(peer::GatherDescriptorHeader));
static_assert(offsetof(KvafSequentialGatherHeader, ready_bytes) ==
              offsetof(peer::GatherDescriptorHeader, ready_bytes));
static_assert(sizeof(KvafSequentialGatherSpan) == sizeof(peer::GatherSpan));
}

class GatherBatchGet final : public SequentialBatchGet {
 public:
  explicit GatherBatchGet(AccessOperatorConfig config);
  ~GatherBatchGet() override;
  Status Initialize(const RemoteRegionDesc& region) override;
  Status Execute(const BatchGetExecRequest& exec) override;
  Status Drain() override;
  Status Shutdown() override;
  Status SetGatherProfiling(bool enabled) override;
  Status CopyGatherProfile(GatherBatchGetProfile* profile) const override;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct GatherBatchGet::Impl {
  explicit Impl(AccessOperatorConfig c) : config(std::move(c)) {}
  AccessOperatorConfig config;
  std::mutex mutex;
  RemoteRegionDesc region{};
  KvafGatherBatchGetArgs args{};
  void* workspace{};
  void* status_device{};
  void* status_host{};
  void* queues{};
  void* profile_device{};
  void* profile_host{};
  bool profiling{}, pending_profile{}, profile_valid{};
  bool check_async_status{}, pending_status_check{};
  uint64_t profile_begin_ns{};
  GatherBatchGetProfile latest_profile{};
  KvafBatchGetTransport* gather_transport{};
  KvafBatchGetLocalBuffer* workspace_buffer{};
  KvafBatchGetLocalBuffer* output_buffer{};
  aclrtStream pending{};
  uint64_t sequence{};
  bool gather_enabled{}, initialized{}, failed{};
  uint64_t StatusBytes() const { return config.npu_core_count * sizeof(KvafSequentialStatus); }
  uint64_t ProfileBytes() const {
    return config.npu_core_count * sizeof(KvafGatherBatchGetCoreProfile);
  }

  Status AclFailure(const char* phase, aclError code) const {
    return Status::DeviceError("AIV sequential BatchGet: mode=" + config.aiv_batch_get_mode +
        " sequence=" + std::to_string(sequence) + " phase=" + phase +
        " acl=" + std::to_string(code) + " probe=" + ProbeName(args.probe) +
        "; resources retained");
  }

  Status Drain() {
    if (failed) return Status::InvalidState("sequential BatchGet failed; shutdown/reinitialize required");
    if (!pending) return Status::OK();
    if (!pending_status_check) {
      // Match direct_read's asynchronous contract: completion is a stream
      // fence, not an implicit device-status readback. Same-stream submissions
      // can stay queued; a stream switch, explicit Drain or shutdown fences it.
      const auto synchronized = aclrtSynchronizeStream(pending);
      if (synchronized != ACL_SUCCESS) {
        failed = true; return AclFailure("synchronize", synchronized);
      }
      pending = nullptr;
      return Status::OK();
    }
    const auto enqueue_begin = pending_profile ? BatchGetProfileNowNs() : 0;
    gather_call::Timer copy_timer(gather_call::kStatusCopy);
    const aclError copied = aclrtMemcpyAsync(status_host, StatusBytes(), status_device,
                                            StatusBytes(), ACL_MEMCPY_DEVICE_TO_HOST, pending);
    if (copied != ACL_SUCCESS) {
      failed = true; return AclFailure("copy_status", copied);
    }
    if (pending_profile) {
      const auto profile_copied = aclrtMemcpyAsync(profile_host, ProfileBytes(), profile_device,
          ProfileBytes(), ACL_MEMCPY_DEVICE_TO_HOST, pending);
      if (profile_copied != ACL_SUCCESS) {
        failed = true; return AclFailure("copy_profile", profile_copied);
      }
    }
    copy_timer.Stop();
    const auto wait_begin = pending_profile ? BatchGetProfileNowNs() : 0;
    if (pending_profile) latest_profile.status_enqueue_ns = wait_begin - enqueue_begin;
    gather_call::Timer sync_timer(gather_call::kStreamSync);
    const aclError synchronized = aclrtSynchronizeStream(pending);
    if (synchronized != ACL_SUCCESS) {
      failed = true; return AclFailure("synchronize", synchronized);
    }
    sync_timer.Stop();
    gather_call::Timer validate_timer(gather_call::kStatusValidate);
    const auto finish_begin = pending_profile ? BatchGetProfileNowNs() : 0;
    if (pending_profile) latest_profile.wait_ns = finish_begin - wait_begin;
    pending = nullptr;
    pending_status_check = false;
    const auto* statuses = static_cast<const KvafSequentialStatus*>(status_host);
    const uint64_t expected_result = args.probe ? KVAF_SEQ_PROBE_COMPLETE : KVAF_SEQ_OK;
    const uint64_t expected_phase = args.probe ? KVAF_GATHER_AFTER_ACK + args.probe - 1 :
        uint64_t(KVAF_SEQ_COMPLETE);
    uint64_t completed_requests = 0, data_wrs = 0;
    for (uint32_t i = 0; i < config.npu_core_count; ++i) {
      const auto& s = statuses[i];
      if (s.magic != KVAF_SEQ_STATUS_MAGIC || s.sequence != sequence || s.result != expected_result ||
          s.phase != expected_phase) {
        failed = true;
        return Status::Internal("AIV sequential BatchGet worker=" + std::to_string(i) +
            " sequence=" + std::to_string(sequence) + " observed_sequence=" +
            std::to_string(s.sequence) + " phase=" + std::to_string(s.phase) +
            " result=" + std::to_string(s.result) +
            " expected_phase=" + std::to_string(expected_phase) +
            " expected_result=" + std::to_string(expected_result));
      }
      const uint64_t expected_wrs = args.probe == KVAF_GATHER_PROBE_FIRST_READ ?
          std::min<uint64_t>(KVAF_GATHER_WRS_PER_POST,
              (s.requests + args.sgl_size - 1) / args.sgl_size) : 0;
      if (args.probe && s.data_wrs != expected_wrs) {
        failed = true;
        return Error("gather probe READ WR count mismatch");
      }
      completed_requests += s.requests;
      data_wrs += s.data_wrs;
    }
    if (output_buffer) KvafBatchGetLocalBufferRelease(output_buffer);
    output_buffer = nullptr;
    if (args.probe) {
      if (!completed_requests) { failed = true; return Error("gather probe completed no requests"); }
      // A prefix is a diagnostic, never successful BatchGet or a perf sample.
      return Status::NotImplemented("[KVAF_AIV_GATHER_PROBE_COMPLETE] stage=" +
          std::string(ProbeName(args.probe)) + " sequence=" + std::to_string(sequence) +
          " cores=" + std::to_string(config.npu_core_count) +
          " requests=" + std::to_string(completed_requests) +
          " data_wrs=" + std::to_string(data_wrs));
    }
    if (pending_profile) {
      const auto captured = CaptureProfile();
      if (!captured.ok()) { failed = true; return captured; }
      const auto finished = BatchGetProfileNowNs();
      latest_profile.finish_ns = finished - finish_begin;
      latest_profile.total_ns = finished - profile_begin_ns;
      profile_valid = true;
      pending_profile = false;
    }
    return Status::OK();
  }

  Status CaptureProfile();

  Status InitializeGather(const peer::PeerRegion& peer) {
    const auto workers = config.npu_core_count;
    args.lane_descriptor_bytes = GatherLaneDescriptorBytes(workers);
    args.lane_staging_bytes = GatherLaneStagingBytes(config.batch_get_token_bytes, workers);
    args.descriptor_offset = workers * uint64_t(kGatherScratchBytes);
    args.workspace_bytes = args.descriptor_offset + workers * args.lane_descriptor_bytes;
    if (region.length > UINT64_MAX - 63 || region.base > UINT64_MAX - region.length - 63)
      return Status::InvalidArgument("gather address overflow");
    args.remote_staging = ((region.base + region.length + 63) & ~uint64_t{63});
    args.remote_descriptor = args.remote_staging + workers * args.lane_staging_bytes;
    args.remote_control = args.remote_descriptor + workers * args.lane_descriptor_bytes;
    if (peer.length < region.length || peer.length - region.length <
        GatherRemoteWorkspaceBytes(config.batch_get_token_bytes, workers))
      return Status::InvalidArgument("remote DDR lacks reserved gather workspace");
    // Prefer huge pages with the Runtime's normal-page fallback. Small queue
    // state allocations may still use normal pages under HUGE_FIRST.
    if (aclrtMalloc(&workspace, args.workspace_bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMemset(workspace, args.workspace_bytes, 0, args.workspace_bytes) != ACL_SUCCESS ||
        aclrtMalloc(&queues, workers * 2 * sizeof(KvafBatchGetQueueState), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
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
    for (auto p : {&workspace, &queues, &status_device, &profile_device}) {
      if (*p && aclrtFree(*p) != ACL_SUCCESS) return Error("free device memory");
      *p = nullptr;
    }
    if (status_host && aclrtFreeHost(status_host) != ACL_SUCCESS) return Error("free status memory");
    status_host = nullptr;
    if (profile_host && aclrtFreeHost(profile_host) != ACL_SUCCESS)
      return Error("free profile memory");
    profile_host = nullptr;
    profiling = pending_profile = profile_valid = false;
    check_async_status = pending_status_check = false;
    latest_profile = {};
    initialized = failed = false; sequence = 0; args = {};
    return execution_status;
  }
};

Status GatherBatchGet::Impl::CaptureProfile() {
  const auto* profiles = static_cast<const KvafGatherBatchGetCoreProfile*>(profile_host);
  const auto* statuses = static_cast<const KvafSequentialStatus*>(status_host);
  latest_profile.cores.clear();
  latest_profile.cores.reserve(config.npu_core_count);
  for (uint32_t i = 0; i < config.npu_core_count; ++i) {
    const auto& p = profiles[i];
    const char* reason = nullptr;
    if (p.magic != KVAF_GATHER_PROFILE_MAGIC) reason = "magic";
    else if (p.sequence != sequence) reason = "sequence";
    else if (p.result != KVAF_SEQ_OK) reason = "result";
    else if (p.valid_mask != KVAF_GATHER_PROFILE_ALL_EVENTS) reason = "valid_mask";
    else if (p.kernel_start_cycle > p.input_done_cycle) reason = "timestamp_start_input";
    else if (p.input_done_cycle > p.descriptor_done_cycle) reason = "timestamp_input_descriptor";
    else if (p.descriptor_done_cycle > p.notify_done_cycle) reason = "timestamp_descriptor_notify";
    else if (p.notify_done_cycle > p.ack_done_cycle) reason = "timestamp_notify_ack";
    else if (!p.pipeline_chunk_bytes && p.ack_done_cycle > p.read_done_cycle) reason = "timestamp_ack_read";
    else if (p.ack_done_cycle > p.kernel_finish_cycle) reason = "timestamp_ack_finish";
    else if (p.notify_done_cycle > p.read_done_cycle) reason = "timestamp_notify_read";
    else if (p.read_done_cycle > p.kernel_finish_cycle) reason = "timestamp_read_finish";
    else if (p.requests != statuses[i].requests) reason = "requests";
    else if (p.data_wrs != statuses[i].data_wrs) reason = "data_wrs";
    else if (p.pipeline_chunk_bytes != (p.requests ? args.pipeline_chunk_bytes : 0)) reason = "pipeline_chunk_bytes";
    else if (p.packed_posts != uint64_t(p.requests && args.pipeline_shards == 2)) reason = "packed_posts";
    else if (p.pipeline_chunk_bytes &&
        (p.first_ready_cycle < p.notify_done_cycle || p.first_ready_cycle > p.first_read_submit_cycle ||
         p.first_read_submit_cycle > p.read_done_cycle || !p.first_read_ready_bytes ||
         p.first_read_ready_bytes > p.requests * config.batch_get_token_bytes))
      reason = "timestamp_ready_submit_read";
    if (reason) {
      // The stream and normal status already passed. Preserve the actual
      // record in the exception before Shutdown frees it: a generic error here
      // cannot distinguish missing publication, ABI mismatch, clock ordering
      // or duplicate request/WR accounting. Never return a partial snapshot.
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
    out.pipeline_shards = p.requests && args.pipeline_shards;
    out.read_vf_calls = p.read_vf_calls;
    out.ready_no_change_queries = p.ready_no_change_queries;
    out.ready_query_skips = p.ready_query_skips;
    out.max_ready_backlog_wrs = p.max_ready_backlog_wrs;
    out.ready_idle_waits = p.ready_idle_waits;
    out.multi_run_posts = p.multi_run_posts;
    out.packed_posts = p.packed_posts;
    out.read_plan_segments = p.read_plan_segments;
    latest_profile.cores.push_back(out);
  }
  return Status::OK();
}

GatherBatchGet::GatherBatchGet(AccessOperatorConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}
GatherBatchGet::~GatherBatchGet() {
  if (!Shutdown().ok()) (void)impl_.release();
}
Status GatherBatchGet::Initialize(const RemoteRegionDesc& region) {
  auto& s = *impl_;
  std::lock_guard<std::mutex> lock(s.mutex);
  if (s.initialized || s.failed || s.workspace || s.status_device || s.status_host)
    return Status::InvalidState("sequential transport already initialized; shutdown before retry");
  if (!s.config.peer_link || !s.config.npu_core_count || s.config.npu_core_count > KVAF_SEQ_MAX_WORKERS ||
      !s.config.batch_get_token_bytes || uint64_t(s.config.batch_get_token_bytes) * KVAF_SEQ_MAX_REQUESTS > UINT32_MAX ||
      !s.config.aiv_gather_sgl_size || s.config.aiv_gather_sgl_size > KVAF_SEQ_MAX_SGES ||
      s.config.aiv_batch_get_mode != "gather_read" ||
      s.config.peer_local_eids.size() > 1)
    return Status::InvalidArgument("invalid sequential BatchGet configuration");
  auto result = ReadProbe(&s.args.probe);
  if (!result.ok()) return result;
  s.args.pipeline_chunk_bytes = kGatherChunkBytes;
  const char* chunk = std::getenv("KVAF_AIV_GATHER_CHUNK_BYTES");
  if (chunk && *chunk) {
    const auto parsed = std::from_chars(chunk, chunk + std::strlen(chunk), s.args.pipeline_chunk_bytes);
    if (parsed.ec != std::errc{} || *parsed.ptr != '\0' ||
        (s.args.pipeline_chunk_bytes &&
         (s.args.pipeline_chunk_bytes < 64 || s.args.pipeline_chunk_bytes % 64)))
      return Status::InvalidArgument("KVAF_AIV_GATHER_CHUNK_BYTES must be 0 or a uint32 multiple of 64");
  }
  // Prefix diagnostics retain their original whole-batch contract. The general
  // split-token encoder is unchanged in this first chunk-consumer version.
  if (s.args.probe || uint64_t(s.config.batch_get_token_bytes) * s.config.aiv_gather_sgl_size > 65536)
    s.args.pipeline_chunk_bytes = 0;
  const char* shards = std::getenv("KVAF_AIV_GATHER_SHARD_PIPELINE");
  if (shards && *shards && std::strcmp(shards, "0") && std::strcmp(shards, "1"))
    return Status::InvalidArgument("KVAF_AIV_GATHER_SHARD_PIPELINE must be 0 or 1");
  s.args.pipeline_shards = s.args.pipeline_chunk_bytes &&
      (!shards || !*shards || std::strcmp(shards, "0"));
  const char* packed = std::getenv("KVAF_AIV_GATHER_PACKED_POSTS");
  if (packed && *packed && std::strcmp(packed, "0") && std::strcmp(packed, "1"))
    return Status::InvalidArgument("KVAF_AIV_GATHER_PACKED_POSTS must be 0 or 1");
  if (s.args.pipeline_shards && (!packed || !*packed || std::strcmp(packed, "0")))
    s.args.pipeline_shards = 2;
  const char* coalesce = std::getenv("KVAF_AIV_GATHER_COALESCE_SGES");
  if (coalesce && *coalesce && std::strcmp(coalesce, "0") && std::strcmp(coalesce, "1"))
    return Status::InvalidArgument("KVAF_AIV_GATHER_COALESCE_SGES must be 0 or 1");
  s.args.coalesce_sges = coalesce && std::strcmp(coalesce, "1") == 0;
  if (s.args.coalesce_sges &&
      uint64_t(s.config.batch_get_token_bytes) * s.config.aiv_gather_sgl_size > 65536)
    return Status::InvalidArgument("coalesced gather requires token_bytes * SGL <= 65536");
  if (s.args.probe > KVAF_GATHER_PROBE_ACK &&
      uint64_t(s.config.batch_get_token_bytes) * s.config.aiv_gather_sgl_size > 65536)
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
  result = s.InitializeGather(peer);
  if (!result.ok()) return result;
  s.args.workspace = Ptr(s.workspace);
  s.initialized = true;
  std::fprintf(stderr, "[KVAF_AIV] GATHER_OPTIONS pipeline=%s scheduler=%s chunk_bytes=%u max_shards=%u\n",
      s.args.pipeline_shards ? "shard" : s.args.pipeline_chunk_bytes ? "prefix" : "whole_batch",
      s.args.pipeline_shards == 2 ? "packed" : "legacy", s.args.pipeline_chunk_bytes, kGatherMaxShards);
  return Status::OK();
}
Status GatherBatchGet::Drain() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->Drain();
}
Status GatherBatchGet::SetGatherProfiling(bool enabled) {
  auto& s = *impl_;
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!s.initialized) return Status::NotInitialized("gather transport is not initialized");
  if (!enabled) { s.profiling = false; return Status::OK(); }
  if (s.args.probe) return Status::InvalidState("gather prefix probes cannot collect performance profiles");
  auto result = s.Drain();
  if (!result.ok()) return result;
  if (!s.profile_device &&
      aclrtMalloc(&s.profile_device, s.ProfileBytes(), ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
    return Error("allocate gather device profile");
  if (!s.profile_host && aclrtMallocHost(&s.profile_host, s.ProfileBytes()) != ACL_SUCCESS)
    return Error("allocate gather Host profile");
  s.profiling = true;
  s.profile_valid = false;
  return Status::OK();
}
Status GatherBatchGet::CopyGatherProfile(GatherBatchGetProfile* profile) const {
  auto& s = *impl_;
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!profile) return Status::InvalidArgument("gather profile output must not be null");
  *profile = {};
  if (!s.initialized) return Status::NotInitialized("gather transport is not initialized");
  if (s.failed || !s.profile_valid)
    return Status::InvalidState("no successful synchronous gather profile is available");
  *profile = s.latest_profile;
  return Status::OK();
}
Status GatherBatchGet::Execute(const BatchGetExecRequest& exec) {
  gather_call::Timer operator_timer(gather_call::kOperator);
  auto& s = *impl_;
  gather_call::Timer mutex_timer(gather_call::kMutexWait);
  std::lock_guard<std::mutex> lock(s.mutex);
  mutex_timer.Stop();
  if (!s.initialized) return Status::NotInitialized("sequential BatchGet is not initialized");
  if (s.failed) return Status::InvalidState("sequential BatchGet failed; shutdown/reinitialize required");
  if (!exec.request || !exec.execution.stream) return Status::InvalidArgument("BatchGet needs request and stream");
  if (!exec.execution.trusted_npu_request) {
    gather_call::Timer capture_timer(gather_call::kCaptureQuery);
    aclmdlRICaptureStatus capture = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    aclmdlRI graph = nullptr;
    if (aclmdlRICaptureGetInfo(static_cast<aclrtStream>(exec.execution.stream), &capture, &graph) != ACL_SUCCESS)
      return Error("query stream capture state");
    if (capture != ACL_MODEL_RI_CAPTURE_STATUS_NONE)
      return Status::NotImplemented("sequential BatchGet supports eager execution only");
  }
  if (s.profiling && !exec.execution.synchronize)
    return Status::InvalidArgument("gather profiling requires synchronize=True");
  if (s.args.probe && !exec.execution.synchronize)
    return Status::InvalidArgument("gather prefix probes require synchronize=True");
  // Stream ordering protects the shared descriptor, staging and SQ/CQ state.
  // Only a stream switch or a pending checked/diagnostic call needs a Host
  // fence before the next launch. Ordinary same-stream calls remain async.
  Status result = Status::OK();
  gather_call::Timer prior_timer(gather_call::kPriorDrain);
  if (s.pending && (s.pending != exec.execution.stream || s.pending_status_check)) {
    result = s.Drain();
    if (!result.ok()) return result;
  }
  prior_timer.Stop();
  s.profile_valid = false;
  if (s.profiling) {
    s.profile_begin_ns = BatchGetProfileNowNs();
    s.latest_profile = {};
  }
  auto args = s.args;
  gather_call::Timer input_timer(gather_call::kInput);
  if (exec.execution.trusted_npu_request) FillInput(*exec.request, s.config, &args.request);
  else result = Input(exec, s.config, &args.request);
  if (!result.ok()) return result;
  input_timer.Stop();
  if (s.sequence == UINT32_MAX) return Status::InvalidState("BatchGet sequence exhausted; rebuild Runtime");
  gather_call::Timer acquire_timer(gather_call::kOutputAcquire);
  if (KvafBatchGetLocalBufferAcquire(s.gather_transport, exec.request->output.data,
      exec.request->output.bytes, &s.output_buffer, &args.request.channel_entity))
    return Error("register output");
  acquire_timer.Stop();
  args.request.core_status = Ptr(s.status_device); args.request.queue_state = Ptr(s.queues);
  s.args.request = args.request;
  args.sequence = ++s.sequence;
  args.profile = s.profiling ? Ptr(s.profile_device) : 0;
  s.pending_profile = s.profiling;
  s.pending_status_check = exec.execution.synchronize || s.profiling ||
      args.probe || s.check_async_status;
  const auto launch_begin = s.profiling ? BatchGetProfileNowNs() : 0;
  if (s.profiling) {
    s.latest_profile.sequence = s.sequence;
    s.latest_profile.prepare_ns = launch_begin - s.profile_begin_ns;
  }
  s.pending = static_cast<aclrtStream>(exec.execution.stream);
  if (args.probe)
    std::fprintf(stderr, "[KVAF_AIV_GATHER_PROBE_BEGIN] stage=%s sequence=%llu cores=%u "
        "token_bytes=%u request_capacity=%u\n", ProbeName(args.probe),
        static_cast<unsigned long long>(args.sequence), args.request.block_count,
        args.request.token_bytes, args.request.request_capacity);
  // Every launched core initializes its complete status and, when enabled,
  // profile record. Host clears add stream tasks without producing new data.
  gather_call::Timer call_launch_timer(gather_call::kLaunch);
  if (KvafGatherBatchGetLaunch(&args, s.pending)) {
    s.failed = true; return Error("enqueue failed; resources retained");
  }
  call_launch_timer.Stop();
  if (s.profiling) s.latest_profile.launch_ns = BatchGetProfileNowNs() - launch_begin;
  if (!s.pending_status_check) {
    // Zero-reference registrations and entity tables remain cached until
    // transport shutdown, exactly as in the direct asynchronous path.
    KvafBatchGetLocalBufferRelease(s.output_buffer);
    s.output_buffer = nullptr;
  }
  return exec.execution.synchronize ? s.Drain() : Status::OK();
}
Status GatherBatchGet::Shutdown() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->Shutdown();
}
std::unique_ptr<SequentialBatchGet> CreateGatherBatchGet(AccessOperatorConfig config) {
  return std::make_unique<GatherBatchGet>(std::move(config));
}
}  // namespace kvaf
