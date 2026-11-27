#include "gather_execution.h"
#include "peer_link.h"
#include "gather_prefix_layout.h"
#include "gather_workspace.h"
#include "sequential_batch_get_common.h"

#include <charconv>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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

Status GatherResources::InitializeGather() {
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
      workers, 2, config.peer_link.get(), config.peer_local_eid.data(), &outputs.transport_))
    return Error("create gather control/data queues");
  if (KvafBatchGetLocalBufferAcquire(outputs.transport_, workspace, args.workspace_bytes,
      &workspace_buffer, &args.workspace_entity)) return Error("register gather workspace");
  gather_enabled = true;  // STOP also handles an uncertain enable response.
  return config.peer_link->EnableGatherLanes(region.base, args.remote_control, workers, true);
}

Status GatherResources::Close() {
  std::lock_guard<std::mutex> lock(mutex);
  if (progress.live_graph) return Status::Busy("release the Gather graph lease before shutdown");
  // Finish pending CPU work; on errors quiesce transports before freeing memory.
  const auto execution_status = progress.health.failed ? Status::OK() :
      GatherHostOperator::Drain(View(), *active, *eager);
  progress.health.failed = true;  // A partially closed transport cannot accept another Execute.
  views.Close();
  if (eager->pending) {
    const aclError synchronized = aclrtSynchronizeStream(eager->pending);
    if (synchronized != ACL_SUCCESS) return View().AclFailure("shutdown_synchronize", synchronized);
  }
  eager->pending = nullptr;
  // Stop CPU copy workers before releasing descriptor/staging resources.
  if (gather_enabled) {
    auto result = config.peer_link->StopGatherLanes(region.base);
    if (!result.ok()) return result;
    gather_enabled = false;
  }
  if (outputs.transport_ && KvafGatherBatchGetTransportQuiesce(outputs.transport_))
    return Error("cannot quiesce gather DMA; resources retained");
  outputs.Release(eager->output_buffer);
  outputs.Release(outputs.retained_);
  if (workspace_buffer) KvafBatchGetLocalBufferRelease(workspace_buffer);
  workspace_buffer = nullptr;
  if (outputs.transport_) KvafBatchGetTransportDestroy(outputs.transport_);
  outputs.transport_ = nullptr;
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
  initialized = false;
  progress = {};
  args = {};
  return execution_status;
}

Status GatherResources::Initialize(const RemoteRegionDesc& region) {
  auto& s = *this;
  std::lock_guard<std::mutex> lock(s.mutex);
  if (s.initialized || s.progress.health.failed || s.workspace || s.status_device || s.status_host)
    return Status::InvalidState("sequential transport already initialized; shutdown before retry");
  if (!s.config.peer_link || !s.config.npu_core_count || s.config.npu_core_count > KVAF_SEQ_MAX_WORKERS ||
      !s.config.batch_get_token_bytes || uint64_t(s.config.batch_get_token_bytes) * kDefaultGetWindow > UINT32_MAX ||
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
  s.views.Initialize(region);
  s.views.Bind(s.active);
  const char* pipeline = s.args.first_chunk_divisor ? "bootstrap_prefix" :
      s.args.cooperative_chunks ? "cooperative_prefix" : s.args.pipeline_chunk_bytes ? "prefix" : "whole_batch";
  std::fprintf(stderr, "[KVAF_AIV] GATHER_OPTIONS pipeline=%s scheduler=%s chunk_bytes=%u prepare_ahead=%u "
      "cooperative_chunks=%u first_chunk_divisor=%u\n", pipeline, s.args.prepare_ahead ? "prepared" : "baseline",
      s.args.pipeline_chunk_bytes, s.args.prepare_ahead, s.args.cooperative_chunks, s.args.first_chunk_divisor);
  return Status::OK();
}

Status GatherResources::SetProfiling(bool enabled) {
  auto& s = *this;
  if (!s.initialized) return Status::NotInitialized("gather transport is not initialized");
  if (!enabled) {
    s.profiling = false;
    views.Prepare(ExecutionDirection::Get);
    return Status::OK();
  }
  if (s.progress.live_graph) return Status::InvalidState("release the Gather graph lease before profiling");
  if (s.args.probe) return Status::InvalidState("gather prefix probes cannot collect performance profiles");
  auto result = GatherHostOperator::Drain(View(), *active, *eager);
  if (!result.ok()) return result;
  if (!s.profile_device &&
      aclrtMalloc(&s.profile_device, s.ProfileBytes(), ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
    return Error("allocate gather device profile");
  if (!s.profile_host && aclrtMallocHost(&s.profile_host, s.ProfileBytes()) != ACL_SUCCESS)
    return Error("allocate gather Host profile");
  s.profiling = true;
  s.eager->profile_valid = false;
  views.Prepare(ExecutionDirection::Get);
  return Status::OK();
}

int GatherOutputRegistration::Acquire(const MutableBufferView& output,
    KvafBatchGetLocalBuffer** buffer, uint64_t* entity) {
  return KvafBatchGetLocalBufferAcquire(transport_, output.data, output.bytes, buffer, entity);
}
void GatherOutputRegistration::Release(KvafBatchGetLocalBuffer*& buffer) {
  if (buffer) KvafBatchGetLocalBufferRelease(std::exchange(buffer, nullptr));
}
void GatherOutputRegistration::Preserve(KvafBatchGetLocalBuffer*& buffer) {
  retained_ = std::exchange(buffer, nullptr);
}

GatherLaunchView GatherResources::View() {
  return {views.View(ExecutionDirection::Get), args, progress, outputs,
          config.npu_core_count, config.batch_get_token_bytes, status_device,
          status_host, queues, profile_device, profile_host, initialized, profiling, check_async_status};
}
}  // namespace kvaf
