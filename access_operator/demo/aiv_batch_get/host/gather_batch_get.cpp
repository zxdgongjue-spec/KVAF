#include "gather_batch_get.h"
#include "sequential_batch_get_common.h"
#include "kvaf_aiv_batch_get.h"
#include "cpu_gather.h"
#include "gather_workspace.h"
#include "peer_link.h"
#include <acl/acl.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <utility>

namespace kvaf {
using namespace sequential_detail;
namespace {
Status Error(const char* what) { return Status::Internal(std::string("AIV sequential BatchGet: ") + what); }
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
  KvafBatchGetTransport* gather_transport{};
  KvafBatchGetLocalBuffer* workspace_buffer{};
  KvafBatchGetLocalBuffer* output_buffer{};
  aclrtStream pending{};
  uint64_t sequence{};
  bool gather_enabled{}, initialized{}, failed{};
  uint64_t StatusBytes() const { return config.npu_core_count * sizeof(KvafSequentialStatus); }

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
    const uint64_t expected_result = KVAF_SEQ_OK;
    const uint64_t expected_phase = KVAF_SEQ_COMPLETE;
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
    }
    if (output_buffer) KvafBatchGetLocalBufferRelease(output_buffer);
    output_buffer = nullptr;
    return Status::OK();
  }

  Status InitializeGather(const peer::PeerRegion& peer) {
    const auto workers = config.npu_core_count;
    args.lane_descriptor_bytes = GatherLaneDescriptorBytes(workers);
    args.lane_staging_bytes = GatherLaneStagingBytes(config.batch_get_token_bytes, workers);
    args.descriptor_offset = workers * 64ULL;
    args.workspace_bytes = args.descriptor_offset + workers * args.lane_descriptor_bytes;
    if (region.length > UINT64_MAX - 63 || region.base > UINT64_MAX - region.length - 63)
      return Status::InvalidArgument("gather address overflow");
    args.remote_staging = ((region.base + region.length + 63) & ~uint64_t{63});
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
  s.region = region;
  if (s.config.peer_local_eids.size() == 1) s.config.peer_local_eid = s.config.peer_local_eids.front();
  peer::PeerRegion peer;
  auto result = s.config.peer_link->LookupRegion(region.base, &peer);
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
  s.args.max_polls = 100000000U;
  s.args.source_base = region.base; s.args.source_bytes = region.length;
  result = s.InitializeGather(peer);
  if (!result.ok()) return result;
  s.args.workspace = Ptr(s.workspace);
  s.initialized = true;
  return Status::OK();
}
Status GatherBatchGet::Drain() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->Drain();
}
Status GatherBatchGet::Execute(const BatchGetExecRequest& exec) {
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
  if (KvafBatchGetLocalBufferAcquire(s.gather_transport, exec.request->output.data,
      exec.request->output.bytes, &s.output_buffer, &args.request.channel_entity))
    return Error("register output");
  args.request.core_status = Ptr(s.status_device); args.request.queue_state = Ptr(s.queues);
  args.sequence = ++s.sequence;
  s.pending = static_cast<aclrtStream>(exec.execution.stream);
  if (aclrtMemsetAsync(s.status_device, s.StatusBytes(), 0, s.StatusBytes(), s.pending) != ACL_SUCCESS ||
      KvafGatherBatchGetLaunch(&args, s.pending)) {
    s.failed = true; return Error("enqueue failed; resources retained");
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
