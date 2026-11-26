#pragma once

#include "gather_batch_get.h"
#include "sequential_batch_get_common.h"
#include "kvaf_aiv_batch_get.h"
#include "gather_prefix_layout.h"
#include "gather_workspace.h"
#include "batch_get_profile.h"
#include "peer_link.h"
#include <acl/acl.h>
#include <algorithm>
#include <mutex>
#include <utility>

namespace kvaf {
namespace sequential_detail {
inline const char* ProbeName(uint32_t probe) {
  switch (probe) {
    case KVAF_GATHER_PROBE_ACK: return "ack";
    case KVAF_GATHER_PROBE_WQE: return "wqe";
    case KVAF_GATHER_PROBE_SQ: return "sq";
    case KVAF_GATHER_PROBE_FIRST_READ: return "first_read";
    default: return "none";
  }
}
}  // namespace sequential_detail

class GatherBatchGet final : public SequentialBatchGet {
 public:
  explicit GatherBatchGet(AccessOperatorConfig config);
  ~GatherBatchGet() override;
  std::unique_ptr<ExecutionResourceOwner> TakeResources() override;
  Status Initialize(const RemoteRegionDesc& region) override;
  Status ExecuteBatchGet(const BatchGetExecRequest& exec) override;
  Status DrainGet() override;
  Status Shutdown() override;
  Status SetGatherBatchGetProfiling(bool enabled) override;
  Status CopyGatherBatchGetProfile(GatherBatchGetProfile* profile) const override;
  std::unique_ptr<ExecutionState> CreateExecutionState() override;
  Status BindExecutionState(ExecutionState*) override;
  Status PrepareGetGraph(const GetGraphOptions&) override;
  Status CheckBatchGetStatus(void*) override;
  Status FinishGetGraph(void*, bool discard) override;
  bool GetGraphNeedsRecovery() const override;
 private:
  struct Resources;
  struct Invocation final : ExecutionState {
    explicit Invocation(const Resources* resources) : owner(resources) {}
    const Resources* owner;
    aclrtStream pending{};
    KvafBatchGetLocalBuffer* output_buffer{};
    bool pending_profile{}, profile_valid{}, pending_status_check{}, head_start{}, recovery_fenced{};
    uint64_t profile_begin_ns{};
    GatherBatchGetProfile latest_profile{};
    struct Graph {
      aclrtStream stream{};
      uint64_t initial_sequence{}, output_entity{};
      bool captured{};
    } graph;
  };
  struct LaunchView {
    Resources& resources;
    Invocation& invocation;
    Invocation* node;
  };
  static Status Launch(const BatchGetExecRequest&, LaunchView);
  std::unique_ptr<Resources> standalone_;
  Resources* resources_;
  Invocation* node_{};
};

struct GatherBatchGet::Resources final : ExecutionResourceOwner {
  explicit Resources(AccessOperatorConfig c) : config(std::move(c)) {}
  Status Initialize(const RemoteRegionDesc&) override;
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
  bool profiling{}, check_async_status{};
  KvafBatchGetTransport* gather_transport{};
  KvafBatchGetLocalBuffer* workspace_buffer{};
  KvafBatchGetLocalBuffer* retained_graph_output{};
  Invocation fallback{this};
  Invocation* eager{&fallback};
  // The sole lease pins the provider's shared workspace and status buffers.
  Invocation* live_graph{};
  uint64_t sequence{};
  bool gather_enabled{}, initialized{}, failed{};
  uint64_t StatusBytes() const { return config.npu_core_count * sizeof(KvafSequentialStatus); }
  uint64_t ProfileBytes() const {
    return config.npu_core_count * sizeof(KvafGatherBatchGetCoreProfile);
  }

  Status AclFailure(const char* phase, aclError code) const {
    return Status::DeviceError("AIV sequential BatchGet: mode=gather_read sequence=" +
        std::to_string(sequence) + " phase=" + phase + " acl=" + std::to_string(code) +
        " probe=" + sequential_detail::ProbeName(args.probe) + "; resources retained");
  }

  Status QueryCapture(aclrtStream stream, bool* capturing) const {
    aclmdlRICaptureStatus capture = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    aclmdlRI id = nullptr;
    if (aclmdlRICaptureGetInfo(stream, &capture, &id) != ACL_SUCCESS)
      return sequential_detail::Error("query stream capture state");
    *capturing = capture != ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    return Status::OK();
  }

  Status ValidateLane(const KvafSequentialStatus& lane, uint32_t worker,
                      uint64_t expected, uint64_t phase, uint64_t result) {
    if (lane.magic == KVAF_SEQ_STATUS_MAGIC && lane.sequence == expected &&
        lane.phase == phase && lane.result == result) return Status::OK();
    failed = true;
    return Status::Internal("AIV sequential BatchGet worker=" + std::to_string(worker) +
        " sequence=" + std::to_string(expected) + " observed_sequence=" +
        std::to_string(lane.sequence) + " phase=" + std::to_string(lane.phase) +
        " result=" + std::to_string(lane.result) + " expected_phase=" +
        std::to_string(phase) + " expected_result=" + std::to_string(result));
  }

  Status ReadGraphStatus(const Invocation& node, bool require_replay) {
    const auto& graph = node.graph;
    const auto copied = aclrtMemcpyAsync(status_host, StatusBytes(), status_device,
                                        StatusBytes(), ACL_MEMCPY_DEVICE_TO_HOST, graph.stream);
    if (copied != ACL_SUCCESS) { failed = true; return AclFailure("copy_graph_status", copied); }
    const auto synchronized = aclrtSynchronizeStream(graph.stream);
    if (synchronized != ACL_SUCCESS) { failed = true; return AclFailure("graph_synchronize", synchronized); }
    const auto* lanes = static_cast<const KvafSequentialStatus*>(status_host);
    const auto observed = lanes[0].sequence;
    for (uint32_t i = 0; i < config.npu_core_count; ++i) {
      auto status = ValidateLane(lanes[i], i, observed, KVAF_SEQ_COMPLETE, KVAF_SEQ_OK);
      if (!status.ok()) return status;
    }
    if (observed < sequence || observed > UINT32_MAX) {
      failed = true;
      return sequential_detail::Error("invalid Gather graph sequence");
    }
    if (require_replay && observed == graph.initial_sequence)
      return Status::InvalidState("Gather graph has not been replayed");
    sequence = observed;
    return Status::OK();
  }

  Status Drain() {
    if (failed) return Status::InvalidState("sequential BatchGet failed; shutdown/reinitialize required");
    if (!eager->pending) return Status::OK();
    if (!eager->pending_status_check) {
      // Ordinary async calls fence only: same-stream launches can remain queued.
      const auto synchronized = aclrtSynchronizeStream(eager->pending);
      if (synchronized != ACL_SUCCESS) {
        failed = true; return AclFailure("synchronize", synchronized);
      }
      eager->pending = nullptr;
      return Status::OK();
    }
    const auto enqueue_begin = eager->pending_profile ? BatchGetProfileNowNs() : 0;
    const aclError copied = aclrtMemcpyAsync(status_host, StatusBytes(), status_device,
                                            StatusBytes(), ACL_MEMCPY_DEVICE_TO_HOST, eager->pending);
    if (copied != ACL_SUCCESS) {
      failed = true; return AclFailure("copy_status", copied);
    }
    if (eager->pending_profile) {
      const auto profile_copied = aclrtMemcpyAsync(profile_host, ProfileBytes(), profile_device,
          ProfileBytes(), ACL_MEMCPY_DEVICE_TO_HOST, eager->pending);
      if (profile_copied != ACL_SUCCESS) {
        failed = true; return AclFailure("copy_profile", profile_copied);
      }
    }
    const auto wait_begin = eager->pending_profile ? BatchGetProfileNowNs() : 0;
    if (eager->pending_profile) eager->latest_profile.status_enqueue_ns = wait_begin - enqueue_begin;
    const aclError synchronized = aclrtSynchronizeStream(eager->pending);
    if (synchronized != ACL_SUCCESS) {
      failed = true; return AclFailure("synchronize", synchronized);
    }
    const auto finish_begin = eager->pending_profile ? BatchGetProfileNowNs() : 0;
    if (eager->pending_profile) eager->latest_profile.wait_ns = finish_begin - wait_begin;
    eager->pending = nullptr;
    eager->pending_status_check = false;
    const auto* statuses = static_cast<const KvafSequentialStatus*>(status_host);
    const uint64_t expected_result = args.probe ? KVAF_SEQ_PROBE_COMPLETE : KVAF_SEQ_OK;
    const uint64_t expected_phase = args.probe ? KVAF_GATHER_AFTER_ACK + args.probe - 1 :
        uint64_t(KVAF_SEQ_COMPLETE);
    uint64_t completed_requests = 0, data_wrs = 0;
    for (uint32_t i = 0; i < config.npu_core_count; ++i) {
      const auto& s = statuses[i];
      auto status = ValidateLane(s, i, sequence, expected_phase, expected_result);
      if (!status.ok()) return status;
      const uint64_t expected_wrs = args.probe == KVAF_GATHER_PROBE_FIRST_READ ?
          std::min<uint64_t>(KVAF_GATHER_WRS_PER_POST,
              (s.requests + args.sgl_size - 1) / args.sgl_size) : 0;
      if (args.probe && s.data_wrs != expected_wrs) {
        failed = true;
        return sequential_detail::Error("gather probe READ WR count mismatch");
      }
      completed_requests += s.requests;
      data_wrs += s.data_wrs;
    }
    if (eager->output_buffer) KvafBatchGetLocalBufferRelease(eager->output_buffer);
    eager->output_buffer = nullptr;
    if (args.probe) {
      if (!completed_requests) { failed = true; return sequential_detail::Error("gather probe completed no requests"); }
      // A prefix is a diagnostic, never successful BatchGet or a perf sample.
      return Status::NotImplemented("[KVAF_AIV_GATHER_PROBE_COMPLETE] stage=" +
          std::string(sequential_detail::ProbeName(args.probe)) + " sequence=" + std::to_string(sequence) +
          " cores=" + std::to_string(config.npu_core_count) +
          " requests=" + std::to_string(completed_requests) +
          " data_wrs=" + std::to_string(data_wrs));
    }
    if (eager->pending_profile) {
      const auto captured = CaptureProfile();
      if (!captured.ok()) { failed = true; return captured; }
      const auto finished = BatchGetProfileNowNs();
      eager->latest_profile.finish_ns = finished - finish_begin;
      eager->latest_profile.total_ns = finished - eager->profile_begin_ns;
      eager->profile_valid = true;
      eager->pending_profile = false;
    }
    return Status::OK();
  }

  Status CaptureProfile();

  Status InitializeGather();
  Status Close() override;
};
}  // namespace kvaf
