#pragma once

#include "scatter_batch_get.h"
#include "sequential_batch_get_common.h"
#include "kvaf_aiv_batch_get.h"
#include "cpu_push_get_layout.h"
#include "get_host_timing.h"
#include "scatter_mailbox_templates.h"
#include "direct_hccp/direct_hccp_session.h"
#include "direct_hccp/target_channel.h"
#include "peer_link.h"
#include <acl/acl.h>
#include <algorithm>
#include <array>
#include <mutex>
#include <utility>

namespace kvaf {
class ScatterBatchGet final : public SequentialBatchGet {
 public:
  explicit ScatterBatchGet(AccessOperatorConfig config);
  ~ScatterBatchGet() override;
  std::unique_ptr<ExecutionResourceOwner> TakeResources() override;
  Status Initialize(const RemoteRegionDesc& region) override;
  Status ExecuteBatchGet(const BatchGetExecRequest& exec) override;
  Status DrainGet() override;
  Status Shutdown() override;
  Status SetBatchGetKernelProfiling(bool enabled) override;
  Status CopyScatterProfiles(std::vector<ScatterBatchGetCoreProfile>* profiles) const override;
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
    bool prepared{}, captured{}, recovery_fenced{};
  };
  std::unique_ptr<Resources> standalone_;
  Resources* resources_;
  Invocation* node_{};
};

struct ScatterBatchGet::Resources final : ExecutionResourceOwner {
  explicit Resources(AccessOperatorConfig c) : config(std::move(c)) {}
  Status Initialize(const RemoteRegionDesc&) override;
  AccessOperatorConfig config;
  std::mutex mutex;
  RemoteRegionDesc region{};
  KvafScatterBatchGetArgs args{};
  void* workspace{};
  void* status_device{};
  void* status_host{};
  direct_hccp::DirectHccpApi api{};
  direct_hccp::DirectHccpSession* session{};
  void* context{};
  void* token{};
  void* memory_handle{};
  void* mailbox_import{};
  uint64_t mailbox_context{};
  std::array<std::unique_ptr<direct_hccp::TargetChannel>, cpu_push::kMaxJettyCount> channels;
  uint32_t payload_jettys{cpu_push::kJettyCount}, payload_jettys_per_core{};
  Invocation fallback{this};
  Invocation* eager{&fallback};
  // Nodes in one graph share the workspace and advance a single device sequence.
  struct Graph {
    aclrtStream stream{};
    aclmdlRI id{};
    uint32_t nodes{};
    uint32_t leases{};
    bool checked{}, retiring{};
  } graph;
  uint64_t sequence{};
  bool push_open{}, initialized{}, failed{}, profile_ready{};
  uint64_t StatusBytes() const { return config.npu_core_count * sizeof(KvafSequentialStatus); }
  uint64_t ProfileBytes() const { return config.npu_core_count * sizeof(ScatterBatchGetCoreProfile); }

  Status CheckExecutionState() const {
    if (!initialized) return Status::NotInitialized("sequential BatchGet is not initialized");
    if (failed) return Status::InvalidState("sequential BatchGet failed; shutdown/reinitialize required");
    return Status::OK();
  }

  Status AclFailure(const char* phase, aclError code) const {
    return Status::DeviceError("AIV sequential BatchGet: mode=cpu_push_scatter sequence=" +
        std::to_string(sequence) + " phase=" + phase + " acl=" + std::to_string(code) +
        "; resources retained");
  }

  Status ReadStatus(aclrtStream stream, GetProviderTiming* timing = nullptr) {
    const auto copy = GetHostTimingStart(timing);
    const aclError copied = aclrtMemcpyAsync(status_host, StatusBytes(), status_device,
                                            StatusBytes(), ACL_MEMCPY_DEVICE_TO_HOST, stream);
    GetHostTimingRecord(timing, &GetProviderTiming::status_copy_ns, copy);
    if (copied != ACL_SUCCESS) {
      failed = true; return AclFailure("copy_status", copied);
    }
    const auto sync = GetHostTimingStart(timing);
    const aclError synchronized = aclrtSynchronizeStream(stream);
    GetHostTimingRecord(timing, &GetProviderTiming::stream_sync_ns, sync);
    if (synchronized != ACL_SUCCESS) {
      failed = true; return AclFailure("synchronize", synchronized);
    }
    return Status::OK();
  }

  Status ValidateStatus(uint64_t expected, GetProviderTiming* timing = nullptr) {
    BatchGetProfileScope validation(timing ? &timing->status_validate_ns : nullptr);
    const auto* statuses = static_cast<const KvafSequentialStatus*>(status_host);
    for (uint32_t i = 0; i < config.npu_core_count; ++i) {
      const auto& s = statuses[i];
      if (expected == 0 && s.sequence == 0 && s.magic == 0) continue;
      if (s.magic != KVAF_SEQ_STATUS_MAGIC || s.sequence != expected ||
          s.result != KVAF_SEQ_OK || s.phase != KVAF_SEQ_COMPLETE) {
        failed = true;
        return Status::Internal("AIV sequential BatchGet worker=" + std::to_string(i) +
            " sequence=" + std::to_string(expected) + " observed_sequence=" +
            std::to_string(s.sequence) + " phase=" + std::to_string(s.phase) +
            " result=" + std::to_string(s.result));
      }
    }
    return Status::OK();
  }

  Status Drain(GetProviderTiming* timing = nullptr) {
    if (failed) return Status::InvalidState("sequential BatchGet failed; shutdown/reinitialize required");
    if (!eager->pending) return Status::OK();
    BatchGetProfileScope completion(timing ? &timing->completion_ns : nullptr);
    auto status = ReadStatus(eager->pending, timing);
    if (!status.ok()) return status;
    eager->pending = nullptr;
    return ValidateStatus(sequence, timing);
  }

  Status QueryCapture(aclrtStream stream, bool* capturing, aclmdlRI* captured = nullptr) const {
    aclmdlRICaptureStatus capture = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    aclmdlRI active = nullptr;
    const auto result = aclmdlRICaptureGetInfo(stream, &capture, &active);
    if (result != ACL_SUCCESS) return sequential_detail::Error("query stream capture state");
    *capturing = capture != ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    if (captured) *captured = active;
    return Status::OK();
  }

  Status RequireOutsideCapture(aclrtStream stream) const {
    bool capturing = false;
    auto status = QueryCapture(stream, &capturing);
    if (!status.ok()) return status;
    return capturing ? Status::InvalidState("Scatter graph operation requires an idle capture stream")
                     : Status::OK();
  }

  // The status slab belongs to the complete graph, not to individual nodes.
  // Its final sequence advances once per captured Scatter kernel. Repeated
  // checks through sibling leases observe the same completed replay.
  Status ReadGraphStatus(bool closing) {
    auto status = ReadStatus(graph.stream);
    if (!status.ok()) return status;
    const auto observed = static_cast<const KvafSequentialStatus*>(status_host)[0].sequence;
    if (observed == sequence && (closing || graph.checked)) return ValidateStatus(sequence);
    if (graph.nodes > UINT32_MAX - sequence) {
      failed = true;
      return Status::InvalidState("BatchGet sequence exhausted; rebuild Runtime");
    }
    status = ValidateStatus(sequence + graph.nodes);
    if (status.ok()) {
      sequence += graph.nodes;
      graph.checked = true;
      profile_ready = ScatterProfileEnabled(args.reserved);
    }
    return status;
  }

  Status InitializePush(const peer::PeerRegion&, bool cpu_gather);
  Status Close() override;
};
}  // namespace kvaf
