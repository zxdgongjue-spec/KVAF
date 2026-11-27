#include "scatter_batch_get.h"
#include "sequential_batch_get_common.h"
#include "get_host_timing.h"

#include <cstddef>

namespace kvaf {
using namespace sequential_detail;
namespace {
Status PrepareScatterBatchGetArguments(const BatchGetExecRequest& exec, const ScatterLaunchView& view,
                                       KvafScatterBatchGetArgs& args) {
  const auto status = exec.execution.trusted_npu_request
      ? sequential_detail::FillInput(*exec.request, view.cores, view.token_bytes, &args.request)
      : sequential_detail::Input(exec, {view.cores, view.token_bytes}, &args.request);
  if (!status.ok()) return status;
  if (exec.execution.trusted_npu_request) args.reserved |= kScatterTrustedInputs;
  if (view.jettys_per_core) args.request.jettys_per_core = view.jettys_per_core;
  args.request.core_status = Ptr(view.status_device);
  args.request.channel_entity = view.mailbox_address;
  args.request.queue_state = view.mailbox_address + offsetof(KvafScatterMailboxContext, queue);
  return Status::OK();
}

Status CheckExecutionState(const ScatterLaunchView& view) {
  if (!view.initialized) return Status::NotInitialized("sequential BatchGet is not initialized");
  if (view.progress.health.failed) return Status::InvalidState("sequential BatchGet failed; shutdown/reinitialize required");
  return Status::OK();
}

Status ReadStatus(const ScatterLaunchView& view, aclrtStream stream, GetProviderTiming* timing) {
  const auto copy = GetHostTimingStart(timing);
  const aclError copied = aclrtMemcpyAsync(view.status_host, view.StatusBytes(), view.status_device,
                                          view.StatusBytes(), ACL_MEMCPY_DEVICE_TO_HOST, stream);
  GetHostTimingRecord(timing, &GetProviderTiming::status_copy_ns, copy);
  if (copied != ACL_SUCCESS) {
    view.progress.health.failed = true; return view.AclFailure("copy_status", copied);
  }
  const auto sync = GetHostTimingStart(timing);
  const aclError synchronized = aclrtSynchronizeStream(stream);
  GetHostTimingRecord(timing, &GetProviderTiming::stream_sync_ns, sync);
  if (synchronized != ACL_SUCCESS) {
    view.progress.health.failed = true; return view.AclFailure("synchronize", synchronized);
  }
  return Status::OK();
}

Status ValidateStatus(const ScatterLaunchView& view, uint64_t expected, GetProviderTiming* timing = nullptr) {
  BatchGetProfileScope validation(timing ? &timing->status_validate_ns : nullptr);
  const auto* statuses = static_cast<const KvafSequentialStatus*>(view.status_host);
  for (uint32_t i = 0; i < view.cores; ++i) {
    const auto& lane = statuses[i];
    if (expected == 0 && lane.sequence == 0 && lane.magic == 0) continue;
    if (lane.magic != KVAF_SEQ_STATUS_MAGIC || lane.sequence != expected ||
        lane.result != KVAF_SEQ_OK || lane.phase != KVAF_SEQ_COMPLETE) {
      view.progress.health.failed = true;
      return Status::Internal("AIV sequential BatchGet worker=" + std::to_string(i) +
          " sequence=" + std::to_string(expected) + " observed_sequence=" +
          std::to_string(lane.sequence) + " phase=" + std::to_string(lane.phase) +
          " result=" + std::to_string(lane.result));
    }
  }
  return Status::OK();
}

Status QueryCapture(aclrtStream stream, bool* capturing, aclmdlRI* captured = nullptr) {
  aclmdlRICaptureStatus capture = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
  aclmdlRI active = nullptr;
  const auto result = aclmdlRICaptureGetInfo(stream, &capture, &active);
  if (result != ACL_SUCCESS) return sequential_detail::Error("query stream capture state");
  *capturing = capture != ACL_MODEL_RI_CAPTURE_STATUS_NONE;
  if (captured) *captured = active;
  return Status::OK();
}

Status RequireOutsideCapture(aclrtStream stream) {
  bool capturing = false;
  auto status = QueryCapture(stream, &capturing);
  if (!status.ok()) return status;
  return capturing ? Status::InvalidState("Scatter graph operation requires an idle capture stream")
                   : Status::OK();
}

// The status slab belongs to the complete graph, not to individual nodes.
// Its final sequence advances once per captured Scatter kernel. Repeated
// checks through sibling leases observe the same completed replay.
Status ReadGraphStatus(const ScatterLaunchView& view, bool closing) {
  auto status = ReadStatus(view, view.progress.graph.stream, nullptr);
  if (!status.ok()) return status;
  const auto observed = static_cast<const KvafSequentialStatus*>(view.status_host)[0].sequence;
  if (observed == view.progress.sequence && (closing || view.progress.graph.checked))
    return ValidateStatus(view, view.progress.sequence);
  if (view.progress.graph.nodes > UINT32_MAX - view.progress.sequence) {
    view.progress.health.failed = true;
    return Status::InvalidState("BatchGet sequence exhausted; rebuild Runtime");
  }
  status = ValidateStatus(view, view.progress.sequence + view.progress.graph.nodes);
  if (status.ok()) {
    view.progress.sequence += view.progress.graph.nodes;
    view.progress.graph.checked = true;
    view.progress.profile_ready = ScatterProfileEnabled(view.args.reserved);
  }
  return status;
}
}  // namespace

Status ScatterLaunchView::AclFailure(const char* phase, aclError code) const {
  return Status::DeviceError("AIV sequential BatchGet: mode=cpu_push_scatter sequence=" +
      std::to_string(progress.sequence) + " phase=" + phase + " acl=" + std::to_string(code) +
      "; resources retained");
}

Status ScatterHostOperator::Drain(const ScatterLaunchView& view, const ScatterInvocation& active,
    ScatterInvocation& eager, GetProviderTiming* timing) {
  if (view.progress.health.failed) return Status::InvalidState("sequential BatchGet failed; shutdown/reinitialize required");
  if (!eager.pending) return Status::OK();
  const auto valid = view.common.Validate(&active, ExecutionDirection::Get);
  if (!valid.ok()) return valid;
  BatchGetProfileScope completion(timing ? &timing->completion_ns : nullptr);
  auto status = ReadStatus(view, eager.pending, timing);
  if (!status.ok()) return status;
  eager.pending = nullptr;
  return ValidateStatus(view, view.progress.sequence, timing);
}

Status ScatterHostOperator::PrepareGraph(const GetGraphOptions& options,
    const ScatterLaunchView& view, ScatterInvocation& node, const ScatterInvocation& eager) {
  const auto status = CheckExecutionState(view);
  if (!status.ok()) return status;
  const auto valid = view.common.Validate(&node, ExecutionDirection::Get);
  if (!valid.ok()) return valid;
  const auto stream = static_cast<aclrtStream>(options.stream);
  if (!node.graph_node || node.prepared || !stream || eager.pending || view.progress.graph.id || view.progress.graph.retiring ||
      (view.progress.graph.stream && view.progress.graph.stream != stream) || view.progress.graph.leases == UINT32_MAX)
    return Status::InvalidState("prepare all Scatter graph leases on one idle stream before capture");
  auto capture = RequireOutsideCapture(stream);
  if (!capture.ok()) return capture;
  view.progress.graph.stream = stream;
  ++view.progress.graph.leases;
  node.prepared = true;
  return Status::OK();
}
Status ScatterHostOperator::CheckGraph(void* stream, const ScatterLaunchView& view, const ScatterInvocation& node) {
  const auto status = CheckExecutionState(view);
  if (!status.ok()) return status;
  if (!node.graph_node || !node.captured || view.progress.graph.stream != stream || view.progress.graph.retiring)
    return Status::InvalidState("check Scatter replay using a captured graph lease and its stream");
  const auto valid = view.common.Validate(&node, ExecutionDirection::Get);
  if (!valid.ok()) return valid;
  auto capture = RequireOutsideCapture(view.progress.graph.stream);
  return capture.ok() ? ReadGraphStatus(view, false) : capture;
}
Status ScatterHostOperator::FinishGraph(void* stream, bool discard,
    const ScatterLaunchView& view, ScatterInvocation& node) {
  if (!node.graph_node || !node.prepared || view.progress.graph.stream != stream)
    return Status::InvalidState("finish Scatter using its prepared graph lease and stream");
  const auto valid = view.common.Validate(&node, ExecutionDirection::Get);
  if (!valid.ok()) return valid;
  if (view.progress.health.failed && !discard)
    return Status::InvalidState("destroy and discard the failed Scatter graph");
  if (!view.progress.graph.retiring) {
    auto status = RequireOutsideCapture(view.progress.graph.stream);
    if (!status.ok()) return status;
    if (view.progress.graph.nodes && !view.progress.health.failed) {
      status = ReadGraphStatus(view, true);
      if (!status.ok() && !discard) return status;
    }
    if (view.progress.health.failed) {
      const auto result = aclrtSynchronizeStream(view.progress.graph.stream);
      if (result != ACL_SUCCESS) return view.AclFailure("external_graph_synchronize", result);
    }
    view.progress.graph.retiring = true;
  }
  node.prepared = false;
  node.recovery_fenced = view.progress.health.failed;
  if (--view.progress.graph.leases == 0) {
    view.progress.graph = {};
    view.progress.profile_ready = false;
  }
  return Status::OK();
}

Status ScatterHostOperator::Execute(const BatchGetExecRequest& exec, const ScatterLaunchView& view,
    ScatterInvocation& active, ScatterInvocation& invocation) {
  auto* timing = exec.host_timing;
  auto state = CheckExecutionState(view);
  if (!state.ok()) return state;
  const auto valid = view.common.Validate(&active, ExecutionDirection::Get);
  if (!valid.ok()) return valid;
  auto& node = active;
  const bool attach_graph = exec.execution.graph_lease != 0;
  if (view.progress.graph.leases && !attach_graph)
    return Status::InvalidState("release Scatter graph leases before another eager BatchGet");
  if (!exec.request || !exec.execution.stream)
    return Status::InvalidArgument("BatchGet needs a request and stream");
  const auto prepare = GetHostTimingStart(timing);
  const auto stream = static_cast<aclrtStream>(exec.execution.stream);
  aclmdlRI external_capture = nullptr;
  if (attach_graph || !exec.execution.trusted_npu_request) {
    bool capturing = false;
    const auto status = QueryCapture(stream, &capturing, &external_capture);
    if (!status.ok()) return status;
    if (attach_graph) {
      if (!capturing || !node.graph_node || !node.prepared || node.captured ||
          view.progress.graph.retiring || invocation.pending || exec.execution.synchronize)
        return Status::InvalidState("capture Scatter with an unused graph lease");
      if (!external_capture || stream != view.progress.graph.stream ||
          (view.progress.graph.id && external_capture != view.progress.graph.id))
        return Status::InvalidState("capture all Scatter leases in the same graph and stream");
      if (view.progress.graph.nodes >= UINT32_MAX - view.progress.sequence)
        return Status::InvalidState("BatchGet sequence exhausted; rebuild Runtime");
    } else if (capturing) {
      return Status::InvalidState("use a Get graph lease for Scatter capture");
    }
  }
  auto result = Drain(view, active, invocation, timing);
  if (!result.ok()) return result;
  auto args = view.args;
  result = PrepareScatterBatchGetArguments(exec, view, args);
  if (!result.ok()) return result;
  if (view.progress.sequence == UINT32_MAX) return Status::InvalidState("BatchGet sequence exhausted; rebuild Runtime");
  GetHostTimingRecord(timing, &GetProviderTiming::prepare_ns, prepare);
  if (attach_graph) {
    args.sequence = 0;
    args.reserved |= kScatterDeviceSequence;
    view.progress.graph.id = external_capture;
    node.captured = true;
    ++view.progress.graph.nodes;
    if (KvafScatterBatchGetLaunch(&args, stream)) {
      view.progress.health.failed = true;
      return Status::CompletionUnknown("Scatter graph launch failed; destroy and discard the graph");
    }
    view.progress.profile_ready = false;
    return Status::OK();
  }
  args.sequence = ++view.progress.sequence;
  view.progress.profile_ready = false;
  invocation.pending = stream;
  // Every launched core overwrites its complete status. Drain checks this
  // request's monotonically increasing sequence, so an old status cannot pass.
  // Avoid submitting a separate device memset task before every kernel.
  const auto launch = GetHostTimingStart(timing);
  const auto launch_result = KvafScatterBatchGetLaunch(&args, invocation.pending);
  if (timing) {
    const auto elapsed = GetHostTimingStart(timing) - launch;
    timing->submit_ns += elapsed;
    timing->aiv_kernel_launch_ns += elapsed;
  }
  if (launch_result) {
    view.progress.health.failed = true; return Error("enqueue failed; resources retained");
  }
  view.progress.profile_ready = ScatterProfileEnabled(args.reserved);
  return exec.execution.synchronize ? Drain(view, active, invocation, timing) : Status::OK();
}

Status ScatterHostOperator::CopyProfiles(std::vector<ScatterBatchGetCoreProfile>* profiles,
    const ScatterLaunchView& view, const ScatterInvocation& active, ScatterInvocation& eager) {
  if (!profiles) return Status::InvalidArgument("scatter profile output is null");
  if (!view.initialized || !ScatterProfileEnabled(view.args.reserved) || !view.progress.profile_ready)
    return Status::InvalidState("enable scatter profiling and complete a BatchGet first");
  const auto valid = view.common.Validate(&active, ExecutionDirection::Get);
  if (!valid.ok()) return valid;
  auto result = Drain(view, active, eager);
  if (!result.ok()) return result;
  profiles->resize(view.cores);
  const auto* source = static_cast<const char*>(view.status_device) + view.StatusBytes();
  const auto copied = aclrtMemcpy(profiles->data(), view.ProfileBytes(), source,
                                 view.ProfileBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
  if (copied != ACL_SUCCESS) { view.progress.health.failed = true; return view.AclFailure("copy_profile", copied); }
  uint64_t requests = 0;
  const auto& first = profiles->front();
  for (uint32_t core = 0; core < profiles->size(); ++core) {
    const auto& p = (*profiles)[core];
    if (p.magic != kScatterProfileMagic || p.sequence != view.progress.sequence ||
        !p.window_count || p.window_count != first.window_count || p.wire_sequence != first.wire_sequence ||
        !p.start_cycle || p.inputs_ready_cycle < p.start_cycle ||
        p.done_cycle < p.inputs_ready_cycle || p.finish_cycle < p.done_cycle ||
        p.pipeline != ScatterOptionEnabled(view.args.reserved, kScatterPipeline) ||
        (p.requests && (!p.scatter_cycles || !p.scatter_chunks ||
            p.scatter_start_cycle < p.inputs_ready_cycle ||
            p.scatter_finish_cycle > p.finish_cycle || p.scatter_finish_cycle <= p.scatter_start_cycle ||
            p.scatter_cycles > p.scatter_finish_cycle - p.scatter_start_cycle)) ||
        p.bytes != p.requests * view.token_bytes ||
        (!core && p.notify_cycle && (p.notify_cycle < p.inputs_ready_cycle || p.notify_cycle > p.done_cycle)))
      return Error("invalid scatter profile; rebuild Host and AIV kernel together");
    requests += p.requests;
  }
  if (requests && (!first.wire_sequence || first.cpu_magic != cpu_push::ProfileToken(first.wire_sequence)))
    return Error("CPU push profile report missing; rebuild and restart the CPU service");
  return Status::OK();
}
}  // namespace kvaf
