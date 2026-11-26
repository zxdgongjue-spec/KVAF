#include "gather_execution.h"

#include <cstdio>

namespace kvaf {
using namespace sequential_detail;
Status GatherBatchGet::Launch(const BatchGetExecRequest& exec, LaunchView view) {
  auto& s = view.resources;
  auto& invocation = view.invocation;
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!s.initialized) return Status::NotInitialized("sequential BatchGet is not initialized");
  if (s.failed) return Status::InvalidState("sequential BatchGet failed; shutdown/reinitialize required");
  if (!exec.request || !exec.execution.stream) return Status::InvalidArgument("BatchGet needs request and stream");
  const bool attach_graph = exec.execution.graph_lease != 0;
  const auto stream = static_cast<aclrtStream>(exec.execution.stream);
  if (s.live_graph && !attach_graph)
    return Status::InvalidState("release the Gather graph lease before eager BatchGet");
  if (attach_graph || !exec.execution.trusted_npu_request) {
    bool capturing = false;
    auto status = s.QueryCapture(stream, &capturing);
    if (!status.ok()) return status;
    if (attach_graph) {
      if (!capturing || !view.node || s.live_graph != view.node ||
          view.node->graph.stream != stream || view.node->graph.captured || exec.execution.synchronize)
        return Status::InvalidState("capture Gather with its unused graph lease and stream");
      auto args = s.args;
      status = Input(exec, s.config, &args.request);
      if (!status.ok()) return status;
      args.head_start = exec.execution.trusted_npu_request && !args.grouped_addresses;
      args.request.core_status = Ptr(s.status_device);
      args.request.queue_state = Ptr(s.queues);
      args.request.channel_entity = view.node->graph.output_entity;
      args.sequence = 0;
      view.node->graph.captured = true;
      if (KvafGatherBatchGetLaunch(&args, stream)) {
        s.failed = true;
        return Status::CompletionUnknown("Gather graph launch failed; destroy and discard the graph");
      }
      return Status::OK();
    }
    if (capturing) return Status::InvalidState("use a Get graph lease for Gather capture");
  }
  if (s.profiling && !exec.execution.synchronize)
    return Status::InvalidArgument("gather profiling requires synchronize=True");
  if (s.args.probe && !exec.execution.synchronize)
    return Status::InvalidArgument("gather prefix probes require synchronize=True");
  // Stream order protects shared workspace/queues; fence on stream switch or checked calls.
  Status result = Status::OK();
  if (invocation.pending && (invocation.pending != exec.execution.stream || invocation.pending_status_check)) {
    result = s.Drain();
    if (!result.ok()) return result;
  }
  invocation.profile_valid = false;
  if (s.profiling) {
    invocation.profile_begin_ns = BatchGetProfileNowNs();
    invocation.latest_profile = {};
  }
  auto args = s.args;
  args.head_start = exec.execution.trusted_npu_request && !args.probe && !args.grouped_addresses;
  if (exec.execution.trusted_npu_request) FillInput(*exec.request, s.config, &args.request);
  else result = Input(exec, s.config, &args.request);
  if (!result.ok()) return result;
  if (s.sequence == UINT32_MAX) return Status::InvalidState("BatchGet sequence exhausted; rebuild Runtime");
  if (KvafBatchGetLocalBufferAcquire(s.gather_transport, exec.request->output.data,
      exec.request->output.bytes, &invocation.output_buffer, &args.request.channel_entity))
    return Error("register output");
  args.request.core_status = Ptr(s.status_device); args.request.queue_state = Ptr(s.queues);
  invocation.head_start = args.head_start;
  args.sequence = ++s.sequence;
  args.profile = s.profiling ? Ptr(s.profile_device) : 0;
  invocation.pending_profile = s.profiling;
  invocation.pending_status_check = exec.execution.synchronize || s.profiling ||
      args.probe || s.check_async_status;
  const auto launch_begin = s.profiling ? BatchGetProfileNowNs() : 0;
  if (s.profiling) {
    invocation.latest_profile.sequence = s.sequence;
    invocation.latest_profile.prepare_ns = launch_begin - invocation.profile_begin_ns;
  }
  invocation.pending = static_cast<aclrtStream>(exec.execution.stream);
  if (args.probe)
    std::fprintf(stderr, "[KVAF_AIV_GATHER_PROBE_BEGIN] stage=%s sequence=%llu cores=%u "
        "token_bytes=%u request_capacity=%u\n", ProbeName(args.probe),
        static_cast<unsigned long long>(args.sequence), args.request.block_count,
        args.request.token_bytes, args.request.request_capacity);
  // Each core initializes status/profile; avoid redundant Host clears.
  if (KvafGatherBatchGetLaunch(&args, invocation.pending)) {
    s.failed = true; return Error("enqueue failed; resources retained");
  }
  if (s.profiling) invocation.latest_profile.launch_ns = BatchGetProfileNowNs() - launch_begin;
  if (!invocation.pending_status_check) {
    // MR/entity cache outlives these references, until transport shutdown.
    KvafBatchGetLocalBufferRelease(invocation.output_buffer);
    invocation.output_buffer = nullptr;
  }
  return exec.execution.synchronize ? s.Drain() : Status::OK();
}
}  // namespace kvaf
