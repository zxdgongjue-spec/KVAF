#include "gather_batch_get.h"
#include "sequential_batch_get_common.h"
#include "gather_prefix_layout.h"

#include <algorithm>
#include <cstdio>
#include <sstream>

namespace kvaf {
using namespace sequential_detail;
namespace {
Status PrepareGatherBatchGetArguments(const BatchGetExecRequest& exec, const GatherLaunchView& view,
                                      KvafGatherBatchGetArgs& args) {
  const auto status = exec.execution.trusted_npu_request && !exec.execution.graph_lease
      ? sequential_detail::FillInput(*exec.request, view.cores, exec.request->output_block_bytes, &args.request)
      : sequential_detail::Input(exec, {view.cores, view.token_bytes}, &args.request);
  if (!status.ok()) return status;
  args.head_start = exec.execution.trusted_npu_request && !args.probe;
  args.request.core_status = Ptr(view.status_device);
  args.request.queue_state = Ptr(view.queues);
  return Status::OK();
}

Status QueryCapture(aclrtStream stream, bool* capturing) {
  aclmdlRICaptureStatus capture = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
  aclmdlRI id = nullptr;
  if (aclmdlRICaptureGetInfo(stream, &capture, &id) != ACL_SUCCESS)
    return sequential_detail::Error("query stream capture state");
  *capturing = capture != ACL_MODEL_RI_CAPTURE_STATUS_NONE;
  return Status::OK();
}

Status ValidateLane(const GatherLaunchView& view, const KvafSequentialStatus& lane, uint32_t worker,
                    uint64_t expected, uint64_t phase, uint64_t result) {
  if (lane.magic == KVAF_SEQ_STATUS_MAGIC && lane.sequence == expected &&
      lane.phase == phase && lane.result == result) return Status::OK();
  view.progress.health.failed = true;
  return Status::Internal("AIV sequential BatchGet worker=" + std::to_string(worker) +
      " sequence=" + std::to_string(expected) + " observed_sequence=" +
      std::to_string(lane.sequence) + " phase=" + std::to_string(lane.phase) +
      " result=" + std::to_string(lane.result) + " expected_phase=" +
      std::to_string(phase) + " expected_result=" + std::to_string(result));
}

Status ReadGraphStatus(const GatherLaunchView& view, const GatherInvocation& node, bool require_replay) {
  const auto& graph = node.graph;
  const auto copied = aclrtMemcpyAsync(view.status_host, view.StatusBytes(), view.status_device,
                                      view.StatusBytes(), ACL_MEMCPY_DEVICE_TO_HOST, graph.stream);
  if (copied != ACL_SUCCESS) { view.progress.health.failed = true; return view.AclFailure("copy_graph_status", copied); }
  const auto synchronized = aclrtSynchronizeStream(graph.stream);
  if (synchronized != ACL_SUCCESS) { view.progress.health.failed = true; return view.AclFailure("graph_synchronize", synchronized); }
  const auto* lanes = static_cast<const KvafSequentialStatus*>(view.status_host);
  const auto observed = lanes[0].sequence;
  for (uint32_t i = 0; i < view.cores; ++i) {
    auto status = ValidateLane(view, lanes[i], i, observed, KVAF_SEQ_COMPLETE, KVAF_SEQ_OK);
    if (!status.ok()) return status;
  }
  if (observed < view.progress.sequence || observed > UINT32_MAX) {
    view.progress.health.failed = true;
    return sequential_detail::Error("invalid Gather graph sequence");
  }
  if (require_replay && observed == graph.initial_sequence)
    return Status::InvalidState("Gather graph has not been replayed");
  view.progress.sequence = observed;
  return Status::OK();
}

Status CaptureProfile(const GatherLaunchView& view, GatherInvocation& eager) {
  const auto* profiles = static_cast<const KvafGatherBatchGetCoreProfile*>(view.profile_host);
  const auto* statuses = static_cast<const KvafSequentialStatus*>(view.status_host);
  eager.latest_profile.cores.clear();
  eager.latest_profile.cores.reserve(view.cores);
  for (uint32_t i = 0; i < view.cores; ++i) {
    const auto& p = profiles[i];
    const auto payload_bytes = p.requests * view.token_bytes;
    const auto chunk_bytes = !p.requests ? 0 :
        view.args.cooperative_chunks ? GatherCooperativeChunkBytes(
            payload_bytes, view.args.cooperative_chunks, view.args.first_chunk_divisor) : view.args.pipeline_chunk_bytes;
    const char* reason = nullptr;
    if (p.magic != KVAF_GATHER_PROFILE_MAGIC) reason = "magic";
    else if (p.sequence != view.progress.sequence) reason = "sequence";
    else if (p.result != KVAF_SEQ_OK) reason = "result";
    else if (p.valid_mask != KVAF_GATHER_PROFILE_ALL_EVENTS) reason = "valid_mask";
    else if (p.kernel_start_cycle > p.input_done_cycle) reason = "timestamp_start_input";
    else if (p.input_done_cycle > p.descriptor_done_cycle) reason = "timestamp_input_descriptor";
    else if (!(eager.head_start && !view.args.probe && p.requests > kGatherHeadRows) &&
             p.descriptor_done_cycle > p.notify_done_cycle) reason = "timestamp_descriptor_notify";
    else if (p.notify_done_cycle > p.ack_done_cycle) reason = "timestamp_notify_ack";
    else if (!p.pipeline_chunk_bytes && p.ack_done_cycle > p.read_done_cycle) reason = "timestamp_ack_read";
    else if (p.ack_done_cycle > p.kernel_finish_cycle) reason = "timestamp_ack_finish";
    else if (p.notify_done_cycle > p.read_done_cycle) reason = "timestamp_notify_read";
    else if (p.read_done_cycle > p.kernel_finish_cycle) reason = "timestamp_read_finish";
    else if (!p.window_count || p.requests > statuses[i].requests ||
             (p.window_count <= 1 && p.requests != statuses[i].requests)) reason = "requests";
    else if (p.data_wrs > statuses[i].data_wrs ||
             (p.window_count <= 1 && p.data_wrs != statuses[i].data_wrs)) reason = "data_wrs";
    else if (p.pipeline_chunk_bytes != chunk_bytes) reason = "pipeline_chunk_bytes";
    else if (p.prepare_ahead != (p.requests ? view.args.prepare_ahead : 0)) reason = "prepare_ahead";
    else if (p.cooperative_chunks != (p.requests ? view.args.cooperative_chunks : 0)) reason = "cooperative_chunks";
    else if (p.pipeline_chunk_bytes &&
        (p.first_ready_cycle < p.notify_done_cycle || p.first_ready_cycle > p.first_read_submit_cycle ||
         p.first_read_submit_cycle > p.read_done_cycle || !p.first_read_ready_bytes ||
         p.first_read_ready_bytes > p.requests * view.token_bytes))
      reason = "timestamp_ready_submit_read";
    if (reason) {
      // Preserve the failed record before Shutdown; never return a partial snapshot.
      std::ostringstream detail;
      detail << "AIV sequential BatchGet: invalid gather profile worker=" << i
             << " sequence=" << view.progress.sequence << " reason=" << reason
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
    out.window_count = p.window_count;
    eager.latest_profile.cores.push_back(out);
  }
  return Status::OK();
}
}  // namespace

Status GatherLaunchView::AclFailure(const char* phase, aclError code) const {
  return Status::DeviceError("AIV sequential BatchGet: mode=gather_read sequence=" +
      std::to_string(progress.sequence) + " phase=" + phase + " acl=" + std::to_string(code) +
      " probe=" + sequential_detail::ProbeName(args.probe) + "; resources retained");
}

Status GatherHostOperator::Drain(const GatherLaunchView& view, const GatherInvocation& active, GatherInvocation& eager) {
  if (view.progress.health.failed) return Status::InvalidState("sequential BatchGet failed; shutdown/reinitialize required");
  if (!eager.pending) return Status::OK();
  const auto valid = view.common.Validate(&active, ExecutionDirection::Get);
  if (!valid.ok()) return valid;
  if (!eager.pending_status_check) {
    // Ordinary async calls fence only: same-stream launches can remain queued.
    const auto synchronized = aclrtSynchronizeStream(eager.pending);
    if (synchronized != ACL_SUCCESS) {
      view.progress.health.failed = true; return view.AclFailure("synchronize", synchronized);
    }
    eager.pending = nullptr;
    return Status::OK();
  }
  const auto enqueue_begin = eager.pending_profile ? BatchGetProfileNowNs() : 0;
  const aclError copied = aclrtMemcpyAsync(view.status_host, view.StatusBytes(), view.status_device,
                                          view.StatusBytes(), ACL_MEMCPY_DEVICE_TO_HOST, eager.pending);
  if (copied != ACL_SUCCESS) {
    view.progress.health.failed = true; return view.AclFailure("copy_status", copied);
  }
  if (eager.pending_profile) {
    const auto profile_copied = aclrtMemcpyAsync(view.profile_host, view.ProfileBytes(), view.profile_device,
        view.ProfileBytes(), ACL_MEMCPY_DEVICE_TO_HOST, eager.pending);
    if (profile_copied != ACL_SUCCESS) {
      view.progress.health.failed = true; return view.AclFailure("copy_profile", profile_copied);
    }
  }
  const auto wait_begin = eager.pending_profile ? BatchGetProfileNowNs() : 0;
  if (eager.pending_profile) eager.latest_profile.status_enqueue_ns = wait_begin - enqueue_begin;
  const aclError synchronized = aclrtSynchronizeStream(eager.pending);
  if (synchronized != ACL_SUCCESS) {
    view.progress.health.failed = true; return view.AclFailure("synchronize", synchronized);
  }
  const auto finish_begin = eager.pending_profile ? BatchGetProfileNowNs() : 0;
  if (eager.pending_profile) eager.latest_profile.wait_ns = finish_begin - wait_begin;
  eager.pending = nullptr;
  eager.pending_status_check = false;
  const auto* statuses = static_cast<const KvafSequentialStatus*>(view.status_host);
  const uint64_t expected_result = view.args.probe ? KVAF_SEQ_PROBE_COMPLETE : KVAF_SEQ_OK;
  const uint64_t expected_phase = view.args.probe ? KVAF_GATHER_AFTER_ACK + view.args.probe - 1 :
      uint64_t(KVAF_SEQ_COMPLETE);
  uint64_t completed_requests = 0, data_wrs = 0;
  for (uint32_t i = 0; i < view.cores; ++i) {
    const auto& lane = statuses[i];
    auto status = ValidateLane(view, lane, i, view.progress.sequence, expected_phase, expected_result);
    if (!status.ok()) return status;
    const uint64_t expected_wrs = view.args.probe == KVAF_GATHER_PROBE_FIRST_READ ?
        std::min<uint64_t>(KVAF_GATHER_WRS_PER_POST,
            (lane.requests + view.args.sgl_size - 1) / view.args.sgl_size) : 0;
    if (view.args.probe && lane.data_wrs != expected_wrs) {
      view.progress.health.failed = true;
      return sequential_detail::Error("gather probe READ WR count mismatch");
    }
    completed_requests += lane.requests;
    data_wrs += lane.data_wrs;
  }
  view.outputs.Release(eager.output_buffer);
  if (view.args.probe) {
    if (!completed_requests) { view.progress.health.failed = true; return sequential_detail::Error("gather probe completed no requests"); }
    // A prefix is a diagnostic, never successful BatchGet or a perf sample.
    return Status::NotImplemented("[KVAF_AIV_GATHER_PROBE_COMPLETE] stage=" +
        std::string(sequential_detail::ProbeName(view.args.probe)) + " sequence=" + std::to_string(view.progress.sequence) +
        " cores=" + std::to_string(view.cores) +
        " requests=" + std::to_string(completed_requests) +
        " data_wrs=" + std::to_string(data_wrs));
  }
  if (eager.pending_profile) {
    const auto captured = CaptureProfile(view, eager);
    if (!captured.ok()) { view.progress.health.failed = true; return captured; }
    const auto finished = BatchGetProfileNowNs();
    eager.latest_profile.finish_ns = finished - finish_begin;
    eager.latest_profile.total_ns = finished - eager.profile_begin_ns;
    eager.profile_valid = true;
    eager.pending_profile = false;
  }
  return Status::OK();
}

Status GatherHostOperator::PrepareGraph(const GetGraphOptions& options,
    const GatherLaunchView& view, GatherInvocation& node, GatherInvocation& eager) {
  if (!view.initialized || view.progress.health.failed || !node.graph_node || view.progress.live_graph || eager.pending ||
      view.profiling || view.args.probe || view.progress.sequence == UINT32_MAX)
    return Status::InvalidState("Gather requires one graph lease on an idle provider without profiling or probes");
  const auto valid = view.common.Validate(&node, ExecutionDirection::Get);
  if (!valid.ok()) return valid;
  if (!options.stream || !options.output.valid() || options.output.device != DeviceKind::kNpu ||
      !options.request_capacity)
    return Status::InvalidArgument("invalid Gather graph stream, output or capacity");
  const auto stream = static_cast<aclrtStream>(options.stream);
  bool capturing = false;
  auto status = QueryCapture(stream, &capturing);
  if (!status.ok()) return status;
  if (capturing) return Status::InvalidState("prepare the Gather graph lease before capture");
  auto* lanes = static_cast<KvafSequentialStatus*>(view.status_host);
  for (uint32_t i = 0; i < view.cores; ++i)
    lanes[i] = {KVAF_SEQ_STATUS_MAGIC, view.progress.sequence, KVAF_SEQ_OK, KVAF_SEQ_COMPLETE, 0, 0, 0, 0};
  const auto copied = aclrtMemcpy(view.status_device, view.StatusBytes(), view.status_host,
                                 view.StatusBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
  if (copied != ACL_SUCCESS) return view.AclFailure("initialize_graph_status", copied);
  uint64_t entity = 0;
  if (view.outputs.Acquire(options.output, &node.output_buffer, &entity)) return Error("register graph output");
  node.graph = {stream, view.progress.sequence, entity, false};
  view.progress.live_graph = &node;
  eager.profile_valid = false;
  return Status::OK();
}
Status GatherHostOperator::CheckGraph(void* stream, const GatherLaunchView& view, const GatherInvocation& node) {
  if (view.progress.live_graph != &node || !node.graph_node || !node.graph.captured || node.graph.stream != stream || view.progress.health.failed)
    return Status::InvalidState("check Gather replay using its captured graph lease and stream");
  const auto valid = view.common.Validate(&node, ExecutionDirection::Get);
  if (!valid.ok()) return valid;
  bool capturing = false;
  auto status = QueryCapture(node.graph.stream, &capturing);
  if (!status.ok()) return status;
  if (capturing) return Status::InvalidState("finish Gather graph capture before checking replay");
  return ReadGraphStatus(view, node, true);
}
Status GatherHostOperator::FinishGraph(void* stream, bool discard,
    const GatherLaunchView& view, GatherInvocation& node) {
  if (view.progress.live_graph != &node || !node.graph_node || !node.graph.stream || node.graph.stream != stream)
    return Status::InvalidState("finish Gather using its prepared graph lease and stream");
  if (view.progress.health.failed && !discard) return Status::InvalidState("destroy and discard the failed Gather graph");
  const auto valid = view.common.Validate(&node, ExecutionDirection::Get);
  if (!valid.ok()) return valid;
  bool capturing = false;
  auto status = QueryCapture(node.graph.stream, &capturing);
  if (!status.ok()) return status;
  if (capturing) return Status::InvalidState("finish Gather graph capture before releasing its lease");
  if (node.graph.captured && !view.progress.health.failed) {
    status = ReadGraphStatus(view, node, false);
    if (!status.ok() && !discard) return status;
  }
  if (view.progress.health.failed) {
    const auto synchronized = aclrtSynchronizeStream(node.graph.stream);
    if (synchronized != ACL_SUCCESS) return view.AclFailure("discard_graph_synchronize", synchronized);
    node.recovery_fenced = true;
    // Failed DMA may still reference the registration after this node is unbound.
    view.outputs.Preserve(node.output_buffer);
  } else {
    view.outputs.Release(node.output_buffer);
  }
  node.graph = {};
  view.progress.live_graph = nullptr;
  return Status::OK();
}

Status GatherHostOperator::CopyProfile(GatherBatchGetProfile* profile, const GatherLaunchView& view,
    const GatherInvocation& active, const GatherInvocation& eager) {
  if (!profile) return Status::InvalidArgument("gather profile output must not be null");
  *profile = {};
  if (!view.initialized) return Status::NotInitialized("gather transport is not initialized");
  if (view.progress.health.failed || !eager.profile_valid)
    return Status::InvalidState("no successful synchronous gather profile is available");
  const auto valid = view.common.Validate(&active, ExecutionDirection::Get);
  if (!valid.ok()) return valid;
  *profile = eager.latest_profile;
  return Status::OK();
}
Status GatherHostOperator::Execute(const BatchGetExecRequest& exec, const GatherLaunchView& view,
    GatherInvocation& active, GatherInvocation& invocation) {
  if (!view.initialized) return Status::NotInitialized("sequential BatchGet is not initialized");
  if (view.progress.health.failed)
    return Status::InvalidState("sequential BatchGet failed; shutdown/reinitialize required");
  const auto valid = view.common.Validate(&active, ExecutionDirection::Get);
  if (!valid.ok()) return valid;
  if (!exec.request || !exec.execution.stream) return Status::InvalidArgument("BatchGet needs request and stream");
  const bool attach_graph = exec.execution.graph_lease != 0;
  const auto stream = static_cast<aclrtStream>(exec.execution.stream);
  if (view.progress.live_graph && !attach_graph)
    return Status::InvalidState("release the Gather graph lease before eager BatchGet");
  if (attach_graph || !exec.execution.trusted_npu_request) {
    bool capturing = false;
    auto status = QueryCapture(stream, &capturing);
    if (!status.ok()) return status;
    if (attach_graph) {
      auto& node = active;
      if (!capturing || !node.graph_node || view.progress.live_graph != &node ||
          node.graph.stream != stream || node.graph.captured || exec.execution.synchronize)
        return Status::InvalidState("capture Gather with its unused graph lease and stream");
      auto args = view.args;
      status = PrepareGatherBatchGetArguments(exec, view, args);
      if (!status.ok()) return status;
      args.request.channel_entity = node.graph.output_entity;
      args.sequence = 0;
      node.graph.captured = true;
      if (KvafGatherBatchGetLaunch(&args, stream)) {
        view.progress.health.failed = true;
        return Status::CompletionUnknown("Gather graph launch failed; destroy and discard the graph");
      }
      return Status::OK();
    }
    if (capturing) return Status::InvalidState("use a Get graph lease for Gather capture");
  }
  if (view.profiling && !exec.execution.synchronize)
    return Status::InvalidArgument("gather profiling requires synchronize=True");
  if (view.args.probe && !exec.execution.synchronize)
    return Status::InvalidArgument("gather prefix probes require synchronize=True");
  // Stream order protects shared workspace/queues; fence on stream switch or checked calls.
  Status result = Status::OK();
  if (invocation.pending && (invocation.pending != exec.execution.stream || invocation.pending_status_check)) {
    result = Drain(view, active, invocation);
    if (!result.ok()) return result;
  }
  invocation.profile_valid = false;
  if (view.profiling) {
    invocation.profile_begin_ns = BatchGetProfileNowNs();
    invocation.latest_profile = {};
  }
  auto args = view.args;
  result = PrepareGatherBatchGetArguments(exec, view, args);
  if (!result.ok()) return result;
  if (view.progress.sequence == UINT32_MAX) return Status::InvalidState("BatchGet sequence exhausted; rebuild Runtime");
  if (view.outputs.Acquire(exec.request->output, &invocation.output_buffer, &args.request.channel_entity))
    return Error("register output");
  invocation.head_start = args.head_start;
  args.sequence = ++view.progress.sequence;
  args.profile = view.profiling ? Ptr(view.profile_device) : 0;
  invocation.pending_profile = view.profiling;
  invocation.pending_status_check = exec.execution.synchronize || view.profiling ||
      args.probe || view.check_async_status;
  const auto launch_begin = view.profiling ? BatchGetProfileNowNs() : 0;
  if (view.profiling) {
    invocation.latest_profile.sequence = view.progress.sequence;
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
    view.progress.health.failed = true; return Error("enqueue failed; resources retained");
  }
  if (view.profiling) invocation.latest_profile.launch_ns = BatchGetProfileNowNs() - launch_begin;
  if (!invocation.pending_status_check) {
    // MR/entity cache outlives these references, until transport shutdown.
    view.outputs.Release(invocation.output_buffer);
  }
  return exec.execution.synchronize ? Drain(view, active, invocation) : Status::OK();
}
}  // namespace kvaf
