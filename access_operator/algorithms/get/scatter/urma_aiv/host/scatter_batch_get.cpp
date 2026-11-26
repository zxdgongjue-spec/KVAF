#include "scatter_execution.h"

namespace kvaf {
using namespace sequential_detail;
namespace {
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
}  // namespace

Status ScatterBatchGet::Launch(const BatchGetExecRequest& exec, LaunchView view) {
  auto* timing = exec.host_timing;
  BatchGetProfileScope total(timing ? &timing->total_ns : nullptr);
  auto& s = view.resources;
  auto& invocation = view.invocation;
  std::lock_guard<std::mutex> lock(s.mutex);
  auto state = s.CheckExecutionState();
  if (!state.ok()) return state;
  const bool attach_graph = exec.execution.graph_lease != 0;
  if (s.graph.leases && !attach_graph)
    return Status::InvalidState("release Scatter graph leases before another eager BatchGet");
  if (!exec.request || !exec.execution.stream)
    return Status::InvalidArgument("BatchGet needs a request and stream");
  const auto prepare = GetHostTimingStart(timing);
  const auto stream = static_cast<aclrtStream>(exec.execution.stream);
  aclmdlRI external_capture = nullptr;
  if (attach_graph || !exec.execution.trusted_npu_request) {
    bool capturing = false;
    const auto status = s.QueryCapture(stream, &capturing, &external_capture);
    if (!status.ok()) return status;
    if (attach_graph) {
      if (!capturing || !view.node || !view.node->prepared || view.node->captured ||
          s.graph.retiring || invocation.pending || exec.execution.synchronize)
        return Status::InvalidState("capture Scatter with an unused graph lease");
      if (!external_capture || stream != s.graph.stream ||
          (s.graph.id && external_capture != s.graph.id))
        return Status::InvalidState("capture all Scatter leases in the same graph and stream");
      if (s.graph.nodes >= UINT32_MAX - s.sequence)
        return Status::InvalidState("BatchGet sequence exhausted; rebuild Runtime");
    } else if (capturing) {
      return Status::InvalidState("use a Get graph lease for Scatter capture");
    }
  }
  auto result = s.Drain(timing);
  if (!result.ok()) return result;
  auto args = s.args;
  if (exec.execution.trusted_npu_request) args.reserved |= kScatterTrustedInputs;
  if (exec.execution.trusted_npu_request) FillTrustedScatterInput(*exec.request, s.config, &args.request);
  else result = Input(exec, s.config, &args.request);
  if (!result.ok()) return result;
  if (s.payload_jettys_per_core) args.request.jettys_per_core = s.payload_jettys_per_core;
  if (s.sequence == UINT32_MAX) return Status::InvalidState("BatchGet sequence exhausted; rebuild Runtime");
  args.request.core_status = Ptr(s.status_device);
  args.request.channel_entity = s.mailbox_context;
  args.request.queue_state = s.mailbox_context + offsetof(KvafScatterMailboxContext, queue);
  GetHostTimingRecord(timing, &GetProviderTiming::prepare_ns, prepare);
  if (attach_graph) {
    args.sequence = 0;
    args.reserved |= kScatterDeviceSequence;
    s.graph.id = external_capture;
    view.node->captured = true;
    ++s.graph.nodes;
    if (KvafScatterBatchGetLaunch(&args, stream)) {
      s.failed = true;
      return Status::CompletionUnknown("Scatter graph launch failed; destroy and discard the graph");
    }
    s.profile_ready = false;
    return Status::OK();
  }
  args.sequence = ++s.sequence;
  s.profile_ready = false;
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
    s.failed = true; return Error("enqueue failed; resources retained");
  }
  s.profile_ready = ScatterProfileEnabled(args.reserved);
  return exec.execution.synchronize ? s.Drain(timing) : Status::OK();
}
}  // namespace kvaf
