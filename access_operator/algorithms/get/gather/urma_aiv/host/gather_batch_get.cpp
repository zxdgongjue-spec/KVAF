#include "gather_batch_get.h"
#include "sequential_batch_get_common.h"

namespace kvaf {
Status PrepareGatherBatchGetArguments(const BatchGetExecRequest& exec, GatherLaunchView view,
                                      KvafGatherBatchGetArgs& args) {
  const auto status = exec.execution.trusted_npu_request && !exec.execution.graph_lease
      ? sequential_detail::FillInput(*exec.request, view.cores, exec.request->output_block_bytes, &args.request)
      : sequential_detail::Input(exec, {view.cores, view.token_bytes}, &args.request);
  if (!status.ok()) return status;
  args.head_start = exec.execution.trusted_npu_request && !args.probe;
  args.request.core_status = view.status_address;
  args.request.queue_state = view.queue_address;
  return Status::OK();
}
}  // namespace kvaf
