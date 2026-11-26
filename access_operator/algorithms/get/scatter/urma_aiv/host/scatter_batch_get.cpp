#include "scatter_batch_get.h"
#include "sequential_batch_get_common.h"

#include <cstddef>

namespace kvaf {
Status PrepareScatterBatchGetArguments(const BatchGetExecRequest& exec, ScatterLaunchView view,
                                       KvafScatterBatchGetArgs& args) {
  if (exec.execution.trusted_npu_request) {
    args.reserved |= kScatterTrustedInputs;
    sequential_detail::FillInput(*exec.request, view.cores, view.token_bytes, &args.request);
  } else {
    const auto status = sequential_detail::Input(exec, {view.cores, view.token_bytes}, &args.request);
    if (!status.ok()) return status;
  }
  if (view.jettys_per_core) args.request.jettys_per_core = view.jettys_per_core;
  args.request.core_status = view.status_address;
  args.request.channel_entity = view.mailbox_address;
  args.request.queue_state = view.mailbox_address + offsetof(KvafScatterMailboxContext, queue);
  return Status::OK();
}
}  // namespace kvaf
