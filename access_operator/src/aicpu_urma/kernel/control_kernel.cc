#include "transport_internal.h"

#include <unistd.h>

#include <cerrno>
#include <cstdio>

using namespace kvaf::aicpu_urma;
using namespace kvaf::aicpu_urma::detail;
extern "C" __attribute__((visibility("default"))) std::uint32_t KvafAicpuUmdkControl(
    UmdkControlArgs* args) {
  if (!args || !args->status) return 1;
  std::unique_lock<std::shared_mutex> lock(state_mutex);
  auto& status = *reinterpret_cast<UmdkControlStatus*>(args->status);
  status = {};
  status.magic = kUmdkControlMagic;
  status.owner_pid = getpid();
  status.session = state.session;

  errno = 0;
  if (args->op == UmdkControlOp::kPrepare) {
    if (!args->worker_count || args->worker_count > 8 || !args->local_buffer ||
        !args->buffer_bytes || args->local_buffer > UINT64_MAX - args->buffer_bytes ||
        args->library[255])
      Fail(status, "invalid UMDK session arguments");
    else if (state.session != 0) {
      status.session = 0;
      Fail(status, "UMDK session already active");
    } else if (!Prepare(*args, status)) {
      UmdkControlStatus cleanup{};
      if (Cleanup(cleanup)) status.session = 0;
    }
  } else if (args->owner_pid != status.owner_pid || !args->session ||
             args->session != state.session)
    Fail(status, "AICPU process/session changed");
  else
    switch (args->op) {
      case UmdkControlOp::kConnect:
        Connect(*args, status);
        break;
      case UmdkControlOp::kQueryWorker:
        DescribeWorker(*args, args->worker_index, status);
        break;
      case UmdkControlOp::kRegisterBuffer:
        RegisterBuffer(*args, status);
        break;
      case UmdkControlOp::kWrite:
        TransferControl(true, reinterpret_cast<const UmdkTransfer*>(args->transfer_spans),
                        args->transfer_count, status);
        break;
      case UmdkControlOp::kDestroy:
        Cleanup(status);
        break;
      default:
        Fail(status, "unknown UMDK operation");
        break;
    }

  if (!status.result) std::snprintf(status.stage, sizeof(status.stage), "OK");
  return 0;
}
