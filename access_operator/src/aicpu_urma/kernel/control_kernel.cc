#include "transport_internal.h"

#include <unistd.h>

#include <cerrno>
#include <cstdio>

using namespace kvaf::aicpu_urma;
using namespace kvaf::aicpu_urma::detail;
extern "C" __attribute__((visibility("default"))) std::uint32_t KvafAicpuUmdkControl(
    UmdkControlArgs* args) {
  if (!args || !args->status) return 1;

  auto& status = *reinterpret_cast<UmdkControlStatus*>(args->status);
  status = {};
  status.magic = kUmdkControlMagic;
  status.owner_pid = getpid();
  auto session = args->op == UmdkControlOp::kPrepare ? CreateSession() : FindSession(args->session);
  if (!session) { Fail(status, "UMDK session was not found"); return 0; }
  const auto session_id = args->op == UmdkControlOp::kPrepare ? session->state.session : args->session;
  std::unique_lock<std::shared_mutex> lock(session->mutex);
  SessionScope scope(*session);
  status.session = CurrentState().session;

  errno = 0;
  if (args->op == UmdkControlOp::kPrepare) {
    if (!args->worker_count || args->worker_count > 8 || !args->local_buffer ||
        !args->buffer_bytes || args->local_buffer > UINT64_MAX - args->buffer_bytes ||
        args->library[255]) {
      Fail(status, "invalid UMDK session arguments");
      CurrentState().session = 0;
      status.session = 0;
    } else if (!Prepare(*args, status)) {
      UmdkControlStatus cleanup{};
      if (Cleanup(cleanup)) status.session = 0;
    }
  } else if (args->owner_pid != status.owner_pid || !args->session ||
             args->session != CurrentState().session)
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
      case UmdkControlOp::kUnregisterBuffers:
        UnregisterBuffers(status);
        break;
      case UmdkControlOp::kDestroy:
        Cleanup(status);
        break;
      default:
        Fail(status, "unknown UMDK operation");
        break;
    }

  if (CurrentState().session == 0) RemoveSession(session_id);
  if (!status.result) std::snprintf(status.stage, sizeof(status.stage), "OK");
  return 0;
}

// Explicit entry capability: old packages lack independent sessions and
// completion-scoped buffer unregistration. Host preflight requires this name.
extern "C" __attribute__((visibility("default"))) std::uint32_t KvafAicpuUmdkControlSessionsV2(
    UmdkControlArgs* args) {
  return KvafAicpuUmdkControl(args);
}
