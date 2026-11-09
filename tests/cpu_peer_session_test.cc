// Exercise the production wire dispatcher with an empty in-process service.
// Including its implementation exposes private handlers without production test
// hooks. No UB device, source allocation or background worker is required.
#include "../runtime/src/cpu_runtime.cc"

#include <cassert>
#include <iostream>

namespace {
using namespace kvaf::peer;

static_assert(ProtocolVersion(Command::kAllocateRegion) == 6U);
static_assert(ProtocolVersion(Command::kEnableGatherLanes) == 6U);
static_assert(ProtocolVersion(Command::kStopGatherLanes) == 6U);
static_assert(ProtocolVersion(Command::kOpenCpuPushGet) == 8U);
static_assert(ProtocolVersion(Command::kStartCpuPushGet) == 8U);
static_assert(ProtocolVersion(Command::kStopCpuPushGet) == 8U);

Result Exchange(kvaf::CpuService& service, Command command,
                bool wrong_size = false, bool truncated = false) {
  int sockets[2];
  assert(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
  CpuPushGetSessionRequest request{};
  request.header = MakeHeader(command, sizeof(request));
  request.session_id = UINT64_MAX;  // Deliberately absent; never starts real I/O.
  if (wrong_size) ++request.header.bytes;
  const auto bytes = truncated ? sizeof(MessageHeader) + 4 : sizeof(request);
  assert(kvaf::WriteAll(sockets[0], &request, bytes));
  assert(shutdown(sockets[0], SHUT_WR) == 0);

  kvaf::HandleClient(sockets[1], &service);
  close(sockets[1]);
  StatusResponse response{};
  assert(kvaf::ReadAll(sockets[0], &response, sizeof(response)));
  close(sockets[0]);
  assert(IsValidResponseHeader(response.header, command, sizeof(response)));
  return static_cast<Result>(response.result);
}
}  // namespace

int main() {
  kvaf::CpuService service(nullptr, 1, 65536, false, kvaf::CpuGatherMode::kAdaptive);
  // Valid START reaches the session lookup. Missing dispatch used to return
  // InvalidRequest instead, including for fully initialized production sessions.
  assert(Exchange(service, Command::kStartCpuPushGet) == Result::kNotFound);
  // STOP is idempotent even after a missing/already released session.
  assert(Exchange(service, Command::kStopCpuPushGet) == Result::kOk);
  assert(Exchange(service, Command::kStopGatherLanes) == Result::kOk);
  for (const auto command : {Command::kStartCpuPushGet, Command::kStopCpuPushGet,
                             Command::kStopGatherLanes}) {
    assert(Exchange(service, command, true) == Result::kInvalidRequest);
    assert(Exchange(service, command, false, true) == Result::kInvalidRequest);
  }
  assert(Exchange(service, static_cast<Command>(0x1234)) == Result::kInvalidRequest);
  std::cout << "PASS: CPU peer session command routing and framing\n";
}
