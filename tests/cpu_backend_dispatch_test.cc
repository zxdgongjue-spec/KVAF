#include "cpu_runtime.h"
#include "peer_protocol.h"

#include <sys/socket.h>
#include <unistd.h>
#include <cstdlib>
#include <iostream>
#include <type_traits>

#define CHECK(condition) do { if (!(condition)) { \
  std::cerr << "line " << __LINE__ << ": " #condition "\n"; std::exit(1); \
} } while (false)

// Exercise the real dispatcher and UBmem backend without installed HAL/URMA.
// In particular, unavailable backends must preserve each response's wire size.
template <class Request, class Response>
void Exchange(kvaf::CpuRuntime& runtime, kvaf::peer::Command command,
              kvaf::peer::Result expected) {
  int sockets[2];
  CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
  Request request{};
  request.header = kvaf::peer::MakeHeader(command, sizeof(request));
  if constexpr (std::is_same_v<Request, kvaf::peer::AllocateUbmemRegionRequest>) {
    request.ddr_bytes = 4096;
    request.slot_size = 64;
  }
  CHECK(send(sockets[0], &request, sizeof(request), 0) == sizeof(request));
  const auto served = runtime.ServeConnection(sockets[1]);
  CHECK(served.ok() != (std::is_same_v<Request, kvaf::peer::AllocateUbmemRegionRequest>));
  Response response{};
  CHECK(recv(sockets[0], &response, sizeof(response), MSG_WAITALL) == sizeof(response));
  CHECK(response.header.command == kvaf::peer::ResponseCommand(command));
  CHECK(response.header.bytes == sizeof(response));
  CHECK(response.result == static_cast<int>(expected));
  close(sockets[0]);
  close(sockets[1]);
}

int main() {
  using namespace kvaf::peer;
  kvaf::CpuRuntime runtime;
  kvaf::CpuRuntimeOptions options;
  options.hal_library = "/nonexistent/kvaf-merge-test-hal.so";
  CHECK(runtime.Initialize(options).ok());  // No backend is eagerly loaded.
  Exchange<AllocateRegionRequest, AllocateRegionResponse>(
      runtime, Command::kAllocateRegion, Result::kResourceError);
  Exchange<CandidateCountRequest, CandidateCountResponse>(
      runtime, Command::kGetCandidateCount, Result::kResourceError);
  Exchange<CandidateInfoRequest, CandidateInfoResponse>(
      runtime, Command::kGetCandidateInfo, Result::kResourceError);
  Exchange<CreateChannelRequest, CreateChannelResponse>(
      runtime, Command::kCreateChannel, Result::kResourceError);
  Exchange<OpenCpuPushGetRequest, OpenCpuPushGetResponse>(
      runtime, Command::kOpenCpuPushGet, Result::kResourceError);
  Exchange<EnableDirectGatherRequest, StatusResponse>(
      runtime, Command::kEnableDirectGather, Result::kResourceError);
  Exchange<EnableGatherLanesRequest, StatusResponse>(
      runtime, Command::kEnableGatherLanes, Result::kResourceError);
  Exchange<CpuPushGetSessionRequest, StatusResponse>(
      runtime, Command::kStartCpuPushGet, Result::kResourceError);
  Exchange<CpuPushGetSessionRequest, StatusResponse>(
      runtime, Command::kStopCpuPushGet, Result::kResourceError);
  Exchange<ReleaseRegionRequest, StatusResponse>(
      runtime, Command::kReleaseRegion, Result::kNotFound);
  Exchange<AllocateUbmemRegionRequest, AllocateUbmemRegionResponse>(
      runtime, Command::kAllocateUbmemRegion, Result::kResourceError);
  CHECK(runtime.Shutdown().ok());
  CHECK(!runtime.initialized());
  std::cout << "CPU backend dispatch and response layout tests passed\n";
}
