#include "cpu_backends.h"
#include "ubmem/hal.h"
#include <sys/socket.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <unordered_set>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); std::abort(); } } while (0)

// Replace hardware URMA only; exercise the actual shared CpuRuntime dispatcher
// and UBmem backend with the same fake HAL used by the mapping tests.
namespace kvaf {
static int urma_initializations = 0;
static bool urma_available = false;
struct UrmaCpuBackend::Impl {
  std::uint64_t* next;
  std::unordered_set<LinkId> sessions;
};
UrmaCpuBackend::UrmaCpuBackend() = default;
UrmaCpuBackend::~UrmaCpuBackend() = default;
Status UrmaCpuBackend::Initialize(const CpuRuntimeOptions&, std::uint64_t& next) {
  ++urma_initializations;
  if (!urma_available) return Status::Internal("test URMA unavailable");
  impl_ = std::make_unique<Impl>(); impl_->next = &next;
  return Status::OK();
}
bool UrmaCpuBackend::initialized() const { return impl_ != nullptr; }
Status UrmaCpuBackend::Shutdown() { impl_.reset(); return Status::OK(); }
Status UrmaCpuBackend::ReleaseLink(LinkId id) {
  return impl_->sessions.erase(id) ? Status::OK() : Status::NotFound("test URMA session");
}
Status UrmaCpuBackend::ServeConnection(int fd, const peer::MessageHeader& header) {
  CHECK(header.command == static_cast<unsigned>(peer::Command::kAllocateRegion));
  peer::AllocateRegionRequest request{};
  CHECK(recv(fd, reinterpret_cast<char*>(&request) + sizeof(header),
      sizeof(request) - sizeof(header), MSG_WAITALL) == sizeof(request) - sizeof(header));
  peer::AllocateRegionResponse response{};
  response.header = peer::MakeHeader(peer::Command::kAllocateRegion, sizeof(response));
  response.header.command = peer::ResponseCommand(peer::Command::kAllocateRegion);
  response.session_id = (*impl_->next)++;
  impl_->sessions.insert(response.session_id);
  CHECK(send(fd, &response, sizeof(response), 0) == sizeof(response));
  return Status::OK();
}
}  // namespace kvaf

template <class Response, class Request>
Response Exchange(kvaf::CpuRuntime& runtime, const Request& request) {
  int sockets[2]; CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
  std::thread server([&] { (void)runtime.ServeConnection(sockets[1]); close(sockets[1]); });
  CHECK(send(sockets[0], &request, sizeof(request), 0) == sizeof(request));
  Response response{};
  CHECK(recv(sockets[0], &response, sizeof(response), MSG_WAITALL) == sizeof(response));
  server.join(); close(sockets[0]);
  CHECK(response.header.bytes == sizeof(response));
  CHECK(response.header.command == (request.header.command | 0x8000U));
  return response;
}
int main(int argc, char** argv) {
  using namespace kvaf;
  using namespace kvaf::peer;
  CHECK(argc == 2);
  CpuRuntimeOptions options; options.hal_library = argv[1];
  CpuRuntime runtime; CHECK(runtime.Initialize(options).ok());
  CHECK(urma_initializations == 0);
  AllocateUbmemRegionRequest u{MakeHeader(Command::kAllocateUbmemRegion, sizeof(u)), 4096, 4096};
  auto first = Exchange<AllocateUbmemRegionResponse>(runtime, u);
  CHECK(first.result == 0 && first.session_id != 0 && urma_initializations == 0);
  CandidateCountRequest candidates{MakeHeader(Command::kGetCandidateCount, sizeof(candidates))};
  CHECK(Exchange<CandidateCountResponse>(runtime, candidates).result != 0);
  CHECK(urma_initializations == 1);
  auto second = Exchange<AllocateUbmemRegionResponse>(runtime, u);
  CHECK(second.result == 0 && second.session_id != first.session_id);
  urma_available = true;
  AllocateRegionRequest r{MakeHeader(Command::kAllocateRegion, sizeof(r)), 4096, 4096, 0, 0};
  auto third = Exchange<AllocateRegionResponse>(runtime, r);
  CHECK(third.result == 0 && third.session_id != first.session_id && third.session_id != second.session_id);
  CHECK(urma_initializations == 2);
  // Mixed releases route by session ownership, including a missing session.
  for (auto id : {second.session_id, third.session_id, first.session_id}) {
    ReleaseRegionRequest release{MakeHeader(Command::kReleaseRegion, sizeof(release)), id};
    CHECK(Exchange<StatusResponse>(runtime, release).result == 0);
    CHECK(Exchange<StatusResponse>(runtime, release).result == static_cast<int>(Result::kNotFound));
  }
  CHECK(urma_initializations == 2);
  CHECK(runtime.Shutdown().ok());
  std::puts("CPU service: lazy backend initialization, mixed allocation sessions, isolated failure and release routing passed");
}
