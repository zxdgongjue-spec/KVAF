#include "cache_region_service.h"
#include "peer_link.h"
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#define CHECK(c)                                        \
  do {                                                  \
    if (!(c)) {                                         \
      std::cerr << "line " << __LINE__ << ": " #c "\n"; \
      std::abort();                                     \
    }                                                   \
  } while (false)
using namespace kvaf;
using namespace kvaf::execution;
namespace {
bool Transfer(int fd, void* data, std::size_t bytes, bool write) {
  auto* p = static_cast<char*>(data);
  while (bytes) {
    auto n = write ? send(fd, p, bytes, MSG_NOSIGNAL) : recv(fd, p, bytes, 0);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return false;
    p += n;
    bytes -= n;
  }
  return true;
}
struct Counts {
  std::atomic<int> allocations{0}, opens{0}, calls{0}, closed{0}, freed{0};
};
struct Pool : Allocation {
  explicit Pool(Counts& c) : c(c) {}
  Status Close() override {
    ++c.freed;
    return Status::OK();
  }
  Counts& c;
};
struct Backend : Session {
  explicit Backend(Counts& c) : c(c) {}
  Status Open(const Request&, Allocation& pool, peer::PeerRegion* out) override {
    ++c.opens;
    *out = pool.descriptor;
    out->session_id = 1;
    out->base += 4096;
    return Status::OK();
  }
  Status Call(const std::vector<std::uint8_t>& request, std::vector<std::uint8_t>* out) override {
    ++c.calls;
    peer::MessageHeader header{};
    std::memcpy(&header, request.data(), sizeof(header));
    if (header.command == static_cast<std::uint16_t>(peer::Command::kPrepareGatherWorkspace)) {
      CHECK(request.size() == sizeof(peer::PrepareGatherWorkspaceRequest));
      peer::PrepareGatherWorkspaceRequest prepared{};
      std::memcpy(&prepared, request.data(), sizeof(prepared));
      CHECK(prepared.session_id == 1 && prepared.bytes == 8192);
      peer::AllocateRegionResponse response{};
      response.header = peer::MakeHeader(peer::Command::kPrepareGatherWorkspace, sizeof(response));
      response.header.command = peer::ResponseCommand(peer::Command::kPrepareGatherWorkspace);
      response.session_id = prepared.session_id;
      response.peer_base = 0x30000;
      response.peer_length = prepared.bytes;
      response.token_id = 7;
      response.token_value = 2;
      response.segment_key_bytes = 8;
      response.segment_key[0] = 13;
      const auto* begin = reinterpret_cast<const std::uint8_t*>(&response);
      out->assign(begin, begin + sizeof(response));
      return Status::OK();
    }
    peer::StatusResponse response{};
    response.header =
        peer::MakeHeader(static_cast<peer::Command>(header.command), sizeof(response));
    response.header.command = header.command | 0x8000;
    const auto* begin = reinterpret_cast<const std::uint8_t*>(&response);
    out->assign(begin, begin + sizeof(response));
    return Status::OK();
  }
  Status Close() override {
    ++c.closed;
    return Status::OK();
  }
  Counts& c;
};
struct Server {
  Counts count;
  CacheRegionService service;
  int listener = -1;
  std::string address;
  std::thread worker;
  std::atomic<bool> stop{false}, drop_call{true};
  bool dropped[8]{};
  Server()
      : service(
            kUrma | kFabric,
            [&](const Request& q, std::unique_ptr<Allocation>* out) {
              ++count.allocations;
              auto pool = std::make_unique<Pool>(count);
              pool->descriptor.base = 0x10000;
              pool->descriptor.length = q.logical_bytes + q.auxiliary_bytes;
              *out = std::move(pool);
              return Status::OK();
            },
            [&](const Request&) { return std::make_unique<Backend>(count); }) {
    listener = socket(AF_INET, SOCK_STREAM, 0);
    CHECK(listener >= 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    CHECK(bind(listener, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0);
    socklen_t size = sizeof(addr);
    CHECK(getsockname(listener, reinterpret_cast<sockaddr*>(&addr), &size) == 0);
    CHECK(listen(listener, 8) == 0);
    address = "127.0.0.1:" + std::to_string(ntohs(addr.sin_port));
    worker = std::thread([&] { Run(); });
  }
  ~Server() {
    stop = true;
    shutdown(listener, SHUT_RDWR);
    close(listener);
    worker.join();
  }
  void Run() {
    while (!stop) {
      int fd = accept(listener, nullptr, nullptr);
      if (fd < 0) break;
      Serve(fd);
      close(fd);
    }
  }
  void Serve(int fd) {
    peer::MessageHeader header{};
    if (!Transfer(fd, &header, sizeof(header), false)) return;
    CHECK(header.command == kCommand && header.bytes >= sizeof(header) && header.bytes <= 16384);
    std::vector<std::uint8_t> bytes(header.bytes - sizeof(header));
    if (!Transfer(fd, bytes.data(), bytes.size(), false)) return;
    Request request;
    CHECK(Decode(bytes, &request).ok());
    auto response = service.Handle(request);
    // Drop after mutation: the peer cannot infer whether creation/close happened.
    bool drop = request.op == Op::SessionCall
                    ? drop_call.exchange(false)
                    : (request.op == Op::CreateRegion || request.op == Op::OpenSession ||
                       request.op == Op::CloseSession || request.op == Op::ReleaseRegion) &&
                          !dropped[int(request.op)];
    dropped[int(request.op)] = true;
    if (drop) return;
    bytes = Encode(response);
    header = peer::MakeHeader(static_cast<peer::Command>(kCommand), sizeof(header) + bytes.size());
    header.command = kCommand | 0x8000;
    CHECK(Transfer(fd, &header, sizeof(header), true) &&
          Transfer(fd, bytes.data(), bytes.size(), true));
  }
};
}  // namespace
int main() {
  Server server;
  auto root = std::make_shared<peer::PeerLink>(server.address);
  peer::PeerRegion region;
  CHECK(root->AllocateCacheRegion(4096, 64, 8192, TransportMode::Urma, &region).code() ==
        StatusCode::kCompletionUnknown);
  Response observed;
  CHECK(root->QueryExecution(&observed).ok() && observed.state == State::Ready);
  CHECK(root->AllocateCacheRegion(4096, 64, 8192, TransportMode::Urma, &region).ok());
  CHECK(server.count.allocations == 1);
  Request config;
  config.executor = Executor::Aicpu;
  config.algorithm = GetAlgorithm::Gather;
  auto session = root->ExecutionSession(config);
  CHECK(session->PrepareExecution().code() == StatusCode::kCompletionUnknown);
  CHECK(session->PrepareExecution().ok() && server.count.opens == 1);
  peer::PeerRegion binding;
  CHECK(session->LookupRegion(region.base, &binding).ok());
  CHECK(binding.base != region.base);
  CHECK(!session->EnableGatherLanes(region.base, region.base + 4096, 2).ok());
  CHECK(!session->StartCpuPushGet(region.base).ok() && server.count.calls == 1);
  CHECK(session->EnableGatherLanes(region.base, region.base + 4096, 2).ok() &&
        server.count.calls == 1);
  server.drop_call = true;
  CHECK(!session->StartCpuPushGet(region.base).ok());
  CHECK(session->ResolveExecutionControl().ok() && server.count.calls == 2);
  CHECK(session->StopCpuPushGet(region.base).ok() && server.count.calls == 3);
  peer::PeerRegion workspace, cached;
  server.drop_call = true;
  CHECK(!session->PrepareGatherWorkspace(region.base, 8192, &workspace).ok());
  CHECK(session->PrepareGatherWorkspace(region.base, 8192, &workspace).ok());
  CHECK(server.count.calls == 4 && server.count.allocations == 1);
  CHECK(workspace.base == 0x30000 && workspace.length == 8192 && workspace.session_id == binding.session_id);
  CHECK(session->LookupRegion(workspace.base, &cached).ok() && cached.token_id == 7 &&
        cached.segment_key[0] == 13 && cached.region_id == binding.region_id && cached.epoch == binding.epoch);
  CHECK(root->ReleaseRegion(region.base).code() ==
        StatusCode::kCompletionUnknown);  // dropped Busy response
  CHECK(root->ReleaseRegion(region.base).code() == StatusCode::kBusy);
  CHECK(session->CloseExecution().code() == StatusCode::kCompletionUnknown);
  CHECK(session->CloseExecution().ok() && server.count.closed == 1);
  CHECK(!session->LookupRegion(workspace.base, &cached).ok());
  CHECK(root->ReleaseRegion(region.base).ok() && server.count.freed == 1);
  std::cout << "lost create/control/close replies recovered with original IDs; no duplicate "
               "allocation or worker\n";
}
