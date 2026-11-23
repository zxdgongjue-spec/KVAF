// Host-only TCP regression: idle persistent and long-wait normal clients cannot
// consume the reserved Query/Cancel/Detach budget; fragmented headers are bounded.
#include "runtime_service_loop.h"
#include "remote_pull_protocol.h"
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
namespace rp = kvaf::remote_pull;
namespace {
void Check(bool value, const char* message) {
  if (!value) { std::cerr << message << '\n'; std::abort(); }
}
bool Read(int fd, void* data, std::size_t bytes) {
  auto* p = static_cast<char*>(data);
  while (bytes) { auto n = recv(fd, p, bytes, 0); if (n <= 0) return false; p += n; bytes -= n; }
  return true;
}
int Connect(const sockaddr_in& address) {
  int fd = socket(AF_INET, SOCK_STREAM, 0); Check(fd >= 0, "socket");
  timeval timeout{2, 0}; setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
#ifdef SO_NOSIGPIPE
  int one = 1; setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
  Check(!connect(fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address)), "connect");
  return fd;
}
void Send(int fd, rp::Command command) {
  kvaf::peer::MessageHeader h{rp::kMagic, rp::kVersion, static_cast<std::uint16_t>(command), sizeof(h)};
  Check(send(fd, &h, sizeof(h), 0) == sizeof(h), "send admission header");
}
void Ack(int fd) { char byte{}; Check(Read(fd, &byte, 1) && byte == 'x', "recovery request must progress"); }
}
int main() {
  int listener = socket(AF_INET, SOCK_STREAM, 0); Check(listener >= 0, "listener");
  sockaddr_in address{}; address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  Check(!bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address)), "bind");
  socklen_t size = sizeof(address); Check(!getsockname(listener, reinterpret_cast<sockaddr*>(&address), &size), "address");
  Check(!listen(listener, 16), "listen");
  std::atomic<bool> running{true}, release{false}, entered{false};
  kvaf::detail::ControlServiceLimits limits; limits.normal_connections = 1; limits.recovery_connections = 2;
  limits.pending_headers = 8; limits.header_timeout_ms = 200;
  std::thread server([&] {
    auto status = kvaf::detail::ServeControlConnections(listener, [&](int fd) {
      kvaf::peer::MessageHeader h{};
      if (!Read(fd, &h, sizeof(h))) return kvaf::Status::Internal("closed");
      if (h.command == static_cast<std::uint16_t>(rp::Command::SubmitAndWait)) {
        entered = true;
        while (!release) std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      const char byte = 'x';
      return send(fd, &byte, 1, 0) == 1 ? kvaf::Status::OK() : kvaf::Status::Internal("reply");
    }, [&] { return running.load(); }, limits);
    Check(status.ok(), "service loop shutdown");
  });
  const int normal = Connect(address); Send(normal, rp::Command::Attach); Ack(normal);
  // Idle long-lived connection occupies all normal capacity.
  const int refused = Connect(address); Send(refused, rp::Command::Attach);
  char byte{}; Check(recv(refused, &byte, 1, 0) <= 0, "extra normal connection must be bounded"); close(refused);
  const int recovery = Connect(address); Send(recovery, rp::Command::Query); Ack(recovery);
  // Reserved admission cannot turn into Submit on the next persistent frame.
  Send(recovery, rp::Command::SubmitAndWait);
  Check(recv(recovery, &byte, 1, 0) <= 0, "reserved connection cannot post a READ"); close(recovery);
  Send(normal, rp::Command::SubmitAndWait);
  for (unsigned i = 0; i < 1000 && !entered; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  Check(entered, "normal SubmitAndWait entered");
  // A slow first header must not block accepting a recovery connection.
  const int partial = Connect(address); byte = 1; Check(send(partial, &byte, 1, 0) == 1, "partial header");
  const int query = Connect(address); Send(query, rp::Command::Cancel); Ack(query);
  Check(recv(partial, &byte, 1, 0) <= 0, "partial header admission times out"); close(partial);
  close(query);
  release = true; Ack(normal); close(normal);
  running = false; server.join(); close(listener);
  std::cout << "CPU control recovery capacity checks passed\n";
}
