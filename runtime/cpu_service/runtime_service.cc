#include "runtime_service.h"
#include "runtime_service_loop.h"
#include "execution_protocol.h"
#include "remote_pull_protocol.h"

#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <fcntl.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace kvaf {
namespace {
using Clock = std::chrono::steady_clock;
int Listen(const RuntimeServiceOptions& options) {
  if (options.address.empty() || options.port == 0U) return -1;
  addrinfo hints{};
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_family = AF_UNSPEC;
  hints.ai_flags = AI_PASSIVE;
  addrinfo* addresses = nullptr;
  const auto port = std::to_string(options.port);
  if (getaddrinfo(options.address.c_str(), port.c_str(), &hints, &addresses)) return -1;
  int listener = -1;
  for (auto* current = addresses; current; current = current->ai_next) {
    listener = socket(current->ai_family, current->ai_socktype, current->ai_protocol);
    if (listener < 0) continue;
    int reuse = 1;
    (void)setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    if (!bind(listener, current->ai_addr, current->ai_addrlen) && !listen(listener, 128)) break;
    close(listener);
    listener = -1;
  }
  freeaddrinfo(addresses);
  return listener;
}
// Classification never consumes data. The owning Runtime remains the only
// parser and journal owner. Partial headers have a bounded admission deadline.
enum class Frame { Incomplete, Invalid, Normal, Recovery };
Frame PeekFrame(int fd) {
  std::array<std::uint8_t, sizeof(peer::MessageHeader) + 16> bytes{};
  const auto n = recv(fd, bytes.data(), bytes.size(), MSG_PEEK | MSG_DONTWAIT);
  if (n < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK))
    return Frame::Incomplete;
  if (n <= 0) return Frame::Invalid;
  if (static_cast<std::size_t>(n) < sizeof(peer::MessageHeader)) return Frame::Incomplete;
  peer::MessageHeader h{};
  std::memcpy(&h, bytes.data(), sizeof(h));
  if (h.bytes < sizeof(h) || (h.command & 0x8000U)) return Frame::Invalid;
  if (h.magic == remote_pull::kMagic) {
    if (h.version != remote_pull::kVersion) return Frame::Invalid;
    using C = remote_pull::Command;
    switch (static_cast<C>(h.command)) {
      case C::Query: case C::Cancel: case C::Detach: case C::Describe:
      case C::InspectSource: return Frame::Recovery;
      default: return Frame::Normal;
    }
  }
  if (h.magic == peer::kProtocolMagic) {
    if (h.version != peer::kProtocolVersion) return Frame::Invalid;
    if (h.command == static_cast<std::uint16_t>(peer::Command::kReleaseRegion))
      return Frame::Recovery;
    if (h.command == execution::kCommand) {
      if (h.bytes < bytes.size() || h.bytes > 16384) return Frame::Invalid;
      if (n < static_cast<decltype(n)>(bytes.size())) return Frame::Incomplete;
      std::uint64_t version = 0, op = 0;
      for (unsigned i = 0; i < 8; ++i) {
        version |= std::uint64_t(bytes[sizeof(h) + i]) << (8 * i);
        op |= std::uint64_t(bytes[sizeof(h) + 8 + i]) << (8 * i);
      }
      if (version != execution::kVersion) return Frame::Invalid;
      if (op == static_cast<std::uint64_t>(execution::Op::Query) ||
          op == static_cast<std::uint64_t>(execution::Op::CloseSession) ||
          op == static_cast<std::uint64_t>(execution::Op::ReleaseRegion))
        return Frame::Recovery;
    }
  }
  // Other Runtime protocols use the normal budget and validate their own frame.
  return Frame::Normal;
}
void ConfigureSocket(int fd, std::uint32_t timeout_ms) {
  int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  timeval timeout{static_cast<time_t>(timeout_ms / 1000),
                  static_cast<suseconds_t>((timeout_ms % 1000) * 1000)};
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
#ifdef SO_NOSIGPIPE
  setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
}
}  // namespace

RuntimeService::RuntimeService(RuntimeServiceOptions options) : options_(std::move(options)) {}

Status detail::ServeControlConnections(int listener, const std::function<Status(int)>& dispatch,
    const std::function<bool()>& keep_running, ControlServiceLimits limits) {
  if (!limits.normal_connections || !limits.recovery_connections || !limits.pending_headers ||
      !limits.header_timeout_ms)
    return Status::InvalidArgument("invalid control service capacity");
  struct Connection {
    int fd;
    bool recovery;
    std::atomic<bool> done{false};
    std::thread worker;
    Connection(int socket, bool reserved) : fd(socket), recovery(reserved) {}
    ~Connection() { if (worker.joinable()) worker.join(); }
  };
  struct Pending { int fd; Clock::time_point expires; };
  struct Connections {
    std::atomic<bool> stopping{false};
    std::vector<std::unique_ptr<Connection>> clients;
    std::vector<Pending> pending;
    ~Connections() {
      stopping = true;
      for (auto p : pending) close(p.fd);
      // Do not close a live fd from another thread or cancel accepted DMA.
      // Socket receive timeouts and polling let handlers leave and join safely.
      clients.clear();
    }
  } connections;
  const int old_flags = fcntl(listener, F_GETFL, 0);
  if (old_flags < 0 || fcntl(listener, F_SETFL, old_flags | O_NONBLOCK))
    return Status::Internal("cannot make control listener nonblocking");
  struct Restore { int fd, flags; ~Restore() { (void)fcntl(fd, F_SETFL, flags); } } restore{listener, old_flags};
  for (;;) {
    if (keep_running && !keep_running()) return Status::OK();
    std::size_t normal = 0, recovery = 0;
    for (auto it = connections.clients.begin(); it != connections.clients.end();) {
      if ((*it)->done) it = connections.clients.erase(it);
      else { ((*it)->recovery ? recovery : normal)++; ++it; }
    }
    // Admission is independent from dispatch workers. An idle persistent socket
    // or a SubmitAndWait can consume only the normal budget. Recovery capacity
    // cannot be converted into a long READ by sending another frame later.
    for (auto it = connections.pending.begin(); it != connections.pending.end();) {
      const auto frame = PeekFrame(it->fd);
      if (frame == Frame::Incomplete && Clock::now() < it->expires) { ++it; continue; }
      const int fd = it->fd;
      it = connections.pending.erase(it);
      const bool reserved = normal >= limits.normal_connections;
      if (frame == Frame::Invalid || frame == Frame::Incomplete ||
          (reserved && (frame != Frame::Recovery || recovery >= limits.recovery_connections))) {
        close(fd);
        continue;
      }
      ConfigureSocket(fd, limits.header_timeout_ms);
      auto client = std::make_unique<Connection>(fd, reserved);
      auto* connection = client.get();
      try {
        connection->worker = std::thread([&, connection] {
          auto partial_since = Clock::time_point{};
          auto idle_since = Clock::now();
          try {
            while (!connections.stopping) {
              pollfd event{connection->fd, POLLIN, 0};
              const int count = poll(&event, 1, 50);
              if (count < 0 && errno == EINTR) continue;
              if (count < 0 || (event.revents & (POLLERR | POLLNVAL))) break;
              if (!count) {
                if (connection->recovery && Clock::now() - idle_since >=
                    std::chrono::milliseconds(limits.header_timeout_ms)) break;
                continue;
              }
              const auto next = PeekFrame(connection->fd);
              if (next == Frame::Incomplete) {
                if (partial_since == Clock::time_point{}) partial_since = Clock::now();
                if (Clock::now() - partial_since >= std::chrono::milliseconds(limits.header_timeout_ms)) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
              }
              if (next == Frame::Invalid || (connection->recovery && next != Frame::Recovery)) break;
              partial_since = {};
              if (!dispatch(connection->fd).ok()) break;
              idle_since = Clock::now();
            }
          } catch (const std::exception& error) {
            std::fprintf(stderr, "[KVAF_CPU] CONTROL_CONNECTION_FAILED detail=%s\n", error.what());
          }
          close(connection->fd);
          connection->done = true;
        });
      } catch (const std::exception&) {
        close(fd);
        return Status::Internal("cannot start control connection handler");
      }
      (reserved ? recovery : normal)++;
      connections.clients.push_back(std::move(client));
    }
    // A short admission tick handles fragmented headers without one peer
    // blocking accept, while avoiding a busy loop on MSG_PEEK partial data.
    pollfd listener_event{listener, POLLIN, 0};
    const int ready = poll(&listener_event, 1, connections.pending.empty() ? 50 : 5);
    if (ready < 0 && errno == EINTR) continue;
    if (ready < 0 || (listener_event.revents & (POLLHUP | POLLERR | POLLNVAL)))
      return Status::Internal("control listener stopped");
    if (!(listener_event.revents & POLLIN)) continue;
    for (std::uint32_t incoming = 0; incoming < limits.pending_headers; ++incoming) {
      const int fd = accept(listener, nullptr, nullptr);
      if (fd < 0) {
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) break;
        return Status::Internal("control accept failed");
      }
      // Accepted sockets are blocking for the existing exact-frame readers.
      const int flags = fcntl(fd, F_GETFL, 0);
      if (flags < 0 || fcntl(fd, F_SETFL, flags & ~O_NONBLOCK) ||
          connections.pending.size() >= limits.pending_headers) { close(fd); continue; }
      connections.pending.push_back({fd, Clock::now() + std::chrono::milliseconds(limits.header_timeout_ms)});
    }
  }
}

Status RuntimeService::Serve(Runtime& runtime) const {
  if (!runtime.initialized()) return Status::NotInitialized("RuntimeService requires an initialized Runtime");
  const int listener = Listen(options_);
  if (listener < 0) return Status::Internal("cannot listen on " + options_.address + ":" +
      std::to_string(options_.port) + ": " + std::strerror(errno));
  std::fprintf(stderr, "[KVAF_CPU] LISTEN address=%s port=%u\n", options_.address.c_str(), options_.port);
  detail::ControlServiceLimits limits;
  limits.normal_connections = options_.normal_connections;
  limits.recovery_connections = options_.recovery_connections;
  limits.pending_headers = options_.pending_headers;
  limits.header_timeout_ms = options_.header_timeout_ms;
  const auto status = detail::ServeControlConnections(listener,
      [&](int fd) { return runtime.ServeConnection(fd); }, {}, limits);
  close(listener);
  return status;
}
}  // namespace kvaf
