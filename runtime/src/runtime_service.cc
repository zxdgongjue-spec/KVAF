#include "runtime_service.h"
#include "runtime_service_loop.h"

#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>
#include <atomic>
#include <memory>
#include <thread>

namespace kvaf {
namespace {

int Listen(const RuntimeServiceOptions& options) {
  if (options.address.empty() || options.port == 0U) return -1;
  addrinfo hints{};
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_family = AF_UNSPEC;
  hints.ai_flags = AI_PASSIVE;
  addrinfo* addresses = nullptr;
  const std::string port = std::to_string(options.port);
  if (getaddrinfo(options.address.c_str(), port.c_str(), &hints, &addresses) != 0) return -1;
  int listener = -1;
  for (addrinfo* current = addresses; current != nullptr; current = current->ai_next) {
    listener = socket(current->ai_family, current->ai_socktype, current->ai_protocol);
    if (listener < 0) continue;
    int reuse = 1;
    (void)setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    if (bind(listener, current->ai_addr, current->ai_addrlen) == 0 && listen(listener, 32) == 0) break;
    close(listener);
    listener = -1;
  }
  freeaddrinfo(addresses);
  return listener;
}

}  // namespace

RuntimeService::RuntimeService(RuntimeServiceOptions options) : options_(std::move(options)) {}

Status detail::ServeControlConnections(int listener, const std::function<Status(int)>& dispatch,
                                       const std::function<bool()>& keep_running) {
  struct Connection {
    int fd;
    std::atomic<bool> done{false};
    std::thread worker;
    explicit Connection(int socket) : fd(socket) {}
    ~Connection() { if (worker.joinable()) worker.join(); }
  };
  struct Connections {
    std::atomic<bool> stopping{false};
    std::vector<std::unique_ptr<Connection>> clients;
    ~Connections() { stopping = true; clients.clear(); }
  } connections;
  for (;;) {
    pollfd listener_event{listener, POLLIN, 0};
    const int ready = poll(&listener_event, 1, keep_running ? 100 : -1);
    if (keep_running && !keep_running()) return Status::OK();
    if (ready < 0 && errno == EINTR) continue;
    if (ready < 0 || (listener_event.revents & (POLLHUP | POLLERR | POLLNVAL)))
      return Status::Internal("control listener stopped");
    if (!(listener_event.revents & POLLIN)) continue;
    const int fd = accept(listener, nullptr, nullptr);
    if (fd < 0) {
      if (errno == EINTR) continue;
      return Status::Internal("accept failed: " + std::string(std::strerror(errno)));
    }
    auto& clients = connections.clients;
    for (auto it = clients.begin(); it != clients.end();)
      if ((*it)->done) it = clients.erase(it); else ++it;
    // Bound live connection handlers. Closing TCP never tears down a Runtime
    // session or cancels a READ that has already been accepted.
    if (clients.size() >= 256) { close(fd); continue; }
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    timeval timeout{5, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
#ifdef SO_NOSIGPIPE
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
    auto client = std::make_unique<Connection>(fd);
    auto* connection = client.get();
    try {
      connection->worker = std::thread([&, connection] {
        const int socket = connection->fd;
        try {
          while (!connections.stopping) {
            pollfd event{socket, POLLIN, 0};
            const int count = poll(&event, 1, 100);
            if (count < 0 && errno == EINTR) continue;
            if (count < 0 || (event.revents & (POLLERR | POLLNVAL))) break;
            if (!count) continue;
            char byte;
            const auto peeked = recv(socket, &byte, 1, MSG_PEEK | MSG_DONTWAIT);
            if (peeked < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
            if (peeked <= 0 || !dispatch(socket).ok()) break;
          }
        } catch (const std::exception& error) {
          std::fprintf(stderr, "[KVAF_CPU] CONTROL_CONNECTION_FAILED detail=%s\n", error.what());
        }
        close(socket);
        connection->done = true;
      });
    } catch (const std::exception&) {
      close(fd);
      return Status::Internal("cannot start control connection handler");
    }
    clients.push_back(std::move(client));
  }
}

Status RuntimeService::Serve(Runtime& runtime) const {
  if (!runtime.initialized()) return Status::NotInitialized("RuntimeService requires an initialized Runtime");
  const int listener = Listen(options_);
  if (listener < 0) {
    return Status::Internal("cannot listen on " + options_.address + ":" +
                            std::to_string(options_.port) + ": " + std::strerror(errno));
  }
  std::fprintf(stderr, "[KVAF_CPU] LISTEN address=%s port=%u\n", options_.address.c_str(), options_.port);
  const auto status = detail::ServeControlConnections(listener,
      [&](int connection) { return runtime.ServeConnection(connection); });
  close(listener);
  return status;
}

}  // namespace kvaf
