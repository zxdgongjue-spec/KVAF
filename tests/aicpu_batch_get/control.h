#pragma once

#include "abi.h"
#include "peer_protocol.h"

#include <cerrno>
#include <cstring>
#include <netdb.h>
#include <netinet/in.h>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <unistd.h>

namespace kvaf::aicpu_test {
// The configure opcode rejects executables with an older wire layout.
constexpr auto kConfigure = static_cast<peer::Command>(0x7016);
constexpr auto kStart = static_cast<peer::Command>(0x7001);
constexpr auto kResetSetup = static_cast<peer::Command>(0x7003);
struct ConfigureRequest {
  peer::MessageHeader header;
  std::uint32_t device_id{};
  std::uint32_t npu_candidate{};
  std::uint32_t reserved{};
  Config config;
};
struct Ack { std::uint64_t magic{kMagic}; std::uint64_t status{}; };
struct Report {
  std::uint64_t magic{kMagic};
  std::uint64_t status{1};
  std::uint64_t rounds{};
  std::uint64_t active_threads{};
  double mean_ns{}, p50_ns{}, p99_ns{}, wait_ns{}, scatter_ns{}, gbps{};
  double prepare_ns{}, verify_ns{};
};
static_assert(sizeof(ConfigureRequest) == 72 && sizeof(Ack) == 16 && sizeof(Report) == 96);

// Literal checks must not construct a temporary string on the success path.
inline void Require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
inline void Require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}
inline std::uint32_t SelectDevice(std::uint32_t requested, bool pinned, std::uint32_t device) {
  Require(requested <= INT32_MAX, "invalid NPU device-id");
  Require(!pinned || requested == device,
          "CPU requested NPU device-id=" + std::to_string(requested) +
          " but server is pinned to device-id=" + std::to_string(device));
  return requested;
}
class Socket {
 public:
  explicit Socket(int fd = -1) : fd_(fd) {}
  ~Socket() { if (fd_ >= 0) close(fd_); }
  Socket(const Socket&) = delete;
  Socket& operator=(const Socket&) = delete;
  int get() const { return fd_; }
 private:
  int fd_;
};
inline void SocketTimeout(int fd, std::uint64_t ms) {
  timeval timeout{static_cast<time_t>(ms / 1000), static_cast<suseconds_t>((ms % 1000) * 1000)};
  Require(setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) == 0 &&
          setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) == 0,
          "cannot set control socket timeout");
}
inline void Receive(int fd, void* data, std::size_t bytes) {
  auto* p = static_cast<char*>(data);
  while (bytes) {
    const auto n = recv(fd, p, bytes, 0);
    if (n < 0 && errno == EINTR) continue;
    Require(n > 0, "control receive failed or peer disconnected");
    bytes -= static_cast<std::size_t>(n); p += n;
  }
}
inline void Send(int fd, const void* data, std::size_t bytes) {
  const auto* p = static_cast<const char*>(data);
  while (bytes) {
#ifdef MSG_NOSIGNAL
    const auto n = send(fd, p, bytes, MSG_NOSIGNAL);
#else
    const auto n = send(fd, p, bytes, 0);
#endif
    if (n < 0 && errno == EINTR) continue;
    Require(n > 0, "control send failed");
    bytes -= static_cast<std::size_t>(n); p += n;
  }
}
inline int Connect(const std::string& endpoint) {
  const auto colon = endpoint.rfind(':');
  Require(colon != std::string::npos, "--peer requires HOST:PORT");
  addrinfo hints{}; hints.ai_socktype = SOCK_STREAM;
  addrinfo* addresses = nullptr;
  Require(getaddrinfo(endpoint.substr(0, colon).c_str(), endpoint.substr(colon + 1).c_str(),
                      &hints, &addresses) == 0, "cannot resolve peer");
  int fd = -1;
  for (auto* p = addresses; p; p = p->ai_next) {
    fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    if (fd >= 0 && connect(fd, p->ai_addr, p->ai_addrlen) == 0) break;
    if (fd >= 0) close(fd);
    fd = -1;
  }
  freeaddrinfo(addresses);
  Require(fd >= 0, "cannot connect to " + endpoint);
  return fd;
}
inline int Listen(std::uint16_t port) {
  const int fd = socket(AF_INET, SOCK_STREAM, 0);
  Require(fd >= 0, "cannot create listener");
  int one = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_ANY);
  address.sin_port = htons(port);
  if (bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) || listen(fd, 16)) {
    close(fd); throw std::runtime_error("cannot bind/listen");
  }
  return fd;
}
inline int Accept(int listener) {
  pollfd p{listener, POLLIN, 0};
  Require(poll(&p, 1, 120000) > 0, "timed out waiting for control connection");
  const int fd = accept(listener, nullptr, nullptr);
  Require(fd >= 0, "accept failed");
  return fd;
}
inline std::uint64_t Number(const char* text) {
  Require(text && *text, "missing numeric value");
  std::uint64_t n = 0;
  for (; *text; ++text) {
    Require(*text >= '0' && *text <= '9' && n <= (UINT64_MAX - (*text - '0')) / 10,
            "invalid or overflowing integer");
    n = n * 10 + (*text - '0');
  }
  return n;
}
}  // namespace kvaf::aicpu_test
