#include "offload_client.h"
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

namespace kvaf::offload {
void SetSocketTimeout(int fd) {
  timeval timeout{5, 0};
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
#ifdef SO_NOSIGPIPE
  int one = 1;
  setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
}
bool ReadBytes(int fd, void* data, std::size_t bytes) {
  auto* p = static_cast<char*>(data);
  while (bytes) {
    const auto n = recv(fd, p, bytes, 0);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return false;
    p += n; bytes -= n;
  }
  return true;
}
bool WriteBytes(int fd, const void* data, std::size_t bytes) {
  const auto* p = static_cast<const char*>(data);
  while (bytes) {
#ifdef MSG_NOSIGNAL
    const auto n = send(fd, p, bytes, MSG_NOSIGNAL);
#else
    const auto n = send(fd, p, bytes, 0);
#endif
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return false;
    p += n; bytes -= n;
  }
  return true;
}
Status Exchange(const std::string& address, Request request,
                const std::vector<Span>& spans, Response* response) {
  if (!response || spans.size() > kMaxSpans) return Status::InvalidArgument("invalid Pull exchange");
  const auto colon = address.rfind(':');
  if (colon == std::string::npos || colon == 0 || colon + 1 == address.size())
    return Status::InvalidArgument("Offload requires host:port");
  addrinfo hints{}, *list = nullptr;
  hints.ai_socktype = SOCK_STREAM;
  if (getaddrinfo(address.substr(0, colon).c_str(), address.substr(colon+1).c_str(), &hints, &list))
    return Status::InDoubt("cannot resolve CPU service");
  int fd = -1;
  for (auto* a = list; a; a = a->ai_next) {
    fd = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
    if (fd < 0) continue;
    const int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK)) { close(fd); fd = -1; continue; }
    int rc = connect(fd, a->ai_addr, a->ai_addrlen);
    if (rc < 0 && errno == EINPROGRESS) {
      pollfd p{fd, POLLOUT, 0};
      int error = 0; socklen_t size = sizeof(error);
      rc = poll(&p, 1, 5000) > 0 && !getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &size) && !error ? 0 : -1;
    }
    if (!rc && !fcntl(fd, F_SETFL, flags)) break;
    close(fd); fd = -1;
  }
  freeaddrinfo(list);
  if (fd < 0) return Status::InDoubt("cannot connect to CPU service");
  SetSocketTimeout(fd);
  request.count = static_cast<std::uint32_t>(spans.size());
  request.header.bytes = sizeof(Request) + request.count * sizeof(Span);
  const bool ok = WriteBytes(fd, &request, sizeof(request)) &&
      WriteBytes(fd, spans.data(), spans.size() * sizeof(Span)) && ReadBytes(fd, response, sizeof(*response));
  close(fd);
  if (!ok) return Status::InDoubt("CPU control response unavailable; retain source and task");
  const auto& h = response->header;
  if (h.magic != kMagic || h.version != kVersion || h.bytes != sizeof(Response) ||
      h.command != (request.header.command | 0x8000U) ||
      response->region != request.region || response->task != request.task ||
      (request.epoch && response->epoch != request.epoch))
    return Status::InDoubt("CPU response identity/version mismatch");
  return Status::OK();
}
}
