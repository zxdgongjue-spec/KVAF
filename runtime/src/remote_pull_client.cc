#include "offload_client.h"
#include "remote_pull_service.h"
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

namespace kvaf::remote_pull {
using offload::ReadBytes;
using offload::WriteBytes;
using offload::SetSocketTimeout;
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
  if (request.header.command == static_cast<std::uint16_t>(Command::Probe)) {
    // The synchronous reference probe may wait for a 30-second READ CQ.
    timeval timeout{35, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  }
  request.count = static_cast<std::uint32_t>(spans.size());
  request.header.bytes = sizeof(Request) + request.count * sizeof(Span);
  const bool ok = WriteBytes(fd, &request, sizeof(request)) &&
      WriteBytes(fd, spans.data(), spans.size() * sizeof(Span)) && ReadBytes(fd, response, sizeof(*response));
  close(fd);
  if (!ok) return Status::InDoubt("CPU control response unavailable; retain source and task");
  const auto& h = response->header;
  if (h.magic != kMagic || h.version != kVersion || h.bytes != sizeof(Response) ||
      h.command != (request.header.command | 0x8000U) ||
      response->region != request.region || response->source_id != request.source_id || response->task != request.task ||
      (request.epoch && response->epoch != request.epoch))
    return Status::InDoubt("CPU response identity/version mismatch");
  return Status::OK();
}
}
