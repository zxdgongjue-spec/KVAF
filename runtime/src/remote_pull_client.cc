#include "offload_client.h"
#include "remote_pull_service.h"
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

namespace kvaf::remote_pull {
using offload::ReadBytes;
using offload::WriteBytes;
using offload::SetSocketTimeout;
ControlChannel::~ControlChannel() { Close(); }
void ControlChannel::Close() {
  if (fd_ >= 0) close(fd_);
  fd_ = -1;
}
Status ControlChannel::Exchange(const std::string& address, Request request,
                                const std::vector<Span>& spans, Response* response) {
  if (!response || spans.size() > kMaxSpans) return Status::InvalidArgument("invalid Pull exchange");
  if (fd_ >= 0 && address_ != address) Close();
  if (fd_ >= 0) {
    pollfd p{fd_, POLLIN, 0};
    if (poll(&p, 1, 0) > 0 && p.revents) {
      char byte;
      // Detect an idle peer close before sending a new request. Never retry
      // after any request bytes have been sent: its task may already be running.
      if ((p.revents & (POLLHUP | POLLERR | POLLNVAL)) ||
          recv(fd_, &byte, 1, MSG_PEEK | MSG_DONTWAIT) == 0) Close();
    }
  }
  if (fd_ < 0) {
    const auto colon = address.rfind(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 == address.size())
      return Status::InvalidArgument("Offload requires host:port");
    addrinfo hints{}, *list = nullptr;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(address.substr(0, colon).c_str(), address.substr(colon+1).c_str(), &hints, &list))
      return Status::InDoubt("cannot resolve CPU service");
    int& fd = fd_;
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
    address_ = address;
  }
  const int fd = fd_;
  SetSocketTimeout(fd);
  int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  if (request.header.command == static_cast<std::uint16_t>(Command::Probe)) {
    // The synchronous reference probe may wait for a 30-second READ CQ.
    timeval timeout{35, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  }
  if (request.header.command == static_cast<std::uint16_t>(Command::SubmitAndWait)) {
    // The server applies timeout_ms to waiting for this READ. Allow the normal
    // control RPC budget in addition to that wait; timeout never cancels DMA.
    const std::uint64_t budget = std::uint64_t(request.timeout_ms) + 5000;
    timeval timeout{static_cast<time_t>(budget / 1000), static_cast<suseconds_t>((budget % 1000) * 1000)};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  }
  request.count = static_cast<std::uint32_t>(spans.size());
  const auto request_bytes = RequestBytes(request.header.command);
  request.header.bytes = request_bytes + request.count * sizeof(Span);
  *response = {};
  const bool sent = WriteBytes(fd, &request, request_bytes) && WriteBytes(fd, spans.data(), spans.size()*sizeof(Span));
  auto& h = response->header;
  const auto response_bytes = ResponseBytes(request.header.command);
  const bool header_ok = sent && ReadBytes(fd, &h, sizeof(h)) && h.magic == kMagic && h.version == kVersion &&
      h.bytes == response_bytes && h.command == (request.header.command | 0x8000U);
  const bool ok = header_ok && ReadBytes(fd, reinterpret_cast<char*>(response)+sizeof(h), response_bytes-sizeof(h));
  if (!ok) {
    Close();
    return Status::InDoubt("CPU control response unavailable or version mismatch; retain source and task");
  }
  if (response->region != request.region || response->source_id != request.source_id || response->task != request.task ||
      response->mr_id != request.mr_id || (request.epoch && response->epoch != request.epoch)) {
    Close();
    return Status::InDoubt("CPU response identity/version mismatch");
  }
  return Status::OK();
}
Status Exchange(const std::string& address, Request request,
                const std::vector<Span>& spans, Response* response) {
  ControlChannel channel;
  return channel.Exchange(address, request, spans, response);
}
}
