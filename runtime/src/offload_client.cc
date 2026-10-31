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
}
