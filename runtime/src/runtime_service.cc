#include "runtime_service.h"

#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>

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

Status RuntimeService::Serve(Runtime& runtime) const {
  if (!runtime.initialized()) return Status::NotInitialized("RuntimeService requires an initialized Runtime");
  const int listener = Listen(options_);
  if (listener < 0) {
    return Status::Internal("cannot listen on " + options_.address + ":" +
                            std::to_string(options_.port) + ": " + std::strerror(errno));
  }
  std::fprintf(stderr, "[KVAF_SERVICE] LISTEN address=%s port=%u\n", options_.address.c_str(), options_.port);
  while (true) {
    const int connection = accept(listener, nullptr, nullptr);
    if (connection < 0) {
      if (errno == EINTR) continue;
      close(listener);
      return Status::Internal("accept failed: " + std::string(std::strerror(errno)));
    }
    (void)runtime.ServeConnection(connection);
    close(connection);
  }
}

}  // namespace kvaf
