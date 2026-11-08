#include "runtime_service.h"

#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <vector>

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
  std::fprintf(stderr, "[KVAF_CPU] LISTEN address=%s port=%u\n", options_.address.c_str(), options_.port);
  // A bounded pool allows independent session setup/control to progress while
  // another session drains. Runtime owns per-record synchronization.
  std::mutex mutex;
  std::condition_variable ready, space;
  std::deque<int> queue;
  bool stopping = false;
  std::vector<std::thread> workers;
  auto worker = [&] {
    for (;;) {
      int connection;
      {
        std::unique_lock<std::mutex> lock(mutex);
        ready.wait(lock, [&] { return stopping || !queue.empty(); });
        if (queue.empty()) return;
        connection = queue.front();
        queue.pop_front();
        space.notify_one();
      }
      try {
        (void)runtime.ServeConnection(connection);
      } catch (const std::exception& e) {
        std::fprintf(stderr, "[KVAF_CPU] control request failed: %s\n", e.what());
      }
      close(connection);
    }
  };
  auto finish = [&] {
    {
      std::lock_guard<std::mutex> lock(mutex);
      stopping = true;
    }
    ready.notify_all();
    for (auto& thread : workers) thread.join();
    close(listener);
  };
  try {
    for (int i = 0; i < 8; ++i) workers.emplace_back(worker);
  } catch (const std::exception& e) {
    finish();
    return Status::Internal(e.what());
  }
  for (;;) {
    const int connection = accept(listener, nullptr, nullptr);
    if (connection < 0) {
      if (errno == EINTR) continue;
      const std::string error = std::strerror(errno);
      finish();
      return Status::Internal("accept failed: " + error);
    }
    timeval timeout{30, 0};
    setsockopt(connection, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(connection, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    std::unique_lock<std::mutex> lock(mutex);
    space.wait(lock, [&] { return queue.size() < 64; });
    queue.push_back(connection);
    ready.notify_one();
  }
}

}  // namespace kvaf
