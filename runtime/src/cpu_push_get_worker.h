#pragma once

#include <atomic>
#include <memory>
#include <thread>

#include "cpu_push_get_transport.h"

namespace kvaf::cpu_push {
class Worker {
 public:
  explicit Worker(std::unique_ptr<Transport> transport) : transport_(std::move(transport)) {}
  ~Worker();
  Status Start();
  Status Stop();
 private:
  void Run();
  std::unique_ptr<Transport> transport_;
  std::thread thread_;
  std::atomic<bool> stop_{false};
};
}  // namespace kvaf::cpu_push
