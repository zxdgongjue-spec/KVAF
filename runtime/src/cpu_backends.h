#pragma once
#include "cpu_runtime.h"
#include "peer_protocol.h"

namespace kvaf {
class UrmaCpuBackend {
 public:
  UrmaCpuBackend();
  ~UrmaCpuBackend();
  Status Initialize(const CpuRuntimeOptions&, std::uint64_t& next_session);
  Status ServeConnection(int fd, const peer::MessageHeader& header);
  Status ReleaseLink(LinkId id);
  Status Shutdown();
  bool initialized() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class UbmemCpuBackend {
 public:
  UbmemCpuBackend(const CpuRuntimeOptions&, std::uint64_t& next_session);
  ~UbmemCpuBackend();
  Status ServeConnection(int fd, const peer::MessageHeader& header);
  Status ReleaseLink(LinkId id);
  Status Shutdown();
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace kvaf
