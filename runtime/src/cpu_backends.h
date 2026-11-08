#pragma once
#include "cpu_runtime.h"
#include "peer_protocol.h"
#include "peer_link.h"

namespace kvaf {
class UrmaCpuBackend {
 public:
  UrmaCpuBackend();
  ~UrmaCpuBackend();
  Status Initialize(const CpuRuntimeOptions&, std::uint64_t& next_session);
  Status AttachRegion(void* memory, std::uint64_t bytes, std::uint64_t slot_size,
                      std::uint32_t candidate, peer::PeerRegion* binding);
  Status Call(const std::vector<std::uint8_t>& request, std::vector<std::uint8_t>* response);
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
