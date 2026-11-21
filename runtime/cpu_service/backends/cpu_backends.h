#pragma once
#include "cpu_runtime.h"
#include "peer_protocol.h"
#include "peer_link.h"

namespace kvaf::offload { class Backend; }
namespace kvaf {
// Allocation-scoped CPU context and DDR registration owner. Execution
// backends keep independent channels, gather workers and Scatter state.
class CpuUrmaMemoryPool {
 public:
  CpuUrmaMemoryPool(const CpuRuntimeOptions&, void* memory, std::uint64_t bytes,
                    std::uint64_t epoch, std::uint64_t allocation_request);
  ~CpuUrmaMemoryPool();
  Status Close();
  Status PrepareCpuPull(std::uint32_t candidate, std::uint32_t token,
                        std::uint64_t logical_bytes, peer::PeerRegion*);
  std::unique_ptr<offload::Backend> CreateCpuPullBackend(std::shared_ptr<void> lease);
 private:
  friend class UrmaCpuBackend;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
class UrmaCpuBackend {
 public:
  UrmaCpuBackend();
  ~UrmaCpuBackend();
  Status Initialize(const CpuRuntimeOptions&, std::uint64_t& next_session,
                    std::shared_ptr<CpuUrmaMemoryPool> memory_pool = {});
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
