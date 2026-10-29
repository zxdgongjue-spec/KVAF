#pragma once

#include "peer_link.h"
#include "runtime.h"
#include "ubmem/hal.h"

namespace kvaf::ubmem {

// Test-only provider: uses Ascend HAL Host DDR/Fabric, with no UMDK or OBMM.
class CpuRuntime final : public kvaf::Runtime {
 public:
  CpuRuntime();
  ~CpuRuntime() override;
  Status Initialize(int numa_node = -1, const std::string& hal_library = "libascend_hal.so");
  Status LinkTo(const LinkTarget&) override;
  Status AcceptLink(const LinkTarget&) override;
  Status ReleaseLink(LinkId session) override;
  Status ServeConnection(int fd) override;
  Status Shutdown() override;
  bool initialized() const override;
  RuntimeCapabilities capabilities() const override { return {false, false, true, false}; }
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class DdrAllocator final : public kvaf::DdrMemoryAllocator {
 public:
  explicit DdrAllocator(std::shared_ptr<peer::PeerLink> link) : link_(std::move(link)) {}
  Status Allocate(const RuntimeInitRequest& request, RemoteRegionDesc* region) override;
  Status Release(const RemoteRegionDesc& region) override;
 private:
  std::shared_ptr<peer::PeerLink> link_;
};
}  // namespace kvaf::ubmem
