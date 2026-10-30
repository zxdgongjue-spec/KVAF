#pragma once
#include "peer_link.h"
#include "runtime.h"

namespace kvaf::ubmem {
class DdrAllocator final : public kvaf::DdrMemoryAllocator {
 public:
  explicit DdrAllocator(std::shared_ptr<peer::PeerLink> link) : link_(std::move(link)) {}
  Status Allocate(const RuntimeInitRequest& request, RemoteRegionDesc* region) override;
  Status Release(const RemoteRegionDesc& region) override;
 private:
  std::shared_ptr<peer::PeerLink> link_;
};
}  // namespace kvaf::ubmem
