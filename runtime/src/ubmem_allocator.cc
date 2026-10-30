#include "ubmem_allocator.h"

namespace kvaf::ubmem {
Status DdrAllocator::Allocate(const RuntimeInitRequest& request, RemoteRegionDesc* region) {
  if (region == nullptr || request.ddr_device != DeviceKind::kCpu) {
    return Status::InvalidArgument("UBmem allocator requires CPU DDR");
  }
  peer::PeerRegion allocated;
  Status status = link_->AllocateUbmemRegion(request.ddr_bytes, request.slot_size, &allocated);
  if (status.ok()) *region = {allocated.base, allocated.length, request.slot_size, DeviceKind::kCpu};
  return status;
}
Status DdrAllocator::Release(const RemoteRegionDesc& region) { return link_->ReleaseRegion(region.base); }
}  // namespace kvaf::ubmem
