#include "ubmem_allocator.h"

namespace kvaf::ubmem {
DdrAllocator::~DdrAllocator() {
  // The Operator destructor may have finished a previously failed unmap.
  if (allocated_.base != 0) (void)Release(allocated_);
}

Status DdrAllocator::Allocate(const RuntimeInitRequest& request, RemoteRegionDesc* region) {
  if (region == nullptr || request.ddr_device != DeviceKind::kCpu) {
    return Status::InvalidArgument("UBmem allocator requires CPU DDR");
  }
  if (allocated_.base != 0) return Status::InvalidState("UBmem DDR is already allocated");
  peer::PeerRegion allocated;
  Status status = link_->AllocateUbmemRegion(request.ddr_bytes, request.slot_size, &allocated);
  if (status.ok()) {
    allocated_ = {allocated.base, allocated.length, request.slot_size, DeviceKind::kCpu};
    *region = allocated_;
  }
  return status;
}
Status DdrAllocator::Release(const RemoteRegionDesc& region) {
  const Status status = link_->ReleaseRegion(region.base);
  if (status.ok() && allocated_.base == region.base) allocated_ = {};
  return status;
}
}  // namespace kvaf::ubmem
