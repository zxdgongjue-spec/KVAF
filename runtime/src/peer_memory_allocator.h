#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "peer_link.h"
#include "runtime.h"

namespace kvaf {

// Allocates one remote region for an NPU initiator. Candidate indices belong
// to the CPU service or to the selected remote NPU, depending on peer_device.
class PeerMemoryAllocator final : public DdrMemoryAllocator {
 public:
  PeerMemoryAllocator(std::shared_ptr<peer::PeerLink> link, DeviceKind peer_device,
                      std::uint32_t peer_device_id)
      : link_(std::move(link)), peer_device_(peer_device), peer_device_id_(peer_device_id) {}

  Status GetCandidateCount(std::uint32_t* count) const {
    if (link_ == nullptr || count == nullptr)
      return Status::InvalidArgument("invalid peer candidate query");
    *count = 0U;
    if (peer_device_ == DeviceKind::kCpu) return link_->GetCandidateCount(count);
    if (peer_device_ != DeviceKind::kNpu)
      return Status::InvalidArgument("unsupported peer device");
    std::vector<std::array<std::uint8_t, peer::kEidBytes>> candidates;
    const Status status = link_->GetHbmEidCandidates(peer_device_id_, &candidates);
    if (status.ok()) *count = static_cast<std::uint32_t>(candidates.size());
    return status;
  }

  void SetCandidate(std::uint32_t candidate_id) { candidate_id_ = candidate_id; }
  bool has_allocation() const { return allocated_.base != 0U; }

  Status Allocate(const RuntimeInitRequest& request, RemoteRegionDesc* region) override {
    if (link_ == nullptr || region == nullptr || request.ddr_device != peer_device_ ||
        (peer_device_ != DeviceKind::kCpu && peer_device_ != DeviceKind::kNpu) ||
        request.ddr_bytes == 0U || request.slot_size == 0U ||
        request.ddr_bytes % request.slot_size != 0U) {
      return Status::InvalidArgument("invalid peer memory allocation request");
    }
    if (has_allocation()) return Status::InvalidState("peer memory is already allocated");
    const Status status = peer_device_ == DeviceKind::kNpu
        ? link_->AllocateHbmRegion(request.ddr_bytes, request.slot_size,
                                   peer_device_id_, candidate_id_, &allocated_)
        : link_->AllocateRegion(request.ddr_bytes, request.slot_size, candidate_id_, &allocated_);
    if (!status.ok()) return status;
    *region = RemoteRegionDesc{allocated_.base, allocated_.length, request.slot_size, peer_device_};
    return Status::OK();
  }

  Status Release(const RemoteRegionDesc& region) override {
    if (!has_allocation() || allocated_.base != region.base)
      return Status::NotFound("peer memory allocation was not found");
    const Status status = peer_device_ == DeviceKind::kNpu
        ? link_->ReleaseHbmRegion(allocated_.session_id)
        : link_->ReleaseRegion(allocated_.base);
    if (status.ok()) allocated_ = {};
    return status;
  }

 private:
  std::shared_ptr<peer::PeerLink> link_;
  DeviceKind peer_device_;
  std::uint32_t peer_device_id_;
  std::uint32_t candidate_id_{0U};
  peer::PeerRegion allocated_{};
};

}  // namespace kvaf
