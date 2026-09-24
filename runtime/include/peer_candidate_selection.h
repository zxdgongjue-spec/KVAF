#pragma once

#include <cstdio>

#include "peer_link.h"

namespace kvaf::detail {

// Discover only metadata: no NPU queues or WQEs exist at this point. Link is
// templated so release-on-mismatch and failure handling can be tested on CPU.
template <typename Link>
Status FindPeerCandidateByEid(Link& link,
    const std::array<std::uint8_t, peer::kEidBytes>& eid,
    std::uint32_t candidate_count, std::uint32_t* selected) {
  if (selected == nullptr) return Status::InvalidArgument("null peer candidate output");
  constexpr std::uint64_t kDiscoveryBytes = 4096U;
  for (std::uint32_t index = 0U; index < candidate_count; ++index) {
    peer::PeerRegion region{};
    const Status allocated = link.AllocateRegion(kDiscoveryBytes, kDiscoveryBytes, index, &region);
    if (!allocated.ok()) {
      std::fprintf(stderr, "[KVAF_AIV] B12_DRIVER_PAIR_CANDIDATE index=%u error=%s\n",
                   index, allocated.message().c_str());
      continue;
    }
    const bool matched = region.eid == eid;
    const Status released = link.ReleaseRegion(region.base);
    if (!released.ok()) return released;
    std::fprintf(stderr, "[KVAF_AIV] B12_DRIVER_PAIR_CANDIDATE index=%u match=%u\n",
                 index, static_cast<unsigned int>(matched));
    if (matched) {
      *selected = index;
      return Status::OK();
    }
  }
  return Status::NotFound("CPU service has no EID matching driver Host/NPU pair");
}

}  // namespace kvaf::detail
