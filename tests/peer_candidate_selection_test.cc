#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "peer_candidate_selection.h"

namespace {
void Check(bool ok) { if (!ok) throw std::runtime_error("peer candidate test failed"); }
struct FakeLink {
  std::vector<std::uint8_t> eids{9, 8, 7, 6, 5};
  int allocation_failure = -1;
  bool release_failure = false;
  std::vector<std::uint32_t> allocated, released;
  kvaf::Status AllocateRegion(std::uint64_t bytes, std::uint64_t slot,
      std::uint32_t index, kvaf::peer::PeerRegion* out) {
    Check(bytes == 4096 && slot == 4096);
    allocated.push_back(index);
    if (static_cast<int>(index) == allocation_failure) return kvaf::Status::Internal("injected allocation error");
    out->base = 4096U * (index + 1U);
    out->eid[0] = eids.at(index);
    return kvaf::Status::OK();
  }
  kvaf::Status ReleaseRegion(std::uint64_t base) {
    released.push_back(static_cast<std::uint32_t>(base / 4096U - 1U));
    return release_failure ? kvaf::Status::Internal("injected release error") : kvaf::Status::OK();
  }
};
}

int main() {
  std::array<std::uint8_t, kvaf::peer::kEidBytes> target{};
  target[0] = 5;
  std::uint32_t selected = 99;
  FakeLink link;
  Check(kvaf::detail::FindPeerCandidateByEid(link, target, 5, &selected).ok());
  Check(selected == 4 && link.allocated == link.released && link.allocated.size() == 5);
  link = FakeLink{};
  link.allocation_failure = 1;
  Check(kvaf::detail::FindPeerCandidateByEid(link, target, 5, &selected).ok());
  Check(selected == 4 && link.released == std::vector<std::uint32_t>({0, 2, 3, 4}));
  link = FakeLink{};
  link.release_failure = true;
  selected = 99;
  Check(!kvaf::detail::FindPeerCandidateByEid(link, target, 5, &selected).ok());
  Check(selected == 99 && link.allocated.size() == 1);
  link = FakeLink{};
  target[0] = 0;
  Check(!kvaf::detail::FindPeerCandidateByEid(link, target, 5, &selected).ok());
  Check(selected == 99 && link.allocated == link.released);
  std::puts("peer_candidate_selection: PASS (exact match, cleanup, allocation/release errors, no match)");
}
