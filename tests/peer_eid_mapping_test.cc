#include "peer_eid_mapping.h"

#include <cstdlib>
#include <iostream>

#define CHECK(x) do { if (!(x)) { std::cerr << "line " << __LINE__ << ": " #x "\n"; std::exit(1); } } while (false)

int main() {
  using namespace kvaf;
  std::vector<peer::PeerCandidateInfo> candidates;
  std::vector<std::string> eids;
  for (uint32_t i = 0U; i < 8U; ++i) {
    eids.push_back(std::string(31, '0') + char('1' + i));
    peer::PeerCandidateInfo info{};
    info.candidate_id = 100U + i;  // Deliberately different from vector indices.
    info.device_name = i < 4U ? "ub0" : "ub1";
    CHECK(detail::ParsePeerEid(eids.back(), &info.eid));
    candidates.push_back(info);
  }
  for (uint32_t cores : {2U, 4U, 8U}) {
    for (bool distinct : {false, true}) {
      std::vector<std::string> configured(eids.begin(), eids.begin() + cores);
      if (!distinct) std::fill(configured.begin(), configured.end(), eids.front());
      CHECK(detail::ValidatePeerEidsByCore(configured, cores, 1U, false).ok());
      std::vector<uint32_t> mapped;
      CHECK(detail::ResolvePeerEidsByCore(configured, candidates, "", &mapped).ok());
      CHECK(mapped.size() == cores);
      for (uint32_t core = 0U; core < cores; ++core) CHECK(mapped[core] == 100U + (distinct ? core : 0U));
    }
  }
  CHECK(detail::ValidatePeerEidsByCore({}, 8U, 4U, true).ok());
  CHECK(!detail::ValidatePeerEidsByCore(eids, 4U, 1U, false).ok());
  CHECK(!detail::ValidatePeerEidsByCore(eids, 8U, 2U, false).ok());
  CHECK(!detail::ValidatePeerEidsByCore(eids, 8U, 1U, true).ok());
  for (const std::string& invalid : std::vector<std::string>{"abc", std::string(32, '0'), std::string(32, 'z')}) {
    CHECK(!detail::ValidatePeerEidsByCore({invalid}, 1U, 0U, false).ok());
  }
  std::vector<uint32_t> mapped{999U};
  CHECK(!detail::ResolvePeerEidsByCore(eids, candidates, "ub0", &mapped).ok());
  CHECK(mapped == std::vector<uint32_t>{999U});  // No partially published map.
  auto duplicate = candidates[0];
  duplicate.device_name = "ub2";
  duplicate.candidate_id = 42U;
  candidates.push_back(duplicate);
  CHECK(!detail::ResolvePeerEidsByCore({eids[0]}, candidates, "", &mapped).ok());
  CHECK(detail::ResolvePeerEidsByCore({eids[0]}, candidates, "ub2", &mapped).ok());
  CHECK(mapped == std::vector<uint32_t>{42U});
  std::cout << "Peer EID mapping: 2/4/8 cores, repeats, filters and invalid input passed\n";
}
