#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "peer_link.h"

namespace kvaf::detail {

inline bool ParsePeerEid(const std::string& text,
                         std::array<uint8_t, peer::kEidBytes>* eid) {
  if (eid == nullptr || text.size() != 2U * peer::kEidBytes) return false;
  const auto digit = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
  };
  for (std::size_t i = 0U; i < eid->size(); ++i) {
    const int high = digit(text[2U * i]);
    const int low = digit(text[2U * i + 1U]);
    if (high < 0 || low < 0) return false;
    (*eid)[i] = static_cast<uint8_t>(high * 16 + low);
  }
  return std::any_of(eid->begin(), eid->end(), [](uint8_t b) { return b != 0U; });
}

inline Status ValidatePeerEidsByCore(const std::vector<std::string>& eids,
                                     uint32_t cores, std::size_t local_eid_count,
                                     bool adaptive) {
  if (eids.empty()) return Status::OK();
  if (cores == 0U || eids.size() != cores) {
    return Status::InvalidArgument("peer_eids_by_core must contain one EID per AIV core");
  }
  if (local_eid_count > 1U || adaptive) {
    return Status::InvalidArgument(
        "peer_eids_by_core requires at most one explicit NPU EID and adaptive_ub_device=False");
  }
  for (std::size_t i = 0U; i < eids.size(); ++i) {
    std::array<uint8_t, peer::kEidBytes> eid{};
    if (!ParsePeerEid(eids[i], &eid)) {
      return Status::InvalidArgument("peer_eids_by_core[" + std::to_string(i) +
                                    "] must be a non-zero 32-digit hexadecimal EID");
    }
  }
  return Status::OK();
}

// Preserve core order and repeated EIDs (shared-EID control experiments).
// Candidate IDs are service-local discovery results, never user configuration.
inline Status ResolvePeerEidsByCore(
    const std::vector<std::string>& eids,
    const std::vector<peer::PeerCandidateInfo>& candidates,
    const std::string& device_filter, std::vector<uint32_t>* resolved) {
  if (resolved == nullptr) return Status::InvalidArgument("missing endpoint mapping output");
  std::vector<uint32_t> result;
  for (std::size_t core = 0U; core < eids.size(); ++core) {
    std::array<uint8_t, peer::kEidBytes> eid{};
    if (!ParsePeerEid(eids[core], &eid)) return Status::InvalidArgument("invalid peer EID");
    const peer::PeerCandidateInfo* match = nullptr;
    for (const auto& candidate : candidates) {
      if (candidate.eid != eid ||
          (!device_filter.empty() && candidate.device_name != device_filter)) continue;
      if (match != nullptr) {
        return Status::InvalidArgument("ambiguous peer EID for core " + std::to_string(core) +
                                       "; specify peer_ub_device");
      }
      match = &candidate;
    }
    if (match == nullptr) {
      return Status::NotFound("configured peer EID unavailable for core " + std::to_string(core) +
                              ": " + eids[core]);
    }
    result.push_back(match->candidate_id);
  }
  *resolved = std::move(result);
  return Status::OK();
}

}  // namespace kvaf::detail
