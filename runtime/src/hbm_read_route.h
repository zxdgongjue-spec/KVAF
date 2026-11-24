#pragma once

#include "status.h"
#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace kvaf::hbm_read {

using RouteId = std::array<std::uint8_t, 16>;
struct RoutePeer {
  RouteId boot_id{};
  std::uint32_t physical_device{UINT32_MAX};
  std::uint8_t pinned{};
};
struct RouteCandidate {
  RouteId eid{};
  std::uint32_t direct_edge{};  // One-based edge_list entry on the shared server.
  char plane[257]{};
  std::uint8_t pinned{};
};
constexpr std::size_t kMaxRouteCandidates = 256;
enum class RouteScope { Unknown, SameHost, Remote };

RoutePeer LocalRoutePeer(std::uint32_t physical_device);
RouteScope ClassifyRoute(const RoutePeer& local, const RoutePeer& peer);
const char* RouteName(RouteScope scope);
Status RouteCandidates(const RoutePeer& local, const RoutePeer& peer, const std::string& fixed_eid,
                       std::vector<RouteCandidate>* out);
// Indices into the reader and producer candidate lists, respectively.
std::vector<std::pair<unsigned, unsigned>> RoutePairs(RouteScope scope,
    const std::vector<RouteCandidate>& reader, const std::vector<RouteCandidate>& producer);
}  // namespace kvaf::hbm_read
