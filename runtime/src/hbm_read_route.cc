#include "hbm_read_route.h"
#include "npu_eid_discovery.h"
#include <boost/property_tree/json_parser.hpp>
#include <filesystem>
#include <map>

namespace kvaf::hbm_read {
namespace {
using Tree = boost::property_tree::ptree;
const Tree& Children(const Tree& node, const char* key) {
  static const Tree empty;
  return node.get_child(key, empty);
}

// Map this rank's direct ports to the edges leading to the requested NPU.
// Zero identifies a mesh port wired to another peer.
std::map<std::string, unsigned> MeshPorts(const Tree& root, const std::string& root_path,
                                         unsigned local_id, unsigned peer_id) {
  std::map<std::string, unsigned> ports;
  const auto topo_path = root.get<std::string>("topo_file_path", "");
  if (topo_path.empty()) return ports;
  Tree topology;
  try {
    boost::property_tree::read_json((std::filesystem::path(root_path).parent_path() / topo_path).string(), topology);
  } catch (const boost::property_tree::json_parser_error&) {
    return ports;  // Optional port mapping; rootinfo still supplies the EIDs.
  }
  unsigned index = 0;
  for (const auto& item : Children(topology, "edge_list")) {
    ++index;
    const auto& edge = item.second;
    if (edge.get<int>("net_layer", -1) != 0 || edge.get<std::string>("link_type", "") != "PEER2PEER" ||
        edge.get<std::string>("position", "DEVICE") != "DEVICE") continue;
    const auto a = edge.get<unsigned>("local_a", UINT32_MAX);
    const auto b = edge.get<unsigned>("local_b", UINT32_MAX);
    if (a != local_id && b != local_id) continue;
    const bool direct = (a == local_id ? b : a) == peer_id;
    for (const auto& port : Children(edge, a == local_id ? "local_a_ports" : "local_b_ports")) {
      auto& preferred = ports[port.second.data()];
      if (direct && !preferred) preferred = index;
    }
  }
  return ports;
}
}  // namespace

RoutePeer LocalRoutePeer(std::uint32_t physical_device) {
  RoutePeer peer{{}, physical_device};
  std::string uuid;
  if (std::ifstream("/proc/sys/kernel/random/boot_id") >> uuid) {
    uuid.erase(std::remove(uuid.begin(), uuid.end(), '-'), uuid.end());
    if (!npu_eid::ParseEid(uuid, &peer.boot_id)) peer.boot_id = {};
  }
  return peer;
}

RouteScope ClassifyRoute(const RoutePeer& local, const RoutePeer& peer) {
  if (local.boot_id == RouteId{} || peer.boot_id == RouteId{}) return RouteScope::Unknown;
  return local.boot_id == peer.boot_id ? RouteScope::SameHost : RouteScope::Remote;
}

const char* RouteName(RouteScope scope) {
  if (scope == RouteScope::SameHost) return "fullmesh";
  if (scope == RouteScope::Remote) return "clos";
  return "unknown";
}

static Status ReadRouteCandidates(const RoutePeer& local, const RoutePeer& peer, std::vector<RouteCandidate>* out) {
  const auto scope = ClassifyRoute(local, peer);
  if (scope == RouteScope::Unknown) return Status::InvalidState("automatic NPU Pull routing requires both host boot IDs");
  if (local.physical_device == UINT32_MAX) return Status::InvalidState("cannot resolve physical NPU for automatic routing");
  const auto* configured = std::getenv("KVAF_HCCL_ROOTINFO_PATH");
  const std::string path = configured && *configured ? configured : "/etc/hccl_rootinfo.json";
  Tree root;
  try {
    boost::property_tree::read_json(path, root);
  } catch (const boost::property_tree::json_parser_error& error) {
    return Status::InvalidArgument("NPU Pull rootinfo: " + std::string(error.what()));
  }
  const bool mesh = scope == RouteScope::SameHost;
  const Tree* rank = nullptr;
  auto peer_id = peer.physical_device;
  for (const auto& entry : Children(root, "rank_list")) {
    const auto device = entry.second.get<unsigned>("device_id", UINT32_MAX);
    if (device == local.physical_device) rank = &entry.second;
    if (device == peer.physical_device) peer_id = entry.second.get<unsigned>("local_id", device);
  }
  if (!rank) return Status::NotFound("no rank for physical device " + std::to_string(local.physical_device) + " in " + path);
  std::map<std::string, unsigned> mesh_ports;
  if (mesh) mesh_ports = MeshPorts(root, path, rank->get<unsigned>("local_id", local.physical_device), peer_id);
  for (const auto& level : Children(*rank, "level_list")) {
    const auto& net = level.second;
    if (net.get<int>("net_layer", -1) != (mesh ? 0 : 1)) continue;
    const auto type = net.get<std::string>("net_type", "");
    if (mesh ? (type != "MESH" && type != "FULLMESH" && type != "1DMESH" && type != "TOPO_FILE_DESC") : type != "CLOS") continue;
    for (const auto& entry : Children(net, "rank_addr_list")) {
      const auto& addr = entry.second;
      RouteCandidate candidate{};
      if (addr.get<std::string>("addr_type", "") != "EID" ||
          !npu_eid::ParsePairInfoEid(addr.get<std::string>("addr", ""), &candidate.eid)) continue;
      bool mesh_port = false;
      for (const auto& port : Children(addr, "ports")) {
        const auto it = mesh_ports.find(port.second.data());
        if (it != mesh_ports.end()) {
          mesh_port = true;
          if (it->second) candidate.direct_edge = it->second;
        }
      }
      if (type == "TOPO_FILE_DESC" && !mesh_ports.empty() && !mesh_port) continue;
      const auto plane = addr.get<std::string>("plane_id", "");
      if (plane.size() >= sizeof(candidate.plane)) return Status::InvalidArgument("NPU Pull plane_id exceeds 256 bytes");
      std::copy(plane.begin(), plane.end(), candidate.plane);
      if (std::none_of(out->begin(), out->end(), [&](const auto& c){return c.eid == candidate.eid;})) {
        if (out->size() == kMaxRouteCandidates) return Status::InvalidArgument("too many NPU Pull EID candidates");
        out->push_back(candidate);
      }
    }
  }
  if (out->empty()) return Status::NotFound("no " + std::string(RouteName(scope)) + " EID for physical device " +
                                            std::to_string(local.physical_device) + " in " + path);
  return Status::OK();
}

Status RouteCandidates(const RoutePeer& local, const RoutePeer& peer, const std::string& fixed_eid,
                       std::vector<RouteCandidate>* out) {
  out->clear();
  if (fixed_eid.empty()) return ReadRouteCandidates(local, peer, out);
  RouteCandidate fixed{};
  if (!npu_eid::ParsePairInfoEid(fixed_eid, &fixed.eid)) return Status::InvalidArgument("invalid NPU Pull EID");
  // A pinned route is always usable without rootinfo. When the other endpoint
  // selects automatically, optional topology metadata can prioritize its matching port/plane.
  if (!peer.pinned) {
    std::vector<RouteCandidate> discovered;
    if (ReadRouteCandidates(local, peer, &discovered).ok()) {
      const auto it = std::find_if(discovered.begin(), discovered.end(), [&](const auto& c){return c.eid == fixed.eid;});
      if (it != discovered.end()) fixed = *it;
    }
  }
  fixed.pinned = 1;
  out->push_back(fixed);
  return Status::OK();
}

std::vector<std::pair<unsigned, unsigned>> RoutePairs(RouteScope scope,
    const std::vector<RouteCandidate>& reader, const std::vector<RouteCandidate>& producer) {
  std::vector<std::pair<unsigned, unsigned>> preferred, fallback;
  for (unsigned b = 0; b < producer.size(); ++b) {
    for (unsigned a = 0; a < reader.size(); ++a) {
      const auto& x = reader[a];
      const auto& y = producer[b];
      const bool match = scope == RouteScope::SameHost
          ? ((x.direct_edge && x.direct_edge == y.direct_edge) || (x.pinned && !x.direct_edge && y.direct_edge) ||
             (y.pinned && !y.direct_edge && x.direct_edge))
          : (x.plane[0] && std::strcmp(x.plane, y.plane) == 0);
      (match ? preferred : fallback).emplace_back(a, b);
    }
  }
  preferred.insert(preferred.end(), fallback.begin(), fallback.end());
  return preferred;
}
}  // namespace kvaf::hbm_read
