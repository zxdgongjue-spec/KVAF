#pragma once
// Shared discovery from aicpu-urma: CLOS candidates, then the active device
// EID from asdrv pair_info. Both users must hold the same selector lock.
#include "peer_protocol.h"
#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <limits>
#include <string>
#include <vector>
#include <utility>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

namespace kvaf::npu_eid {
using AclrtGetPhyDevIdByLogicDevId = int (*)(int32_t, int32_t*);

struct NpuLocalEidCandidate {
  std::array<std::uint8_t, peer::kEidBytes> eid{};
  std::string source;
};

struct DiscoveryPaths {
  std::string rootinfo;
  std::string selector{"/proc/asdrv_ub/dev_id"};
  std::string pair_info{"/proc/asdrv_ub/pair_info"};
  std::string lock{"/tmp/kvaf_npu_runtime_asdrv_ub.lock"};
};

inline std::string EidToHex(const std::array<std::uint8_t, peer::kEidBytes>& eid) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string text;
  text.reserve(eid.size() * 2U);
  for (std::uint8_t byte : eid) {
    text.push_back(kHex[byte >> 4U]);
    text.push_back(kHex[byte & 0x0fU]);
  }
  return text;
}

inline bool ParseJsonUnsigned(const std::string& line, const char* field, std::uint32_t* value) {
  if (value == nullptr) return false;
  const std::string marker = std::string{"\""} + field + "\"";
  const std::size_t field_position = line.find(marker);
  const std::size_t colon = field_position == std::string::npos
                                ? std::string::npos
                                : line.find(':', field_position + marker.size());
  const std::size_t first = colon == std::string::npos
                                ? std::string::npos
                                : line.find_first_of("0123456789", colon + 1U);
  if (first == std::string::npos) return false;
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(line.c_str() + first, &end, 10);
  if (end == line.c_str() + first || parsed > std::numeric_limits<std::uint32_t>::max()) return false;
  *value = static_cast<std::uint32_t>(parsed);
  return true;
}

inline bool ParseJsonString(const std::string& line, const char* field, std::string* value) {
  if (value == nullptr) return false;
  const std::string marker = std::string{"\""} + field + "\"";
  const std::size_t field_position = line.find(marker);
  const std::size_t colon = field_position == std::string::npos
                                ? std::string::npos
                                : line.find(':', field_position + marker.size());
  const std::size_t first_quote = colon == std::string::npos
                                      ? std::string::npos
                                      : line.find('"', colon + 1U);
  const std::size_t second_quote = first_quote == std::string::npos
                                       ? std::string::npos
                                       : line.find('"', first_quote + 1U);
  if (second_quote == std::string::npos) return false;
  *value = line.substr(first_quote + 1U, second_quote - first_quote - 1U);
  return true;
}

inline bool ParseEid(const std::string& text,
              std::array<std::uint8_t, peer::kEidBytes>* eid) {
  if (eid == nullptr || text.size() != eid->size() * 2U) return false;
  const auto hex = [](char value) -> int {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
  };
  for (std::size_t index = 0; index < eid->size(); ++index) {
    const int high = hex(text[index * 2U]);
    const int low = hex(text[index * 2U + 1U]);
    if (high < 0 || low < 0) return false;
    (*eid)[index] = static_cast<std::uint8_t>((high << 4U) | low);
  }
  return std::any_of(eid->begin(), eid->end(), [](std::uint8_t byte) { return byte != 0; });
}

inline bool ParsePairInfoEid(const std::string& text,
                      std::array<std::uint8_t, peer::kEidBytes>* eid) {
  if (eid == nullptr) return false;
  std::string compact;
  compact.reserve(eid->size() * 2U);
  for (const unsigned char value : text) {
    if (std::isxdigit(value)) {
      compact.push_back(static_cast<char>(value));
    } else if (value != ':') {
      return false;
    }
  }
  return ParseEid(compact, eid);
}

inline std::string Trim(std::string text) {
  const std::size_t first = text.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return {};
  const std::size_t last = text.find_last_not_of(" \t\r\n");
  return text.substr(first, last - first + 1U);
}

inline bool ResolvePhysicalDeviceId(std::uint32_t logical_device_id, std::uint32_t* physical_device_id) {
  if (physical_device_id == nullptr ||
      logical_device_id > static_cast<std::uint32_t>(std::numeric_limits<int32_t>::max())) {
    return false;
  }
  auto get_physical = reinterpret_cast<AclrtGetPhyDevIdByLogicDevId>(
      dlsym(RTLD_DEFAULT, "aclrtGetPhyDevIdByLogicDevId"));
  if (get_physical == nullptr) {
    void* ascendcl = dlopen("libascendcl.so", RTLD_NOW | RTLD_NOLOAD);
    if (ascendcl == nullptr) ascendcl = dlopen("libascendcl.so", RTLD_NOW | RTLD_LOCAL);
    if (ascendcl != nullptr) {
      get_physical = reinterpret_cast<AclrtGetPhyDevIdByLogicDevId>(
          dlsym(ascendcl, "aclrtGetPhyDevIdByLogicDevId"));
    }
  }
  int32_t physical = -1;
  if (get_physical == nullptr ||
      get_physical(static_cast<int32_t>(logical_device_id), &physical) != 0 || physical < 0) {
    return false;
  }
  *physical_device_id = static_cast<std::uint32_t>(physical);
  return true;
}

inline bool ReadHcclClosEids(const std::string& path, std::uint32_t physical_device_id,
                      std::vector<std::array<std::uint8_t, peer::kEidBytes>>* eids,
                      std::string* error) {
  std::ifstream input(path);
  if (!input.is_open()) {
    *error = "cannot open " + path;
    return false;
  }
  bool selected_rank = false;
  bool found_rank = false;
  bool clos_level_one = false;
  bool clos_type = false;
  bool expect_eid_address = false;
  std::string line;
  while (std::getline(input, line)) {
    std::uint32_t value = 0U;
    if (ParseJsonUnsigned(line, "device_id", &value)) {
      selected_rank = value == physical_device_id;
      found_rank = found_rank || selected_rank;
      clos_level_one = clos_type = expect_eid_address = false;
      continue;
    }
    if (!selected_rank) continue;
    if (ParseJsonUnsigned(line, "net_layer", &value)) {
      clos_level_one = value == 1U;
      clos_type = expect_eid_address = false;
      continue;
    }
    if (!clos_level_one) continue;
    if (line.find("\"net_type\"") != std::string::npos) {
      clos_type = line.find("\"CLOS\"") != std::string::npos;
      continue;
    }
    if (clos_type && line.find("\"addr_type\"") != std::string::npos) {
      expect_eid_address = line.find("\"EID\"") != std::string::npos;
      continue;
    }
    if (!expect_eid_address) continue;
    std::string eid_text;
    if (!ParseJsonString(line, "addr", &eid_text)) continue;
    expect_eid_address = false;
    std::array<std::uint8_t, peer::kEidBytes> eid{};
    if (!ParseEid(eid_text, &eid)) {
      *error = "invalid CLOS EID for device_id=" + std::to_string(physical_device_id);
      return false;
    }
    if (std::find(eids->begin(), eids->end(), eid) == eids->end()) eids->push_back(eid);
  }
  if (!found_rank || eids->empty()) {
    *error = found_rank ? "no CLOS EID at net_layer=1 for physical device_id=" +
                              std::to_string(physical_device_id)
                        : "no rank for physical device_id=" + std::to_string(physical_device_id);
    return false;
  }
  return true;
}

inline bool ReadPairInfoDeviceEid(std::uint32_t physical_device_id,
                           std::array<std::uint8_t, peer::kEidBytes>* eid,
                           std::string* error, const DiscoveryPaths& paths = {}) {
  if (eid == nullptr || error == nullptr) return false;
  const int lock_fd = open(paths.lock.c_str(), O_CREAT | O_RDWR, 0600);
  if (lock_fd < 0 || flock(lock_fd, LOCK_EX) != 0) {
    *error = "cannot lock " + paths.lock + ": " + std::strerror(errno);
    if (lock_fd >= 0) close(lock_fd);
    return false;
  }
  bool success = false;
  {
    std::ofstream selector(paths.selector);
    if (!selector.is_open()) {
      *error = "cannot open " + std::string(paths.selector);
    } else {
      selector << physical_device_id << '\n';
      selector.close();
      if (!selector) {
        *error = "cannot select physical device in " + paths.selector;
        (void)flock(lock_fd, LOCK_UN); close(lock_fd); return false;
      }
      std::ifstream pair_info(paths.pair_info);
      if (!pair_info.is_open()) {
        *error = "cannot open " + std::string(paths.pair_info);
      } else {
        const std::string selected_marker = "dev_id=" + std::to_string(physical_device_id) + ";";
        bool selected_device = false;
        bool active_channel = false;
        std::uint32_t active_channels = 0U;
        std::uint32_t device_eids = 0U;
        std::string line;
        while (std::getline(pair_info, line)) {
          if (line.find("Dev id info.") != std::string::npos) {
            selected_device = line.find(selected_marker) != std::string::npos;
            active_channel = false;
            continue;
          }
          if (!selected_device) continue;
          if (line.find("chan_id=") != std::string::npos) {
            active_channel = line.find("flag=1") != std::string::npos;
            if (active_channel) ++active_channels;
            continue;
          }
          constexpr char kRemotePrefix[] = "remote_eid:";
          const std::size_t remote_pos = line.find(kRemotePrefix);
          if (active_channel && remote_pos != std::string::npos &&
              ParsePairInfoEid(Trim(line.substr(remote_pos + sizeof(kRemotePrefix) - 1U)), eid)) {
            ++device_eids;
          }
        }
        success = selected_device && active_channels == 1U && device_eids == 1U;
        if (!success) {
          *error = "expected one active pair_info device EID for physical device_id=" +
                   std::to_string(physical_device_id) + "; active=" +
                   std::to_string(active_channels) + " device_eids=" + std::to_string(device_eids);
        }
      }
    }
  }
  (void)flock(lock_fd, LOCK_UN);
  close(lock_fd);
  return success;
}

inline void AddNpuLocalEidCandidate(
    std::uint32_t physical_device_id,
    const std::array<std::uint8_t, peer::kEidBytes>& eid,
    std::string source, std::vector<NpuLocalEidCandidate>* candidates) {
  if (std::find_if(candidates->begin(), candidates->end(),
                   [&eid](const NpuLocalEidCandidate& candidate) {
                     return candidate.eid == eid;
                   }) != candidates->end()) {
    return;
  }
  candidates->push_back(NpuLocalEidCandidate{eid, std::move(source)});
  std::fprintf(stderr,
               "[KVAF_AIV] PEER_LINK_NPU_EID_DISCOVERED physical_device=%u source=%s eid=%s\n",
               physical_device_id, candidates->back().source.c_str(),
               EidToHex(candidates->back().eid).c_str());
}

inline bool DiscoverPhysicalNpuEids(std::uint32_t physical_device_id,
                                   std::vector<NpuLocalEidCandidate>* candidates,
                                   std::string* error, const DiscoveryPaths& paths = {}) {
  if (candidates == nullptr || error == nullptr) return false;
  candidates->clear();
  const char* configured = std::getenv("KVAF_HCCL_ROOTINFO_PATH");
  const std::string rootinfo = !paths.rootinfo.empty() ? paths.rootinfo : configured != nullptr && configured[0] != '\0'
                                   ? configured
                                   : "/etc/hccl_rootinfo.json";
  std::string clos_error;
  std::vector<std::array<std::uint8_t, peer::kEidBytes>> clos_eids;
  (void)ReadHcclClosEids(rootinfo, physical_device_id, &clos_eids, &clos_error);
  for (std::size_t index = 0; index < clos_eids.size(); ++index) {
    AddNpuLocalEidCandidate(physical_device_id, clos_eids[index],
                            "hccl_clos_pg_" + std::to_string(index), candidates);
  }
  if (clos_eids.empty()) {
    std::fprintf(stderr, "[KVAF_AIV] PEER_LINK_NPU_EID_DISCOVERY_UNAVAILABLE path=%s reason=%s\n",
                 rootinfo.c_str(), clos_error.c_str());
  }
  std::array<std::uint8_t, peer::kEidBytes> pair_info_eid{};
  std::string pair_info_error;
  if (ReadPairInfoDeviceEid(physical_device_id, &pair_info_eid, &pair_info_error, paths)) {
    AddNpuLocalEidCandidate(physical_device_id, pair_info_eid,
                            "asdrv_pair_info_device_eid", candidates);
  } else {
    std::fprintf(stderr, "[KVAF_AIV] PEER_LINK_NPU_EID_FALLBACK_UNAVAILABLE reason=%s\n",
                 pair_info_error.c_str());
  }
  if (!candidates->empty()) return true;
  *error = "no usable NPU EID candidate; CLOS=" + clos_error + "; pair_info=" + pair_info_error;
  return false;
}

inline bool DiscoverNpuLocalEidCandidates(std::uint32_t logical_device_id,
    std::vector<NpuLocalEidCandidate>* candidates, std::string* error) {
  std::uint32_t physical{};
  if (!ResolvePhysicalDeviceId(logical_device_id, &physical)) {
    *error = "cannot resolve physical device from logical device " + std::to_string(logical_device_id);
    return false;
  }
  return DiscoverPhysicalNpuEids(physical, candidates, error);
}

inline bool ParseConfiguredNpuLocalEids(
    std::uint32_t logical_device_id, const std::vector<std::string>& configured,
    std::vector<NpuLocalEidCandidate>* candidates, std::string* error) {
  if (candidates == nullptr || error == nullptr || configured.empty()) return false;
  candidates->clear();
  std::uint32_t physical_device_id = 0U;
  if (!ResolvePhysicalDeviceId(logical_device_id, &physical_device_id)) {
    *error = "cannot resolve physical device from logical device " +
             std::to_string(logical_device_id);
    return false;
  }
  for (std::size_t index = 0U; index < configured.size(); ++index) {
    std::array<std::uint8_t, peer::kEidBytes> eid{};
    if (!ParseEid(configured[index], &eid) ||
        std::all_of(eid.begin(), eid.end(),
                    [](std::uint8_t byte) { return byte == 0U; })) {
      *error = "npu_eids[" + std::to_string(index) +
               "] must be a non-zero 32-digit hexadecimal EID";
      return false;
    }
    if (std::find_if(candidates->begin(), candidates->end(),
                     [&eid](const NpuLocalEidCandidate& candidate) {
                       return candidate.eid == eid;
                     }) != candidates->end()) {
      *error = "npu_eids contains duplicate EID " + configured[index];
      return false;
    }
    AddNpuLocalEidCandidate(physical_device_id, eid,
                            "explicit_npu_eid_" + std::to_string(index),
                            candidates);
  }
  return true;
}

} // namespace kvaf::npu_eid
