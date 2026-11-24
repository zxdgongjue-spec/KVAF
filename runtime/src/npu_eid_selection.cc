#include "npu_eid_selection.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <string_view>
#include <sys/file.h>
#include <unistd.h>

namespace kvaf::detail {
std::string EidToHex(const NpuEid& eid) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string text;
  text.reserve(eid.size() * 2U);
  for (std::uint8_t byte : eid) {
    text.push_back(kHex[byte >> 4U]);
    text.push_back(kHex[byte & 0x0fU]);
  }
  return text;
}

namespace {
constexpr char kUbDevIdPath[] = "/proc/asdrv_ub/dev_id";
constexpr char kUbPairInfoPath[] = "/proc/asdrv_ub/pair_info";
constexpr char kUbSelectorLockPath[] = "/tmp/kvaf_npu_runtime_asdrv_ub.lock";

bool ParseJsonUnsigned(const std::string& line, const char* field, std::uint32_t* value) {
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

bool ParseJsonString(const std::string& line, const char* field, std::string* value) {
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

bool ParseEid(const std::string& text, NpuEid* eid) {
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
  return true;
}

bool ParsePairInfoEid(const std::string& text, NpuEid* eid) {
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

std::string Trim(std::string text) {
  const std::size_t first = text.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return {};
  const std::size_t last = text.find_last_not_of(" \t\r\n");
  return text.substr(first, last - first + 1U);
}

std::optional<std::string_view> PairInfoField(std::string_view line, std::string_view key) {
  constexpr std::string_view whitespace = " \t\r\n\f\v";
  constexpr std::string_view boundaries = "(),; \t\r\n\f\v";
  for (auto pos = line.find(key); pos != line.npos; pos = line.find(key, pos + key.size())) {
    if (pos != 0 && boundaries.find(line[pos - 1]) == boundaries.npos) continue;
    const auto equal = line.find_first_not_of(whitespace, pos + key.size());
    if (equal == line.npos || line[equal] != '=') continue;
    const auto first = line.find_first_not_of(whitespace, equal + 1);
    if (first == line.npos) return std::string_view{};
    return line.substr(first, line.find_first_of(boundaries, first) - first);
  }
  return std::nullopt;
}

bool ReadHcclClosEids(const std::string& path, std::uint32_t physical_device_id,
                      std::vector<NpuEid>* eids,
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
    NpuEid eid{};
    if (!ParseEid(eid_text, &eid) || eid == NpuEid{}) {
      continue;
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

}  // namespace

PairEids ParsePairInfo(std::istream& input, std::uint32_t physical_device) {
  const std::string device_id = std::to_string(physical_device);
  PairEids pair;
  bool selected = false, active = false;
  unsigned active_channels = 0, local_fields = 0, remote_fields = 0;
  std::string line;
  while (std::getline(input, line)) {
    if (line.find("Dev id info.") != std::string::npos) {
      selected = PairInfoField(line, "dev_id") == device_id;
      active = false;
    } else if (selected && PairInfoField(line, "chan_id")) {
      active = PairInfoField(line, "flag") == "1";
      if (active) ++active_channels;
    } else if (active) {
      const auto colon = line.find(':');
      if (colon == std::string::npos) continue;
      const auto key = Trim(line.substr(0, colon));
      if (key != "local_eid" && key != "remote_eid") continue;
      const bool cpu = key == "local_eid";
      ++(cpu ? local_fields : remote_fields);
      NpuEid eid{};
      if (ParsePairInfoEid(Trim(line.substr(colon + 1)), &eid) && eid != NpuEid{})
        (cpu ? pair.cpu : pair.npu) = eid;
    }
  }
  if (input.bad() || active_channels != 1) return {};
  if (local_fields != 1) pair.cpu.reset();
  if (remote_fields != 1) pair.npu.reset();
  return pair;
}

PairEids ReadDevicePair(std::uint32_t physical_device) {
  PairEids pair;
  const int fd = open(kUbSelectorLockPath, O_CREAT | O_RDWR, 0600);
  if (fd >= 0 && flock(fd, LOCK_EX) == 0) {
    std::ofstream selector(kUbDevIdPath);
    selector << physical_device << '\n';
    selector.close();
    if (selector) {
      std::ifstream input(kUbPairInfoPath);
      pair = ParsePairInfo(input, physical_device);
    }
  }
  if (fd >= 0) close(fd);  // Releases the selector lock.
  std::fprintf(stderr, "[KVAF] PAIR_INFO device=%u cpu=%s npu=%s\n", physical_device,
               pair.cpu ? EidToHex(*pair.cpu).c_str() : "unavailable",
               pair.npu ? EidToHex(*pair.npu).c_str() : "unavailable");
  return pair;
}

Status SelectNpuEids(std::uint32_t physical_device, const std::vector<std::string>& configured,
                     const PairEids& pair, std::vector<NpuEid>* selected) {
  selected->clear();
  const char* source = "explicit";
  if (!configured.empty()) {
    for (const auto& text : configured) {
      NpuEid eid{};
      if (!ParseEid(text, &eid) || eid == NpuEid{})
        return Status::InvalidArgument("npu_eids must contain non-zero 32-digit hexadecimal EIDs");
      if (std::find(selected->begin(), selected->end(), eid) != selected->end())
        return Status::InvalidArgument("npu_eids contains duplicate EID " + text);
      selected->push_back(eid);
    }
  } else if (pair.npu) {
    source = "pair_info.remote_eid";
    selected->push_back(*pair.npu);
  } else {
    source = "hccl_clos";
    const char* path = std::getenv("KVAF_HCCL_ROOTINFO_PATH");
    std::vector<NpuEid> clos;
    std::string error;
    if (!ReadHcclClosEids(path && *path ? path : "/etc/hccl_rootinfo.json", physical_device,
                          &clos, &error))
      return Status::NotFound(error);
    selected->push_back(clos.front());
  }
  for (const auto& eid : *selected)
    std::fprintf(stderr, "[KVAF] PEER_LINK_NPU_EID_DISCOVERED physical_device=%u source=%s eid=%s\n",
                 physical_device, source, EidToHex(eid).c_str());
  return Status::OK();
}
}  // namespace kvaf::detail
