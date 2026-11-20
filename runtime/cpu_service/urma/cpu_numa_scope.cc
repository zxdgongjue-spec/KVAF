#include "cpu_service/urma/cpu_numa_scope.h"

#include <sched.h>

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

namespace kvaf {
namespace {
template <typename T>
bool ReadNumber(const std::string& path, T* value) {
  std::ifstream input(path);
  std::string word;
  if (!(input >> word)) return false;
  const bool hex = word.compare(0, 2, "0x") == 0 || word.compare(0, 2, "0X") == 0;
  const char* end = word.data() + word.size();
  auto result = std::from_chars(word.data() + (hex ? 2 : 0), end, *value, hex ? 16 : 10);
  return result.ec == std::errc{} && result.ptr == end;
}

// Linux sysfs CPU and node lists use comma-separated IDs and inclusive ranges.
bool ReadIds(const std::string& path, std::vector<int>* ids) {
  std::ifstream input(path);
  std::string line;
  if (!std::getline(input, line)) return false;
  const char* cursor = line.data();
  const char* end = cursor + line.size();
  while (cursor < end) {
    int first = -1, last = -1;
    auto id = std::from_chars(cursor, end, first);
    if (id.ec != std::errc{} || first < 0) return false;
    cursor = id.ptr;
    last = first;
    if (cursor < end && *cursor == '-') {
      id = std::from_chars(cursor + 1, end, last);
      if (id.ec != std::errc{} || last < first) return false;
      cursor = id.ptr;
    }
    for (int i = first; i <= last; ++i) ids->push_back(i);
    if (cursor < end && *cursor++ != ',') return false;
  }
  return true;
}

int DevicePackage(const std::string& device) {
  // On deployed udmac<package>d<die>e<FE> devices c identifies the Linux CPU package.
  unsigned package, die, fe;
  int consumed = 0;
  if (std::sscanf(device.c_str(), "udmac%ud%ue%u%n", &package, &die, &fe, &consumed) != 3 ||
      static_cast<std::size_t>(consumed) != device.size()) return -1;
  return static_cast<int>(package);
}

// ACPI SRAT GICC affinity records map ARM processor UIDs to firmware
// proximity domains. Linux node IDs can differ from those domain numbers.
std::map<unsigned, unsigned> ReadCpuDomains(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  const std::vector<unsigned char> data{std::istreambuf_iterator<char>(input), {}};
  const auto u32 = [&](std::size_t i) {
    return unsigned(data[i]) | unsigned(data[i + 1]) << 8 |
           unsigned(data[i + 2]) << 16 | unsigned(data[i + 3]) << 24;
  };
  constexpr std::size_t header_size = 48;
  if (data.size() < header_size || !std::equal(data.begin(), data.begin() + 4, "SRAT") ||
      u32(4) != data.size() || (std::accumulate(data.begin(), data.end(), 0u) & 255u)) return {};
  std::map<unsigned, unsigned> domains;
  for (std::size_t pos = header_size; pos < data.size();) {
    if (data.size() - pos < 2) return {};
    const auto length = data[pos + 1];
    if (length < 2 || length > data.size() - pos) return {};
    if (data[pos] == 3 && length >= 18 && (u32(pos + 10) & 1u)) {
      if (!domains.emplace(u32(pos + 6), u32(pos + 2)).second) return {};
    }
    pos += length;
  }
  return domains;
}
}  // namespace

Status CpuNumaScope::Capture(std::string sysfs) {
  sysfs_ = std::move(sysfs);
  const std::string system = sysfs_ + "/devices/system/";
  std::vector<int> possible, online;
  if (!ReadIds(system + "cpu/possible", &possible) || possible.empty() ||
      !ReadIds(system + "cpu/online", &online) || online.empty())
    return Status::Internal("cannot read possible/online CPUs from sysfs");
  const auto cpu_count = possible.back() + 1;
  const auto free_mask = [](cpu_set_t* mask) { CPU_FREE(mask); };
  std::unique_ptr<cpu_set_t, decltype(free_mask)> mask(CPU_ALLOC(cpu_count), free_mask);
  const auto bytes = CPU_ALLOC_SIZE(cpu_count);
  if (!mask || sched_getaffinity(0, bytes, mask.get()) != 0)
    return Status::Internal("cannot read CPU service affinity");
  const auto allowed = [&](int cpu) {
    return cpu < cpu_count && CPU_ISSET_S(cpu, bytes, mask.get());
  };
  unrestricted_ = std::all_of(online.begin(), online.end(), allowed);
  if (unrestricted_) return Status::OK();

  std::vector<int> nodes;
  if (!ReadIds(system + "node/online", &nodes))
    return Status::Internal("cannot map service CPU affinity to NUMA nodes");
  std::set<int> mapped;
  std::set<int> affinity_nodes;
  std::map<int, std::set<int>> package_nodes;
  auto domains = ReadCpuDomains(sysfs_ + "/firmware/acpi/tables/SRAT");
  bool domains_known = !domains.empty();
  for (int node : nodes) {
    std::vector<int> cpus;
    if (!ReadIds(system + "node/node" + std::to_string(node) + "/cpulist", &cpus))
      return Status::Internal("cannot read CPUs for NUMA node " + std::to_string(node));
    for (int cpu : cpus) {
      if (!std::binary_search(online.begin(), online.end(), cpu)) continue;
      mapped.insert(cpu);
      int package = -1;
      const auto cpu_path = system + "cpu/cpu" + std::to_string(cpu);
      if (!ReadNumber(cpu_path + "/topology/physical_package_id", &package) || package < 0)
        return Status::Internal("cannot determine physical package for CPU " + std::to_string(cpu));
      package_nodes[package].insert(node);
      if (allowed(cpu)) {
        affinity_nodes.insert(node);
        packages_.insert(package);
      }
      if (domains_known) {
        unsigned uid;
        if (ReadNumber(cpu_path + "/firmware_node/uid", &uid) && domains.count(uid))
          domain_nodes_[domains.at(uid)].insert(node);
        else
          domains_known = false;
      }
    }
  }
  if (!domains_known) domain_nodes_.clear();
  if (packages_.empty() || mapped.size() != online.size())
    return Status::Internal("incomplete CPU to NUMA topology");
  for (int node : affinity_nodes) std::fprintf(stderr, "[KVAF_CPU] AFFINITY_NUMA_NODE node=%d\n", node);
  for (int package : packages_) {
    std::fprintf(stderr, "[KVAF_CPU] CANDIDATE_CPU_CHIP package=%d\n", package);
    nodes_.insert(package_nodes[package].begin(), package_nodes[package].end());
  }
  // A node shared with an excluded package cannot establish device ownership by itself.
  for (const auto& entry : package_nodes)
    if (!packages_.count(entry.first))
      for (int node : entry.second) nodes_.erase(node);
  return Status::OK();
}

bool CpuNumaScope::Allows(const std::string& device) const {
  if (unrestricted_) return true;
  const auto package = DevicePackage(device);
  if (package >= 0) return packages_.count(package) != 0;
  for (const auto* root : {"/class/ubcore/", "/class/uburma/"}) {
    const auto parent = sysfs_ + root + device + "/device/";
    int node = -1;
    if (ReadNumber(parent + "numa_node", &node) && node >= 0)
      return nodes_.count(node) != 0;
    unsigned domain;
    if (ReadNumber(parent + "numa", &domain)) {
      const auto it = domain_nodes_.find(domain);
      if (it != domain_nodes_.end())
        return std::includes(nodes_.begin(), nodes_.end(), it->second.begin(), it->second.end());
    }
  }
  return false;
}
}  // namespace kvaf
