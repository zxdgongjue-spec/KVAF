#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "cpu_numa_binding.h"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <utility>

#include <linux/mempolicy.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace kvaf::cpu_numa {
namespace {

bool Number(const std::string& text, int* value) {
  const auto result = std::from_chars(text.data(), text.data() + text.size(), *value);
  return result.ec == std::errc{} && result.ptr == text.data() + text.size() && *value >= 0;
}

bool ReadNumber(const std::string& path, int* value) {
  std::ifstream input(path);
  std::string text;
  return static_cast<bool>(input >> text) && Number(text, value);
}

Status ReadIds(const std::string& path, std::vector<int>* ids) {
  std::ifstream input(path);
  if (!input) return Status::NotFound("cannot read NUMA topology: " + path);
  std::string line;
  std::getline(input, line);
  std::istringstream parts(line);
  std::string part;
  ids->clear();
  while (std::getline(parts, part, ',')) {
    const auto dash = part.find('-');
    int first = 0, last = 0;
    if (!Number(part.substr(0, dash), &first) ||
        (dash != std::string::npos && !Number(part.substr(dash + 1), &last)))
      return Status::InvalidArgument("invalid CPU/node list: " + path);
    if (dash == std::string::npos) last = first;
    if (last < first)
      return Status::InvalidArgument("invalid CPU/node range: " + path);
    for (int id = first; id <= last; ++id) ids->push_back(id);
  }
  return Status::OK();
}

bool Contains(const std::vector<int>& ids, int value) {
  return std::binary_search(ids.begin(), ids.end(), value);
}

struct Node {
  int id;
  int package{-1};
  std::vector<int> cpus;
};

}  // namespace

std::string FormatIds(const std::vector<int>& ids) {
  std::string text;
  for (const int id : ids) {
    if (!text.empty()) text += ',';
    text += std::to_string(id);
  }
  return text;
}

Status Resolve(const std::string& device, std::optional<int> requested_node, Binding& binding) {
  const std::string system = "/sys/devices/system/";
  std::vector<int> possible_cpus, possible_nodes, online_cpus, online_nodes, memory_nodes;
  for (const auto& item : {
           std::make_pair(system + "cpu/possible", &possible_cpus),
           std::make_pair(system + "node/possible", &possible_nodes),
           std::make_pair(system + "cpu/online", &online_cpus),
           std::make_pair(system + "node/online", &online_nodes),
           std::make_pair(system + "node/has_memory", &memory_nodes)}) {
    const auto status = ReadIds(item.first, item.second);
    if (!status.ok()) return status;
    if (item.second->empty()) return Status::NotFound("empty NUMA topology: " + item.first);
  }
  binding.possible_cpus = possible_cpus.back() + 1;
  binding.possible_nodes = possible_nodes.back() + 1;
  if (requested_node.has_value()) {
    if (!Contains(online_nodes, *requested_node) || !Contains(memory_nodes, *requested_node))
      return Status::InvalidArgument("requested NUMA node is not online with memory: " +
                                     std::to_string(*requested_node));
    binding.node = *requested_node;
    binding.nodes = {*requested_node};
    binding.strict = true;
    binding.source = "command_line";
    return Status::OK();
  }
  std::vector<Node> nodes;
  for (const int id : online_nodes) {
    Node node{id, -1, {}};
    std::vector<int> cpus;
    auto status = ReadIds(system + "node/node" + std::to_string(id) + "/cpulist", &cpus);
    if (!status.ok()) return status;
    for (const int cpu : cpus)
      if (Contains(online_cpus, cpu)) node.cpus.push_back(cpu);
    if (node.cpus.empty()) continue;
    // A CPU-local NUMA node belongs to one package; one online core identifies it.
    const auto path = system + "cpu/cpu" + std::to_string(node.cpus.front()) +
                      "/topology/physical_package_id";
    if (!ReadNumber(path, &node.package)) return Status::NotFound("cannot read CPU package: " + path);
    nodes.push_back(std::move(node));
  }

  // On the deployed udmac devices, cN identifies Linux CPU package N.
  // The UB device/numa attribute is a firmware proximity domain (PXM), so it
  // must not be used directly as a Linux node ID.
  if (device.compare(0, 5, "udmac") != 0)
    return Status::NotFound("cannot resolve CPU package from device name: " + device);
  const char* end = device.data() + device.size();
  const auto parsed = std::from_chars(device.data() + 5, end, binding.package);
  if (parsed.ec != std::errc{} || parsed.ptr == end || *parsed.ptr != 'd' || binding.package < 0)
    return Status::InvalidArgument("cannot read CPU package from device name: " + device);
  binding.source = "device_name_c_package";
  for (const auto& node : nodes) {
    if (node.package != binding.package) continue;
    if (Contains(memory_nodes, node.id)) binding.nodes.push_back(node.id);
    binding.cpus.insert(binding.cpus.end(), node.cpus.begin(), node.cpus.end());
  }
  std::sort(binding.cpus.begin(), binding.cpus.end());
  if (binding.nodes.empty() || binding.cpus.empty())
    return Status::NotFound("no online CPUs and memory nodes for device CPU package " +
                            std::to_string(binding.package));
  return Status::OK();
}

Memory::~Memory() {
  if (mapping != nullptr) (void)munmap(mapping, mapping_bytes);
}

Status Allocate(Binding& binding, std::size_t bytes, bool huge_pages,
                std::unique_ptr<Memory>& memory) {
  const auto error = [](const char* operation) {
    return Status::Internal(std::string(operation) + ": " + std::strerror(errno));
  };
  const auto mask_bytes = CPU_ALLOC_SIZE(binding.possible_cpus);
  const auto free_mask = [](cpu_set_t* mask) { CPU_FREE(mask); };
  std::unique_ptr<cpu_set_t, decltype(free_mask)> allowed(CPU_ALLOC(binding.possible_cpus), free_mask);
  std::unique_ptr<cpu_set_t, decltype(free_mask)> selected(CPU_ALLOC(binding.possible_cpus), free_mask);
  if (!allowed || !selected) return Status::OutOfMemory("cannot allocate CPU affinity mask");
  if (sched_getaffinity(0, mask_bytes, allowed.get()) != 0) return error("sched_getaffinity");
  CPU_ZERO_S(mask_bytes, selected.get());
  std::vector<int> cpus;
  for (int cpu = 0; cpu < binding.possible_cpus; ++cpu) {
    if (CPU_ISSET_S(cpu, mask_bytes, allowed.get()) &&
        (binding.strict || Contains(binding.cpus, cpu))) {
      CPU_SET_S(cpu, mask_bytes, selected.get());
      cpus.push_back(cpu);
    }
  }
  if (cpus.empty())
    return Status::InvalidArgument("device CPU package has no CPUs in inherited affinity; check numactl/taskset/cpuset");

  constexpr auto word_bits = sizeof(unsigned long) * 8U;
  const auto words = (binding.possible_nodes + word_bits - 1U) / word_bits;
  std::vector<unsigned long> allowed_nodes(words, 0UL), selected_nodes(words, 0UL);
  const auto maxnode = words * word_bits;
  if (syscall(SYS_get_mempolicy, nullptr, allowed_nodes.data(), maxnode, nullptr,
              MPOL_F_MEMS_ALLOWED) != 0) return error("get_mempolicy(MEMS_ALLOWED)");
  std::vector<int> nodes;
  for (const int node : binding.nodes) {
    const auto bit = 1UL << (node % word_bits);
    if (allowed_nodes[node / word_bits] & bit) {
      nodes.push_back(node);
    }
  }
  if (nodes.empty())
    return Status::InvalidArgument("target NUMA nodes are not allowed by cpuset");
  // Select a deterministic preferred node within the device's package. With
  // an explicit node, this list has exactly that one entry. The kernel handles
  // preferred-node fallback; MPOL_BIND disables it.
  binding.node = nodes.front();
  selected_nodes[binding.node / word_bits] |= 1UL << (binding.node % word_bits);

  const long page_size_result = sysconf(_SC_PAGESIZE);
  if (page_size_result <= 0) return Status::Internal("cannot determine system page size");
  const auto page_size = static_cast<std::size_t>(page_size_result);
  const auto alignment = huge_pages ? std::max<std::size_t>(page_size, 2U * 1024U * 1024U) : page_size;
  if (bytes > std::numeric_limits<std::size_t>::max() - (page_size - 1U) - alignment)
    return Status::InvalidArgument("NUMA allocation size overflow");
  auto allocation = std::make_unique<Memory>();
  const auto bound_bytes = (bytes + page_size - 1U) / page_size * page_size;
  // Keep the alignment padding as untouched virtual space in the owned mapping.
  allocation->mapping_bytes = bound_bytes + alignment;
  void* mapping = mmap(nullptr, allocation->mapping_bytes, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (mapping == MAP_FAILED) return error("mmap(NUMA region)");
  allocation->mapping = mapping;
  const auto address = reinterpret_cast<std::uintptr_t>(mapping);
  allocation->data = reinterpret_cast<void*>((address + alignment - 1U) / alignment * alignment);
  const int policy = binding.strict ? MPOL_BIND : MPOL_PREFERRED;
  if (syscall(SYS_mbind, allocation->data, bound_bytes, policy,
              selected_nodes.data(), maxnode, 0UL) != 0) return error("mbind");
  if (huge_pages && madvise(allocation->data, bound_bytes, MADV_HUGEPAGE) != 0)
    return error("madvise(MADV_HUGEPAGE)");

  // Nothing between binding and restoring can throw or return early. Even
  // default-page regions are touched here, before the UB driver sees them.
  if (!binding.strict && sched_setaffinity(0, mask_bytes, selected.get()) != 0)
    return error("sched_setaffinity");
  std::memset(allocation->data, 0, bytes);
  if (!binding.strict && sched_setaffinity(0, mask_bytes, allowed.get()) != 0)
    return error("restore sched_setaffinity after NUMA allocation");
  binding.cpus = std::move(cpus);
  memory = std::move(allocation);
  return Status::OK();
}

}  // namespace kvaf::cpu_numa
