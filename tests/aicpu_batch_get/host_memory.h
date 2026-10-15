#pragma once

#include "control.h"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>
#include <linux/mempolicy.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>

namespace kvaf::aicpu_test {
inline std::vector<unsigned> ParseCpuList(const std::string& text) {
  std::vector<unsigned> result;
  std::istringstream items(text); std::string item;
  while (std::getline(items, item, ',')) {
    const auto dash = item.find('-');
    const auto first = Number(item.substr(0, dash).c_str());
    const auto last = dash == std::string::npos ? first : Number(item.substr(dash + 1).c_str());
    Require(first <= last && last < CPU_SETSIZE, "NUMA CPU list exceeds affinity mask capacity");
    for (auto n = first; n <= last; ++n) result.push_back(static_cast<unsigned>(n));
  }
  return result;
}
inline std::string ReadTopology(const std::string& path) {
  std::ifstream f(path); std::string value; std::getline(f, value); return value;
}
// -1 means the endpoint exposes no usable NUMA topology. Auto placement is
// best effort; explicit node selection and binding failures remain strict.
template <class Read>
inline int FindUrmaNumaNode(const std::string& device, Read read) {
  Require(!device.empty() && device.find('/') == std::string::npos && device != "." && device != "..",
          "invalid selected UB device name");
  for (const auto* kind : {"ubcore", "infiniband"}) {
    const auto text = read(std::string("/sys/class/") + kind + "/" + device + "/device/numa_node");
    if (!text.empty() && text != "-1") {
      const auto node = Number(text.c_str());
      Require(node < 4096, "invalid UB NUMA node"); return static_cast<int>(node);
    }
  }
  return -1;
}
inline int UrmaNumaNode(const std::string& device) {
  return FindUrmaNumaNode(device, ReadTopology);
}
struct HostMemoryOptions {
  std::string numa_node{"none"};  // none, best-effort auto (selected UB endpoint), or node id
  int cpu{-1};
};
// Declare before the transport so registered memory outlives transport even
// on exception. Allocate only after the actual UB endpoint has been selected.
class HostMemory {
 public:
  HostMemory() = default;
  HostMemory(const HostMemory&) = delete;
  HostMemory& operator=(const HostMemory&) = delete;
  ~HostMemory() {
    if (data_) munmap(data_, bytes_);
    if (affinity_changed_) sched_setaffinity(0, sizeof(original_), &original_);
  }
  void Allocate(std::uint64_t requested, const HostMemoryOptions& options, const std::string& device) {
    Require(!data_, "host allocation already initialized");
    int node = -1;
    if (options.numa_node == "auto") {
      node = UrmaNumaNode(device);
      if (node < 0)
        std::fprintf(stderr, "[AICPU_HOST_MEMORY] NUMA_AUTO_UNAVAILABLE ub_device=%s "
            "fallback=none memory_policy=inherited reason=no_usable_sysfs_numa_node "
            "explicit_node_option=--host-numa-node\n", device.c_str());
    }
    else if (options.numa_node != "none") {
      const auto n = Number(options.numa_node.c_str());
      Require(n < 4096, "NUMA node exceeds 4095"); node = static_cast<int>(n);
    }
    Require(sched_getaffinity(0, sizeof(original_), &original_) == 0, "cannot read host CPU affinity");
    cpu_set_t selected = original_;
    if (node >= 0) {
      const auto list = ReadTopology("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist");
      Require(!list.empty(), "NUMA node cpulist not found");
      CPU_ZERO(&selected);
      for (auto cpu : ParseCpuList(list)) if (CPU_ISSET(cpu, &original_)) CPU_SET(cpu, &selected);
    }
    if (options.cpu >= 0) {
      Require(options.cpu < CPU_SETSIZE && CPU_ISSET(options.cpu, &selected),
              "--host-cpu is outside allowed CPUs or the selected NUMA node");
      CPU_ZERO(&selected); CPU_SET(options.cpu, &selected);
    }
    Require(CPU_COUNT(&selected) > 0, "no allowed host CPUs on selected NUMA node");
    if (node >= 0 || options.cpu >= 0) {
      Require(sched_setaffinity(0, sizeof(selected), &selected) == 0, "cannot bind host CPU affinity");
      affinity_changed_ = true;
    }
    const auto page = sysconf(_SC_PAGESIZE);
    Require(page > 0, "cannot query host page size");
    const auto alignment = static_cast<std::uint64_t>(page);
    Require(requested > 0 && requested <= kMaxAllocation && requested <= SIZE_MAX - alignment + 1,
            "host allocation exceeds limit");
    bytes_ = (requested + alignment - 1) / alignment * alignment;
    Require(bytes_ <= kMaxAllocation, "rounded host allocation exceeds limit");
    data_ = mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (data_ == MAP_FAILED) {
      data_ = nullptr;
      throw std::runtime_error(std::string("host mmap failed: ") + std::strerror(errno));
    }
    if (node >= 0) {
      constexpr auto bits = sizeof(unsigned long) * 8;
      std::vector<unsigned long> nodes(node / bits + 1);
      nodes[node / bits] |= 1UL << (node % bits);
      // Install before first touch; no migration of registered/in-flight pages.
      const auto bound = syscall(SYS_mbind, data_, bytes_, MPOL_BIND | MPOL_F_STATIC_NODES,
                                 nodes.data(), node + 1UL, 0);
      Require(bound == 0, "NUMA mbind failed: " + std::string(std::strerror(errno)));
    }
    std::memset(data_, 0, bytes_);
    std::ostringstream cpus;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) if (CPU_ISSET(cpu, &selected)) {
      if (cpus.tellp() > 0) cpus << ':';
      cpus << cpu;
    }
    std::fprintf(stderr, "[AICPU_HOST_MEMORY] ub_device=%s numa_request=%s numa_node=%d "
        "allowed_cpus=%s first_touch_cpu=%d bytes=%zu "
        "allocation=after_endpoint_selection affinity_scope=submitter\n",
        device.c_str(), options.numa_node.c_str(), node, cpus.str().c_str(), sched_getcpu(),
        bytes_);
    // Query a small page sample outside timing. Failure (e.g. container policy)
    // is diagnostic, not evidence of local placement.
    void* pages[16]{}; int status[16]{};
    const auto samples = std::min<std::size_t>(16, bytes_ / page);
    for (std::size_t i = 0; i < samples; ++i)
      pages[i] = static_cast<char*>(data_) + (i * (bytes_ / page) / samples) * page;
    const auto query = syscall(SYS_move_pages, 0, samples, pages, nullptr, status, 0);
    if (query < 0) std::fprintf(stderr, "[AICPU_HOST_MEMORY] placement_query=unavailable errno=%d\n", errno);
    else {
      std::fprintf(stderr, "[AICPU_HOST_MEMORY] sampled_page_nodes=");
      for (std::size_t i = 0; i < samples; ++i) std::fprintf(stderr, "%s%d", i ? ":" : "", status[i]);
      std::fprintf(stderr, "\n");
    }
  }
  void* data() const { return data_; }
  std::size_t bytes() const { return bytes_; }
 private:
  void* data_{};
  std::size_t bytes_{};
  cpu_set_t original_{};
  bool affinity_changed_{};
};
}  // namespace kvaf::aicpu_test
