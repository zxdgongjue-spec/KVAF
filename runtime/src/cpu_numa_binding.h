#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "status.h"

namespace kvaf::cpu_numa {

struct Binding {
  int package{-1};
  bool strict{false};
  std::string source;
  std::vector<int> nodes;
  std::vector<int> cpus;
  int possible_cpus{0};
  int possible_nodes{0};
};

// Prefer the device's local node set unless an explicit node overrides it.
Status Resolve(const std::string& device, std::optional<int> requested_node, Binding& binding);

// A dedicated mapping keeps its NUMA policy independent of other sessions and
// malloc's reusable heap. Release only after unregistering the UB segment.
struct Memory {
  Memory() = default;
  ~Memory();
  Memory(const Memory&) = delete;
  Memory& operator=(const Memory&) = delete;
  void* data{nullptr};
  void* mapping{nullptr};
  std::size_t mapping_bytes{0};
};

// Use MPOL_PREFERRED_MANY normally, MPOL_BIND for an explicit node. Default placement
// temporarily binds the touching thread to device-local CPUs; explicit-node
// placement leaves CPU affinity alone. Neither changes the thread memory policy.
Status Allocate(Binding& binding, std::size_t bytes, bool huge_pages,
                std::unique_ptr<Memory>& memory);
std::string FormatIds(const std::vector<int>& ids);

}  // namespace kvaf::cpu_numa
