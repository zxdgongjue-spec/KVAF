#pragma once

#include <array>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "cpu_service/urma/cpu_numa_scope.h"

namespace kvaf {
struct CpuUrmaCandidate {
  std::string device_name;
  std::uint32_t eid_index{0};
  std::array<std::uint8_t, 16> eid{};
};

// One immutable candidate list per CPU service, shared by discovery and
// allocation-scoped context pools. The first successful URMA query publishes it.
class CpuUrmaDiscovery {
 public:
  ~CpuUrmaDiscovery();
  CpuUrmaDiscovery(std::string device, bool has_eid_index, std::uint32_t eid_index);
  CpuUrmaDiscovery(const CpuUrmaDiscovery&) = delete;
  CpuUrmaDiscovery& operator=(const CpuUrmaDiscovery&) = delete;

  Status Initialize();
  const std::vector<CpuUrmaCandidate>& candidates() const { return candidates_; }

 private:
  bool initialized_{false};
  std::mutex mutex_;
  CpuNumaScope scope_;
  Status scope_status_;
  std::string device_;
  bool has_eid_index_;
  std::uint32_t eid_index_;
  std::vector<CpuUrmaCandidate> candidates_;
};
}  // namespace kvaf
