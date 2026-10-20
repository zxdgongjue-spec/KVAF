#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "status.h"

namespace kvaf {
struct CpuUrmaCandidate {
  std::string device_name;
  std::uint32_t eid_index{0};
  std::array<std::uint8_t, 16> eid{};
};

// Owns process-wide URMA initialization only when this instance initialized
// it. Keep it alive until all candidate resources/transports are destroyed.
// Shared by CpuRuntime's resource pool and standalone transport clients.
class CpuUrmaDiscovery {
 public:
  ~CpuUrmaDiscovery();
  CpuUrmaDiscovery() = default;
  CpuUrmaDiscovery(const CpuUrmaDiscovery&) = delete;
  CpuUrmaDiscovery& operator=(const CpuUrmaDiscovery&) = delete;

  Status Initialize(const std::string& device = {}, bool has_eid_index = false,
                    std::uint32_t eid_index = 0);
  void Shutdown();
  const std::vector<CpuUrmaCandidate>& candidates() const { return candidates_; }

 private:
  bool initialized_{false};
  bool owns_urma_{false};
  std::vector<CpuUrmaCandidate> candidates_;
};
}  // namespace kvaf
