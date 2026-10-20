#include "cpu_urma_discovery.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <memory>
#include <tuple>

#include "urma_transport.h"

namespace kvaf {
CpuUrmaDiscovery::~CpuUrmaDiscovery() { Shutdown(); }

Status CpuUrmaDiscovery::Initialize(const std::string& requested_device, bool has_eid_index,
                                    std::uint32_t eid_index) {
  if (initialized_) return Status::InvalidState("CPU URMA discovery already initialized");
  urma_init_attr_t init{};
  const int rc = urma_init(&init);
  if (rc != URMA_SUCCESS && rc != EEXIST && rc != -EEXIST)
    return Status::Internal("urma_init failed (rc=" + std::to_string(rc) + ")");
  initialized_ = true;
  owns_urma_ = rc == URMA_SUCCESS;
  const bool enumerated = urma::EnumerateEndpoints(
      urma::NativeApi{}, requested_device.c_str(), has_eid_index ? eid_index : UINT32_MAX,
      [&](auto* device, const auto& info) {
        CpuUrmaCandidate candidate{device->name, info.eid_index, {}};
        std::copy_n(info.eid.raw, candidate.eid.size(), candidate.eid.begin());
        candidates_.push_back(candidate);
        return false;
      });
  if (!enumerated) return Status::NotFound("urma_get_device_list returned no devices");
  std::sort(candidates_.begin(), candidates_.end(), [](const auto& a, const auto& b) {
    return std::tie(a.device_name, a.eid_index) < std::tie(b.device_name, b.eid_index);
  });
  if (candidates_.empty()) return Status::NotFound("no usable CPU UB/EID candidates");
  return Status::OK();
}

void CpuUrmaDiscovery::Shutdown() {
  candidates_.clear();
  if (initialized_ && owns_urma_) (void)urma_uninit();
  owns_urma_ = false;
  initialized_ = false;
}
}  // namespace kvaf
