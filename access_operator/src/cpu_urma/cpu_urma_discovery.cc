#include "cpu_urma/cpu_urma_discovery.h"

#include <urma_api.h>
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <memory>
#include <tuple>

namespace kvaf {
CpuUrmaDiscovery::~CpuUrmaDiscovery() { Shutdown(); }

Status CpuUrmaDiscovery::Initialize(const std::string& requested_device,
                                   bool has_eid_index, std::uint32_t eid_index) {
  if (initialized_) return Status::InvalidState("CPU URMA discovery already initialized");
  urma_init_attr_t init{};
  const int rc = urma_init(&init);
  if (rc != URMA_SUCCESS && rc != EEXIST && rc != -EEXIST)
    return Status::Internal("urma_init failed (rc=" + std::to_string(rc) + ")");
  initialized_ = true;
  owns_urma_ = rc == URMA_SUCCESS;
  int device_count = 0;
  std::unique_ptr<urma_device_t*, decltype(&urma_free_device_list)> devices(
      urma_get_device_list(&device_count), urma_free_device_list);
  if (!devices) return Status::NotFound("urma_get_device_list returned no devices");
  for (int i = 0; i < device_count; ++i) {
    auto* device = devices.get()[i];
    if (!device || device->type != URMA_TRANSPORT_UB ||
        (!requested_device.empty() && requested_device != device->name)) continue;
    std::uint32_t count = 0;
    std::unique_ptr<urma_eid_info_t, decltype(&urma_free_eid_list)> eids(
        urma_get_eid_list(device, &count), urma_free_eid_list);
    if (!eids) {
      std::fprintf(stderr, "[KVAF_CPU] CANDIDATE_ENUM_FAILED ub_device=%s errno=%d\n", device->name, errno);
      continue;
    }
    for (std::uint32_t j = 0; j < count; ++j) {
      const auto& info = eids.get()[j];
      if ((has_eid_index && info.eid_index != eid_index) ||
          std::all_of(info.eid.raw, info.eid.raw + 16, [](auto b) { return b == 0; })) continue;
      CpuUrmaCandidate candidate{device->name, info.eid_index, {}};
      std::copy_n(info.eid.raw, candidate.eid.size(), candidate.eid.begin());
      candidates_.push_back(candidate);
    }
  }
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
