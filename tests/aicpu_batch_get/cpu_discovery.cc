#include "cpu_discovery.h"
#include "control.h"

#include <urma_api.h>
#include <algorithm>
#include <memory>
#include <tuple>

namespace kvaf::aicpu_test {
UrmaLifetime::UrmaLifetime() {
  urma_init_attr_t attr{};
  const int rc = urma_init(&attr);
  Require(rc == URMA_SUCCESS || rc == EEXIST || rc == -EEXIST,
          "urma_init failed: " + std::to_string(rc));
  owned_ = rc == URMA_SUCCESS;
}
UrmaLifetime::~UrmaLifetime() { if (owned_) (void)urma_uninit(); }

std::vector<CpuCandidate> DiscoverCpuCandidates(const CpuUrmaTransportOptions& filter) {
  int count = 0;
  std::unique_ptr<urma_device_t*, decltype(&urma_free_device_list)> devices(
      urma_get_device_list(&count), urma_free_device_list);
  Require(devices != nullptr, "cannot enumerate CPU URMA devices");
  std::vector<CpuCandidate> candidates;
  for (int i = 0; i < count; ++i) {
    auto* device = devices.get()[i];
    if (!device || device->type != URMA_TRANSPORT_UB ||
        (!filter.ub_device.empty() && filter.ub_device != device->name)) continue;
    std::uint32_t eid_count = 0;
    std::unique_ptr<urma_eid_info_t, decltype(&urma_free_eid_list)> eids(
        urma_get_eid_list(device, &eid_count), urma_free_eid_list);
    if (!eids) continue;
    for (std::uint32_t j = 0; j < eid_count; ++j) {
      const auto& entry = eids.get()[j];
      if ((filter.has_eid_index && entry.eid_index != filter.eid_index) ||
          std::all_of(entry.eid.raw, entry.eid.raw + 16, [](auto b) { return b == 0; })) continue;
      CpuCandidate candidate{device->name, entry.eid_index, {}};
      std::copy_n(entry.eid.raw, candidate.eid.size(), candidate.eid.begin());
      candidates.push_back(candidate);
    }
  }
  std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
    return std::tie(a.device, a.eid_index) < std::tie(b.device, b.eid_index);
  });
  Require(!candidates.empty(), "no usable CPU UB/EID candidates");
  return candidates;
}
}  // namespace kvaf::aicpu_test
