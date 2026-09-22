#pragma once

#include "cpu_urma/cpu_urma_transport.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace kvaf::aicpu_test {
struct CpuCandidate {
  std::string device;
  std::uint32_t eid_index;
  std::array<std::uint8_t, 16> eid;
};
// Keep URMA alive while individual setup attempts are destroyed.
class UrmaLifetime {
 public:
  UrmaLifetime();
  ~UrmaLifetime();
  UrmaLifetime(const UrmaLifetime&) = delete;
  UrmaLifetime& operator=(const UrmaLifetime&) = delete;
 private:
  bool owned_{};
};
std::vector<CpuCandidate> DiscoverCpuCandidates(const CpuUrmaTransportOptions& filter);

// Retry only channel setup (no data WR has been posted). Failed validation
// after successful setup is fatal: do not recycle a target after uncertain DMA.
template <class Transport, class Reset>
Status SelectCpuTransport(CpuUrmaTransportOptions options,
                          const std::vector<CpuCandidate>& candidates,
                          Transport& transport, Reset reset) {
  Status last = Status::NotFound("no matching CPU UB/EID candidates");
  for (const auto& candidate : candidates) {
    options.ub_device = candidate.device;
    options.has_eid_index = true; options.eid_index = candidate.eid_index;
    options.has_preferred_eid = true; options.preferred_eid = candidate.eid;
    std::fprintf(stderr, "AICPU_GET_CPU LINK_ATTEMPT npu_device=%u ub_device=%s eid_index=%u\n",
                 options.remote_device_id, candidate.device.c_str(), candidate.eid_index);
    last = transport.Initialize(options);
    if (last.ok()) {
      last = transport.ValidateLink();
      if (!last.ok()) return last;
      std::fprintf(stderr, "AICPU_GET_CPU LINK_SELECTED npu_device=%u ub_device=%s eid_index=%u\n",
                   options.remote_device_id, candidate.device.c_str(), candidate.eid_index);
      return last;
    }
    std::fprintf(stderr, "AICPU_GET_CPU LINK_FAILED %s\n", last.message().c_str());
    const auto stopped = transport.Shutdown();
    if (!stopped.ok()) return stopped;
    const auto cleaned = reset();
    if (!cleaned.ok()) return cleaned;
  }
  return last;
}
}  // namespace kvaf::aicpu_test
