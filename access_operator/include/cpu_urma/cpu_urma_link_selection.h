#pragma once

#include "cpu_urma/cpu_urma_discovery.h"
#include "cpu_urma/cpu_urma_transport.h"
#include "peer_link.h"

#include <cstddef>
#include <cstdio>

namespace kvaf {
// The caller owns allocation, transport and cleanup. A retry is permitted only
// after that caller has cleaned up the failed attempt. This keeps Runtime's
// allocation policy out of standalone clients while sharing route selection.
struct CpuUrmaLinkAttempt {
  Status status;
  bool retry{false};
};

inline CpuUrmaTransportOptions BindCpuUrmaEndpoint(CpuUrmaTransportOptions options,
                                                  const CpuUrmaCandidate& cpu,
                                                  std::uint32_t device_id,
                                                  std::uint32_t npu_candidate) {
  options.ub_device = cpu.device_name;
  options.has_eid_index = true;
  options.eid_index = cpu.eid_index;
  options.has_preferred_eid = true;
  options.preferred_eid = cpu.eid;
  options.remote_hbm = true;
  options.remote_device_id = device_id;
  options.remote_eid_index = npu_candidate;
  return options;
}

// Preserve Runtime's CPU-major, NPU-minor candidate order and NPU device-ID
// lookup through PeerLink. Peer is injectable for host-only protocol tests.
// UINT32_MAX tries every remote candidate; any other index pins only the
// remote side, leaving local CPU candidate filtering to the caller.
template <class Peer, class Candidates, class Attempt>
Status SelectCpuUrmaDeviceLink(Peer& peer, std::uint32_t device_id,
                               const Candidates& cpus, Attempt attempt,
                               std::uint32_t npu_candidate = UINT32_MAX) {
  std::vector<std::array<std::uint8_t, peer::kEidBytes>> npus;
  Status last = peer.GetHbmEidCandidates(device_id, &npus);
  if (!last.ok()) return last;
  if (cpus.empty() || npus.empty()) return Status::NotFound("empty CPU/NPU EID candidate list");
  const bool fixed_npu = npu_candidate != UINT32_MAX;
  if (fixed_npu && npu_candidate >= npus.size()) {
    return Status::InvalidArgument("remote EID candidate index is out of range");
  }
  const std::uint32_t first_npu = fixed_npu ? npu_candidate : 0U;
  const std::size_t end_npu = fixed_npu
      ? static_cast<std::size_t>(npu_candidate) + 1U : npus.size();
  for (const auto& cpu : cpus) {
    for (std::uint32_t index = first_npu; index < end_npu; ++index) {
      std::fprintf(stderr, "[KVAF_CPU] LINK_ATTEMPT npu_device=%u ub_device=%s eid_index=%u npu_candidate=%u\n",
                   device_id, cpu.device_name.c_str(), cpu.eid_index, index);
      const auto result = attempt(cpu, index);
      last = result.status;
      if (last.ok()) {
        std::fprintf(stderr, "[KVAF_CPU] LINK_SELECTED npu_device=%u ub_device=%s eid_index=%u npu_candidate=%u\n",
                     device_id, cpu.device_name.c_str(), cpu.eid_index, index);
        return last;
      }
      std::fprintf(stderr, "[KVAF_CPU] LINK_FAILED reason=%s retry=%d\n", last.message().c_str(), result.retry);
      if (!result.retry) return last;
    }
  }
  return last;
}
}  // namespace kvaf
