#include "cpu_urma/cpu_urma_link_selection.h"

#include <iostream>
#include <stdexcept>
#include <utility>

namespace {
using namespace kvaf;
void Check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
struct Peer {
  std::uint32_t requested{UINT32_MAX};
  bool fail{false};
  bool empty{false};
  Status GetHbmEidCandidates(std::uint32_t device,
                             std::vector<std::array<std::uint8_t, 16>>* out) {
    requested = device;
    if (fail) return Status::NotFound("unknown NPU device");
    if (!empty) {
      out->resize(2);
      (*out)[0][15] = 11; (*out)[1][15] = 12;
    }
    return Status::OK();
  }
};
std::vector<CpuUrmaCandidate> Candidates() {
  return {{"ub0", 2, {}}, {"ub1", 7, {}}};
}
void PairOrderAndReuse() {
  const auto cpus = Candidates();
  Peer peer;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> attempted;
  bool previous_cleaned = true;
  const auto status = SelectCpuUrmaDeviceLink(peer, 4, cpus,
      [&](const CpuUrmaCandidate& cpu, std::uint32_t npu) -> CpuUrmaLinkAttempt {
    Check(previous_cleaned, "next pair attempted before caller cleanup");
    attempted.emplace_back(cpu.eid_index, npu);
    if (cpu.eid_index == 7 && npu == 0) return {Status::OK(), false};
    previous_cleaned = false;
    // Runtime or standalone client cleans its resources before permitting retry.
    previous_cleaned = true;
    return {Status::Internal("no route"), true};
  });
  Check(status.ok() && peer.requested == 4, "device-ID discovery not used");
  Check(attempted == std::vector<std::pair<std::uint32_t, std::uint32_t>>{{2, 0}, {2, 1}, {7, 0}},
        "CPU-major/NPU-minor order changed");

  CpuUrmaTransportOptions base;
  base.peer_address = "npu:29991"; base.remote_sgl_size = 16; base.jetty_count = 4;
  auto cpu = cpus[1]; cpu.eid[15] = 23;
  const auto selected = BindCpuUrmaEndpoint(base, cpu, 4, 1);
  Check(selected.ub_device == "ub1" && selected.has_eid_index && selected.eid_index == 7 &&
        selected.has_preferred_eid && selected.preferred_eid == cpu.eid &&
        selected.remote_device_id == 4 && selected.remote_eid_index == 1 && selected.remote_hbm,
        "CPU EID and NPU device/candidate IDs were conflated");
  Check(selected.peer_address == base.peer_address && selected.remote_sgl_size == 16 &&
        selected.jetty_count == 4, "binding destroyed caller transport configuration");
}
void StopAndFailure() {
  Peer peer;
  int attempts = 0;
  const auto no_retry = [&](const CpuUrmaCandidate&, std::uint32_t) -> CpuUrmaLinkAttempt {
    ++attempts; return {Status::Internal("cleanup failed or uncertain DMA"), false};
  };
  Check(!SelectCpuUrmaDeviceLink(peer, 4, Candidates(), no_retry).ok() && attempts == 1,
        "fatal attempt should stop selection");
  attempts = 0; peer.fail = true;
  Check(!SelectCpuUrmaDeviceLink(peer, 4, Candidates(), no_retry).ok() && attempts == 0,
        "attempted transport after failed device lookup");
  peer.fail = false; peer.empty = true;
  Check(!SelectCpuUrmaDeviceLink(peer, 4, Candidates(), no_retry).ok() && attempts == 0,
        "empty NPU candidate list accepted");
  peer.empty = false;
  Check(!SelectCpuUrmaDeviceLink(peer, 4, std::vector<CpuUrmaCandidate>{}, no_retry).ok(),
        "empty CPU candidate list accepted");
  const auto all_fail = [&](const CpuUrmaCandidate&, std::uint32_t) -> CpuUrmaLinkAttempt {
    ++attempts; return {Status::Internal("attempt " + std::to_string(attempts)), true};
  };
  const auto status = SelectCpuUrmaDeviceLink(peer, 4, Candidates(), all_fail);
  Check(!status.ok() && attempts == 4 && status.message() == "attempt 4", "last error was lost");
}
void FixedRemoteCandidate() {
  Peer peer;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> attempted;
  const auto attempt = [&](const CpuUrmaCandidate& cpu, std::uint32_t npu) -> CpuUrmaLinkAttempt {
    attempted.emplace_back(cpu.eid_index, npu);
    return {cpu.eid_index == 7 ? Status::OK() : Status::Internal("no route"), true};
  };
  Check(SelectCpuUrmaDeviceLink(peer, 6, Candidates(), attempt, 1).ok() &&
        attempted == std::vector<std::pair<std::uint32_t, std::uint32_t>>{{2, 1}, {7, 1}},
        "fixed remote candidate changed local selection or tried another remote EID");
  attempted.clear();
  Check(!SelectCpuUrmaDeviceLink(peer, 6, Candidates(), attempt, 2).ok() && attempted.empty(),
        "out-of-range remote candidate must fail before allocation");
}
}  // namespace
int main() {
  try { PairOrderAndReuse(); StopAndFailure(); FixedRemoteCandidate(); std::cout << "CPU URMA link selection: PASS\n"; return 0; }
  catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
