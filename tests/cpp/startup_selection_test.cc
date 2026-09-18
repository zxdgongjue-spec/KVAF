#include "startup_selection.h"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <tuple>

using kvaf::Status;
using kvaf::detail::SelectStartupDeviceCandidate;
using kvaf::detail::ShouldTuneStartupCandidates;
using kvaf::detail::StartupCandidate;

namespace {

void Check(bool condition) {
  if (!condition) {
    std::cerr << "startup selection expectation failed\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  Check(ShouldTuneStartupCandidates(true, true));
  Check(!ShouldTuneStartupCandidates(true, false));
  Check(!ShouldTuneStartupCandidates(false, true));

  std::vector<std::tuple<unsigned, std::size_t, bool>> calls;
  bool allocated = false;
  int releases = 0;
  auto cleanup = [&]() {
    Check(allocated);
    allocated = false;
    ++releases;
    return Status::OK();
  };

  // Each device contributes only its first usable EID/N-EID pair. Invalid
  // timings are rejected, so the next pair for that device is tried.
  const std::vector<std::vector<StartupCandidate>> devices{
      {{0, 0}, {0, 1}}, {{1, 0}, {1, 1}}, {{2, 0}, {2, 1}}};
  auto initialize = [&](const StartupCandidate& candidate, bool measure,
                        double* cycles) {
    Check(!allocated);
    allocated = true;
    calls.emplace_back(candidate.peer, candidate.local_group, measure);
    if (candidate.peer == 0 && candidate.local_group == 0) {
      return Status::Internal("unreachable");
    }
    if (candidate.peer == 2 && candidate.local_group == 0) {
      *cycles = std::numeric_limits<double>::quiet_NaN();
      return Status::OK();
    }
    *cycles = candidate.peer == 1 ? 20.0 : 30.0;
    return Status::OK();
  };
  Check(SelectStartupDeviceCandidate(devices, initialize, cleanup).ok());
  Check(calls.size() == 6 && releases == 5 && allocated);
  Check(calls[0] == std::make_tuple(0U, std::size_t{0}, true));
  Check(calls[1] == std::make_tuple(0U, std::size_t{1}, true));
  Check(calls[2] == std::make_tuple(1U, std::size_t{0}, true));
  Check(calls[3] == std::make_tuple(2U, std::size_t{0}, true));
  Check(calls[4] == std::make_tuple(2U, std::size_t{1}, true));
  Check(calls[5] == std::make_tuple(1U, std::size_t{0}, false));
  cleanup();

  // If the fastest device cannot be rebuilt with the formal region, use the
  // next measured device and leave no temporary allocation behind.
  calls.clear();
  releases = 0;
  const std::vector<std::vector<StartupCandidate>> rebuild_devices{
      {{0, 0}}, {{1, 0}}};
  auto rebuild_failure = [&](const StartupCandidate& candidate, bool measure,
                             double* cycles) {
    Check(!allocated);
    allocated = true;
    calls.emplace_back(candidate.peer, candidate.local_group, measure);
    *cycles = 10.0 + candidate.peer;
    if (!measure && candidate.peer == 0) {
      return Status::Internal("rebuild failed");
    }
    return Status::OK();
  };
  Check(SelectStartupDeviceCandidate(rebuild_devices, rebuild_failure, cleanup).ok());
  Check(calls.size() == 4 && releases == 3 && allocated);
  Check(calls.back() == std::make_tuple(1U, std::size_t{0}, false));
  cleanup();

  allocated = false;
  auto cleanup_failure = [&]() { return Status::Internal("cleanup failure"); };
  auto failed = [&](const StartupCandidate&, bool, double*) {
    allocated = true;
    return Status::Internal("probe failure");
  };
  const Status status = SelectStartupDeviceCandidate({{{0, 0}}}, failed,
                                                      cleanup_failure);
  Check(status.message() == "cleanup failure");
  std::cout << "startup selection tests passed\n";
}
