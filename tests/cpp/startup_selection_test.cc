#include "startup_selection.h"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <tuple>

using kvaf::Status;
using kvaf::detail::SelectStartupCandidate;
using kvaf::detail::BuildStartupCandidates;
using kvaf::detail::ShouldTuneStartupCandidates;
using kvaf::detail::StartupCandidate;

void Check(bool condition) {
  if (!condition) {
    std::cerr << "startup selection expectation failed\n";
    std::exit(1);
  }
}

int main() {
  Check(ShouldTuneStartupCandidates(true, true));
  Check(!ShouldTuneStartupCandidates(true, false));
  Check(!ShouldTuneStartupCandidates(false, true));

  std::vector<StartupCandidate> fixed;
  Check(BuildStartupCandidates(5, 3, 2, &fixed).ok());
  Check(fixed.size() == 3 && fixed.front().peer == 2 &&
        fixed.front().local_group == 0 && fixed.back().local_group == 2);
  Check(!BuildStartupCandidates(5, 1, 5, &fixed).ok());
  Check(BuildStartupCandidates(2, 2, -1, &fixed).ok());
  Check(fixed.size() == 4 && fixed.front().peer == 0 && fixed.back().peer == 1);

  const std::vector<StartupCandidate> candidates{{0, 0}, {1, 0}, {2, 0}, {3, 0}, {3, 1}};
  std::vector<std::tuple<unsigned, std::size_t, bool>> calls;
  bool allocated = false;
  int releases = 0;
  auto cleanup = [&]() {
    Check(allocated);
    allocated = false;
    ++releases;
    return Status::OK();
  };
  auto initialize = [&](const StartupCandidate& c, bool measure, double* cycles) {
    Check(!allocated);
    allocated = true;
    calls.emplace_back(c.peer, c.local_group, measure);
    *cycles = c.local_group == 1 ? 5.0 : 100.0 - c.peer * 10.0;
    if (c.peer == 1) return Status::Internal("unreachable");
    return Status::OK();
  };
  // Candidate count has no hard cap; every explicitly shortlisted EID pair is measured.
  Check(SelectStartupCandidate(true, candidates, initialize, cleanup).ok());
  Check(calls.size() == 6 && releases == 5 && allocated);
  Check(calls.back() == std::make_tuple(3U, std::size_t{1}, false));
  cleanup();

  calls.clear(); releases = 0;
  Check(SelectStartupCandidate(false, candidates, initialize, cleanup).ok());
  Check(calls.size() == 1 && !std::get<2>(calls.front()) && releases == 0);
  cleanup();
  // Selection has no process-global winner: another N card can prefer peer 0.
  auto other_card = [&](const StartupCandidate& c, bool measure, double* cycles) {
    Check(!allocated); allocated = true;
    calls.emplace_back(c.peer, c.local_group, measure);
    *cycles = 10.0 + c.peer;
    return Status::OK();
  };
  Check(SelectStartupCandidate(true, candidates, other_card, cleanup).ok());
  Check(std::get<0>(calls.back()) == 0);
  cleanup();

  // Failed winner reconstruction falls back in measured order; ties are stable.
  calls.clear();
  auto rebuild_failure = [&](const StartupCandidate& c, bool measure, double* cycles) {
    Check(!allocated); allocated = true;
    calls.emplace_back(c.peer, c.local_group, measure);
    *cycles = 10.0;
    return !measure && c.peer == 0 ? Status::Internal("rebuild failed") : Status::OK();
  };
  Check(SelectStartupCandidate(true, {{0, 0}, {1, 0}}, rebuild_failure, cleanup).ok());
  Check(calls.size() == 4 && std::get<0>(calls.back()) == 1);
  cleanup();

  auto invalid = [&](const StartupCandidate&, bool measure, double* cycles) {
    Check(measure); allocated = true;
    *cycles = std::numeric_limits<double>::quiet_NaN();
    return Status::OK();
  };
  Check(!SelectStartupCandidate(true, {{0, 0}}, invalid, cleanup).ok());
  Check(!allocated);
  auto failed = [&](const StartupCandidate&, bool, double*) {
    allocated = true;
    return Status::Internal("probe failure");
  };
  Check(!SelectStartupCandidate(true, {{0, 0}}, failed, cleanup).ok());
  auto cleanup_failure = [&]() { return Status::Internal("cleanup failure"); };
  const auto status = SelectStartupCandidate(true, {{0, 0}, {0, 1}}, failed, cleanup_failure);
  Check(status.message() == "cleanup failure");
  std::cout << "startup selection tests passed\n";
}
