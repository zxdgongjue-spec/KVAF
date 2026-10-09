#pragma once

#include "abi.h"

namespace kvaf::aicpu_test {
// Runs on each device block after the coordinator has prepared output and
// published this round. The same wait/consume path is exercised by host UTs.
// Ready is per range, not the global producer-done record. The producer must
// finish every WR for this range before publishing its sequence.
template <class IO>
Result RunPipelineRange(std::uint32_t worker, std::uint64_t sequence,
                        std::uint64_t round_start, std::uint64_t timeout_ticks, IO& io) {
  if (worker >= kMaxScatterThreads || !sequence || !timeout_ticks)
    return Result::kInvalidArgs;
  for (;;) {
    const auto status = io.Status();
    if (status != Result::kPending) return status;
    const auto done = io.ReadRangeDone(worker);
    const auto observed = io.Now();
    if (done > sequence) return Result::kSequenceError;
    if (observed - round_start >= timeout_ticks) return Result::kTimeout;
    if (done == sequence) {
      io.RangeReady(worker, observed);
      io.Scatter(worker, round_start);
      const auto after = io.Status();
      if (after != Result::kPending) return after;
      return io.Now() - round_start >= timeout_ticks ? Result::kTimeout : Result::kOk;
    }
    io.Relax();
  }
}
}  // namespace kvaf::aicpu_test
