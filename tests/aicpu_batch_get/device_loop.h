#pragma once

#include "abi.h"

namespace kvaf::aicpu_test {
// IO is a device adapter in production and a deterministic simulator in UT.
// This is the same sequencing loop in both, not a separate mock algorithm.
template <class IO>
Result RunDevice(const Config& config, IO& io, Sample* samples, KernelResult* result) {
  Layout layout{};
  if (!MakeLayout(config, &layout) || !samples || !result) return Result::kInvalidArgs;
  *result = {};
  result->counter_frequency = io.Frequency();
  if (result->counter_frequency == 0 || result->counter_frequency > UINT64_MAX / config.timeout_ms)
    return result->status = Result::kInvalidArgs;
  const std::uint64_t timeout_ticks = result->counter_frequency * config.timeout_ms / 1000;
  for (std::uint64_t seq = 1; seq <= layout.rounds; ++seq) {
    result->failed_sequence = seq;
    const auto start = io.Now();
    io.PublishRequest(seq);
    for (;;) {
      const auto done = io.ReadDone();
      if (done == seq) break;
      if (done > seq) return result->status = Result::kSequenceError;
      if (io.Now() - start >= timeout_ticks) return result->status = Result::kTimeout;
    }
    const auto ready = io.Now();
    io.Scatter(config, layout);
    const auto end = io.Now();
    // Verification is outside the reported per-round timing, but before the
    // next request; the producer cannot overwrite staging during checking.
    if (!io.VerifyTags(seq, config, layout)) return result->status = Result::kDataMismatch;
    if (seq > config.warmup)
      samples[seq - config.warmup - 1] = {end - start, ready - start, end - ready};
    result->completed_rounds = seq;
  }
  result->failed_sequence = 0;
  return result->status = Result::kOk;
}
}  // namespace kvaf::aicpu_test
