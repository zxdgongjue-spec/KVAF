#pragma once

#include "abi.h"

namespace kvaf::aicpu_test {
// IO is a device adapter in production and a deterministic simulator in UT.
// This is the same sequencing loop in both, not a separate mock algorithm.
template <class IO>
Result RunDevice(const Config& config, IO& io, Sample* samples, KernelResult* result) {
  Layout layout{};
  // Pipeline needs independently running blocks; never silently use the
  // whole-batch wait/pthread path for a pipeline configuration.
  if (!MakeLayout(config, &layout) || IsScatterPipeline(config) || !samples || !result)
    return Result::kInvalidArgs;
  *result = {};
  result->requested_threads = config.scatter_threads;
  result->active_threads = ActiveThreads(config);
  result->kernel_revision = kKernelRevision;
  result->counter_frequency = io.Frequency();
  if (result->counter_frequency == 0 || result->counter_frequency > UINT64_MAX / config.timeout_ms)
    return result->status = Result::kInvalidArgs;
  const std::uint64_t timeout_ticks = result->counter_frequency * config.timeout_ms / 1000;
  for (std::uint64_t seq = 1; seq <= layout.rounds; ++seq) {
    result->failed_sequence = seq;
    const bool measure_transfer = config.profile_transfer && seq > config.warmup;
    const auto prepare_start = measure_transfer ? io.Now() : 0;
    // Reset output cache residency left by allocation/previous verification.
    // Complete before timing or notifying the producer, including warmup.
    io.PrepareRound(config, layout);
    const auto start = io.Now();
    io.PublishRequest(seq);
    const auto published = measure_transfer ? io.Now() : 0;
    std::uint64_t done_reads = 0;
    for (;;) {
      const auto done = io.ReadDone();
      if (measure_transfer) ++done_reads;
      if (done == seq) break;
      if (done > seq) return result->status = Result::kSequenceError;
      if (io.Now() - start >= timeout_ticks) return result->status = Result::kTimeout;
    }
    const auto ready = io.Now();
    if (IsDirectOutput(config)) io.CompleteDirectOutput();
    else io.Scatter(config, layout);
    const auto end = io.Now();
    // Verification is outside the reported per-round timing, but before the
    // next request; the producer cannot overwrite payload during checking.
    // PrepareRound removes its cache effects before the next sample starts.
    if (!io.VerifyTags(seq, config, layout)) return result->status = Result::kDataMismatch;
    const auto verified = measure_transfer ? io.Now() : 0;
    if (seq > config.warmup) {
      samples[seq - config.warmup - 1] = IsDirectOutput(config)
          ? Sample{end - start, end - start, 0}
          : Sample{end - start, ready - start, end - ready};
      if (measure_transfer) {
        auto& sample = samples[seq - config.warmup - 1];
        sample.request_publish_ticks = published - start;
        sample.done_poll_count = done_reads;
        sample.cache_prepare_ticks = start - prepare_start;
        sample.verify_ticks = verified - end;
      }
    }
    result->completed_rounds = seq;
  }
  result->failed_sequence = 0;
  return result->status = Result::kOk;
}
}  // namespace kvaf::aicpu_test
