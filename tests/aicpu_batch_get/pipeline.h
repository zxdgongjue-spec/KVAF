#pragma once

#include "abi.h"

namespace kvaf::aicpu_test {
inline std::uint64_t PipelineChunkCount(const Config& c, std::uint32_t worker) {
  const auto r = ScatterRange(c, worker, ActiveThreads(c));
  return (r.end - r.begin + c.pipeline_chunk_rows - 1) / c.pipeline_chunk_rows;
}
inline RowRange PipelineChunkRange(const Config& c, std::uint32_t worker, std::uint64_t chunk) {
  const auto r = ScatterRange(c, worker, ActiveThreads(c));
  const auto begin = r.begin + chunk * c.pipeline_chunk_rows;
  const auto end = begin + c.pipeline_chunk_rows;
  return {begin, end < r.end ? end : r.end};
}
inline std::uint64_t PipelineProgressToken(std::uint64_t sequence, std::uint64_t chunks) {
  return (sequence << 32) | chunks;
}
// A queue marker fences only THAT queue. Never release a range from queue 0
// alone. A later marker subsumes earlier chunks on the same ordered queue.
template <class Read>
std::uint64_t ChainReadyProgress(const Config& c, std::uint32_t worker,
                                 std::uint64_t sequence, Read read) {
  const auto chunks = PipelineChunkCount(c, worker);
  auto progress = chunks;
  bool stale = false;
  for (std::uint32_t q = 0; q < c.pipeline_jettys; ++q) {
    const auto token = read(q), seq = token >> 32, value = token & UINT32_MAX;
    if (seq > sequence || (seq == sequence && (!value || value > chunks)))
      return PipelineProgressToken(sequence + 1, 0); // Consumer's sequence-error path.
    stale |= seq < sequence;
    if (seq == sequence && value < progress) progress = value;
  }
  return stale ? 0 : PipelineProgressToken(sequence, progress);
}
template <class Emit> void ForEachPipelineChunk(const Config& c, Emit emit) {
  const auto workers = ActiveThreads(c);
  std::uint64_t max = 0;
  for (std::uint32_t w = 0; w < workers; ++w) {
    const auto count = PipelineChunkCount(c, w); if (count > max) max = count;
  }
  for (std::uint64_t chunk = 0; chunk < max; ++chunk)
    for (std::uint32_t w = 0; w < workers; ++w)
      if (chunk < PipelineChunkCount(c, w)) emit(w, chunk, PipelineChunkRange(c, w, chunk));
}
// Consume the entire contiguous ready prefix in one invalidate+copy span.
// A progress jump skips notifications, NEVER data. Clean once per worker.
template <class IO>
Result RunPipelineChunks(const Config& c, std::uint32_t worker, std::uint64_t sequence,
                         std::uint64_t start, std::uint64_t timeout, IO& io) {
  if (!c.pipeline_chunk_rows || worker >= ActiveThreads(c) || !sequence ||
      sequence > UINT32_MAX || !timeout) return Result::kInvalidArgs;
  const auto chunks = PipelineChunkCount(c, worker);
  std::uint64_t consumed = 0, previous = 0;
  while (consumed < chunks) {
    for (;;) {
      const auto status = io.Status();
      if (status != Result::kPending) return status;
      const auto token = io.ReadRangeDone(worker), observed = io.Now();
      const auto seq = token >> 32, progress = token & UINT32_MAX;
      if (seq > sequence) return Result::kSequenceError;
      if (seq == sequence) {
        if (!progress || progress > chunks || progress < previous) return Result::kSequenceError;
        previous = progress;
      }
      if (observed - start >= timeout) return Result::kTimeout;
      if (seq == sequence && progress > consumed) {
        if (!consumed) io.RangeReady(observed);
        const auto first = PipelineChunkRange(c, worker, consumed);
        const auto last = PipelineChunkRange(c, worker, progress - 1);
        io.CopySpan({first.begin, last.end});
        consumed = progress;
        if (io.Status() != Result::kPending) return io.Status();
        if (io.Now() - start >= timeout) return Result::kTimeout;
        break;
      }
      io.Relax();
    }
  }
  io.CleanRange(worker);
  if (io.Status() != Result::kPending) return io.Status();
  return io.Now() - start >= timeout ? Result::kTimeout : Result::kOk;
}
}  // namespace kvaf::aicpu_test
