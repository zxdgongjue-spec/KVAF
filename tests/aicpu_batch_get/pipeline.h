#pragma once

#include "abi.h"
#if defined(KVAF_AICPU_TEST_DEVICE_KERNEL)
#include <securec.h>
#else
#include <cstddef>
#include <cstring>
#endif

namespace kvaf::aicpu_test {
inline std::uint64_t PipelineChunkCount(const Config& c, std::uint32_t worker) {
  const auto r = ScatterRange(c, worker, ActiveThreads(c));
  return (r.end - r.begin + kPipelineChunkRows - 1) / kPipelineChunkRows;
}
inline RowRange PipelineChunkRange(const Config& c, std::uint32_t worker, std::uint64_t chunk) {
  const auto r = ScatterRange(c, worker, ActiveThreads(c));
  const auto begin = r.begin + chunk * kPipelineChunkRows;
  const auto end = begin + kPipelineChunkRows;
  return {begin, end < r.end ? end : r.end};
}
inline std::uint64_t PipelineProgressToken(std::uint64_t sequence, std::uint64_t chunks) {
  return (sequence << 32) | chunks;
}
// Release only the progress reached by every Jetty.
template <class Read>
std::uint64_t ChainReadyProgress(const Config& c, std::uint32_t worker,
                                 std::uint64_t sequence, Read read) {
  const auto chunks = PipelineChunkCount(c, worker);
  auto progress = chunks;
  bool stale = false;
  for (std::uint32_t q = 0; q < kJettyCount; ++q) {
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
// A progress jump releases all intervening chunks; finalize once per worker.
template <class IO>
Result RunPipelineChunks(const Config& c, std::uint32_t worker, std::uint64_t sequence,
                         std::uint64_t start, std::uint64_t timeout, IO& io) {
  if (worker >= ActiveThreads(c) || !sequence ||
      sequence > UINT32_MAX || !timeout) return Result::kInvalidArgs;
  const auto chunks = PipelineChunkCount(c, worker);
  std::uint64_t consumed = 0;
  while (consumed < chunks) {
    for (;;) {
      const auto status = io.Status();
      if (status != Result::kPending) return status;
      const auto token = io.ReadRangeDone(worker), observed = io.Now();
      const auto seq = token >> 32, progress = token & UINT32_MAX;
      if (seq > sequence) return Result::kSequenceError;
      if (seq == sequence && (!progress || progress > chunks || progress < consumed))
        return Result::kSequenceError;
      if (observed - start >= timeout) return Result::kTimeout;
      if (seq == sequence && progress > consumed) {
        if (!consumed) io.RangeReady(observed);
        const auto first = PipelineChunkRange(c, worker, consumed);
        const auto last = PipelineChunkRange(c, worker, progress - 1);
        io.CopySpan({first.begin, last.end});
        consumed = progress;
        if (const auto copied = io.Status(); copied != Result::kPending) return copied;
        if (io.Now() - start >= timeout) return Result::kTimeout;
        break;
      }
      io.Relax();
    }
  }
  io.FinishRange();
  if (const auto finished = io.Status(); finished != Result::kPending) return finished;
  return io.Now() - start >= timeout ? Result::kTimeout : Result::kOk;
}
// Copy packed staging rows into strided output slots.
inline bool CopyScatterRows(std::uint8_t* output, const std::uint8_t* source,
                             std::uint64_t rows, std::uint64_t bytes, std::uint64_t stride) {
  for (std::uint64_t row = 0; row < rows; ++row) {
#if defined(KVAF_AICPU_TEST_DEVICE_KERNEL)
    if (memcpy_sp(output + row * stride, bytes, source + row * bytes, bytes) != EOK) return false;
#else
    std::memcpy(output + row * stride, source + row * bytes, static_cast<std::size_t>(bytes));
#endif
  }
  return true;
}

// Leave output untouched until the external reader acknowledges it.
template <class IO>
Result VerifyOutputViaUrma(std::uint64_t sequence, std::uint64_t timeout, IO& io) {
  const auto begin = io.Now();
  io.PublishOutputReady(sequence);
  for (;;) {
    const auto ack = io.ReadOutputVerified();
    const auto ack_sequence = ack >> 1;
    if (ack_sequence > sequence) return Result::kSequenceError;
    if (ack_sequence == sequence) return (ack & 1) ? Result::kDataMismatch : Result::kOk;
    if (io.Now() - begin >= timeout) return Result::kTimeout;
    io.Relax();
  }
}
}  // namespace kvaf::aicpu_test
