#pragma once

#include <array>

#include "cpu_push_get_layout.h"
#if defined(KVAF_CPU_PUSH_DEVICE_KERNEL)
#include <securec.h>
#else
#include <cstring>
#endif

namespace kvaf::cpu_push {
using ReadySnapshot = std::array<std::uint64_t, kMaxJettyCount>;

// Advance each Jetty independently. Coalesce consecutive staging rows from
// the same snapshot, including across chunk boundaries when all are ready.
template <class IO>
Result Consume(std::uint32_t count, std::uint32_t bytes, std::uint32_t workers,
               std::uint32_t worker, std::uint64_t sequence, std::uint64_t start,
               std::uint64_t timeout, IO& io, std::uint32_t* consumed,
               std::uint32_t jettys = kJettyCount) {
  const auto range = WorkerRange(count, bytes, worker, workers);
  const auto chunks = ChunkCount(range);
  if (!chunks) return Result::kOk;
  std::array<std::uint32_t, kMaxJettyCount> copied{}, required{};
  for (std::uint32_t q = 0; q < jettys; ++q) {
    const auto last = JettyRange(ChunkRange(range, chunks - 1), q, jettys);
    required[q] = chunks - (last.begin == last.end);
  }
  while (*consumed < chunks) {
    if (const auto status = io.Status(); status != Result::kPending) return status;
    const auto snapshot = io.ReadReady(worker);
    auto ready = copied;
    auto first = chunks;
    std::uint32_t last = 0;
    for (std::uint32_t q = 0; q < jettys; ++q) {
      const auto seq = snapshot[q] >> 32;
      const auto progress = static_cast<std::uint32_t>(snapshot[q]);
      if (seq > sequence || (seq == sequence &&
          (!progress || progress > chunks || progress < copied[q])))
        return Result::kSequenceError;
      if (seq == sequence) ready[q] = std::min(progress, required[q]);
      if (ready[q] > copied[q]) {
        first = std::min(first, copied[q]);
        last = std::max(last, ready[q]);
      }
    }
    if (io.Now() - start >= timeout) return Result::kTimeout;
    if (first == chunks) { io.Relax(); continue; }
    Range span{};
    for (auto chunk = first; chunk < last; ++chunk) {
      const auto part = ChunkRange(range, chunk);
      for (std::uint32_t q = 0; q < jettys; ++q) {
        if (chunk < copied[q] || chunk >= ready[q]) continue;
        const auto shard = JettyRange(part, q, jettys);
        if (span.end != shard.begin) {
          if (span.begin != span.end && !io.Copy(span)) return Result::kCopyError;
          span.begin = shard.begin;
        }
        span.end = shard.end;
      }
    }
    if (!io.Copy(span)) return Result::kCopyError;
    copied = ready;
    *consumed = chunks;
    for (std::uint32_t q = 0; q < jettys; ++q)
      if (copied[q] < required[q]) *consumed = std::min(*consumed, copied[q]);
  }
  io.FinishRange();
  if (const auto status = io.Status(); status != Result::kPending) return status;
  return io.Now() - start >= timeout ? Result::kTimeout : Result::kOk;
}

// Preserve caller order and copy each staging row to its output slot.
inline bool CopyMappedRows(std::uint8_t* output, const std::uint8_t* staging,
                           const std::int32_t* output_indices, Range range,
                           std::uint32_t bytes) {
  for (auto i = range.begin; i < range.end; ++i) {
    auto* destination = output + std::uint64_t(output_indices[i]) * bytes;
    const auto* source = staging + std::uint64_t{i} * bytes;
#if defined(KVAF_CPU_PUSH_DEVICE_KERNEL)
    if (memcpy_sp(destination, bytes, source, bytes) != EOK) return false;
#else
    std::memcpy(destination, source, bytes);
#endif
  }
  return true;
}
}  // namespace kvaf::cpu_push
