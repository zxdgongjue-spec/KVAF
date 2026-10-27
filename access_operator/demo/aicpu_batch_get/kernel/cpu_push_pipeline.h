#pragma once

#include "cpu_push_get_layout.h"
#if defined(KVAF_CPU_PUSH_DEVICE_KERNEL)
#include <securec.h>
#else
#include <cstring>
#endif

namespace kvaf::cpu_push {
// Release only the prefix reached by every Jetty.
template <class Read>
std::uint64_t ChainReadyProgress(std::uint32_t chunks, std::uint64_t sequence, Read read) {
  auto progress = chunks;
  bool stale = false;
  for (std::uint32_t q = 0; q < kJettyCount; ++q) {
    const auto token = read(q), seq = token >> 32, value = token & UINT32_MAX;
    if (seq > sequence || (seq == sequence && (!value || value > chunks)))
      return Progress(sequence, UINT32_MAX);
    stale |= seq < sequence;
    if (seq == sequence) progress = std::min(progress, static_cast<std::uint32_t>(value));
  }
  return stale ? 0 : Progress(sequence, progress);
}

// Merge ready chunks into one copy span and finalize once per worker.
template <class IO>
Result Consume(std::uint32_t count, std::uint32_t bytes, std::uint32_t workers,
               std::uint32_t worker, std::uint64_t sequence, std::uint64_t start,
               std::uint64_t timeout, IO& io, std::uint32_t* consumed) {
  const auto range = WorkerRange(count, bytes, worker, workers);
  const auto chunks = ChunkCount(range);
  if (!chunks) return Result::kOk;
  std::uint32_t cursor = 0;
  Result copy_status = Result::kPending;
  while (cursor < chunks) {
    if (copy_status != Result::kPending) return copy_status;
    if (const auto status = io.Status(); status != Result::kPending) return status;
    const auto token = io.ReadRangeDone(worker), observed = io.Now();
    const auto seq = token >> 32;
    const auto progress = static_cast<std::uint32_t>(token);
    if (seq > sequence || (seq == sequence && (!progress || progress > chunks || progress < cursor)))
      return Result::kSequenceError;
    if (observed - start >= timeout) return Result::kTimeout;
    if (seq == sequence && progress > cursor) {
      if (io.Copy({ChunkRange(range, cursor).begin, ChunkRange(range, progress - 1).end}))
        *consumed = progress;
      else copy_status = Result::kCopyError;
      cursor = progress;
    } else io.Relax();
  }
  io.FinishRange();
  if (copy_status != Result::kPending) return copy_status;
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
