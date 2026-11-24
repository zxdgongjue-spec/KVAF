#pragma once

#include <algorithm>
#include <cstdint>

#include "cpu_push_get_abi.h"
#include "get_window.h"

namespace kvaf::cpu_push {
inline constexpr std::uint64_t Align64(std::uint64_t n) { return (n + 63) & ~std::uint64_t{63}; }
struct Layout {
  std::uint64_t request{0}, notify{64}, done{128}, ready{192};
  std::uint64_t descriptor{}, staging{}, bytes{};
};
inline bool MakeLayout(std::uint32_t token_bytes, std::uint32_t workers, Layout* out,
                       std::uint32_t capacity = kDefaultGetWindow) {
  if (!out || !token_bytes || !workers || workers > kMaxWorkers ||
      !capacity || capacity > INT32_MAX || capacity % kGetWindowGranularity ||
      std::uint64_t{token_bytes} * capacity > UINT32_MAX) return false;
  Layout l;
  l.descriptor = l.ready + workers * sizeof(Ready);
  l.staging = Align64(l.descriptor + std::uint64_t{capacity} * sizeof(std::uint64_t));
  l.bytes = Align64(l.staging + std::uint64_t{token_bytes} * capacity);
  *out = l;
  return true;
}
inline std::uint32_t ScatterWindowCapacity(std::uint32_t token_bytes, std::uint32_t workers,
                                         std::uint64_t budget) {
  if (!budget) {
    Layout layout;
    return MakeLayout(token_bytes, workers, &layout) ? kDefaultGetWindow : 0U;
  }
  return LargestGetWindow([&](std::uint32_t capacity) {
    Layout layout;
    return MakeLayout(token_bytes, workers, &layout, capacity) && layout.bytes <= budget;
  });
}
struct Range { std::uint32_t begin{}, end{}; };
inline std::uint32_t RowsPerCacheGroup(std::uint32_t bytes) {
  auto divisor = 64U;
  while (divisor) {
    const auto remainder = bytes % divisor;
    bytes = divisor; divisor = remainder;
  }
  return 64 / bytes;
}
inline std::uint32_t ActiveWorkers(std::uint32_t count, std::uint32_t bytes,
                                   std::uint32_t workers) {
  const auto quantum = RowsPerCacheGroup(bytes);
  return std::min(workers, (count + quantum - 1) / quantum);
}
// Worker boundaries own complete staging cache lines.
inline Range WorkerRange(std::uint32_t count, std::uint32_t bytes,
                         std::uint32_t worker, std::uint32_t workers) {
  const auto active = ActiveWorkers(count, bytes, workers);
  if (worker >= active) return {count, count};
  const auto quantum = RowsPerCacheGroup(bytes);
  const auto groups = (count + quantum - 1) / quantum;
  return {static_cast<std::uint32_t>(std::uint64_t{groups} * worker / active * quantum),
          static_cast<std::uint32_t>(std::min(std::uint64_t{groups} * (worker + 1) / active * quantum,
                                            std::uint64_t{count}))};
}
inline std::uint32_t ChunkCount(Range r) { return (r.end - r.begin + kChunkRows - 1) / kChunkRows; }
inline Range ChunkRange(Range r, std::uint32_t chunk) {
  const auto begin = r.begin + chunk * kChunkRows;
  return {begin, std::min(r.end, begin + kChunkRows)};
}
inline Range JettyRange(Range range, std::uint32_t queue) {
  const auto rows = range.end - range.begin;
  const auto base = rows / kJettyCount, remainder = rows % kJettyCount;
  const auto begin = range.begin + queue * base + std::min(queue, remainder);
  return {begin, begin + base + (queue < remainder ? 1U : 0U)};
}
template <class Emit>
void ForEachPipelineChunk(std::uint32_t count, std::uint32_t bytes,
                          std::uint32_t workers, Emit emit) {
  const auto active = ActiveWorkers(count, bytes, workers);
  std::uint32_t max = 0;
  for (std::uint32_t w = 0; w < active; ++w)
    max = std::max(max, ChunkCount(WorkerRange(count, bytes, w, active)));
  for (std::uint32_t chunk = 0; chunk < max; ++chunk)
    for (std::uint32_t w = 0; w < active; ++w) {
      const auto range = WorkerRange(count, bytes, w, active);
      if (chunk < ChunkCount(range)) emit(w, chunk, ChunkRange(range, chunk));
    }
}
inline bool SourceRange(std::uint64_t base, std::uint64_t bytes,
                        std::uint64_t address, std::uint64_t length) {
  return length && address >= base && length <= bytes && address - base <= bytes - length;
}
}  // namespace kvaf::cpu_push
