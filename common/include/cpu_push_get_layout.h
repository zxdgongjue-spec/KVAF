#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

#include "cpu_push_get_abi.h"

namespace kvaf::cpu_push {
inline constexpr std::uint64_t Align64(std::uint64_t n) { return (n + 63) & ~std::uint64_t{63}; }
struct Layout {
  std::uint64_t request{0}, notify{64}, done{128}, ready{192};
  std::uint64_t output_target{}, descriptor{}, staging{}, bytes{};
  std::uint64_t DstRows(std::uint32_t count) const {
    return Align64(descriptor + std::uint64_t{count} * sizeof(std::uint64_t));
  }
};
inline bool MakeLayout(std::uint32_t token_bytes, std::uint32_t workers, Layout* out,
                       bool direct_output = false) {
  if (!out || !token_bytes || !workers || workers > kMaxWorkers ||
      std::uint64_t{token_bytes} * kMaxRequests > UINT32_MAX) return false;
  Layout l;
  l.descriptor = l.ready + workers * sizeof(Ready);
  if (direct_output) {
    l.output_target = l.descriptor;
    l.descriptor += sizeof(OutputTarget);
  }
  l.staging = l.DstRows(kMaxRequests);
  if (direct_output) l.staging = Align64(l.staging + kMaxRequests * sizeof(std::uint32_t));
  l.bytes = direct_output ? l.staging : Align64(l.staging + std::uint64_t{token_bytes} * kMaxRequests);
  *out = l;
  return true;
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
  return {groups * worker / active * quantum,
          std::min(groups * (worker + 1) / active * quantum, count)};
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
inline std::uint32_t DirectRowsPerWrite(std::uint32_t token_bytes, std::uint32_t max_sge,
                                        std::uint32_t max_bytes) {
  return token_bytes ? std::min(max_sge, max_bytes / token_bytes) : 0;
}

// IDs follow worker, chunk, Jetty, then row order, independently of posting order.
struct DirectWriteGrouping {
  std::uint32_t rows_per_wr;

  std::uint32_t Count(Range rows) const {
    return (rows.end - rows.begin + rows_per_wr - 1) / rows_per_wr;
  }

  std::uint32_t ChunkWrCount(std::uint32_t rows) const {
    std::uint32_t total = 0;
    for (std::uint32_t jetty = 0; jetty < kJettyCount; ++jetty)
      total += Count(JettyRange({0, rows}, jetty));
    return total;
  }

  std::array<std::uint32_t, kMaxWorkers + 1> WorkerOffsets(
      std::uint32_t count, std::uint32_t token_bytes, std::uint32_t workers) const {
    std::array<std::uint32_t, kMaxWorkers + 1> offsets{};
    const auto full_chunk_wrs = ChunkWrCount(kChunkRows);
    for (std::uint32_t worker = 0; worker < workers; ++worker) {
      const auto range = WorkerRange(count, token_bytes, worker, workers);
      const auto rows = range.end - range.begin;
      offsets[worker + 1] = offsets[worker] + rows / kChunkRows * full_chunk_wrs +
          ChunkWrCount(rows % kChunkRows);
    }
    return offsets;
  }

  template <class Emit>
  bool ForEachWrite(std::uint32_t worker_offset, std::uint32_t chunk, Range rows,
                    std::uint32_t jetty, Emit emit) const {
    auto id = worker_offset + chunk * ChunkWrCount(kChunkRows);
    for (std::uint32_t q = 0; q < jetty; ++q) id += Count(JettyRange(rows, q));
    const auto shard = JettyRange(rows, jetty);
    for (auto first = shard.begin; first < shard.end; first += rows_per_wr)
      if (!emit(id++, Range{first, std::min(shard.end, first + rows_per_wr)})) return false;
    return true;
  }
};

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
