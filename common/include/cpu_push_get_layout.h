#pragma once

#include <algorithm>
#include <cstdint>

#include "cpu_push_get_abi.h"

namespace kvaf::cpu_push {
inline constexpr std::uint64_t Align64(std::uint64_t n) { return (n + 63) & ~std::uint64_t{63}; }
struct Layout {
  std::uint64_t request{0}, notify{64}, done{128}, ready{192};
  std::uint64_t descriptor{}, staging{}, bytes{};
};
inline bool MakeLayout(std::uint32_t token_bytes, std::uint32_t workers, Layout* out) {
  if (!out || !token_bytes || !workers || workers > kMaxWorkers ||
      std::uint64_t{token_bytes} * kMaxRequests > UINT32_MAX) return false;
  Layout l;
  l.descriptor = l.ready + workers * sizeof(Ready);
  l.staging = Align64(l.descriptor + kMaxRequests * sizeof(std::uint64_t));
  l.bytes = Align64(l.staging + std::uint64_t{token_bytes} * kMaxRequests);
  *out = l;
  return true;
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
