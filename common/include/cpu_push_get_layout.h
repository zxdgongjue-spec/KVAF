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
  l.staging = Align64(l.descriptor + kMaxRequests * sizeof(peer::GatherSpan));
  l.bytes = Align64(l.staging + std::uint64_t{token_bytes} * kMaxRequests);
  *out = l;
  return true;
}
struct Range { std::uint32_t begin{}, end{}; };
inline std::uint32_t RowsPerCacheGroup(std::uint32_t bytes) {
  std::uint32_t a = bytes, b = 64;
  while (b) { const auto remainder = a % b; a = b; b = remainder; }
  return 64 / a;
}
inline Range WorkerRange(std::uint32_t count, std::uint32_t bytes,
                         std::uint32_t worker, std::uint32_t workers) {
  const auto quantum = RowsPerCacheGroup(bytes);
  const auto groups = (count + quantum - 1) / quantum;
  return {std::min(count, static_cast<std::uint32_t>(std::uint64_t{groups} * worker / workers) * quantum),
          std::min(count, static_cast<std::uint32_t>(std::uint64_t{groups} * (worker + 1) / workers) * quantum)};
}
inline std::uint32_t ChunkCount(Range r) { return (r.end - r.begin + kChunkRows - 1) / kChunkRows; }
inline Range ChunkRange(Range r, std::uint32_t chunk) {
  const auto begin = r.begin + chunk * kChunkRows;
  return {begin, std::min(r.end, begin + kChunkRows)};
}
inline bool SourceRange(std::uint64_t base, std::uint64_t bytes,
                        std::uint64_t address, std::uint64_t length) {
  return length && address >= base && length <= bytes && address - base <= bytes - length;
}
}  // namespace kvaf::cpu_push
