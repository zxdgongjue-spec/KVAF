#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "cpu_push_get_layout.h"

namespace kvaf::cpu_push {
// A wave can gather disjoint ranges. In an interleaved wave each range belongs
// to the Jetty at the same index.
using GatherWave = std::array<Range, kMaxJettyCount>;

inline std::vector<GatherWave> InterleavedGatherWaves(std::uint32_t count,
    std::uint32_t bytes, std::uint32_t workers, std::uint32_t head_rows,
    std::uint32_t jettys = kJettyCount) {
  struct Shard {
    Range rows;
    std::uint32_t queue;
  };
  std::array<std::vector<Shard>, kMaxAivWorkers> pending;
  const auto active = ActiveWorkers(count, bytes, workers);
  if (active < 2U) return {};
  const auto quantum = RowsPerCacheGroup(bytes);
  bool aligned = true;
  ForEachPipelineChunk(count, bytes, workers, [&](std::uint32_t lane, std::uint32_t, Range chunk) {
    for (std::uint32_t turn = 0; turn < jettys; ++turn) {
      const auto queue = (lane + turn) % jettys;
      const auto rows = JettyRange(chunk, queue, jettys);
      if (rows.begin == rows.end || rows.end <= head_rows) continue;
      // Keep DMA and workers copying subsequent waves on separate cache
      // lines. Leave mixed head/suffix shards to the contiguous schedule.
      if (rows.begin < head_rows || rows.begin % quantum ||
          (rows.end != count && rows.end % quantum)) {
        aligned = false;
        continue;
      }
      pending[lane].push_back({rows, queue});
    }
  });
  if (!aligned) return {};

  std::vector<GatherWave> waves;
  std::array<std::size_t, kMaxAivWorkers> next{};
  std::uint32_t first_lane = 0;
  for (;;) {
    GatherWave wave{};
    bool any = false;
    for (std::uint32_t i = 0; i < active; ++i) {
      const auto lane = (first_lane + i) % active;
      if (next[lane] == pending[lane].size()) continue;
      const auto& shard = pending[lane][next[lane]];
      auto& range = wave[shard.queue];
      if (range.begin != range.end) continue;
      range = shard.rows;
      ++next[lane];
      any = true;
    }
    if (!any) return waves;
    waves.push_back(wave);
    // Rotate priority when multiple lanes compete for a Jetty.
    first_lane = (first_lane + 1U) % active;
  }
}
}  // namespace kvaf::cpu_push
