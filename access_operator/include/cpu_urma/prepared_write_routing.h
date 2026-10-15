#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace kvaf::cpu_urma_detail {
struct PreparedQueueRows { std::size_t first{}, count{}; };
// Partition a batch across the data queues in contiguous row ranges.
inline PreparedQueueRows PreparedRowsForQueue(std::size_t rows, std::size_t queues,
    std::size_t queue) {
  if (!queues || queue >= queues) return {};
  const auto base = rows / queues, remainder = rows % queues;
  return {queue * base + std::min(queue, remainder), base + (queue < remainder ? 1U : 0U)};
}
}  // namespace kvaf::cpu_urma_detail
