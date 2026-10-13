#pragma once

#include "cpu_gather.h"
#include "gather_output.h"

#include <utility>
#include <vector>

namespace kvaf {

struct GatherHostPlan {
  std::uint32_t total_bytes{0U};
  std::uint64_t output_extent{0U};
  std::vector<std::pair<std::uint64_t, std::uint64_t>> output_ranges;
};

bool PrepareGatherHostLayout(std::uint64_t base, std::uint64_t region_bytes,
                             std::uint64_t staging, std::uint64_t descriptor,
                             const std::vector<peer::GatherSpan>& sources,
                             std::uint64_t output_bytes,
                             const std::vector<GatherOutputSpan>& targets,
                             GatherHostPlan* plan);

}
