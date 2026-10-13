#include "gather_host_plan.h"

#include "gather_output_layout.h"
#include "peer_protocol.h"

#include <algorithm>

namespace kvaf {

bool PrepareGatherHostLayout(std::uint64_t base, std::uint64_t region_bytes,
                             std::uint64_t staging, std::uint64_t descriptor,
                             const std::vector<peer::GatherSpan>& sources,
                             std::uint64_t output_bytes,
                             const std::vector<GatherOutputSpan>& targets,
                             GatherHostPlan* plan) {
  if (plan == nullptr || sources.empty() || sources.size() > peer::kMaxGatherSpans ||
      targets.size() > peer::kMaxGatherSpans || region_bytes < 64U ||
      base > UINT64_MAX - region_bytes) return false;
  const auto data_bytes = peer::GatherControlOffset(region_bytes);
  if (staging < base || staging - base >= data_bytes) return false;
  const auto descriptor_bytes = peer::GatherDescriptorBytes(sources.size());
  if (descriptor != 0U && (descriptor % 64U != 0U ||
      !peer::GatherRange(base, data_bytes, descriptor, descriptor_bytes))) return false;
  std::uint64_t staging_limit = base + data_bytes;
  std::uint64_t total = 0U;
  for (const auto& source : sources) {
    if (source.reserved != 0U ||
        !peer::GatherRange(base, data_bytes, source.address, source.length) ||
        total > UINT32_MAX - source.length) return false;
    const auto end = source.address + source.length;
    if (source.address < staging) {
      if (end > staging) return false;
    } else {
      staging_limit = std::min(staging_limit, source.address);
    }
    if (descriptor != 0U && descriptor < end &&
        source.address < descriptor + descriptor_bytes) return false;
    total += source.length;
  }
  if (total > staging_limit - staging ||
      (descriptor != 0U && descriptor < staging + total &&
       staging < descriptor + descriptor_bytes)) return false;
  const GatherOutputSpan contiguous{0U, static_cast<std::uint32_t>(total), 0U};
  std::uint64_t extent = 0U;
  if (!ValidateGatherOutput(output_bytes, targets.empty() ? &contiguous : targets.data(),
                            targets.empty() ? 1U : targets.size(),
                            static_cast<std::uint32_t>(total), &extent,
                            plan->output_ranges)) return false;
  plan->total_bytes = static_cast<std::uint32_t>(total);
  plan->output_extent = extent;
  return true;
}

}
