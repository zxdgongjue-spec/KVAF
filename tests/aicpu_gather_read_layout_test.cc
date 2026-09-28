#include "aicpu_gather_read_layout.h"
#include "gather_output_layout.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

void ProbeRotatesFirstSlot(std::uint32_t blocks, std::uint32_t block_bytes) {
  const std::uint64_t base = 0x10000000ULL;
  const auto extra = kvaf::aicpu_gather_read_client::ReservedSlots(blocks, block_bytes);
  const auto slots = static_cast<std::uint64_t>(2U) * blocks + extra;
  const auto bytes = slots * block_bytes;
  assert(bytes % 4096U == 0U);
  std::vector<std::uint64_t> reserved;
  for (std::uint64_t slot = blocks + 1U; slot < slots; ++slot)
    reserved.push_back(base + slot * block_bytes);
  reserved.push_back(base);
  assert(reserved.size() == blocks + extra);
  std::uint64_t staging = 0U;
  assert(kvaf::aicpu_gather_read_client::SelectStagingAddress(
      base, bytes, block_bytes, blocks, reserved, &staging));
  assert(staging == base + static_cast<std::uint64_t>(blocks + 1U) * block_bytes);
  std::vector<kvaf::peer::GatherSpan> sources;
  for (std::uint64_t slot = 1U; slot <= blocks; ++slot)
    sources.push_back({base + slot * block_bytes, block_bytes, 0U});
  std::uint32_t total = 0U;
  assert(!kvaf::peer::ValidateGather(base, bytes, base +
      static_cast<std::uint64_t>(blocks) * block_bytes, sources, &total));
  assert(kvaf::peer::ValidateGather(base, bytes, staging, sources, &total));
  assert(total == static_cast<std::uint64_t>(blocks) * block_bytes);
}

}  // namespace

int main() {
  ProbeRotatesFirstSlot(8U, 4096U);
  ProbeRotatesFirstSlot(4096U, 656U);
  std::uint64_t extent = 0U;
  const std::vector<kvaf::GatherOutputSpan> scattered{
      {64U, 16U, 0U}, {0U, 16U, 0U}, {32U, 16U, 0U}};
  assert(kvaf::ValidateGatherOutput(192U, scattered, 48U, &extent));
  assert(extent == 80U);
  assert(kvaf::ValidateGatherOutput(192U,
      {{0U, 16U, 0U}, {32U, 16U, 0U}, {64U, 16U, 0U}}, 48U, &extent));
  assert(kvaf::ValidateGatherOutput(192U,
      {{64U, 16U, 0U}, {32U, 16U, 0U}, {0U, 16U, 0U}}, 48U, &extent));
  assert(!kvaf::ValidateGatherOutput(127U, scattered, 48U, &extent));
  assert(!kvaf::ValidateGatherOutput(192U, scattered, 47U, &extent));
  assert(!kvaf::ValidateGatherOutput(192U,
      {{0U, 16U, 0U}, {8U, 16U, 0U}}, 32U, &extent));
  assert(!kvaf::ValidateGatherOutput(192U,
      {{16U, 16U, 0U}, {8U, 16U, 0U}}, 32U, &extent));
  assert(!kvaf::ValidateGatherOutput(192U,
      {{0U, 16U, 1U}, {64U, 16U, 0U}}, 32U, &extent));
  std::cout << "AICPU GatherRead layout: PASS\n";
}
