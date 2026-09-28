#include "aicpu_gather_read_layout.h"

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
  std::cout << "AICPU GatherRead layout: PASS\n";
}
