#pragma once

#include "cpu_gather.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace kvaf::aicpu_gather_read_client {

inline std::size_t ReservedSlots(std::size_t blocks, std::size_t block_bytes) {
  const std::size_t page_slots = 4096U / std::gcd(block_bytes, std::size_t{4096U});
  const std::size_t descriptor_slots =
      (peer::GatherDescriptorBytes(blocks) + 63U + block_bytes - 1U) / block_bytes;
  const std::size_t minimum = (std::size_t{128U} + block_bytes - 1U) / block_bytes + 1U;
  const std::size_t extra = descriptor_slots + minimum;
  const std::size_t used = 2U * blocks + extra;
  return extra + (page_slots - used % page_slots) % page_slots;
}

inline bool SelectStagingAddress(std::uint64_t region_base, std::uint64_t region_bytes,
                                 std::uint32_t block_bytes, std::uint32_t blocks,
                                 std::vector<std::uint64_t> reserved_addresses,
                                 std::uint64_t* staging_address,
                                 std::uint64_t* descriptor_address) {
  if (staging_address == nullptr || descriptor_address == nullptr ||
      block_bytes == 0U || blocks == 0U ||
      region_bytes < 64U || region_base > UINT64_MAX - region_bytes ||
      reserved_addresses.size() < blocks) return false;
  std::sort(reserved_addresses.begin(), reserved_addresses.end());
  const std::uint64_t payload = static_cast<std::uint64_t>(blocks) * block_bytes;
  const std::uint64_t descriptor_bytes = peer::GatherDescriptorBytes(blocks);
  const std::uint64_t data_bytes = peer::GatherControlOffset(region_bytes);
  std::uint64_t run_start = 0U;
  std::uint64_t previous = 0U;
  std::uint32_t run_length = 0U;
  for (const auto address : reserved_addresses) {
    if (!peer::GatherRange(region_base, region_bytes, address, block_bytes) ||
        (run_length != 0U && address == previous)) return false;
    if (run_length != 0U && previous <= UINT64_MAX - block_bytes &&
        address == previous + block_bytes) {
      ++run_length;
    } else {
      run_start = address;
      run_length = 1U;
    }
    const auto descriptor = (run_start + payload + 63U) & ~std::uint64_t{63U};
    if (run_length >= blocks &&
        peer::GatherRange(region_base, data_bytes, run_start, payload) &&
        peer::GatherRange(region_base, data_bytes, descriptor, descriptor_bytes) &&
        descriptor + descriptor_bytes <= run_start +
            static_cast<std::uint64_t>(run_length) * block_bytes) {
      *staging_address = run_start;
      *descriptor_address = descriptor;
      return true;
    }
    previous = address;
  }
  return false;
}

}  // namespace kvaf::aicpu_gather_read_client
