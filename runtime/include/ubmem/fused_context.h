#pragma once

#include <cstdint>

namespace kvaf {

// Host-side ownership descriptor, not a kernel launch ABI. Every output view
// shares one arena/lease; only the original descriptor may release that lease.
// It intentionally contains no URMA queue, channel, or completion pointers.
struct UbmemFusedContext {
  std::uint32_t abi_version{1};
  std::uint32_t device_id{0};
  std::uint64_t lease_id{0};
  std::uint64_t region_base{0};
  std::uint64_t region_bytes{0};
  std::uint64_t slot_bytes{0};
  std::uint64_t output_address{0};
  std::uint64_t output_bytes{0};
  std::uint32_t token_bytes{0};
  std::uint32_t tokens_per_block{0};
};

}  // namespace kvaf
