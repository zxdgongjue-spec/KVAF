#pragma once
#include <cstdint>
#include <limits>
#include "status.h"
#include "types.h"
namespace kvaf {
// Stable identity and logical bounds; bindings keep their own mapped/registered
// lengths. Padding and staging are never included in logical_bytes.
struct CacheRegionDesc {
  std::uint64_t region_id{0}, epoch{0};
  Address canonical_base{0};
  std::uint64_t logical_bytes{0}, slot_size{0};
  Status Translate(Address address, std::uint64_t bytes, Address binding_base, Address* out) const {
    if (!out || address < canonical_base || address - canonical_base > logical_bytes ||
        bytes > logical_bytes - (address - canonical_base))
      return Status::InvalidArgument("cache address is outside the logical region");
    const auto offset = address - canonical_base;
    if (offset > std::numeric_limits<Address>::max() - binding_base ||
        bytes > std::numeric_limits<Address>::max() - (binding_base + offset))
      return Status::InvalidArgument("cache address translation overflow");
    *out = binding_base + offset;
    return Status::OK();
  }
};
}  // namespace kvaf
