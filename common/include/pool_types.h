#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "status.h"
#include "types.h"

namespace kvaf {

struct PoolHandle {
  std::uint64_t owner{0}, id{0};
};

struct LayerHandle {
  std::uint64_t owner{0}, id{0};
  bool empty() const { return owner == 0 && id == 0; }
};

struct PoolSpec {
  std::string name;
  std::uint64_t slot_size{0}, slot_count{0};
  std::uint32_t token_bytes{0}, tokens_per_block{0};
};

struct LayerLayout {
  LayerHandle layer;
  PoolHandle pool;
  std::uint64_t namespace_id{0}, layer_id{0};
  PoolSpec spec;
  RemoteRegionDesc region;
};

struct SlotLease {
  PoolHandle pool;
  std::uint64_t slot{0};
  Address address{0};
};

struct PoolInfo {
  PoolHandle handle;
  PoolSpec spec;
  RemoteRegionDesc region;
  std::uint64_t free_slots{0}, reserved_slots{0}, committed_slots{0};
  std::uint64_t quarantined_slots{0}, bound_layers{0};
  bool quarantined{false};
};

// Pool owns slots; metadata owns key visibility and pins. Implementations must
// validate the complete lease batch before changing any allocation state.
class SlotAllocationProvider {
 public:
  virtual ~SlotAllocationProvider() = default;
  virtual Status Reserve(PoolHandle, std::size_t, std::vector<SlotLease>*) = 0;
  virtual Status Commit(const std::vector<SlotLease>&) = 0;
  virtual Status Release(const std::vector<SlotLease>&) = 0;
};

}  // namespace kvaf
