#pragma once

#include <memory>
#include <vector>

#include "pool_types.h"

namespace kvaf {
struct PutContext;
struct LockContext;
struct PutAccessDesc;
struct FlatAccessDesc;

// The multi-pool directory of the same logical MetaManager. Legacy DSA calls
// keep their fixed-slot semantics; scoped pool calls share this one directory.
class PoolMetadata {
 public:
  PoolMetadata();
  ~PoolMetadata();
  PoolMetadata(const PoolMetadata&) = delete;
  PoolMetadata& operator=(const PoolMetadata&) = delete;
  Status PreparePut(SlotAllocationProvider&, const LayerLayout&,
                    const std::vector<KeyId>&, const std::vector<std::uint64_t>&,
                    PutContext*, PutAccessDesc*);
  Status CommitPut(SlotAllocationProvider&, const PutContext&);
  Status AbortPut(SlotAllocationProvider&, const PutContext&);
  Status LockKeys(const LayerLayout&, const std::vector<KeyId>&, LockContext*);
  Status Lookup(const std::vector<KeyId>&, const LockContext&, FlatAccessDesc*) const;
  Status UnlockKeys(const LockContext&);
  Status Free(SlotAllocationProvider&, const LayerLayout&, const std::vector<KeyId>&);
  std::size_t CountLayer(LayerHandle) const;
  std::size_t CountPool(PoolHandle) const;
  // Only after the region's operator has successfully stopped all DMA.
  Status ErasePool(PoolHandle);
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace kvaf
