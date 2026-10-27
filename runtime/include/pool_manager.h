#pragma once

#include <functional>
#include <memory>

#include "access_operator.h"
#include "pool_types.h"

namespace kvaf {
class DdrMemoryAllocator;
using AccessOperatorFactory = std::function<std::unique_ptr<AccessOperator>()>;

// One region/operator context per pool; no nested RuntimeCore instances.
// Host API calls are serialized by the caller, as for the legacy RuntimeCore.
class PoolManager final : public SlotAllocationProvider {
 public:
  PoolManager(DdrMemoryAllocator&, AccessOperatorFactory);
  ~PoolManager();
  Status Create(const PoolSpec&, PoolHandle*);
  Status Info(PoolHandle, PoolInfo*) const;
  Status Bind(std::uint64_t namespace_id, std::uint64_t layer_id, PoolHandle, LayerHandle*);
  Status FindLayer(std::uint64_t namespace_id, std::uint64_t layer_id, LayerHandle*) const;
  Status Resolve(LayerHandle, LayerLayout*, bool allow_quarantined = false) const;
  Status Unbind(LayerHandle);
  void UnbindPool(PoolHandle);
  AccessOperator* Operator(PoolHandle) const;
  Status Stop(PoolHandle);
  Status Destroy(PoolHandle);
  void Quarantine(PoolHandle);
  std::vector<PoolHandle> Handles() const;
  Status Reserve(PoolHandle, std::size_t, std::vector<SlotLease>*) override;
  Status Commit(const std::vector<SlotLease>&) override;
  Status Release(const std::vector<SlotLease>&) override;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace kvaf
