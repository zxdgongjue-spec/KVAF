#include "pool_manager.h"
#include "runtime.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <new>
#include <unordered_map>
#include <unordered_set>

namespace kvaf {
namespace {
std::atomic<std::uint64_t> next_owner{1};
enum class Slot { kFree, kReserved, kCommitted, kQuarantined };
}
struct PoolManager::Impl {
  struct Pool {
    PoolHandle handle;
    PoolSpec spec;
    RemoteRegionDesc region;
    std::unique_ptr<AccessOperator> op;
    std::vector<Slot> slots;
    std::vector<std::uint64_t> free;
    bool stopped{false}, quarantined{false};
  };
  DdrMemoryAllocator& allocator;
  AccessOperatorFactory factory;
  const std::uint64_t owner{next_owner.fetch_add(1)};
  std::uint64_t next_pool{1}, next_layer{1};
  std::unordered_map<std::uint64_t, std::unique_ptr<Pool>> pools;
  std::unordered_map<std::uint64_t, LayerLayout> layers;
  Impl(DdrMemoryAllocator& a, AccessOperatorFactory f) : allocator(a), factory(std::move(f)) {}
  Pool* Find(PoolHandle handle) const {
    auto found = pools.find(handle.id);
    return handle.owner == owner && found != pools.end() ? found->second.get() : nullptr;
  }
  Status Transition(const std::vector<SlotLease>& leases, bool commit) {
    std::unordered_set<std::uint64_t> seen;
    if (leases.empty()) return Status::InvalidArgument("empty leases");
    auto* pool = Find(leases.front().pool);
    if (!pool) return Status::NotFound("pool handle not found");
    try {
      seen.reserve(leases.size());
      for (const auto& lease : leases) {
        if (lease.pool.owner != owner || lease.pool.id != pool->handle.id ||
            lease.slot >= pool->slots.size() || !seen.insert(lease.slot).second ||
            lease.address != pool->region.base + lease.slot * pool->spec.slot_size ||
            (commit ? pool->slots[lease.slot] != Slot::kReserved :
                      pool->slots[lease.slot] == Slot::kFree))
          return Status::InvalidState("invalid slot lease transition");
      }
    } catch (const std::bad_alloc&) { return Status::OutOfMemory("lease validation allocation failed"); }
    for (const auto& lease : leases) {
      pool->slots[lease.slot] = commit ? Slot::kCommitted : Slot::kFree;
      if (!commit) pool->free.push_back(lease.slot);
    }
    return Status::OK();
  }
};

PoolManager::PoolManager(DdrMemoryAllocator& allocator, AccessOperatorFactory factory)
    : impl_(std::make_unique<Impl>(allocator, std::move(factory))) {}
PoolManager::~PoolManager() = default;

Status PoolManager::Create(const PoolSpec& spec, PoolHandle* output) {
  if (!output || !impl_->factory || spec.name.empty() || !spec.slot_count ||
      !spec.slot_size || !spec.token_bytes || !spec.tokens_per_block ||
      std::uint64_t{spec.token_bytes} * spec.tokens_per_block > spec.slot_size ||
      spec.slot_size > std::numeric_limits<std::uint64_t>::max() / spec.slot_count ||
      spec.slot_count > std::vector<Slot>().max_size() ||
      spec.slot_count > std::vector<std::uint64_t>().max_size() ||
      spec.slot_count * spec.slot_size > std::numeric_limits<std::size_t>::max())
    return Status::InvalidArgument("invalid pool capacity or layout");
  for (const auto& item : impl_->pools)
    if (item.second->spec.name == spec.name) return Status::AlreadyExists("duplicate pool name");
  auto pool = std::make_unique<Impl::Pool>();
  try {
    pool->spec = spec;
    pool->slots.assign(static_cast<std::size_t>(spec.slot_count), Slot::kFree);
    pool->free.reserve(static_cast<std::size_t>(spec.slot_count));
    for (auto i = spec.slot_count; i > 0; --i) pool->free.push_back(i - 1);
    pool->op = impl_->factory();
    impl_->pools.reserve(impl_->pools.size() + 1);
  } catch (const std::bad_alloc&) { return Status::OutOfMemory("pool allocation failed"); }
  if (!pool->op) return Status::InvalidArgument("operator factory returned null");
  pool->handle = PoolHandle{impl_->owner, impl_->next_pool++};
  const auto handle = pool->handle;
  // Publish an internal owner before allocation. Failed initialization is not
  // returned to the caller, but resources survive failed cleanup for retry.
  auto* state = pool.get();
  try { impl_->pools.emplace(handle.id, std::move(pool)); }
  catch (const std::bad_alloc&) { return Status::OutOfMemory("pool publication failed"); }
  const RuntimeInitRequest request{spec.slot_count * spec.slot_size, spec.slot_size, DeviceKind::kCpu};
  auto status = impl_->allocator.Allocate(request, &state->region);
  if (!status.ok()) { impl_->pools.erase(handle.id); return status; }
  const auto& region = state->region;
  auto discard = [&] {
    auto released = impl_->allocator.Release(region);
    if (released.ok()) impl_->pools.erase(handle.id);
    else { state->stopped = true; Quarantine(handle); }
  };
  if (!region.base || region.length != request.ddr_bytes || region.slot_size != spec.slot_size ||
      region.device != DeviceKind::kCpu || region.length > std::numeric_limits<Address>::max() - region.base) {
    discard();
    return Status::InvalidArgument("allocator returned invalid pool region");
  }
  for (const auto& item : impl_->pools) {
    if (item.first == handle.id) continue;
    const auto& existing = item.second->region;
    if (region.base < existing.base + existing.length && existing.base < region.base + region.length) {
      discard();
      return Status::InvalidArgument("allocator returned overlapping pool regions");
    }
  }
  status = state->op->Initialize(region);
  if (!status.ok()) {
    auto stopped = state->op->Shutdown();
    if (stopped.ok()) discard();
    else Quarantine(handle);
    return status;
  }
  *output = handle;
  return Status::OK();
}

Status PoolManager::Info(PoolHandle handle, PoolInfo* output) const {
  auto* pool = impl_->Find(handle);
  if (!pool) return Status::NotFound("pool handle not found");
  if (!output) return Status::InvalidArgument("null pool info");
  PoolInfo info;
  info.handle = handle; info.spec = pool->spec; info.region = pool->region;
  info.quarantined = pool->quarantined;
  for (auto slot : pool->slots) {
    info.free_slots += slot == Slot::kFree;
    info.reserved_slots += slot == Slot::kReserved;
    info.committed_slots += slot == Slot::kCommitted;
    info.quarantined_slots += slot == Slot::kQuarantined;
  }
  for (const auto& layer : impl_->layers) info.bound_layers += layer.second.pool.id == handle.id;
  *output = std::move(info);
  return Status::OK();
}

Status PoolManager::Bind(std::uint64_t ns, std::uint64_t layer_id, PoolHandle handle, LayerHandle* output) {
  auto* pool = impl_->Find(handle);
  if (!pool) return Status::NotFound("pool handle not found");
  if (!output) return Status::InvalidArgument("null layer handle");
  if (pool->quarantined || pool->stopped) return Status::InvalidState("pool unavailable");
  for (const auto& layer : impl_->layers)
    if (layer.second.namespace_id == ns && layer.second.layer_id == layer_id)
      return Status::AlreadyExists("layer already bound");
  LayerHandle layer{impl_->owner, impl_->next_layer++};
  impl_->layers.emplace(layer.id, LayerLayout{layer, handle, ns, layer_id, pool->spec, pool->region});
  *output = layer;
  return Status::OK();
}
Status PoolManager::FindLayer(std::uint64_t ns, std::uint64_t layer_id, LayerHandle* output) const {
  if (!output) return Status::InvalidArgument("null layer handle");
  for (const auto& layer : impl_->layers) {
    if (layer.second.namespace_id == ns && layer.second.layer_id == layer_id) {
      *output = layer.second.layer;
      return Status::OK();
    }
  }
  return Status::NotFound("layer not bound");
}
Status PoolManager::Resolve(LayerHandle handle, LayerLayout* output, bool allow_quarantined) const {
  if (!output) return Status::InvalidArgument("null layer layout");
  auto found = impl_->layers.find(handle.id);
  if (handle.owner != impl_->owner || found == impl_->layers.end())
    return Status::NotFound("stale or foreign layer handle");
  auto* pool = impl_->Find(found->second.pool);
  if (!pool || (!allow_quarantined && (pool->quarantined || pool->stopped)))
    return Status::InvalidState("pool unavailable");
  *output = found->second;
  return Status::OK();
}
Status PoolManager::Unbind(LayerHandle handle) {
  LayerLayout layout;
  auto status = Resolve(handle, &layout, true);
  if (status.ok()) impl_->layers.erase(handle.id);
  return status;
}
AccessOperator* PoolManager::Operator(PoolHandle handle) const {
  auto* pool = impl_->Find(handle);
  return pool && !pool->stopped ? pool->op.get() : nullptr;
}
void PoolManager::UnbindPool(PoolHandle handle) {
  for (auto it = impl_->layers.begin(); it != impl_->layers.end();) {
    if (it->second.pool.id == handle.id) it = impl_->layers.erase(it);
    else ++it;
  }
}
Status PoolManager::Stop(PoolHandle handle) {
  auto* pool = impl_->Find(handle);
  if (!pool) return Status::NotFound("pool handle not found");
  if (pool->stopped) return Status::OK();
  auto status = pool->op->Shutdown();
  if (status.ok()) pool->stopped = true;
  else Quarantine(handle);
  return status;
}
Status PoolManager::Destroy(PoolHandle handle) {
  PoolInfo info;
  auto status = Info(handle, &info);
  if (!status.ok()) return status;
  if (info.bound_layers) return Status::Busy("pool still has bound layers");
  status = Stop(handle);
  if (!status.ok()) return status;
  auto* pool = impl_->Find(handle);
  status = impl_->allocator.Release(pool->region);
  if (!status.ok()) { Quarantine(handle); return status; }
  impl_->pools.erase(handle.id);
  return Status::OK();
}
void PoolManager::Quarantine(PoolHandle handle) {
  auto* pool = impl_->Find(handle);
  if (!pool) return;
  pool->quarantined = true;
  for (auto& slot : pool->slots) if (slot == Slot::kReserved) slot = Slot::kQuarantined;
}
std::vector<PoolHandle> PoolManager::Handles() const {
  std::vector<PoolHandle> handles;
  for (const auto& item : impl_->pools) handles.push_back(item.second->handle);
  return handles;
}
Status PoolManager::Reserve(PoolHandle handle, std::size_t count, std::vector<SlotLease>* output) {
  auto* pool = impl_->Find(handle);
  if (!pool) return Status::NotFound("pool handle not found");
  if (!output || !count) return Status::InvalidArgument("invalid reservation");
  if (pool->quarantined || pool->stopped) return Status::InvalidState("pool unavailable");
  if (pool->free.size() < count) return Status::OutOfMemory("pool has insufficient free slots");
  std::vector<SlotLease> leases;
  try {
    leases.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      const auto slot = pool->free[pool->free.size() - i - 1];
      leases.push_back(SlotLease{handle, slot, pool->region.base + slot * pool->spec.slot_size});
    }
  } catch (const std::bad_alloc&) { return Status::OutOfMemory("lease allocation failed"); }
  for (const auto& lease : leases) pool->slots[lease.slot] = Slot::kReserved;
  pool->free.resize(pool->free.size() - count);
  *output = std::move(leases);
  return Status::OK();
}
Status PoolManager::Commit(const std::vector<SlotLease>& leases) { return impl_->Transition(leases, true); }
Status PoolManager::Release(const std::vector<SlotLease>& leases) { return impl_->Transition(leases, false); }
}  // namespace kvaf
