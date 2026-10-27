#include "pool_metadata.h"
#include "meta_manager.h"

#include <limits>
#include <mutex>
#include <new>
#include <unordered_map>
#include <unordered_set>

namespace kvaf {
namespace {
struct Key {
  std::uint64_t owner;
  std::uint64_t layer;
  KeyId key;
  bool operator==(const Key& rhs) const { return owner == rhs.owner && layer == rhs.layer && key == rhs.key; }
};
struct Hash {
  std::size_t operator()(const Key& k) const {
    return std::hash<std::uint64_t>{}(k.owner) ^
        (std::hash<std::uint64_t>{}(k.layer) << 1) ^ (std::hash<KeyId>{}(k.key) << 2);
  }
};
bool Distinct(const std::vector<KeyId>& keys) {
  return std::unordered_set<KeyId>(keys.begin(), keys.end()).size() == keys.size();
}
}

struct PoolMetadata::Impl {
  struct Record { SlotLease lease; std::uint64_t length; std::uint32_t pins{0}; bool active{false}; };
  struct Pending { std::vector<Key> keys; std::vector<SlotLease> leases; };
  struct Locked { std::vector<KeyId> keys; std::vector<Record*> records; };
  mutable std::mutex mutex;
  std::unordered_map<Key, Record, Hash> records;
  std::unordered_map<std::uint64_t, Pending> pending;
  std::unordered_map<std::uint64_t, Locked> locks;
  std::uint64_t next_put{1}, next_lock{1};
};

PoolMetadata::PoolMetadata() : impl_(std::make_unique<Impl>()) {}
PoolMetadata::~PoolMetadata() = default;

Status PoolMetadata::PreparePut(SlotAllocationProvider& slots, const LayerLayout& layer,
    const std::vector<KeyId>& keys, const std::vector<std::uint64_t>& lengths,
    PutContext* context, PutAccessDesc* output) {
  if (!context || !output || keys.empty() || keys.size() != lengths.size() || !Distinct(keys))
    return Status::InvalidArgument("invalid scoped Put keys/lengths");
  std::lock_guard<std::mutex> lock(impl_->mutex);
  for (std::size_t i = 0; i < keys.size(); ++i) {
    if (!lengths[i] || lengths[i] > layer.spec.slot_size)
      return Status::InvalidArgument("Put length exceeds pool slot_size");
    if (impl_->records.count(Key{layer.layer.owner, layer.layer.id, keys[i]}))
      return Status::AlreadyExists("key exists or is pending in this layer");
  }
  if (impl_->next_put == std::numeric_limits<std::uint64_t>::max())
    return Status::InvalidState("Put context IDs exhausted");
  Impl::Pending pending;
  PutAccessDesc result;
  try {
    pending.keys.reserve(keys.size());
    result.entries.reserve(keys.size());
    impl_->records.reserve(impl_->records.size() + keys.size());
    impl_->pending.reserve(impl_->pending.size() + 1);
    for (KeyId key : keys) pending.keys.push_back(Key{layer.layer.owner, layer.layer.id, key});
  } catch (const std::bad_alloc&) { return Status::OutOfMemory("Put metadata allocation failed"); }
  auto status = slots.Reserve(layer.pool, keys.size(), &pending.leases);
  if (!status.ok()) return status;
  std::size_t inserted = 0;
  try {
    for (std::size_t i = 0; i < keys.size(); ++i) {
      impl_->records.emplace(pending.keys[i], Impl::Record{pending.leases[i], lengths[i]});
      ++inserted;
      result.entries.push_back(PutAllocation{keys[i], pending.leases[i].address, lengths[i]});
    }
    // Copy before publication: on allocation failure pending still owns leases.
    impl_->pending.emplace(impl_->next_put, pending);
  } catch (const std::bad_alloc&) {
    for (std::size_t i = 0; i < inserted; ++i) impl_->records.erase(pending.keys[i]);
    (void)slots.Release(pending.leases);
    return Status::OutOfMemory("Put metadata allocation failed");
  }
  context->id = impl_->next_put++;
  *output = std::move(result);
  return Status::OK();
}

Status PoolMetadata::CommitPut(SlotAllocationProvider& slots, const PutContext& context) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  auto found = impl_->pending.find(context.id);
  if (found == impl_->pending.end()) return Status::NotFound("Put context not found");
  auto status = slots.Commit(found->second.leases);
  if (!status.ok()) return status;
  for (const auto& key : found->second.keys) impl_->records.at(key).active = true;
  impl_->pending.erase(found);
  return Status::OK();
}

Status PoolMetadata::AbortPut(SlotAllocationProvider& slots, const PutContext& context) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  auto found = impl_->pending.find(context.id);
  if (found == impl_->pending.end()) return Status::NotFound("Put context not found");
  auto status = slots.Release(found->second.leases);
  if (!status.ok()) return status;
  for (const auto& key : found->second.keys) impl_->records.erase(key);
  impl_->pending.erase(found);
  return Status::OK();
}

Status PoolMetadata::LockKeys(const LayerLayout& layer, const std::vector<KeyId>& keys,
                              LockContext* context) {
  if (!context || keys.empty()) return Status::InvalidArgument("empty keys or null LockContext");
  if (keys.size() > std::numeric_limits<std::uint32_t>::max())
    return Status::InvalidArgument("too many keys for pin counts");
  std::lock_guard<std::mutex> lock(impl_->mutex);
  Impl::Locked locked;
  try {
    locked.keys = keys;
    locked.records.reserve(keys.size());
    for (KeyId key : keys) {
      auto record = impl_->records.find(Key{layer.layer.owner, layer.layer.id, key});
      if (record == impl_->records.end() || !record->second.active)
        return Status::NotFound("key not found in this layer");
      if (record->second.pins > std::numeric_limits<std::uint32_t>::max() - keys.size())
        return Status::InvalidState("pin count exhausted");
      locked.records.push_back(&record->second);
    }
    if (impl_->next_lock == std::numeric_limits<std::uint64_t>::max())
      return Status::InvalidState("Lock context IDs exhausted");
    auto inserted = impl_->locks.emplace(impl_->next_lock, std::move(locked));
    for (auto* record : inserted.first->second.records) ++record->pins;
  } catch (const std::bad_alloc&) { return Status::OutOfMemory("Lock metadata allocation failed"); }
  context->id = impl_->next_lock++;
  return Status::OK();
}

Status PoolMetadata::Lookup(const std::vector<KeyId>& keys, const LockContext& context,
                            FlatAccessDesc* output) const {
  if (!output) return Status::InvalidArgument("null lookup output");
  std::lock_guard<std::mutex> lock(impl_->mutex);
  auto found = impl_->locks.find(context.id);
  if (found == impl_->locks.end()) return Status::NotFound("Lock context not found");
  if (keys != found->second.keys) return Status::InvalidArgument("keys do not match lock");
  FlatAccessDesc result;
  try {
    for (auto* record : found->second.records) {
      result.addresses.push_back(record->lease.address);
      result.lengths.push_back(record->length);
      result.valid.push_back(1);
    }
  } catch (const std::bad_alloc&) { return Status::OutOfMemory("Lookup allocation failed"); }
  *output = std::move(result);
  return Status::OK();
}

Status PoolMetadata::UnlockKeys(const LockContext& context) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  auto found = impl_->locks.find(context.id);
  if (found == impl_->locks.end()) return Status::NotFound("Lock context not found");
  for (auto* record : found->second.records) --record->pins;
  impl_->locks.erase(found);
  return Status::OK();
}

Status PoolMetadata::Free(SlotAllocationProvider& slots, const LayerLayout& layer,
                          const std::vector<KeyId>& keys) {
  if (keys.empty() || !Distinct(keys)) return Status::InvalidArgument("empty or duplicate Free keys");
  std::lock_guard<std::mutex> lock(impl_->mutex);
  std::vector<SlotLease> leases;
  try { leases.reserve(keys.size()); }
  catch (const std::bad_alloc&) { return Status::OutOfMemory("Free metadata allocation failed"); }
  for (KeyId key : keys) {
    auto found = impl_->records.find(Key{layer.layer.owner, layer.layer.id, key});
    if (found == impl_->records.end() || !found->second.active)
      return Status::NotFound("Free key not found in this layer");
    if (found->second.pins) return Status::Busy("key pinned by AccessTable");
    leases.push_back(found->second.lease);
  }
  auto status = slots.Release(leases);
  if (!status.ok()) return status;
  for (KeyId key : keys) impl_->records.erase(Key{layer.layer.owner, layer.layer.id, key});
  return Status::OK();
}

std::size_t PoolMetadata::CountLayer(LayerHandle layer) const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  std::size_t count = 0;
  for (const auto& item : impl_->records)
    count += item.first.owner == layer.owner && item.first.layer == layer.id;
  return count;
}
std::size_t PoolMetadata::CountPool(PoolHandle pool) const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  std::size_t count = 0;
  for (const auto& item : impl_->records)
    count += item.second.lease.pool.owner == pool.owner && item.second.lease.pool.id == pool.id;
  return count;
}
Status PoolMetadata::ErasePool(PoolHandle pool) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  for (const auto& item : impl_->records)
    if (item.second.lease.pool.owner == pool.owner && item.second.lease.pool.id == pool.id && item.second.pins)
      return Status::Busy("pool metadata is pinned");
  for (auto it = impl_->pending.begin(); it != impl_->pending.end();) {
    if (!it->second.leases.empty() && it->second.leases.front().pool.owner == pool.owner &&
        it->second.leases.front().pool.id == pool.id)
      it = impl_->pending.erase(it);
    else ++it;
  }
  for (auto it = impl_->records.begin(); it != impl_->records.end();) {
    if (it->second.lease.pool.owner == pool.owner && it->second.lease.pool.id == pool.id)
      it = impl_->records.erase(it);
    else ++it;
  }
  return Status::OK();
}
}  // namespace kvaf
