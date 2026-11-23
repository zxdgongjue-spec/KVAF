#include "meta_manager.h"

#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace kvaf {
namespace {

// Pending and active objects both own a reference. A slot is recyclable only
// after the last object disappears; individual holes and closed-slot tails
// are not reused. All allocator and metadata operations share this mutex.
class AppendOnlyMetaManager final : public MetaManager {
  struct Slot { std::uint64_t used{0}; std::size_t references{0}; };
  struct Record {
    Address address;
    std::uint64_t length;
    std::size_t slot;
    std::size_t pins{0};
    bool committed{false};
  };
  struct Locked { KeyId key; Record* record; };
  struct Undo { std::size_t id; Slot before; };
  static constexpr std::size_t kNoSlot = std::numeric_limits<std::size_t>::max();

 public:
  Status RegisterRemoteRegion(const RemoteRegionDesc& region) override {
    std::lock_guard<std::mutex> guard(mutex_);
    if (registered_) return Status::AlreadyExists("remote region already registered");
    if (!region.length || !region.slot_size || region.length % region.slot_size ||
        region.base > UINT64_MAX - (region.length - 1) ||
        region.length / region.slot_size > std::numeric_limits<std::size_t>::max())
      return Status::InvalidArgument("invalid append_only region/slot extent");
    try {
      const auto count = static_cast<std::size_t>(region.length / region.slot_size);
      std::vector<Slot> slots(count);
      std::vector<std::size_t> free;
      free.reserve(count);
      for (auto i = count; i > 0; --i) free.push_back(i - 1);
      slots_ = std::move(slots);
      free_ = std::move(free);
    } catch (const std::bad_alloc&) {
      return Status::OutOfMemory("append_only slot metadata allocation failed");
    } catch (const std::length_error&) {
      return Status::OutOfMemory("append_only slot metadata capacity exceeded");
    }
    region_ = region;
    registered_ = true;
    return Status::OK();
  }

  Status PreparePut(const std::vector<KeyId>& keys,
                    const std::vector<std::uint64_t>& lengths,
                    PutContext* context, PutAccessDesc* desc) override {
    if (!context || !desc || keys.empty() || keys.size() != lengths.size())
      return Status::InvalidArgument("invalid append_only Put arguments");
    context->id = 0;
    desc->entries.clear();
    std::lock_guard<std::mutex> guard(mutex_);
    if (!registered_) return Status::NotInitialized("region not registered");
    if (!next_put_) return Status::InvalidState("Put context IDs exhausted");
    const auto original_current = current_;
    std::vector<Undo> undo;
    std::vector<std::size_t> taken;
    std::vector<KeyId> inserted;
    auto rollback = [&] {
      for (auto key : inserted) records_.erase(key);
      for (const auto& item : undo) slots_[item.id] = item.before;
      for (auto it = taken.rbegin(); it != taken.rend(); ++it) free_.push_back(*it);
      current_ = original_current;
      desc->entries.clear();
    };
    try {
      std::unordered_set<KeyId> distinct;
      distinct.reserve(keys.size());
      for (std::size_t i = 0; i < keys.size(); ++i) {
        if (!distinct.insert(keys[i]).second || !lengths[i] || lengths[i] > region_.slot_size)
          return Status::InvalidArgument("duplicate Key or length outside one slot");
        if (records_.count(keys[i]))
          return Status::AlreadyExists("Key already exists or is pending");
      }
      undo.reserve(keys.size());
      taken.reserve(keys.size());
      inserted.reserve(keys.size());
      desc->entries.reserve(keys.size());
      std::vector<KeyId> pending(keys);
      for (std::size_t i = 0; i < keys.size(); ++i) {
        if (current_ == kNoSlot || lengths[i] > region_.slot_size - slots_[current_].used) {
          if (free_.empty()) {
            rollback();
            return Status::OutOfMemory("no empty slot for append_only Put");
          }
          current_ = free_.back();
          free_.pop_back();
          taken.push_back(current_);
        }
        if (undo.empty() || undo.back().id != current_)
          undo.push_back(Undo{current_, slots_[current_]});
        auto& slot = slots_[current_];
        const auto address = region_.base + current_ * region_.slot_size + slot.used;
        slot.used += lengths[i];
        ++slot.references;
        records_.emplace(keys[i], Record{address, lengths[i], current_});
        inserted.push_back(keys[i]);
        // The transport expects the actual WRITE length, not slot capacity.
        desc->entries.push_back(PutAllocation{keys[i], address, lengths[i]});
      }
      pending_.emplace(next_put_, std::move(pending));
      context->id = next_put_++;
    } catch (const std::bad_alloc&) {
      rollback();
      return Status::OutOfMemory("append_only Put metadata allocation failed");
    } catch (const std::length_error&) {
      rollback();
      return Status::OutOfMemory("append_only Put metadata capacity exceeded");
    }
    return Status::OK();
  }

  Status CommitPut(const PutContext& context) override {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!registered_) return Status::NotInitialized("region not registered");
    auto it = pending_.find(context.id);
    if (it == pending_.end()) return Status::NotFound("Put context not found");
    for (auto key : it->second) {
      auto record = records_.find(key);
      if (record == records_.end() || record->second.committed)
        return Status::InvalidState("lost pending Key");
    }
    for (auto key : it->second) records_.at(key).committed = true;
    pending_.erase(it);
    return Status::OK();
  }

  Status AbortPut(const PutContext& context) override {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!registered_) return Status::NotInitialized("region not registered");
    auto it = pending_.find(context.id);
    if (it == pending_.end()) return Status::NotFound("Put context not found");
    for (auto key : it->second) {
      auto record = records_.find(key);
      if (record == records_.end() || record->second.committed)
        return Status::InvalidState("lost pending Key");
    }
    // Unlike a failed PreparePut under the mutex, later reservations can now
    // exist. Never rewind their cursor; aborted bytes become holes until reset.
    for (auto key : it->second) ReleaseRecord(key);
    pending_.erase(it);
    return Status::OK();
  }

  Status LockKeys(const std::vector<KeyId>& keys, MetaManagerLockMode mode,
                  LockContext* context) override {
    if (!context || keys.empty() || mode != MetaManagerLockMode::kStableReadAllowAppend)
      return Status::InvalidArgument("invalid lock arguments");
    std::lock_guard<std::mutex> guard(mutex_);
    if (!registered_) return Status::NotInitialized("region not registered");
    if (!next_lock_) return Status::InvalidState("lock context IDs exhausted");
    try {
      std::vector<Locked> items;
      items.reserve(keys.size());
      for (auto key : keys) {
        auto it = records_.find(key);
        if (it == records_.end() || !it->second.committed)
          return Status::NotFound("Key not found");
        // unordered_map node references survive rehash. Pinned records cannot
        // be erased, so later appends do not invalidate this snapshot.
        items.push_back(Locked{key, &it->second});
      }
      auto it = locks_.emplace(next_lock_, std::move(items)).first;
      for (auto& item : it->second) ++item.record->pins;
      context->id = next_lock_++;
    } catch (const std::bad_alloc&) {
      return Status::OutOfMemory("lock snapshot allocation failed");
    } catch (const std::length_error&) {
      return Status::OutOfMemory("lock snapshot capacity exceeded");
    }
    return Status::OK();
  }

  Status BatchLookup(const std::vector<KeyId>& keys, const LockContext& context,
                     FlatAccessDesc* desc) override {
    if (!desc) return Status::InvalidArgument("lookup output is null");
    std::lock_guard<std::mutex> guard(mutex_);
    if (!registered_) return Status::NotInitialized("region not registered");
    auto it = locks_.find(context.id);
    if (it == locks_.end()) return Status::NotFound("lock context not found");
    if (keys.size() != it->second.size()) return Status::InvalidArgument("lookup shape mismatch");
    for (std::size_t i = 0; i < keys.size(); ++i)
      if (keys[i] != it->second[i].key) return Status::InvalidArgument("lookup Key order mismatch");
    try {
      desc->addresses.clear(); desc->lengths.clear(); desc->valid.clear();
      desc->addresses.reserve(keys.size()); desc->lengths.reserve(keys.size()); desc->valid.reserve(keys.size());
      for (const auto& item : it->second) {
        desc->addresses.push_back(item.record->address);
        desc->lengths.push_back(item.record->length);
        desc->valid.push_back(1);
      }
    } catch (const std::bad_alloc&) {
      return Status::OutOfMemory("lookup allocation failed");
    } catch (const std::length_error&) {
      return Status::OutOfMemory("lookup capacity exceeded");
    }
    return Status::OK();
  }

  Status UnlockKeys(const LockContext& context) override {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!registered_) return Status::NotInitialized("region not registered");
    auto it = locks_.find(context.id);
    if (it == locks_.end()) return Status::NotFound("lock context not found");
    for (auto& item : it->second) --item.record->pins;
    locks_.erase(it);
    return Status::OK();
  }

  Status BatchFree(const std::vector<KeyId>& keys) override {
    if (keys.empty()) return Status::InvalidArgument("empty free request");
    std::lock_guard<std::mutex> guard(mutex_);
    if (!registered_) return Status::NotInitialized("region not registered");
    try {
      std::unordered_set<KeyId> distinct;
      distinct.reserve(keys.size());
      for (auto key : keys) {
        if (!distinct.insert(key).second) return Status::InvalidArgument("duplicate free Key");
        auto it = records_.find(key);
        if (it == records_.end() || !it->second.committed) return Status::NotFound("Key not found");
        if (it->second.pins) return Status::Busy("Key pinned by an AccessTable");
      }
    } catch (const std::bad_alloc&) {
      return Status::OutOfMemory("free validation allocation failed");
    } catch (const std::length_error&) {
      return Status::OutOfMemory("free validation capacity exceeded");
    }
    for (auto key : keys) ReleaseRecord(key);
    return Status::OK();
  }

 private:
  void ReleaseRecord(KeyId key) {
    auto it = records_.find(key);
    const auto id = it->second.slot;
    records_.erase(it);
    if (--slots_[id].references == 0) {
      slots_[id].used = 0;
      if (current_ == id) current_ = kNoSlot;
      free_.push_back(id);  // capacity reserved at registration; cannot allocate
    }
  }

  std::mutex mutex_;
  RemoteRegionDesc region_;
  bool registered_{false};
  std::size_t current_{kNoSlot};
  std::vector<Slot> slots_;
  std::vector<std::size_t> free_;
  std::unordered_map<KeyId, Record> records_;
  std::unordered_map<std::uint64_t, std::vector<KeyId>> pending_;
  std::unordered_map<std::uint64_t, std::vector<Locked>> locks_;
  std::uint64_t next_put_{1}, next_lock_{1};
};
}  // namespace

std::unique_ptr<MetaManager> CreateInMemoryMetaManager(AllocationMode mode) {
  switch (mode) {
    case AllocationMode::kFixed: return CreateInMemoryMetaManager();
    case AllocationMode::kAppendOnly: return std::make_unique<AppendOnlyMetaManager>();
  }
  throw std::invalid_argument("unknown allocation mode");
}
}  // namespace kvaf
