#include "meta_manager.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace kvaf {

namespace {

// 单条 key 当前绑定的地址记录。
struct AddressRecord {
  std::uint64_t record_id{0};
  Address uba{0};
  std::uint64_t valid_length{0};
  std::size_t slot_id{0};
  std::uint32_t pin_count{0};
};

// PreparePut 阶段的临时预留项。
struct PendingPutItem {
  KeyId key{0};
  std::size_t slot_id{0};
  std::uint64_t valid_length{0};
  std::size_t flat_index_slot{0};
};

// 一次 PreparePut 对应的事务上下文。
struct PendingPutContext {
  std::vector<PendingPutItem> items;
};

// LockKeys 固定下来的一条快照项。
struct LockedItem {
  KeyId key{0};
  AddressRecord* record{nullptr};
};

// 一次稳定读快照包含的全部 key。
struct ActiveLockContext {
  std::vector<LockedItem> items;
};

Status InvalidArgument(std::string message) {
  return Status::InvalidArgument(std::move(message));
}

Status NotInitialized(std::string message) {
  return Status::NotInitialized(std::move(message));
}

Status NotFound(std::string message) {
  return Status::NotFound(std::move(message));
}

Status AlreadyExists(std::string message) {
  return Status::Error(StatusCode::kAlreadyExists, std::move(message));
}

Status Busy(std::string message) {
  return Status::Error(StatusCode::kBusy, std::move(message));
}

Status InvalidState(std::string message) {
  return Status::InvalidState(std::move(message));
}

// 批量接口要求 key 唯一，避免重复申请或重复释放。
bool HasDistinctKeys(const std::vector<KeyId>& keys) {
  // BatchPut callers commonly provide monotonically increasing key ranges.
  // This avoids allocating a temporary hash node for every key on that path.
  if (keys.size() < 2) {
    return true;
  }
  bool strictly_increasing = true;
  for (std::size_t index = 1; index < keys.size(); ++index) {
    if (keys[index] <= keys[index - 1]) {
      strictly_increasing = false;
      break;
    }
  }
  if (strictly_increasing) {
    return true;
  }

  std::unordered_set<KeyId> seen;
  seen.reserve(keys.size());
  for (KeyId key : keys) {
    if (!seen.insert(key).second) {
      return false;
    }
  }
  return true;
}

bool CalculateTableCapacity(std::size_t expected_entries,
                            std::size_t* capacity) {
  if (capacity == nullptr ||
      expected_entries >
          (std::numeric_limits<std::size_t>::max() - 1U) / 2U) {
    return false;
  }
  const std::size_t desired = std::max<std::size_t>(
      1U, expected_entries == 0U ? 1U : expected_entries * 2U);
  std::size_t table_capacity = 8U;
  while (table_capacity < desired) {
    if (table_capacity > std::numeric_limits<std::size_t>::max() / 2U) {
      return false;
    }
    table_capacity *= 2U;
  }
  *capacity = table_capacity;
  return true;
}

std::size_t MixKey(KeyId key) {
  std::uint64_t value = key + UINT64_C(0x9e3779b97f4a7c15);
  value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);
  value ^= value >> 31U;
  return static_cast<std::size_t>(value);
}

// Fixed-capacity open-addressing index. A key is inserted once as pending by
// PreparePut and changed in place to active by CommitPut. This avoids a second
// hash-table insertion and keeps both visibility states in one preallocated
// table.
class FlatKeyIndex final {
 public:
  bool Reserve(std::size_t expected_entries) {
    std::size_t desired_capacity = 0U;
    if (!CalculateTableCapacity(expected_entries, &desired_capacity)) {
      return false;
    }
    if (entries_.size() >= desired_capacity) {
      return true;
    }

    std::vector<Entry> old_entries = std::move(entries_);
    entries_.assign(desired_capacity, Entry{});
    mask_ = desired_capacity - 1U;
    size_ = 0U;
    for (const Entry& entry : old_entries) {
      if (entry.state == kPending || entry.state == kActive) {
        if (!Insert(entry.key, entry.slot_id, entry.state, nullptr)) {
          return false;
        }
      }
    }
    return true;
  }

  bool empty() const { return size_ == 0U; }
  std::size_t size() const { return size_; }

  bool FindActive(KeyId key, std::size_t* slot_id) const {
    const Entry* entry = FindEntry(key);
    if (entry == nullptr || entry->state != kActive) {
      return false;
    }
    if (slot_id != nullptr) {
      *slot_id = entry->slot_id;
    }
    return true;
  }

  bool InsertPending(KeyId key, std::size_t slot_id,
                     std::size_t* flat_index_slot) {
    return Insert(key, slot_id, kPending, flat_index_slot);
  }

  bool ActivatePendingAt(KeyId key, std::size_t slot_id,
                         std::size_t flat_index_slot) {
    if (flat_index_slot >= entries_.size()) {
      return false;
    }
    Entry& entry = entries_[flat_index_slot];
    if (entry.state != kPending || entry.key != key ||
        entry.slot_id != slot_id) {
      return false;
    }
    entry.state = kActive;
    return true;
  }

  bool ErasePendingAt(KeyId key, std::size_t slot_id,
                      std::size_t flat_index_slot) {
    if (flat_index_slot >= entries_.size()) {
      return false;
    }
    Entry& entry = entries_[flat_index_slot];
    if (entry.state != kPending || entry.key != key ||
        entry.slot_id != slot_id) {
      return false;
    }
    entry.state = kDeleted;
    --size_;
    return true;
  }

  bool EraseActive(KeyId key, std::size_t slot_id) {
    return Erase(key, slot_id, kActive);
  }

  void Clear() {
    for (Entry& entry : entries_) {
      entry.state = kEmpty;
    }
    size_ = 0U;
  }

 private:
  enum : std::uint8_t {
    kEmpty = 0U,
    kPending = 1U,
    kActive = 2U,
    kDeleted = 3U,
  };

  struct Entry {
    KeyId key{0};
    std::size_t slot_id{0};
    std::uint8_t state{kEmpty};
  };

  const Entry* FindEntry(KeyId key) const {
    if (entries_.empty()) {
      return nullptr;
    }
    const std::size_t start = MixKey(key) & mask_;
    for (std::size_t probe = 0U; probe < entries_.size(); ++probe) {
      const Entry& entry = entries_[(start + probe) & mask_];
      if (entry.state == kEmpty) {
        return nullptr;
      }
      if ((entry.state == kPending || entry.state == kActive) &&
          entry.key == key) {
        return &entry;
      }
    }
    return nullptr;
  }

  Entry* FindEntry(KeyId key) {
    return const_cast<Entry*>(
        static_cast<const FlatKeyIndex*>(this)->FindEntry(key));
  }

  bool Insert(KeyId key, std::size_t slot_id, std::uint8_t state,
              std::size_t* flat_index_slot) {
    if (entries_.empty()) {
      return false;
    }
    const std::size_t start = MixKey(key) & mask_;
    std::size_t first_deleted = entries_.size();
    for (std::size_t probe = 0U; probe < entries_.size(); ++probe) {
      Entry& entry = entries_[(start + probe) & mask_];
      if (entry.state == kPending || entry.state == kActive) {
        if (entry.key == key) {
          return false;
        }
        continue;
      }
      if (entry.state == kDeleted) {
        if (first_deleted == entries_.size()) {
          first_deleted = (start + probe) & mask_;
        }
        continue;
      }
      const std::size_t target = first_deleted == entries_.size()
                                     ? (start + probe) & mask_
                                     : first_deleted;
      entries_[target] = Entry{key, slot_id, state};
      ++size_;
      if (flat_index_slot != nullptr) {
        *flat_index_slot = target;
      }
      return true;
    }
    if (first_deleted != entries_.size()) {
      entries_[first_deleted] = Entry{key, slot_id, state};
      ++size_;
      if (flat_index_slot != nullptr) {
        *flat_index_slot = first_deleted;
      }
      return true;
    }
    return false;
  }

  bool Erase(KeyId key, std::size_t slot_id, std::uint8_t expected_state) {
    Entry* entry = FindEntry(key);
    if (entry == nullptr || entry->state != expected_state ||
        entry->slot_id != slot_id) {
      return false;
    }
    entry->state = kDeleted;
    --size_;
    return true;
  }

  std::vector<Entry> entries_;
  std::size_t mask_{0U};
  std::size_t size_{0U};
};

class InMemoryMetaManager final : public MetaManager {
 public:
  // 注册单个远端 region，并按固定 slot 粒度初始化空闲队列。
  Status RegisterRemoteRegion(const RemoteRegionDesc& region) override {
    if (region.length == 0) {
      return InvalidArgument("remote region length must be greater than zero");
    }
    if (region.slot_size == 0) {
      return InvalidArgument("slot_size must be greater than zero");
    }
    if (region.length % region.slot_size != 0) {
      return InvalidArgument(
          "remote region length must be a multiple of slot_size");
    }

    std::lock_guard<std::mutex> lock(meta_mutex_);
    if (region_registered_) {
      return AlreadyExists("remote region has already been registered");
    }

    const std::uint64_t slot_count = region.length / region.slot_size;
    if (slot_count > std::numeric_limits<std::size_t>::max()) {
      return InvalidArgument("remote region slot count exceeds host size_t");
    }
    const std::size_t slot_count_size = static_cast<std::size_t>(slot_count);
    try {
      records_.assign(slot_count_size, AddressRecord{});
      if (!key_index_.Reserve(slot_count_size)) {
        return Status::OutOfMemory("failed to reserve metadata index capacity");
      }
      free_slots_.reserve(slot_count_size);
      // BatchPut normally has one in-flight context. Reserve its bucket so
      // steady-state emplace does not trigger a bucket-table rehash.
      pending_puts_.reserve(1U);
    } catch (const std::bad_alloc&) {
      return Status::OutOfMemory("failed to allocate metadata index storage");
    }

    region_ = region;
    region_registered_ = true;
    // Push in reverse so the first allocation still receives slot zero.
    for (std::uint64_t i = slot_count; i > 0U; --i) {
      free_slots_.push_back(static_cast<std::size_t>(i - 1U));
    }

    return Status::OK();
  }

  // 为一批新 key 预留 slot，并返回写路径需要的地址描述。
  Status PreparePut(const std::vector<KeyId>& flat_keys,
                    const std::vector<std::uint64_t>& lengths,
                    PutContext* put_context,
                    PutAccessDesc* put_desc) override {
    if (put_context == nullptr || put_desc == nullptr) {
      return InvalidArgument("put_context and put_desc must not be null");
    }
    if (flat_keys.empty()) {
      return InvalidArgument("PreparePut requires at least one key");
    }
    if (flat_keys.size() != lengths.size()) {
      return InvalidArgument("keys and lengths must have the same size");
    }
    if (!HasDistinctKeys(flat_keys)) {
      return InvalidArgument("PreparePut does not accept duplicate keys");
    }
    std::lock_guard<std::mutex> lock(meta_mutex_);
    if (!region_registered_) {
      return NotInitialized("remote region has not been registered");
    }
    if (free_slots_.size() < flat_keys.size()) {
      return Status::Error(StatusCode::kOutOfMemory,
                           "not enough free slots for PreparePut");
    }

    for (std::size_t i = 0; i < flat_keys.size(); ++i) {
      if (lengths[i] == 0) {
        return InvalidArgument("PreparePut length must be greater than zero");
      }
      if (lengths[i] > region_.slot_size) {
        return InvalidArgument("PreparePut length exceeds slot_size");
      }
    }
    // The flat index was sized from the region slot count at registration, so
    // publishing this transaction does not rehash or allocate hash nodes.
    const std::uint64_t ctx_id = next_put_context_id_++;
    PendingPutContext pending_context;
    pending_context.items.reserve(flat_keys.size());
    put_desc->entries.clear();
    put_desc->entries.reserve(flat_keys.size());

    // Slots leave the free stack here, but remain invisible until CommitPut.
    for (std::size_t i = 0; i < flat_keys.size(); ++i) {
      const std::size_t slot_id = free_slots_.back();
      free_slots_.pop_back();
      pending_context.items.push_back(PendingPutItem{0U, slot_id, 0U, 0U});
    }
    for (std::size_t i = 0; i < flat_keys.size(); ++i) {
      PendingPutItem& item = pending_context.items[i];
      item.key = flat_keys[i];
      item.valid_length = lengths[i];
      put_desc->entries.push_back(PutAllocation{
          item.key, SlotAddress(item.slot_id), item.valid_length});
    }
    for (std::size_t index = 0U; index < flat_keys.size(); ++index) {
      PendingPutItem& item = pending_context.items[index];
      if (!key_index_.InsertPending(item.key, item.slot_id,
                                    &item.flat_index_slot)) {
        // A free slot proves the preallocated index has capacity. InsertPending
        // can therefore fail only for a committed or pending key. Restore the
        // transaction state before returning the duplicate-key error.
        for (std::size_t rollback = 0U; rollback < index; ++rollback) {
          const PendingPutItem& reserved = pending_context.items[rollback];
          key_index_.ErasePendingAt(reserved.key, reserved.slot_id,
                                    reserved.flat_index_slot);
        }
        for (auto item = pending_context.items.rbegin();
             item != pending_context.items.rend(); ++item) {
          free_slots_.push_back(item->slot_id);
        }
        return AlreadyExists("PreparePut key already exists or is pending");
      }
    }
    pending_puts_.emplace(ctx_id, std::move(pending_context));
    put_context->id = ctx_id;
    return Status::OK();
  }

  // 将 PreparePut 产生的预留记录提交为正式可见映射。
  Status CommitPut(const PutContext& put_context) override {
    if (put_context.id == 0) {
      return InvalidArgument("PutContext id must be non-zero");
    }

    std::lock_guard<std::mutex> lock(meta_mutex_);
    if (!region_registered_) {
      return NotInitialized("remote region has not been registered");
    }

    auto pending_it = pending_puts_.find(put_context.id);
    if (pending_it == pending_puts_.end()) {
      return NotFound("PutContext was not found");
    }

    // PreparePut recorded the stable flat-table position for every pending
    // key while holding meta_mutex_. CommitPut changes those exact entries to
    // active without another key hash probe or insertion.
    for (const PendingPutItem& item : pending_it->second.items) {
      AddressRecord& record = records_[item.slot_id];
      record.record_id = next_record_id_++;
      record.uba = SlotAddress(item.slot_id);
      record.valid_length = item.valid_length;
      record.slot_id = item.slot_id;
      record.pin_count = 0;
    }
    for (const PendingPutItem& item : pending_it->second.items) {
      if (!key_index_.ActivatePendingAt(item.key, item.slot_id,
                                         item.flat_index_slot)) {
        return InvalidState("CommitPut lost a pending key index entry");
      }
    }
    pending_puts_.erase(pending_it);
    return Status::OK();
  }

  // 放弃一次未提交的 put，并把预留 slot 归还空闲队列。
  Status AbortPut(const PutContext& put_context) override {
    if (put_context.id == 0) {
      return InvalidArgument("PutContext id must be non-zero");
    }

    std::lock_guard<std::mutex> lock(meta_mutex_);
    if (!region_registered_) {
      return NotInitialized("remote region has not been registered");
    }

    auto pending_it = pending_puts_.find(put_context.id);
    if (pending_it == pending_puts_.end()) {
      return NotFound("PutContext was not found");
    }

    for (auto item = pending_it->second.items.rbegin();
         item != pending_it->second.items.rend(); ++item) {
      (void)key_index_.ErasePendingAt(item->key, item->slot_id,
                                      item->flat_index_slot);
      free_slots_.push_back(item->slot_id);
    }

    pending_puts_.erase(pending_it);
    return Status::OK();
  }

  // 为一批 key 建立稳定读快照，并通过 pin_count 阻止并发 free。
  Status LockKeys(const std::vector<KeyId>& flat_keys,
                  MetaManagerLockMode lock_mode,
                  LockContext* lock_context) override {
    if (lock_context == nullptr) {
      return InvalidArgument("lock_context must not be null");
    }
    if (flat_keys.empty()) {
      return InvalidArgument("LockKeys requires at least one key");
    }
    if (lock_mode != MetaManagerLockMode::kStableReadAllowAppend) {
      return InvalidArgument("unsupported lock mode");
    }

    std::lock_guard<std::mutex> lock(meta_mutex_);
    if (!region_registered_) {
      return NotInitialized("remote region has not been registered");
    }

    std::vector<LockedItem> items;
    items.reserve(flat_keys.size());
    for (KeyId key : flat_keys) {
      std::size_t slot_id = 0U;
      if (!key_index_.FindActive(key, &slot_id)) {
        // Pins are taken while building the snapshot so this path needs to
        // roll them back if a later key is missing.
        for (const LockedItem& item : items) {
          --item.record->pin_count;
        }
        return NotFound("LockKeys key was not found");
      }

      // records_ is allocated once during region registration and never
      // resized, so this pointer remains stable until the manager is destroyed.
      AddressRecord* record = &records_[slot_id];
      ++record->pin_count;
      items.push_back(LockedItem{key, record});
    }

    const std::uint64_t ctx_id = next_lock_context_id_++;
    lock_contexts_.emplace(ctx_id, ActiveLockContext{std::move(items)});
    lock_context->id = ctx_id;
    return Status::OK();
  }

  // 读取 LockKeys 固定下来的那一版地址信息。
  Status BatchLookup(const std::vector<KeyId>& flat_keys,
                     const LockContext& lock_context,
                     FlatAccessDesc* access_desc) override {
    if (access_desc == nullptr) {
      return InvalidArgument("access_desc must not be null");
    }
    if (lock_context.id == 0) {
      return InvalidArgument("LockContext id must be non-zero");
    }

    std::lock_guard<std::mutex> lock(meta_mutex_);
    if (!region_registered_) {
      return NotInitialized("remote region has not been registered");
    }

    auto lock_it = lock_contexts_.find(lock_context.id);
    if (lock_it == lock_contexts_.end()) {
      return NotFound("LockContext was not found");
    }
    if (flat_keys.size() != lock_it->second.items.size()) {
      return InvalidArgument("BatchLookup keys size does not match LockContext");
    }

    access_desc->addresses.clear();
    access_desc->lengths.clear();
    access_desc->valid.clear();
    access_desc->addresses.reserve(flat_keys.size());
    access_desc->lengths.reserve(flat_keys.size());
    access_desc->valid.reserve(flat_keys.size());

    for (std::size_t i = 0; i < flat_keys.size(); ++i) {
      const LockedItem& item = lock_it->second.items[i];
      if (flat_keys[i] != item.key) {
        return InvalidArgument("BatchLookup keys do not match LockContext");
      }

      if (item.record == nullptr || item.record->pin_count == 0) {
        return InvalidState("BatchLookup lost a locked record");
      }

      access_desc->addresses.push_back(item.record->uba);
      access_desc->lengths.push_back(item.record->valid_length);
      access_desc->valid.push_back(1);
    }

    return Status::OK();
  }

  // 结束一次稳定读，释放对应记录上的 pin。
  Status UnlockKeys(const LockContext& lock_context) override {
    if (lock_context.id == 0) {
      return InvalidArgument("LockContext id must be non-zero");
    }

    std::lock_guard<std::mutex> lock(meta_mutex_);
    if (!region_registered_) {
      return NotInitialized("remote region has not been registered");
    }

    auto lock_it = lock_contexts_.find(lock_context.id);
    if (lock_it == lock_contexts_.end()) {
      return NotFound("LockContext was not found");
    }

    for (const LockedItem& item : lock_it->second.items) {
      if (item.record == nullptr || item.record->pin_count == 0) {
        return InvalidState("UnlockKeys found a zero pin_count");
      }

      --item.record->pin_count;
    }

    lock_contexts_.erase(lock_it);
    return Status::OK();
  }

  // 释放一批 key 占用的 slot。
  // 先整体校验，再统一回收，保持 batch 级 all-or-nothing 语义。
  Status BatchFree(const std::vector<KeyId>& flat_keys) override {
    if (flat_keys.empty()) {
      return InvalidArgument("BatchFree requires at least one key");
    }
    if (!HasDistinctKeys(flat_keys)) {
      return InvalidArgument("BatchFree does not accept duplicate keys");
    }

    std::lock_guard<std::mutex> lock(meta_mutex_);
    if (!region_registered_) {
      return NotInitialized("remote region has not been registered");
    }

    std::vector<std::size_t> slots_to_recycle;
    slots_to_recycle.reserve(flat_keys.size());
    for (KeyId key : flat_keys) {
      std::size_t slot_id = 0U;
      if (!key_index_.FindActive(key, &slot_id)) {
        return NotFound("BatchFree key was not found");
      }

      const AddressRecord& record = records_[slot_id];
      if (record.pin_count > 0) {
        return Busy("BatchFree key is still pinned by a LockContext");
      }

      slots_to_recycle.push_back(record.slot_id);
    }

    // The benchmark and normal table teardown free the complete active set.
    // Resetting the flat table in one pass avoids probing each key again.
    if (key_index_.size() == flat_keys.size()) {
      key_index_.Clear();
    } else {
      for (std::size_t index = 0U; index < flat_keys.size(); ++index) {
        (void)key_index_.EraseActive(flat_keys[index],
                                     slots_to_recycle[index]);
      }
    }

    for (auto slot = slots_to_recycle.rbegin();
         slot != slots_to_recycle.rend(); ++slot) {
      free_slots_.push_back(*slot);
    }

    return Status::OK();
  }

 private:
  Address SlotAddress(std::size_t slot_id) const {
    return region_.base + static_cast<Address>(slot_id) * region_.slot_size;
  }

  // 当前 POC 使用一把总锁保护元数据和预分配的空闲 slot stack。
  std::mutex meta_mutex_;
  bool region_registered_{false};
  RemoteRegionDesc region_;
  std::vector<std::size_t> free_slots_;
  std::vector<AddressRecord> records_;
  FlatKeyIndex key_index_;
  std::unordered_map<std::uint64_t, PendingPutContext> pending_puts_;
  std::unordered_map<std::uint64_t, ActiveLockContext> lock_contexts_;
  std::uint64_t next_record_id_{1};
  std::uint64_t next_put_context_id_{1};
  std::uint64_t next_lock_context_id_{1};
};

}  // namespace

std::unique_ptr<MetaManager> CreateInMemoryMetaManager() {
  return std::make_unique<InMemoryMetaManager>();
}

}  // namespace kvaf
