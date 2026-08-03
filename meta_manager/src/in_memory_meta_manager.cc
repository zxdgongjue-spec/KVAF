#include "kvaf/meta_manager/meta_manager.h"

#include <chrono>
#include <iostream>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace kvaf {

namespace {

// 统一的短错误标签，避免不同分支拼接不同格式的字符串。
constexpr std::size_t kLockKeysGroupedThreshold = 256;

enum class ErrorTag {
  kInvalidRequest,
  kInvalidRegion,
  kRegionExists,
  kRegionMissing,
  kPutContextMissing,
  kLockContextMissing,
  kKeyExists,
  kKeyMissing,
  kOutOfSlots,
  kKeyBusy,
  kStateConflict,
  kUnsupportedMode,
};

Status Error(StatusCode code, ErrorTag tag) {
  switch (tag) {
    case ErrorTag::kInvalidRequest:
      return Status::Error(code, "invalid_request");
    case ErrorTag::kInvalidRegion:
      return Status::Error(code, "invalid_region");
    case ErrorTag::kRegionExists:
      return Status::Error(code, "region_exists");
    case ErrorTag::kRegionMissing:
      return Status::Error(code, "region_missing");
    case ErrorTag::kPutContextMissing:
      return Status::Error(code, "put_context_missing");
    case ErrorTag::kLockContextMissing:
      return Status::Error(code, "lock_context_missing");
    case ErrorTag::kKeyExists:
      return Status::Error(code, "key_exists");
    case ErrorTag::kKeyMissing:
      return Status::Error(code, "key_missing");
    case ErrorTag::kOutOfSlots:
      return Status::Error(code, "out_of_slots");
    case ErrorTag::kKeyBusy:
      return Status::Error(code, "key_busy");
    case ErrorTag::kStateConflict:
      return Status::Error(code, "state_conflict");
    case ErrorTag::kUnsupportedMode:
      return Status::Error(code, "unsupported_mode");
  }

  return Status::Error(StatusCode::kInternal, "internal_error");
}

// 作用域计时器：函数退出时按需打印耗时。
class ScopedMethodTimer {
 public:
  ScopedMethodTimer(const std::atomic<bool>* enabled, const char* method_name)
      : enabled_(enabled),
        method_name_(method_name),
        start_(std::chrono::steady_clock::now()) {}

  ~ScopedMethodTimer() {
    if (!enabled_->load(std::memory_order_relaxed)) {
      return;
    }

    const auto elapsed_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start_)
            .count();
    std::cout << "[MetaManagerTiming] method=" << method_name_
              << " elapsed_ns=" << elapsed_ns << '\n';
  }

 private:
  const std::atomic<bool>* enabled_;
  const char* method_name_;
  std::chrono::steady_clock::time_point start_;
};

}  // namespace

Status MetaManager::RegisterRemoteRegion(const RemoteRegionDesc& region) {
  ScopedMethodTimer timer(&timing_log_enabled_, "RegisterRemoteRegion");
  if (region_state_.load(std::memory_order_acquire) !=
      RegionState::kUninitialized) {
    return Error(StatusCode::kAlreadyExists, ErrorTag::kRegionExists);
  }

  // 当前 POC 要求：
  // 1. short / long 两类 slot 都必须存在
  // 2. long slot 必须严格大于 short slot
  // 3. 两类 slot 数量相同
  if (region.short_slot_size == 0 || region.long_slot_size == 0 ||
      region.slot_count_per_class == 0 ||
      region.short_slot_size >= region.long_slot_size) {
    return Error(StatusCode::kInvalidArgument, ErrorTag::kInvalidRegion);
  }

  const std::uint64_t total_slot_size =
      region.short_slot_size + region.long_slot_size;
  if (region.length != region.slot_count_per_class * total_slot_size) {
    return Error(StatusCode::kInvalidArgument, ErrorTag::kInvalidRegion);
  }

  RegionState expected = RegionState::kUninitialized;
  if (!region_state_.compare_exchange_strong(
          expected, RegionState::kInitializing, std::memory_order_acq_rel,
          std::memory_order_acquire)) {
    return Error(StatusCode::kAlreadyExists, ErrorTag::kRegionExists);
  }

  region_ = region;

  for (std::uint64_t i = 0; i < region.slot_count_per_class; ++i) {
    short_free_slots_.push(static_cast<std::size_t>(i));
    long_free_slots_.push(static_cast<std::size_t>(i));
  }

  const std::size_t max_active_keys =
      static_cast<std::size_t>(region.slot_count_per_class) * 2u;
  const std::size_t reserve_per_shard =
      (max_active_keys + kKeyShardCount - 1u) / kKeyShardCount;
  for (KeyShard& shard : key_shards_) {
    shard.entries.max_load_factor(0.5f);
    shard.entries.reserve(reserve_per_shard);
  }

  {
    std::lock_guard<std::mutex> context_lock(context_mutex_);
    pending_puts_.reserve(1024);
    lock_contexts_.reserve(1024);
  }

  region_state_.store(RegionState::kReady, std::memory_order_release);
  return Status::OK();
}

Status MetaManager::PreparePut(const std::vector<KeyId>& flat_keys,
                               const std::vector<std::uint64_t>& lengths,
                               PutContext* put_context,
                               PutAccessDesc* put_desc) {
  ScopedMethodTimer timer(&timing_log_enabled_, "PreparePut");
  if (put_context == nullptr || put_desc == nullptr || flat_keys.empty() ||
      flat_keys.size() != lengths.size()) {
    return Error(StatusCode::kInvalidArgument, ErrorTag::kInvalidRequest);
  }
  if (region_state_.load(std::memory_order_acquire) != RegionState::kReady) {
    return Error(StatusCode::kNotInitialized, ErrorTag::kRegionMissing);
  }

  // 这一步完全是线程私有工作：先做长度判断。
  // 默认模式下仍保留批次内重复 key 校验；性能模式可跳过这段预检查。
  std::vector<SlotClass> slot_classes(flat_keys.size());
  std::unordered_set<std::string_view> batch_seen;
  if (!performance_mode_enabled_) {
    batch_seen.reserve(flat_keys.size());
  }
  std::size_t short_needed = 0;
  std::size_t long_needed = 0;
  for (std::size_t i = 0; i < flat_keys.size(); ++i) {
    if (!performance_mode_enabled_ && !batch_seen.emplace(flat_keys[i]).second) {
      return Error(StatusCode::kInvalidArgument, ErrorTag::kInvalidRequest);
    }

    if (!TryResolveSlotClass(lengths[i], &slot_classes[i])) {
      return Error(StatusCode::kInvalidArgument, ErrorTag::kInvalidRequest);
    }

    if (slot_classes[i] == SlotClass::kShort) {
      ++short_needed;
    } else {
      ++long_needed;
    }
  }

  std::vector<std::size_t> key_shard_ids;
  std::vector<std::size_t> shard_order;
  CollectShardPlan(flat_keys, &key_shard_ids, &shard_order);

  // 按 shard id 升序加锁，避免多个线程交叉锁 shard 时死锁。
  std::vector<std::unique_lock<std::mutex>> shard_locks;
  shard_locks.reserve(shard_order.size());
  for (std::size_t shard_id : shard_order) {
    shard_locks.emplace_back(key_shards_[shard_id].mutex);
  }

  PendingPutContext pending_context;
  pending_context.items.reserve(flat_keys.size());
  std::vector<std::pair<std::size_t, const KeyId*>> new_keys;
  new_keys.reserve(flat_keys.size());

  auto rollback_new_keys = [&]() {
    for (auto it = new_keys.rbegin(); it != new_keys.rend(); ++it) {
      key_shards_[it->first].entries.erase(*it->second);
    }
  };

  for (std::size_t i = 0; i < flat_keys.size(); ++i) {
    KeyShard& shard = key_shards_[key_shard_ids[i]];
    auto key_it = shard.entries.find(flat_keys[i]);
    if (key_it == shard.entries.end()) {
      key_it = shard.entries.emplace(flat_keys[i], KeyEntry{}).first;
      new_keys.push_back({key_shard_ids[i], &key_it->first});
    }

    KeyEntry& key_entry = key_it->second;
    if (key_entry.has_active_record || key_entry.is_pending) {
      rollback_new_keys();
      return Error(StatusCode::kAlreadyExists, ErrorTag::kKeyExists);
    }

    pending_context.items.push_back(PendingPutItem{
        key_shard_ids[i], &key_entry, &key_it->first,
        SlotRef{slot_classes[i], 0}, lengths[i]});
  }

  std::unique_lock<std::mutex> free_queue_lock;
  if (short_needed > 0 || long_needed > 0) {
    free_queue_lock = std::unique_lock<std::mutex>(free_queue_mutex_);
  }

  if (short_free_slots_.size() < short_needed ||
      long_free_slots_.size() < long_needed) {
    rollback_new_keys();
    return Error(StatusCode::kOutOfMemory, ErrorTag::kOutOfSlots);
  }

  put_desc->entries.resize(flat_keys.size());
  for (std::size_t i = 0; i < pending_context.items.size(); ++i) {
    PendingPutItem& item = pending_context.items[i];
    item.slot = AcquireSlot(item.slot.slot_class);
    item.key_entry->is_pending = true;
    FillAccessEntry(item.key, SlotAddress(item.slot), item.valid_length,
                    &put_desc->entries[i]);
  }

  {
    std::lock_guard<std::mutex> context_lock(context_mutex_);
    const std::uint64_t ctx_id = next_put_context_id_++;
    pending_puts_.emplace(ctx_id, std::move(pending_context));
    put_context->id = ctx_id;
  }

  return Status::OK();
}

Status MetaManager::CommitPut(const PutContext& put_context) {
  ScopedMethodTimer timer(&timing_log_enabled_, "CommitPut");
  if (region_state_.load(std::memory_order_acquire) != RegionState::kReady) {
    return Error(StatusCode::kNotInitialized, ErrorTag::kRegionMissing);
  }

  PendingPutContext pending_context;
  {
    std::lock_guard<std::mutex> context_lock(context_mutex_);
    auto pending_it = pending_puts_.find(put_context.id);
    if (pending_it == pending_puts_.end()) {
      return Error(StatusCode::kNotFound, ErrorTag::kPutContextMissing);
    }
    pending_context = std::move(pending_it->second);
    pending_puts_.erase(pending_it);
  }

  std::array<bool, kKeyShardCount> shard_used{};
  std::vector<std::size_t> shard_order;
  for (const PendingPutItem& item : pending_context.items) {
    if (!shard_used[item.shard_id]) {
      shard_used[item.shard_id] = true;
      shard_order.push_back(item.shard_id);
    }
  }

  std::vector<std::unique_lock<std::mutex>> shard_locks;
  shard_locks.reserve(shard_order.size());
  for (std::size_t shard_id : shard_order) {
    shard_locks.emplace_back(key_shards_[shard_id].mutex);
  }

  // Commit 是写路径的可见性边界；提交后才变成 active record。
  for (const PendingPutItem& item : pending_context.items) {
    if (item.key_entry == nullptr || !item.key_entry->is_pending ||
        item.key_entry->has_active_record) {
      return Error(StatusCode::kInvalidState, ErrorTag::kStateConflict);
    }
    item.key_entry->is_pending = false;
    item.key_entry->has_active_record = true;
    item.key_entry->active_record = {item.slot, item.valid_length, 0};
  }

  return Status::OK();
}

Status MetaManager::AbortPut(const PutContext& put_context) {
  ScopedMethodTimer timer(&timing_log_enabled_, "AbortPut");
  if (region_state_.load(std::memory_order_acquire) != RegionState::kReady) {
    return Error(StatusCode::kNotInitialized, ErrorTag::kRegionMissing);
  }

  PendingPutContext pending_context;
  {
    std::lock_guard<std::mutex> context_lock(context_mutex_);
    auto pending_it = pending_puts_.find(put_context.id);
    if (pending_it == pending_puts_.end()) {
      return Error(StatusCode::kNotFound, ErrorTag::kPutContextMissing);
    }
    pending_context = std::move(pending_it->second);
    pending_puts_.erase(pending_it);
  }

  std::array<bool, kKeyShardCount> shard_used{};
  std::vector<std::size_t> shard_order;
  std::size_t short_count = 0;
  std::size_t long_count = 0;
  for (const PendingPutItem& item : pending_context.items) {
    if (!shard_used[item.shard_id]) {
      shard_used[item.shard_id] = true;
      shard_order.push_back(item.shard_id);
    }
    if (item.slot.slot_class == SlotClass::kShort) {
      ++short_count;
    } else {
      ++long_count;
    }
  }

  std::vector<std::unique_lock<std::mutex>> shard_locks;
  shard_locks.reserve(shard_order.size());
  for (std::size_t shard_id : shard_order) {
    shard_locks.emplace_back(key_shards_[shard_id].mutex);
  }

  std::unique_lock<std::mutex> free_queue_lock;
  if (short_count > 0 || long_count > 0) {
    free_queue_lock = std::unique_lock<std::mutex>(free_queue_mutex_);
  }

  for (const PendingPutItem& item : pending_context.items) {
    if (item.key_entry == nullptr || !item.key_entry->is_pending) {
      return Error(StatusCode::kInvalidState, ErrorTag::kStateConflict);
    }
    item.key_entry->is_pending = false;
    ReleaseSlot(item.slot);
  }

  return Status::OK();
}

Status MetaManager::LockKeys(const std::vector<KeyId>& flat_keys,
                             MetaManagerLockMode lock_mode,
                             LockContext* lock_context) {
  ScopedMethodTimer timer(&timing_log_enabled_, "LockKeys");
  if (lock_context == nullptr || flat_keys.empty()) {
    return Error(StatusCode::kInvalidArgument, ErrorTag::kInvalidRequest);
  }
  if (lock_mode != MetaManagerLockMode::kStableReadAllowAppend) {
    return Error(StatusCode::kInvalidArgument, ErrorTag::kUnsupportedMode);
  }
  if (region_state_.load(std::memory_order_acquire) != RegionState::kReady) {
    return Error(StatusCode::kNotInitialized, ErrorTag::kRegionMissing);
  }

  ActiveLockContext active_context;
  std::vector<AddressRecord*> pinned_records;
  pinned_records.reserve(flat_keys.size());

  auto rollback_pins = [&]() {
    for (AddressRecord* record : pinned_records) {
      DecrementPinCount(record);
    }
  };

  if (flat_keys.size() < kLockKeysGroupedThreshold) {
    active_context.items.reserve(flat_keys.size());

  // 逐 key 加 shard 锁，避免先整批做一次 shard 规划、再整批做一次查表。
  // 在当前 benchmark 的单线程场景里，这条路径比“多锁统一规划”更轻。
  for (const KeyId& key : flat_keys) {
    KeyShard& shard = key_shards_[ShardIndexForKey(key)];
    std::lock_guard<std::mutex> shard_lock(shard.mutex);
    auto key_it = shard.entries.find(key);
    if (key_it == shard.entries.end() || !key_it->second.has_active_record) {
      rollback_pins();
      return Error(StatusCode::kNotFound, ErrorTag::kKeyMissing);
    }

    AddressRecord* record = &key_it->second.active_record;
    IncrementPinCount(record);
    pinned_records.push_back(record);
      active_context.items.push_back(LockedRecordRef{&key_it->first, record});
    }
  } else {
    std::vector<std::size_t> key_shard_ids;
    std::vector<std::size_t> shard_order;
    CollectShardPlan(flat_keys, &key_shard_ids, &shard_order);

    std::array<std::size_t, kKeyShardCount> shard_counts{};
    for (std::size_t shard_id : key_shard_ids) {
      ++shard_counts[shard_id];
    }

    std::array<std::vector<std::size_t>, kKeyShardCount> shard_key_indices;
    for (std::size_t shard_id : shard_order) {
      shard_key_indices[shard_id].reserve(shard_counts[shard_id]);
    }
    for (std::size_t i = 0; i < key_shard_ids.size(); ++i) {
      shard_key_indices[key_shard_ids[i]].push_back(i);
    }

    active_context.items.resize(flat_keys.size());
    for (std::size_t shard_id : shard_order) {
      KeyShard& shard = key_shards_[shard_id];
      std::lock_guard<std::mutex> shard_lock(shard.mutex);
      for (std::size_t key_index : shard_key_indices[shard_id]) {
        auto key_it = shard.entries.find(flat_keys[key_index]);
        if (key_it == shard.entries.end() ||
            !key_it->second.has_active_record) {
          rollback_pins();
          return Error(StatusCode::kNotFound, ErrorTag::kKeyMissing);
        }

        AddressRecord* record = &key_it->second.active_record;
        IncrementPinCount(record);
        pinned_records.push_back(record);
        active_context.items[key_index] =
            LockedRecordRef{&key_it->first, record};
      }
    }
  }

  {
    std::lock_guard<std::mutex> context_lock(context_mutex_);
    const std::uint64_t ctx_id = next_lock_context_id_++;
    lock_contexts_.emplace(ctx_id, std::move(active_context));
    lock_context->id = ctx_id;
  }

  return Status::OK();
}

Status MetaManager::BatchLookup(const std::vector<KeyId>& flat_keys,
                                const LockContext& lock_context,
                                FlatAccessDesc* access_desc) {
  ScopedMethodTimer timer(&timing_log_enabled_, "BatchLookup");
  if (access_desc == nullptr) {
    return Error(StatusCode::kInvalidArgument, ErrorTag::kInvalidRequest);
  }
  if (region_state_.load(std::memory_order_acquire) != RegionState::kReady) {
    return Error(StatusCode::kNotInitialized, ErrorTag::kRegionMissing);
  }

  // BatchLookup 只读 lock 上下文；这里持有 context 锁即可保证：
  // 1. 同一个 lock context 不会在 lookup 途中被 Unlock 删除
  // 2. 对应 pin_count 在 lookup 期间不会被减到 0
  std::vector<AccessEntry> snapshot_entries(flat_keys.size());
  {
    std::lock_guard<std::mutex> context_lock(context_mutex_);
    auto lock_it = lock_contexts_.find(lock_context.id);
    if (lock_it == lock_contexts_.end()) {
      return Error(StatusCode::kNotFound, ErrorTag::kLockContextMissing);
    }
    if (lock_it->second.items.size() != flat_keys.size()) {
      return Error(StatusCode::kInvalidArgument, ErrorTag::kInvalidRequest);
    }

    for (std::size_t i = 0; i < lock_it->second.items.size(); ++i) {
      const LockedRecordRef& item = lock_it->second.items[i];
      if (item.key == nullptr || item.record == nullptr) {
        return Error(StatusCode::kInvalidState, ErrorTag::kStateConflict);
      }
      if (!performance_mode_enabled_ && *item.key != flat_keys[i]) {
        return Error(StatusCode::kInvalidArgument, ErrorTag::kInvalidRequest);
      }
      FillAccessEntry(item.key, SlotAddress(item.record->slot),
                      item.record->valid_length, &snapshot_entries[i]);
    }
  }

  access_desc->entries = std::move(snapshot_entries);
  return Status::OK();
}

Status MetaManager::UnlockKeys(const LockContext& lock_context) {
  ScopedMethodTimer timer(&timing_log_enabled_, "UnlockKeys");
  if (region_state_.load(std::memory_order_acquire) != RegionState::kReady) {
    return Error(StatusCode::kNotInitialized, ErrorTag::kRegionMissing);
  }

  ActiveLockContext active_context;
  {
    std::lock_guard<std::mutex> context_lock(context_mutex_);
    auto lock_it = lock_contexts_.find(lock_context.id);
    if (lock_it == lock_contexts_.end()) {
      return Error(StatusCode::kNotFound, ErrorTag::kLockContextMissing);
    }
    active_context = std::move(lock_it->second);
    lock_contexts_.erase(lock_it);
  }

  // Unlock 不再回头做“key 去重 + shard 规划 + 二次加锁”。
  // lock context 已经缓存了稳定 record 指针，这里直接原子减 pin 即可。
  for (const LockedRecordRef& item : active_context.items) {
    if (item.record == nullptr || LoadPinCount(*item.record) == 0) {
      return Error(StatusCode::kInvalidState, ErrorTag::kStateConflict);
    }
  }

  for (const LockedRecordRef& item : active_context.items) {
    DecrementPinCount(item.record);
  }

  return Status::OK();
}

Status MetaManager::BatchFree(const std::vector<KeyId>& flat_keys) {
  ScopedMethodTimer timer(&timing_log_enabled_, "BatchFree");
  if (flat_keys.empty()) {
    return Error(StatusCode::kInvalidArgument, ErrorTag::kInvalidRequest);
  }
  if (region_state_.load(std::memory_order_acquire) != RegionState::kReady) {
    return Error(StatusCode::kNotInitialized, ErrorTag::kRegionMissing);
  }

  // BatchFree 默认保留重复 key 的静默去重；性能模式可跳过，
  // 由上层保证输入不重复。
  std::unordered_set<std::string_view> batch_seen;
  std::vector<KeyId> unique_keys;
  unique_keys.reserve(flat_keys.size());
  if (performance_mode_enabled_) {
    unique_keys = flat_keys;
  } else {
    batch_seen.reserve(flat_keys.size());
    for (const KeyId& key : flat_keys) {
      if (batch_seen.emplace(key).second) {
        unique_keys.push_back(key);
      }
    }
  }

  std::vector<std::size_t> key_shard_ids;
  std::vector<std::size_t> shard_order;
  CollectShardPlan(unique_keys, &key_shard_ids, &shard_order);

  std::vector<std::unique_lock<std::mutex>> shard_locks;
  shard_locks.reserve(shard_order.size());
  for (std::size_t shard_id : shard_order) {
    shard_locks.emplace_back(key_shards_[shard_id].mutex);
  }

  std::vector<SlotRef> slots_to_recycle;
  slots_to_recycle.reserve(unique_keys.size());
  std::vector<KeyEntry*> entries_to_clear;
  entries_to_clear.reserve(unique_keys.size());
  std::size_t short_count = 0;
  std::size_t long_count = 0;

  for (std::size_t i = 0; i < unique_keys.size(); ++i) {
    KeyShard& shard = key_shards_[key_shard_ids[i]];
    auto key_it = shard.entries.find(unique_keys[i]);
    if (key_it == shard.entries.end() || !key_it->second.has_active_record) {
      return Error(StatusCode::kNotFound, ErrorTag::kKeyMissing);
    }
    if (LoadPinCount(key_it->second.active_record) > 0) {
      return Error(StatusCode::kBusy, ErrorTag::kKeyBusy);
    }

    const SlotRef slot = key_it->second.active_record.slot;
    if (slot.slot_class == SlotClass::kShort) {
      ++short_count;
    } else {
      ++long_count;
    }
    slots_to_recycle.push_back(slot);
    entries_to_clear.push_back(&key_it->second);
  }

  std::unique_lock<std::mutex> free_queue_lock;
  if (short_count > 0 || long_count > 0) {
    free_queue_lock = std::unique_lock<std::mutex>(free_queue_mutex_);
  }

  for (KeyEntry* entry : entries_to_clear) {
    entry->has_active_record = false;
    entry->active_record = {};
  }
  for (const SlotRef& slot : slots_to_recycle) {
    ReleaseSlot(slot);
  }

  return Status::OK();
}

Address MetaManager::SlotAddress(const SlotRef& slot) const {
  // region 前半段是 short slot 区，后半段是 long slot 区。
  if (slot.slot_class == SlotClass::kShort) {
    return region_.base +
           static_cast<Address>(slot.slot_index) * region_.short_slot_size;
  }

  const Address long_region_base =
      region_.base + region_.slot_count_per_class * region_.short_slot_size;
  return long_region_base +
         static_cast<Address>(slot.slot_index) * region_.long_slot_size;
}

bool MetaManager::TryResolveSlotClass(std::uint64_t length,
                                      SlotClass* slot_class) const {
  // 长度为 0 非法；否则优先走更省空间的 short slot。
  if (length == 0 || slot_class == nullptr) {
    return false;
  }
  if (length <= region_.short_slot_size) {
    *slot_class = SlotClass::kShort;
    return true;
  }
  if (length <= region_.long_slot_size) {
    *slot_class = SlotClass::kLong;
    return true;
  }
  return false;
}

void MetaManager::FillAccessEntry(const KeyId* key,
                                  Address address,
                                  std::uint64_t length,
                                  AccessEntry* entry) const {
  if (entry == nullptr) {
    return;
  }

  entry->key = key;
  entry->address = address;
  entry->length = length;
  entry->valid = 1;
}

std::uint32_t MetaManager::LoadPinCount(const AddressRecord& record) const {
  return record.pin_count.load(std::memory_order_relaxed);
}

void MetaManager::IncrementPinCount(AddressRecord* record) {
  if (record == nullptr) {
    return;
  }
  record->pin_count.fetch_add(1, std::memory_order_relaxed);
}

void MetaManager::DecrementPinCount(AddressRecord* record) {
  if (record == nullptr) {
    return;
  }
  record->pin_count.fetch_sub(1, std::memory_order_relaxed);
}

MetaManager::SlotRef MetaManager::AcquireSlot(SlotClass slot_class) {
  std::queue<std::size_t>& free_slots =
      slot_class == SlotClass::kShort ? short_free_slots_ : long_free_slots_;
  const std::size_t slot_index = free_slots.front();
  free_slots.pop();
  return SlotRef{slot_class, slot_index};
}

void MetaManager::ReleaseSlot(const SlotRef& slot) {
  std::queue<std::size_t>& free_slots =
      slot.slot_class == SlotClass::kShort ? short_free_slots_ : long_free_slots_;
  free_slots.push(slot.slot_index);
}

std::size_t MetaManager::ShardIndexForKey(const KeyId& key) const {
  return std::hash<KeyId>{}(key) % kKeyShardCount;
}

void MetaManager::CollectShardPlan(const std::vector<KeyId>& flat_keys,
                                   std::vector<std::size_t>* key_shard_ids,
                                   std::vector<std::size_t>* shard_order) const {
  key_shard_ids->resize(flat_keys.size());
  shard_order->clear();

  std::array<bool, kKeyShardCount> shard_used{};
  for (std::size_t i = 0; i < flat_keys.size(); ++i) {
    const std::size_t shard_id = ShardIndexForKey(flat_keys[i]);
    (*key_shard_ids)[i] = shard_id;
    shard_used[shard_id] = true;
  }

  shard_order->reserve(kKeyShardCount);
  for (std::size_t shard_id = 0; shard_id < kKeyShardCount; ++shard_id) {
    if (shard_used[shard_id]) {
      shard_order->push_back(shard_id);
    }
  }
}

void MetaManager::SetTimingLogEnabled(bool enabled) {
  timing_log_enabled_.store(enabled, std::memory_order_relaxed);
}

void MetaManager::SetPerformanceModeEnabled(bool enabled) {
  performance_mode_enabled_ = enabled;
}

std::unique_ptr<MetaManager> CreateInMemoryMetaManager() {
  return std::make_unique<MetaManager>();
}

}  // namespace kvaf
