#include "kvaf/meta_manager/meta_manager.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <mutex>
#include <new>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace kvaf {

namespace {

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
  kHostOutOfMemory,
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
    case ErrorTag::kHostOutOfMemory:
      return Status::Error(code, "host_out_of_memory");
  }

  return Status::Error(StatusCode::kInternal, "internal_error");
}

class TimingAggregator {
 public:
  void Add(const char* method_name,
           const char* phase_name,
           std::uint64_t elapsed_ns) {
    std::lock_guard<std::mutex> lock(mutex_);
    samples_[{method_name, phase_name == nullptr ? "" : phase_name}]
        .push_back(elapsed_ns);
  }

  void Flush() {
    std::map<std::pair<std::string, std::string>,
             std::vector<std::uint64_t>>
        samples;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      samples.swap(samples_);
    }

    for (auto& entry : samples) {
      std::vector<std::uint64_t>& values = entry.second;
      if (values.empty()) {
        continue;
      }

      std::sort(values.begin(), values.end());
      std::uint64_t total_ns = 0;
      for (const std::uint64_t value : values) {
        total_ns += value;
      }

      const auto& key = entry.first;
      std::cerr << "[MetaManagerTimingSummary] method=" << key.first;
      if (!key.second.empty()) {
        std::cerr << " phase=" << key.second;
      }
      std::cerr << " count=" << values.size()
                << " total_ns=" << total_ns
                << " avg_ns="
                << static_cast<double>(total_ns) / values.size()
                << " min_ns=" << values.front()
                << " p50_ns=" << values[values.size() / 2]
                << " p95_ns=" << values[(values.size() * 95) / 100]
                << " max_ns=" << values.back() << '\n';
    }
  }

 private:
  std::mutex mutex_;
  std::map<std::pair<std::string, std::string>,
           std::vector<std::uint64_t>>
      samples_;
};

TimingAggregator& GetTimingAggregator() {
  static TimingAggregator aggregator;
  return aggregator;
}

class ScopedMethodTimer {
 public:
  ScopedMethodTimer(const std::atomic<bool>* enabled,
                   const char* method_name,
                   const char* phase_name = nullptr)
      : method_name_(method_name),
        phase_name_(phase_name),
        active_(enabled->load(std::memory_order_relaxed)) {
    if (active_) {
      start_ = std::chrono::steady_clock::now();
    }
  }

  ~ScopedMethodTimer() {
    if (!active_) {
      return;
    }

    const auto elapsed_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - start_)
            .count();
    GetTimingAggregator().Add(method_name_, phase_name_,
                               static_cast<std::uint64_t>(elapsed_ns));
  }

 private:
  const char* method_name_;
  const char* phase_name_;
  bool active_{false};
  std::chrono::steady_clock::time_point start_;
};

}  // namespace

void FlushMetaManagerTimingSummary() {
  GetTimingAggregator().Flush();
}

Status MetaManager::RegisterRemoteRegion(const RemoteRegionDesc& region) {
  ScopedMethodTimer timer(&timing_log_enabled_, "RegisterRemoteRegion");
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

  try {
    std::queue<std::size_t> short_free_slots;
    std::queue<std::size_t> long_free_slots;
    for (std::uint64_t i = 0; i < region.slot_count_per_class; ++i) {
      short_free_slots.push(static_cast<std::size_t>(i));
      long_free_slots.push(static_cast<std::size_t>(i));
    }

    const std::size_t max_active_keys =
        static_cast<std::size_t>(region.slot_count_per_class) * 2u;
    entries_.max_load_factor(0.5f);
    entries_.reserve(max_active_keys);
    pending_puts_.reserve(1024);
    lock_contexts_.reserve(128);

    region_ = region;
    short_free_slots_ = std::move(short_free_slots);
    long_free_slots_ = std::move(long_free_slots);
  } catch (const std::bad_alloc&) {
    region_state_.store(RegionState::kUninitialized, std::memory_order_release);
    return Error(StatusCode::kOutOfMemory, ErrorTag::kHostOutOfMemory);
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

  std::unique_lock<std::mutex> write_lock(write_mutex_, std::defer_lock);
  {
    ScopedMethodTimer phase_timer(&timing_log_enabled_, "PreparePut", "lock");
    write_lock.lock();
  }

  std::vector<SlotClass> slot_classes;
  std::unordered_set<std::string_view> batch_seen;
  const bool performance_mode =
      performance_mode_enabled_.load(std::memory_order_relaxed);

  std::size_t short_needed = 0;
  std::size_t long_needed = 0;
  {
    ScopedMethodTimer phase_timer(&timing_log_enabled_, "PreparePut",
                                  "classify");
    slot_classes.resize(flat_keys.size());
    if (!performance_mode) {
      batch_seen.reserve(flat_keys.size());
    }

    for (std::size_t i = 0; i < flat_keys.size(); ++i) {
      if (!performance_mode && !batch_seen.emplace(flat_keys[i]).second) {
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
  }

  PendingPutContext pending_context;
  std::vector<const KeyId*> new_keys;
  {
    ScopedMethodTimer phase_timer(&timing_log_enabled_, "PreparePut",
                                  "reserve");
    try {
      pending_context.items.reserve(flat_keys.size());
      new_keys.reserve(flat_keys.size());
    } catch (const std::bad_alloc&) {
      return Error(StatusCode::kOutOfMemory, ErrorTag::kHostOutOfMemory);
    }
  }

  auto rollback_new_keys = [&]() {
    for (auto it = new_keys.rbegin(); it != new_keys.rend(); ++it) {
      entries_.erase(**it);
    }
  };

  {
    ScopedMethodTimer phase_timer(&timing_log_enabled_, "PreparePut",
                                  "index_keys");
    for (std::size_t i = 0; i < flat_keys.size(); ++i) {
      auto key_it = entries_.find(flat_keys[i]);
      if (key_it == entries_.end()) {
        key_it = entries_.emplace(flat_keys[i], KeyEntry{}).first;
        new_keys.push_back(&key_it->first);
      }

      KeyEntry& key_entry = key_it->second;
      if (key_entry.has_active_record || key_entry.is_pending) {
        rollback_new_keys();
        return Error(StatusCode::kAlreadyExists, ErrorTag::kKeyExists);
      }

      pending_context.items.push_back(PendingPutItem{
          &key_entry, &key_it->first, SlotRef{slot_classes[i], 0}, lengths[i]});
    }
  }

  {
    ScopedMethodTimer phase_timer(&timing_log_enabled_, "PreparePut",
                                  "check_capacity");
    if (short_free_slots_.size() < short_needed ||
        long_free_slots_.size() < long_needed) {
      rollback_new_keys();
      return Error(StatusCode::kOutOfMemory, ErrorTag::kOutOfSlots);
    }
  }

  {
    ScopedMethodTimer phase_timer(&timing_log_enabled_, "PreparePut",
                                  "resize_desc");
    try {
      ResizeAccessDesc(flat_keys.size(), put_desc);
    } catch (const std::bad_alloc&) {
      rollback_new_keys();
      return Error(StatusCode::kOutOfMemory, ErrorTag::kHostOutOfMemory);
    }
  }

  {
    ScopedMethodTimer phase_timer(&timing_log_enabled_, "PreparePut",
                                  "allocate_slots");
    for (std::size_t i = 0; i < pending_context.items.size(); ++i) {
      PendingPutItem& item = pending_context.items[i];
      item.slot = AcquireSlot(item.slot.slot_class);
      item.key_entry->is_pending = true;
      FillAccessDescAt(item.key, SlotAddress(item.slot), item.valid_length, i,
                       put_desc);
    }
  }

  {
    ScopedMethodTimer phase_timer(&timing_log_enabled_, "PreparePut",
                                  "publish_context");
    try {
      const std::uint64_t ctx_id = next_put_context_id_++;
      pending_puts_.emplace(ctx_id, std::move(pending_context));
      put_context->id = ctx_id;
    } catch (const std::bad_alloc&) {
      for (const PendingPutItem& item : pending_context.items) {
        item.key_entry->is_pending = false;
        ReleaseSlot(item.slot);
      }
      rollback_new_keys();
      return Error(StatusCode::kOutOfMemory, ErrorTag::kHostOutOfMemory);
    }
  }

  return Status::OK();
}

Status MetaManager::CommitPut(const PutContext& put_context) {
  ScopedMethodTimer timer(&timing_log_enabled_, "CommitPut");
  if (region_state_.load(std::memory_order_acquire) != RegionState::kReady) {
    return Error(StatusCode::kNotInitialized, ErrorTag::kRegionMissing);
  }

  std::lock_guard<std::mutex> write_lock(write_mutex_);

  PendingPutContext pending_context;
  auto pending_it = pending_puts_.find(put_context.id);
  if (pending_it == pending_puts_.end()) {
    return Error(StatusCode::kNotFound, ErrorTag::kPutContextMissing);
  }
  pending_context = std::move(pending_it->second);
  pending_puts_.erase(pending_it);

  for (const PendingPutItem& item : pending_context.items) {
    if (item.key_entry == nullptr || !item.key_entry->is_pending ||
        item.key_entry->has_active_record) {
      return Error(StatusCode::kInvalidState, ErrorTag::kStateConflict);
    }

    item.key_entry->is_pending = false;
    item.key_entry->has_active_record = true;
    item.key_entry->active_record = AddressRecord{
        item.slot, item.valid_length, 0};
  }

  return Status::OK();
}

Status MetaManager::AbortPut(const PutContext& put_context) {
  ScopedMethodTimer timer(&timing_log_enabled_, "AbortPut");
  if (region_state_.load(std::memory_order_acquire) != RegionState::kReady) {
    return Error(StatusCode::kNotInitialized, ErrorTag::kRegionMissing);
  }

  std::lock_guard<std::mutex> write_lock(write_mutex_);

  PendingPutContext pending_context;
  auto pending_it = pending_puts_.find(put_context.id);
  if (pending_it == pending_puts_.end()) {
    return Error(StatusCode::kNotFound, ErrorTag::kPutContextMissing);
  }
  pending_context = std::move(pending_it->second);
  pending_puts_.erase(pending_it);

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
  {
    ScopedMethodTimer phase_timer(&timing_log_enabled_, "LockKeys", "reserve");
    try {
      active_context.items.reserve(flat_keys.size());
      pinned_records.reserve(flat_keys.size());
    } catch (const std::bad_alloc&) {
      return Error(StatusCode::kOutOfMemory, ErrorTag::kHostOutOfMemory);
    }
  }

  auto rollback_pins = [&]() {
    for (AddressRecord* record : pinned_records) {
      DecrementPinCount(record);
    }
  };

  {
    ScopedMethodTimer phase_timer(&timing_log_enabled_, "LockKeys",
                                  "find_and_pin");
    for (const KeyId& key : flat_keys) {
      auto key_it = entries_.find(key);
      if (key_it == entries_.end() || !key_it->second.has_active_record) {
        rollback_pins();
        return Error(StatusCode::kNotFound, ErrorTag::kKeyMissing);
      }

      AddressRecord* record = &key_it->second.active_record;
      IncrementPinCount(record);
      pinned_records.push_back(record);
      active_context.items.push_back(LockedRecordRef{&key_it->first, record});
    }
  }

  {
    ScopedMethodTimer phase_timer(&timing_log_enabled_, "LockKeys",
                                  "register_context");
    const std::uint64_t ctx_id =
        next_lock_context_id_.fetch_add(1, std::memory_order_relaxed);
    try {
      std::unique_lock<std::shared_mutex> context_lock(lock_context_mutex_);
      auto inserted = lock_contexts_.try_emplace(ctx_id);
      if (!inserted.second) {
        rollback_pins();
        return Error(StatusCode::kInvalidState, ErrorTag::kStateConflict);
      }
      inserted.first->second.items.swap(active_context.items);
    } catch (const std::bad_alloc&) {
      rollback_pins();
      return Error(StatusCode::kOutOfMemory, ErrorTag::kHostOutOfMemory);
    }

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

  std::shared_lock<std::shared_mutex> context_lock(lock_context_mutex_);
  auto context_it = lock_contexts_.find(lock_context.id);
  if (context_it == lock_contexts_.end()) {
    return Error(StatusCode::kNotFound, ErrorTag::kLockContextMissing);
  }
  const ActiveLockContext& active_context = context_it->second;

  if (active_context.items.size() != flat_keys.size()) {
    return Error(StatusCode::kInvalidArgument, ErrorTag::kInvalidRequest);
  }

  const bool performance_mode =
      performance_mode_enabled_.load(std::memory_order_relaxed);
  try {
    ResizeAccessDesc(active_context.items.size(), access_desc);
  } catch (const std::bad_alloc&) {
    return Error(StatusCode::kOutOfMemory, ErrorTag::kHostOutOfMemory);
  }

  for (std::size_t i = 0; i < active_context.items.size(); ++i) {
    const LockedRecordRef& item = active_context.items[i];
    if (item.key == nullptr || item.record == nullptr) {
      return Error(StatusCode::kInvalidState, ErrorTag::kStateConflict);
    }
    if (!performance_mode && *item.key != flat_keys[i]) {
      return Error(StatusCode::kInvalidArgument, ErrorTag::kInvalidRequest);
    }

    FillAccessDescAt(item.key, SlotAddress(item.record->slot),
                     item.record->valid_length, i, access_desc);
  }

  return Status::OK();
}

Status MetaManager::UnlockKeys(const LockContext& lock_context) {
  ScopedMethodTimer timer(&timing_log_enabled_, "UnlockKeys");
  if (region_state_.load(std::memory_order_acquire) != RegionState::kReady) {
    return Error(StatusCode::kNotInitialized, ErrorTag::kRegionMissing);
  }

  std::unique_lock<std::shared_mutex> context_lock(lock_context_mutex_);
  auto context_it = lock_contexts_.find(lock_context.id);
  if (context_it == lock_contexts_.end()) {
    return Error(StatusCode::kNotFound, ErrorTag::kLockContextMissing);
  }

  ActiveLockContext& active_context = context_it->second;
  for (const LockedRecordRef& item : active_context.items) {
    if (item.record == nullptr || LoadPinCount(*item.record) == 0) {
      return Error(StatusCode::kInvalidState, ErrorTag::kStateConflict);
    }
  }

  for (const LockedRecordRef& item : active_context.items) {
    DecrementPinCount(item.record);
  }

  lock_contexts_.erase(context_it);
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

  std::unique_lock<std::mutex> write_lock(write_mutex_, std::defer_lock);
  {
    ScopedMethodTimer phase_timer(&timing_log_enabled_, "BatchFree", "lock");
    write_lock.lock();
  }

  std::vector<KeyId> unique_keys;
  const bool performance_mode =
      performance_mode_enabled_.load(std::memory_order_relaxed);
  {
    ScopedMethodTimer phase_timer(&timing_log_enabled_, "BatchFree",
                                  "deduplicate");
    if (performance_mode) {
      try {
        unique_keys = flat_keys;
      } catch (const std::bad_alloc&) {
        return Error(StatusCode::kOutOfMemory, ErrorTag::kHostOutOfMemory);
      }
    } else {
      std::unordered_set<std::string_view> batch_seen;
      try {
        batch_seen.reserve(flat_keys.size());
        unique_keys.reserve(flat_keys.size());
      } catch (const std::bad_alloc&) {
        return Error(StatusCode::kOutOfMemory, ErrorTag::kHostOutOfMemory);
      }

      for (const KeyId& key : flat_keys) {
        if (batch_seen.emplace(key).second) {
          unique_keys.push_back(key);
        }
      }
    }
  }

  std::vector<SlotRef> slots_to_recycle;
  std::vector<KeyEntry*> entries_to_clear;
  {
    ScopedMethodTimer phase_timer(&timing_log_enabled_, "BatchFree",
                                  "reserve");
    try {
      slots_to_recycle.reserve(unique_keys.size());
      entries_to_clear.reserve(unique_keys.size());
    } catch (const std::bad_alloc&) {
      return Error(StatusCode::kOutOfMemory, ErrorTag::kHostOutOfMemory);
    }
  }

  {
    ScopedMethodTimer phase_timer(&timing_log_enabled_, "BatchFree",
                                  "validate_keys");
    for (const KeyId& key : unique_keys) {
      auto key_it = entries_.find(key);
      if (key_it == entries_.end() || !key_it->second.has_active_record) {
        return Error(StatusCode::kNotFound, ErrorTag::kKeyMissing);
      }
      if (LoadPinCount(key_it->second.active_record) > 0) {
        return Error(StatusCode::kBusy, ErrorTag::kKeyBusy);
      }

      const SlotRef slot = key_it->second.active_record.slot;
      slots_to_recycle.push_back(slot);
      entries_to_clear.push_back(&key_it->second);
    }
  }

  {
    ScopedMethodTimer phase_timer(&timing_log_enabled_, "BatchFree",
                                  "clear_entries");
    for (KeyEntry* entry : entries_to_clear) {
      entry->has_active_record = false;
      entry->active_record = {};
    }
  }
  {
    ScopedMethodTimer phase_timer(&timing_log_enabled_, "BatchFree",
                                  "release_slots");
    for (const SlotRef& slot : slots_to_recycle) {
      ReleaseSlot(slot);
    }
  }

  return Status::OK();
}

void MetaManager::SetTimingLogEnabled(bool enabled) {
  timing_log_enabled_.store(enabled, std::memory_order_relaxed);
}

void MetaManager::SetPerformanceModeEnabled(bool enabled) {
  performance_mode_enabled_.store(enabled, std::memory_order_relaxed);
}

Address MetaManager::SlotAddress(const SlotRef& slot) const {
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

void MetaManager::ResizeAccessDesc(std::size_t count,
                                   AccessDesc* access_desc) const {
  if (access_desc == nullptr) {
    return;
  }

  access_desc->keys.resize(count);
  access_desc->addresses.resize(count);
  access_desc->lengths.resize(count);
  access_desc->valid.resize(count);
}

void MetaManager::FillAccessDescAt(const KeyId* key,
                                   Address address,
                                   std::uint64_t length,
                                   std::size_t index,
                                   AccessDesc* access_desc) const {
  if (access_desc == nullptr) {
    return;
  }

  access_desc->keys[index] = key;
  access_desc->addresses[index] = address;
  access_desc->lengths[index] = length;
  access_desc->valid[index] = 1;
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
  if (slot_class == SlotClass::kShort) {
    const std::size_t slot_index = short_free_slots_.front();
    short_free_slots_.pop();
    return SlotRef{SlotClass::kShort, slot_index};
  }

  const std::size_t slot_index = long_free_slots_.front();
  long_free_slots_.pop();
  return SlotRef{SlotClass::kLong, slot_index};
}

void MetaManager::ReleaseSlot(const SlotRef& slot) {
  if (slot.slot_class == SlotClass::kShort) {
    short_free_slots_.push(slot.slot_index);
  } else {
    long_free_slots_.push(slot.slot_index);
  }
}

std::unique_ptr<MetaManager> CreateInMemoryMetaManager() {
  return std::make_unique<MetaManager>();
}

}  // namespace kvaf
