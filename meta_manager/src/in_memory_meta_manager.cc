#include "kvaf/meta_manager/meta_manager.h"

#include <algorithm>
#include <chrono>
#include <deque>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <string>
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

struct MetaManager::EntryIndex {
  struct Bucket {
    std::size_t hash{0};
    const KeyId* key{nullptr};
    SlotRef slot;
    std::uint64_t valid_length{0};
    RecordState state{RecordState::kUnused};
  };

  struct InsertResult {
    Bucket* bucket{nullptr};
    const KeyId* key{nullptr};
    std::size_t hash{0};
    bool inserted{false};
  };

  static constexpr std::uint8_t kEmptyControl = 0;

  void Reserve(std::size_t max_entries) {
    const std::size_t max_size = std::numeric_limits<std::size_t>::max();
    const std::size_t required_capacity =
        max_entries > max_size / 2
            ? max_size
            : std::max<std::size_t>(8, max_entries * 2);
    std::size_t capacity = 1;
    while (capacity < required_capacity && capacity <= max_size / 2) {
      capacity <<= 1;
    }
    if (capacity < required_capacity) {
      capacity = required_capacity;
    }
    if (capacity > buckets_.size()) {
      Rehash(capacity);
    }
  }

  Bucket* Find(const KeyId& key) {
    const std::size_t hash = std::hash<KeyId>{}(key);
    return FindWithHash(key, hash);
  }

  Bucket* FindStored(const KeyId* key, std::size_t hash) {
    if (key == nullptr || buckets_.empty()) {
      return nullptr;
    }

    const std::uint8_t fingerprint = Fingerprint(hash);
    const std::size_t mask = buckets_.size() - 1;
    std::size_t bucket = hash & mask;
    for (std::size_t probes = 0; probes < buckets_.size(); ++probes) {
      const std::uint8_t control = control_[bucket];
      if (control == kEmptyControl) {
        return nullptr;
      }
      if (control == fingerprint) {
        Bucket& candidate = buckets_[bucket];
        if (candidate.hash == hash && candidate.key == key) {
          return &candidate;
        }
      }
      bucket = (bucket + 1) & mask;
    }
    return nullptr;
  }

  InsertResult TryEmplace(const KeyId& key) {
    if (buckets_.empty()) {
      Reserve(8);
    }
    if ((size_ + 1) * 2 >= buckets_.size()) {
      Rehash(buckets_.size() * 2);
    }

    const std::size_t mask = buckets_.size() - 1;
    const std::size_t hash = std::hash<KeyId>{}(key);
    const std::uint8_t fingerprint = Fingerprint(hash);
    std::size_t bucket = hash & mask;
    for (;;) {
      const std::uint8_t control = control_[bucket];
      if (control == kEmptyControl) {
        keys_.emplace_back(key);
        const KeyId* stored_key = &keys_.back();
        Bucket& new_bucket = buckets_[bucket];
        new_bucket = Bucket{hash, stored_key};
        control_[bucket] = fingerprint;
        ++size_;
        return InsertResult{&new_bucket, stored_key, hash, true};
      }
      if (control == fingerprint) {
        Bucket& candidate = buckets_[bucket];
        if (candidate.hash == hash && *candidate.key == key) {
          return InsertResult{&candidate, candidate.key, hash, false};
        }
      }
      bucket = (bucket + 1) & mask;
    }
  }

 private:
  Bucket* FindWithHash(const KeyId& key, std::size_t hash) {
    if (buckets_.empty()) {
      return nullptr;
    }

    const std::uint8_t fingerprint = Fingerprint(hash);
    const std::size_t mask = buckets_.size() - 1;
    std::size_t bucket = hash & mask;
    for (std::size_t probes = 0; probes < buckets_.size(); ++probes) {
      const std::uint8_t control = control_[bucket];
      if (control == kEmptyControl) {
        return nullptr;
      }
      if (control == fingerprint) {
        Bucket& candidate = buckets_[bucket];
        if (candidate.hash == hash && *candidate.key == key) {
          return &candidate;
        }
      }
      bucket = (bucket + 1) & mask;
    }
    return nullptr;
  }

  static std::uint8_t Fingerprint(std::size_t hash) {
    const std::uint64_t mixed_hash = static_cast<std::uint64_t>(hash);
    const std::uint8_t low_byte = static_cast<std::uint8_t>(
        mixed_hash ^ (mixed_hash >> 32));
    return static_cast<std::uint8_t>((low_byte & 0xfeu) | 2u);
  }

  void Rehash(std::size_t capacity) {
    std::vector<Bucket> new_buckets(capacity);
    std::vector<std::uint8_t> new_control(capacity, kEmptyControl);
    const std::size_t mask = capacity - 1;
    for (std::size_t old_bucket = 0; old_bucket < buckets_.size();
         ++old_bucket) {
      if (control_[old_bucket] == kEmptyControl) {
        continue;
      }

      const Bucket& entry = buckets_[old_bucket];
      std::size_t bucket = entry.hash & mask;
      while (new_control[bucket] != kEmptyControl) {
        bucket = (bucket + 1) & mask;
      }
      new_buckets[bucket] = entry;
      new_control[bucket] = Fingerprint(entry.hash);
    }

    buckets_.swap(new_buckets);
    control_.swap(new_control);
  }

  std::deque<KeyId> keys_;
  std::vector<Bucket> buckets_;
  std::vector<std::uint8_t> control_;
  std::size_t size_{0};
};

MetaManager::MetaManager()
    : entries_(std::make_unique<EntryIndex>()) {}

MetaManager::~MetaManager() = default;

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
    entries_->Reserve(max_active_keys);
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
    for (const KeyId* key : new_keys) {
      EntryIndex::Bucket* bucket =
          entries_->FindStored(key, std::hash<KeyId>{}(*key));
      if (bucket != nullptr) {
        bucket->state = RecordState::kFreed;
      }
    }
  };

  {
    ScopedMethodTimer phase_timer(&timing_log_enabled_, "PreparePut",
                                  "index_keys");
    try {
      for (std::size_t i = 0; i < flat_keys.size(); ++i) {
        const EntryIndex::InsertResult result =
            entries_->TryEmplace(flat_keys[i]);
        if (result.inserted) {
          new_keys.push_back(result.key);
        }

        EntryIndex::Bucket& bucket = *result.bucket;
        if (!result.inserted || bucket.state != RecordState::kUnused) {
          rollback_new_keys();
          return Error(StatusCode::kAlreadyExists, ErrorTag::kKeyExists);
        }

        bucket.state = RecordState::kPending;
        pending_context.items.push_back(PendingPutItem{
            result.key, result.hash, SlotRef{slot_classes[i], 0}, lengths[i]});
      }
    } catch (const std::bad_alloc&) {
      rollback_new_keys();
      return Error(StatusCode::kOutOfMemory, ErrorTag::kHostOutOfMemory);
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
      FillAccessDescAt(&flat_keys[i], SlotAddress(item.slot), item.valid_length,
                       i, put_desc);
    }
  }

  {
    ScopedMethodTimer phase_timer(&timing_log_enabled_, "PreparePut",
                                  "publish_context");
    try {
      const std::uint64_t ctx_id = next_put_context_id_++;
      auto inserted = pending_puts_.try_emplace(ctx_id);
      if (!inserted.second) {
        for (const PendingPutItem& item : pending_context.items) {
          EntryIndex::Bucket* bucket =
              entries_->FindStored(item.key, item.hash);
          if (bucket != nullptr) {
            bucket->state = RecordState::kFreed;
          }
          ReleaseSlot(item.slot);
        }
        rollback_new_keys();
        return Error(StatusCode::kInvalidState, ErrorTag::kStateConflict);
      }
      inserted.first->second.items.swap(pending_context.items);
      put_context->id = ctx_id;
    } catch (const std::bad_alloc&) {
      for (const PendingPutItem& item : pending_context.items) {
        EntryIndex::Bucket* bucket =
            entries_->FindStored(item.key, item.hash);
        if (bucket != nullptr) {
          bucket->state = RecordState::kFreed;
        }
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
    EntryIndex::Bucket* bucket = entries_->FindStored(item.key, item.hash);
    if (bucket == nullptr || bucket->state != RecordState::kPending) {
      return Error(StatusCode::kInvalidState, ErrorTag::kStateConflict);
    }
  }

  for (const PendingPutItem& item : pending_context.items) {
    EntryIndex::Bucket* bucket = entries_->FindStored(item.key, item.hash);
    bucket->slot = item.slot;
    bucket->valid_length = item.valid_length;
    bucket->state = RecordState::kActive;
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
    EntryIndex::Bucket* bucket = entries_->FindStored(item.key, item.hash);
    if (bucket == nullptr || bucket->state != RecordState::kPending) {
      return Error(StatusCode::kInvalidState, ErrorTag::kStateConflict);
    }
  }

  for (const PendingPutItem& item : pending_context.items) {
    EntryIndex::Bucket* bucket = entries_->FindStored(item.key, item.hash);
    bucket->state = RecordState::kFreed;
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
  {
    ScopedMethodTimer phase_timer(&timing_log_enabled_, "LockKeys",
                                  "read_phase");
    active_context.read_phase_lock =
        std::shared_lock<std::shared_mutex>(read_phase_mutex_);
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
        return Error(StatusCode::kInvalidState, ErrorTag::kStateConflict);
      }
      inserted.first->second.read_phase_lock =
          std::move(active_context.read_phase_lock);
    } catch (const std::bad_alloc&) {
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

  std::shared_lock<std::shared_mutex> context_lock;
  {
    ScopedMethodTimer phase_timer(&timing_log_enabled_, "BatchLookup",
                                  "lock_context");
    context_lock = std::shared_lock<std::shared_mutex>(lock_context_mutex_);
    if (lock_contexts_.find(lock_context.id) == lock_contexts_.end()) {
      return Error(StatusCode::kNotFound, ErrorTag::kLockContextMissing);
    }
  }

  {
    ScopedMethodTimer phase_timer(&timing_log_enabled_, "BatchLookup",
                                  "resize_desc");
    try {
      ResizeAccessDesc(flat_keys.size(), access_desc);
    } catch (const std::bad_alloc&) {
      return Error(StatusCode::kOutOfMemory, ErrorTag::kHostOutOfMemory);
    }
  }

  {
    ScopedMethodTimer phase_timer(&timing_log_enabled_, "BatchLookup",
                                  "find");
    for (std::size_t i = 0; i < flat_keys.size(); ++i) {
      const EntryIndex::Bucket* bucket = entries_->Find(flat_keys[i]);
      if (bucket == nullptr || bucket->state != RecordState::kActive) {
        return Error(StatusCode::kNotFound, ErrorTag::kKeyMissing);
      }

      FillAccessDescAt(&flat_keys[i], SlotAddress(bucket->slot),
                       bucket->valid_length, i, access_desc);
    }
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
  if (!active_context.read_phase_lock.owns_lock()) {
    return Error(StatusCode::kInvalidState, ErrorTag::kStateConflict);
  }

  // Erasing the context releases its read lease after the context is removed
  // from the lookup table, so BatchLookup cannot observe a reclaimed context.
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

  std::unique_lock<std::shared_mutex> read_phase_lock(read_phase_mutex_,
                                                       std::defer_lock);
  {
    ScopedMethodTimer phase_timer(&timing_log_enabled_, "BatchFree",
                                  "write_phase");
    if (!read_phase_lock.try_lock()) {
      return Error(StatusCode::kBusy, ErrorTag::kKeyBusy);
    }
  }

  std::unique_lock<std::mutex> write_lock(write_mutex_, std::defer_lock);
  {
    ScopedMethodTimer phase_timer(&timing_log_enabled_, "BatchFree", "lock");
    write_lock.lock();
  }

  std::vector<KeyId> unique_keys;
  const std::vector<KeyId>* keys_to_free = &flat_keys;
  const bool performance_mode =
      performance_mode_enabled_.load(std::memory_order_relaxed);
  {
    ScopedMethodTimer phase_timer(&timing_log_enabled_, "BatchFree",
                                  "deduplicate");
    if (performance_mode) {
      // Performance mode already skips duplicate-key validation, so avoid
      // copying the input strings before the lookup pass.
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
      keys_to_free = &unique_keys;
    }
  }

  std::vector<SlotRef> slots_to_recycle;
  std::vector<EntryIndex::Bucket*> entries_to_clear;
  {
    ScopedMethodTimer phase_timer(&timing_log_enabled_, "BatchFree",
                                  "reserve");
    try {
      slots_to_recycle.reserve(keys_to_free->size());
      entries_to_clear.reserve(keys_to_free->size());
    } catch (const std::bad_alloc&) {
      return Error(StatusCode::kOutOfMemory, ErrorTag::kHostOutOfMemory);
    }
  }

  {
    ScopedMethodTimer phase_timer(&timing_log_enabled_, "BatchFree",
                                  "validate_keys");
    for (const KeyId& key : *keys_to_free) {
      EntryIndex::Bucket* bucket = entries_->Find(key);
      if (bucket == nullptr || bucket->state != RecordState::kActive) {
        return Error(StatusCode::kNotFound, ErrorTag::kKeyMissing);
      }

      slots_to_recycle.push_back(bucket->slot);
      entries_to_clear.push_back(bucket);
    }
  }

  {
    ScopedMethodTimer phase_timer(&timing_log_enabled_, "BatchFree",
                                  "clear_entries");
    for (EntryIndex::Bucket* bucket : entries_to_clear) {
      bucket->slot = {};
      bucket->valid_length = 0;
      bucket->state = RecordState::kFreed;
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
