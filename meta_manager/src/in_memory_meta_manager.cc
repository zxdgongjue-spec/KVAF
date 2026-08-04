#include "kvaf/meta_manager/meta_manager.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace kvaf {
namespace {

// This implementation deliberately keeps the existing public MetaManager API.
// It does not use a fixed number of key shards. The hot lookup path uses a
// shared index lock, immutable shared records, atomic pin counts, and direct
// lock-context references. Insert/erase operations briefly take the exclusive
// index lock; slow data transfers remain outside all metadata locks.

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
    // Keep this disabled in throughput tests. std::cout itself serializes.
    std::cout << "[MetaManagerTiming] method=" << method_name_
              << " elapsed_ns=" << elapsed_ns << '\n';
  }

 private:
  const std::atomic<bool>* enabled_;
  const char* method_name_;
  std::chrono::steady_clock::time_point start_;
};

enum class InternalSlotClass : std::uint8_t {
  kShort,
  kLong,
};

struct InternalSlotRef {
  InternalSlotClass slot_class{InternalSlotClass::kShort};
  std::size_t slot_index{0};
};

// A lock-free bitmap allocator for one slot class. There is no fixed shard
// count and no queue-wide mutex. The hint only spreads concurrent scans.
class AtomicSlotPool {
 public:
  AtomicSlotPool() = default;
  AtomicSlotPool(const AtomicSlotPool&) = delete;
  AtomicSlotPool& operator=(const AtomicSlotPool&) = delete;

  void Initialize(std::size_t slot_count) {
    slot_count_ = slot_count;
    word_count_ = (slot_count + 63U) / 64U;

    std::unique_ptr<std::atomic<std::uint64_t>[]> new_words;
    if (word_count_ != 0) {
      new_words = std::make_unique<std::atomic<std::uint64_t>[]>(word_count_);
      for (std::size_t i = 0; i < word_count_; ++i) {
        new_words[i].store(~std::uint64_t{0}, std::memory_order_relaxed);
      }
      const std::size_t valid_bits = slot_count & 63U;
      if (valid_bits != 0) {
        new_words[word_count_ - 1].store(
            (std::uint64_t{1} << valid_bits) - 1,
            std::memory_order_relaxed);
      }
    }

    words_ = std::move(new_words);
    next_word_.store(0, std::memory_order_relaxed);
    free_count_.store(slot_count, std::memory_order_release);
  }

  bool TryAcquire(std::size_t count, std::vector<std::size_t>* out) {
    if (out == nullptr) {
      return false;
    }
    out->clear();
    if (count == 0) {
      return true;
    }
    if (free_count_.load(std::memory_order_acquire) < count) {
      return false;
    }

    out->reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      std::size_t slot_index = 0;
      if (!TryAcquireOne(&slot_index)) {
        for (const std::size_t acquired : *out) {
          Release(acquired);
        }
        out->clear();
        return false;
      }
      out->push_back(slot_index);
    }
    return true;
  }

  void Release(std::size_t slot_index) {
    if (slot_index >= slot_count_) {
      return;
    }
    const std::size_t word_index = slot_index >> 6U;
    const std::uint64_t mask = std::uint64_t{1} << (slot_index & 63U);
    const std::uint64_t previous =
        words_[word_index].fetch_or(mask, std::memory_order_acq_rel);
    // A set bit means a double release. Do not inflate free_count in that case.
    if ((previous & mask) == 0) {
      free_count_.fetch_add(1, std::memory_order_release);
    }
  }

  std::size_t FreeCount() const {
    return free_count_.load(std::memory_order_acquire);
  }

 private:
  static unsigned FirstSetBit(std::uint64_t value) {
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<unsigned>(__builtin_ctzll(value));
#else
    unsigned bit = 0;
    while ((value & std::uint64_t{1}) == 0) {
      value >>= 1U;
      ++bit;
    }
    return bit;
#endif
  }

  bool TryAcquireOne(std::size_t* slot_index) {
    if (slot_index == nullptr || word_count_ == 0) {
      return false;
    }

    const std::size_t start =
        next_word_.fetch_add(1, std::memory_order_relaxed) % word_count_;
    for (std::size_t offset = 0; offset < word_count_; ++offset) {
      const std::size_t word_index = (start + offset) % word_count_;
      std::uint64_t available =
          words_[word_index].load(std::memory_order_acquire);
      while (available != 0) {
        const unsigned bit = FirstSetBit(available);
        const std::uint64_t mask = std::uint64_t{1} << bit;
        const std::uint64_t desired = available & ~mask;
        if (words_[word_index].compare_exchange_weak(
                available, desired, std::memory_order_acq_rel,
                std::memory_order_acquire)) {
          *slot_index = (word_index << 6U) + bit;
          free_count_.fetch_sub(1, std::memory_order_acq_rel);
          return true;
        }
        // compare_exchange_weak refreshes available on failure.
      }
    }
    return false;
  }

  std::size_t slot_count_{0};
  std::size_t word_count_{0};
  std::unique_ptr<std::atomic<std::uint64_t>[]> words_;
  std::atomic<std::size_t> next_word_{0};
  std::atomic<std::size_t> free_count_{0};
};

enum class PutGroupState : std::uint8_t {
  kPending,
  kVisible,
  kAborted,
};

struct PutGroup {
  std::atomic<PutGroupState> state{PutGroupState::kPending};
};

struct KeyRecord {
  KeyRecord(KeyId input_key,
            InternalSlotRef input_slot,
            std::uint64_t input_valid_length,
            std::shared_ptr<PutGroup> input_group)
      : key(std::move(input_key)),
        slot(input_slot),
        valid_length(input_valid_length),
        group(std::move(input_group)) {}

  const KeyId key;
  const InternalSlotRef slot;
  const std::uint64_t valid_length;
  const std::shared_ptr<PutGroup> group;
  std::atomic<std::uint64_t> pin_count{0};
  std::atomic<bool> deleted{false};
};

using KeyRecordPtr = std::shared_ptr<KeyRecord>;

struct PendingPutContextState {
  std::shared_ptr<PutGroup> group;
  std::vector<KeyRecordPtr> items;
};

struct ActiveLockContextState {
  mutable std::shared_mutex lifecycle_mutex;
  std::vector<KeyRecordPtr> items;
};

struct ManagerState {
  std::mutex register_mutex;
  RemoteRegionDesc region{};
  std::atomic<bool> region_registered{false};

  // No fixed key shards. Lookups share this lock; insert/erase hold it only for
  // the map mutation window. Key records stay alive through shared_ptr.
  mutable std::shared_mutex key_index_mutex;
  std::unordered_map<KeyId, KeyRecordPtr> key_index;

  mutable std::shared_mutex pending_context_mutex;
  std::unordered_map<std::uint64_t,
                     std::shared_ptr<PendingPutContextState>>
      pending_puts;

  mutable std::shared_mutex lock_context_mutex;
  std::unordered_map<std::uint64_t,
                     std::shared_ptr<ActiveLockContextState>>
      lock_contexts;

  AtomicSlotPool short_slots;
  AtomicSlotPool long_slots;

  std::atomic<std::uint64_t> next_put_context_id{1};
  std::atomic<std::uint64_t> next_lock_context_id{1};
  std::atomic<bool> timing_log_enabled{false};
  std::atomic<bool> performance_mode_enabled{false};
};

// The existing header has no pImpl field. To provide a source-only drop-in
// implementation, state is kept in a side table keyed by MetaManager*. The
// intended deployment has one process-lifetime manager. Create resets an entry
// if the allocator reuses an old object address; all states are released at
// process shutdown by the registry destructor.
std::mutex g_manager_registry_mutex;
std::unordered_map<const MetaManager*, std::shared_ptr<ManagerState>>
    g_manager_registry;

// Avoid turning the source-only side table into another global hot lock. Each
// worker resolves a manager once, then reuses the process-lifetime state.
thread_local const MetaManager* g_cached_manager = nullptr;
thread_local ManagerState* g_cached_state = nullptr;

ManagerState* GetManagerState(const MetaManager* manager) {
  if (g_cached_manager == manager && g_cached_state != nullptr) {
    return g_cached_state;
  }

  std::lock_guard<std::mutex> lock(g_manager_registry_mutex);
  auto it = g_manager_registry.find(manager);
  if (it == g_manager_registry.end()) {
    it = g_manager_registry
             .emplace(manager, std::make_shared<ManagerState>())
             .first;
  }

  g_cached_manager = manager;
  g_cached_state = it->second.get();
  return g_cached_state;
}

void ResetManagerState(const MetaManager* manager) {
  auto state = std::make_shared<ManagerState>();
  ManagerState* const state_ptr = state.get();
  {
    std::lock_guard<std::mutex> lock(g_manager_registry_mutex);
    g_manager_registry[manager] = std::move(state);
  }
  g_cached_manager = manager;
  g_cached_state = state_ptr;
}

bool CheckedAdd(std::uint64_t lhs,
                std::uint64_t rhs,
                std::uint64_t* result) {
  if (result == nullptr ||
      lhs > std::numeric_limits<std::uint64_t>::max() - rhs) {
    return false;
  }
  *result = lhs + rhs;
  return true;
}

bool CheckedMultiply(std::uint64_t lhs,
                     std::uint64_t rhs,
                     std::uint64_t* result) {
  if (result == nullptr ||
      (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs)) {
    return false;
  }
  *result = lhs * rhs;
  return true;
}

bool RegionReady(const ManagerState& state) {
  return state.region_registered.load(std::memory_order_acquire);
}

bool ResolveSlotClassForState(const ManagerState& state,
                         std::uint64_t length,
                         InternalSlotClass* slot_class) {
  if (slot_class == nullptr || length == 0) {
    return false;
  }
  if (length <= state.region.short_slot_size) {
    *slot_class = InternalSlotClass::kShort;
    return true;
  }
  if (length <= state.region.long_slot_size) {
    *slot_class = InternalSlotClass::kLong;
    return true;
  }
  return false;
}

Address ComputeSlotAddressForState(const ManagerState& state, const InternalSlotRef& slot) {
  if (slot.slot_class == InternalSlotClass::kShort) {
    return state.region.base +
           static_cast<Address>(slot.slot_index) *
               state.region.short_slot_size;
  }
  const Address long_region_base =
      state.region.base +
      state.region.slot_count_per_class * state.region.short_slot_size;
  return long_region_base +
         static_cast<Address>(slot.slot_index) * state.region.long_slot_size;
}

void FillAccessEntryFromRecord(const KeyRecordPtr& record, AccessEntry* entry) {
  if (entry == nullptr || record == nullptr) {
    return;
  }
  entry->key = &record->key;
  entry->address = 0;  // Set by caller because address depends on ManagerState.
  entry->length = record->valid_length;
  entry->valid = 1;
}

void ReleaseSlotToState(ManagerState* state, const InternalSlotRef& slot) {
  if (slot.slot_class == InternalSlotClass::kShort) {
    state->short_slots.Release(slot.slot_index);
  } else {
    state->long_slots.Release(slot.slot_index);
  }
}

bool ReserveSlotsFromState(
    ManagerState* state,
    const std::vector<InternalSlotClass>& slot_classes,
    std::vector<InternalSlotRef>* result) {
  if (state == nullptr || result == nullptr) {
    return false;
  }

  std::size_t short_count = 0;
  std::size_t long_count = 0;
  for (const InternalSlotClass slot_class : slot_classes) {
    if (slot_class == InternalSlotClass::kShort) {
      ++short_count;
    } else {
      ++long_count;
    }
  }

  std::vector<std::size_t> short_indices;
  std::vector<std::size_t> long_indices;
  short_indices.reserve(short_count);
  long_indices.reserve(long_count);
  result->clear();
  result->reserve(slot_classes.size());

  if (!state->short_slots.TryAcquire(short_count, &short_indices)) {
    return false;
  }
  if (!state->long_slots.TryAcquire(long_count, &long_indices)) {
    for (const std::size_t index : short_indices) {
      state->short_slots.Release(index);
    }
    return false;
  }

  std::size_t short_position = 0;
  std::size_t long_position = 0;
  for (const InternalSlotClass slot_class : slot_classes) {
    if (slot_class == InternalSlotClass::kShort) {
      result->push_back(
          InternalSlotRef{slot_class, short_indices[short_position++]});
    } else {
      result->push_back(
          InternalSlotRef{slot_class, long_indices[long_position++]});
    }
  }
  return true;
}

}  // namespace

Status MetaManager::RegisterRemoteRegion(const RemoteRegionDesc& region) {
  const auto state = GetManagerState(this);
  ScopedMethodTimer timer(&state->timing_log_enabled, "RegisterRemoteRegion");
  std::lock_guard<std::mutex> lock(state->register_mutex);

  if (RegionReady(*state)) {
    return Error(StatusCode::kAlreadyExists, ErrorTag::kRegionExists);
  }
  if (region.short_slot_size == 0 || region.long_slot_size == 0 ||
      region.slot_count_per_class == 0 ||
      region.short_slot_size >= region.long_slot_size) {
    return Error(StatusCode::kInvalidArgument, ErrorTag::kInvalidRegion);
  }

  std::uint64_t total_slot_size = 0;
  std::uint64_t expected_length = 0;
  if (!CheckedAdd(region.short_slot_size, region.long_slot_size,
                  &total_slot_size) ||
      !CheckedMultiply(region.slot_count_per_class, total_slot_size,
                       &expected_length) ||
      region.length != expected_length) {
    return Error(StatusCode::kInvalidArgument, ErrorTag::kInvalidRegion);
  }
  if (region.slot_count_per_class >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return Error(StatusCode::kInvalidArgument, ErrorTag::kInvalidRegion);
  }

  try {
    const std::size_t slot_count =
        static_cast<std::size_t>(region.slot_count_per_class);
    state->short_slots.Initialize(slot_count);
    state->long_slots.Initialize(slot_count);

    std::unique_lock<std::shared_mutex> index_lock(state->key_index_mutex);
    const std::uint64_t maximum_records_u64 =
        region.slot_count_per_class >
                std::numeric_limits<std::uint64_t>::max() / 2
            ? std::numeric_limits<std::uint64_t>::max()
            : region.slot_count_per_class * 2;
    const std::size_t maximum_records =
        maximum_records_u64 >
                static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max())
            ? std::numeric_limits<std::size_t>::max()
            : static_cast<std::size_t>(maximum_records_u64);
    state->key_index.reserve(maximum_records);
  } catch (const std::bad_alloc&) {
    return Error(StatusCode::kOutOfMemory, ErrorTag::kHostOutOfMemory);
  }

  state->region = region;
  state->region_registered.store(true, std::memory_order_release);
  return Status::OK();
}

Status MetaManager::PreparePut(const std::vector<KeyId>& flat_keys,
                               const std::vector<std::uint64_t>& lengths,
                               PutContext* put_context,
                               AccessDesc* put_desc) {
  const auto state = GetManagerState(this);
  ScopedMethodTimer timer(&state->timing_log_enabled, "PreparePut");

  if (put_context == nullptr || put_desc == nullptr || flat_keys.empty() ||
      flat_keys.size() != lengths.size()) {
    return Error(StatusCode::kInvalidArgument, ErrorTag::kInvalidRequest);
  }
  if (!RegionReady(*state)) {
    return Error(StatusCode::kNotInitialized, ErrorTag::kRegionMissing);
  }

  std::vector<InternalSlotClass> slot_classes;
  std::unordered_set<KeyId> batch_seen;
  try {
    slot_classes.reserve(flat_keys.size());
    batch_seen.reserve(flat_keys.size());
    for (std::size_t i = 0; i < flat_keys.size(); ++i) {
      if (!batch_seen.emplace(flat_keys[i]).second) {
        return Error(StatusCode::kInvalidArgument, ErrorTag::kInvalidRequest);
      }
      InternalSlotClass slot_class;
      if (!ResolveSlotClassForState(*state, lengths[i], &slot_class)) {
        return Error(StatusCode::kInvalidArgument, ErrorTag::kInvalidRequest);
      }
      slot_classes.push_back(slot_class);
    }
  } catch (const std::bad_alloc&) {
    return Error(StatusCode::kOutOfMemory, ErrorTag::kHostOutOfMemory);
  }

  std::vector<InternalSlotRef> slots;
  try {
    if (!ReserveSlotsFromState(state, slot_classes, &slots)) {
      return Error(StatusCode::kOutOfMemory, ErrorTag::kOutOfSlots);
    }
  } catch (const std::bad_alloc&) {
    return Error(StatusCode::kOutOfMemory, ErrorTag::kHostOutOfMemory);
  }

  std::shared_ptr<PutGroup> group;
  std::shared_ptr<PendingPutContextState> context;
  std::vector<AccessEntry> entries;
  try {
    group = std::make_shared<PutGroup>();
    context = std::make_shared<PendingPutContextState>();
    context->group = group;
    context->items.reserve(flat_keys.size());
    entries.reserve(flat_keys.size());

    for (std::size_t i = 0; i < flat_keys.size(); ++i) {
      auto record = std::make_shared<KeyRecord>(
          flat_keys[i], slots[i], lengths[i], group);
      context->items.push_back(record);

      AccessEntry entry;
      FillAccessEntryFromRecord(record, &entry);
      entry.address = ComputeSlotAddressForState(*state, record->slot);
      entries.push_back(std::move(entry));
    }
  } catch (const std::bad_alloc&) {
    for (const InternalSlotRef& slot : slots) {
      ReleaseSlotToState(state, slot);
    }
    return Error(StatusCode::kOutOfMemory, ErrorTag::kHostOutOfMemory);
  }

  const std::uint64_t context_id =
      state->next_put_context_id.fetch_add(1, std::memory_order_relaxed);

  bool context_inserted = false;
  std::vector<KeyId> inserted_keys;
  try {
    inserted_keys.reserve(flat_keys.size());

    // Global lock order: pending_context_mutex -> key_index_mutex.
    std::unique_lock<std::shared_mutex> pending_lock(
        state->pending_context_mutex);
    std::unique_lock<std::shared_mutex> index_lock(state->key_index_mutex);

    for (const KeyId& key : flat_keys) {
      if (state->key_index.find(key) != state->key_index.end()) {
        index_lock.unlock();
        pending_lock.unlock();
        for (const InternalSlotRef& slot : slots) {
          ReleaseSlotToState(state, slot);
        }
        return Error(StatusCode::kAlreadyExists, ErrorTag::kKeyExists);
      }
    }

    context_inserted =
        state->pending_puts.emplace(context_id, context).second;
    if (!context_inserted) {
      index_lock.unlock();
      pending_lock.unlock();
      for (const InternalSlotRef& slot : slots) {
        ReleaseSlotToState(state, slot);
      }
      return Error(StatusCode::kInvalidState, ErrorTag::kStateConflict);
    }

    for (const KeyRecordPtr& record : context->items) {
      const auto inserted = state->key_index.emplace(record->key, record);
      if (!inserted.second) {
        for (const KeyId& key : inserted_keys) {
          state->key_index.erase(key);
        }
        state->pending_puts.erase(context_id);
        index_lock.unlock();
        pending_lock.unlock();
        for (const InternalSlotRef& slot : slots) {
          ReleaseSlotToState(state, slot);
        }
        return Error(StatusCode::kInvalidState, ErrorTag::kStateConflict);
      }
      inserted_keys.push_back(record->key);
    }
  } catch (const std::bad_alloc&) {
    std::unique_lock<std::shared_mutex> pending_lock(
        state->pending_context_mutex);
    std::unique_lock<std::shared_mutex> index_lock(state->key_index_mutex);
    for (const KeyId& key : inserted_keys) {
      state->key_index.erase(key);
    }
    if (context_inserted) {
      state->pending_puts.erase(context_id);
    }
    index_lock.unlock();
    pending_lock.unlock();
    for (const InternalSlotRef& slot : slots) {
      ReleaseSlotToState(state, slot);
    }
    return Error(StatusCode::kOutOfMemory, ErrorTag::kHostOutOfMemory);
  }

  put_desc->entries.swap(entries);
  put_context->id = context_id;
  return Status::OK();
}

Status MetaManager::CommitPut(const PutContext& put_context) {
  const auto state = GetManagerState(this);
  ScopedMethodTimer timer(&state->timing_log_enabled, "CommitPut");
  if (!RegionReady(*state)) {
    return Error(StatusCode::kNotInitialized, ErrorTag::kRegionMissing);
  }

  std::unique_lock<std::shared_mutex> lock(state->pending_context_mutex);
  const auto it = state->pending_puts.find(put_context.id);
  if (it == state->pending_puts.end()) {
    return Error(StatusCode::kNotFound, ErrorTag::kPutContextMissing);
  }

  PutGroupState expected = PutGroupState::kPending;
  if (!it->second->group->state.compare_exchange_strong(
          expected, PutGroupState::kVisible, std::memory_order_release,
          std::memory_order_acquire)) {
    return Error(StatusCode::kInvalidState, ErrorTag::kStateConflict);
  }

  // One atomic group-state transition makes every key in the batch visible.
  state->pending_puts.erase(it);
  return Status::OK();
}

Status MetaManager::AbortPut(const PutContext& put_context) {
  const auto state = GetManagerState(this);
  ScopedMethodTimer timer(&state->timing_log_enabled, "AbortPut");
  if (!RegionReady(*state)) {
    return Error(StatusCode::kNotInitialized, ErrorTag::kRegionMissing);
  }

  std::shared_ptr<PendingPutContextState> context;
  {
    // Same lock order as PreparePut.
    std::unique_lock<std::shared_mutex> pending_lock(
        state->pending_context_mutex);
    const auto context_it = state->pending_puts.find(put_context.id);
    if (context_it == state->pending_puts.end()) {
      return Error(StatusCode::kNotFound, ErrorTag::kPutContextMissing);
    }
    context = context_it->second;

    std::unique_lock<std::shared_mutex> index_lock(state->key_index_mutex);
    PutGroupState expected = PutGroupState::kPending;
    if (!context->group->state.compare_exchange_strong(
            expected, PutGroupState::kAborted, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
      return Error(StatusCode::kInvalidState, ErrorTag::kStateConflict);
    }

    for (const KeyRecordPtr& record : context->items) {
      const auto key_it = state->key_index.find(record->key);
      if (key_it == state->key_index.end() ||
          key_it->second.get() != record.get()) {
        return Error(StatusCode::kInvalidState, ErrorTag::kStateConflict);
      }
    }
    for (const KeyRecordPtr& record : context->items) {
      record->deleted.store(true, std::memory_order_release);
      state->key_index.erase(record->key);
    }
    state->pending_puts.erase(context_it);
  }

  for (const KeyRecordPtr& record : context->items) {
    ReleaseSlotToState(state, record->slot);
  }
  return Status::OK();
}

Status MetaManager::LockKeys(const std::vector<KeyId>& flat_keys,
                             MetaManagerLockMode lock_mode,
                             LockContext* lock_context) {
  const auto state = GetManagerState(this);
  ScopedMethodTimer timer(&state->timing_log_enabled, "LockKeys");

  if (lock_context == nullptr || flat_keys.empty()) {
    return Error(StatusCode::kInvalidArgument, ErrorTag::kInvalidRequest);
  }
  if (lock_mode != MetaManagerLockMode::kStableReadAllowAppend) {
    return Error(StatusCode::kInvalidArgument, ErrorTag::kUnsupportedMode);
  }
  if (!RegionReady(*state)) {
    return Error(StatusCode::kNotInitialized, ErrorTag::kRegionMissing);
  }

  std::shared_ptr<ActiveLockContextState> context;
  try {
    context = std::make_shared<ActiveLockContextState>();
    context->items.reserve(flat_keys.size());
  } catch (const std::bad_alloc&) {
    return Error(StatusCode::kOutOfMemory, ErrorTag::kHostOutOfMemory);
  }

  {
    // Shared map lock prevents erase while we take pins. Different readers do
    // not block one another, and the slow data read happens after this returns.
    std::shared_lock<std::shared_mutex> index_lock(state->key_index_mutex);
    for (const KeyId& key : flat_keys) {
      const auto it = state->key_index.find(key);
      if (it == state->key_index.end() ||
          it->second->deleted.load(std::memory_order_acquire) ||
          it->second->group->state.load(std::memory_order_acquire) !=
              PutGroupState::kVisible) {
        return Error(StatusCode::kNotFound, ErrorTag::kKeyMissing);
      }
      context->items.push_back(it->second);
    }
    for (const KeyRecordPtr& record : context->items) {
      record->pin_count.fetch_add(1, std::memory_order_acq_rel);
    }
  }

  const std::uint64_t context_id =
      state->next_lock_context_id.fetch_add(1, std::memory_order_relaxed);
  try {
    std::unique_lock<std::shared_mutex> context_lock(
        state->lock_context_mutex);
    if (!state->lock_contexts.emplace(context_id, context).second) {
      for (const KeyRecordPtr& record : context->items) {
        record->pin_count.fetch_sub(1, std::memory_order_acq_rel);
      }
      return Error(StatusCode::kInvalidState, ErrorTag::kStateConflict);
    }
  } catch (const std::bad_alloc&) {
    for (const KeyRecordPtr& record : context->items) {
      record->pin_count.fetch_sub(1, std::memory_order_acq_rel);
    }
    return Error(StatusCode::kOutOfMemory, ErrorTag::kHostOutOfMemory);
  }

  lock_context->id = context_id;
  return Status::OK();
}

Status MetaManager::BatchLookup(const std::vector<KeyId>& flat_keys,
                                const LockContext& lock_context,
                                AccessDesc* access_desc) {
  const auto state = GetManagerState(this);
  ScopedMethodTimer timer(&state->timing_log_enabled, "BatchLookup");

  if (access_desc == nullptr) {
    return Error(StatusCode::kInvalidArgument, ErrorTag::kInvalidRequest);
  }
  if (!RegionReady(*state)) {
    return Error(StatusCode::kNotInitialized, ErrorTag::kRegionMissing);
  }

  std::shared_ptr<ActiveLockContextState> context;
  std::shared_lock<std::shared_mutex> lifecycle_lock;
  {
    std::shared_lock<std::shared_mutex> context_map_lock(
        state->lock_context_mutex);
    const auto it = state->lock_contexts.find(lock_context.id);
    if (it == state->lock_contexts.end()) {
      return Error(StatusCode::kNotFound, ErrorTag::kLockContextMissing);
    }
    context = it->second;
    lifecycle_lock =
        std::shared_lock<std::shared_mutex>(context->lifecycle_mutex);
  }

  if (context->items.size() != flat_keys.size()) {
    return Error(StatusCode::kInvalidArgument, ErrorTag::kInvalidRequest);
  }

  std::vector<AccessEntry> entries;
  try {
    entries.reserve(context->items.size());
    const bool performance_mode =
        state->performance_mode_enabled.load(std::memory_order_relaxed);
    for (std::size_t i = 0; i < context->items.size(); ++i) {
      const KeyRecordPtr& record = context->items[i];
      if (record == nullptr ||
          record->deleted.load(std::memory_order_acquire)) {
        return Error(StatusCode::kInvalidState, ErrorTag::kStateConflict);
      }
      if (!performance_mode && record->key != flat_keys[i]) {
        return Error(StatusCode::kInvalidArgument, ErrorTag::kInvalidRequest);
      }
      AccessEntry entry;
      FillAccessEntryFromRecord(record, &entry);
      entry.address = ComputeSlotAddressForState(*state, record->slot);
      entries.push_back(std::move(entry));
    }
  } catch (const std::bad_alloc&) {
    return Error(StatusCode::kOutOfMemory, ErrorTag::kHostOutOfMemory);
  }

  access_desc->entries.swap(entries);
  return Status::OK();
}

Status MetaManager::UnlockKeys(const LockContext& lock_context) {
  const auto state = GetManagerState(this);
  ScopedMethodTimer timer(&state->timing_log_enabled, "UnlockKeys");
  if (!RegionReady(*state)) {
    return Error(StatusCode::kNotInitialized, ErrorTag::kRegionMissing);
  }

  std::shared_ptr<ActiveLockContextState> context;
  {
    std::unique_lock<std::shared_mutex> context_map_lock(
        state->lock_context_mutex);
    const auto it = state->lock_contexts.find(lock_context.id);
    if (it == state->lock_contexts.end()) {
      return Error(StatusCode::kNotFound, ErrorTag::kLockContextMissing);
    }
    context = it->second;

    // Lock order is map -> lifecycle. BatchLookup follows the same order.
    std::unique_lock<std::shared_mutex> lifecycle_lock(
        context->lifecycle_mutex);
    for (const KeyRecordPtr& record : context->items) {
      if (record == nullptr ||
          record->pin_count.load(std::memory_order_acquire) == 0) {
        return Error(StatusCode::kInvalidState, ErrorTag::kStateConflict);
      }
    }
    state->lock_contexts.erase(it);
    for (const KeyRecordPtr& record : context->items) {
      record->pin_count.fetch_sub(1, std::memory_order_acq_rel);
    }
  }
  return Status::OK();
}

Status MetaManager::BatchFree(const std::vector<KeyId>& flat_keys) {
  const auto state = GetManagerState(this);
  ScopedMethodTimer timer(&state->timing_log_enabled, "BatchFree");
  if (flat_keys.empty()) {
    return Error(StatusCode::kInvalidArgument, ErrorTag::kInvalidRequest);
  }
  if (!RegionReady(*state)) {
    return Error(StatusCode::kNotInitialized, ErrorTag::kRegionMissing);
  }

  std::vector<KeyId> unique_keys;
  std::unordered_set<KeyId> seen;
  try {
    unique_keys.reserve(flat_keys.size());
    seen.reserve(flat_keys.size());
    for (const KeyId& key : flat_keys) {
      if (seen.emplace(key).second) {
        unique_keys.push_back(key);
      }
    }
  } catch (const std::bad_alloc&) {
    return Error(StatusCode::kOutOfMemory, ErrorTag::kHostOutOfMemory);
  }

  std::vector<KeyRecordPtr> records;
  try {
    records.reserve(unique_keys.size());
  } catch (const std::bad_alloc&) {
    return Error(StatusCode::kOutOfMemory, ErrorTag::kHostOutOfMemory);
  }

  {
    // Exclusive lock closes the race with LockKeys: either the pin is taken
    // first and free returns busy, or erase wins and the later lock sees miss.
    std::unique_lock<std::shared_mutex> index_lock(state->key_index_mutex);
    for (const KeyId& key : unique_keys) {
      const auto it = state->key_index.find(key);
      if (it == state->key_index.end() ||
          it->second->deleted.load(std::memory_order_acquire) ||
          it->second->group->state.load(std::memory_order_acquire) !=
              PutGroupState::kVisible) {
        return Error(StatusCode::kNotFound, ErrorTag::kKeyMissing);
      }
      if (it->second->pin_count.load(std::memory_order_acquire) != 0) {
        return Error(StatusCode::kBusy, ErrorTag::kKeyBusy);
      }
      records.push_back(it->second);
    }

    for (const KeyRecordPtr& record : records) {
      record->deleted.store(true, std::memory_order_release);
      state->key_index.erase(record->key);
    }
  }

  for (const KeyRecordPtr& record : records) {
    ReleaseSlotToState(state, record->slot);
  }
  return Status::OK();
}

void MetaManager::SetTimingLogEnabled(bool enabled) {
  GetManagerState(this)->timing_log_enabled.store(enabled,
                                                   std::memory_order_relaxed);
}

void MetaManager::SetPerformanceModeEnabled(bool enabled) {
  GetManagerState(this)->performance_mode_enabled.store(
      enabled, std::memory_order_relaxed);
}

std::unique_ptr<MetaManager> CreateInMemoryMetaManager() {
  auto manager = std::make_unique<MetaManager>();
  ResetManagerState(manager.get());
  return manager;
}

}  // namespace kvaf
