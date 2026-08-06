#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include "kvaf/status.h"
#include "kvaf/types.h"

namespace kvaf {

// Writes and reads are executed in separate phases. Multiple read requests
// may run concurrently, but writes must not overlap with Lock/Lookup/Unlock.
enum class MetaManagerLockMode {
  kStableReadAllowAppend = 0,
};

// PreparePut returns a write transaction handle that must later be committed
// or aborted.
struct PutContext {
  std::uint64_t id{0};
};

// LockKeys returns a read-snapshot handle that must later be used for lookup
// and unlock.
struct LockContext {
  std::uint64_t id{0};
};

// PreparePut / BatchLookup share one batched return layout. All vectors use
// the same index to describe the same key record.
struct AccessDesc {
  std::vector<const KeyId*> keys;
  std::vector<Address> addresses;
  std::vector<std::uint64_t> lengths;
  std::vector<std::uint8_t> valid;
};

using PutAccessDesc = AccessDesc;
using FlatAccessDesc = AccessDesc;

class MetaManager {
 public:
  MetaManager() = default;

  Status RegisterRemoteRegion(const RemoteRegionDesc& region);
  Status PreparePut(const std::vector<KeyId>& flat_keys,
                    const std::vector<std::uint64_t>& lengths,
                    PutContext* put_context,
                    PutAccessDesc* put_desc);
  Status CommitPut(const PutContext& put_context);
  Status AbortPut(const PutContext& put_context);
  Status LockKeys(const std::vector<KeyId>& flat_keys,
                  MetaManagerLockMode lock_mode,
                  LockContext* lock_context);
  Status BatchLookup(const std::vector<KeyId>& flat_keys,
                     const LockContext& lock_context,
                     FlatAccessDesc* access_desc);
  Status UnlockKeys(const LockContext& lock_context);
  Status BatchFree(const std::vector<KeyId>& flat_keys);
  void SetTimingLogEnabled(bool enabled);
  void SetPerformanceModeEnabled(bool enabled);

 private:
  enum class RegionState {
    kUninitialized = 0,
    kInitializing,
    kReady,
  };

  enum class SlotClass {
    kShort = 0,
    kLong,
  };

  // Internally a slot is tracked by class and index. The UBA is derived only
  // when an external address must be returned.
  struct SlotRef {
    SlotClass slot_class{SlotClass::kShort};
    std::size_t slot_index{0};
  };

  struct AddressRecord {
    SlotRef slot;
    std::uint64_t valid_length{0};
    std::atomic<std::uint32_t> pin_count{0};

    AddressRecord() = default;

    AddressRecord(SlotRef slot_value,
                  std::uint64_t valid_length_value,
                  std::uint32_t pin_count_value)
        : slot(slot_value),
          valid_length(valid_length_value),
          pin_count(pin_count_value) {}

    AddressRecord(const AddressRecord& other)
        : slot(other.slot),
          valid_length(other.valid_length),
          pin_count(other.pin_count.load(std::memory_order_relaxed)) {}

    AddressRecord& operator=(const AddressRecord& other) {
      if (this == &other) {
        return *this;
      }
      slot = other.slot;
      valid_length = other.valid_length;
      pin_count.store(other.pin_count.load(std::memory_order_relaxed),
                      std::memory_order_relaxed);
      return *this;
    }
  };

  struct KeyEntry {
    AddressRecord active_record;
    bool has_active_record{false};
    bool is_pending{false};
  };

  struct PendingPutItem {
    KeyEntry* key_entry{nullptr};
    const KeyId* key{nullptr};
    SlotRef slot;
    std::uint64_t valid_length{0};
  };

  struct PendingPutContext {
    std::vector<PendingPutItem> items;
  };

  struct LockedRecordRef {
    const KeyId* key{nullptr};
    AddressRecord* record{nullptr};
  };

  struct ActiveLockContext {
    std::vector<LockedRecordRef> items;
  };

  Address SlotAddress(const SlotRef& slot) const;
  bool TryResolveSlotClass(std::uint64_t length, SlotClass* slot_class) const;
  void ResizeAccessDesc(std::size_t count, AccessDesc* access_desc) const;
  void FillAccessDescAt(const KeyId* key,
                        Address address,
                        std::uint64_t length,
                        std::size_t index,
                        AccessDesc* access_desc) const;
  std::uint32_t LoadPinCount(const AddressRecord& record) const;
  void IncrementPinCount(AddressRecord* record);
  void DecrementPinCount(AddressRecord* record);
  SlotRef AcquireSlot(SlotClass slot_class);
  void ReleaseSlot(const SlotRef& slot);

  std::atomic<RegionState> region_state_{RegionState::kUninitialized};
  RemoteRegionDesc region_;

  // Write operations are serialized here. The caller must also keep this
  // phase separate from the read phase, so read methods do not lock entries_.
  std::mutex write_mutex_;
  std::queue<std::size_t> short_free_slots_;
  std::queue<std::size_t> long_free_slots_;
  std::unordered_map<KeyId, KeyEntry> entries_;

  std::unordered_map<std::uint64_t, PendingPutContext> pending_puts_;
  std::uint64_t next_put_context_id_{1};

  mutable std::shared_mutex lock_context_mutex_;
  std::unordered_map<std::uint64_t, ActiveLockContext> lock_contexts_;
  std::atomic<std::uint64_t> next_lock_context_id_{1};

  std::atomic<bool> timing_log_enabled_{false};
  std::atomic<bool> performance_mode_enabled_{false};
};

std::unique_ptr<MetaManager> CreateInMemoryMetaManager();

// Flushes aggregated method/phase timing collected while timing logging is
// enabled. Output is written to stderr and the collected samples are cleared.
void FlushMetaManagerTimingSummary();

}  // namespace kvaf
