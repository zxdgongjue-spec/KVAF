#pragma once

#include <array>
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

// 当前 POC 只支持一种稳定读语义：
// 锁住一批 key 后，允许其他新 key 继续写入，
// 但不允许这批 key 在解锁前被回收。
enum class MetaManagerLockMode {
  kStableReadAllowAppend = 0,
};

// PreparePut 返回的上下文，后续需要用它继续 Commit 或 Abort。
struct PutContext {
  std::uint64_t id{0};
};

// LockKeys 返回的上下文，后续需要用它继续 BatchLookup 或 Unlock。
struct LockContext {
  std::uint64_t id{0};
};

// 一条对外返回的访问结果。
// key 是指向内部稳定存储字符串的指针，避免热路径重复复制字符串。
struct AccessEntry {
  const KeyId* key{nullptr};
  Address address{0};
  std::uint64_t length{0};
  std::uint8_t valid{0};
};

// PreparePut / BatchLookup 统一使用这类批量返回结构。
struct AccessDesc {
  std::vector<AccessEntry> entries;
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
  static constexpr std::size_t kKeyShardCount = 64;
  static constexpr std::size_t kLockContextShardCount = 64;

  enum class RegionState {
    kUninitialized = 0,
    kInitializing,
    kReady,
  };

  // slot 根据长度分成两类。
  enum class SlotClass {
    kShort = 0,
    kLong,
  };

  // 内部只保存 slot 类型和槽位下标，需要返回真实地址时再换算。
  struct SlotRef {
    SlotClass slot_class{SlotClass::kShort};
    std::size_t slot_index{0};
  };

  // 一条已提交地址记录。
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

  // 一个 key 的完整生命周期状态。
  // 这样热路径只需要在所属 shard 内查一次表，
  // 不必再拆成 active/pending 多张表反复查找。
  struct KeyEntry {
    AddressRecord active_record;
    bool has_active_record{false};
    bool is_pending{false};
  };

  // 一个 key shard：自己的锁 + 自己那部分 key 元数据。
  struct KeyShard {
    mutable std::shared_mutex mutex;
    std::unordered_map<KeyId, KeyEntry> entries;
  };

  // PreparePut 暂存但尚未 Commit 的预留项。
  struct PendingPutItem {
    std::size_t shard_id{0};
    KeyEntry* key_entry{nullptr};
    const KeyId* key{nullptr};
    SlotRef slot;
    std::uint64_t valid_length{0};
  };

  struct PendingPutContext {
    std::vector<PendingPutItem> items;
  };

  // LockKeys 固定下来的记录指针，后续 lookup/unlock 不再重复查表。
  struct LockedRecordRef {
    const KeyId* key{nullptr};
    AddressRecord* record{nullptr};
  };

  struct ActiveLockContext {
    mutable std::shared_mutex lifecycle_mutex;
    std::vector<LockedRecordRef> items;
  };

  struct LockContextShard {
    std::mutex mutex;
    std::unordered_map<std::uint64_t, ActiveLockContext> items;
  };

  Address SlotAddress(const SlotRef& slot) const;
  bool TryResolveSlotClass(std::uint64_t length, SlotClass* slot_class) const;
  void FillAccessEntry(const KeyId* key,
                       Address address,
                       std::uint64_t length,
                       AccessEntry* entry) const;
  std::uint32_t LoadPinCount(const AddressRecord& record) const;
  void IncrementPinCount(AddressRecord* record);
  void DecrementPinCount(AddressRecord* record);
  SlotRef AcquireSlot(SlotClass slot_class);
  void ReleaseSlot(const SlotRef& slot);
  std::size_t ShardIndexForKey(const KeyId& key) const;
  std::size_t LockContextShardIndex(std::uint64_t context_id) const;
  void CollectShardPlan(const std::vector<KeyId>& flat_keys,
                        std::vector<std::size_t>* key_shard_ids,
                        std::vector<std::size_t>* shard_order) const;

  // region 只允许注册一次；注册后 region_ 只读，因此只需要注册锁。
  std::atomic<RegionState> region_state_{RegionState::kUninitialized};
  RemoteRegionDesc region_;

  // 两类空闲队列各自独立加锁，不再被全局锁串行化。
  std::mutex free_queue_mutex_;
  std::queue<std::size_t> short_free_slots_;
  std::queue<std::size_t> long_free_slots_;

  // key 按 shard 切分；不同 shard 之间可以并行操作。
  std::array<KeyShard, kKeyShardCount> key_shards_;

  // put / lock 上下文统一由 context 锁保护。
  std::mutex context_mutex_;
  std::unordered_map<std::uint64_t, PendingPutContext> pending_puts_;
  std::uint64_t next_put_context_id_{1};
  std::array<LockContextShard, kLockContextShardCount> lock_context_shards_;
  std::atomic<std::uint64_t> next_lock_context_id_{1};

  std::atomic<bool> timing_log_enabled_{false};
  std::atomic<bool> performance_mode_enabled_{false};
};

std::unique_ptr<MetaManager> CreateInMemoryMetaManager();

}  // namespace kvaf
