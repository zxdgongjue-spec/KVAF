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

// 写入和读取处于不同阶段。多个读请求可以并发执行，但写入不能与
// Lock/Lookup/Unlock 阶段重叠。
enum class MetaManagerLockMode {
  kStableReadAllowAppend = 0,
};

// PreparePut 返回一个写事务句柄，后续必须提交或中止该事务。
struct PutContext {
  std::uint64_t id{0};
};

// LockKeys 返回全局读快照句柄。它不会扫描或固定单个 key，后续需要使用该句柄
// 执行查询和解锁。
struct LockContext {
  std::uint64_t id{0};
};

// PreparePut / BatchLookup 共用批量返回布局。所有 vector 使用相同下标描述同一条
// key 记录，keys 中的指针借用对应输入批次中的字符串。
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
  MetaManager();
  ~MetaManager();

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
  // BatchLookup 在 lock_context 持有全局读快照期间，执行 KeyId 到地址的实际查询。
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

  // 内部只记录 slot 类别和索引，只有需要返回外部地址时才计算 UBA。
  struct SlotRef {
    SlotClass slot_class{SlotClass::kShort};
    std::size_t slot_index{0};
  };

  enum class RecordState : std::uint8_t {
    kUnused = 0,
    kPending,
    kActive,
    kFreed,
  };

  struct PendingPutItem {
    const KeyId* key{nullptr};
    std::size_t hash{0};
    SlotRef slot;
    std::uint64_t valid_length{0};
  };

  struct PendingPutContext {
    std::vector<PendingPutItem> items;
  };

  struct ActiveLockContext {
    // 在 UnlockKeys 前保持完整的 key 到地址表稳定。
    std::shared_lock<std::shared_mutex> read_phase_lock;
  };

  Address SlotAddress(const SlotRef& slot) const;
  bool TryResolveSlotClass(std::uint64_t length, SlotClass* slot_class) const;
  void ResizeAccessDesc(std::size_t count, AccessDesc* access_desc) const;
  void FillAccessDescAt(const KeyId* key,
                        Address address,
                        std::uint64_t length,
                        std::size_t index,
                        AccessDesc* access_desc) const;
  SlotRef AcquireSlot(SlotClass slot_class);
  void ReleaseSlot(const SlotRef& slot);

  std::atomic<RegionState> region_state_{RegionState::kUninitialized};
  RemoteRegionDesc region_;

  // 写操作在这里串行化。调用方必须将 put 操作与读阶段分离，因此读方法不锁定
  // entries_。
  std::mutex write_mutex_;
  std::queue<std::size_t> short_free_slots_;
  std::queue<std::size_t> long_free_slots_;

  // EntryIndex 将地址元数据直接存储在开放寻址桶中。KeyId 单独追加保存，是因为
  // 桶重哈希时仍需要保持字符串地址稳定。
  struct EntryIndex;
  std::unique_ptr<EntryIndex> entries_;

  // 大批量 BatchLookup 复用少量 worker，避免每次请求都创建和回收线程。
  class LookupWorkerPool;
  std::unique_ptr<LookupWorkerPool> lookup_workers_;

  std::unordered_map<std::uint64_t, PendingPutContext> pending_puts_;
  std::uint64_t next_put_context_id_{1};

  // LockKeys 在读快照整个生命周期内持有共享租约。BatchFree 清理记录和回收 slot
  // 前需要获取排他侧。
  mutable std::shared_mutex read_phase_mutex_;
  mutable std::shared_mutex lock_context_mutex_;
  std::unordered_map<std::uint64_t, ActiveLockContext> lock_contexts_;
  std::atomic<std::uint64_t> next_lock_context_id_{1};

  std::atomic<bool> timing_log_enabled_{false};
  std::atomic<bool> performance_mode_enabled_{false};
};

std::unique_ptr<MetaManager> CreateInMemoryMetaManager();

// 输出计时日志开启期间收集的方法/阶段汇总，并清空已收集的样本。
void FlushMetaManagerTimingSummary();

}  // namespace kvaf
