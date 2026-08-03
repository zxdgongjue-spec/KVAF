#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "kvaf/status.h"
#include "kvaf/types.h"

namespace kvaf {

// 当前 POC 只支持一种稳定读语义：锁定后允许其它新 key 继续追加，
// 但不允许把本次锁定的 key 回收掉。
enum class MetaManagerLockMode {
  kStableReadAllowAppend = 0,
};

// 单个远端 region 的配置。
// 当前实现把 region 拆成两段：前半段放短 slot，后半段放长 slot。
struct RemoteRegionDesc {
  Address base{0};
  std::uint64_t length{0};
  std::uint64_t short_slot_size{0};
  std::uint64_t long_slot_size{0};
  std::uint64_t slot_count_per_class{0};
  DeviceKind device{DeviceKind::kCpu};
};

// PreparePut 返回的事务句柄。调用方写完数据后需要带着它继续 Commit/Abort。
struct PutContext {
  std::uint64_t id{0};
};

// LockKeys 返回的快照句柄。调用方用它把 lock / get / unlock 串起来。
struct LockContext {
  std::uint64_t id{0};
};

// 单个 key 本次被分配到的写入地址。
struct PutAllocation {
  KeyId key{};
  Address address{0};
  std::uint64_t reserved_length{0};
};

// PreparePut 返回的一批写入地址描述。
struct PutAccessDesc {
  std::vector<PutAllocation> entries;
};

// BatchLookup 返回的一批读地址描述，三个数组按相同下标一一对应。
struct FlatAccessDesc {
  std::vector<Address> addresses;
  std::vector<std::uint64_t> lengths;
  std::vector<std::uint8_t> valid;
};

// MetaManager 直接持有 POC 所需的全部状态：
// 1. slot 分配/回收
// 2. key 到地址记录的映射
// 3. pending put 和活动 lock 的上下文
class MetaManager {
 public:
  MetaManager() = default;

  // 注册唯一的远端 region，并初始化短/长 slot 的空闲队列。
  Status RegisterRemoteRegion(const RemoteRegionDesc& region);
  // 为一批 key 预留地址；真正可见前还需要 CommitPut。
  Status PreparePut(const std::vector<KeyId>& flat_keys,
                    const std::vector<std::uint64_t>& lengths,
                    PutContext* put_context,
                    PutAccessDesc* put_desc);
  // 把 PreparePut 产生的预留结果提交为正式映射。
  Status CommitPut(const PutContext& put_context);
  // 放弃一次未提交的 put，并把预留 slot 归还。
  Status AbortPut(const PutContext& put_context);
  // 固定一批 key，阻止它们在读期间被回收。
  Status LockKeys(const std::vector<KeyId>& flat_keys,
                  MetaManagerLockMode lock_mode,
                  LockContext* lock_context);
  // 读取 LockKeys 固定下来的地址与长度信息。
  Status BatchLookup(const std::vector<KeyId>& flat_keys,
                     const LockContext& lock_context,
                     FlatAccessDesc* access_desc);
  // 释放一次稳定读快照。
  Status UnlockKeys(const LockContext& lock_context);
  // 回收一批 key 当前占用的 slot。
  Status BatchFree(const std::vector<KeyId>& flat_keys);
  // 控制是否输出每个接口的耗时日志；默认关闭。
  void SetTimingLogEnabled(bool enabled);

 private:
  // slot 按长度分为两类；PreparePut 会根据 payload 长度选择其一。
  enum class SlotClass {
    kShort = 0,
    kLong,
  };

  // 内部只保存 slot 类型和 slot 下标，需要对外返回地址时再做换算。
  struct SlotRef {
    SlotClass slot_class{SlotClass::kShort};
    std::size_t slot_index{0};
  };

  // 一条已提交的地址记录。
  struct AddressRecord {
    SlotRef slot;
    std::uint64_t valid_length{0};
    std::uint32_t pin_count{0};
  };

  // 当前 POC 中，一个 key 只保留一条 active_record。
  struct KeyEntry {
    AddressRecord active_record;
  };

  // PreparePut 期间暂存的预留项；Commit 前不可见。
  struct PendingPutItem {
    KeyId key{};
    SlotRef slot;
    std::uint64_t valid_length{0};
  };

  // 一次 PreparePut 对应的上下文。
  struct PendingPutContext {
    std::vector<PendingPutItem> items;
  };

  // 一次 LockKeys 固定下来的 key 列表。
  struct ActiveLockContext {
    std::vector<KeyId> keys;
  };

  // 按 slot 类型和下标换算出真实远端地址。
  Address SlotAddress(const SlotRef& slot) const;
  // 根据 payload 长度判断应该落到短 slot 还是长 slot。
  bool TryResolveSlotClass(std::uint64_t length, SlotClass* slot_class) const;
  // 从对应空闲队列取出一个 slot。
  SlotRef AcquireSlot(SlotClass slot_class);
  // 把 slot 归还到对应空闲队列。
  void ReleaseSlot(const SlotRef& slot);

  // 当前实现用一把总锁保护全部元数据状态。
  std::mutex meta_mutex_;
  bool region_registered_{false};
  RemoteRegionDesc region_;
  // 两类 slot 各自维护独立的 FIFO 空闲队列。
  std::queue<std::size_t> short_free_slots_;
  std::queue<std::size_t> long_free_slots_;
  // 已提交 key、pending put、pending key 和活动读快照。
  std::unordered_map<KeyId, KeyEntry> key_index_;
  std::unordered_map<std::uint64_t, PendingPutContext> pending_puts_;
  std::unordered_set<KeyId> pending_keys_;
  std::unordered_map<std::uint64_t, ActiveLockContext> lock_contexts_;
  // 递增上下文 id，用于标识一次 put 或一次 lock。
  std::uint64_t next_put_context_id_{1};
  std::uint64_t next_lock_context_id_{1};
  // 计时日志默认关闭，避免影响正常路径。
  std::atomic<bool> timing_log_enabled_{false};
};

// 创建一个基于内存数据结构的 MetaManager POC。
std::unique_ptr<MetaManager> CreateInMemoryMetaManager();

}  // namespace kvaf
