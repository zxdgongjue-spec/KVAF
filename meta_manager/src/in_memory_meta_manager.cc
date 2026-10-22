#include "meta_manager.h"

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

// DSA keys are bounded slot IDs. Visibility belongs to each fixed slot.
enum class SlotState : std::uint8_t { kEmpty, kPending, kActive };

// 单条 key 当前绑定的地址记录。
struct AddressRecord {
  std::uint64_t record_id{0};
  Address uba{0};
  std::uint64_t valid_length{0};
  std::size_t slot_id{0};
  std::uint32_t pin_count{0};
  SlotState state{SlotState::kEmpty};
};

// PreparePut 阶段的临时预留项。
struct PendingPutItem {
  KeyId key{0};
  std::size_t slot_id{0};
  std::uint64_t valid_length{0};
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

class InMemoryMetaManager final : public MetaManager {
 public:
  // Register the complete DSA key space: key_id == slot_id.
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
      // BatchPut normally has one in-flight context. Reserve its bucket so
      // steady-state emplace does not trigger a bucket-table rehash.
      pending_puts_.reserve(1U);
    } catch (const std::bad_alloc&) {
      return Status::OutOfMemory("failed to allocate DSA slot metadata");
    }

    region_ = region;
    region_registered_ = true;
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
    // Validate the whole batch before reserving any fixed slot.
    for (std::size_t i = 0; i < flat_keys.size(); ++i) {
      if (flat_keys[i] >= records_.size()) {
        return InvalidArgument(
            "PreparePut key must be a DSA slot ID inside the region");
      }
      if (lengths[i] == 0) {
        return InvalidArgument("PreparePut length must be greater than zero");
      }
      if (lengths[i] > region_.slot_size) {
        return InvalidArgument("PreparePut length exceeds slot_size");
      }
      if (records_[flat_keys[i]].state != SlotState::kEmpty) {
        return AlreadyExists("PreparePut key already exists or is pending");
      }
    }

    const std::uint64_t ctx_id = next_put_context_id_++;
    PendingPutContext pending_context;
    pending_context.items.reserve(flat_keys.size());
    put_desc->entries.clear();
    put_desc->entries.reserve(flat_keys.size());
    for (std::size_t i = 0; i < flat_keys.size(); ++i) {
      const std::size_t slot_id = static_cast<std::size_t>(flat_keys[i]);
      pending_context.items.push_back(PendingPutItem{flat_keys[i], slot_id, lengths[i]});
      put_desc->entries.push_back(
          PutAllocation{flat_keys[i], SlotAddress(slot_id), lengths[i]});
    }
    // Finish all allocations before changing slot visibility.
    auto pending_it =
        pending_puts_.emplace(ctx_id, std::move(pending_context)).first;
    for (const PendingPutItem& item : pending_it->second.items) {
      records_[item.slot_id].state = SlotState::kPending;
    }
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

    for (const PendingPutItem& item : pending_it->second.items) {
      if (records_[item.slot_id].state != SlotState::kPending) {
        return InvalidState("CommitPut lost a pending DSA slot");
      }
    }
    // The mutex makes publication of the whole batch atomic to readers.
    for (const PendingPutItem& item : pending_it->second.items) {
      AddressRecord& record = records_[item.slot_id];
      record.record_id = next_record_id_++;
      record.uba = SlotAddress(item.slot_id);
      record.valid_length = item.valid_length;
      record.slot_id = item.slot_id;
      record.pin_count = 0;
      record.state = SlotState::kActive;
    }
    pending_puts_.erase(pending_it);
    return Status::OK();
  }

  // 放弃一次未提交的 put，使该 Key 的固定 slot 恢复 EMPTY。
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

    for (const PendingPutItem& item : pending_it->second.items) {
      records_[item.slot_id] = AddressRecord{};
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
      if (key >= records_.size() || records_[key].state != SlotState::kActive) {
        // Pins are taken while building the snapshot so this path needs to
        // roll them back if a later key is missing.
        for (const LockedItem& item : items) {
          --item.record->pin_count;
        }
        return NotFound("LockKeys key was not found");
      }

      // records_ is allocated once during region registration and never
      // resized, so this pointer remains stable until the manager is destroyed.
      AddressRecord* record = &records_[key];
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

    for (KeyId key : flat_keys) {
      if (key >= records_.size() || records_[key].state != SlotState::kActive) {
        return NotFound("BatchFree key was not found");
      }
      const AddressRecord& record = records_[key];
      if (record.pin_count > 0) {
        return Busy("BatchFree key is still pinned by a LockContext");
      }
    }

    for (KeyId key : flat_keys) {
      records_[key] = AddressRecord{};
    }

    return Status::OK();
  }

 private:
  Address SlotAddress(std::size_t slot_id) const {
    return region_.base + static_cast<Address>(slot_id) * region_.slot_size;
  }

  // One mutex protects fixed-slot states, transactions and read snapshots.
  std::mutex meta_mutex_;
  bool region_registered_{false};
  RemoteRegionDesc region_;
  std::vector<AddressRecord> records_;
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
