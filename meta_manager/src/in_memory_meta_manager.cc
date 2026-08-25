#include "kvaf/meta_manager/meta_manager.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
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
  std::uint32_t layer_id{0};
  std::size_t slot_id{0};
  std::uint32_t pin_count{0};
};

// A caller key is meaningful only within one transformer's layer partition.
// Keeping the pair in the metadata index permits every layer to use the same
// compact local key range (for example pool_row * qcap + block_q).
struct LayerKey {
  std::uint32_t layer_id{0};
  KeyId key{0};

  bool operator==(const LayerKey& other) const {
    return layer_id == other.layer_id && key == other.key;
  }
};

struct LayerKeyHash {
  std::size_t operator()(const LayerKey& key) const {
    const std::size_t first = std::hash<std::uint32_t>{}(key.layer_id);
    const std::size_t second = std::hash<KeyId>{}(key.key);
    return first ^ (second + 0x9e3779b9U + (first << 6U) + (first >> 2U));
  }
};

// key 到当前可见记录的索引项。
struct KeyEntry {
  AddressRecord active_record;
};

// PreparePut 阶段的临时预留项。
struct PendingPutItem {
  LayerKey layer_key;
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

    if (region.layer_count == 0) {
      return InvalidArgument(
          "remote region layer_count must be greater than zero");
    }
    const std::uint64_t slot_count = region.length / region.slot_size;
    if (slot_count % region.layer_count != 0) {
      return InvalidArgument(
          "remote region slot count must divide evenly across layer_count");
    }
    if (region.slot_allocation_order != SlotAllocationOrder::kFifo &&
        region.slot_allocation_order != SlotAllocationOrder::kShuffled) {
      return InvalidArgument("remote region slot_allocation_order is invalid");
    }

    const std::uint64_t slots_per_layer = slot_count / region.layer_count;
    std::vector<std::queue<std::size_t>> free_slots_by_layer(
        region.layer_count);
    for (std::uint32_t layer_id = 0; layer_id < region.layer_count;
         ++layer_id) {
      auto& free_slots = free_slots_by_layer[layer_id];
      std::vector<std::size_t> slot_order(
          static_cast<std::size_t>(slots_per_layer));
      std::iota(slot_order.begin(), slot_order.end(), std::size_t{0});
      if (region.slot_allocation_order == SlotAllocationOrder::kShuffled) {
        // Give each layer a distinct, deterministic permutation while keeping
        // the seed supplied by Python sufficient to reproduce the layout.
        std::mt19937_64 generator(
            region.slot_shuffle_seed +
            0x9e3779b97f4a7c15ULL * static_cast<std::uint64_t>(layer_id));
        std::shuffle(slot_order.begin(), slot_order.end(), generator);
      }
      for (std::size_t slot_id : slot_order) {
        free_slots.push(slot_id);
      }
    }

    region_ = region;
    slots_per_layer_ = slots_per_layer;
    free_slots_by_layer_ = std::move(free_slots_by_layer);
    region_registered_ = true;

    return Status::OK();
  }

  // 为一批新 key 预留 slot，并返回写路径需要的地址描述。
  Status PreparePut(const std::vector<KeyId>& flat_keys,
                    const std::vector<std::uint64_t>& lengths,
                    PutContext* put_context,
                    PutAccessDesc* put_desc,
                    std::uint32_t layer_id) override {
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
    Status status = ValidateLayerId(layer_id);
    if (!status.ok()) {
      return status;
    }
    auto& free_slots = free_slots_by_layer_[layer_id];
    if (free_slots.size() < flat_keys.size()) {
      return Status::Error(StatusCode::kOutOfMemory,
                           "not enough free slots in requested layer for "
                           "PreparePut");
    }

    for (std::size_t i = 0; i < flat_keys.size(); ++i) {
      if (lengths[i] == 0) {
        return InvalidArgument("PreparePut length must be greater than zero");
      }
      if (lengths[i] > region_.slot_size) {
        return InvalidArgument("PreparePut length exceeds slot_size");
      }
      const LayerKey layer_key{layer_id, flat_keys[i]};
      if (key_index_.find(layer_key) != key_index_.end() ||
          pending_keys_.find(layer_key) != pending_keys_.end()) {
        return AlreadyExists("PreparePut key already exists or is pending");
      }
    }

    const std::uint64_t ctx_id = next_put_context_id_++;
    PendingPutContext pending_context;
    pending_context.items.reserve(flat_keys.size());
    put_desc->entries.clear();
    put_desc->entries.reserve(flat_keys.size());

    // slot 在这里离开 free queue，但在 CommitPut 前仍然不可见。
    for (std::size_t i = 0; i < flat_keys.size(); ++i) {
      const std::size_t slot_id = free_slots.front();
      free_slots.pop();

      const LayerKey layer_key{layer_id, flat_keys[i]};
      const Address address = SlotAddress(layer_id, slot_id);
      pending_context.items.push_back(
          PendingPutItem{layer_key, slot_id, lengths[i]});
      put_desc->entries.push_back(
          PutAllocation{flat_keys[i], address, lengths[i]});
      pending_keys_.insert(layer_key);
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

    for (const PendingPutItem& item : pending_it->second.items) {
      if (key_index_.find(item.layer_key) != key_index_.end()) {
        return InvalidState("CommitPut found an already committed key");
      }
    }

    // Commit 是写路径对读路径的可见性边界。
    for (const PendingPutItem& item : pending_it->second.items) {
      AddressRecord record;
      record.record_id = next_record_id_++;
      record.uba = SlotAddress(item.layer_key.layer_id, item.slot_id);
      record.valid_length = item.valid_length;
      record.layer_id = item.layer_key.layer_id;
      record.slot_id = item.slot_id;
      record.pin_count = 0;

      key_index_.emplace(item.layer_key, KeyEntry{record});
      pending_keys_.erase(item.layer_key);
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

    for (const PendingPutItem& item : pending_it->second.items) {
      free_slots_by_layer_[item.layer_key.layer_id].push(item.slot_id);
      pending_keys_.erase(item.layer_key);
    }

    pending_puts_.erase(pending_it);
    return Status::OK();
  }

  // 为一批 key 建立稳定读快照，并通过 pin_count 阻止并发 free。
  Status LockKeys(const std::vector<KeyId>& flat_keys,
                  MetaManagerLockMode lock_mode,
                  LockContext* lock_context,
                  std::uint32_t layer_id) override {
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
    Status status = ValidateLayerId(layer_id);
    if (!status.ok()) {
      return status;
    }

    std::vector<LockedItem> items;
    items.reserve(flat_keys.size());
    for (KeyId key : flat_keys) {
      auto key_it = key_index_.find(LayerKey{layer_id, key});
      if (key_it == key_index_.end()) {
        // Pins are taken while building the snapshot so this path needs to
        // roll them back if a later key is missing.
        for (const LockedItem& item : items) {
          --item.record->pin_count;
        }
        return NotFound("LockKeys key was not found");
      }

      // unordered_map node addresses remain stable across rehash. The pin
      // prevents BatchFree from erasing this record until UnlockKeys, so this
      // pointer is a valid stable-read snapshot and avoids two more hash
      // lookups per key in BatchLookup/UnlockKeys.
      AddressRecord* record = &key_it->second.active_record;
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
  Status BatchFree(const std::vector<KeyId>& flat_keys,
                   std::uint32_t layer_id) override {
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
    Status status = ValidateLayerId(layer_id);
    if (!status.ok()) {
      return status;
    }

    std::vector<std::size_t> slots_to_recycle;
    slots_to_recycle.reserve(flat_keys.size());
    for (KeyId key : flat_keys) {
      auto key_it = key_index_.find(LayerKey{layer_id, key});
      if (key_it == key_index_.end()) {
        return NotFound("BatchFree key was not found");
      }

      const AddressRecord& record = key_it->second.active_record;
      if (record.pin_count > 0) {
        return Busy("BatchFree key is still pinned by a LockContext");
      }

      slots_to_recycle.push_back(record.slot_id);
    }

    for (KeyId key : flat_keys) {
      key_index_.erase(LayerKey{layer_id, key});
    }

    for (std::size_t slot_id : slots_to_recycle) {
      free_slots_by_layer_[layer_id].push(slot_id);
    }

    return Status::OK();
  }

 private:
  Status ValidateLayerId(std::uint32_t layer_id) const {
    if (layer_id >= region_.layer_count) {
      return InvalidArgument("layer_id is outside registered remote region");
    }
    return Status::OK();
  }

  Address SlotAddress(std::uint32_t layer_id, std::size_t slot_id) const {
    const std::uint64_t absolute_slot =
        static_cast<std::uint64_t>(layer_id) * slots_per_layer_ + slot_id;
    return region_.base +
           static_cast<Address>(absolute_slot) * region_.slot_size;
  }

  // 当前 POC 使用一把总锁保护元数据和空闲 slot 队列。
  std::mutex meta_mutex_;
  bool region_registered_{false};
  RemoteRegionDesc region_;
  std::uint64_t slots_per_layer_{0};
  std::vector<std::queue<std::size_t>> free_slots_by_layer_;
  std::unordered_map<LayerKey, KeyEntry, LayerKeyHash> key_index_;
  std::unordered_map<std::uint64_t, PendingPutContext> pending_puts_;
  std::unordered_set<LayerKey, LayerKeyHash> pending_keys_;
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
