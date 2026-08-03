#include "kvaf/meta_manager/meta_manager.h"

#include <chrono>
#include <iostream>
#include <utility>

namespace kvaf {

namespace {

// 统一内部错误标签，避免每个分支都拼接一套长字符串。
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

// 把内部错误标签转换成稳定的外部错误消息。
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

// 批量接口要求 key 唯一，避免重复申请、重复锁定或重复回收。
bool HasDistinctKeys(const std::vector<KeyId>& keys) {
  std::unordered_set<KeyId> seen;
  seen.reserve(keys.size());
  for (const KeyId& key : keys) {
    if (!seen.insert(key).second) {
      return false;
    }
  }
  return true;
}

// 作用域计时器：函数退出时按需打印耗时，避免每个 return 分支都手动写日志。
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

    const auto end = std::chrono::steady_clock::now();
    const auto elapsed_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_)
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
  std::lock_guard<std::mutex> lock(meta_mutex_);
  if (region_registered_) {
    return Error(StatusCode::kAlreadyExists, ErrorTag::kRegionExists);
  }

  // 当前 POC 约束：
  // 1. 必须同时提供短/长两类 slot
  // 2. 长 slot 必须严格大于短 slot
  // 3. 两类 slot 的数量必须相同
  if (region.short_slot_size == 0 || region.long_slot_size == 0 ||
      region.slot_count_per_class == 0 ||
      region.short_slot_size >= region.long_slot_size) {
    return Error(StatusCode::kInvalidArgument, ErrorTag::kInvalidRegion);
  }

  // region 的总长度必须正好等于“短区 + 长区”的总和。
  const std::uint64_t total_slot_size =
      region.short_slot_size + region.long_slot_size;
  if (region.length != region.slot_count_per_class * total_slot_size) {
    return Error(StatusCode::kInvalidArgument, ErrorTag::kInvalidRegion);
  }

  region_ = region;
  region_registered_ = true;
  // 初始化两类空闲队列；下标相同不代表同一地址，只是各自队列里的第几个 slot。
  for (std::uint64_t i = 0; i < region.slot_count_per_class; ++i) {
    short_free_slots_.push(static_cast<std::size_t>(i));
    long_free_slots_.push(static_cast<std::size_t>(i));
  }

  return Status::OK();
}

Status MetaManager::PreparePut(const std::vector<KeyId>& flat_keys,
                               const std::vector<std::uint64_t>& lengths,
                               PutContext* put_context,
                               PutAccessDesc* put_desc) {
  ScopedMethodTimer timer(&timing_log_enabled_, "PreparePut");
  if (put_context == nullptr || put_desc == nullptr || flat_keys.empty() ||
      flat_keys.size() != lengths.size() || !HasDistinctKeys(flat_keys)) {
    return Error(StatusCode::kInvalidArgument, ErrorTag::kInvalidRequest);
  }

  std::lock_guard<std::mutex> lock(meta_mutex_);
  if (!region_registered_) {
    return Error(StatusCode::kNotInitialized, ErrorTag::kRegionMissing);
  }

  // 先统计本次需要多少短 slot、多少长 slot，避免取了一半才发现队列不够。
  std::vector<SlotClass> slot_classes;
  slot_classes.reserve(flat_keys.size());

  std::size_t short_needed = 0;
  std::size_t long_needed = 0;
  for (std::size_t i = 0; i < flat_keys.size(); ++i) {
    SlotClass slot_class;
    if (!TryResolveSlotClass(lengths[i], &slot_class)) {
      return Error(StatusCode::kInvalidArgument, ErrorTag::kInvalidRequest);
    }
    if (key_index_.find(flat_keys[i]) != key_index_.end() ||
        pending_keys_.find(flat_keys[i]) != pending_keys_.end()) {
      return Error(StatusCode::kAlreadyExists, ErrorTag::kKeyExists);
    }

    slot_classes.push_back(slot_class);
    if (slot_class == SlotClass::kShort) {
      ++short_needed;
    } else {
      ++long_needed;
    }
  }

  if (short_free_slots_.size() < short_needed ||
      long_free_slots_.size() < long_needed) {
    return Error(StatusCode::kOutOfMemory, ErrorTag::kOutOfSlots);
  }

  const std::uint64_t ctx_id = next_put_context_id_++;
  PendingPutContext pending_context;
  pending_context.items.reserve(flat_keys.size());

  // put_desc 是“本次调用的输出缓冲区”，进入前先清空旧结果。
  put_desc->entries.clear();
  put_desc->entries.reserve(flat_keys.size());

  for (std::size_t i = 0; i < flat_keys.size(); ++i) {
    // 这里 slot 已经离开空闲队列，但在 CommitPut 之前仍然对读路径不可见。
    const SlotRef slot = AcquireSlot(slot_classes[i]);
    pending_context.items.push_back(
        PendingPutItem{flat_keys[i], slot, lengths[i]});
    put_desc->entries.push_back(
        PutAllocation{flat_keys[i], SlotAddress(slot), lengths[i]});
    pending_keys_.insert(flat_keys[i]);
  }

  pending_puts_.emplace(ctx_id, std::move(pending_context));
  put_context->id = ctx_id;
  return Status::OK();
}

Status MetaManager::CommitPut(const PutContext& put_context) {
  ScopedMethodTimer timer(&timing_log_enabled_, "CommitPut");
  std::lock_guard<std::mutex> lock(meta_mutex_);
  if (!region_registered_) {
    return Error(StatusCode::kNotInitialized, ErrorTag::kRegionMissing);
  }

  auto pending_it = pending_puts_.find(put_context.id);
  if (pending_it == pending_puts_.end()) {
    return Error(StatusCode::kNotFound, ErrorTag::kPutContextMissing);
  }

  // CommitPut 是写路径的可见性边界：提交后 key 才进入 key_index_。
  for (const PendingPutItem& item : pending_it->second.items) {
    key_index_.emplace(
        item.key, KeyEntry{AddressRecord{item.slot, item.valid_length, 0}});
    pending_keys_.erase(item.key);
  }

  pending_puts_.erase(pending_it);
  return Status::OK();
}

Status MetaManager::AbortPut(const PutContext& put_context) {
  ScopedMethodTimer timer(&timing_log_enabled_, "AbortPut");
  std::lock_guard<std::mutex> lock(meta_mutex_);
  if (!region_registered_) {
    return Error(StatusCode::kNotInitialized, ErrorTag::kRegionMissing);
  }

  auto pending_it = pending_puts_.find(put_context.id);
  if (pending_it == pending_puts_.end()) {
    return Error(StatusCode::kNotFound, ErrorTag::kPutContextMissing);
  }

  // AbortPut 回滚一次未提交的 put，把已预留的 slot 重新放回空闲队列。
  for (const PendingPutItem& item : pending_it->second.items) {
    ReleaseSlot(item.slot);
    pending_keys_.erase(item.key);
  }

  pending_puts_.erase(pending_it);
  return Status::OK();
}

Status MetaManager::LockKeys(const std::vector<KeyId>& flat_keys,
                             MetaManagerLockMode lock_mode,
                             LockContext* lock_context) {
  ScopedMethodTimer timer(&timing_log_enabled_, "LockKeys");
  if (lock_context == nullptr || flat_keys.empty() ||
      !HasDistinctKeys(flat_keys)) {
    return Error(StatusCode::kInvalidArgument, ErrorTag::kInvalidRequest);
  }
  if (lock_mode != MetaManagerLockMode::kStableReadAllowAppend) {
    return Error(StatusCode::kInvalidArgument, ErrorTag::kUnsupportedMode);
  }

  std::lock_guard<std::mutex> lock(meta_mutex_);
  if (!region_registered_) {
    return Error(StatusCode::kNotInitialized, ErrorTag::kRegionMissing);
  }

  for (const KeyId& key : flat_keys) {
    if (key_index_.find(key) == key_index_.end()) {
      return Error(StatusCode::kNotFound, ErrorTag::kKeyMissing);
    }
  }

  // pin_count 用来阻止并发 BatchFree；只要还有活跃 lock，就不允许回收。
  for (const KeyId& key : flat_keys) {
    ++key_index_.find(key)->second.active_record.pin_count;
  }

  const std::uint64_t ctx_id = next_lock_context_id_++;
  lock_contexts_.emplace(ctx_id, ActiveLockContext{flat_keys});
  lock_context->id = ctx_id;
  return Status::OK();
}

Status MetaManager::BatchLookup(const std::vector<KeyId>& flat_keys,
                                const LockContext& lock_context,
                                FlatAccessDesc* access_desc) {
  ScopedMethodTimer timer(&timing_log_enabled_, "BatchLookup");
  if (access_desc == nullptr) {
    return Error(StatusCode::kInvalidArgument, ErrorTag::kInvalidRequest);
  }

  std::lock_guard<std::mutex> lock(meta_mutex_);
  if (!region_registered_) {
    return Error(StatusCode::kNotInitialized, ErrorTag::kRegionMissing);
  }

  auto lock_it = lock_contexts_.find(lock_context.id);
  if (lock_it == lock_contexts_.end()) {
    return Error(StatusCode::kNotFound, ErrorTag::kLockContextMissing);
  }
  // BatchLookup 必须和 LockKeys 使用同一批 key，确保返回结果按同一顺序一一对应。
  if (lock_it->second.keys != flat_keys) {
    return Error(StatusCode::kInvalidArgument, ErrorTag::kInvalidRequest);
  }

  access_desc->addresses.clear();
  access_desc->lengths.clear();
  access_desc->valid.clear();
  access_desc->addresses.reserve(flat_keys.size());
  access_desc->lengths.reserve(flat_keys.size());
  access_desc->valid.reserve(flat_keys.size());

  for (const KeyId& key : flat_keys) {
    auto key_it = key_index_.find(key);
    if (key_it == key_index_.end()) {
      return Error(StatusCode::kInvalidState, ErrorTag::kStateConflict);
    }

    // 内部记录只保存 slot 引用；对外返回时再换算成真实地址。
    const AddressRecord& record = key_it->second.active_record;
    access_desc->addresses.push_back(SlotAddress(record.slot));
    access_desc->lengths.push_back(record.valid_length);
    access_desc->valid.push_back(1);
  }

  return Status::OK();
}

Status MetaManager::UnlockKeys(const LockContext& lock_context) {
  ScopedMethodTimer timer(&timing_log_enabled_, "UnlockKeys");
  std::lock_guard<std::mutex> lock(meta_mutex_);
  if (!region_registered_) {
    return Error(StatusCode::kNotInitialized, ErrorTag::kRegionMissing);
  }

  auto lock_it = lock_contexts_.find(lock_context.id);
  if (lock_it == lock_contexts_.end()) {
    return Error(StatusCode::kNotFound, ErrorTag::kLockContextMissing);
  }

  // 先整体检查，再统一减 pin_count，避免中途失败留下部分解锁的状态。
  for (const KeyId& key : lock_it->second.keys) {
    auto key_it = key_index_.find(key);
    if (key_it == key_index_.end() || key_it->second.active_record.pin_count == 0) {
      return Error(StatusCode::kInvalidState, ErrorTag::kStateConflict);
    }
  }

  for (const KeyId& key : lock_it->second.keys) {
    --key_index_.find(key)->second.active_record.pin_count;
  }

  lock_contexts_.erase(lock_it);
  return Status::OK();
}

Status MetaManager::BatchFree(const std::vector<KeyId>& flat_keys) {
  ScopedMethodTimer timer(&timing_log_enabled_, "BatchFree");
  if (flat_keys.empty() || !HasDistinctKeys(flat_keys)) {
    return Error(StatusCode::kInvalidArgument, ErrorTag::kInvalidRequest);
  }

  std::lock_guard<std::mutex> lock(meta_mutex_);
  if (!region_registered_) {
    return Error(StatusCode::kNotInitialized, ErrorTag::kRegionMissing);
  }

  // BatchFree 保持 all-or-nothing：先整批校验，再统一删除和回收。
  std::vector<SlotRef> slots_to_recycle;
  slots_to_recycle.reserve(flat_keys.size());
  for (const KeyId& key : flat_keys) {
    auto key_it = key_index_.find(key);
    if (key_it == key_index_.end()) {
      return Error(StatusCode::kNotFound, ErrorTag::kKeyMissing);
    }
    if (key_it->second.active_record.pin_count > 0) {
      return Error(StatusCode::kBusy, ErrorTag::kKeyBusy);
    }
    slots_to_recycle.push_back(key_it->second.active_record.slot);
  }

  for (const KeyId& key : flat_keys) {
    key_index_.erase(key);
  }
  for (const SlotRef& slot : slots_to_recycle) {
    ReleaseSlot(slot);
  }

  return Status::OK();
}

Address MetaManager::SlotAddress(const SlotRef& slot) const {
  // region 前半段是短 slot 区，后半段是长 slot 区。
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
  // 长度为 0 无效；否则优先使用更省空间的短 slot。
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

MetaManager::SlotRef MetaManager::AcquireSlot(SlotClass slot_class) {
  // 两类 slot 各自独立排队，避免不同长度的空间互相挤占。
  std::queue<std::size_t>& free_slots =
      slot_class == SlotClass::kShort ? short_free_slots_ : long_free_slots_;
  const std::size_t slot_index = free_slots.front();
  free_slots.pop();
  return SlotRef{slot_class, slot_index};
}

void MetaManager::ReleaseSlot(const SlotRef& slot) {
  // 回收时按 slot 类型放回原来的队列，保持地址空间分层不变。
  std::queue<std::size_t>& free_slots =
      slot.slot_class == SlotClass::kShort ? short_free_slots_ : long_free_slots_;
  free_slots.push(slot.slot_index);
}

void MetaManager::SetTimingLogEnabled(bool enabled) {
  timing_log_enabled_.store(enabled, std::memory_order_relaxed);
}

std::unique_ptr<MetaManager> CreateInMemoryMetaManager() {
  // 当前 POC 直接返回一个带完整状态的 MetaManager 实例。
  return std::make_unique<MetaManager>();
}

}  // namespace kvaf
