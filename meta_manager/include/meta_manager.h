#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "status.h"
#include "types.h"

namespace kvaf {

enum class AllocationMode { kFixed, kAppendOnly };

inline const char* AllocationModeName(AllocationMode mode) {
  switch (mode) {
    case AllocationMode::kFixed: return "fixed";
    case AllocationMode::kAppendOnly: return "append_only";
  }
  return "unknown";
}

inline Status ParseAllocationMode(const std::string& name, AllocationMode* mode) {
  if (mode == nullptr) return Status::InvalidArgument("allocation mode output is null");
  if (name == "fixed") *mode = AllocationMode::kFixed;
  else if (name == "append_only") *mode = AllocationMode::kAppendOnly;
  else return Status::InvalidArgument("allocation_mode must be fixed or append_only");
  return Status::OK();
}

// 读锁模式。当前 POC 只支持“稳定读 + 允许后续追加新 key”这一种语义。
enum class MetaManagerLockMode {
  kStableReadAllowAppend = 0,
};

// put 上下文，表示一次“已预留地址但尚未对读路径可见”的写事务。
struct PutContext {
  std::uint64_t id{0};
};

// lock 上下文，表示一次稳定读快照。
// 调用方需要用它把 lock / get / unlock 串起来。
struct LockContext {
  std::uint64_t id{0};
};

// 单个 key 对应的一段预留写入地址。
// AccessOperator 在拿到该结构后，把 payload 写到 address 指向的远端地址。
struct PutAllocation {
  KeyId key{0};
  Address address{0};
  std::uint64_t reserved_length{0};
};

// PreparePut 返回的批量地址描述。
struct PutAccessDesc {
  std::vector<PutAllocation> entries;
};

// BatchLookup 返回的扁平化访问描述。
// 上层可以按相同下标把 address / length / valid 重新映射回请求 key。
struct FlatAccessDesc {
  std::vector<Address> addresses;
  std::vector<std::uint64_t> lengths;
  std::vector<std::uint8_t> valid;
};

// MetaManager 负责维护 key 到远端地址的元数据映射，并提供:
// 1. put 路径上的地址预留与提交
// 2. get 路径上的稳定读快照
// 3. free 路径上的地址回收
class MetaManager {
 public:
  virtual ~MetaManager() = default;

  // 注册远端区域。fixed 模式下 slot_id = key，Key 必须在 slot 数范围内。
  // append_only 模式下 Key 独立于 slot，按真实长度紧密追加，单 Key 不跨 slot。
  virtual Status RegisterRemoteRegion(const RemoteRegionDesc& region) = 0;

  // 为一批 key 预留远端空间，具体 slot/offset 由分配模式决定。
  // 该阶段只分配地址，不暴露映射；只有 CommitPut 之后读路径才能看到这些 key。
  virtual Status PreparePut(const std::vector<KeyId>& flat_keys,
                            const std::vector<std::uint64_t>& lengths,
                            PutContext* put_context,
                            PutAccessDesc* put_desc) = 0;

  // 将 PreparePut 预留的地址正式提交为可见映射。
  virtual Status CommitPut(const PutContext& put_context) = 0;

  // 放弃一次尚未提交的 put。append_only 只在整个 slot 无引用时回收空间，
  // 不回退仍有其他预留/已提交对象的 slot 游标。
  virtual Status AbortPut(const PutContext& put_context) = 0;

  // 为一批 key 建立稳定读快照。
  // 成功后，这批 key 对应的记录会被 pin，直到 UnlockKeys 才允许 free 回收。
  virtual Status LockKeys(const std::vector<KeyId>& flat_keys,
                          MetaManagerLockMode lock_mode,
                          LockContext* lock_context) = 0;

  // 读取 LockKeys 固定下来的那一版地址信息。
  // 如果 key 顺序不匹配，或者锁定期间记录发生变化，则返回错误。
  virtual Status BatchLookup(const std::vector<KeyId>& flat_keys,
                             const LockContext& lock_context,
                             FlatAccessDesc* access_desc) = 0;

  // 释放稳定读快照，解除对应记录的 pin。
  virtual Status UnlockKeys(const LockContext& lock_context) = 0;

  // 删除一批未被 pin 的 key。fixed 回收各 Key 的 slot；append_only
  // 只有在最后一个预留/已提交对象被移除后才回收整个 slot。
  virtual Status BatchFree(const std::vector<KeyId>& flat_keys) = 0;
};

// 创建一个基于内存数据结构的 MetaManager POC 实现。
std::unique_ptr<MetaManager> CreateInMemoryMetaManager();
// Opt-in policy. No-argument factory above keeps the existing DSA implementation.
std::unique_ptr<MetaManager> CreateInMemoryMetaManager(AllocationMode mode);

}  // namespace kvaf
