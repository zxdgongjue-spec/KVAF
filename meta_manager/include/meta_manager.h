#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "status.h"
#include "types.h"

namespace kvaf {

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

  // 注册一块可管理的远端内存区域。
  virtual Status RegisterRemoteRegion(const RemoteRegionDesc& region) = 0;

  // 为一批 key 预留远端 slot。
  // 该阶段只分配地址，不暴露映射；只有 CommitPut 之后读路径才能看到这些 key。
  virtual Status PreparePut(const std::vector<KeyId>& flat_keys,
                            const std::vector<std::uint64_t>& lengths,
                            PutContext* put_context,
                            PutAccessDesc* put_desc) = 0;

  // 将 PreparePut 预留的地址正式提交为可见映射。
  virtual Status CommitPut(const PutContext& put_context) = 0;

  // 放弃一次尚未提交的 put，并把预留 slot 归还到空闲 slot 池。
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

  // 释放一批 key 当前占用的 slot，并放回空闲 slot 池。
  virtual Status BatchFree(const std::vector<KeyId>& flat_keys) = 0;
};

// 创建一个基于内存数据结构的 MetaManager POC 实现。
std::unique_ptr<MetaManager> CreateInMemoryMetaManager();

}  // namespace kvaf
