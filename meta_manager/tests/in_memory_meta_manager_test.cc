#include "kvaf/meta_manager/meta_manager.h"

#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using kvaf::Address;
using kvaf::CreateInMemoryMetaManager;
using kvaf::FlatAccessDesc;
using kvaf::KeyId;
using kvaf::LockContext;
using kvaf::MetaManager;
using kvaf::MetaManagerLockMode;
using kvaf::PutAccessDesc;
using kvaf::PutContext;
using kvaf::RemoteRegionDesc;
using kvaf::Status;
using kvaf::StatusCode;

constexpr std::uint64_t kBaseAddress = 4096;
constexpr std::uint64_t kShortSlotSize = 128;
constexpr std::uint64_t kLongSlotSize = 256;

// 轻量断言助手：测试失败时直接打印原因，便于独立执行时定位问题。
bool Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "EXPECT FAILED: " << message << '\n';
    return false;
  }
  return true;
}

// 检查返回码是否符合预期，同时把 detail 一并带出来。
bool ExpectStatusCode(const Status& status,
                      StatusCode code,
                      const std::string& message) {
  if (status.code() != code) {
    std::cerr << "EXPECT FAILED: " << message
              << " expected=" << static_cast<int>(code)
              << " actual=" << static_cast<int>(status.code())
              << " detail=" << status.message() << '\n';
    return false;
  }
  return true;
}

bool ExpectOk(const Status& status, const std::string& message) {
  return ExpectStatusCode(status, StatusCode::kOk, message);
}

// 构造一个同时包含短 slot 区和长 slot 区的测试 region。
RemoteRegionDesc MakeRegion(std::uint64_t slot_count_per_class) {
  RemoteRegionDesc region;
  region.base = kBaseAddress;
  region.short_slot_size = kShortSlotSize;
  region.long_slot_size = kLongSlotSize;
  region.slot_count_per_class = slot_count_per_class;
  region.length = slot_count_per_class * (kShortSlotSize + kLongSlotSize);
  return region;
}

// 长 slot 区紧跟在短 slot 区之后，因此它的起始地址可以直接算出来。
Address LongSlotBase(std::uint64_t slot_count_per_class) {
  return kBaseAddress + slot_count_per_class * kShortSlotSize;
}

// 创建并初始化一个可直接使用的 MetaManager，减少各个用例里的重复样板代码。
std::unique_ptr<MetaManager> MakeInitializedManager(
    std::uint64_t slot_count_per_class) {
  auto manager = CreateInMemoryMetaManager();
  const Status status =
      manager->RegisterRemoteRegion(MakeRegion(slot_count_per_class));
  if (!status.ok()) {
    std::cerr << "Failed to initialize manager: " << status.message() << '\n';
    return nullptr;
  }
  return manager;
}

// 验证 region 注册阶段的参数检查，以及“只允许注册一次”的约束。
bool TestRegisterRemoteRegionValidation() {
  auto manager = CreateInMemoryMetaManager();

  RemoteRegionDesc invalid = MakeRegion(1);
  invalid.short_slot_size = 0;
  if (!ExpectStatusCode(manager->RegisterRemoteRegion(invalid),
                        StatusCode::kInvalidArgument,
                        "zero short slot size should be rejected")) {
    return false;
  }

  RemoteRegionDesc valid = MakeRegion(2);
  if (!ExpectOk(manager->RegisterRemoteRegion(valid),
                "RegisterRemoteRegion should accept a valid region")) {
    return false;
  }

  return ExpectStatusCode(manager->RegisterRemoteRegion(valid),
                          StatusCode::kAlreadyExists,
                          "RegisterRemoteRegion should reject a second call");
}

// 验证短 slot 路径上的完整主流程：预留、提交、查询、解锁、回收、复用。
bool TestPutCommitLookupUnlockFreeRoundTrip() {
  auto manager = MakeInitializedManager(2);
  if (!manager) {
    return false;
  }

  const std::vector<KeyId> keys{"k11", "k12"};
  const std::vector<std::uint64_t> lengths{64, 120};
  PutContext put_context;
  PutAccessDesc put_desc;

  if (!ExpectOk(manager->PreparePut(keys, lengths, &put_context, &put_desc),
                "PreparePut should succeed")) {
    return false;
  }
  if (!Expect(put_desc.entries.size() == 2,
              "PreparePut should return one address per key")) {
    return false;
  }
  if (!Expect(put_desc.entries[0].address == kBaseAddress,
              "first short reservation should use the first short slot")) {
    return false;
  }
  if (!Expect(put_desc.entries[1].address == kBaseAddress + kShortSlotSize,
              "second short reservation should use the second short slot")) {
    return false;
  }

  LockContext invisible_lock;
  if (!ExpectStatusCode(
          manager->LockKeys(keys, MetaManagerLockMode::kStableReadAllowAppend,
                            &invisible_lock),
          StatusCode::kNotFound,
          "uncommitted keys should stay invisible")) {
    return false;
  }

  if (!ExpectOk(manager->CommitPut(put_context), "CommitPut should succeed")) {
    return false;
  }

  LockContext lock_context;
  if (!ExpectOk(
          manager->LockKeys(keys, MetaManagerLockMode::kStableReadAllowAppend,
                            &lock_context),
          "LockKeys should succeed after commit")) {
    return false;
  }

  FlatAccessDesc access_desc;
  if (!ExpectOk(manager->BatchLookup(keys, lock_context, &access_desc),
                "BatchLookup should succeed")) {
    return false;
  }
  if (!Expect(access_desc.addresses ==
                  std::vector<Address>{kBaseAddress,
                                       kBaseAddress + kShortSlotSize},
              "BatchLookup should return committed short addresses")) {
    return false;
  }
  if (!Expect(access_desc.lengths == lengths,
              "BatchLookup should return committed lengths")) {
    return false;
  }
  if (!Expect(access_desc.valid == std::vector<std::uint8_t>{1, 1},
              "BatchLookup should mark committed records valid")) {
    return false;
  }

  if (!ExpectOk(manager->UnlockKeys(lock_context), "UnlockKeys should succeed")) {
    return false;
  }
  if (!ExpectOk(manager->BatchFree(keys), "BatchFree should succeed")) {
    return false;
  }

  PutContext recycled_put;
  PutAccessDesc recycled_desc;
  if (!ExpectOk(manager->PreparePut({"k13"}, {32}, &recycled_put, &recycled_desc),
                "freed short slots should be reusable")) {
    return false;
  }
  return Expect(recycled_desc.entries[0].address == kBaseAddress,
                "BatchFree should recycle the first short slot");
}

// 验证较大的 payload 会进入长 slot 区，而不是挤占短 slot。
bool TestLongSlotAllocation() {
  auto manager = MakeInitializedManager(2);
  if (!manager) {
    return false;
  }

  const std::vector<KeyId> keys{"short-key", "long-key"};
  const std::vector<std::uint64_t> lengths{96, 200};
  PutContext put_context;
  PutAccessDesc put_desc;

  if (!ExpectOk(manager->PreparePut(keys, lengths, &put_context, &put_desc),
                "PreparePut should place keys into different slot classes")) {
    return false;
  }
  if (!Expect(put_desc.entries[0].address == kBaseAddress,
              "short payload should use short slot space")) {
    return false;
  }
  if (!Expect(put_desc.entries[1].address == LongSlotBase(2),
              "long payload should use long slot space")) {
    return false;
  }

  if (!ExpectOk(manager->CommitPut(put_context), "CommitPut should succeed")) {
    return false;
  }

  LockContext lock_context;
  if (!ExpectOk(
          manager->LockKeys(keys, MetaManagerLockMode::kStableReadAllowAppend,
                            &lock_context),
          "LockKeys should succeed for mixed slot classes")) {
    return false;
  }

  FlatAccessDesc access_desc;
  if (!ExpectOk(manager->BatchLookup(keys, lock_context, &access_desc),
                "BatchLookup should succeed for mixed slot classes")) {
    return false;
  }
  if (!Expect(access_desc.addresses ==
                  std::vector<Address>{kBaseAddress, LongSlotBase(2)},
              "BatchLookup should preserve short and long addresses")) {
    return false;
  }

  return ExpectOk(manager->UnlockKeys(lock_context),
                  "UnlockKeys should succeed for mixed slot classes");
}

// 验证 AbortPut 不会留下可见映射，并且会把短 slot 归还给空闲队列。
bool TestAbortPutRecyclesSlot() {
  auto manager = MakeInitializedManager(1);
  if (!manager) {
    return false;
  }

  PutContext first_put;
  PutAccessDesc first_desc;
  if (!ExpectOk(manager->PreparePut({"k21"}, {32}, &first_put, &first_desc),
                "first PreparePut should succeed")) {
    return false;
  }
  if (!ExpectOk(manager->AbortPut(first_put), "AbortPut should succeed")) {
    return false;
  }

  PutContext second_put;
  PutAccessDesc second_desc;
  if (!ExpectOk(manager->PreparePut({"k22"}, {48}, &second_put, &second_desc),
                "second PreparePut should reuse the freed short slot")) {
    return false;
  }
  return Expect(second_desc.entries[0].address == first_desc.entries[0].address,
                "AbortPut should recycle the reserved short slot");
}

// 验证被 LockKeys 固定住的 key 在解锁前不能被 BatchFree 回收。
bool TestBatchFreeFailsWhileLocked() {
  auto manager = MakeInitializedManager(1);
  if (!manager) {
    return false;
  }

  PutContext put_context;
  PutAccessDesc put_desc;
  if (!ExpectOk(manager->PreparePut({"k31"}, {80}, &put_context, &put_desc),
                "PreparePut should succeed")) {
    return false;
  }
  if (!ExpectOk(manager->CommitPut(put_context), "CommitPut should succeed")) {
    return false;
  }

  LockContext lock_context;
  if (!ExpectOk(
          manager->LockKeys({"k31"}, MetaManagerLockMode::kStableReadAllowAppend,
                            &lock_context),
          "LockKeys should succeed")) {
    return false;
  }

  if (!ExpectStatusCode(manager->BatchFree({"k31"}), StatusCode::kBusy,
                        "BatchFree should reject locked keys")) {
    return false;
  }
  if (!ExpectOk(manager->UnlockKeys(lock_context), "UnlockKeys should succeed")) {
    return false;
  }
  return ExpectOk(manager->BatchFree({"k31"}),
                  "BatchFree should succeed after unlock");
}

// 验证重复 key、错误 key 顺序等典型非法输入分支。
bool TestInvalidArgumentAndDuplicateKeyPaths() {
  auto manager = MakeInitializedManager(2);
  if (!manager) {
    return false;
  }

  PutContext duplicate_put;
  PutAccessDesc duplicate_desc;
  if (!ExpectStatusCode(
          manager->PreparePut({"dup", "dup"}, {8, 8}, &duplicate_put,
                              &duplicate_desc),
          StatusCode::kInvalidArgument,
          "PreparePut should reject duplicate keys")) {
    return false;
  }

  PutContext put_context;
  PutAccessDesc put_desc;
  if (!ExpectOk(manager->PreparePut({"k42"}, {16}, &put_context, &put_desc),
                "PreparePut should succeed for a fresh key")) {
    return false;
  }
  if (!ExpectOk(manager->CommitPut(put_context), "CommitPut should succeed")) {
    return false;
  }

  PutContext conflict_put;
  PutAccessDesc conflict_desc;
  if (!ExpectStatusCode(manager->PreparePut({"k42"}, {32}, &conflict_put,
                                            &conflict_desc),
                        StatusCode::kAlreadyExists,
                        "PreparePut should reject an existing key")) {
    return false;
  }

  LockContext lock_context;
  if (!ExpectOk(
          manager->LockKeys({"k42"}, MetaManagerLockMode::kStableReadAllowAppend,
                            &lock_context),
          "LockKeys should succeed")) {
    return false;
  }

  FlatAccessDesc access_desc;
  if (!ExpectStatusCode(
          manager->BatchLookup({"other"}, lock_context, &access_desc),
          StatusCode::kInvalidArgument,
          "BatchLookup should reject keys that do not match the lock context")) {
    return false;
  }
  return ExpectOk(manager->UnlockKeys(lock_context), "UnlockKeys should succeed");
}

// 验证 region 尚未注册时，核心接口都会返回未初始化错误。
bool TestNotInitializedPaths() {
  auto manager = CreateInMemoryMetaManager();

  PutContext put_context;
  PutAccessDesc put_desc;
  if (!ExpectStatusCode(
          manager->PreparePut({"k51"}, {8}, &put_context, &put_desc),
          StatusCode::kNotInitialized,
          "PreparePut should fail before RegisterRemoteRegion")) {
    return false;
  }

  LockContext lock_context;
  if (!ExpectStatusCode(
          manager->LockKeys({"k51"}, MetaManagerLockMode::kStableReadAllowAppend,
                            &lock_context),
          StatusCode::kNotInitialized,
          "LockKeys should fail before RegisterRemoteRegion")) {
    return false;
  }

  return ExpectStatusCode(manager->BatchFree({"k51"}), StatusCode::kNotInitialized,
                          "BatchFree should fail before RegisterRemoteRegion");
}

// 验证未知上下文 id 和空输出指针等错误使用场景。
bool TestInvalidContextPaths() {
  auto manager = MakeInitializedManager(1);
  if (!manager) {
    return false;
  }

  if (!ExpectStatusCode(manager->CommitPut(PutContext{999}),
                        StatusCode::kNotFound,
                        "CommitPut should reject an unknown PutContext")) {
    return false;
  }
  if (!ExpectStatusCode(manager->AbortPut(PutContext{888}),
                        StatusCode::kNotFound,
                        "AbortPut should reject an unknown PutContext")) {
    return false;
  }
  if (!ExpectStatusCode(manager->UnlockKeys(LockContext{777}),
                        StatusCode::kNotFound,
                        "UnlockKeys should reject an unknown LockContext")) {
    return false;
  }
  return ExpectStatusCode(
      manager->BatchLookup({"k1"}, LockContext{666}, nullptr),
      StatusCode::kInvalidArgument,
      "BatchLookup should reject a null output pointer");
}

// 验证超过长 slot 容量的 payload 会被拒绝。
bool TestPreparePutRejectsOversizedLength() {
  auto manager = MakeInitializedManager(1);
  if (!manager) {
    return false;
  }

  PutContext put_context;
  PutAccessDesc put_desc;
  return ExpectStatusCode(
      manager->PreparePut({"k61"}, {kLongSlotSize + 1}, &put_context, &put_desc),
      StatusCode::kInvalidArgument,
      "PreparePut should reject payload larger than the long slot");
}

// 验证某一类 slot 队列耗尽后，会返回 OOM，而不是错误复用已有空间。
bool TestPreparePutOutOfMemory() {
  auto manager = MakeInitializedManager(1);
  if (!manager) {
    return false;
  }

  PutContext put_context;
  PutAccessDesc put_desc;
  return ExpectStatusCode(
      manager->PreparePut({"k71", "k72"}, {16, 32}, &put_context, &put_desc),
      StatusCode::kOutOfMemory,
      "PreparePut should fail when the short slot queue is exhausted");
}

// 验证同一个 lock context 只能成功解锁一次，避免 pin_count 被重复扣减。
bool TestDoubleUnlockReturnsNotFound() {
  auto manager = MakeInitializedManager(1);
  if (!manager) {
    return false;
  }

  PutContext put_context;
  PutAccessDesc put_desc;
  if (!ExpectOk(manager->PreparePut({"k81"}, {24}, &put_context, &put_desc),
                "PreparePut should succeed")) {
    return false;
  }
  if (!ExpectOk(manager->CommitPut(put_context), "CommitPut should succeed")) {
    return false;
  }

  LockContext lock_context;
  if (!ExpectOk(
          manager->LockKeys({"k81"}, MetaManagerLockMode::kStableReadAllowAppend,
                            &lock_context),
          "LockKeys should succeed")) {
    return false;
  }
  if (!ExpectOk(manager->UnlockKeys(lock_context), "first UnlockKeys should succeed")) {
    return false;
  }
  return ExpectStatusCode(manager->UnlockKeys(lock_context), StatusCode::kNotFound,
                          "second UnlockKeys should fail because context is gone");
}

struct TestCase {
  const char* name;
  std::function<bool()> run;
};

}  // namespace

int main() {
  // 把所有用例放进一张表里统一执行，便于后续继续扩充覆盖面。
  const std::vector<TestCase> tests{
      {"RegisterRemoteRegionValidation", TestRegisterRemoteRegionValidation},
      {"PutCommitLookupUnlockFreeRoundTrip",
       TestPutCommitLookupUnlockFreeRoundTrip},
      {"LongSlotAllocation", TestLongSlotAllocation},
      {"AbortPutRecyclesSlot", TestAbortPutRecyclesSlot},
      {"BatchFreeFailsWhileLocked", TestBatchFreeFailsWhileLocked},
      {"InvalidArgumentAndDuplicateKeyPaths",
       TestInvalidArgumentAndDuplicateKeyPaths},
      {"NotInitializedPaths", TestNotInitializedPaths},
      {"InvalidContextPaths", TestInvalidContextPaths},
      {"PreparePutRejectsOversizedLength", TestPreparePutRejectsOversizedLength},
      {"PreparePutOutOfMemory", TestPreparePutOutOfMemory},
      {"DoubleUnlockReturnsNotFound", TestDoubleUnlockReturnsNotFound},
  };

  bool all_passed = true;
  for (const TestCase& test : tests) {
    const bool passed = test.run();
    std::cout << (passed ? "[PASS] " : "[FAIL] ") << test.name << '\n';
    all_passed = all_passed && passed;
  }

  return all_passed ? 0 : 1;
}
