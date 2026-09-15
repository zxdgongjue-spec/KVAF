#include "meta_manager.h"

#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

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
constexpr std::uint64_t kSlotSize = 256;

// 最轻量的断言助手。失败时直接输出错误信息，便于脱离 gtest 独立运行。
bool Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "EXPECT FAILED: " << message << '\n';
    return false;
  }
  return true;
}

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

// 构造一个按固定 slot 切分的远端 region。
RemoteRegionDesc MakeRegion(std::uint64_t slot_count) {
  RemoteRegionDesc region;
  region.base = kBaseAddress;
  region.length = slot_count * kSlotSize;
  region.slot_size = kSlotSize;
  return region;
}

std::unique_ptr<MetaManager> MakeInitializedManager(std::uint64_t slot_count) {
  auto manager = CreateInMemoryMetaManager();
  const Status status = manager->RegisterRemoteRegion(MakeRegion(slot_count));
  if (!status.ok()) {
    std::cerr << "Failed to initialize manager: " << status.message() << '\n';
    return nullptr;
  }
  return manager;
}

// 验证 region 注册阶段的参数检查与重复注册保护。
bool TestRegisterRemoteRegionValidation() {
  auto manager = CreateInMemoryMetaManager();

  RemoteRegionDesc invalid = MakeRegion(1);
  invalid.slot_size = 0;
  if (!ExpectStatusCode(manager->RegisterRemoteRegion(invalid),
                        StatusCode::kInvalidArgument,
                        "slot_size=0 should be rejected")) {
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

// 验证 put -> commit -> lock -> get -> unlock -> free 主流程，
// 同时确认 free 后 slot 会重新进入可分配的 free stack。
bool TestPutCommitLookupUnlockFreeRoundTrip() {
  auto manager = MakeInitializedManager(2);
  if (!manager) {
    return false;
  }

  const std::vector<KeyId> keys{11, 12};
  const std::vector<std::uint64_t> lengths{64, 128};
  PutContext put_context;
  PutAccessDesc put_desc;

  if (!ExpectOk(manager->PreparePut(keys, lengths, &put_context, &put_desc),
                "PreparePut should succeed")) {
    return false;
  }
  if (!Expect(put_desc.entries.size() == 2,
              "PreparePut should return one reserved address per key")) {
    return false;
  }
  if (!Expect(put_desc.entries[0].address == kBaseAddress,
              "first reservation should use the first slot")) {
    return false;
  }
  if (!Expect(put_desc.entries[1].address == kBaseAddress + kSlotSize,
              "second reservation should use the second slot")) {
    return false;
  }

  LockContext invisible_lock;
  if (!ExpectStatusCode(
          manager->LockKeys(keys, MetaManagerLockMode::kStableReadAllowAppend,
                            &invisible_lock),
          StatusCode::kNotFound,
          "uncommitted keys should stay invisible to readers")) {
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
                "BatchLookup should succeed for a locked snapshot")) {
    return false;
  }
  if (!Expect(access_desc.addresses == std::vector<std::uint64_t>{
                                           kBaseAddress,
                                           kBaseAddress + kSlotSize},
              "BatchLookup should return committed addresses in key order")) {
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

  LockContext free_lock;
  if (!ExpectStatusCode(
      manager->LockKeys(keys, MetaManagerLockMode::kStableReadAllowAppend,
                        &free_lock),
      StatusCode::kNotFound,
      "freed keys should no longer be visible")) {
    return false;
  }

  PutContext recycled_put;
  PutAccessDesc recycled_desc;
  if (!ExpectOk(manager->PreparePut({13}, {32}, &recycled_put, &recycled_desc),
                "freed slots should be reusable by later puts")) {
    return false;
  }
  return Expect(recycled_desc.entries[0].address == kBaseAddress,
                "BatchFree should recycle the first freed slot back to the stack");
}

// 验证 AbortPut 不会留下可见映射，并且会回收之前预留的 slot。
bool TestAbortPutRecyclesSlot() {
  auto manager = MakeInitializedManager(1);
  if (!manager) {
    return false;
  }

  PutContext first_put;
  PutAccessDesc first_desc;
  if (!ExpectOk(manager->PreparePut({21}, {32}, &first_put, &first_desc),
                "first PreparePut should succeed")) {
    return false;
  }
  if (!ExpectOk(manager->AbortPut(first_put), "AbortPut should succeed")) {
    return false;
  }

  LockContext invisible_lock;
  if (!ExpectStatusCode(
          manager->LockKeys({21}, MetaManagerLockMode::kStableReadAllowAppend,
                            &invisible_lock),
          StatusCode::kNotFound,
          "aborted puts should not leave visible mappings behind")) {
    return false;
  }

  PutContext second_put;
  PutAccessDesc second_desc;
  if (!ExpectOk(manager->PreparePut({22}, {48}, &second_put, &second_desc),
                "second PreparePut should reuse the freed slot")) {
    return false;
  }
  if (!Expect(second_desc.entries.size() == 1,
              "PreparePut should return one reservation")) {
    return false;
  }
  return Expect(second_desc.entries[0].address == first_desc.entries[0].address,
                "AbortPut should recycle the reserved slot back to the stack");
}

// 验证 pending 和 active 共用一个索引时，提交或回滚一个上下文不会改变另一个
// in-flight 上下文的可见性或 slot 归属。
bool TestPendingToActiveTransitionIsContextLocal() {
  auto manager = MakeInitializedManager(3);
  if (!manager) {
    return false;
  }

  PutContext first_put;
  PutAccessDesc first_desc;
  if (!ExpectOk(manager->PreparePut({23}, {32}, &first_put, &first_desc),
                "first PreparePut should succeed")) {
    return false;
  }
  PutContext second_put;
  PutAccessDesc second_desc;
  if (!ExpectOk(manager->PreparePut({24}, {48}, &second_put, &second_desc),
                "second PreparePut should succeed while first is pending")) {
    return false;
  }

  PutContext duplicate_pending_put;
  PutAccessDesc duplicate_pending_desc;
  if (!ExpectStatusCode(
          manager->PreparePut({24}, {48}, &duplicate_pending_put,
                              &duplicate_pending_desc),
          StatusCode::kAlreadyExists,
          "PreparePut should reject a key owned by another pending context")) {
    return false;
  }

  if (!ExpectOk(manager->CommitPut(first_put),
                "CommitPut should activate only its own pending keys")) {
    return false;
  }
  LockContext first_lock;
  if (!ExpectOk(
          manager->LockKeys({23}, MetaManagerLockMode::kStableReadAllowAppend,
                            &first_lock),
          "committed key should become visible")) {
    return false;
  }
  if (!ExpectOk(manager->UnlockKeys(first_lock),
                "UnlockKeys should release the committed-key lock")) {
    return false;
  }

  LockContext pending_lock;
  if (!ExpectStatusCode(
          manager->LockKeys({24}, MetaManagerLockMode::kStableReadAllowAppend,
                            &pending_lock),
          StatusCode::kNotFound,
          "a different pending context must remain invisible after commit")) {
    return false;
  }
  if (!ExpectOk(manager->AbortPut(second_put),
                "AbortPut should remove only its pending keys")) {
    return false;
  }

  PutContext remaining_slot_put;
  PutAccessDesc remaining_slot_desc;
  if (!ExpectOk(manager->PreparePut({25}, {64}, &remaining_slot_put,
                                    &remaining_slot_desc),
                "remaining free slot should be allocatable")) {
    return false;
  }
  PutContext recycled_put;
  PutAccessDesc recycled_desc;
  if (!ExpectOk(manager->PreparePut({26}, {64}, &recycled_put, &recycled_desc),
                "aborted pending slot should be reusable")) {
    return false;
  }
  return Expect(
      remaining_slot_desc.entries[0].address == second_desc.entries[0].address,
      "AbortPut should return the pending context's slot to the top of the stack");
}

// 验证只要 key 仍被稳定读快照 pin 住，就不允许 free。
bool TestBatchFreeFailsWhileLocked() {
  auto manager = MakeInitializedManager(1);
  if (!manager) {
    return false;
  }

  PutContext put_context;
  PutAccessDesc put_desc;
  if (!ExpectOk(manager->PreparePut({31}, {80}, &put_context, &put_desc),
                "PreparePut should succeed")) {
    return false;
  }
  if (!ExpectOk(manager->CommitPut(put_context), "CommitPut should succeed")) {
    return false;
  }

  LockContext lock_context;
  if (!ExpectOk(
          manager->LockKeys({31}, MetaManagerLockMode::kStableReadAllowAppend,
                            &lock_context),
          "LockKeys should succeed")) {
    return false;
  }

  if (!ExpectStatusCode(manager->BatchFree({31}), StatusCode::kBusy,
                        "BatchFree should reject locked keys")) {
    return false;
  }
  if (!ExpectOk(manager->UnlockKeys(lock_context), "UnlockKeys should succeed")) {
    return false;
  }
  return ExpectOk(manager->BatchFree({31}),
                  "BatchFree should succeed after unlock");
}

// 验证重复 key、错误查询 key 顺序等非法输入分支。
bool TestInvalidArgumentAndDuplicateKeyPaths() {
  auto manager = MakeInitializedManager(2);
  if (!manager) {
    return false;
  }

  PutContext duplicate_put;
  PutAccessDesc duplicate_desc;
  if (!ExpectStatusCode(manager->PreparePut({41, 41}, {8, 8}, &duplicate_put,
                                            &duplicate_desc),
                        StatusCode::kInvalidArgument,
                        "PreparePut should reject duplicate keys in one batch")) {
    return false;
  }

  PutContext put_context;
  PutAccessDesc put_desc;
  if (!ExpectOk(manager->PreparePut({42}, {16}, &put_context, &put_desc),
                "PreparePut should succeed for a fresh key")) {
    return false;
  }
  if (!ExpectOk(manager->CommitPut(put_context), "CommitPut should succeed")) {
    return false;
  }

  PutContext conflict_put;
  PutAccessDesc conflict_desc;
  if (!ExpectStatusCode(manager->PreparePut({42}, {32}, &conflict_put,
                                            &conflict_desc),
                        StatusCode::kAlreadyExists,
                        "PreparePut should reject an already committed key")) {
    return false;
  }

  LockContext lock_context;
  if (!ExpectOk(
          manager->LockKeys({42}, MetaManagerLockMode::kStableReadAllowAppend,
                            &lock_context),
          "LockKeys should succeed")) {
    return false;
  }

  FlatAccessDesc access_desc;
  if (!ExpectStatusCode(manager->BatchLookup({99}, lock_context, &access_desc),
                        StatusCode::kInvalidArgument,
                        "BatchLookup should validate key order against LockContext")) {
    return false;
  }
  return ExpectOk(manager->UnlockKeys(lock_context), "UnlockKeys should succeed");
}

// 验证在 region 尚未注册前，核心接口都会返回未初始化错误。
bool TestNotInitializedPaths() {
  auto manager = CreateInMemoryMetaManager();

  PutContext put_context;
  PutAccessDesc put_desc;
  if (!ExpectStatusCode(manager->PreparePut({51}, {8}, &put_context, &put_desc),
                        StatusCode::kNotInitialized,
                        "PreparePut should fail before RegisterRemoteRegion")) {
    return false;
  }

  LockContext lock_context;
  if (!ExpectStatusCode(
          manager->LockKeys({51}, MetaManagerLockMode::kStableReadAllowAppend,
                            &lock_context),
          StatusCode::kNotInitialized,
          "LockKeys should fail before RegisterRemoteRegion")) {
    return false;
  }

  return ExpectStatusCode(manager->BatchFree({51}), StatusCode::kNotInitialized,
                          "BatchFree should fail before RegisterRemoteRegion");
}

// 验证非法上下文 id 会被明确拒绝，避免调用方误用旧上下文。
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
  return ExpectStatusCode(manager->BatchLookup({1}, LockContext{666}, nullptr),
                          StatusCode::kInvalidArgument,
                          "BatchLookup should validate nullptr access_desc first");
}

// 验证单条写入长度不能超过 slot 容量。
bool TestPreparePutRejectsOversizedLength() {
  auto manager = MakeInitializedManager(1);
  if (!manager) {
    return false;
  }

  PutContext put_context;
  PutAccessDesc put_desc;
  return ExpectStatusCode(
      manager->PreparePut({61}, {kSlotSize + 1}, &put_context, &put_desc),
      StatusCode::kInvalidArgument,
      "PreparePut should reject payload larger than one slot");
}

// 验证 free queue 耗尽后会返回 OOM，而不是产生重复分配。
bool TestPreparePutOutOfMemory() {
  auto manager = MakeInitializedManager(1);
  if (!manager) {
    return false;
  }

  PutContext first_put;
  PutAccessDesc first_desc;
  if (!ExpectOk(manager->PreparePut({71}, {16}, &first_put, &first_desc),
                "first PreparePut should consume the only slot")) {
    return false;
  }

  PutContext second_put;
  PutAccessDesc second_desc;
  return ExpectStatusCode(manager->PreparePut({72}, {16}, &second_put, &second_desc),
                          StatusCode::kOutOfMemory,
                          "PreparePut should fail when no free slot remains");
}

// 验证同一个 lock context 只能释放一次，避免 pin_count 被重复扣减。
bool TestDoubleUnlockReturnsNotFound() {
  auto manager = MakeInitializedManager(1);
  if (!manager) {
    return false;
  }

  PutContext put_context;
  PutAccessDesc put_desc;
  if (!ExpectOk(manager->PreparePut({81}, {24}, &put_context, &put_desc),
                "PreparePut should succeed")) {
    return false;
  }
  if (!ExpectOk(manager->CommitPut(put_context), "CommitPut should succeed")) {
    return false;
  }

  LockContext lock_context;
  if (!ExpectOk(
          manager->LockKeys({81}, MetaManagerLockMode::kStableReadAllowAppend,
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
  const std::vector<TestCase> tests{
      {"RegisterRemoteRegionValidation", TestRegisterRemoteRegionValidation},
      {"PutCommitLookupUnlockFreeRoundTrip",
       TestPutCommitLookupUnlockFreeRoundTrip},
      {"AbortPutRecyclesSlot", TestAbortPutRecyclesSlot},
      {"PendingToActiveTransitionIsContextLocal",
       TestPendingToActiveTransitionIsContextLocal},
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
