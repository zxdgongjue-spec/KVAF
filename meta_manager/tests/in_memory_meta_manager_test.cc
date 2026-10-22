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

constexpr KeyId DsaKey(KeyId pool_row, KeyId layer, KeyId block_q) {
  constexpr KeyId kLayers = 2, kQCapacity = 2;
  return pool_row * (kLayers * kQCapacity) + layer * kQCapacity + block_q;
}

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
// 同时确认相同 DSA key 在 free 后会复用自己的固定 slot。
bool TestPutCommitLookupUnlockFreeRoundTrip() {
  auto manager = MakeInitializedManager(2);
  if (!manager) {
    return false;
  }

  const std::vector<KeyId> keys{DsaKey(0, 0, 0), DsaKey(0, 0, 1)};
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
  if (!ExpectOk(manager->PreparePut({DsaKey(0, 0, 0)}, {32}, &recycled_put, &recycled_desc),
                "freed slots should be reusable by later puts")) {
    return false;
  }
  return Expect(recycled_desc.entries[0].address == kBaseAddress,
                "BatchFree should allow the same key to reuse its slot");
}

// 验证 AbortPut 不会留下可见映射，并且会回收之前预留的 slot。
bool TestAbortPutRecyclesSlot() {
  auto manager = MakeInitializedManager(1);
  if (!manager) {
    return false;
  }

  PutContext first_put;
  PutAccessDesc first_desc;
  if (!ExpectOk(manager->PreparePut({DsaKey(0, 0, 0)}, {32}, &first_put, &first_desc),
                "first PreparePut should succeed")) {
    return false;
  }
  if (!ExpectOk(manager->AbortPut(first_put), "AbortPut should succeed")) {
    return false;
  }

  LockContext invisible_lock;
  if (!ExpectStatusCode(
          manager->LockKeys({DsaKey(0, 0, 0)}, MetaManagerLockMode::kStableReadAllowAppend,
                            &invisible_lock),
          StatusCode::kNotFound,
          "aborted puts should not leave visible mappings behind")) {
    return false;
  }

  PutContext second_put;
  PutAccessDesc second_desc;
  if (!ExpectOk(manager->PreparePut({DsaKey(0, 0, 0)}, {48}, &second_put, &second_desc),
                "second PreparePut should reuse the freed slot")) {
    return false;
  }
  if (!Expect(second_desc.entries.size() == 1,
              "PreparePut should return one reservation")) {
    return false;
  }
  return Expect(second_desc.entries[0].address == first_desc.entries[0].address,
                "AbortPut should allow the same key to reuse its slot");
}

// 验证只要 key 仍被稳定读快照 pin 住，就不允许 free。
bool TestBatchFreeFailsWhileLocked() {
  auto manager = MakeInitializedManager(1);
  if (!manager) {
    return false;
  }

  PutContext put_context;
  PutAccessDesc put_desc;
  if (!ExpectOk(manager->PreparePut({DsaKey(0, 0, 0)}, {80}, &put_context, &put_desc),
                "PreparePut should succeed")) {
    return false;
  }
  if (!ExpectOk(manager->CommitPut(put_context), "CommitPut should succeed")) {
    return false;
  }

  LockContext lock_context;
  if (!ExpectOk(
          manager->LockKeys({DsaKey(0, 0, 0)}, MetaManagerLockMode::kStableReadAllowAppend,
                            &lock_context),
          "LockKeys should succeed")) {
    return false;
  }

  if (!ExpectStatusCode(manager->BatchFree({DsaKey(0, 0, 0)}), StatusCode::kBusy,
                        "BatchFree should reject locked keys")) {
    return false;
  }
  if (!ExpectOk(manager->UnlockKeys(lock_context), "UnlockKeys should succeed")) {
    return false;
  }
  return ExpectOk(manager->BatchFree({DsaKey(0, 0, 0)}),
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
  if (!ExpectStatusCode(manager->PreparePut({DsaKey(0, 0, 0), DsaKey(0, 0, 0)}, {8, 8}, &duplicate_put,
                                            &duplicate_desc),
                        StatusCode::kInvalidArgument,
                        "PreparePut should reject duplicate keys in one batch")) {
    return false;
  }

  PutContext put_context;
  PutAccessDesc put_desc;
  if (!ExpectOk(manager->PreparePut({DsaKey(0, 0, 1)}, {16}, &put_context, &put_desc),
                "PreparePut should succeed for a fresh key")) {
    return false;
  }
  if (!ExpectOk(manager->CommitPut(put_context), "CommitPut should succeed")) {
    return false;
  }

  PutContext conflict_put;
  PutAccessDesc conflict_desc;
  if (!ExpectStatusCode(manager->PreparePut({DsaKey(0, 0, 1)}, {32}, &conflict_put,
                                            &conflict_desc),
                        StatusCode::kAlreadyExists,
                        "PreparePut should reject an already committed key")) {
    return false;
  }

  LockContext lock_context;
  if (!ExpectOk(
          manager->LockKeys({DsaKey(0, 0, 1)}, MetaManagerLockMode::kStableReadAllowAppend,
                            &lock_context),
          "LockKeys should succeed")) {
    return false;
  }

  FlatAccessDesc access_desc;
  if (!ExpectStatusCode(manager->BatchLookup({DsaKey(0, 0, 0)}, lock_context, &access_desc),
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
  if (!ExpectStatusCode(manager->PreparePut({DsaKey(0, 0, 0)}, {8}, &put_context, &put_desc),
                        StatusCode::kNotInitialized,
                        "PreparePut should fail before RegisterRemoteRegion")) {
    return false;
  }

  LockContext lock_context;
  if (!ExpectStatusCode(
          manager->LockKeys({DsaKey(0, 0, 0)}, MetaManagerLockMode::kStableReadAllowAppend,
                            &lock_context),
          StatusCode::kNotInitialized,
          "LockKeys should fail before RegisterRemoteRegion")) {
    return false;
  }

  return ExpectStatusCode(manager->BatchFree({DsaKey(0, 0, 0)}), StatusCode::kNotInitialized,
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
      manager->PreparePut({DsaKey(0, 0, 0)}, {kSlotSize + 1}, &put_context, &put_desc),
      StatusCode::kInvalidArgument,
      "PreparePut should reject payload larger than one slot");
}

// 固定映射要求 Key 小于注册的 slot 数量。
bool TestPreparePutRejectsOutOfRangeKey() {
  auto manager = MakeInitializedManager(1);
  if (!manager) {
    return false;
  }

  PutContext first_put;
  PutAccessDesc first_desc;
  if (!ExpectOk(manager->PreparePut({DsaKey(0, 0, 0)}, {16}, &first_put, &first_desc),
                "first PreparePut should consume the only slot")) {
    return false;
  }

  PutContext second_put;
  PutAccessDesc second_desc;
  return ExpectStatusCode(manager->PreparePut({DsaKey(0, 0, 1)}, {16}, &second_put, &second_desc),
                          StatusCode::kInvalidArgument,
                          "PreparePut should reject a key outside the region");
}

// 验证同一个 lock context 只能释放一次，避免 pin_count 被重复扣减。
bool TestDoubleUnlockReturnsNotFound() {
  auto manager = MakeInitializedManager(1);
  if (!manager) {
    return false;
  }

  PutContext put_context;
  PutAccessDesc put_desc;
  if (!ExpectOk(manager->PreparePut({DsaKey(0, 0, 0)}, {24}, &put_context, &put_desc),
                "PreparePut should succeed")) {
    return false;
  }
  if (!ExpectOk(manager->CommitPut(put_context), "CommitPut should succeed")) {
    return false;
  }

  LockContext lock_context;
  if (!ExpectOk(
          manager->LockKeys({DsaKey(0, 0, 0)}, MetaManagerLockMode::kStableReadAllowAppend,
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

// DSA directory and prefix keys select fixed addresses even in reverse order.
bool TestDsaKeySelectsFixedSlot() {
  constexpr KeyId kDirectoryCells = 2 * 2 * 2;
  const KeyId prefix_key = kDirectoryCells + 1 * 2 + 1;
  const std::vector<KeyId> keys{prefix_key, DsaKey(1, 1, 1), DsaKey(0, 0, 0)};
  auto manager = MakeInitializedManager(kDirectoryCells + 2 * 2);
  if (!manager) return false;
  PutContext context;
  PutAccessDesc desc;
  if (!ExpectOk(manager->PreparePut(keys, {8, 16, 32}, &context, &desc),
                "DSA PreparePut should succeed")) return false;
  for (std::size_t i = 0; i < keys.size(); ++i) {
    if (!Expect(desc.entries[i].address == kBaseAddress + keys[i] * kSlotSize,
                "DSA key must select its fixed slot")) return false;
  }
  return ExpectOk(manager->AbortPut(context), "DSA AbortPut should succeed");
}

struct TestCase {
  const char* name;
  std::function<bool()> run;
};

}  // namespace

int main() {
  const std::vector<TestCase> tests{
      {"DsaKeySelectsFixedSlot", TestDsaKeySelectsFixedSlot},
      {"RegisterRemoteRegionValidation", TestRegisterRemoteRegionValidation},
      {"PutCommitLookupUnlockFreeRoundTrip",
       TestPutCommitLookupUnlockFreeRoundTrip},
      {"AbortPutRecyclesSlot", TestAbortPutRecyclesSlot},
      {"BatchFreeFailsWhileLocked", TestBatchFreeFailsWhileLocked},
      {"InvalidArgumentAndDuplicateKeyPaths",
       TestInvalidArgumentAndDuplicateKeyPaths},
      {"NotInitializedPaths", TestNotInitializedPaths},
      {"InvalidContextPaths", TestInvalidContextPaths},
      {"PreparePutRejectsOversizedLength", TestPreparePutRejectsOversizedLength},
      {"PreparePutRejectsOutOfRangeKey", TestPreparePutRejectsOutOfRangeKey},
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
