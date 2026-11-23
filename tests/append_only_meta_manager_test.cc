#include "meta_manager.h"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <vector>

// Fail one allocation, then re-enable allocation so error Status creation and
// rollback can run normally. Exercise failures after partial slot/Key mutation.
static long fail_after = -1;
void* operator new(std::size_t n) {
  if (fail_after == 0) { fail_after = -1; throw std::bad_alloc(); }
  if (fail_after > 0) --fail_after;
  if (void* p = std::malloc(n ? n : 1)) return p;
  throw std::bad_alloc();
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

#define CHECK(x) do { if (!(x)) { std::cerr << "line " << __LINE__ << ": " << #x << '\n'; std::exit(1); } } while (false)
using namespace kvaf;
constexpr Address base = 0x100000;
struct Fixture {
  std::unique_ptr<MetaManager> meta = CreateInMemoryMetaManager(AllocationMode::kAppendOnly);
  Fixture(std::size_t slots, std::uint64_t capacity = 64) {
    CHECK(meta->RegisterRemoteRegion({base, slots*capacity, capacity, DeviceKind::kCpu}).ok());
  }
  PutAccessDesc Put(std::vector<KeyId> keys, std::vector<std::uint64_t> lengths) {
    PutContext context;
    PutAccessDesc desc;
    CHECK(meta->PreparePut(keys, lengths, &context, &desc).ok());
    CHECK(meta->CommitPut(context).ok());
    return desc;
  }
  Address Lookup(KeyId key) {
    LockContext lock;
    FlatAccessDesc desc;
    CHECK(meta->LockKeys({key}, MetaManagerLockMode::kStableReadAllowAppend, &lock).ok());
    CHECK(meta->BatchLookup({key}, lock, &desc).ok());
    CHECK(meta->UnlockKeys(lock).ok());
    return desc.addresses[0];
  }
  void Missing(KeyId key) {
    LockContext lock;
    CHECK(meta->LockKeys({key}, MetaManagerLockMode::kStableReadAllowAppend, &lock).code() == StatusCode::kNotFound);
  }
};

void PackingAndRecycling() {
  Fixture f(2);
  auto d = f.Put({100, 200, 300}, {20, 30, 24});
  CHECK(d.entries[0].address == base && d.entries[1].address == base+20);
  CHECK(d.entries[2].address == base+64);
  CHECK(d.entries[1].reserved_length == 30);
  CHECK(f.Put({400}, {40}).entries[0].address == base+88);  // exact tail fit
  PutContext c; PutAccessDesc out;
  CHECK(f.meta->PreparePut({500}, {1}, &c, &out).code() == StatusCode::kOutOfMemory);
  CHECK(f.meta->BatchFree({200}).ok());
  f.Missing(200);
  CHECK(f.meta->PreparePut({500}, {1}, &c, &out).code() == StatusCode::kOutOfMemory);
  CHECK(f.Lookup(100) == base && f.Lookup(300) == base+64);
  CHECK(f.meta->BatchFree({100}).ok());
  CHECK(f.Put({500}, {64}).entries[0].address == base);  // whole empty slot reused
  CHECK(f.Lookup(300) == base+64);
}

void FailedPrepareIsAtomic() {
  Fixture f(2, 16);
  f.Put({10}, {4});
  PutContext c{999}; PutAccessDesc d;
  CHECK(f.meta->PreparePut({20,30,40}, {10,16,1}, &c, &d).code() == StatusCode::kOutOfMemory);
  CHECK(c.id == 0 && d.entries.empty());
  f.Missing(20); f.Missing(30); f.Missing(40);
  auto next = f.Put({20,30}, {12,16});
  CHECK(next.entries[0].address == base+4 && next.entries[1].address == base+16);
}

void AbortPreservesLaterReservations() {
  Fixture f(1);
  PutContext a, b, c; PutAccessDesc da, db, dc;
  CHECK(f.meta->PreparePut({10}, {20}, &a, &da).ok());
  f.Missing(10);
  CHECK(f.meta->PreparePut({10}, {1}, &c, &dc).code() == StatusCode::kAlreadyExists);
  CHECK(f.meta->PreparePut({20}, {30}, &b, &db).ok());
  CHECK(db.entries[0].address == base+20);
  CHECK(f.meta->CommitPut(b).ok());
  CHECK(f.meta->AbortPut(a).ok());
  CHECK(f.meta->AbortPut(a).code() == StatusCode::kNotFound);
  CHECK(f.meta->PreparePut({30}, {14}, &c, &dc).ok());
  CHECK(dc.entries[0].address == base+50);  // never rewinds over B
  CHECK(f.meta->BatchFree({20}).ok());
  PutContext later; PutAccessDesc dl;
  CHECK(f.meta->PreparePut({40}, {1}, &later, &dl).code() == StatusCode::kOutOfMemory);
  CHECK(f.meta->AbortPut(c).ok());  // pending reference was the last owner
  CHECK(f.Put({40}, {64}).entries[0].address == base);
}

void PinsAndStableSnapshots() {
  Fixture f(1, 16384);
  const KeyId anchor = std::numeric_limits<KeyId>::max();
  f.Put({anchor}, {1});
  LockContext lock;
  CHECK(f.meta->LockKeys({anchor, anchor}, MetaManagerLockMode::kStableReadAllowAppend, &lock).ok());
  std::vector<KeyId> keys;
  for (KeyId k = 0; k < 4096; ++k) keys.push_back(k);
  f.Put(keys, std::vector<std::uint64_t>(keys.size(), 1));  // forces map rehash
  FlatAccessDesc view;
  CHECK(f.meta->BatchLookup({anchor, anchor}, lock, &view).ok());
  CHECK(view.addresses == std::vector<Address>({base, base}));
  CHECK(f.meta->BatchFree({0,anchor}).code() == StatusCode::kBusy);
  CHECK(f.Lookup(0) == base+1);  // failed batch Free is atomic
  LockContext missing;
  CHECK(f.meta->LockKeys({0,999999}, MetaManagerLockMode::kStableReadAllowAppend, &missing).code() == StatusCode::kNotFound);
  CHECK(f.meta->BatchFree({0}).ok());  // failed lock did not leave pins
  CHECK(f.meta->BatchFree({1,1}).code() == StatusCode::kInvalidArgument);
  CHECK(f.Lookup(1) == base+2);
  CHECK(f.meta->UnlockKeys(lock).ok());
  CHECK(f.meta->UnlockKeys(lock).code() == StatusCode::kNotFound);
  CHECK(f.meta->BatchLookup({anchor,anchor}, lock, &view).code() == StatusCode::kNotFound);
  CHECK(f.meta->BatchFree({anchor}).ok());
  CHECK(f.Put({anchor}, {1}).entries[0].address == base+4097);  // Key reused, hole not reused
}

void InvalidAndDefaultMode() {
  auto append = CreateInMemoryMetaManager(AllocationMode::kAppendOnly);
  PutContext c; PutAccessDesc d;
  CHECK(append->PreparePut({1}, {1}, &c, &d).code() == StatusCode::kNotInitialized);
  CHECK(append->RegisterRemoteRegion({base,65,64,DeviceKind::kCpu}).code() == StatusCode::kInvalidArgument);
  CHECK(append->RegisterRemoteRegion({UINT64_MAX,64,64,DeviceKind::kCpu}).code() == StatusCode::kInvalidArgument);
  CHECK(append->RegisterRemoteRegion({base,64,64,DeviceKind::kCpu}).ok());
  CHECK(append->RegisterRemoteRegion({base,64,64,DeviceKind::kCpu}).code() == StatusCode::kAlreadyExists);
  CHECK(append->PreparePut({1,1}, {1,1}, &c, &d).code() == StatusCode::kInvalidArgument);
  CHECK(append->PreparePut({1}, {65}, &c, &d).code() == StatusCode::kInvalidArgument);
  CHECK(append->PreparePut({1}, {0}, &c, &d).code() == StatusCode::kInvalidArgument);
  CHECK(append->PreparePut({1}, {1}, &c, &d).ok());
  CHECK(append->AbortPut(c).ok());
  for (bool explicit_mode : {false, true}) {
    auto fixed = explicit_mode ? CreateInMemoryMetaManager(AllocationMode::kFixed) : CreateInMemoryMetaManager();
    CHECK(fixed->RegisterRemoteRegion({base,128,64,DeviceKind::kCpu}).ok());
    CHECK(fixed->PreparePut({1}, {1}, &c, &d).ok() && d.entries[0].address == base+64);
    CHECK(fixed->AbortPut(c).ok());
    CHECK(fixed->PreparePut({2}, {1}, &c, &d).code() == StatusCode::kInvalidArgument);
  }
}

void AllocationFailureRollback() {
  int failures = 0;
  for (long countdown = 0; countdown < 40; ++countdown) {
    Fixture f(2,16);
    f.Put({10},{4});
    const std::vector<KeyId> keys{20,30};
    const std::vector<std::uint64_t> lengths{10,16};
    PutContext c; PutAccessDesc d;
    fail_after = countdown;
    const auto status = f.meta->PreparePut(keys, lengths, &c, &d);
    fail_after = -1;
    if (status.ok()) { CHECK(f.meta->AbortPut(c).ok()); continue; }
    ++failures;
    CHECK(status.code() == StatusCode::kOutOfMemory && c.id == 0 && d.entries.empty());
    f.Missing(20); f.Missing(30);
    auto retry = f.Put(keys,lengths);
    CHECK(retry.entries[0].address == base+4 && retry.entries[1].address == base+16);
    CHECK(f.Lookup(10) == base);
  }
  CHECK(failures > 5);
}

int main() {
  PackingAndRecycling(); FailedPrepareIsAtomic(); AbortPreservesLaterReservations();
  PinsAndStableSnapshots(); InvalidAndDefaultMode(); AllocationFailureRollback();
  std::cout << "APPEND_ONLY_META_PASS: packing, rollover, whole-slot recycling, atomic failure, pending/abort, pins, rehash and allocation fault injection\n";
}
