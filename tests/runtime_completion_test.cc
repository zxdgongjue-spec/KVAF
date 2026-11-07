#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "meta_manager.h"
#include "runtime.h"

namespace {
void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

class Allocator final : public kvaf::DdrMemoryAllocator {
 public:
  kvaf::Status Allocate(const kvaf::RuntimeInitRequest& request,
                       kvaf::RemoteRegionDesc* region) override {
    storage.resize(request.ddr_bytes);
    *region = {reinterpret_cast<kvaf::Address>(storage.data()), request.ddr_bytes,
               request.slot_size, kvaf::DeviceKind::kCpu};
    return kvaf::Status::OK();
  }
  kvaf::Status Release(const kvaf::RemoteRegionDesc&) override {
    ++releases;
    if (fail_release) return kvaf::Status::Internal("unmap failed");
    return kvaf::Status::OK();
  }
  bool fail_release{false};
  int releases{0};
  std::vector<std::uint8_t> storage;
};

class Operator final : public kvaf::AccessOperator {
 public:
  kvaf::Status Initialize(const kvaf::RemoteRegionDesc&) override { return kvaf::Status::OK(); }
  kvaf::Status Shutdown() override { ++shutdowns; return kvaf::Status::OK(); }
  kvaf::Status ExecuteBatchGet(const kvaf::BatchGetExecRequest&) override { return kvaf::Status::OK(); }
  kvaf::Status ExecuteBatchPut(const kvaf::BatchPutExecRequest&) override {
    ++writes;
    return result;
  }
  kvaf::Status result;
  int writes{0};
  int shutdowns{0};
};

void TestPutCompletion(bool unknown) {
  Allocator allocator;
  Operator op;
  auto metadata = kvaf::CreateInMemoryMetaManager();
  kvaf::RuntimeCore core(*metadata, op, allocator);
  Require(core.Initialize({4096, 256, kvaf::DeviceKind::kCpu}).ok(), "initialize");
  std::uint8_t input[8]{};
  const kvaf::BatchPutRequest put{{0}, {input, sizeof(input), kvaf::DeviceKind::kCpu}, {0}, {8}};
  op.result = unknown ? kvaf::Status::CompletionUnknown("submitted, synchronization failed")
                      : kvaf::Status::InvalidArgument("not submitted");
  Require(!core.BatchPut(put).ok(), "failed Put must not commit");
  if (unknown) {
    kvaf::PutContext context;
    kvaf::PutAccessDesc destinations;
    Require(!metadata->PreparePut({0}, {8}, &context, &destinations).ok(), "unknown Put must retain slot");
    Require(core.BatchPut(put).code() == kvaf::StatusCode::kCompletionUnknown, "reject subsequent Put");
    Require(core.BatchFree({{0}}).code() == kvaf::StatusCode::kCompletionUnknown, "reject free");
    kvaf::AccessTableHandle table;
    Require(core.PrepareAccessTable({{0}}, &table).code() == kvaf::StatusCode::kCompletionUnknown,
            "reject reads from quarantined Runtime");
    Require(core.Shutdown().code() == kvaf::StatusCode::kCompletionUnknown, "retain DDR on shutdown");
    Require(allocator.releases == 0 && op.writes == 1, "no release or new device work");
  } else {
    op.result = kvaf::Status::OK();
    Require(core.BatchPut(put).ok(), "unsubmitted failure must release its reservation");
    kvaf::AccessTableHandle table;
    Require(core.PrepareAccessTable({{0}}, &table).ok(), "read committed key");
    kvaf::AccessTableView view;
    Require(core.GetAccessTableView(table, &view).ok() && view.size == 1 && view.valid[0], "visible only after completion");
    Require(core.ReleaseAccessTable(table).ok(), "release read lock");
    Require(core.BatchFree({{0}}).ok(), "free committed key");
    Require(core.Shutdown().ok() && allocator.releases == 1, "normal teardown");
  }
}

void TestPartialTeardown() {
  Allocator allocator;
  Operator op;
  auto metadata = kvaf::CreateInMemoryMetaManager();
  kvaf::RuntimeCore core(*metadata, op, allocator);
  Require(core.Initialize({4096, 256, kvaf::DeviceKind::kCpu}).ok(), "initialize cleanup test");
  allocator.fail_release = true;
  Require(!core.Shutdown().ok() && !core.initialized() && core.needs_cleanup(), "retain incomplete cleanup state");
  Require(core.BatchFree({{0}}).code() == kvaf::StatusCode::kNotInitialized, "reject work during cleanup");
  allocator.fail_release = false;
  Require(core.Shutdown().ok(), "retry arena release");
  Require(op.shutdowns == 2 && allocator.releases == 2, "retry idempotent operator cleanup and retained allocation");
}

void TestFusedReadLease() {
  Allocator allocator;
  Operator op;
  auto metadata = kvaf::CreateInMemoryMetaManager();
  kvaf::RuntimeCore core(*metadata, op, allocator);
  Require(core.Initialize({4096, 256, kvaf::DeviceKind::kCpu}).ok(), "initialize fused lease test");
  std::uint8_t input[16]{};
  Require(core.BatchPut({{0, 1}, {input, sizeof(input), kvaf::DeviceKind::kCpu},
                          {0, 8}, {8, 8}}).ok(), "publish keys before fusion");
  kvaf::AccessTableHandle table;
  Require(core.PrepareAccessTable({{0}}, &table).ok(), "prepare pinned table");
  int fences = 0;
  Require(core.AcquireFusedReadLease([&] { ++fences; return kvaf::Status::OK(); }).ok(),
          "acquire external reader");
  Require(core.AcquireFusedReadLease([] { return kvaf::Status::OK(); }).code() ==
              kvaf::StatusCode::kBusy, "reject duplicate lease");
  Require(!core.RetainFusedAccessTable(table, 256).ok(), "validate table block lengths");
  Require(core.RetainFusedAccessTable(table, 8).ok(), "retain optional table");
  Require(core.ReleaseAccessTable(table).code() == kvaf::StatusCode::kBusy,
          "fused consumer preserves original table metadata pin");
  Require(core.BatchFree({{0}}).code() == kvaf::StatusCode::kBusy && fences == 1,
          "registered keys stay pinned across a request retirement fence");
  Require(core.BatchFree({{1}}).ok() && fences == 2, "unrelated retired keys can be freed");
  Require(core.BatchPut({{2}, {input, sizeof(input), kvaf::DeviceKind::kCpu}, {0}, {8}}).ok(),
          "append Put remains available during a long-lived fused lease");
  Require(fences == 2, "append has no external-reader fence");
  Require(core.Shutdown().code() == kvaf::StatusCode::kBusy && op.shutdowns == 0 &&
              allocator.releases == 0, "lease prevents arena teardown");
  Require(core.ReleaseFusedReadLease().code() == kvaf::StatusCode::kBusy,
          "cannot drop lease before its extra table readers");
  Require(core.FenceFusedReads().ok() && fences == 3, "explicit release fence");
  Require(core.ReleaseFusedAccessTable(table).ok(), "remove extra reader");
  Require(core.ReleaseFusedReadLease().ok(), "release external reader");
  Require(core.ReleaseAccessTable(table).ok(), "caller releases its original table");
  Require(core.BatchFree({{0, 2}}).ok() && fences == 3, "ordinary free needs no fused fence");
  Require(core.Shutdown().ok(), "safe shutdown after release");
}

void TestFusedFenceFailure() {
  Allocator allocator;
  Operator op;
  auto metadata = kvaf::CreateInMemoryMetaManager();
  kvaf::RuntimeCore core(*metadata, op, allocator);
  Require(core.Initialize({4096, 256, kvaf::DeviceKind::kCpu}).ok(), "initialize failed fence test");
  std::uint8_t input[8]{};
  Require(core.BatchPut({{0}, {input, sizeof(input), kvaf::DeviceKind::kCpu}, {0}, {8}}).ok(),
          "publish key before failed retirement");
  Require(core.AcquireFusedReadLease([] {
    return kvaf::Status::CompletionUnknown("external kernel synchronization failed");
  }).ok(), "acquire failing external reader");
  Require(core.BatchFree({{0}}).code() == kvaf::StatusCode::kCompletionUnknown,
          "failed fence prevents key reuse");
  kvaf::PutContext pending;
  kvaf::PutAccessDesc destinations;
  Require(metadata->PreparePut({0}, {8}, &pending, &destinations).code() ==
              kvaf::StatusCode::kAlreadyExists, "metadata retains the key after failed free");
  Require(core.ReleaseFusedReadLease().code() == kvaf::StatusCode::kCompletionUnknown,
          "unknown completion cannot release the lease");
  Require(core.Shutdown().code() == kvaf::StatusCode::kCompletionUnknown &&
              op.shutdowns == 0 && allocator.releases == 0, "retain arena and operator on failure");
}
}  // namespace

int main() {
  TestPutCompletion(false);
  TestPutCompletion(true);
  TestPartialTeardown();
  TestFusedReadLease();
  TestFusedFenceFailure();
  std::cout << "PASS: Runtime Put completion and slot lifetime\n";
}
