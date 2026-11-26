#include "runtime.h"
#include "batch_get_profile.h"

#include <cstdlib>
#include <iostream>

#define CHECK(condition) do { if (!(condition)) { \
  std::cerr << "line " << __LINE__ << ": " #condition "\n"; std::exit(1); \
} } while (false)

namespace {
using namespace kvaf;

struct Operator final : AccessOperator {
  bool fail_initialize{false}, fail_shutdown{false};
  bool tracks_get_completion{}, fail_get_drain{};
  int shutdown_calls{0};
  int get_calls{0};
  const BatchGetRequest* last_request{};
  BatchGetExecutionOptions last_execution;
  Status Initialize(const RemoteRegionDesc&) override {
    return fail_initialize ? Status::Internal("initialize failed") : Status::OK();
  }
  Status Shutdown() override {
    ++shutdown_calls;
    return fail_shutdown ? Status::Internal("drain failed") : Status::OK();
  }
  Status ExecuteBatchGet(const BatchGetExecRequest& exec) override {
    ++get_calls;
    last_request = exec.request;
    last_execution = exec.execution;
    return Status::OK();
  }
  bool TracksGetCompletion() const override { return tracks_get_completion; }
  Status DrainGet() override {
    return fail_get_drain ? Status::Internal("Get drain failed") : Status::OK();
  }
  Status ExecuteBatchPut(const BatchPutExecRequest&) override { return Status::OK(); }
};

struct Allocator final : DdrMemoryAllocator {
  bool fail_release{false}, allocated{false};
  int release_calls{0};
  Status Allocate(const RuntimeInitRequest& request, RemoteRegionDesc* region) override {
    CHECK(!allocated);
    allocated = true;
    *region = {4096, request.ddr_bytes, request.slot_size, DeviceKind::kCpu};
    return Status::OK();
  }
  Status Release(const RemoteRegionDesc& region) override {
    CHECK(allocated && region.base == 4096);
    ++release_calls;
    if (fail_release) return Status::Internal("release failed");
    allocated = false;
    return Status::OK();
  }
};

void CheckDrainFailurePreservesTable() {
  auto meta = CreateInMemoryMetaManager();
  Operator op;
  Allocator allocator;
  RuntimeCore runtime(*meta, op, allocator);
  CHECK(runtime.Initialize({128, 64, DeviceKind::kCpu}).ok());
  PutContext put;
  PutAccessDesc desc;
  CHECK(meta->PreparePut({0}, {64}, &put, &desc).ok());
  CHECK(meta->CommitPut(put).ok());
  AccessTableHandle table;
  CHECK(runtime.PrepareAccessTable({{0}}, &table).ok());

  op.fail_shutdown = true;
  CHECK(!runtime.Shutdown().ok());
  CHECK(!runtime.initialized() && runtime.needs_cleanup());
  CHECK(op.shutdown_calls == 1 && allocator.release_calls == 0);
  CHECK(!meta->BatchFree({0}).ok());  // The outstanding transfer still owns its table.
  CHECK(!runtime.Initialize({128, 64, DeviceKind::kCpu}).ok());

  op.fail_shutdown = false;
  CHECK(runtime.Shutdown().ok());
  CHECK(op.shutdown_calls == 2 && allocator.release_calls == 1);
  CHECK(meta->BatchFree({0}).ok());
  CHECK(!runtime.needs_cleanup() && !allocator.allocated);
}

void CheckReleaseRetry() {
  auto meta = CreateInMemoryMetaManager();
  Operator op;
  Allocator allocator;
  RuntimeCore runtime(*meta, op, allocator);
  CHECK(runtime.Initialize({128, 64, DeviceKind::kCpu}).ok());
  allocator.fail_release = true;
  CHECK(!runtime.Shutdown().ok());
  CHECK(!runtime.initialized() && runtime.needs_cleanup());
  CHECK(runtime.remote_region().base == 4096 && allocator.allocated);
  CHECK(!runtime.Initialize({128, 64, DeviceKind::kCpu}).ok());
  allocator.fail_release = false;
  CHECK(runtime.Shutdown().ok());
  CHECK(allocator.release_calls == 2 && !runtime.needs_cleanup());
  CHECK(runtime.remote_region().base == 0);
}

void CheckPartialInitializationRetry() {
  auto meta = CreateInMemoryMetaManager();
  Operator op;
  Allocator allocator;
  {
    RuntimeCore runtime(*meta, op, allocator);
    op.fail_initialize = op.fail_shutdown = true;
    CHECK(!runtime.Initialize({128, 64, DeviceKind::kCpu}).ok());
    CHECK(runtime.needs_cleanup() && !runtime.initialized());
    CHECK(allocator.allocated && allocator.release_calls == 0);
    op.fail_shutdown = false;
    allocator.fail_release = true;
    CHECK(!runtime.Shutdown().ok());
    CHECK(runtime.needs_cleanup() && allocator.allocated);
    allocator.fail_release = false;
    // Destruction also retries cleanup after a failed initialization.
  }
  CHECK(!allocator.allocated && allocator.release_calls == 2);
}

void CheckTrustedNpuDispatch() {
  auto meta = CreateInMemoryMetaManager();
  Operator op;
  Allocator allocator;
  RuntimeCore runtime(*meta, op, allocator);
  BatchGetRequest request;
  request.indices.device = DeviceKind::kNpu;
  BatchGetProfile profile;
  BatchGetExecutionOptions execution;
  execution.stream = reinterpret_cast<void*>(1);
  execution.synchronize = true;
  execution.profile = &profile;
  execution.trusted_npu_request = true;
  CHECK(runtime.BatchGet(request, execution).code() == StatusCode::kNotInitialized);
  CHECK(runtime.Initialize({128, 64, DeviceKind::kCpu}).ok());
  execution.stream = nullptr;
  CHECK(runtime.BatchGet(request, execution).code() == StatusCode::kInvalidArgument);
  execution.stream = reinterpret_cast<void*>(1);
  request.indices.device = DeviceKind::kCpu;
  CHECK(runtime.BatchGet(request, execution).code() == StatusCode::kInvalidArgument);
  CHECK(op.get_calls == 0);
  request.indices.device = DeviceKind::kNpu;
  request.access_table.opaque = 123;
  CHECK(runtime.BatchGet(request, execution).code() == StatusCode::kNotFound);
  request.access_table = {};
  CHECK(runtime.BatchGet(request, execution).ok());
  CHECK(op.get_calls == 1 && op.last_request == &request);
  CHECK(op.last_execution.stream == execution.stream && op.last_execution.synchronize);
  CHECK(op.last_execution.profile == &profile && op.last_execution.trusted_npu_request);
  request.access_table = {};
  execution.graph_lease = 1;
  CHECK(runtime.BatchGet(request, execution).code() == StatusCode::kNotFound);
  CHECK(op.get_calls == 1);
  execution.graph_lease = 0;
  request.access_table.opaque = 123;
  execution.trusted_npu_request = false;
  CHECK(runtime.BatchGet(request, execution).code() == StatusCode::kNotFound);
  CHECK(op.get_calls == 1);
  CHECK(runtime.Shutdown().ok());
}

void CheckTrustedNpuPinsSourceUntilDrain() {
  auto meta = CreateInMemoryMetaManager();
  Operator op;
  op.tracks_get_completion = true;
  Allocator allocator;
  RuntimeCore runtime(*meta, op, allocator);
  CHECK(runtime.Initialize({128, 64, DeviceKind::kCpu}).ok());
  PutContext put;
  PutAccessDesc desc;
  CHECK(meta->PreparePut({0}, {64}, &put, &desc).ok());
  CHECK(meta->CommitPut(put).ok());
  BatchGetRequest request;
  CHECK(runtime.PrepareAccessTable({{0}}, &request.access_table).ok());
  request.indices.device = DeviceKind::kNpu;
  BatchGetExecutionOptions execution;
  execution.stream = reinterpret_cast<void*>(1);
  execution.trusted_npu_request = true;
  CHECK(runtime.BatchGet(request, execution).ok());
  CHECK(runtime.ReleaseAccessTable(request.access_table).ok());
  CHECK(!meta->BatchFree({0}).ok());
  op.fail_get_drain = true;
  CHECK(!runtime.DrainGet().ok());
  CHECK(!meta->BatchFree({0}).ok());
  op.fail_get_drain = false;
  CHECK(runtime.DrainGet().ok());
  CHECK(meta->BatchFree({0}).ok());
  CHECK(runtime.Shutdown().ok());
}

}  // namespace

int main() {
  CheckDrainFailurePreservesTable();
  CheckReleaseRetry();
  CheckPartialInitializationRetry();
  CheckTrustedNpuDispatch();
  CheckTrustedNpuPinsSourceUntilDrain();
  std::cout << "Runtime cleanup ordering and retry tests passed\n";
}
