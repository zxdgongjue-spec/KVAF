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
  request.access_table.opaque = 123;  // Trusted NPU tables come from device metadata.
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
  CHECK(runtime.BatchGet(request, execution).ok());
  CHECK(op.get_calls == 1 && op.last_request == &request);
  CHECK(op.last_execution.stream == execution.stream && op.last_execution.synchronize);
  CHECK(op.last_execution.profile == &profile && op.last_execution.trusted_npu_request);
  execution.trusted_npu_request = false;
  CHECK(runtime.BatchGet(request, execution).code() == StatusCode::kNotFound);
  CHECK(op.get_calls == 1);
  CHECK(runtime.Shutdown().ok());
}

void CheckScatterGraphControlDispatch() {
  auto meta = CreateInMemoryMetaManager();
  Operator op;
  Allocator allocator;
  RuntimeCore runtime(*meta, op, allocator);
  CHECK(runtime.Initialize({128, 64, DeviceKind::kCpu}).ok());
  BatchGetRequest request;
  BatchGetExecutionOptions execution;
  execution.trusted_npu_request = true;
  execution.stream = reinterpret_cast<void*>(1);
  execution.scatter_graph = ScatterGraphAction::kAttach;
  CHECK(runtime.BatchGet(request, execution).code() == StatusCode::kInvalidArgument);
  // Replay checks and graph cleanup carry no input tensors. The owner checks
  // their graph/stream identity; only capture and payload launches need inputs.
  execution.scatter_graph = ScatterGraphAction::kCheck;
  execution.stream = nullptr;
  CHECK(runtime.BatchGet(request, execution).code() == StatusCode::kInvalidArgument);
  execution.stream = reinterpret_cast<void*>(1);
  CHECK(runtime.BatchGet(request, execution).ok());
  execution.stream = nullptr;
  execution.scatter_graph = ScatterGraphAction::kDetach;
  CHECK(runtime.BatchGet(request, execution).ok());
  CHECK(op.last_execution.scatter_graph == ScatterGraphAction::kDetach);
  CHECK(op.get_calls == 2);
  op.fail_shutdown = true;
  CHECK(!runtime.Shutdown().ok() && runtime.needs_cleanup());
  execution.scatter_graph = ScatterGraphAction::kDetach;
  CHECK(runtime.BatchGet(request, execution).ok());
  execution.scatter_graph = ScatterGraphAction::kCheck;
  CHECK(runtime.BatchGet(request, execution).code() == StatusCode::kNotInitialized);
  op.fail_shutdown = false;
  CHECK(runtime.Shutdown().ok());
}
}  // namespace

int main() {
  CheckDrainFailurePreservesTable();
  CheckReleaseRetry();
  CheckPartialInitializationRetry();
  CheckTrustedNpuDispatch();
  CheckScatterGraphControlDispatch();
  std::cout << "Runtime cleanup ordering and retry tests passed\n";
}
