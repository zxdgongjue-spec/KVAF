#include "runtime.h"
#include <cassert>
#include <cstdint>
#include <iostream>

namespace {
struct RecordingOperator final : kvaf::AccessOperator {
  int calls{};
  bool fail{}, trusted{}, synchronized{};
  const kvaf::BatchGetRequest* request{};
  void* stream{};
  kvaf::Status Initialize(const kvaf::RemoteRegionDesc&) override { return kvaf::Status::OK(); }
  kvaf::Status Shutdown() override { return kvaf::Status::OK(); }
  kvaf::Status ExecuteBatchGet(const kvaf::BatchGetExecRequest& exec) override {
    ++calls;
    request = exec.request;
    trusted = exec.execution.trusted_npu_request;
    synchronized = exec.execution.synchronize;
    stream = exec.execution.stream;
    assert(!exec.host_access_table.addresses);
    return fail ? kvaf::Status::Internal("transport failure") : kvaf::Status::OK();
  }
  kvaf::Status ExecuteBatchPut(const kvaf::BatchPutExecRequest&) override {
    return kvaf::Status::NotImplemented("unused");
  }
};
}

int main() {
  using namespace kvaf;
  auto meta = CreateInMemoryMetaManager();
  auto allocator = CreateHostDdrMemoryAllocator();
  RecordingOperator op;
  RuntimeCore runtime(*meta, op, *allocator);
  std::uint64_t address = 0x1000;
  std::uint8_t valid = 1, output[64]{};
  std::int32_t index = 0, count = 1;
  BatchGetRequest request;
  request.indices = request.offsets = request.output_indices = {&index, 4, DeviceKind::kNpu};
  request.miss_count = {&count, 4, DeviceKind::kNpu};
  request.output = {output, sizeof(output), DeviceKind::kNpu};
  request.output_block_bytes = 64; request.tokens_per_block = 1;
  request.device_access_table = {{&address, 8, DeviceKind::kNpu}, {&valid, 1, DeviceKind::kNpu}};
  BatchGetExecutionOptions execution{reinterpret_cast<void*>(1), true, true};
  assert(runtime.BatchGet(request, execution).code() == StatusCode::kNotInitialized);
  assert(!op.calls);
  assert(runtime.Initialize({4096, 64, DeviceKind::kCpu}).ok());
  execution.trusted_npu_request = false;
  assert(runtime.BatchGet(request, execution).ok());
  assert(op.calls == 1 && !op.trusted);
  // The caller's live device table is sufficient on the trusted path; no Host
  // handle lookup/validation or CPU read of the NPU miss_count is needed.
  request.access_table.opaque = 98765;
  assert(runtime.BatchGet(request, execution).code() == StatusCode::kNotFound);
  assert(op.calls == 1);
  execution.trusted_npu_request = true;
  assert(runtime.BatchGet(request, execution).ok());
  assert(op.calls == 2 && op.trusted && op.synchronized);
  assert(op.request == &request && op.stream == execution.stream);
  request.indices.device = DeviceKind::kCpu;
  assert(runtime.BatchGet(request, execution).code() == StatusCode::kInvalidArgument);
  request.indices.device = DeviceKind::kNpu;
  const auto stream = execution.stream;
  execution.stream = nullptr;
  assert(runtime.BatchGet(request, execution).code() == StatusCode::kInvalidArgument);
  execution.stream = stream;
  op.fail = true;
  const auto failed = runtime.BatchGet(request, execution);
  assert(failed.code() == StatusCode::kInternal && failed.message() == "transport failure");
  assert(op.calls == 3);
  assert(runtime.Shutdown().ok());
  std::cout << "Runtime BatchGet contract: PASS\n";
}
