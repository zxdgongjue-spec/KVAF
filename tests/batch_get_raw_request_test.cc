#include "runtime.h"

#include <cstdlib>
#include <iostream>
#include <limits>

#define CHECK(condition) do { if (!(condition)) { \
  std::cerr << "line " << __LINE__ << ": " #condition "\n"; std::exit(1); \
} } while (false)

class RecordingOperator final : public kvaf::AccessOperator {
 public:
  int calls = 0;
  kvaf::BatchGetRequest last{};
  kvaf::Status Initialize(const kvaf::RemoteRegionDesc&) override { return kvaf::Status::OK(); }
  kvaf::Status Shutdown() override { return kvaf::Status::OK(); }
  kvaf::Status ExecuteBatchPut(const kvaf::BatchPutExecRequest&) override { return kvaf::Status::OK(); }
  kvaf::Status ExecuteBatchGet(const kvaf::BatchGetExecRequest& exec) override {
    CHECK(exec.host_access_table.size == 0);
    CHECK(exec.execution.stream == reinterpret_cast<void*>(0x8000));
    CHECK(!exec.execution.synchronize);
    last = *exec.request;
    ++calls;
    return kvaf::Status::OK();
  }
};

int main() {
  using namespace kvaf;
  auto meta = CreateInMemoryMetaManager();
  auto allocator = CreateHostDdrMemoryAllocator();
  RecordingOperator op;
  RuntimeCore runtime(*meta, op, *allocator);
  BatchGetRequest request{};
  // Deliberately inaccessible device pointers: host validation must forward
  // the descriptors without dereferencing or resolving their contents.
  request.remote_addresses = {reinterpret_cast<void*>(0x1000), 24, DeviceKind::kNpu};
  request.local_addresses = {reinterpret_cast<void*>(0x2000), 24, DeviceKind::kNpu};
  request.miss_count = {reinterpret_cast<void*>(0x3000), 4, DeviceKind::kNpu};
  request.output = {reinterpret_cast<void*>(0x4000), 31, DeviceKind::kNpu};
  request.output_block_bytes = 7;  // containing MR need not be a multiple
  const BatchGetExecutionOptions execution{reinterpret_cast<void*>(0x8000), false};
  CHECK(!runtime.BatchGet(request, execution).ok());
  CHECK(runtime.Initialize({4096, 64}).ok());
  CHECK(runtime.BatchGet(request, execution).ok());
  CHECK(op.calls == 1);
  CHECK(op.last.remote_addresses.data == request.remote_addresses.data);
  CHECK(op.last.local_addresses.data == request.local_addresses.data);
  CHECK(op.last.output.bytes == 31 && op.last.output_block_bytes == 7);

  const auto reject = [&](const BatchGetRequest& invalid) {
    const auto status = runtime.BatchGet(invalid, execution);
    CHECK(status.code() == StatusCode::kInvalidArgument);
    CHECK(op.calls == 1);
  };
  auto bad = request; bad.local_addresses = {}; reject(bad);
  bad = request; bad.remote_addresses = {}; reject(bad);
  bad = request; bad.local_addresses.bytes = 16; reject(bad);
  bad = request; bad.local_addresses.bytes = bad.remote_addresses.bytes = 23; reject(bad);
  bad = request; bad.remote_addresses.data = nullptr; reject(bad);
  bad = request; bad.remote_addresses.device = DeviceKind::kCpu; reject(bad);
  bad = request; bad.local_addresses.device = DeviceKind::kCpu; reject(bad);
  bad = request; bad.miss_count.device = DeviceKind::kCpu; reject(bad);
  bad = request; bad.miss_count.bytes = 8; reject(bad);
  bad = request; bad.miss_count.data = nullptr; reject(bad);
  bad = request; bad.output.device = DeviceKind::kCpu; reject(bad);
  bad = request; bad.output.data = nullptr; reject(bad);
  bad = request; bad.output_block_bytes = 0; reject(bad);
  bad = request; bad.output_block_bytes = 32; reject(bad);
  bad = request; bad.access_table.opaque = 123; reject(bad);
  bad = request; bad.indices = request.miss_count; reject(bad);
  bad = request; bad.offsets = request.miss_count; reject(bad);
  bad = request; bad.output_indices = request.miss_count; reject(bad);
  bad = request; bad.device_access_table.addresses = request.remote_addresses; reject(bad);
  bad = request; bad.tokens_per_block = 1; reject(bad);
  bad = request;
  bad.remote_addresses.bytes = bad.local_addresses.bytes =
      (static_cast<std::size_t>(std::numeric_limits<int32_t>::max()) + 1) * sizeof(Address);
  reject(bad);
  bad = request;
  bad.output.bytes = bad.output_block_bytes = uint64_t{1} << 32;
  reject(bad);
  CHECK(runtime.Shutdown().ok());
  std::cout << "Raw BatchGet Runtime forwarding and layout validation passed\n";
}
