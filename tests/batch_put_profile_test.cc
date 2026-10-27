#include "runtime.h"

#include <array>
#include <cstdlib>
#include <iostream>

#define CHECK(condition) do { if (!(condition)) { \
  std::cerr << "line " << __LINE__ << ": " #condition "\n"; std::exit(1); \
} } while (false)

class Operator final : public kvaf::AccessOperator {
 public:
  bool profiled{false}, fail{false};
  kvaf::Status Initialize(const kvaf::RemoteRegionDesc&) override { return kvaf::Status::OK(); }
  kvaf::Status Shutdown() override { return kvaf::Status::OK(); }
  kvaf::Status ExecuteBatchGet(const kvaf::BatchGetExecRequest&) override { return kvaf::Status::OK(); }
  kvaf::Status ExecuteBatchPut(const kvaf::BatchPutExecRequest& exec) override {
    profiled = exec.profile != nullptr;
    return fail ? kvaf::Status::Internal("injected failure") : kvaf::Status::OK();
  }
};

int main() {
  auto meta = kvaf::CreateInMemoryMetaManager();
  auto allocator = kvaf::CreateHostDdrMemoryAllocator();
  Operator op;
  kvaf::RuntimeCore runtime(*meta, op, *allocator);
  kvaf::BatchPutProfile profile;
  CHECK(!runtime.SetBatchPutProfiling(true).ok());
  CHECK(runtime.Initialize({256, 64, kvaf::DeviceKind::kCpu}).ok());
  std::array<std::uint8_t, 100> source{};
  kvaf::BatchPutRequest request{{0, 1, 2},
      {source.data(), source.size(), kvaf::DeviceKind::kCpu}, {0, 16, 48}, {7, 19, 31}};
  CHECK(runtime.BatchPut(request).ok());
  CHECK(!op.profiled);
  CHECK(runtime.BatchFree({request.keys}).ok());
  CHECK(runtime.SetBatchPutProfiling(true).ok());
  CHECK(!runtime.CopyBatchPutProfile(&profile).ok());
  CHECK(runtime.BatchPut(request).ok());
  CHECK(op.profiled);
  CHECK(runtime.CopyBatchPutProfile(&profile).ok());
  CHECK(profile.valid && profile.row_count == 3);
  CHECK(profile.payload_bytes == 57);  // Useful bytes, not the padded 100-B input.
  CHECK(profile.runtime_total_ns >= profile.prepare_put_ns + profile.commit_put_ns);
  CHECK(runtime.BatchFree({request.keys}).ok());
  op.fail = true;
  CHECK(!runtime.BatchPut(request).ok());
  CHECK(!runtime.CopyBatchPutProfile(&profile).ok());  // No stale success snapshot.
  op.fail = false;
  CHECK(runtime.BatchPut(request).ok());  // Failed put was aborted.
  CHECK(runtime.BatchFree({request.keys}).ok());
  CHECK(runtime.SetBatchPutProfiling(false).ok());
  CHECK(!runtime.CopyBatchPutProfile(&profile).ok());
  CHECK(runtime.BatchPut(request).ok() && !op.profiled);
  CHECK(runtime.Shutdown().ok());
  std::cout << "BatchPut profiling lifecycle and useful byte accounting PASS\n";
}
