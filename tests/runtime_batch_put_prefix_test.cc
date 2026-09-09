#include "access_operator.h"
#include "meta_manager.h"
#include "runtime.h"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

namespace {

class RecordingOperator final : public kvaf::AccessOperator {
 public:
  kvaf::Status Initialize(const kvaf::RemoteRegionDesc&) override {
    initialized_ = true;
    return kvaf::Status::OK();
  }

  kvaf::Status Shutdown() override {
    initialized_ = false;
    return kvaf::Status::OK();
  }

  kvaf::Status ExecuteBatchGet(const kvaf::BatchGetExecRequest&) override {
    return kvaf::Status::NotImplemented("unused by compact-prefix test");
  }

  kvaf::Status ExecuteBatchPut(
      const kvaf::BatchPutExecRequest& request) override {
    if (!initialized_ || request.plan == nullptr || request.put_desc == nullptr) {
      return kvaf::Status::InvalidArgument("invalid recording BatchPut request");
    }
    ++put_calls;
    keys = request.plan->flat_keys;
    offsets = request.plan->offsets;
    lengths = request.plan->lengths;
    allocation_count = request.put_desc->entries.size();
    return kvaf::Status::OK();
  }

  std::size_t put_calls{0};
  std::size_t allocation_count{0};
  std::vector<kvaf::KeyId> keys;
  std::vector<std::uint64_t> offsets;
  std::vector<std::uint64_t> lengths;

 private:
  bool initialized_{false};
};

bool Expect(bool condition, const char* message) {
  if (condition) return true;
  std::fprintf(stderr, "FAILED: %s\n", message);
  return false;
}

}  // namespace

int main() {
  std::unique_ptr<kvaf::MetaManager> meta = kvaf::CreateInMemoryMetaManager();
  std::unique_ptr<kvaf::DdrMemoryAllocator> allocator =
      kvaf::CreateHostDdrMemoryAllocator();
  RecordingOperator op;
  kvaf::RuntimeCore runtime(*meta, op, *allocator);
  if (!Expect(runtime.Initialize({128U, 32U, kvaf::DeviceKind::kCpu}).ok(),
              "Runtime initialize")) {
    return 1;
  }

  std::uint8_t source[4] = {0x11U, 0x22U, 0x33U, 0x44U};
  kvaf::BatchPutRequest prefix;
  prefix.groups.push_back(kvaf::BatchPutGroup{
      {201U, 202U, 201U, 203U},
      {source, sizeof(source), kvaf::DeviceKind::kCpu},
      {0U, 1U, 999U, 999U},
      {1U, 1U, 0U, 0U}});
  prefix.miss_count = 2U;
  if (!Expect(runtime.BatchPut(prefix).ok(), "compact-prefix BatchPut")) {
    return 1;
  }
  if (!Expect(op.put_calls == 1U && op.allocation_count == 2U,
              "only active rows reach the operator") ||
      !Expect(op.keys == std::vector<kvaf::KeyId>({201U, 202U}),
              "operator receives the active key prefix") ||
      !Expect(op.offsets == std::vector<std::uint64_t>({0U, 1U}) &&
                  op.lengths == std::vector<std::uint64_t>({1U, 1U}),
              "operator receives the active descriptor prefix")) {
    return 1;
  }

  kvaf::AccessTableHandle active_table;
  if (!Expect(runtime.PrepareAccessTable({{201U, 202U}}, &active_table).ok(),
              "active keys are committed") ||
      !Expect(runtime.ReleaseAccessTable(active_table).ok(),
              "release active table")) {
    return 1;
  }
  kvaf::AccessTableHandle inactive_table;
  const kvaf::Status inactive =
      runtime.PrepareAccessTable({{203U}}, &inactive_table);
  if (!Expect(inactive.code() == kvaf::StatusCode::kNotFound,
              "inactive suffix key is not committed")) {
    return 1;
  }

  kvaf::BatchPutRequest empty;
  empty.groups.push_back(kvaf::BatchPutGroup{
      {301U}, {source, sizeof(source), kvaf::DeviceKind::kCpu}, {999U}, {0U}});
  empty.miss_count = 0U;
  if (!Expect(runtime.BatchPut(empty).ok(), "zero miss_count is a no-op") ||
      !Expect(op.put_calls == 1U, "zero miss_count does not call operator")) {
    return 1;
  }

  kvaf::BatchPutRequest oversized;
  oversized.groups.push_back(kvaf::BatchPutGroup{
      {401U}, {source, sizeof(source), kvaf::DeviceKind::kCpu}, {0U}, {1U}});
  oversized.miss_count = 2U;
  const kvaf::Status invalid = runtime.BatchPut(oversized);
  if (!Expect(invalid.code() == kvaf::StatusCode::kInvalidArgument,
              "miss_count larger than capacity is rejected")) {
    return 1;
  }

  if (!Expect(runtime.Shutdown().ok(), "Runtime shutdown")) return 1;
  std::printf("runtime BatchPut compact-prefix tests passed\n");
  return 0;
}
