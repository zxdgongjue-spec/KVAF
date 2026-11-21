#include "runtime.h"
#include <cstring>
#include <iostream>
#include <vector>
#include <cstdlib>

#define CHECK(x) do { if (!(x)) { std::cerr << "line " << __LINE__ << ": " << #x << '\n'; std::exit(1); } } while (false)
using namespace kvaf;

namespace {
struct Operator : AccessOperator {
  bool supported = true;
  int calls = 0;
  Status Initialize(const RemoteRegionDesc&) override { return Status::OK(); }
  Status Shutdown() override { return Status::OK(); }
  Status ExecuteBatchGet(const BatchGetExecRequest&) override { return Status::OK(); }
  Status ExecuteBatchPut(const BatchPutExecRequest& e) override {
    for (std::size_t i = 0; i < e.request->keys.size(); ++i)
      std::memcpy(reinterpret_cast<void*>(e.put_desc->entries[i].address),
          static_cast<const uint8_t*>(e.request->input.data) + e.request->input_offsets[i],
          e.request->lengths[i]);
    return Status::OK();
  }
  Status ExecuteBatchGetRanges(const BatchGetRangesExecRequest& e) override {
    if (!supported) return AccessOperator::ExecuteBatchGetRanges(e);
    ++calls;
    const auto& r = *e.request;
    const auto count = *static_cast<const int32_t*>(r.request_count.data);
    const auto* indices = static_cast<const int32_t*>(r.indices.data);
    const auto* offsets = static_cast<const int64_t*>(r.source_offsets.data);
    const auto* lengths = static_cast<const int64_t*>(r.lengths.data);
    const auto* targets = static_cast<const int64_t*>(r.output_offsets.data);
    for (int i = 0; i < count; ++i)
      std::memcpy(static_cast<uint8_t*>(r.output.data) + targets[i],
          reinterpret_cast<const uint8_t*>(e.host_access_table.addresses[indices[i]]) + offsets[i], lengths[i]);
    return Status::OK();
  }
};
template <class T> ConstBufferView Buffer(const std::vector<T>& x) {
  return {x.data(), x.size() * sizeof(T), DeviceKind::kCpu};
}
}

int main() {
  auto meta = CreateInMemoryMetaManager();
  auto allocator = CreateHostDdrMemoryAllocator();
  Operator op;
  RuntimeCore runtime(*meta, op, *allocator);
  CHECK(runtime.Initialize({8192, 4096, DeviceKind::kCpu}).ok());
  std::vector<uint8_t> oversized(4097);
  CHECK(runtime.BatchPut({{1}, Buffer(oversized), {0}, {oversized.size()}}).code() ==
        StatusCode::kInvalidArgument);
  std::vector<uint8_t> packed(3648);
  for (std::size_t i = 0; i < packed.size(); ++i) packed[i] = (i * 31) % 251;
  CHECK(runtime.BatchPut({{0}, Buffer(packed), {0}, {packed.size()}}).ok());
  AccessTableHandle table;
  CHECK(runtime.PrepareAccessTable({{0}}, &table).ok());
  CHECK(runtime.BatchFree({{0}}).code() == StatusCode::kBusy);
  std::vector<int32_t> indices{0, 0, -1};
  std::vector<int64_t> starts{656, 0, -1}, lengths{1968, 656, -1}, targets{0, 1968, -1};
  int32_t count = 2;
  std::vector<uint8_t> output(3000, 0xa5);
  BatchGetRangesRequest request{table, Buffer(indices), Buffer(starts), Buffer(lengths),
      {&count, sizeof(count), DeviceKind::kCpu}, Buffer(targets),
      {output.data(), output.size(), DeviceKind::kCpu}, {}};
  CHECK(runtime.BatchGetRanges(request).ok());
  CHECK(std::memcmp(output.data(), packed.data() + 656, 1968) == 0);
  CHECK(std::memcmp(output.data() + 1968, packed.data(), 656) == 0);
  CHECK(output[2624] == 0xa5);
  count = 0;
  const auto before = output;
  CHECK(runtime.BatchGetRanges(request).ok() && output == before);
  count = 4;
  CHECK(runtime.BatchGetRanges(request).code() == StatusCode::kInvalidArgument);
  count = 2;
  lengths[0] = 4000;
  CHECK(runtime.BatchGetRanges(request).code() == StatusCode::kInvalidArgument);
  lengths[0] = 1968;
  targets[0] = 2000;
  CHECK(runtime.BatchGetRanges(request).code() == StatusCode::kInvalidArgument);
  targets[0] = 0;
  const auto bytes = request.lengths.bytes;
  request.lengths.bytes -= sizeof(int64_t);
  CHECK(runtime.BatchGetRanges(request).code() == StatusCode::kInvalidArgument);
  request.lengths.bytes = bytes;
  // The same nonuniform range request layout is valid for device dispatch;
  // missing actual object lengths must not fall back to fixed token_bytes.
  auto npu_request = request;
  npu_request.indices.device = npu_request.source_offsets.device =
      npu_request.lengths.device = npu_request.request_count.device =
      npu_request.output_offsets.device = npu_request.output.device = DeviceKind::kNpu;
  AccessTableView host_view;
  CHECK(runtime.GetAccessTableView(table, &host_view).ok());
  npu_request.device_access_table = {
      {host_view.addresses, sizeof(Address), DeviceKind::kNpu},
      {host_view.valid, 1, DeviceKind::kNpu}, {}};
  CHECK(runtime.BatchGetRanges(npu_request).code() == StatusCode::kInvalidArgument);
  npu_request.device_access_table.lengths = {host_view.lengths, sizeof(uint64_t), DeviceKind::kNpu};
  CHECK(runtime.BatchGetRanges(npu_request).ok());
  op.supported = false;
  CHECK(runtime.BatchGetRanges(request).code() == StatusCode::kNotImplemented);
  op.supported = true;
  CHECK(runtime.ReleaseAccessTable(table).ok());
  CHECK(runtime.BatchGetRanges(request).code() == StatusCode::kNotFound);
  CHECK(runtime.BatchFree({{0}}).ok());
  CHECK(runtime.BatchPut({{0}, Buffer(packed), {0}, {packed.size()}}).ok());
  CHECK(runtime.Shutdown().ok());
  std::cout << "RANGE_GET_RUNTIME_PASS: packed objects, partial prefix, bounds, pin/free/reuse and unsupported backend\n";
}
