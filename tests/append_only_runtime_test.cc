#include "runtime.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::cerr << "line " << __LINE__ << ": " << #x << '\n'; std::exit(1); } } while (false)
using namespace kvaf;
struct Operator : AccessOperator {
  bool fail_write = false;
  int fixed_get_calls = 0;
  Status Initialize(const RemoteRegionDesc&) override { return Status::OK(); }
  Status Shutdown() override { return Status::OK(); }
  Status ExecuteBatchGet(const BatchGetExecRequest&) override {
    ++fixed_get_calls;
    return Status::OK();
  }
  Status ExecuteBatchPut(const BatchPutExecRequest& e) override {
    for (std::size_t i = 0; i < e.request->keys.size(); ++i)
      std::memcpy(reinterpret_cast<void*>(e.put_desc->entries[i].address),
                  static_cast<const uint8_t*>(e.request->input.data) + e.request->input_offsets[i],
                  e.request->lengths[i]);
    return fail_write ? Status::Internal("completed write failed") : Status::OK();
  }
  Status ExecuteBatchGetRanges(const BatchGetRangesExecRequest& e) override {
    const auto& r = *e.request;
    const auto n = *static_cast<const int32_t*>(r.request_count.data);
    for (int i = 0; i < n; ++i) {
      const auto key_index = static_cast<const int32_t*>(r.indices.data)[i];
      std::memcpy(static_cast<uint8_t*>(r.output.data) + static_cast<const int64_t*>(r.output_offsets.data)[i],
          reinterpret_cast<const uint8_t*>(e.host_access_table.addresses[key_index]) +
              static_cast<const int64_t*>(r.source_offsets.data)[i],
          static_cast<const int64_t*>(r.lengths.data)[i]);
    }
    return Status::OK();
  }
};
template<class T> ConstBufferView Buffer(const std::vector<T>& data) {
  return {data.data(), data.size()*sizeof(T), DeviceKind::kCpu};
}

int main() {
  auto meta = CreateInMemoryMetaManager(AllocationMode::kAppendOnly);
  auto allocator = CreateHostDdrMemoryAllocator();
  Operator op;
  RuntimeCore runtime(*meta, op, *allocator);
  CHECK(runtime.Initialize({128,64,DeviceKind::kCpu}).ok());
  const auto base = runtime.remote_region().base;
  std::vector<uint8_t> source(74);
  for (std::size_t i = 0; i < source.size(); ++i) source[i] = i*17;
  CHECK(runtime.BatchPut({{100,200,300}, Buffer(source), {0,20,50}, {20,30,24}}).ok());
  AccessTableHandle h;
  CHECK(runtime.PrepareAccessTable({{200,100}}, &h).ok());
  AccessTableView view;
  CHECK(runtime.GetAccessTableView(h, &view).ok());
  CHECK(view.addresses[0] == base+20 && view.addresses[1] == base);
  CHECK(view.lengths[0] == 30 && view.lengths[1] == 20);
  std::vector<uint8_t> added(8, 0xbb);
  CHECK(runtime.BatchPut({{400}, Buffer(added), {0}, {8}}).ok());
  CHECK(runtime.BatchFree({{200}}).code() == StatusCode::kBusy);
  std::vector<int32_t> indices{0,1};
  std::vector<int64_t> offsets{0,0}, lengths{30,20}, targets{0,30};
  int32_t count = 2;
  std::vector<uint8_t> output(54, 0xa5);
  BatchGetRangesRequest r{h, Buffer(indices), Buffer(offsets), Buffer(lengths),
      {&count,4,DeviceKind::kCpu}, Buffer(targets), {output.data(),output.size(),DeviceKind::kCpu}, {}};
  CHECK(runtime.BatchGetRanges(r).ok());
  CHECK(!std::memcmp(output.data(), source.data()+20,30));
  CHECK(!std::memcmp(output.data()+30, source.data(),20) && output[50] == 0xa5);

  // Fixed NPU Get validates the entire table, even if the active prefix
  // references only an object with the requested token size.
  int32_t row = 0, active = 1;
  const auto npu = DeviceKind::kNpu;
  const ConstBufferView row_buffer{&row, sizeof(row), npu};
  BatchGetRequest fixed{h, row_buffer, row_buffer, {&active, sizeof(active), npu},
      row_buffer, {output.data(), 30, npu}, 30, 1,
      {{view.addresses, view.size * sizeof(Address), npu}, {view.valid, view.size, npu}, {}}};
  CHECK(runtime.BatchGet(fixed).code() == StatusCode::kInvalidArgument);
  CHECK(op.fixed_get_calls == 0);
  AccessTableHandle fixed_handle;
  CHECK(runtime.PrepareAccessTable({{200}}, &fixed_handle).ok());
  CHECK(runtime.GetAccessTableView(fixed_handle, &view).ok());
  fixed.access_table = fixed_handle;
  fixed.device_access_table = {{view.addresses, sizeof(Address), npu}, {view.valid, 1, npu}, {}};
  CHECK(runtime.BatchGet(fixed).ok());
  CHECK(op.fixed_get_calls == 1);
  CHECK(runtime.ReleaseAccessTable(fixed_handle).ok());

  CHECK(runtime.ReleaseAccessTable(h).ok());
  CHECK(runtime.BatchFree({{200,100}}).ok());
  std::vector<uint8_t> full(64, 0xcc);
  CHECK(runtime.BatchPut({{500}, Buffer(full), {0}, {64}}).ok());
  CHECK(runtime.BatchGetRanges(r).code() == StatusCode::kNotFound);
  CHECK(runtime.BatchFree({{500}}).ok());
  op.fail_write = true;
  CHECK(runtime.BatchPut({{600}, Buffer(added), {0}, {8}}).code() == StatusCode::kInternal);
  CHECK(runtime.PrepareAccessTable({{600}}, &h).code() == StatusCode::kNotFound);
  op.fail_write = false;
  CHECK(runtime.BatchPut({{700}, Buffer(full), {0}, {64}}).ok());
  CHECK(runtime.PrepareAccessTable({{300,400,700}}, &h).ok());
  CHECK(runtime.GetAccessTableView(h, &view).ok());
  CHECK(!std::memcmp(reinterpret_cast<void*>(view.addresses[0]),source.data()+50,24));
  CHECK(!std::memcmp(reinterpret_cast<void*>(view.addresses[1]),added.data(),8));
  CHECK(view.addresses[2] == base);
  CHECK(runtime.ReleaseAccessTable(h).ok());
  CHECK(runtime.Shutdown().ok());
  std::cout << "APPEND_ONLY_RUNTIME_PASS: unchanged Put/range Get APIs, payload, pin, reuse and failed-write Abort\n";
}
