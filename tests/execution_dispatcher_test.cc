#include "execution_dispatcher.h"
#include "runtime.h"
#include "cache_region.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <mutex>
#include <sstream>
#include <stdexcept>
using namespace kvaf;
using namespace std::chrono_literals;
#define CHECK(c) do { if (!(c)) throw std::runtime_error(std::string("line ") + std::to_string(__LINE__) + ": " #c); } while (0)
namespace {
constexpr PutExecutionConfig put{Transport::Urma, Executor::Aiv};
constexpr GetExecutionConfig get{Transport::Urma, Executor::Aiv, GetAlgorithm::Direct};
constexpr GetExecutionConfig gather{Transport::Urma, Executor::Aicpu, GetAlgorithm::Gather};
constexpr GetExecutionConfig scatter{Transport::Urma, Executor::Aicpu, GetAlgorithm::Scatter};
ExecutionPolicy Policy() {
  ExecutionPolicy p;
  return p;
}
struct State {
  std::atomic<int> creates{0}, calls{0}, drains{0}, closes{0}, bindings{0}, releases{0};
  bool fail_bind{false}, fail_release{false};
  std::atomic<bool> fail_create{false}, fail_put{false}, fail_drain{false}, fail_close{false};
  std::mutex mutex;
  std::condition_variable changed;
  bool block_create{false}, entered{false};
};
class Provider final : public AccessOperator {
 public:
  explicit Provider(State& s) : s_(s) {}
  Status Initialize(const RemoteRegionDesc&) override {
    ++s_.creates;
    std::unique_lock<std::mutex> lock(s_.mutex);
    s_.entered = true;
    s_.changed.notify_all();
    s_.changed.wait(lock, [&] { return !s_.block_create; });
    return s_.fail_create ? Status::Internal("create") : Status::OK();
  }
  Status Shutdown() override {
    ++s_.closes;
    if (s_.fail_close) return Status::Internal("close");
    return DrainGet();
  }
  Status ExecuteBatchPut(const BatchPutExecRequest& e) override {
    ++s_.calls;
    CHECK(e.execution.config == put);
    if (s_.fail_put) return Status::Internal("uncertain DMA");
    for (std::size_t i = 0; i < e.request->keys.size(); ++i)
      std::memcpy(reinterpret_cast<void*>(e.put_desc->entries[i].address),
          static_cast<const char*>(e.request->input.data) + e.request->input_offsets[i], e.request->lengths[i]);
    return Status::OK();
  }
  Status ExecuteBatchGet(const BatchGetExecRequest& e) override {
    ++s_.calls;
    CHECK(e.execution.config == get || e.execution.config == gather || e.execution.config == scatter);
    if (e.host_timing) {
      e.host_timing->prepare_ns = 11;
      e.host_timing->submit_ns = 12;
      e.host_timing->completion_ns = 13;
      e.host_timing->cleanup_ns = 14;
    }
    request_ = *e.request;
    table_ = e.host_access_table;
    pending_ = true;
    return e.execution.synchronize ? DrainGet() : Status::OK();
  }
  Status PrepareGetOutput(const GetOutputOptions&) override {
    ++s_.bindings;
    return s_.fail_bind ? Status::Internal("bind") : Status::OK();
  }
  Status FinishGetOutput() override {
    ++s_.releases;
    return s_.fail_release ? Status::Internal("release") : DrainGet();
  }
  Status PrepareGetGraph(const GetGraphOptions&) override { return Status::OK(); }
  Status FinishGetGraph(void*, bool) override { return DrainGet(); }
  Status DrainGet() override {
    if (!pending_) return Status::OK();
    ++s_.drains;
    if (s_.fail_drain) return Status::Internal("drain");
    const auto n = *static_cast<const int*>(request_.miss_count.data);
    for (int i = 0; i < n; ++i) {
      const auto index = static_cast<const int*>(request_.indices.data)[i];
      const auto offset = static_cast<const int*>(request_.offsets.data)[i];
      const auto out = static_cast<const int*>(request_.output_indices.data)[i];
      std::memcpy(static_cast<char*>(request_.output.data) + out * request_.output_block_bytes,
          reinterpret_cast<const char*>(table_.addresses[index]) + offset, request_.output_block_bytes);
    }
    pending_ = false;
    return Status::OK();
  }
 private:
  State& s_;
  bool pending_{false};
  BatchGetRequest request_;
  AccessTableView table_;
};
void Register(ExecutionDispatcher& d, State& p, State& g, State& a) {
  CHECK(d.RegisterPut(put, [&] { return std::make_unique<Provider>(p); }).ok());
  CHECK(d.RegisterGet(get, [&] { return std::make_unique<Provider>(g); }).ok());
  CHECK(d.RegisterGet(gather, [&] { return std::make_unique<Provider>(a); }).ok());
}
void Configurations() {
  ExecutionPolicy p;
  CHECK(NormalizeExecutionPolicy(&p).ok());
  CHECK(p.default_put == put && p.default_get == get);
  CHECK(ValidateExecution(TransportMode::Urma, {Transport::Urma, Executor::Aiv, GetAlgorithm::Scatter}).ok());
  CHECK(!ValidateExecution(TransportMode::Ubmem, {Transport::Ubmem, Executor::Aiv, GetAlgorithm::Gather}).ok());
  CHECK(!ValidateExecution(TransportMode::Urma, {Transport::Urma, Executor::Aicpu, GetAlgorithm::Direct}).ok());
  CHECK(!ValidateExecution(TransportMode::Urma, PutExecutionConfig{Transport::Ubmem, Executor::Aiv}).ok());
  PutExecutionConfig c;
  GetExecutionConfig selected;
  CHECK(ResolveExecution(p, std::nullopt, &c).ok() && c == put);
  CHECK(ResolveExecution(p, gather, &selected).ok() && selected == gather);
  const PutExecutionConfig ubmem{Transport::Ubmem, Executor::Aiv};
  CHECK(!ResolveExecution(p, ubmem, &c).ok());
  p.transport_mode = TransportMode::UrmaAndUbmem;
  CHECK(NormalizeExecutionPolicy(&p).ok());
  CHECK(ResolveExecution(p, ubmem, &c).ok() && c == ubmem);
  p = {}; p.transport_mode = TransportMode::Ubmem;
  CHECK(NormalizeExecutionPolicy(&p).ok());
  CHECK(p.default_put == ubmem && p.default_get->transport == Transport::Ubmem);
  p = Policy(); p.default_get = {Transport::Urma, Executor::Aiv, GetAlgorithm::Scatter};
  CHECK(NormalizeExecutionPolicy(&p).ok());  // Reserved providers fail at use, not policy creation.
  p = {}; p.default_get = {Transport::Ubmem, Executor::Aiv, GetAlgorithm::Direct};
  CHECK(!NormalizeExecutionPolicy(&p).ok());
  CHECK(!p.default_put);  // Failed normalization leaves the policy untouched.
  CacheRegionDesc region{1, 2, 100, 64, 8}; Address translated;
  CHECK(region.Translate(120, 8, 1000, &translated).ok() && translated == 1020);
  CHECK(!region.Translate(99, 1, 1000, &translated).ok());
  CHECK(!region.Translate(160, 8, 1000, &translated).ok());
  CHECK(!region.Translate(120, 8, UINT64_MAX - 10, &translated).ok());
  ExecutionDispatcher missing(Policy());
  CHECK(missing.Initialize({4096, 128, 64, DeviceKind::kCpu}).ok());
  CHECK(missing.Warmup(put, {}).code() == StatusCode::kNotImplemented);
  CHECK(missing.Warmup(get, {}).code() == StatusCode::kNotImplemented);
  CHECK(missing.Shutdown().ok());
}
void LazyAndIndependent() {
  State p, g, a;
  ExecutionDispatcher d(Policy()); Register(d, p, g, a);
  CHECK(d.Initialize({4096, 128, 64, DeviceKind::kCpu}).ok());
  CHECK(p.creates == 0 && g.creates == 0 && a.creates == 0);
  g.block_create = true;
  auto first = std::async(std::launch::async, [&] { return d.Warmup(get); });
  {
    std::unique_lock<std::mutex> lock(g.mutex);
    CHECK(g.changed.wait_for(lock, 2s, [&] { return g.entered; }));
  }
  auto second = std::async(std::launch::async, [&] { return d.Warmup(get); });
  auto independent = std::async(std::launch::async, [&] { return d.Warmup(put); });
  CHECK(independent.wait_for(2s) == std::future_status::ready);
  CHECK(independent.get().ok() && p.creates == 1);
  {
    std::lock_guard<std::mutex> lock(g.mutex); g.block_create = false;
  }
  g.changed.notify_all();
  CHECK(first.get().ok() && second.get().ok()); CHECK(g.creates == 1 && a.creates == 0);
  a.fail_create = true;
  CHECK(!d.Warmup(gather).ok()); a.fail_create = false;
  CHECK(!d.Warmup(gather).ok() && a.creates == 1);  // no blind retry
  CHECK(d.Recover(gather).ok() && a.creates == 2);
  a.fail_close = true; CHECK(!d.Shutdown().ok()); a.fail_close = false;
  CHECK(d.Shutdown().ok());
}
void RuntimePinsAndTransactions() {
  State p, g, a;
  auto meta = CreateInMemoryMetaManager();
  auto alloc = CreateHostDdrMemoryAllocator();
  ExecutionDispatcher d(Policy()); Register(d, p, g, a);
  RuntimeCore runtime(*meta, d, *alloc);
  CHECK(runtime.Initialize({128, 16, DeviceKind::kCpu}).ok());
  char input[16] = "initial payload", output[16]{};
  auto write = [&](KeyId key) {
    return runtime.BatchPut({{key}, {input, 16, DeviceKind::kCpu}, {0}, {16}});
  };
  CHECK(write(1).ok()); CHECK(write(1).code() == StatusCode::kAlreadyExists);
  AccessTableHandle table;
  CHECK(runtime.PrepareAccessTable({{1}}, &table).ok());
  int index=0, offset=0, count=1, out=0;
  BatchGetRequest r{table, {&index,4,DeviceKind::kCpu}, {&offset,4,DeviceKind::kCpu},
      {&count,4,DeviceKind::kCpu}, {&out,4,DeviceKind::kCpu}, {output,16,DeviceKind::kCpu},16,1,{}};
  CHECK(runtime.BatchGet(r).ok()); CHECK(g.drains == 0);
  CHECK(write(2).ok()); CHECK(g.drains == 0); // Put never drains Get
  CHECK(runtime.ReleaseAccessTable(table).ok());
  CHECK(runtime.BatchFree({{1}}).code() == StatusCode::kBusy);
  CHECK(runtime.BatchGet(r).code() == StatusCode::kNotFound);
  g.fail_drain = true;
  CHECK(!runtime.DrainGet().ok()); CHECK(runtime.BatchFree({{1}}).code() == StatusCode::kBusy);
  CHECK(write(3).ok()); // Get failure is isolated
  g.fail_drain = false;
  CHECK(runtime.DrainGet().ok()); CHECK(std::memcmp(input, output, 16) == 0);
  CHECK(runtime.BatchFree({{1}}).ok());
  CHECK(runtime.PrepareAccessTable({{2}}, &table).ok()); r.access_table = table;
  BatchGetExecutionOptions execution; execution.config = gather; execution.synchronize = true;
  a.block_create = true;
  auto preparing = std::async(std::launch::async, [&] { return runtime.BatchGet(r, execution); });
  {
    std::unique_lock<std::mutex> lock(a.mutex);
    CHECK(a.changed.wait_for(lock, 2s, [&] { return a.entered; }));
  }
  CHECK(runtime.ReleaseAccessTable(table).ok());
  CHECK(runtime.BatchFree({{2}}).code() == StatusCode::kBusy);
  CHECK(write(6).ok()); // resource preparation does not block the opposite domain
  {
    std::lock_guard<std::mutex> lock(a.mutex); a.block_create = false;
  }
  a.changed.notify_all();
  CHECK(preparing.get().ok()); CHECK(a.creates == 1);
  CHECK(runtime.BatchFree({{2}}).ok());
  p.fail_put = true;
  auto failed = write(4);
  if (failed.code() != StatusCode::kCompletionUnknown) std::cerr << failed.message() << " code=" << int(failed.code()) << "\n";
  CHECK(failed.code() == StatusCode::kCompletionUnknown);
  CHECK(write(5).code() == StatusCode::kBusy);
  PutContext pending; PutAccessDesc desc;
  CHECK(meta->PreparePut({4}, {16}, &pending, &desc).code() == StatusCode::kAlreadyExists);
  p.fail_close = true; CHECK(!runtime.Shutdown().ok()); CHECK(runtime.needs_cleanup());
  p.fail_close = false; CHECK(runtime.Shutdown().ok()); CHECK(!runtime.needs_cleanup());
}
void GraphLeasePinsAndIsolation() {
  State p, g, a;
  ExecutionDispatcher dispatch(Policy());
  Register(dispatch, p, g, a);
  auto meta = CreateInMemoryMetaManager();
  auto allocator = CreateHostDdrMemoryAllocator();
  RuntimeCore runtime(*meta, dispatch, *allocator);
  CHECK(runtime.Initialize({128, 16, DeviceKind::kCpu}).ok());
  char input[16] = "graph payload", output[16]{};
  CHECK(runtime.BatchPut({{1}, {input, 16, DeviceKind::kCpu}, {0}, {16}}).ok());
  AccessTableHandle table;
  CHECK(runtime.PrepareAccessTable({{1}}, &table).ok());
  std::uint64_t lease = 0;
  CHECK(runtime
            .BeginGetGraph(get, {{output, 16, DeviceKind::kCpu}, reinterpret_cast<void*>(1), 1},
                           &lease)
            .ok());
  int index = 0, offset = 0, count = 1, row = 0;
  BatchGetRequest request{table,
                          {&index, 4, DeviceKind::kCpu},
                          {&offset, 4, DeviceKind::kCpu},
                          {&count, 4, DeviceKind::kCpu},
                          {&row, 4, DeviceKind::kCpu},
                          {output, 16, DeviceKind::kCpu},
                          16,
                          1,
                          {}};
  CHECK(runtime.BatchGet(request).code() == StatusCode::kBusy);
  CHECK(dispatch.Recover(get).code() == StatusCode::kBusy);
  BatchGetExecutionOptions options;
  options.config = get;
  options.stream = reinterpret_cast<void*>(1);
  options.graph_lease = lease;
  auto bad_options = options;
  bad_options.stream = reinterpret_cast<void*>(2);
  CHECK(runtime.BatchGet(request, bad_options).code() == StatusCode::kInvalidArgument);
  CHECK(runtime.BatchGet(request, options).ok());
  const auto drains = g.drains.load();
  CHECK(runtime.BatchGet(request, options).code() == StatusCode::kInvalidState);
  CHECK(g.drains == drains);
  CHECK(runtime.ReleaseAccessTable(table).ok());
  CHECK(runtime.BatchFree({{1}}).code() == StatusCode::kBusy);
  CHECK(runtime.BatchPut({{2}, {input, 16, DeviceKind::kCpu}, {0}, {16}}).ok());
  CHECK(g.drains == drains);
  CHECK(runtime.Shutdown().code() == StatusCode::kBusy);
  CHECK(runtime.initialized());
  g.fail_drain = true;
  CHECK(!runtime.EndGetGraph(lease, reinterpret_cast<void*>(1)).ok());
  CHECK(runtime.BatchFree({{1}}).code() == StatusCode::kBusy);
  g.fail_drain = false;
  CHECK(runtime.EndGetGraph(lease, reinterpret_cast<void*>(1)).ok());
  CHECK(std::memcmp(input, output, 16) == 0);
  CHECK(runtime.BatchFree({{1}}).ok());
  CHECK(runtime.Shutdown().ok());
}
void OutputLeaseReuseAndPins() {
  State p, g, a;
  ExecutionDispatcher dispatch(Policy()); Register(dispatch, p, g, a);
  auto meta = CreateInMemoryMetaManager();
  auto allocator = CreateHostDdrMemoryAllocator();
  RuntimeCore runtime(*meta, dispatch, *allocator);
  CHECK(runtime.Initialize({128, 16, DeviceKind::kCpu}).ok());
  char input[16] = "bound output", output[16]{}, other[16]{};
  CHECK(runtime.BatchPut({{1}, {input, 16, DeviceKind::kCpu}, {0}, {16}}).ok());
  AccessTableHandle table;
  CHECK(runtime.PrepareAccessTable({{1}}, &table).ok());
  int index = 0, offset = 0, count = 1, row = 0;
  BatchGetRequest request{table, {&index,4,DeviceKind::kNpu}, {&offset,4,DeviceKind::kNpu},
      {&count,4,DeviceKind::kNpu}, {&row,4,DeviceKind::kNpu},
      {output,16,DeviceKind::kNpu},16,1,{}};
  Address address = 0; std::uint8_t valid = 1;
  request.device_access_table = {{&address,8,DeviceKind::kNpu}, {&valid,1,DeviceKind::kNpu}};
  GetOutputOptions binding{request.output, reinterpret_cast<void*>(1), 1};
  std::uint64_t lease = 0, unused = 0;
  CHECK(runtime.BeginGetOutput(get, binding, &lease).ok() && lease && g.bindings == 1);
  CHECK(runtime.BeginGetOutput(gather, binding, &unused).ok());
  CHECK(runtime.EndGetOutput(unused).ok());
  CHECK(runtime.BeginGetGraph(get, {binding.output, binding.stream, 1}, &unused).code() == StatusCode::kBusy);
  CHECK(dispatch.Recover(get).code() == StatusCode::kBusy);
  CHECK(dispatch.Warmup(gather).ok());
  CHECK(runtime.BatchGet(request).code() == StatusCode::kBusy);
  BatchGetExecutionOptions options; options.config = get; options.output_lease = lease;
  auto changed = request; changed.output.data = other;
  CHECK(!runtime.BatchGet(changed, options).ok());
  changed = request; changed.indices.bytes = changed.offsets.bytes = changed.output_indices.bytes = 8;
  CHECK(!runtime.BatchGet(changed, options).ok());
  auto mismatched = options; mismatched.config = gather;
  CHECK(!runtime.BatchGet(request, mismatched).ok());
  CHECK(g.calls == 0 && a.creates == 1);
  for (int i = 0; i < 3; ++i) {
    options.synchronize = i % 2;
    CHECK(runtime.BatchGet(request, options).ok());
    CHECK(runtime.DrainGet().ok());
    CHECK(std::memcmp(input, output, 16) == 0);
  }
  CHECK(g.bindings == 1 && g.releases == 0);
  options.synchronize = false;
  CHECK(runtime.BatchGet(request, options).ok());
  CHECK(runtime.ReleaseAccessTable(table).ok());
  CHECK(runtime.BatchFree({{1}}).code() == StatusCode::kBusy);
  CHECK(runtime.BatchPut({{2}, {input, 16, DeviceKind::kCpu}, {0}, {16}}).ok());
  g.fail_drain = true;
  CHECK(!runtime.EndGetOutput(lease).ok() && g.releases == 0);
  CHECK(runtime.BatchFree({{1}}).code() == StatusCode::kBusy);
  g.fail_drain = false;
  CHECK(runtime.DrainGet().ok());
  CHECK(runtime.BatchFree({{1}}).ok()); // Tables do not stay pinned for the output lease.
  g.fail_release = true;
  CHECK(!runtime.EndGetOutput(lease).ok());
  CHECK(runtime.BeginGetOutput(get, binding, &unused).code() == StatusCode::kBusy);
  g.fail_release = false;
  CHECK(runtime.EndGetOutput(lease).ok());
  CHECK(runtime.EndGetOutput(lease).code() == StatusCode::kNotFound);
  CHECK(runtime.BeginGetOutput(get, binding, &unused).ok() && unused != lease);
  CHECK(runtime.EndGetOutput(lease).code() == StatusCode::kNotFound);
  g.fail_close = true;
  CHECK(!runtime.Shutdown().ok());
  g.fail_close = false;
  CHECK(runtime.Shutdown().ok()); // Also releases an idle output lease.

  auto fresh_meta = CreateInMemoryMetaManager();
  RuntimeCore retry_runtime(*fresh_meta, dispatch, *allocator);
  CHECK(retry_runtime.Initialize({128, 16, DeviceKind::kCpu}).ok());
  a.fail_bind = true; a.fail_close = true;
  CHECK(!retry_runtime.BeginGetOutput(gather, binding, &unused).ok() && !unused);
  CHECK(!retry_runtime.DrainGet().ok()); // Failed preparation retains partial resources.
  a.fail_close = false;
  CHECK(retry_runtime.DrainGet().ok());
  a.fail_bind = false;
  CHECK(dispatch.Recover(gather).ok());
  CHECK(retry_runtime.BeginGetOutput(gather, binding, &unused).ok());
  CHECK(retry_runtime.EndGetOutput(unused).ok());
  CHECK(retry_runtime.Shutdown().ok());
}
void TraceOutputAndOrdinary() {
  const std::string path = "/tmp/kvaf-get-dispatch-test-" +
      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".csv";
  std::remove(path.c_str());
  CHECK(setenv("KVAF_GET_DISPATCH_TRACE", path.c_str(), 1) == 0);
  State p, g, a;
  ExecutionDispatcher d(Policy());
  CHECK(unsetenv("KVAF_GET_DISPATCH_TRACE") == 0);
  Register(d, p, g, a);
  CHECK(d.Initialize({4096, 128, 64, DeviceKind::kCpu}).ok());
  char input[16] = "trace payload", output[16]{};
  Address addresses[]{reinterpret_cast<Address>(input)};
  std::uint64_t lengths[]{16};
  std::uint8_t valid[]{1};
  int index = 0, offset = 0, count = 1, output_index = 0;
  BatchGetRequest request{};
  request.indices = {&index, 4, DeviceKind::kNpu};
  request.offsets = {&offset, 4, DeviceKind::kNpu};
  request.miss_count = {&count, 4, DeviceKind::kNpu};
  request.output_indices = {&output_index, 4, DeviceKind::kNpu};
  request.output = {output, sizeof(output), DeviceKind::kNpu};
  request.output_block_bytes = sizeof(output);
  BatchGetExecutionOptions options{};
  options.stream = reinterpret_cast<void*>(1);
  options.synchronize = true;
  options.config = get;
  std::uint64_t lease{};
  CHECK(d.BeginGetOutput(get, {request.output, options.stream, 1}, &lease).ok());
  options.output_lease = lease;
  CHECK(d.ExecuteBatchGet({{addresses, lengths, valid, 1}, &request, options}).ok());
  CHECK(std::memcmp(input, output, sizeof(output)) == 0);
  CHECK(d.EndGetOutput(lease).ok());
  options.output_lease = 0;
  options.config = gather;
  CHECK(d.ExecuteBatchGet({{addresses, lengths, valid, 1}, &request, options}).ok());
  CHECK(d.ExecuteBatchGet({{addresses, lengths, valid, 1}, &request, options}).ok());
  CHECK(d.Shutdown().ok());
  std::ifstream trace(path);
  CHECK(trace.good());
  const std::string contents((std::istreambuf_iterator<char>(trace)), std::istreambuf_iterator<char>());
  CHECK(contents.find("algorithm,path") != std::string::npos);
  CHECK(contents.find("provider_prepare_ns,provider_submit_ns,provider_completion_ns,provider_cleanup_ns")
        != std::string::npos);
  CHECK(contents.find(",aiv_direct,output,") != std::string::npos);
  CHECK(contents.find(",gather,ordinary,") != std::string::npos);
  CHECK(contents.find(",11,12,13,14,") != std::string::npos);
  std::istringstream lines(contents);
  std::string line, last;
  std::getline(lines, line);
  const auto fields = [](const std::string& input) {
    std::vector<std::string> result;
    std::istringstream stream(input);
    std::string value;
    while (std::getline(stream, value, ',')) result.push_back(value);
    return result;
  };
  const auto columns = fields(line);
  while (std::getline(lines, line)) if (!line.empty()) last = line;
  const auto values = fields(last);
  CHECK(values.size() == columns.size());
  for (std::size_t i = 0; i < columns.size(); ++i)
    if (columns[i] == "describe_ns" || columns[i] == "bind_ns" || columns[i] == "warmup_ns")
      CHECK(values[i] == "0");
  std::remove(path.c_str());
}

void IndependentBindingsAndOtherPendingRequest() {
  State p, g, a, b;
  auto policy = Policy();
  ExecutionDispatcher dispatch(policy);
  Register(dispatch, p, g, a);
  CHECK(dispatch.RegisterGet(scatter, [&] { return std::make_unique<Provider>(b); }).ok());
  auto meta = CreateInMemoryMetaManager();
  auto allocator = CreateHostDdrMemoryAllocator();
  RuntimeCore runtime(*meta, dispatch, *allocator);
  CHECK(runtime.Initialize({128, 16, DeviceKind::kCpu}).ok());
  char input[16] = "mixed outputs", direct_output[16]{}, gather_output[16]{}, other_output[16]{};
  CHECK(runtime.BatchPut({{1}, {input, 16, DeviceKind::kCpu}, {0}, {16}}).ok());
  AccessTableHandle table;
  CHECK(runtime.PrepareAccessTable({{1}}, &table).ok());
  int index = 0, offset = 0, count = 1, row = 0;
  Address address = 0;
  std::uint8_t valid = 1;
  BatchGetRequest request{table, {&index,4,DeviceKind::kNpu}, {&offset,4,DeviceKind::kNpu},
      {&count,4,DeviceKind::kNpu}, {&row,4,DeviceKind::kNpu},
      {direct_output,16,DeviceKind::kNpu},16,1,{}};
  request.device_access_table = {{&address,8,DeviceKind::kNpu}, {&valid,1,DeviceKind::kNpu}};
  void* stream = reinterpret_cast<void*>(1);
  std::uint64_t direct_id{}, gather_id{};
  CHECK(runtime.BeginGetOutput(get, {request.output, stream, 1}, &direct_id).ok());
  request.output.data = gather_output;
  CHECK(runtime.BeginGetOutput(gather, {request.output, stream, 1}, &gather_id).ok());
  CHECK(direct_id != gather_id);
  BatchGetExecutionOptions options;
  options.stream = stream;
  options.synchronize = true;
  for (int iteration = 0; iteration < 4; ++iteration) {
    const bool direct = iteration % 2 == 0;
    options.config = direct ? get : gather;
    options.output_lease = direct ? direct_id : gather_id;
    // An output binding prewarms one stream, but later calls may use another.
    options.stream = reinterpret_cast<void*>(static_cast<std::uintptr_t>(iteration % 2 + 1));
    request.output.data = direct ? direct_output : gather_output;
    CHECK(runtime.BatchGet(request, options).ok());
    CHECK(std::memcmp(input, request.output.data, sizeof(input)) == 0);
  }
  CHECK(g.creates == 1 && a.creates == 1 && g.bindings == 1 && a.bindings == 1);
  options.config = gather;
  options.output_lease = direct_id;
  CHECK(runtime.BatchGet(request, options).code() == StatusCode::kBusy);
  CHECK(dispatch.Recover(get).code() == StatusCode::kBusy);
  options.config = scatter;
  options.output_lease = 0;
  options.synchronize = false;
  request.output.data = other_output;
  CHECK(runtime.BatchGet(request, options).ok());
  CHECK(b.drains == 0);
  CHECK(runtime.EndGetOutput(direct_id).ok());
  CHECK(runtime.EndGetOutput(gather_id).ok());
  CHECK(b.drains == 0); // Ending idle bindings does not complete the third algorithm.
  CHECK(dispatch.Recover(get).code() == StatusCode::kBusy);
  CHECK(runtime.ReleaseAccessTable(table).ok());
  CHECK(runtime.BatchFree({{1}}).code() == StatusCode::kBusy);
  CHECK(runtime.DrainGet().ok());
  CHECK(std::memcmp(input, other_output, sizeof(input)) == 0);
  CHECK(runtime.BatchFree({{1}}).ok());
  CHECK(runtime.EndGetOutput(direct_id).code() == StatusCode::kNotFound);
  CHECK(runtime.Shutdown().ok());
}
void TraceLegacyProvider() {
  class LegacyProvider final : public AccessOperator {
   public:
    Status Initialize(const RemoteRegionDesc&) override { return Status::OK(); }
    Status Shutdown() override { return Status::OK(); }
    Status ExecuteBatchPut(const BatchPutExecRequest&) override { return Status::OK(); }
    Status ExecuteBatchGet(const BatchGetExecRequest& r) override {
      CHECK(r.host_timing != nullptr);
      r.host_timing->prepare_ns = 21;
      r.host_timing->submit_ns = 22;
      r.host_timing->completion_ns = 23;
      return Status::OK();
    }
  } provider;
  const std::string path = "/tmp/kvaf-get-legacy-trace-test-" +
      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".csv";
  std::remove(path.c_str());
  CHECK(setenv("KVAF_GET_DISPATCH_TRACE", path.c_str(), 1) == 0);
  auto meta = CreateInMemoryMetaManager();
  auto allocator = CreateHostDdrMemoryAllocator();
  RuntimeCore runtime(*meta, provider, *allocator);
  CHECK(unsetenv("KVAF_GET_DISPATCH_TRACE") == 0);
  CHECK(runtime.Initialize({128, 16, DeviceKind::kCpu}).ok());
  Address addresses[]{1};
  std::uint8_t valid[]{1};
  int index = 0, offset = 0, count = 0, output_index = 0;
  char output[16]{};
  BatchGetRequest request{};
  request.indices = {&index, 4, DeviceKind::kNpu};
  request.offsets = {&offset, 4, DeviceKind::kNpu};
  request.miss_count = {&count, 4, DeviceKind::kNpu};
  request.output_indices = {&output_index, 4, DeviceKind::kNpu};
  request.output = {output, sizeof(output), DeviceKind::kNpu};
  request.output_block_bytes = sizeof(output);
  request.tokens_per_block = 1;
  request.device_access_table = {{addresses, sizeof(addresses), DeviceKind::kNpu},
                                 {valid, sizeof(valid), DeviceKind::kNpu}};
  CHECK(runtime.BatchGet(request).ok());
  CHECK(runtime.Shutdown().ok());
  std::ifstream trace(path);
  CHECK(trace.good());
  const std::string contents((std::istreambuf_iterator<char>(trace)), std::istreambuf_iterator<char>());
  CHECK(contents.find(",legacy,legacy,") != std::string::npos);
  CHECK(contents.find(",21,22,23,0,0,0,0,0\n") != std::string::npos);
  std::remove(path.c_str());
}
}  // namespace
int main() {
  Configurations();
  LazyAndIndependent();
  RuntimePinsAndTransactions();
  GraphLeasePinsAndIsolation();
  OutputLeaseReuseAndPins();
  IndependentBindingsAndOtherPendingRequest();
  TraceOutputAndOrdinary();
  TraceLegacyProvider();
  std::cout << "Execution registry, lazy domains, pins, overlap and recovery passed\n";
}
