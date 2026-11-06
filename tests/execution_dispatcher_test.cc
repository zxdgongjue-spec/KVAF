#include "execution_dispatcher.h"
#include "runtime.h"
#include "cache_region.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>
using namespace kvaf;
using namespace std::chrono_literals;
#define CHECK(c) do { if (!(c)) throw std::runtime_error(std::string("line ") + std::to_string(__LINE__) + ": " #c); } while (0)
namespace {
constexpr PutExecutionConfig put{Transport::Urma, Executor::Aiv};
constexpr GetExecutionConfig get{Transport::Urma, Executor::Aiv, GetAlgorithm::Direct};
constexpr GetExecutionConfig gather{Transport::Urma, Executor::Aicpu, GetAlgorithm::Gather};
ExecutionPolicy Policy() {
  ExecutionPolicy p;
  p.enabled_put_configs = std::vector<PutExecutionConfig>{put};
  p.enabled_get_configs = std::vector<GetExecutionConfig>{get, gather};
  return p;
}
struct State {
  std::atomic<int> creates{0}, calls{0}, drains{0}, closes{0};
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
    CHECK(e.execution.config == get || e.execution.config == gather);
    request_ = *e.request;
    table_ = e.host_access_table;
    pending_ = true;
    return e.execution.synchronize ? DrainGet() : Status::OK();
  }
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
  CHECK(p.enabled_put_configs->size() == 2 && p.enabled_get_configs->size() == 5);
  CHECK(ValidateExecution(TransportMode::Urma, {Transport::Urma, Executor::Aiv, GetAlgorithm::Scatter}).ok());
  CHECK(!ValidateExecution(TransportMode::Ubmem, {Transport::Ubmem, Executor::Aiv, GetAlgorithm::Gather}).ok());
  CHECK(!ValidateExecution(TransportMode::Urma, {Transport::Urma, Executor::Aicpu, GetAlgorithm::Direct}).ok());
  CHECK(!ValidateExecution(TransportMode::Urma, PutExecutionConfig{Transport::Ubmem, Executor::Aiv}).ok());
  p = {}; p.transport_mode = TransportMode::UrmaAndUbmem;
  CHECK(NormalizeExecutionPolicy(&p).ok());
  CHECK(p.enabled_put_configs->size() == 3 && p.enabled_get_configs->size() == 6);
  p = Policy(); p.enabled_put_configs = std::vector<PutExecutionConfig>{};
  CHECK(NormalizeExecutionPolicy(&p).ok() && !p.default_put);
  PutExecutionConfig c;
  CHECK(ResolveExecution(p, put, &c).code() == StatusCode::kNotImplemented);
  p.default_put = put;
  CHECK(!NormalizeExecutionPolicy(&p).ok());
  p = Policy(); p.default_get = {Transport::Urma, Executor::Aiv, GetAlgorithm::Scatter};
  CHECK(!NormalizeExecutionPolicy(&p).ok());
  p = Policy(); p.enabled_put_configs->push_back(put);
  CHECK(!NormalizeExecutionPolicy(&p).ok());
  CacheRegionDesc region{1, 2, 100, 64, 8}; Address translated;
  CHECK(region.Translate(120, 8, 1000, &translated).ok() && translated == 1020);
  CHECK(!region.Translate(99, 1, 1000, &translated).ok());
  CHECK(!region.Translate(160, 8, 1000, &translated).ok());
  CHECK(!region.Translate(120, 8, UINT64_MAX - 10, &translated).ok());
  ExecutionDispatcher missing(Policy());
  CHECK(missing.Initialize({4096, 128, 64, DeviceKind::kCpu}).code() == StatusCode::kNotImplemented);
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
}  // namespace
int main() {
  Configurations(); LazyAndIndependent(); RuntimePinsAndTransactions();
  std::cout << "Execution registry, lazy domains, pins, overlap and recovery passed\n";
}
