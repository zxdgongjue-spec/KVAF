// Host-only fault injection against the real Dispatcher, RuntimeCore and table
// pins. No ACL/URMA stubs or device SDK are needed; see this directory's README.
#include <cstdlib>
#include <iostream>
#include "execution_dispatcher.h"
#include "runtime.h"

using namespace kvaf;

namespace {
void Check(bool condition, const char* message) {
  if (!condition) { std::cerr << message << '\n'; std::abort(); }
}
void Ok(Status status) { Check(status.ok(), status.message().c_str()); }

struct Faults {
  bool device_error{true}, close_error{false}, init_error{false}, finish_error{false};
  int initializes{}, closes{}, launches{}, drains{};
};
struct Call : ExecutionState { bool failed{false}, output{false}; };
class FailingGet final : public AccessOperator {
 public:
  explicit FailingGet(Faults& faults) : faults_(faults) {}
  Status Initialize(const RemoteRegionDesc&) override {
    ++faults_.initializes;
    return faults_.init_error ? Status::Internal("injected init failure") : Status::OK();
  }
  Status Shutdown() override {
    ++faults_.closes;
    if (faults_.close_error) return Status::CompletionUnknown("injected unsafe close");
    call_->failed = call_->output = false;
    return Status::OK();
  }
  std::unique_ptr<ExecutionState> CreateExecutionState() override { return std::make_unique<Call>(); }
  Status BindExecutionState(ExecutionState* state) override {
    call_ = state ? static_cast<Call*>(state) : &fallback_;
    return Status::OK();
  }
  Status ExecuteBatchGet(const BatchGetExecRequest& request) override {
    ++faults_.launches;
    call_->failed = faults_.device_error;
    return request.execution.synchronize ? DrainGet() : Status::OK();
  }
  Status ExecuteBatchPut(const BatchPutExecRequest&) override { return Status::OK(); }
  Status DrainGet() override {
    ++faults_.drains;
    return call_->failed ? Status::Internal("injected persistent device error") : Status::OK();
  }
  Status PrepareGetOutput(const GetOutputOptions&) override { call_->output = true; return Status::OK(); }
  Status FinishGetOutput() override {
    if (call_->failed || faults_.finish_error) return Status::Internal("injected output retirement failure");
    call_->output = false;
    return Status::OK();
  }
  Status PrepareGetGraph(const GetGraphOptions&) override { return Status::OK(); }
  Status FinishGetGraph(void*, bool) override { return Status::OK(); }
 private:
  Faults& faults_;
  Call fallback_;
  Call* call_{&fallback_};
};

const GetExecutionConfig kGather{Transport::Urma, Executor::Aicpu, GetAlgorithm::Gather};
const GetExecutionConfig kScatter{Transport::Urma, Executor::Aicpu, GetAlgorithm::Scatter};

struct Fixture {
  Faults gather, scatter;
  ExecutionDispatcher dispatcher{ExecutionPolicy{TransportMode::Urma, {}, kGather}};
  std::unique_ptr<MetaManager> meta{CreateInMemoryMetaManager()};
  std::unique_ptr<DdrMemoryAllocator> allocator{CreateHostDdrMemoryAllocator()};
  RuntimeCore core{*meta, dispatcher, *allocator};
  AccessTableHandle table;
  std::int32_t index{0}, offset{0}, count{1};
  std::uint64_t address{0};
  std::uint8_t valid{1}, output[16]{};
  BatchGetRequest request;
  void* stream{reinterpret_cast<void*>(1)};

  Fixture() {
    Ok(dispatcher.RegisterGet(kGather, [&] { return std::make_unique<FailingGet>(gather); }));
    Ok(dispatcher.RegisterGet(kScatter, [&] { return std::make_unique<FailingGet>(scatter); }));
    Ok(core.Initialize({4096, 16, DeviceKind::kCpu}));
    PutContext transaction;
    PutAccessDesc allocation;
    Ok(meta->PreparePut({0}, {16}, &transaction, &allocation));
    Ok(meta->CommitPut(transaction));
    Ok(core.PrepareAccessTable({{0}}, &table));
    address = allocation.entries[0].address;
    request.access_table = table;
    request.indices = {&index, sizeof(index), DeviceKind::kNpu};
    request.offsets = {&offset, sizeof(offset), DeviceKind::kNpu};
    request.output_indices = request.indices;
    request.miss_count = {&count, sizeof(count), DeviceKind::kNpu};
    request.output = {output, sizeof(output), DeviceKind::kNpu};
    request.output_block_bytes = 16;
    request.tokens_per_block = 1;
    request.device_access_table = {{&address, sizeof(address), DeviceKind::kNpu},
                                   {&valid, sizeof(valid), DeviceKind::kNpu}};
  }
  ~Fixture() { Ok(core.Shutdown()); }
  Status Submit(GetExecutionConfig config = kGather, std::uint64_t lease = 0, bool sync = false) {
    BatchGetExecutionOptions options;
    options.config = config;
    options.stream = stream;
    options.output_lease = lease;
    options.synchronize = sync;
    return core.BatchGet(request, options);
  }
  std::uint64_t BindOutput() {
    std::uint64_t lease{};
    Ok(core.BeginGetOutput(kGather, {request.output, stream, 1}, &lease));
    return lease;
  }
  void DropHandle() { Ok(core.ReleaseAccessTable(table)); }
  void CheckPinned() { Check(!meta->BatchFree({0}).ok(), "unsafe retirement released the table pin"); }
  void CheckUnpinned() { Ok(meta->BatchFree({0})); }
};

void StickyFailure() {
  for (bool synchronous : {false, true}) {
    Fixture f;
    auto submitted = f.Submit(kGather, 0, synchronous);
    Check(submitted.ok() != synchronous, "wrong original request result");
    Check(!f.core.DrainGet().ok(), "device failure was lost");
    Check(f.core.RecoverGet(kScatter).code() == StatusCode::kBusy, "recovery discarded another algorithm's pending request");
    Check(f.scatter.closes == 0 && f.core.HasPendingGet(), "wrong-instance recovery changed ownership");
    const auto drains = f.gather.drains;
    f.gather.device_error = false;
    Ok(f.core.RecoverGet(kGather));
    Check(f.gather.closes == 1 && f.gather.initializes == 2, "recovery never closed/rebuilt the provider");
    Check(f.gather.drains == drains && f.gather.launches == 1, "recovery retried the failed request");
    Check(!f.core.HasPendingGet(), "closed request remained pending");
    Ok(f.Submit());
    Ok(f.core.DrainGet());
  }
}

void RetryCloseAndRebuild() {
  for (bool fail_close : {true, false}) {
    Fixture f;
    Ok(f.Submit());
    Check(!f.core.DrainGet().ok(), "expected device error");
    f.DropHandle();
    f.CheckPinned();
    f.gather.close_error = fail_close;
    f.gather.init_error = !fail_close;
    Check(!f.core.RecoverGet(kGather).ok(), "injected recovery failure was ignored");
    Check(f.core.HasPendingGet() == fail_close, "close and rebuild failures have different ownership");
    if (fail_close) f.CheckPinned(); else f.CheckUnpinned();
    f.gather.close_error = f.gather.init_error = f.gather.device_error = false;
    Ok(f.core.RecoverGet(kGather));
    Check(!f.core.HasPendingGet(), "retry did not retire the request");
    if (fail_close) f.CheckUnpinned();
  }
}

void FailedOutputBinding() {
  Fixture f;
  const auto lease = f.BindOutput();
  Ok(f.Submit(kGather, lease));
  Check(!f.core.DrainGet().ok(), "expected device error");
  f.DropHandle();
  f.gather.close_error = true;
  Check(!f.core.EndGetOutput(lease).ok(), "failed close discarded the binding");
  f.CheckPinned();
  const auto closes = f.gather.closes;
  Check(!f.core.DrainGet().ok(), "Drain discarded an unfinished output binding");
  Check(f.gather.closes == closes, "Drain must leave bound teardown to EndGetOutput");
  Check(f.core.RecoverGet(kGather).code() == StatusCode::kBusy, "recovery invalidated a live output ID");
  f.gather.close_error = f.gather.device_error = false;
  Ok(f.core.EndGetOutput(lease));
  Check(!f.core.HasPendingGet(), "ended output retained the failed request");
  f.CheckUnpinned();
  Check(!f.core.EndGetOutput(lease).ok(), "retired lease was reused");
  Ok(f.core.RecoverGet(kGather));
}

void KeepOtherRequestAndGraphs() {
  Fixture f;
  const auto lease = f.BindOutput();
  f.scatter.device_error = false;
  Ok(f.Submit(kScatter));
  Ok(f.core.RecoverGet(kScatter));
  Ok(f.Submit(kScatter));
  Check(f.core.RecoverGet(kGather).code() == StatusCode::kBusy, "live output binding must remain explicit");
  f.gather.finish_error = true;
  f.gather.close_error = true;
  Check(!f.core.EndGetOutput(lease).ok(), "injected output close failure was ignored");
  Check(f.core.HasPendingGet(), "ending idle output released another request");
  Check(f.scatter.closes == 1, "idle output teardown closed another algorithm");
  Ok(f.core.DrainGet());
  f.gather.close_error = false;
  Ok(f.core.EndGetOutput(lease));
  f.gather.finish_error = false;
  Ok(f.core.RecoverGet(kGather));
  std::uint64_t graph{};
  Ok(f.core.BeginGetGraph(kGather, {f.request.output, f.stream, 1}, &graph));
  const auto closes = f.gather.closes;
  Check(f.core.RecoverGet(kGather).code() == StatusCode::kBusy, "recovery closed a live graph");
  Check(f.gather.closes == closes, "graph protection ran after closing resources");
  Ok(f.core.EndGetGraph(graph, f.stream, true));
}
}  // namespace

int main() {
  StickyFailure();
  RetryCloseAndRebuild();
  FailedOutputBinding();
  KeepOtherRequestAndGraphs();
  std::cout << "Get recovery, close/rebuild retries, table pins and output ownership PASS\n";
}
