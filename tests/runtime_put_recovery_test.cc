#include "execution_dispatcher.h"
#include "runtime.h"

#include <cstring>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <unistd.h>
#include <vector>
#include <stdexcept>

#define CHECK(c) do { if (!(c)) { \
  std::cerr << "line " << __LINE__ << ": " #c "\n"; std::exit(1); \
} } while (false)

namespace {
using namespace kvaf;
constexpr PutExecutionConfig put{Transport::Urma, Executor::Aiv};
constexpr PutExecutionConfig other{Transport::Urma, Executor::Aicpu};

class FaultMeta final : public MetaManager {
 public:
  int fail_prepare{0}, fail_commit{0}, fail_abort{0}, prepares{0}, commits{0}, aborts{0};
  bool throw_prepare{false};
  std::unique_ptr<MetaManager> real = CreateInMemoryMetaManager();
  Status RegisterRemoteRegion(const RemoteRegionDesc& r) override { return real->RegisterRemoteRegion(r); }
  Status PreparePut(const std::vector<KeyId>& keys, const std::vector<std::uint64_t>& lengths,
                    PutContext* txn, PutAccessDesc* desc) override {
    ++prepares;
    if (throw_prepare) throw std::runtime_error("injected metadata preparation exception");
    if (fail_prepare) { --fail_prepare; return Status::OutOfMemory("injected metadata preparation failure"); }
    return real->PreparePut(keys, lengths, txn, desc);
  }
  Status CommitPut(const PutContext& txn) override {
    ++commits;
    if (fail_commit) { --fail_commit; return Status::Internal("injected commit failure"); }
    return real->CommitPut(txn);
  }
  Status AbortPut(const PutContext& txn) override {
    ++aborts;
    if (fail_abort) { --fail_abort; return Status::Internal("injected abort failure"); }
    return real->AbortPut(txn);
  }
  Status LockKeys(const std::vector<KeyId>& keys, MetaManagerLockMode mode, LockContext* lock) override {
    return real->LockKeys(keys, mode, lock);
  }
  Status BatchLookup(const std::vector<KeyId>& keys, const LockContext& lock, FlatAccessDesc* desc) override {
    return real->BatchLookup(keys, lock, desc);
  }
  Status UnlockKeys(const LockContext& lock) override { return real->UnlockKeys(lock); }
  Status BatchFree(const std::vector<KeyId>& keys) override { return real->BatchFree(keys); }
};

struct ProviderState {
  Status launch;
  PutCompletion completion;
  int launches{0}, resolves{0}, fail_ack{0};
  int input_setups{0}, input_finishes{0}, states{0}, prepares{0};
  int binds{0}, describes{0}, warmups{0};
  int launched_state{0}, resolved_state{0}, acknowledged_state{0};
  bool fail_input_setup{false}, fail_input_finish{false};
  bool fail_shutdown{false};
  bool throw_after_submit{false};
  bool fail_bind{false};
  bool throw_bind{false};
};
struct ProviderInvocation final : ExecutionState {
  explicit ProviderInvocation(int value) : id(value) {}
  int id;
};
class Provider final : public AccessOperator {
 public:
  explicit Provider(ProviderState& state) : state_(state) {}
  Status Initialize(const RemoteRegionDesc&) override { return Status::OK(); }
  Status Shutdown() override {
    return state_.fail_shutdown ? Status::CompletionUnknown("DMA still active") : Status::OK();
  }
  Status ExecuteBatchGet(const BatchGetExecRequest&) override { return Status::OK(); }
  Status BindExecutionState(ExecutionState* state) override {
    if (!state) { active_ = nullptr; return Status::OK(); }
    if (state_.throw_bind) throw std::runtime_error("injected state binding exception");
    if (state_.fail_bind) return Status::Internal("injected state binding failure");
    active_ = dynamic_cast<ProviderInvocation*>(state);
    CHECK(active_);
    ++state_.binds;
    return Status::OK();
  }
  std::unique_ptr<ExecutionState> CreateExecutionState() override {
    return std::make_unique<ProviderInvocation>(++state_.states);
  }
  Status DescribeRequirements(const BatchPutRequest& request, std::uint32_t* capacity) const override {
    ++state_.describes;
    return AccessOperator::DescribeRequirements(request, capacity);
  }
  Status Warmup(const ExecutionWarmup&) override {
    ++state_.warmups;
    return Status::OK();
  }
  Status PreparePut(const BatchPutRequest&, const BatchPutExecutionOptions&) override {
    ++state_.prepares;
    return Status::OK();
  }
  Status PreparePutInput(const PutInputOptions&) override {
    ++state_.input_setups;
    return state_.fail_input_setup ? Status::Internal("injected input setup failure") : Status::OK();
  }
  Status FinishPutInput() override {
    ++state_.input_finishes;
    return state_.fail_input_finish ? Status::Internal("injected input finish failure") : Status::OK();
  }
  Status ExecuteBatchPut(const BatchPutExecRequest& request) override {
    CHECK(active_);
    state_.launched_state = active_->id;
    ++state_.launches;
    if (auto* t = request.execution.host_timing) {
      t->transport = "urma"; t->executor = "aiv";
      t->prepare_ns = 5; t->registration_ns = 2; t->metadata_ns = 7;
      t->submit_ns = 11; t->completion_ns = 13; t->cleanup_ns = 17;
    }
    for (std::size_t i = 0; i < request.request->keys.size(); ++i)
      std::memcpy(reinterpret_cast<void*>(request.put_desc->entries[i].address),
                  static_cast<const char*>(request.request->input.data) + request.request->input_offsets[i],
                  request.request->lengths[i]);
    if (state_.throw_after_submit) throw std::runtime_error("injected failure after DMA submission");
    return state_.launch;
  }
  Status ResolvePut(PutCompletion* completion) override {
    CHECK(active_);
    state_.resolved_state = active_->id;
    ++state_.resolves;
    *completion = state_.completion;
    return Status::OK();
  }
  Status AcknowledgePut() override {
    CHECK(active_);
    state_.acknowledged_state = active_->id;
    if (state_.fail_ack) { --state_.fail_ack; return Status::Internal("injected acknowledgement failure"); }
    return Status::OK();
  }
 private:
  ProviderState& state_;
  ProviderInvocation* active_{nullptr};
};

ExecutionPolicy Policy() {
  ExecutionPolicy policy;
  return policy;
}

void CheckRuntimePreparationGuard() {
  class PreparationProbe final : public AccessOperator {
   public:
    bool reserved{false};
    int cancellations{0}, launches{0};
    Status Initialize(const RemoteRegionDesc&) override { return Status::OK(); }
    Status Shutdown() override { CHECK(!reserved); return Status::OK(); }
    Status PreparePut(const BatchPutRequest&, const BatchPutExecutionOptions&) override {
      CHECK(!reserved);
      reserved = true;
      return Status::OK();
    }
    Status CancelPreparedPut() override {
      if (reserved) { reserved = false; ++cancellations; }
      return Status::OK();
    }
    Status ExecuteBatchPut(const BatchPutExecRequest&) override {
      CHECK(reserved);
      ++launches;
      // A rejection before accepting the launch leaves preparation to Runtime.
      return Status::InvalidArgument("injected rejection before submission");
    }
    Status ExecuteBatchGet(const BatchGetExecRequest&) override { return Status::OK(); }
  } provider;
  FaultMeta meta;
  auto allocator = CreateHostDdrMemoryAllocator();
  RuntimeCore runtime(meta, provider, *allocator);
  CHECK(runtime.Initialize({256, 16, DeviceKind::kCpu}).ok());
  char payload[16]{};
  const BatchPutRequest request{{0}, {payload, 16, DeviceKind::kCpu}, {0}, {16}};
  meta.fail_prepare = 1;
  CHECK(runtime.BatchPut(request).code() == StatusCode::kOutOfMemory);
  CHECK(!provider.reserved && provider.cancellations == 1 && provider.launches == 0);
  CHECK(!runtime.HasPendingPut() && meta.aborts == 0);

  meta.throw_prepare = true;
  bool threw = false;
  try { (void)runtime.BatchPut(request); }
  catch (const std::runtime_error&) { threw = true; }
  CHECK(threw && !provider.reserved && provider.cancellations == 2 && provider.launches == 0);
  CHECK(!runtime.HasPendingPut() && meta.aborts == 0);
  meta.throw_prepare = false;

  CHECK(runtime.BatchPut(request).code() == StatusCode::kInvalidArgument);
  CHECK(!provider.reserved && provider.cancellations == 3 && provider.launches == 1);
  CHECK(!runtime.HasPendingPut() && meta.aborts == 1);
  CHECK(runtime.Shutdown().ok());
}

void CheckResidentPutInstances() {
  FaultMeta meta;
  ProviderState state, alternate;
  auto allocator = CreateHostDdrMemoryAllocator();
  ExecutionDispatcher dispatch(Policy());
  CHECK(dispatch.RegisterPut(put, [&] { return std::make_unique<Provider>(state); }).ok());
  CHECK(dispatch.RegisterPut(other, [&] { return std::make_unique<Provider>(alternate); }).ok());
  RuntimeCore runtime(meta, dispatch, *allocator);
  CHECK(runtime.Initialize({512, 16, DeviceKind::kCpu}).ok());
  char payload[32]{};
  void* stream = reinterpret_cast<void*>(1);
  auto write = [&](KeyId key, PutExecutionConfig config, void* on_stream) {
    return runtime.BatchPut({{key}, {payload, sizeof(payload), DeviceKind::kCpu}, {0}, {16}},
                            {on_stream, config});
  };
  CHECK(write(0, put, stream).ok());
  CHECK(state.states == 1 && state.binds == 1 && state.describes == 1 && state.warmups == 1);
  CHECK(write(1, put, stream).ok());
  CHECK(write(2, other, stream).ok());
  CHECK(write(3, put, stream).ok());
  CHECK(state.states == 1 && state.binds == 1 && state.describes == 1 && state.warmups == 1);
  CHECK(alternate.states == 1 && alternate.binds == 1);
  CHECK(state.launched_state == 1 && state.acknowledged_state == 1);

  // Growing capacity and changing stream may prepare resources again, but must
  // retain the invocation state rather than rebind an unrelated context.
  CHECK(runtime.BatchPut({{4, 5}, {payload, sizeof(payload), DeviceKind::kCpu}, {0, 16}, {16, 16}},
                         {stream, put}).ok());
  CHECK(write(6, put, reinterpret_cast<void*>(2)).ok());
  CHECK(state.states == 1 && state.binds == 1 && state.warmups > 1);

  // An input lease temporarily uses the already resident algorithm instance.
  const PutInputOptions input{{payload, sizeof(payload), DeviceKind::kNpu}, stream, 2};
  std::uint64_t id{};
  CHECK(runtime.BeginPutInput(put, input, &id).ok());
  CHECK(runtime.BatchPut({{7}, input.input, {0}, {16}}, {stream, put, id}).ok());
  CHECK(runtime.EndPutInput(id).ok());
  CHECK(write(8, put, stream).ok());
  CHECK(state.states == 1 && state.binds == 1 && state.input_setups == 1 && state.input_finishes == 1);

  CHECK(runtime.RecoverPut(other).ok());
  CHECK(write(9, other, stream).ok());
  CHECK(alternate.states == 2 && alternate.binds == 2);
  CHECK(write(10, put, stream).ok() && state.states == 1 && state.binds == 1);
  CHECK(runtime.Shutdown().ok());
}

void CheckMetadataPreparationCancellation() {
  FaultMeta meta;
  ProviderState state, alternate;
  auto allocator = CreateHostDdrMemoryAllocator();
  ExecutionDispatcher dispatch(Policy());
  CHECK(dispatch.RegisterPut(put, [&] { return std::make_unique<Provider>(state); }).ok());
  CHECK(dispatch.RegisterPut(other, [&] { return std::make_unique<Provider>(alternate); }).ok());
  RuntimeCore runtime(meta, dispatch, *allocator);
  CHECK(runtime.Initialize({256, 16, DeviceKind::kCpu}).ok());
  char payload[32]{};
  void* stream = reinterpret_cast<void*>(1);
  auto write = [&](KeyId key, PutExecutionConfig config = put, std::uint64_t bytes = 16) {
    return runtime.BatchPut({{key}, {payload, sizeof(payload), DeviceKind::kCpu}, {0}, {bytes}},
                            {stream, config});
  };
  CHECK(write(0).ok());
  CHECK(write(0).code() == StatusCode::kAlreadyExists);
  CHECK(!runtime.HasPendingPut() && state.launches == 1 && meta.aborts == 0);
  CHECK(write(1, other).ok());
  CHECK(write(16).code() == StatusCode::kInvalidArgument);
  CHECK(!runtime.HasPendingPut() && write(2).ok());
  CHECK(write(3, put, 32).code() == StatusCode::kInvalidArgument);
  CHECK(!runtime.HasPendingPut() && write(3).ok());
  meta.fail_prepare = 1;
  CHECK(write(4).code() == StatusCode::kOutOfMemory);
  CHECK(!runtime.HasPendingPut() && write(4).ok());
  meta.throw_prepare = true;
  bool threw = false;
  try { (void)write(5); }
  catch (const std::runtime_error&) { threw = true; }
  CHECK(threw && !runtime.HasPendingPut());
  meta.throw_prepare = false;
  CHECK(write(5).ok());
  CHECK(state.states == 1 && state.binds == 1 && state.launches == 5);
  CHECK(alternate.states == 1 && alternate.binds == 1 && alternate.launches == 1);
  CHECK(meta.aborts == 0 && meta.commits == 6);
  CHECK(runtime.Shutdown().ok());
}

void CheckOnlineRecovery() {
  FaultMeta meta;
  ProviderState state, unrelated;
  auto allocator = CreateHostDdrMemoryAllocator();
  ExecutionDispatcher dispatch(Policy());
  CHECK(dispatch.RegisterPut(put, [&] { return std::make_unique<Provider>(state); }).ok());
  CHECK(dispatch.RegisterPut(other, [&] { return std::make_unique<Provider>(unrelated); }).ok());
  RuntimeCore runtime(meta, dispatch, *allocator);
  CHECK(runtime.Initialize({256, 16, DeviceKind::kCpu}).ok());
  const auto base = runtime.remote_region().base;
  char payload[16] = "recovery-data";
  auto write = [&](KeyId key) {
    return runtime.BatchPut({{key}, {payload, sizeof(payload), DeviceKind::kCpu}, {0}, {sizeof(payload)}});
  };
  auto readable = [&](KeyId key) {
    AccessTableHandle table;
    if (!runtime.PrepareAccessTable({{key}}, &table).ok()) return false;
    AccessTableView view;
    CHECK(runtime.GetAccessTableView(table, &view).ok());
    CHECK(std::memcmp(reinterpret_cast<const void*>(view.addresses[0]), payload, sizeof(payload)) == 0);
    CHECK(runtime.ReleaseAccessTable(table).ok());
    return true;
  };
  CHECK(write(0).ok());
  state.launch = Status::CompletionUnknown("submitted, completion not yet known");
  CHECK(write(1).code() == StatusCode::kCompletionUnknown);
  CHECK(runtime.HasPendingPut() && !readable(1) && readable(0));
  CHECK(write(2).code() == StatusCode::kBusy);
  CHECK(!runtime.RecoverPut(other).ok());
  CHECK(state.resolves == 0 && unrelated.resolves == 0);
  CHECK(runtime.RecoverPut(put).code() == StatusCode::kCompletionUnknown);
  CHECK(runtime.HasPendingPut() && !readable(1));
  state.completion = {PutOutcome::Succeeded, Status::OK()};
  CHECK(runtime.RecoverPut(put).ok());
  CHECK(!runtime.HasPendingPut() && runtime.initialized() && readable(1));
  CHECK(state.launches == 2 && runtime.remote_region().base == base);

  // A terminal failure is aborted, reported to the caller and no longer owns
  // the source buffer. It must not publish partially written data.
  CHECK(write(2).code() == StatusCode::kCompletionUnknown);
  state.completion = {PutOutcome::Failed, Status::Internal("device rejected write")};
  meta.fail_abort = 1;
  CHECK(!runtime.RecoverPut(put).ok());
  CHECK(runtime.HasPendingPut() && !readable(2));
  CHECK(!runtime.RecoverPut(put).ok());
  CHECK(!runtime.HasPendingPut() && !readable(2));
  CHECK(state.launches == 3);
  state.launch = Status::OK();
  CHECK(write(2).ok() && readable(2));

  // A completed DMA with a metadata error retries metadata only.
  meta.fail_commit = 1;
  const auto before_resolve = state.resolves;
  CHECK(!write(3).ok() && runtime.HasPendingPut() && !readable(3));
  CHECK(runtime.RecoverPut(put).ok() && readable(3));
  CHECK(state.launches == 5 && state.resolves == before_resolve);

  // Acknowledgement failure happens after publication. Retry must not commit
  // twice: the real MetaManager rejects a second commit of the same txn ID.
  state.fail_ack = 1;
  CHECK(!write(4).ok() && runtime.HasPendingPut() && readable(4));
  const auto commits = meta.commits;
  CHECK(runtime.RecoverPut(put).ok() && !runtime.HasPendingPut());
  CHECK(meta.commits == commits && state.launches == 6);
  CHECK(state.states == 1 && state.binds == 1);
  CHECK(state.launched_state == 1 && state.resolved_state == 1 && state.acknowledged_state == 1);

  state.launch = Status::CompletionUnknown("pending until teardown");
  CHECK(!write(5).ok());
  state.fail_shutdown = true;
  CHECK(!runtime.Shutdown().ok() && runtime.HasPendingPut());
  state.fail_shutdown = false;
  CHECK(runtime.Shutdown().ok() && !runtime.HasPendingPut());
}

void CheckAcknowledgementAndException(bool throw_after_submit) {
  // A fresh Runtime/MetaManager, matching NpuRuntime's recreation on initialize.
  FaultMeta meta;
  ProviderState state, unrelated;
  auto allocator = CreateHostDdrMemoryAllocator();
  ExecutionDispatcher dispatch(Policy());
  CHECK(dispatch.RegisterPut(put, [&] { return std::make_unique<Provider>(state); }).ok());
  CHECK(dispatch.RegisterPut(other, [&] { return std::make_unique<Provider>(unrelated); }).ok());
  RuntimeCore runtime(meta, dispatch, *allocator);
  CHECK(runtime.Initialize({256, 16, DeviceKind::kCpu}).ok());
  char payload[16] = "recovery-data";
  state.throw_after_submit = throw_after_submit;
  state.fail_ack = throw_after_submit ? 0 : 1;
  const auto written = runtime.BatchPut({{0}, {payload, 16, DeviceKind::kCpu}, {0}, {16}});
  CHECK(!written.ok() && runtime.HasPendingPut());
  AccessTableHandle table;
  if (throw_after_submit) {
    CHECK(written.code() == StatusCode::kCompletionUnknown);
    CHECK(!runtime.PrepareAccessTable({{0}}, &table).ok());
    state.throw_after_submit = false;
    state.completion = {PutOutcome::Succeeded, Status::OK()};
    CHECK(runtime.RecoverPut(put).ok() && !runtime.HasPendingPut());
    CHECK(state.launches == 1);
  }
  CHECK(runtime.PrepareAccessTable({{0}}, &table).ok());
  CHECK(runtime.ReleaseAccessTable(table).ok());
  // Publication succeeded even if ACK failed. Shutdown must not try to abort
  // this already committed transaction (which the real MetaManager rejects).
  const auto aborts = meta.aborts;
  CHECK(runtime.Shutdown().ok() && !runtime.HasPendingPut());
  CHECK(meta.aborts == aborts);
}

void CheckPreparationFailure(bool throw_bind) {
  FaultMeta meta;
  ProviderState state, unrelated;
  auto allocator = CreateHostDdrMemoryAllocator();
  ExecutionDispatcher dispatch(Policy());
  CHECK(dispatch.RegisterPut(put, [&] { return std::make_unique<Provider>(state); }).ok());
  CHECK(dispatch.RegisterPut(other, [&] { return std::make_unique<Provider>(unrelated); }).ok());
  RuntimeCore runtime(meta, dispatch, *allocator);
  CHECK(runtime.Initialize({256, 16, DeviceKind::kCpu}).ok());
  char payload[16] = "prepare-failure";
  const BatchPutRequest request{{0}, {payload, 16, DeviceKind::kCpu}, {0}, {16}};
  state.fail_bind = !throw_bind;
  state.throw_bind = throw_bind;
  CHECK(!runtime.BatchPut(request).ok());
  // Binding now precedes MetaManager::PreparePut: there is no transaction to
  // abort, or original DMA to resolve, after a failed/throwing bind.
  CHECK(!runtime.HasPendingPut() && meta.prepares == 0 && meta.aborts == 0 && meta.commits == 0);
  CHECK(!runtime.HasPendingPut() && state.launches == 0 && state.resolves == 0);
  state.fail_bind = false;
  state.throw_bind = false;
  CHECK(runtime.RecoverPut(put).ok());
  CHECK(runtime.BatchPut(request).ok());
  CHECK(state.launches == 1 && runtime.Shutdown().ok());
}
void CheckInputBinding(PutExecutionConfig config = put) {
  const auto mismatch = config == put ? other : put;
  FaultMeta meta;
  ProviderState state, unrelated;
  auto allocator = CreateHostDdrMemoryAllocator();
  ExecutionDispatcher dispatch(Policy());
  CHECK(dispatch.RegisterPut(config, [&] { return std::make_unique<Provider>(state); }).ok());
  CHECK(dispatch.RegisterPut(mismatch, [&] { return std::make_unique<Provider>(unrelated); }).ok());
  RuntimeCore runtime(meta, dispatch, *allocator);
  CHECK(runtime.Initialize({256, 16, DeviceKind::kCpu}).ok());
  char payload[16] = "bound-input", alternate[16]{};
  void* stream = reinterpret_cast<void*>(1);
  const PutInputOptions binding{{payload, 16, DeviceKind::kNpu}, stream, 1};
  std::uint64_t id{};
  CHECK(runtime.BeginPutInput(config, binding, &id).ok() && id);
  CHECK(state.states == 1 && state.input_setups == 1);
  auto write = [&](KeyId key) {
    return runtime.BatchPut({{key}, binding.input, {0}, {16}}, {stream, config, id});
  };
  auto rejected = [&](BatchPutRequest request, BatchPutExecutionOptions options) {
    const auto calls = state.launches, commits = meta.commits;
    CHECK(!runtime.BatchPut(request, options).ok());
    CHECK(!runtime.HasPendingPut() && state.launches == calls && meta.commits == commits);
  };
  rejected({{0}, {alternate, 16, DeviceKind::kNpu}, {0}, {16}}, {stream, config, id});
  rejected({{0}, {payload, 8, DeviceKind::kNpu}, {0}, {8}}, {stream, config, id});
  rejected({{0}, binding.input, {0}, {16}}, {reinterpret_cast<void*>(2), config, id});
  rejected({{0, 1}, binding.input, {0, 8}, {8, 8}}, {stream, config, id});
  rejected({{0}, binding.input, {0}, {16}}, {stream, mismatch, id});
  rejected({{0}, binding.input, {0}, {16}}, {stream, config, 0});
  CHECK(!dispatch.Warmup(config).ok() && !runtime.RecoverPut(config).ok());
  CHECK(write(0).ok() && write(1).ok());
  CHECK(state.states == 1 && state.binds == 1 && state.input_setups == 1 && state.input_finishes == 0);
  CHECK(write(0).code() == StatusCode::kAlreadyExists && !runtime.HasPendingPut());
  CHECK(state.launches == 2);  // Metadata rejection also cancels bound preparation.

  // Both unknown completion and metadata errors retain the same bound context.
  state.launch = Status::CompletionUnknown("unknown completion");
  CHECK(!write(2).ok() && runtime.HasPendingPut());
  CHECK(runtime.EndPutInput(id).code() == StatusCode::kBusy);
  state.completion = {PutOutcome::Succeeded, Status::OK()};
  CHECK(runtime.RecoverPut(config).ok() && state.launches == 3);
  state.launch = Status::OK();
  meta.fail_commit = 1;
  CHECK(!write(3).ok() && runtime.HasPendingPut());
  CHECK(runtime.EndPutInput(id).code() == StatusCode::kBusy);
  CHECK(runtime.RecoverPut(config).ok() && state.launches == 4);
  state.fail_ack = 1;
  CHECK(!write(4).ok() && runtime.HasPendingPut());
  const auto commits = meta.commits;
  CHECK(runtime.EndPutInput(id).code() == StatusCode::kBusy);
  CHECK(runtime.RecoverPut(config).ok() && meta.commits == commits && state.launches == 5);
  CHECK(write(5).ok() && state.states == 1);

  state.fail_input_finish = true;
  CHECK(!runtime.EndPutInput(id).ok());
  state.fail_input_finish = false;
  CHECK(runtime.EndPutInput(id).ok());
  CHECK(!runtime.EndPutInput(id).ok());
  rejected({{6}, binding.input, {0}, {16}}, {stream, config, id});
  const auto previous_id = id;
  CHECK(runtime.BeginPutInput(config, binding, &id).ok() && id != previous_id);
  CHECK(write(6).ok() && state.states == 1 && state.binds == 1);
  state.fail_shutdown = true;
  CHECK(!runtime.Shutdown().ok());
  state.fail_shutdown = false;
  CHECK(runtime.Shutdown().ok());
}

void CheckFailedInputSetup() {
  FaultMeta meta;
  ProviderState state, unrelated;
  auto allocator = CreateHostDdrMemoryAllocator();
  ExecutionDispatcher dispatch(Policy());
  CHECK(dispatch.RegisterPut(put, [&] { return std::make_unique<Provider>(state); }).ok());
  CHECK(dispatch.RegisterPut(other, [&] { return std::make_unique<Provider>(unrelated); }).ok());
  RuntimeCore runtime(meta, dispatch, *allocator);
  CHECK(runtime.Initialize({256, 16, DeviceKind::kCpu}).ok());
  char payload[16]{};
  PutInputOptions binding{{payload, 16, DeviceKind::kNpu}, reinterpret_cast<void*>(1), 1};
  std::uint64_t id = 99;
  state.fail_input_setup = state.fail_shutdown = true;
  CHECK(!runtime.BeginPutInput(put, binding, &id).ok() && id == 0);
  CHECK(!runtime.BatchPut({{0}, binding.input, {0}, {16}}, {binding.stream, put}).ok());
  CHECK(!runtime.BeginPutInput(put, binding, &id).ok());
  CHECK(!runtime.RecoverPut(put).ok() && !runtime.Shutdown().ok());
  state.fail_shutdown = false;
  CHECK(runtime.Shutdown().ok() && state.launches == 0);
}

void CheckPutTrace() {
  const auto trace = "/tmp/kvaf-put-trace-" + std::to_string(getpid()) + ".csv";
  std::remove(trace.c_str());
  CHECK(setenv("KVAF_PUT_DISPATCH_TRACE", trace.c_str(), 1) == 0);
  CheckInputBinding();
  CHECK(unsetenv("KVAF_PUT_DISPATCH_TRACE") == 0);
  std::ifstream input(trace);
  std::string line;
  CHECK(std::getline(input, line));
  auto split = [](const std::string& value) {
    std::vector<std::string> fields;
    std::istringstream row(value);
    std::string field;
    while (std::getline(row, field, ',')) fields.push_back(field);
    return fields;
  };
  const auto header = split(line);
  auto column = [&](const char* name) {
    for (std::size_t i = 0; i < header.size(); ++i) if (header[i] == name) return i;
    CHECK(false); return header.size();
  };
  int success = 0, recovery = 0, failures = 0;
  while (std::getline(input, line)) {
    const auto fields = split(line);
    CHECK(fields.size() == header.size());
    CHECK(std::stoull(fields[column("total_ns")]) > 0);
    if (fields[column("event")] == "recover") { ++recovery; continue; }
    if (fields[column("result")] != "0") { ++failures; continue; }
    CHECK(fields[column("transport")] == "urma" && fields[column("executor")] == "aiv");
    CHECK(fields[column("path")] == "input");
    CHECK(fields[column("provider_registration_ns")] == "2");
    CHECK(fields[column("provider_prepare_ns")] == "5");
    // A bound input keeps its resident reference; each transaction still takes
    // a short execution lease, without describing or binding another state.
    CHECK(fields[column("describe_ns")] == "0" && fields[column("bind_ns")] == "0");
    CHECK(std::stoull(fields[column("provider_ns")]) > 0);
    ++success;
  }
  CHECK(success == 4 && recovery == 4 && failures >= 6);
  std::remove(trace.c_str());
}

}

int main() {
  CheckRuntimePreparationGuard();
  CheckResidentPutInstances();
  CheckMetadataPreparationCancellation();
  CheckOnlineRecovery();
  CheckInputBinding();
  CheckInputBinding(other);
  CheckFailedInputSetup();
  CheckPutTrace();
  CheckAcknowledgementAndException(false);
  CheckAcknowledgementAndException(true);
  CheckPreparationFailure(false);
  CheckPreparationFailure(true);
  std::cout << "Put recovery preserves transactions, commits/aborts once and never replays DMA\n";
}
