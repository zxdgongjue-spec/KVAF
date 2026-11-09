#include "execution_dispatcher.h"
#include "runtime.h"

#include <cstring>
#include <cstdlib>
#include <iostream>
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
  int fail_commit{0}, fail_abort{0}, commits{0}, aborts{0};
  std::unique_ptr<MetaManager> real = CreateInMemoryMetaManager();
  Status RegisterRemoteRegion(const RemoteRegionDesc& r) override { return real->RegisterRemoteRegion(r); }
  Status PreparePut(const std::vector<KeyId>& keys, const std::vector<std::uint64_t>& lengths,
                    PutContext* txn, PutAccessDesc* desc) override {
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
  bool fail_shutdown{false};
  bool throw_after_submit{false};
  bool fail_bind{false};
  bool throw_bind{false};
};
class Provider final : public AccessOperator {
 public:
  explicit Provider(ProviderState& state) : state_(state) {}
  Status Initialize(const RemoteRegionDesc&) override { return Status::OK(); }
  Status Shutdown() override {
    return state_.fail_shutdown ? Status::CompletionUnknown("DMA still active") : Status::OK();
  }
  Status ExecuteBatchGet(const BatchGetExecRequest&) override { return Status::OK(); }
  Status BindExecutionState(ExecutionState*) override {
    if (state_.throw_bind) throw std::runtime_error("injected state binding exception");
    return state_.fail_bind ? Status::Internal("injected state binding failure") : Status::OK();
  }
  Status ExecuteBatchPut(const BatchPutExecRequest& request) override {
    ++state_.launches;
    for (std::size_t i = 0; i < request.request->keys.size(); ++i)
      std::memcpy(reinterpret_cast<void*>(request.put_desc->entries[i].address),
                  static_cast<const char*>(request.request->input.data) + request.request->input_offsets[i],
                  request.request->lengths[i]);
    if (state_.throw_after_submit) throw std::runtime_error("injected failure after DMA submission");
    return state_.launch;
  }
  Status ResolvePut(PutCompletion* completion) override {
    ++state_.resolves;
    *completion = state_.completion;
    return Status::OK();
  }
  Status AcknowledgePut() override {
    if (state_.fail_ack) { --state_.fail_ack; return Status::Internal("injected acknowledgement failure"); }
    return Status::OK();
  }
 private:
  ProviderState& state_;
};

ExecutionPolicy Policy() {
  ExecutionPolicy policy;
  policy.enabled_put_configs = std::vector<PutExecutionConfig>{put, other};
  policy.enabled_get_configs = std::vector<GetExecutionConfig>{};
  return policy;
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
  meta.fail_abort = 1;
  CHECK(!runtime.BatchPut(request).ok());
  CHECK(runtime.HasPendingPut() && state.launches == 0);
  CHECK(!runtime.RecoverPut(other).ok() && runtime.HasPendingPut());
  CHECK(!runtime.RecoverPut(put).ok());  // Original bind failure is reported.
  CHECK(!runtime.HasPendingPut() && state.launches == 0 && state.resolves == 0);
  state.fail_bind = false;
  state.throw_bind = false;
  CHECK(runtime.BatchPut(request).ok());
  CHECK(state.launches == 1 && runtime.Shutdown().ok());
}
}

int main() {
  CheckOnlineRecovery();
  CheckAcknowledgementAndException(false);
  CheckAcknowledgementAndException(true);
  CheckPreparationFailure(false);
  CheckPreparationFailure(true);
  std::cout << "Put recovery preserves transactions, commits/aborts once and never replays DMA\n";
}
