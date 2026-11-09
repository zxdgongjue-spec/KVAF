#include "execution_context.h"
#include "execution_dispatcher.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

using namespace kvaf;
using namespace std::chrono_literals;
#define CHECK(c) do { if (!(c)) throw std::runtime_error(std::string("line ") + std::to_string(__LINE__) + ": " #c); } while (0)
namespace {
constexpr PutExecutionConfig put{Transport::Urma, Executor::Aiv};
constexpr GetExecutionConfig get{Transport::Urma, Executor::Aiv, GetAlgorithm::Direct};
constexpr RemoteRegionDesc region{4096, 1024, 64, DeviceKind::kCpu};

// These fakes deliberately separate physical ownership, borrowed operation and
// per-invocation state. An AccessOperator-only fake would miss ownership bugs.
struct Counters {
  std::atomic<int> factories{0}, initializes{0}, closes{0}, owners{0}, operators{0};
  std::atomic<int> live_states{0}, state_destroys{0}, legacy_shutdowns{0}, close_with_state{0};
  std::atomic<int> put_launches{0}, get_launches{0}, resolves{0};
  std::atomic<bool> fail_initialize{false}, fail_close{false}, fail_drain{false};
  std::atomic<bool> fail_put{false}, fail_resolve{false}, fail_ack{false}, fail_get{false};
  bool throw_get{false};
  std::atomic<bool> bad_order{false}, copied_metadata_ok{true};
  std::uint32_t capacity{65536};
  std::mutex mutex;
  std::condition_variable changed;
  bool block_initialize{false}, initialize_entered{false};
  std::vector<std::string> events;
  void Event(const char* value) {
    std::lock_guard<std::mutex> lock(mutex);
    events.emplace_back(value);
  }
};
struct Invocation final : ExecutionState {
  explicit Invocation(Counters& c) : counters(c) { ++counters.live_states; }
  ~Invocation() override {
    --counters.live_states;
    ++counters.state_destroys;
    counters.Event("state destroyed");
  }
  Counters& counters;
  const BatchPutRequest* put_request{};
  const PutAccessDesc* put_destination{};
  const BatchGetRequest* get_request{};
  bool pending_put{}, pending_get{};
};
struct PhysicalState {
  explicit PhysicalState(Counters& c) : counters(c) {}
  Counters& counters;
  Invocation* invocation{};
};
class Owner final : public ExecutionResourceOwner {
 public:
  explicit Owner(Counters& c) : physical(c) { ++c.owners; }
  ~Owner() override {
    if (physical.counters.operators != 0) physical.counters.bad_order = true;
    --physical.counters.owners;
    physical.counters.Event("owner destroyed");
  }
  Status Close() override {
    auto& c = physical.counters;
    ++c.closes;
    if (physical.invocation) {
      ++c.close_with_state;
      if (c.live_states == 0) c.bad_order = true;
    }
    if (c.fail_close) return Status::CompletionUnknown("resource close pending");
    if (physical.invocation) {
      physical.invocation->pending_put = false;
      physical.invocation->pending_get = false;
    }
    c.Event("owner closed");
    return Status::OK();
  }
  PhysicalState physical;
};
class BorrowedOperation final : public AccessOperator {
 public:
  explicit BorrowedOperation(Counters& c) : owner_(std::make_unique<Owner>(c)), physical_(&owner_->physical) {
    ++c.operators;
  }
  ~BorrowedOperation() override {
    auto& c = physical_->counters;
    if (!c.owners || physical_->invocation) c.bad_order = true;
    --c.operators;
    c.Event("operator destroyed");
  }
  std::unique_ptr<ExecutionResourceOwner> TakeResources() override { return std::move(owner_); }
  Status ValidateRequestCapacity(std::uint32_t count) const override {
    return count <= physical_->counters.capacity ? Status::OK()
        : Status::InvalidArgument("request exceeds resource capacity");
  }
  std::unique_ptr<ExecutionState> CreateExecutionState() override {
    return std::make_unique<Invocation>(physical_->counters);
  }
  Status BindExecutionState(ExecutionState* state) override {
    if (!state) { physical_->invocation = nullptr; return Status::OK(); }
    auto* invocation = dynamic_cast<Invocation*>(state);
    if (!invocation || physical_->invocation) return Status::InvalidState("invalid invocation bind");
    physical_->invocation = invocation;
    return Status::OK();
  }
  Status Initialize(const RemoteRegionDesc&) override {
    auto& c = physical_->counters;
    if (owner_) c.bad_order = true; // ownership must move before initialization
    ++c.initializes;
    std::unique_lock<std::mutex> lock(c.mutex);
    c.initialize_entered = true;
    c.changed.notify_all();
    c.changed.wait(lock, [&] { return !c.block_initialize; });
    return c.fail_initialize ? Status::Internal("initialization failed") : Status::OK();
  }
  Status Shutdown() override {
    ++physical_->counters.legacy_shutdowns;
    return Status::Internal("borrowed operator must not own resource shutdown");
  }
  Status ExecuteBatchPut(const BatchPutExecRequest& e) override {
    auto& c = physical_->counters;
    auto* state = physical_->invocation;
    if (!state) return Status::InvalidState("missing Put state");
    ++c.put_launches;
    state->put_request = e.request;
    state->put_destination = e.put_desc;
    state->pending_put = true;
    return c.fail_put ? Status::CompletionUnknown("WRITE submitted, wait interrupted") : Status::OK();
  }
  Status ResolvePut(PutCompletion* out) override {
    auto& c = physical_->counters;
    auto* state = physical_->invocation;
    if (!state || !state->pending_put) return Status::InvalidState("missing original WRITE");
    ++c.resolves;
    if (state->put_request->keys != std::vector<KeyId>{42} ||
        state->put_destination->entries[0].address != 4096)
      c.copied_metadata_ok = false;
    if (c.fail_resolve) return Status::CompletionUnknown("original WRITE still pending");
    *out = {PutOutcome::Succeeded, Status::OK()};
    return Status::OK();
  }
  Status AcknowledgePut() override {
    if (physical_->counters.fail_ack) return Status::Internal("metadata acknowledgement failed");
    physical_->invocation->pending_put = false;
    return Status::OK();
  }
  Status ExecuteBatchGet(const BatchGetExecRequest& e) override {
    auto& c = physical_->counters;
    auto* state = physical_->invocation;
    if (!state) return Status::InvalidState("missing Get state");
    ++c.get_launches;
    state->get_request = e.request;
    state->pending_get = true;
    if (c.throw_get) throw std::runtime_error("Get threw after submission");
    return c.fail_get ? Status::CompletionUnknown("READ submitted, wait interrupted") : Status::OK();
  }
  Status DrainGet() override {
    auto& c = physical_->counters;
    auto* state = physical_->invocation;
    if (!state || !state->pending_get) return Status::OK();
    if (state->get_request->output_block_bytes != 64) c.copied_metadata_ok = false;
    if (c.fail_drain) return Status::CompletionUnknown("READ still pending");
    state->pending_get = false;
    return Status::OK();
  }
  Status PrepareGetGraph(const GetGraphOptions&) override {
    return Status::Internal("registration partially prepared");
  }
 private:
  std::unique_ptr<Owner> owner_;
  PhysicalState* physical_;
};
auto Factory(Counters& c) {
  return [&c] {
    ++c.factories;
    return std::make_unique<BorrowedOperation>(c);
  };
}
void CheckDestroyed(const Counters& c) {
  CHECK(c.owners == 0 && c.operators == 0 && c.live_states == 0);
  CHECK(!c.bad_order && c.legacy_shutdowns == 0);
}
ExecutionPolicy Policy() {
  ExecutionPolicy p;
  p.enabled_put_configs = std::vector<PutExecutionConfig>{put};
  p.enabled_get_configs = std::vector<GetExecutionConfig>{get};
  return p;
}
void Register(ExecutionDispatcher& d, Counters& p, Counters& g) {
  CHECK(d.RegisterPut(put, Factory(p)).ok());
  CHECK(d.RegisterGet(get, Factory(g)).ok());
  CHECK(d.Initialize(region).ok());
}

void ExclusiveLeaseAndIndependentOwner() {
  Counters c;
  ExecutionResourceManager manager;
  CHECK(manager.Register(get, Factory(c)).ok());
  CHECK(manager.Initialize(region).ok());
  auto context = std::make_unique<GetExecutionContext>();
  CHECK(manager.AcquireGet({get, 1}, &context->lease).ok());
  CHECK(context->Bind().ok());
  GetResourceLease competing;
  CHECK(manager.AcquireGet({get, 1}, &competing).code() == StatusCode::kBusy);
  CHECK(!competing);
  CHECK(manager.Recover(get).code() == StatusCode::kBusy);
  context->request.output_block_bytes = 64;
  CHECK(context->lease.view().Launch(context->invocation()).ok());
  CHECK(manager.Shutdown().code() == StatusCode::kBusy);
  CHECK(c.closes == 0 && c.live_states == 1 && c.owners == 1);
  CHECK(manager.Recover(get).code() == StatusCode::kNotInitialized);
  c.fail_close = true;
  CHECK(!manager.Close(context->lease).ok());
  CHECK(c.live_states == 1 && c.owners == 1 && c.operators == 1);
  c.fail_close = false;
  CHECK(manager.Close(context->lease).ok());
  // Close may drain through the owner but may not destroy the borrowed operator
  // or state before the exclusive lease holder has unbound its invocation.
  CHECK(c.live_states == 1 && c.owners == 1 && c.operators == 1);
  context.reset();
  CHECK(c.close_with_state == 2);
  CHECK(manager.Shutdown().ok());
  CheckDestroyed(c);
  CHECK(c.events == std::vector<std::string>({"owner closed", "state destroyed",
                                           "operator destroyed", "owner destroyed"}));
}

void ConcurrentFirstUseAndDirectionIsolation() {
  Counters p, g;
  ExecutionResourceManager manager;
  CHECK(manager.Register(put, Factory(p)).ok());
  CHECK(manager.Register(get, Factory(g)).ok());
  CHECK(manager.Initialize(region).ok());
  g.block_initialize = true;
  auto first = std::async(std::launch::async, [&] {
    GetResourceLease lease;
    return manager.AcquireGet({get, 1}, &lease);
  });
  bool entered;
  {
    std::unique_lock<std::mutex> lock(g.mutex);
    entered = g.changed.wait_for(lock, 2s, [&] { return g.initialize_entered; });
  }
  auto second = std::async(std::launch::async, [&] {
    GetResourceLease lease;
    return manager.AcquireGet({get, 1}, &lease);
  });
  auto opposite = std::async(std::launch::async, [&] {
    PutResourceLease lease;
    return manager.AcquirePut({put, 1}, &lease);
  });
  const bool put_ready = opposite.wait_for(2s) == std::future_status::ready;
  const bool one_create = g.factories == 1;
  // Always release the gate before asserting, so a failed test cannot strand
  // async futures (and hide the actual failure behind the test timeout).
  {
    std::lock_guard<std::mutex> lock(g.mutex);
    g.block_initialize = false;
  }
  g.changed.notify_all();
  const auto first_status = first.get(), second_status = second.get(), put_status = opposite.get();
  CHECK(entered && put_ready && one_create);
  CHECK(first_status.ok() && put_status.ok());
  CHECK(second_status.ok() || second_status.code() == StatusCode::kBusy);
  GetResourceLease after_first;
  CHECK(manager.AcquireGet({get, 1}, &after_first).ok());
  after_first = {};
  CHECK(g.factories == 1 && g.initializes == 1 && p.factories == 1);
  CHECK(manager.Shutdown().ok());
  CheckDestroyed(g); CheckDestroyed(p);
}

void CapacityFailureDoesNotPoisonResources() {
  Counters p, g;
  p.capacity = 2;
  g.capacity = 3;
  ExecutionResourceManager manager;
  CHECK(manager.Register(put, Factory(p)).ok());
  CHECK(manager.Register(get, Factory(g)).ok());
  CHECK(manager.Initialize(region).ok());
  PutResourceLease write;
  GetResourceLease read;
  CHECK(manager.AcquirePut({put, 3}, &write).code() == StatusCode::kInvalidArgument);
  CHECK(manager.AcquireGet({get, 4}, &read).code() == StatusCode::kInvalidArgument);
  CHECK(!write && !read);
  CHECK(manager.AcquirePut({put, 2}, &write).ok());
  CHECK(manager.AcquireGet({get, 3}, &read).ok());
  CHECK(p.factories == 1 && g.factories == 1);
  write = {}; read = {};
  CHECK(manager.Shutdown().ok());
  CheckDestroyed(p); CheckDestroyed(g);
}

void FailedInitializationRequiresExplicitRecovery() {
  Counters c;
  ExecutionResourceManager manager;
  CHECK(manager.Register(get, Factory(c)).ok());
  CHECK(manager.Initialize(region).ok());
  c.fail_initialize = true;
  GetResourceLease lease;
  CHECK(!manager.AcquireGet({get, 1}, &lease).ok() && !lease);
  c.fail_initialize = false;
  CHECK(!manager.AcquireGet({get, 1}, &lease).ok() && !lease);
  CHECK(c.factories == 1 && c.owners == 1);
  c.fail_close = true;
  CHECK(!manager.Recover(get).ok());
  CHECK(!manager.AcquireGet({get, 1}, &lease).ok());
  CHECK(c.factories == 1 && c.owners == 1);
  c.fail_close = false;
  CHECK(manager.Recover(get).ok());
  CHECK(c.factories == 2 && c.initializes == 2);
  CHECK(manager.AcquireGet({get, 1}, &lease).ok());
  lease = {};
  CHECK(manager.Shutdown().ok());
  CheckDestroyed(c);
}

void FailedCloseMustNotReplaceLiveOwner() {
  Counters c;
  ExecutionResourceManager manager;
  CHECK(manager.Register(get, Factory(c)).ok());
  CHECK(manager.Initialize(region).ok());
  GetResourceLease lease;
  CHECK(manager.AcquireGet({get, 1}, &lease).ok());
  lease = {};
  c.fail_close = true;
  CHECK(!manager.Recover(get).ok());
  const auto acquire = manager.AcquireGet({get, 1}, &lease);
  const bool retained = !acquire.ok() && !lease && c.factories == 1 && c.owners == 1;
  // Clean up even when regressing: old code replaced the owner here.
  lease = {};
  c.fail_close = false;
  CHECK(manager.Recover(get).ok());
  CHECK(manager.Shutdown().ok());
  CHECK(retained);
  CheckDestroyed(c);
}

void PutStateSurvivesResolutionAndAcknowledgementFailure() {
  Counters p, g;
  ExecutionDispatcher dispatch(Policy());
  Register(dispatch, p, g);
  p.fail_put = p.fail_resolve = true;
  char data[64]{};
  BatchPutRequest request{{42}, {data, sizeof(data), DeviceKind::kCpu}, {0}, {64}};
  PutAccessDesc destination{{{42, 4096, 64}}};
  CHECK(dispatch.ExecuteBatchPut({&request, &destination, {}}).code() == StatusCode::kCompletionUnknown);
  request.keys[0] = 99;
  destination.entries[0].address = 9999;
  CHECK(p.live_states == 1 && p.owners == 1 && p.put_launches == 1);
  CHECK(dispatch.Warmup(put).code() == StatusCode::kBusy);
  CHECK(dispatch.Warmup(get).ok());
  PutCompletion completion;
  CHECK(!dispatch.ResolvePut(put, &completion).ok());
  CHECK(p.live_states == 1 && p.copied_metadata_ok);
  p.fail_resolve = false;
  CHECK(dispatch.ResolvePut(put, &completion).ok());
  CHECK(completion.outcome == PutOutcome::Succeeded && p.put_launches == 1);
  p.fail_ack = true;
  CHECK(!dispatch.AcknowledgePut().ok());
  CHECK(p.live_states == 1 && p.owners == 1);
  p.fail_ack = false;
  CHECK(dispatch.AcknowledgePut().ok());
  CHECK(p.live_states == 0 && p.owners == 1 && p.state_destroys == 1);
  CHECK(dispatch.Shutdown().ok());
  CheckDestroyed(p); CheckDestroyed(g);
}

void GetFailureRetainsCopiedContextThroughShutdown(bool throw_get) {
  Counters p, g;
  ExecutionDispatcher dispatch(Policy());
  Register(dispatch, p, g);
  g.fail_get = g.fail_drain = g.fail_close = true;
  g.throw_get = throw_get;
  BatchGetRequest request;
  request.output_block_bytes = 64;
  CHECK(!dispatch.ExecuteBatchGet({{}, &request, {}}).ok());
  request.output_block_bytes = 999;
  CHECK(!dispatch.DrainGet().ok());
  CHECK(g.copied_metadata_ok && g.live_states == 1 && g.owners == 1);
  CHECK(dispatch.Warmup(put).ok());
  CHECK(!dispatch.Shutdown().ok());
  CHECK(g.live_states == 1 && g.owners == 1 && g.operators == 1);
  g.fail_close = false;
  CHECK(dispatch.Shutdown().ok());
  CHECK(g.close_with_state == 2 && g.state_destroys == 1);
  CheckDestroyed(g); CheckDestroyed(p);
}

void FailedGraphPreparationMustRetryClose() {
  Counters p, g;
  ExecutionDispatcher dispatch(Policy());
  Register(dispatch, p, g);
  g.fail_close = true;
  std::uint64_t id = 123;
  CHECK(!dispatch.BeginGetGraph(get, {}, &id).ok() && id == 0);
  CHECK(g.live_states == 1 && g.closes == 1);
  // No Get was submitted: the provider's ordinary Drain would succeed. Only
  // retrying the failed owner close can safely release the graph registration.
  CHECK(!dispatch.DrainGet().ok());
  CHECK(g.live_states == 1 && g.closes == 2);
  CHECK(dispatch.Warmup(put).ok());
  g.fail_close = false;
  CHECK(dispatch.DrainGet().ok());
  CHECK(g.live_states == 0 && g.closes == 3);
  CheckDestroyed(g);
  CHECK(dispatch.Warmup(get).ok());
  CHECK(g.factories == 2);
  CHECK(dispatch.Shutdown().ok());
  CheckDestroyed(g); CheckDestroyed(p);
}
}  // namespace
int main() {
  try {
    ExclusiveLeaseAndIndependentOwner();
    ConcurrentFirstUseAndDirectionIsolation();
    CapacityFailureDoesNotPoisonResources();
    FailedInitializationRequiresExplicitRecovery();
    FailedCloseMustNotReplaceLiveOwner();
    PutStateSurvivesResolutionAndAcknowledgementFailure();
    GetFailureRetainsCopiedContextThroughShutdown(false);
    GetFailureRetainsCopiedContextThroughShutdown(true);
    FailedGraphPreparationMustRetryClose();
    std::cout << "Resource ownership, exclusive leases, parallel initialization and retained contexts passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
