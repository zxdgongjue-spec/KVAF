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
  std::atomic<int> state_creates{0}, binds{0}, unbinds{0};
  std::atomic<int> put_launches{0}, get_launches{0}, resolves{0};
  std::atomic<int> legacy_initializes{0}, legacy_warmups{0}, warmups{0}, descriptions{0};
  std::uint32_t requirement_padding{0}, warmed_capacity{0};
  std::atomic<bool> fail_initialize{false}, fail_close{false}, fail_drain{false};
  std::atomic<bool> fail_put{false}, fail_resolve{false}, fail_ack{false}, fail_get{false};
  bool throw_get{false}, throw_put{false};
  std::atomic<bool> bad_order{false}, copied_metadata_ok{true};
  bool put_request_borrowed{false}, put_state_ok{true};
  const BatchPutRequest* expected_put_request{};
  const PutAccessDesc* expected_put_destination{};
  const BatchGetRequest* last_get_request{};
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
  explicit Invocation(Counters& c) : counters(c) { ++counters.live_states; ++counters.state_creates; }
  ~Invocation() override {
    --counters.live_states;
    ++counters.state_destroys;
    counters.Event("state destroyed");
  }
  Counters& counters;
  // A provider retains its submitted state, not the caller's request vectors.
  std::size_t put_rows{};
  KeyId put_key{};
  Address put_destination{};
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
  Status Initialize(const RemoteRegionDesc&) override {
    auto& c = physical.counters;
    ++c.initializes;
    std::unique_lock<std::mutex> lock(c.mutex);
    c.initialize_entered = true;
    c.changed.notify_all();
    c.changed.wait(lock, [&] { return !c.block_initialize; });
    return c.fail_initialize ? Status::Internal("initialization failed") : Status::OK();
  }
  Status Warmup(const ExecutionWarmup& r) override {
    ++physical.counters.warmups;
    physical.counters.warmed_capacity = r.request_capacity;
    return Status::OK();
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
    if (!state) { ++physical_->counters.unbinds; physical_->invocation = nullptr; return Status::OK(); }
    auto* invocation = dynamic_cast<Invocation*>(state);
    if (!invocation || physical_->invocation) return Status::InvalidState("invalid invocation bind");
    physical_->invocation = invocation;
    ++physical_->counters.binds;
    return Status::OK();
  }
  Status Initialize(const RemoteRegionDesc&) override {
    ++physical_->counters.legacy_initializes;
    return Status::Internal("borrowed operator must not initialize resources");
  }
  Status Warmup(const ExecutionWarmup&) override {
    ++physical_->counters.legacy_warmups;
    return Status::Internal("borrowed operator must not allocate workspace");
  }
  Status DescribeRequirements(const BatchGetRequest& r, std::uint32_t* capacity) const override {
    ++physical_->counters.descriptions;
    if (owner_) physical_->counters.bad_order = true;
    auto s = AccessOperator::DescribeRequirements(r, capacity);
    if (!s.ok()) return s;
    *capacity += physical_->counters.requirement_padding;
    return ValidateRequestCapacity(*capacity);
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
    c.put_request_borrowed = e.request == c.expected_put_request &&
                             e.put_desc == c.expected_put_destination;
    CHECK(e.request && e.put_desc && e.request->keys.size() == 1 &&
          e.put_desc->entries.size() == 1);
    state->put_rows = e.request->keys.size();
    state->put_key = e.request->keys[0];
    state->put_destination = e.put_desc->entries[0].address;
    state->pending_put = true;
    if (c.throw_put) throw std::runtime_error("WRITE threw after submission");
    return c.fail_put ? Status::CompletionUnknown("WRITE submitted, wait interrupted") : Status::OK();
  }
  Status ResolvePut(PutCompletion* out) override {
    auto& c = physical_->counters;
    auto* state = physical_->invocation;
    if (!state || !state->pending_put) return Status::InvalidState("missing original WRITE");
    ++c.resolves;
    if (state->put_rows != 1 || state->put_key != 42 || state->put_destination != 4096)
      c.put_state_ok = false;
    if (c.fail_resolve) return Status::CompletionUnknown("original WRITE still pending");
    *out = {PutOutcome::Succeeded, Status::OK()};
    return Status::OK();
  }
  Status AcknowledgePut() override {
    auto* state = physical_->invocation;
    if (!state || state->put_rows != 1 || state->put_key != 42 || state->put_destination != 4096)
      physical_->counters.put_state_ok = false;
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
    c.last_get_request = e.request;
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
  CHECK(!c.bad_order && c.legacy_shutdowns == 0 && c.legacy_initializes == 0 && c.legacy_warmups == 0);
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

void DescriptionBeforeAllocation() {
  Counters c;
  c.capacity = 8;
  c.requirement_padding = 2; // The provider, not the dispatcher, specifies capacity.
  ExecutionResourceManager manager;
  CHECK(manager.Register(get, Factory(c)).ok());
  CHECK(manager.Initialize(region).ok());
  BatchGetRequest request;
  request.indices.bytes = 7 * sizeof(std::int32_t);
  GetResourceRequirements requirements;
  CHECK(manager.Describe(get, request, &requirements).code() == StatusCode::kInvalidArgument);
  CHECK(c.factories == 1 && c.initializes == 0);
  request.indices.bytes = 6 * sizeof(std::int32_t);
  CHECK(manager.Describe(get, request, &requirements).ok());
  CHECK(requirements.config == get && requirements.capacity == 8 && c.initializes == 0);
  GetResourceLease lease;
  CHECK(manager.AcquireGet(requirements, &lease).ok() && c.initializes == 1);
  CHECK(lease.view().Warmup({requirements.capacity, nullptr}).ok());
  CHECK(c.warmups == 1 && c.warmed_capacity == 8);
  lease = {};
  request.indices.bytes = 3;
  CHECK(manager.Describe(get, request, &requirements).code() == StatusCode::kInvalidArgument);
  request.indices.bytes = (std::uint64_t(UINT32_MAX) + 1) * sizeof(std::int32_t);
  CHECK(manager.Describe(get, request, &requirements).code() == StatusCode::kInvalidArgument);
  CHECK(manager.Shutdown().ok());
  CheckDestroyed(c);
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

void ResidentReferenceIsNotExecutionOccupancy() {
  Counters c;
  ExecutionResourceManager manager;
  CHECK(manager.Register(get, Factory(c)).ok());
  CHECK(manager.Initialize(region).ok());
  auto context = std::make_unique<GetExecutionContext>();
  CHECK(manager.PinGet({get, 4}, &context->resources).ok());
  CHECK(!context->lease && context->Bind().ok());
  GetResourceLease first, competing;
  CHECK(context->resources.Acquire(&first).ok());
  CHECK(context->resources.Acquire(&competing).code() == StatusCode::kBusy);
  first = {};
  CHECK(context->resources.Acquire(&competing).ok());
  competing = {};
  CHECK(c.state_creates == 1 && c.binds == 1 && c.initializes == 1);
  CHECK(manager.AcquireGet({get, 4}, &competing).code() == StatusCode::kBusy);
  CHECK(manager.Recover(get).code() == StatusCode::kBusy);
  CHECK(manager.Shutdown().code() == StatusCode::kBusy);
  CHECK(c.closes == 0 && c.live_states == 1);
  CHECK(context->resources.Acquire(&competing).code() == StatusCode::kNotInitialized);
  CHECK(manager.Close(context->resources).ok());
  CHECK(context->resources.Acquire(&competing).code() == StatusCode::kNotInitialized);
  CHECK(c.close_with_state == 1 && c.operators == 1);
  context.reset();
  CHECK(manager.Shutdown().ok());
  CheckDestroyed(c);
}

void ResidentContextReusesPreparationAndRetiresRequest() {
  Counters p, g;
  g.capacity = 8;
  g.requirement_padding = 2;
  ExecutionDispatcher dispatch(Policy());
  Register(dispatch, p, g);
  BatchGetRequest request;
  request.indices.bytes = 4 * sizeof(std::int32_t);
  request.output_block_bytes = 64;
  BatchGetExecutionOptions options;
  options.config = get;
  options.stream = reinterpret_cast<void*>(1);
  for (int i = 0; i < 4; ++i) {
    CHECK(dispatch.ExecuteBatchGet({{}, &request, options}).ok());
    CHECK(dispatch.DrainGet().ok());
    CHECK(g.last_get_request->indices.bytes == 0 && g.last_get_request->output_block_bytes == 0);
  }
  CHECK(g.factories == 1 && g.initializes == 1 && g.state_creates == 1 && g.binds == 1);
  CHECK(g.live_states == 1 && g.state_destroys == 0 && g.unbinds == 0);
  CHECK(g.descriptions == 1 && g.warmups == 1 && g.warmed_capacity == 6);
  request.indices.bytes = 6 * sizeof(std::int32_t);
  CHECK(dispatch.ExecuteBatchGet({{}, &request, options}).ok());
  CHECK(dispatch.DrainGet().ok());
  CHECK(g.descriptions == 2 && g.warmups == 2 && g.warmed_capacity == 8);
  request.indices.bytes = 7 * sizeof(std::int32_t);
  CHECK(dispatch.ExecuteBatchGet({{}, &request, options}).code() == StatusCode::kInvalidArgument);
  CHECK(g.warmups == 2 && g.state_creates == 1);
  request.indices.bytes = 3;
  CHECK(dispatch.ExecuteBatchGet({{}, &request, options}).code() == StatusCode::kInvalidArgument);
  CHECK(g.descriptions == 3);
  request.indices.bytes = 4 * sizeof(std::int32_t);
  options.stream = reinterpret_cast<void*>(2);
  CHECK(dispatch.ExecuteBatchGet({{}, &request, options}).ok());
  CHECK(dispatch.DrainGet().ok());
  CHECK(g.warmups == 3 && g.warmed_capacity == 8 && g.state_creates == 1);
  CHECK(dispatch.ExecuteBatchGet({{}, &request, options}).ok());
  g.fail_drain = true;
  CHECK(!dispatch.Recover(get).ok());
  CHECK(g.live_states == 1 && g.factories == 1);
  g.fail_drain = false;
  CHECK(dispatch.Recover(get).ok());
  CHECK(g.state_destroys == 1 && g.factories == 2 && g.initializes == 2);
  CHECK(dispatch.ExecuteBatchGet({{}, &request, options}).ok());
  CHECK(dispatch.DrainGet().ok());
  CHECK(g.state_creates == 2 && g.binds == 2 && g.unbinds == 1);
  CHECK(dispatch.Shutdown().ok());
  CheckDestroyed(g); CheckDestroyed(p);
}

void FailedResidentCloseRetainsOnlyAffectedInstance() {
  constexpr GetExecutionConfig second{Transport::Urma, Executor::Aicpu, GetAlgorithm::Gather};
  Counters p, g, other;
  auto policy = Policy();
  policy.enabled_get_configs->push_back(second);
  ExecutionDispatcher dispatch(policy);
  CHECK(dispatch.RegisterPut(put, Factory(p)).ok());
  CHECK(dispatch.RegisterGet(get, Factory(g)).ok());
  CHECK(dispatch.RegisterGet(second, Factory(other)).ok());
  CHECK(dispatch.Initialize(region).ok());
  CHECK(dispatch.Warmup(get, {4, nullptr}).ok());
  CHECK(dispatch.Warmup(second, {4, nullptr}).ok());
  CHECK(g.live_states == 1 && other.live_states == 1);
  g.fail_close = true;
  CHECK(!dispatch.Shutdown().ok());
  CHECK(g.live_states == 1 && g.owners == 1 && g.operators == 1);
  CheckDestroyed(other);
  g.fail_close = false;
  CHECK(dispatch.Shutdown().ok());
  CHECK(g.close_with_state == 2 && g.unbinds == 1);
  CheckDestroyed(g); CheckDestroyed(p);
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

void PutStateSurvivesResolutionAndAcknowledgementFailure(bool throw_put) {
  Counters p, g;
  ExecutionDispatcher dispatch(Policy());
  Register(dispatch, p, g);
  p.fail_put = p.fail_resolve = true;
  p.throw_put = throw_put;
  char data[64]{};
  {
    BatchPutRequest request{{42}, {data, sizeof(data), DeviceKind::kCpu}, {0}, {64}};
    PutAccessDesc destination{{{42, 4096, 64}}};
    p.expected_put_request = &request;
    p.expected_put_destination = &destination;
    CHECK(dispatch.ExecuteBatchPut({&request, &destination, {}}).code() == StatusCode::kCompletionUnknown);
    CHECK(p.put_request_borrowed); // Launch sees the originals, without dispatcher copies.
    request.keys[0] = 99;
    destination.entries[0].address = 9999;
    PutCompletion unresolved;
    CHECK(!dispatch.ResolvePut(put, &unresolved).ok() && p.put_state_ok);
  }
  // Request objects and their vector storage are gone; input storage remains
  // alive until the original write is resolved and acknowledged.
  p.expected_put_request = nullptr;
  p.expected_put_destination = nullptr;
  CHECK(p.live_states == 1 && p.owners == 1 && p.put_launches == 1);
  CHECK(dispatch.Warmup(put).code() == StatusCode::kBusy);
  CHECK(dispatch.Warmup(get).ok());
  PutCompletion completion;
  CHECK(!dispatch.ResolvePut(put, &completion).ok());
  CHECK(p.live_states == 1 && p.put_state_ok);
  p.fail_resolve = false;
  CHECK(dispatch.ResolvePut(put, &completion).ok());
  CHECK(completion.outcome == PutOutcome::Succeeded && p.put_launches == 1);
  p.fail_ack = true;
  CHECK(!dispatch.AcknowledgePut().ok());
  CHECK(p.live_states == 1 && p.owners == 1 && p.put_state_ok);
  p.fail_ack = false;
  CHECK(dispatch.AcknowledgePut().ok());
  CHECK(p.live_states == 0 && p.owners == 1 && p.state_destroys == 1 && p.put_state_ok);
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
  std::uint64_t invalid_id = 123;
  CHECK(dispatch.BeginGetGraph(get, {}, &invalid_id).code() == StatusCode::kInvalidArgument);
  CHECK(invalid_id == 0 && g.factories == 0 && g.initializes == 0 && g.warmups == 0);
  g.fail_close = true;
  std::uint64_t id = 123;
  CHECK(!dispatch.BeginGetGraph(get, {{}, nullptr, 1}, &id).ok() && id == 0);
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
    DescriptionBeforeAllocation();
    ExclusiveLeaseAndIndependentOwner();
    ResidentReferenceIsNotExecutionOccupancy();
    ResidentContextReusesPreparationAndRetiresRequest();
    FailedResidentCloseRetainsOnlyAffectedInstance();
    ConcurrentFirstUseAndDirectionIsolation();
    CapacityFailureDoesNotPoisonResources();
    FailedInitializationRequiresExplicitRecovery();
    FailedCloseMustNotReplaceLiveOwner();
    PutStateSurvivesResolutionAndAcknowledgementFailure(false);
    PutStateSurvivesResolutionAndAcknowledgementFailure(true);
    GetFailureRetainsCopiedContextThroughShutdown(false);
    GetFailureRetainsCopiedContextThroughShutdown(true);
    FailedGraphPreparationMustRetryClose();
    std::cout << "Resource ownership, exclusive leases, parallel initialization and retained contexts passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
