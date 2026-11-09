#include "execution_dispatcher.h"
#include <exception>

namespace kvaf {
namespace {
Status LaunchGet(GetExecutionContext& context) {
  try {
    return context.lease.view().Launch(context.invocation());
  } catch (const std::exception& e) {
    return Status::CompletionUnknown(std::string("provider Get threw after submission: ") + e.what());
  } catch (...) {
    return Status::CompletionUnknown("provider Get threw after submission");
  }
}
}  // namespace
ExecutionDispatcher::ExecutionDispatcher(ExecutionPolicy policy) : policy_(std::move(policy)) {
  policy_status_ = NormalizeExecutionPolicy(&policy_);
}
ExecutionDispatcher::~ExecutionDispatcher() {
  if (!Shutdown().ok()) {
    // Retain invocation state as well as the resource lease after unsafe close.
    (void)pending_put_.release();
    (void)pending_get_.release();
    (void)graph_context_.release();
  }
}
Status ExecutionDispatcher::RegisterPut(PutExecutionConfig c, Factory f) {
  std::unique_lock<std::shared_mutex> lock(lifecycle_);
  if (ready_ || region_.base) return Status::InvalidState("register providers before initialization");
  auto s = ValidateExecution(policy_.transport_mode, c);
  return s.ok() ? resources_.Register(c, std::move(f)) : s;
}
Status ExecutionDispatcher::RegisterGet(GetExecutionConfig c, Factory f) {
  std::unique_lock<std::shared_mutex> lock(lifecycle_);
  if (ready_ || region_.base) return Status::InvalidState("register providers before initialization");
  auto s = ValidateExecution(policy_.transport_mode, c);
  return s.ok() ? resources_.Register(c, std::move(f)) : s;
}
Status ExecutionDispatcher::Initialize(const RemoteRegionDesc& region) {
  std::unique_lock<std::shared_mutex> lock(lifecycle_);
  if (!policy_status_.ok()) return policy_status_;
  if (ready_ || region_.base) return Status::InvalidState("dispatcher requires shutdown before initialize");
  for (auto c : *policy_.enabled_put_configs)
    if (!resources_.Contains(c)) return Status::NotImplemented("enabled Put provider is not built or registered");
  for (auto c : *policy_.enabled_get_configs)
    if (!resources_.Contains(c))
      return Status::NotImplemented("enabled Get provider is not built or registered (AIV Gather/Scatter reserved)");
  auto s = resources_.Initialize(region);
  if (s.ok()) { region_ = region; ready_ = true; }
  return s;
}
Status ExecutionDispatcher::PreparePut(const BatchPutExecutionOptions& options) {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(put_mutex_);
  if (!ready_) return Status::NotInitialized("dispatcher is not accepting work");
  if (pending_put_) return Status::InvalidState("Put awaits completion/transaction acknowledgement");
  PutExecutionConfig c;
  auto s = ResolveExecution(policy_, options.config, &c);
  PutResourceLease lease;
  return s.ok() ? resources_.AcquirePut({c}, &lease) : s;
}
Status ExecutionDispatcher::PreparePut(const BatchPutRequest& request,
                                       const BatchPutExecutionOptions& options) {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(put_mutex_);
  if (!ready_) return Status::NotInitialized("dispatcher is not accepting work");
  if (pending_put_) return Status::InvalidState("Put awaits completion/transaction acknowledgement");
  PutExecutionConfig c;
  auto s = ResolveExecution(policy_, options.config, &c);
  PutResourceLease lease;
  if (s.ok()) s = resources_.AcquirePut({c, static_cast<std::uint32_t>(request.keys.size())}, &lease);
  return s.ok() ? lease.view().Prepare(request, options) : s;
}
Status ExecutionDispatcher::Recover(PutExecutionConfig c) {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(put_mutex_);
  if (!ready_) return Status::NotInitialized("dispatcher is not accepting work");
  if (pending_put_) return Status::Busy("resolve pending Put through Runtime before resource recovery");
  PutExecutionConfig resolved;
  auto s = ResolveExecution(policy_, c, &resolved);
  return s.ok() ? resources_.Recover(resolved) : s;
}
Status ExecutionDispatcher::Recover(GetExecutionConfig c) {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(get_mutex_);
  if (!ready_) return Status::NotInitialized("dispatcher is not accepting work");
  GetExecutionConfig resolved;
  auto s = ResolveExecution(policy_, c, &resolved);
  if (s.ok()) s = DrainGetLocked();
  return s.ok() ? resources_.Recover(resolved) : s;
}
Status ExecutionDispatcher::ExecuteBatchPut(const BatchPutExecRequest& request) {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(put_mutex_);
  if (!ready_) return Status::NotInitialized("dispatcher is not accepting work");
  if (pending_put_) return Status::InvalidState("Put awaits completion/transaction acknowledgement");
  if (!request.request || !request.put_desc) return Status::InvalidArgument("missing Put request");
  auto context = std::make_unique<PutExecutionContext>();
  auto s = ResolveExecution(policy_, request.execution.config, &context->config);
  if (!s.ok()) return s;
  try {
    context->request = *request.request;
    context->destination = *request.put_desc;
    context->options = request.execution;
    context->options.config = context->config;
    s = resources_.AcquirePut({context->config, static_cast<std::uint32_t>(context->request.keys.size())}, &context->lease);
    if (s.ok()) s = context->Bind();
  } catch (const std::exception& e) {
    s = Status::Internal(std::string("Put preparation failed before submission: ") + e.what());
  } catch (...) {
    s = Status::Internal("Put preparation failed before submission");
  }
  if (!s.ok()) {
    context->completion = {PutOutcome::Failed, s};
    pending_put_ = std::move(context);
    return s;
  }
  pending_put_ = std::move(context);
  pending_put_->launched = true;
  try {
    s = pending_put_->lease.view().Launch(pending_put_->invocation());
  } catch (const std::exception& e) {
    s = Status::CompletionUnknown(std::string("provider Put threw after submission: ") + e.what());
  } catch (...) {
    s = Status::CompletionUnknown("provider Put threw after submission");
  }
  if (s.ok()) pending_put_->completion = {PutOutcome::Succeeded, Status::OK()};
  else pending_put_->completion.result = s;
  return s.ok() ? s : Status::CompletionUnknown(s.message());
}
Status ExecutionDispatcher::ResolvePut(PutExecutionConfig c, PutCompletion* out) {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(put_mutex_);
  if (!out) return Status::InvalidArgument("null Put completion output");
  if (!pending_put_) return Status::NotFound("no pending Put execution");
  if (!(pending_put_->config == c)) return Status::InvalidArgument("recovery configuration differs from original Put");
  if (pending_put_->completion.outcome == PutOutcome::Unknown) {
    PutCompletion result;
    auto s = pending_put_->lease.view().Resolve(&result);
    if (!s.ok()) return s;
    if ((result.outcome == PutOutcome::Succeeded && !result.result.ok()) ||
        (result.outcome == PutOutcome::Failed && result.result.ok()))
      return Status::Internal("inconsistent provider Put completion");
    pending_put_->completion = result;
  }
  *out = pending_put_->completion;
  return Status::OK();
}
Status ExecutionDispatcher::AcknowledgePut() {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(put_mutex_);
  if (!pending_put_) return Status::OK();
  if (pending_put_->completion.outcome == PutOutcome::Unknown)
    return Status::Busy("cannot acknowledge unresolved Put");
  auto s = pending_put_->launched ? pending_put_->lease.view().Acknowledge() : Status::OK();
  if (s.ok()) pending_put_.reset();
  return s;
}
Status ExecutionDispatcher::DrainGetLocked() {
  if (graph_id_) return Status::Busy("Get graph lease owns this direction");
  if (!pending_get_) return Status::OK();
  auto s = pending_get_->cleanup_only ? resources_.Close(pending_get_->lease)
                                      : pending_get_->lease.view().Drain();
  if (s.ok()) pending_get_.reset();
  return s;
}
Status ExecutionDispatcher::ExecuteBatchGet(const BatchGetExecRequest& request) {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(get_mutex_);
  if (!ready_) return Status::NotInitialized("dispatcher is not accepting work");
  if (!request.request) return Status::InvalidArgument("missing Get request");
  GetExecutionConfig c;
  auto s = ResolveExecution(policy_, request.execution.config, &c);
  if (!s.ok()) return s;
  if (graph_id_ || request.execution.graph_lease) {
    if (!graph_id_ || request.execution.graph_lease != graph_id_ || !(c == graph_context_->config))
      return Status::Busy("Get graph lease/configuration mismatch");
    if (graph_captured_ || request.execution.synchronize || request.execution.profile)
      return Status::InvalidState("one asynchronous Get capture is allowed per lease");
    graph_context_->request = *request.request;
    graph_context_->table = request.host_access_table;
    graph_context_->options = request.execution;
    graph_context_->options.config = c;
    graph_captured_ = true;
    return LaunchGet(*graph_context_);
  }
  s = DrainGetLocked();
  if (!s.ok()) return s;
  auto context = std::make_unique<GetExecutionContext>();
  context->config = c;
  context->request = *request.request;
  context->table = request.host_access_table;
  context->options = request.execution;
  context->options.config = c;
  s = resources_.AcquireGet({c, static_cast<std::uint32_t>(request.request->indices.bytes / 4)}, &context->lease);
  if (s.ok()) s = context->Bind();
  if (!s.ok()) return s;
  pending_get_ = std::move(context);
  s = LaunchGet(*pending_get_);
  if (s.ok() && request.execution.synchronize) pending_get_.reset();
  return s;
}
Status ExecutionDispatcher::DrainGet() {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(get_mutex_);
  return DrainGetLocked();
}
Status ExecutionDispatcher::Warmup(PutExecutionConfig c, const ExecutionWarmup& r) {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(put_mutex_);
  if (pending_put_) return Status::Busy("Put awaits completion/transaction acknowledgement");
  PutExecutionConfig resolved;
  auto s = ResolveExecution(policy_, c, &resolved);
  PutResourceLease lease;
  if (s.ok()) s = resources_.AcquirePut({resolved, r.request_capacity}, &lease);
  return s.ok() ? lease.view().Warmup(r) : s;
}
Status ExecutionDispatcher::Warmup(GetExecutionConfig c, const ExecutionWarmup& r) {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(get_mutex_);
  GetExecutionConfig resolved;
  auto s = ResolveExecution(policy_, c, &resolved);
  if (s.ok()) s = DrainGetLocked();
  GetResourceLease lease;
  if (s.ok()) s = resources_.AcquireGet({resolved, r.request_capacity}, &lease);
  return s.ok() ? lease.view().Warmup(r) : s;
}
Status ExecutionDispatcher::BeginGetGraph(GetExecutionConfig c, const GetGraphOptions& r, std::uint64_t* id) {
  if (!id) return Status::InvalidArgument("null graph lease output");
  *id = 0;
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(get_mutex_);
  if (c.executor != Executor::Aiv || c.algorithm != GetAlgorithm::Direct)
    return Status::NotImplemented("only AIV Direct Get supports graph leases");
  auto context = std::make_unique<GetExecutionContext>();
  auto s = ResolveExecution(policy_, c, &context->config);
  if (s.ok()) s = DrainGetLocked();
  if (s.ok()) s = resources_.AcquireGet({c, r.request_capacity}, &context->lease);
  if (s.ok()) s = context->Bind();
  if (!s.ok()) return s;
  if (!next_graph_id_) return Status::InvalidState("graph lease ID exhausted");
  try {
    s = context->lease.view().PrepareGraph(r);
  } catch (const std::exception& e) {
    s = Status::Internal(std::string("graph preparation failed: ") + e.what());
  } catch (...) {
    s = Status::Internal("graph preparation failed");
  }
  if (!s.ok()) {
    context->cleanup_only = true;
    const auto closed = resources_.Close(context->lease);
    if (!closed.ok()) pending_get_ = std::move(context);
    return s;
  }
  graph_id_ = next_graph_id_++;
  graph_context_ = std::move(context);
  graph_captured_ = false;
  *id = graph_id_;
  return s;
}
Status ExecutionDispatcher::EndGetGraph(std::uint64_t id, void* stream, bool discard) {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(get_mutex_);
  if (!id || id != graph_id_) return Status::NotFound("Get graph lease not found");
  auto s = graph_context_->lease.view().FinishGraph(stream, discard);
  if (s.ok()) { graph_id_ = 0; graph_context_.reset(); graph_captured_ = false; }
  return s;
}
Status ExecutionDispatcher::Shutdown() {
  std::unique_lock<std::shared_mutex> life(lifecycle_);
  if (graph_id_) return Status::Busy("release Get graph lease before shutdown");
  ready_ = false;
  Status first;
  auto close = [&](auto& context) {
    if (!context) return;
    auto s = resources_.Close(context->lease);
    if (s.ok()) context.reset();
    else if (first.ok()) first = s;
  };
  close(pending_put_);
  close(pending_get_);
  auto s = resources_.Shutdown();
  if (!s.ok() && first.ok()) first = s;
  if (first.ok()) region_ = {};
  return first;
}
}  // namespace kvaf
