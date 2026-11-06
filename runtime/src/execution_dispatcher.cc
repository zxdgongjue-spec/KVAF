#include "execution_dispatcher.h"
#include <algorithm>
#include <exception>

namespace kvaf {
ExecutionDispatcher::ExecutionDispatcher(ExecutionPolicy policy) : policy_(std::move(policy)) {
  policy_status_ = NormalizeExecutionPolicy(&policy_);
}
ExecutionDispatcher::~ExecutionDispatcher() {
  if (!Shutdown().ok()) {
    // Failed shutdown cannot relinquish DMA-owned memory through destructors.
    for (auto& r : puts_) (void)r.owner.release();
    for (auto& r : gets_) (void)r.owner.release();
  }
}
Status ExecutionDispatcher::RegisterPut(PutExecutionConfig c, Factory f) {
  std::unique_lock<std::shared_mutex> lock(lifecycle_);
  if (ready_ || region_.base) return Status::InvalidState("register providers before initialization");
  if (!f) return Status::InvalidArgument("null provider factory");
  auto s = ValidateExecution(policy_.transport_mode, c);
  if (!s.ok()) return s;
  for (auto& r : puts_) if (r.config == c) return Status::AlreadyExists("Put provider already registered");
  puts_.push_back({c, std::move(f), {}, {}, false});
  return Status::OK();
}
Status ExecutionDispatcher::RegisterGet(GetExecutionConfig c, Factory f) {
  std::unique_lock<std::shared_mutex> lock(lifecycle_);
  if (ready_ || region_.base) return Status::InvalidState("register providers before initialization");
  if (!f) return Status::InvalidArgument("null provider factory");
  auto s = ValidateExecution(policy_.transport_mode, c);
  if (!s.ok()) return s;
  for (auto& r : gets_) if (r.config == c) return Status::AlreadyExists("Get provider already registered");
  gets_.push_back({c, std::move(f), {}, {}, false});
  return Status::OK();
}
Status ExecutionDispatcher::Initialize(const RemoteRegionDesc& region) {
  std::unique_lock<std::shared_mutex> lock(lifecycle_);
  if (!policy_status_.ok()) return policy_status_;
  if (ready_ || region_.base) return Status::InvalidState("dispatcher requires shutdown before initialize");
  if (!region.base || !region.length || region.length > UINT64_MAX - region.base)
    return Status::InvalidArgument("invalid cache region");
  // Discover known capability gaps before any session factory runs.
  for (auto c : *policy_.enabled_put_configs) {
    if (std::none_of(puts_.begin(), puts_.end(), [c](const auto& r) { return r.config == c; }))
      return Status::NotImplemented("enabled Put provider is not built or registered");
  }
  for (auto c : *policy_.enabled_get_configs) {
    if (std::none_of(gets_.begin(), gets_.end(), [c](const auto& r) { return r.config == c; }))
      return Status::NotImplemented("enabled Get provider is not built or registered (AIV Gather/Scatter reserved)");
  }
  region_ = region;
  ready_ = true;
  return Status::OK();
}
template<class C>
Status ExecutionDispatcher::Acquire(std::vector<Resource<C>>& resources, C c, AccessOperator** out) {
  if (!ready_) return Status::NotInitialized("dispatcher is not accepting work");
  for (auto& r : resources) {
    if (!(r.config == c)) continue;
    if (!r.failure.ok()) return r.failure;
    if (!r.ready) {
      try {
        r.owner = r.factory();
        r.failure = r.owner ? r.owner->Initialize(region_) : Status::Internal("provider factory returned null");
      } catch (const std::exception& e) {
        r.failure = Status::Internal(std::string("provider initialization failed: ") + e.what());
      }
      if (!r.failure.ok()) return r.failure;
      r.ready = true;
    }
    *out = r.owner.get();
    return Status::OK();
  }
  return Status::NotImplemented("execution provider is unavailable");
}
Status ExecutionDispatcher::PreparePut(const BatchPutExecutionOptions& options) {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(put_mutex_);
  PutExecutionConfig c;
  auto s = ResolveExecution(policy_, options.config, &c);
  AccessOperator* op = nullptr;
  if (s.ok() && put_failed_) return Status::InvalidState("Put completion is unresolved; shutdown required");
  return s.ok() ? Acquire(puts_, c, &op) : s;
}
Status ExecutionDispatcher::ExecuteBatchPut(const BatchPutExecRequest& request) {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(put_mutex_);
  PutExecutionConfig c;
  auto s = ResolveExecution(policy_, request.execution.config, &c);
  if (!s.ok()) return s;
  if (put_failed_) return Status::InvalidState("Put completion is unresolved; shutdown required");
  AccessOperator* op = nullptr;
  s = Acquire(puts_, c, &op);
  if (!s.ok()) return s;
  auto invocation = request;
  invocation.execution.config = c;
  s = op->ExecuteBatchPut(invocation);
  if (!s.ok()) {
    put_failed_ = true;
    return Status::CompletionUnknown(s.message());
  }
  return s;
}
Status ExecutionDispatcher::DrainGetLocked() {
  if (!pending_get_) return Status::OK();
  auto s = pending_get_->DrainGet();
  if (s.ok()) pending_get_ = nullptr;
  return s;
}
Status ExecutionDispatcher::ExecuteBatchGet(const BatchGetExecRequest& request) {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(get_mutex_);
  GetExecutionConfig c;
  auto s = ResolveExecution(policy_, request.execution.config, &c);
  if (!s.ok()) return s;
  s = DrainGetLocked();
  if (!s.ok()) return s;
  AccessOperator* op = nullptr;
  s = Acquire(gets_, c, &op);
  if (!s.ok()) return s;
  auto invocation = request;
  invocation.execution.config = c;
  pending_get_ = op;  // Even a failed launch may have submitted work.
  s = op->ExecuteBatchGet(invocation);
  if (s.ok() && invocation.execution.synchronize) pending_get_ = nullptr;
  return s;
}
Status ExecutionDispatcher::DrainGet() {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(get_mutex_);
  return DrainGetLocked();
}
Status ExecutionDispatcher::Warmup(PutExecutionConfig c) { return PreparePut({nullptr, c}); }
Status ExecutionDispatcher::Warmup(GetExecutionConfig c) {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(get_mutex_);
  GetExecutionConfig resolved;
  auto s = ResolveExecution(policy_, c, &resolved);
  if (s.ok()) s = DrainGetLocked();
  AccessOperator* op = nullptr;
  return s.ok() ? Acquire(gets_, resolved, &op) : s;
}
Status ExecutionDispatcher::Shutdown() {
  std::unique_lock<std::shared_mutex> life(lifecycle_);
  ready_ = false;
  Status first;
  auto close = [&first](auto& resources) {
    for (auto& r : resources) {
      if (r.owner) {
        auto s = r.owner->Shutdown();
        if (!s.ok()) { if (first.ok()) first = s; continue; }
        r.owner.reset();
      }
      r.ready = false;
      r.failure = Status::OK();
    }
  };
  // Try both domains; a failed Get does not prevent Put cleanup.
  close(puts_);
  close(gets_);
  if (first.ok()) { pending_get_ = nullptr; region_ = {}; put_failed_ = false; }
  else {
    // A successfully closed pending owner must not leave a dangling pointer.
    bool retained = false;
    for (auto& r : gets_) retained |= r.owner.get() == pending_get_;
    if (!retained) pending_get_ = nullptr;
  }
  return first;
}
}  // namespace kvaf
