#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <vector>
#include "access_operator.h"

namespace kvaf {
namespace detail {
enum class ResourceState { Empty, Creating, Ready, Recovering, Closed };
struct ExecutionResourceRecord {
  using Factory = std::function<std::unique_ptr<AccessOperator>()>;
  explicit ExecutionResourceRecord(Factory f) : factory(std::move(f)) {}
  std::mutex mutex;
  Factory factory;
  std::unique_ptr<ExecutionResourceOwner> resources;
  std::unique_ptr<AccessOperator> operation;
  ResourceState state{ResourceState::Empty};
  Status failure;
  bool leased{false};
  Status Initialize(const RemoteRegionDesc&);
  Status Close();
  void Reset();
};
}  // namespace detail

// Direction-specific views deliberately expose only the permitted operation.
// Concrete providers validate their own typed ExecutionState at this boundary.
class PutResourceView {
 public:
  Status Prepare(const BatchPutRequest& r, const BatchPutExecutionOptions& o) const {
    return op_->PreparePut(r, o);
  }
  Status Launch(const BatchPutExecRequest& r) const { return op_->ExecuteBatchPut(r); }
  Status Resolve(PutCompletion* result) const { return op_->ResolvePut(result); }
  Status Acknowledge() const { return op_->AcknowledgePut(); }
  Status Warmup(const ExecutionWarmup& r) const { return op_->Warmup(r); }
  std::unique_ptr<ExecutionState> CreateState() const { return op_->CreateExecutionState(); }
  Status Bind(ExecutionState* s) const { return op_->BindExecutionState(s); }
 private:
  friend class PutResourceLease;
  explicit PutResourceView(AccessOperator* op) : op_(op) {}
  AccessOperator* op_;
};
class GetResourceView {
 public:
  Status Launch(const BatchGetExecRequest& r) const { return op_->ExecuteBatchGet(r); }
  Status Drain() const { return op_->DrainGet(); }
  Status Warmup(const ExecutionWarmup& r) const { return op_->Warmup(r); }
  Status PrepareGraph(const GetGraphOptions& r) const { return op_->PrepareGetGraph(r); }
  Status FinishGraph(void* stream, bool discard) const { return op_->FinishGetGraph(stream, discard); }
  std::unique_ptr<ExecutionState> CreateState() const { return op_->CreateExecutionState(); }
  Status Bind(ExecutionState* s) const { return op_->BindExecutionState(s); }
 private:
  friend class GetResourceLease;
  explicit GetResourceView(AccessOperator* op) : op_(op) {}
  AccessOperator* op_;
};

class ResourceLease {
 public:
  ResourceLease() = default;
  ResourceLease(ResourceLease&& other) noexcept : record_(std::move(other.record_)) {}
  ResourceLease& operator=(ResourceLease&& other) noexcept;
  ResourceLease(const ResourceLease&) = delete;
  ResourceLease& operator=(const ResourceLease&) = delete;
  ~ResourceLease();
  explicit operator bool() const { return bool(record_); }
 protected:
  friend class ExecutionResourceManager;
  std::shared_ptr<detail::ExecutionResourceRecord> record_;
  void Reset();
};
class PutResourceLease final : public ResourceLease {
 public:
  PutResourceView view() const { return PutResourceView(record_->operation.get()); }
};
class GetResourceLease final : public ResourceLease {
 public:
  GetResourceView view() const { return GetResourceView(record_->operation.get()); }
};
struct PutResourceRequirements { PutExecutionConfig config; std::uint32_t capacity{0}; };
struct GetResourceRequirements { GetExecutionConfig config; std::uint32_t capacity{0}; };

// A manager is scoped to one Runtime and one immutable Region generation.
// Factories capture the immutable device/EID/peer/ABI/profile of each entry;
// no entry or registration is reused across managers or Region generations.
class ExecutionResourceManager {
 public:
  using Factory = detail::ExecutionResourceRecord::Factory;
  ~ExecutionResourceManager();
  Status Register(PutExecutionConfig, Factory);
  Status Register(GetExecutionConfig, Factory);
  bool Contains(PutExecutionConfig) const;
  bool Contains(GetExecutionConfig) const;
  Status Initialize(const RemoteRegionDesc&);
  Status AcquirePut(const PutResourceRequirements&, PutResourceLease*);
  Status AcquireGet(const GetResourceRequirements&, GetResourceLease*);
  Status Recover(PutExecutionConfig);
  Status Recover(GetExecutionConfig);
  // Only the exclusive lease holder may close a still-leased resource. This
  // drains/tears down through its owner while invocation state is still alive.
  Status Close(PutResourceLease&);
  Status Close(GetResourceLease&);
  Status Shutdown();
 private:
  template<class C> struct Entry { C config; std::shared_ptr<detail::ExecutionResourceRecord> record; };
  template<class C> Status Register(std::vector<Entry<C>>&, C, Factory);
  template<class C> std::shared_ptr<detail::ExecutionResourceRecord> Find(const std::vector<Entry<C>>&, C) const;
  template<class C> Status Acquire(const std::vector<Entry<C>>&, C, std::uint32_t, ResourceLease*);
  template<class C> Status Recover(const std::vector<Entry<C>>&, C);
  Status Close(ResourceLease&);
  mutable std::shared_mutex lifecycle_;
  RemoteRegionDesc region_{};
  bool ready_{false};
  std::vector<Entry<PutExecutionConfig>> puts_;
  std::vector<Entry<GetExecutionConfig>> gets_;
};
}  // namespace kvaf
