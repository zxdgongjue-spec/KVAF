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
  Status Create();
  Status Initialize(const RemoteRegionDesc&);
  Status Close();
  void Reset();
};
}  // namespace detail

// Direction-specific dispatch adapters for the public AccessOperator contract.
// Hardware providers prepare concrete resource views before calling stateless
// algorithms; these adapters route lifecycle work directly to the resource owner.
class PutResourceView {
 public:
  Status Prepare(const BatchPutRequest& r, const BatchPutExecutionOptions& o) const {
    std::uint32_t capacity{};
    auto s = op_->DescribeRequirements(r, &capacity);
    if (s.ok()) s = Warmup({capacity, o.stream});
    return s.ok() ? op_->PreparePut(r, o) : s;
  }
  Status Launch(const BatchPutExecRequest& r) const { return op_->ExecuteBatchPut(r); }
  Status Resolve(PutCompletion* result) const { return op_->ResolvePut(result); }
  Status Acknowledge() const { return op_->AcknowledgePut(); }
  Status Warmup(const ExecutionWarmup& r) const {
    return resources_ ? resources_->Warmup(r) : op_->Warmup(r);
  }
  std::unique_ptr<ExecutionState> CreateState() const { return op_->CreateExecutionState(); }
  Status Bind(ExecutionState* s) const { return op_->BindExecutionState(s); }
 private:
  friend class PutResourceLease;
  explicit PutResourceView(AccessOperator* op, ExecutionResourceOwner* resources)
      : op_(op), resources_(resources) {}
  AccessOperator* op_;
  ExecutionResourceOwner* resources_;
};
class GetResourceView {
 public:
  Status Launch(const BatchGetExecRequest& r) const {
    if (!r.request) return Status::InvalidArgument("missing Get request");
    std::uint32_t capacity{};
    auto s = op_->DescribeRequirements(*r.request, &capacity);
    if (s.ok() && !r.execution.graph_lease) s = Warmup({capacity, r.execution.stream});
    return s.ok() ? op_->ExecuteBatchGet(r) : s;
  }
  Status Drain() const { return op_->DrainGet(); }
  Status Warmup(const ExecutionWarmup& r) const {
    return resources_ ? resources_->Warmup(r) : op_->Warmup(r);
  }
  Status PrepareGraph(const GetGraphOptions& r) const {
    if (!r.request_capacity) return Status::InvalidArgument("graph capacity must be positive");
    auto s = Warmup({r.request_capacity, r.stream});
    return s.ok() ? op_->PrepareGetGraph(r) : s;
  }
  Status FinishGraph(void* stream, bool discard) const { return op_->FinishGetGraph(stream, discard); }
  std::unique_ptr<ExecutionState> CreateState() const { return op_->CreateExecutionState(); }
  Status Bind(ExecutionState* s) const { return op_->BindExecutionState(s); }
 private:
  friend class GetResourceLease;
  explicit GetResourceView(AccessOperator* op, ExecutionResourceOwner* resources)
      : op_(op), resources_(resources) {}
  AccessOperator* op_;
  ExecutionResourceOwner* resources_;
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
  PutResourceView view() const { return PutResourceView(record_->operation.get(), record_->resources.get()); }
};
class GetResourceLease final : public ResourceLease {
 public:
  GetResourceView view() const { return GetResourceView(record_->operation.get(), record_->resources.get()); }
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
  Status Describe(PutExecutionConfig, const BatchPutRequest&, PutResourceRequirements*);
  Status Describe(GetExecutionConfig, const BatchGetRequest&, GetResourceRequirements*);
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
  template<class C, class R, class Q> Status Describe(const std::vector<Entry<C>>&, C, const R&, Q*);
  template<class C> Status Recover(const std::vector<Entry<C>>&, C);
  Status Close(ResourceLease&);
  mutable std::shared_mutex lifecycle_;
  RemoteRegionDesc region_{};
  bool ready_{false};
  std::vector<Entry<PutExecutionConfig>> puts_;
  std::vector<Entry<GetExecutionConfig>> gets_;
};
}  // namespace kvaf
