#pragma once

#include <chrono>
#include <cstdint>
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
  std::size_t resident_refs{0};
  bool accepting_work{false};
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
  Status Describe(const BatchPutRequest& r, std::uint32_t* capacity) const {
    return op_->DescribeRequirements(r, capacity);
  }
  Status ValidateCapacity(std::uint32_t capacity) const { return op_->ValidateRequestCapacity(capacity); }
  Status Prepare(const BatchPutRequest& r, const BatchPutExecutionOptions& o) const {
    std::uint32_t capacity{};
    auto s = op_->DescribeRequirements(r, &capacity);
    if (s.ok()) s = Warmup({capacity, o.stream});
    return s.ok() ? op_->PreparePut(r, o) : s;
  }
  Status PrepareBound(const BatchPutRequest& r, const BatchPutExecutionOptions& o) const {
    return op_->PreparePut(r, o);
  }
  Status Launch(const BatchPutExecRequest& r) const { return op_->ExecuteBatchPut(r); }
  Status Resolve(PutCompletion* result) const { return op_->ResolvePut(result); }
  Status Acknowledge() const { return op_->AcknowledgePut(); }
  Status PrepareInput(const PutInputOptions& r) const { return op_->PreparePutInput(r); }
  Status FinishInput() const { return op_->FinishPutInput(); }
  Status Warmup(const ExecutionWarmup& r) const {
    return resources_ ? resources_->Warmup(r) : op_->Warmup(r);
  }
  std::unique_ptr<ExecutionState> CreateState() const { return op_->CreateExecutionState(); }
  Status Bind(ExecutionState* s) const { return op_->BindExecutionState(s); }
 private:
  friend class PutResourceLease;
  friend class PutResourceReference;
  explicit PutResourceView(AccessOperator* op, ExecutionResourceOwner* resources)
      : op_(op), resources_(resources) {}
  AccessOperator* op_;
  ExecutionResourceOwner* resources_;
};
struct GetLaunchTiming {
  std::uint64_t requirements_ns{}, warmup_ns{}, provider_ns{};
  GetProviderTiming provider_detail;
};
class GetResourceView {
 public:
  Status Describe(const BatchGetRequest& r, std::uint32_t* capacity) const {
    return op_->DescribeRequirements(r, capacity);
  }
  Status ValidateCapacity(std::uint32_t capacity) const { return op_->ValidateRequestCapacity(capacity); }
  // Preparation is owned by the resident dispatcher instance. Never warm or
  // allocate here: a graph invocation must use its already pinned resources.
  Status LaunchPrepared(const BatchGetExecRequest& r, GetLaunchTiming* timing = nullptr) const {
    if (!r.request) return Status::InvalidArgument("missing Get request");
    if (!timing) return op_->ExecuteBatchGet(r);
    const auto start = std::chrono::steady_clock::now();
    auto profiled = r;
    profiled.host_timing = &timing->provider_detail;
    auto s = op_->ExecuteBatchGet(profiled);
    timing->provider_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count();
    return s;
  }
  Status Launch(const BatchGetExecRequest& r, GetLaunchTiming* timing = nullptr) const {
    if (!r.request) return Status::InvalidArgument("missing Get request");
    if (!timing) {
      std::uint32_t capacity{};
      auto s = op_->DescribeRequirements(*r.request, &capacity);
      if (s.ok() && !r.execution.graph_lease) s = Warmup({capacity, r.execution.stream});
      return s.ok() ? op_->ExecuteBatchGet(r) : s;
    }
    const auto now = [] {
      return std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    std::uint32_t capacity{};
    auto start = now();
    auto s = op_->DescribeRequirements(*r.request, &capacity);
    timing->requirements_ns = now() - start;
    if (s.ok() && !r.execution.graph_lease) {
      start = now();
      s = Warmup({capacity, r.execution.stream});
      timing->warmup_ns = now() - start;
    }
    if (s.ok()) {
      start = now();
      auto profiled = r;
      profiled.host_timing = &timing->provider_detail;
      s = op_->ExecuteBatchGet(profiled);
      timing->provider_ns = now() - start;
    }
    return s;
  }
  Status Drain(BatchGetHostTiming* timing = nullptr) const {
    return timing ? op_->DrainGet(timing) : op_->DrainGet();
  }
  Status PrepareOutput(const GetOutputOptions& r) const {
    return op_->PrepareGetOutput(r);
  }
  Status FinishOutput() const { return op_->FinishGetOutput(); }
  Status Warmup(const ExecutionWarmup& r) const {
    return resources_ ? resources_->Warmup(r) : op_->Warmup(r);
  }
  Status PrepareGraph(const GetGraphOptions& r) const {
    if (!r.request_capacity) return Status::InvalidArgument("graph capacity must be positive");
    auto s = Warmup({r.request_capacity, r.stream});
    return s.ok() ? op_->PrepareGetGraph(r) : s;
  }
  Status PrepareGraphPrepared(const GetGraphOptions& r) const { return op_->PrepareGetGraph(r); }
  Status FinishGraph(void* stream, bool discard) const { return op_->FinishGetGraph(stream, discard); }
  std::unique_ptr<ExecutionState> CreateState() const { return op_->CreateExecutionState(); }
  Status Bind(ExecutionState* s) const { return op_->BindExecutionState(s); }
  AccessOperator* operation() const { return op_; }
 private:
  friend class GetResourceLease;
  friend class GetResourceReference;
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
  friend class PutResourceReference;
  friend class GetResourceReference;
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
// A resident state pins its provider/owner without claiming execution. Only its
// holder can acquire the short exclusive lease; completion returns that lease
// while keeping this reference and the bound provider state alive.
class PutResourceReference final {
 public:
  PutResourceReference() = default;
  PutResourceReference(PutResourceReference&& other) noexcept : record_(std::move(other.record_)) {}
  PutResourceReference& operator=(PutResourceReference&&) noexcept;
  PutResourceReference(const PutResourceReference&) = delete;
  PutResourceReference& operator=(const PutResourceReference&) = delete;
  ~PutResourceReference();
  explicit operator bool() const { return bool(record_); }
  PutResourceView view() const { return PutResourceView(record_->operation.get(), record_->resources.get()); }
  Status Acquire(PutResourceLease*) const;
 private:
  friend class ExecutionResourceManager;
  std::shared_ptr<detail::ExecutionResourceRecord> record_;
  void Reset();
};
class GetResourceReference final {
 public:
  GetResourceReference() = default;
  GetResourceReference(GetResourceReference&& other) noexcept : record_(std::move(other.record_)) {}
  GetResourceReference& operator=(GetResourceReference&&) noexcept;
  GetResourceReference(const GetResourceReference&) = delete;
  GetResourceReference& operator=(const GetResourceReference&) = delete;
  ~GetResourceReference();
  explicit operator bool() const { return bool(record_); }
  GetResourceView view() const { return GetResourceView(record_->operation.get(), record_->resources.get()); }
  Status Acquire(GetResourceLease*) const;
 private:
  friend class ExecutionResourceManager;
  std::shared_ptr<detail::ExecutionResourceRecord> record_;
  void Reset();
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
  Status PinPut(const PutResourceRequirements&, PutResourceReference*);
  Status PinGet(const GetResourceRequirements&, GetResourceReference*);
  Status Recover(PutExecutionConfig);
  Status Recover(GetExecutionConfig);
  // Only the exclusive lease holder may close a still-leased resource. This
  // drains/tears down through its owner while invocation state is still alive.
  Status Close(PutResourceLease&);
  Status Close(GetResourceLease&);
  Status Close(PutResourceReference&);
  Status Close(GetResourceReference&);
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
