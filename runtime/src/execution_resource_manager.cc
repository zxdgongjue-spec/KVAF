#include "execution_resource_manager.h"

#include <exception>

namespace kvaf {
namespace detail {
Status ExecutionResourceRecord::Create() {
  if (!failure.ok()) return failure;
  if (operation) return Status::OK();
  try {
    operation = factory();
    if (!operation) failure = Status::Internal("provider factory returned null");
    else resources = operation->TakeResources();
  } catch (const std::exception& e) {
    failure = Status::Internal(std::string("provider creation failed: ") + e.what());
  } catch (...) {
    failure = Status::Internal("provider creation failed");
  }
  if (!failure.ok()) state = ResourceState::Recovering;
  return failure;
}
Status ExecutionResourceRecord::Initialize(const RemoteRegionDesc& region) {
  auto s = Create();
  if (!s.ok() || state == ResourceState::Ready) return s;
  state = ResourceState::Creating;
  try {
    failure = resources ? resources->Initialize(region) : operation->Initialize(region);
  } catch (const std::exception& e) {
    failure = Status::Internal(std::string("resource initialization failed: ") + e.what());
  } catch (...) {
    failure = Status::Internal("resource initialization failed");
  }
  state = failure.ok() ? ResourceState::Ready : ResourceState::Recovering;
  accepting_work = failure.ok();
  return failure;
}
Status ExecutionResourceRecord::Close() {
  accepting_work = false;
  if (state == ResourceState::Closed) return Status::OK();
  auto s = resources ? resources->Close() : operation ? operation->Shutdown() : Status::OK();
  if (!s.ok()) { state = ResourceState::Recovering; failure = s; return s; }
  state = ResourceState::Closed;
  failure = Status::OK();
  if (!leased && !resident_refs) Reset();
  return s;
}
void ExecutionResourceRecord::Reset() {
  // Borrowed operators must disappear before their concrete resource views.
  operation.reset();
  resources.reset();
}
}  // namespace detail

ResourceLease::~ResourceLease() { Reset(); }
ResourceLease& ResourceLease::operator=(ResourceLease&& other) noexcept {
  if (this != &other) { Reset(); record_ = std::move(other.record_); }
  return *this;
}
void ResourceLease::Reset() {
  if (!record_) return;
  auto record = std::move(record_);
  {
    std::lock_guard<std::mutex> lock(record->mutex);
    record->leased = false;
    if (record->state == detail::ResourceState::Closed && !record->resident_refs) record->Reset();
  }
}
GetResourceReference::~GetResourceReference() { Reset(); }
GetResourceReference& GetResourceReference::operator=(GetResourceReference&& other) noexcept {
  if (this != &other) { Reset(); record_ = std::move(other.record_); }
  return *this;
}
void GetResourceReference::Reset() {
  if (!record_) return;
  auto record = std::move(record_);
  std::lock_guard<std::mutex> lock(record->mutex);
  --record->resident_refs;
  if (!record->resident_refs && !record->leased && record->state == detail::ResourceState::Closed)
    record->Reset();
}
Status GetResourceReference::Acquire(GetResourceLease* out) const {
  if (!record_ || !out || *out) return Status::InvalidArgument("Get execution lease must be empty");
  std::lock_guard<std::mutex> lock(record_->mutex);
  if (!record_->accepting_work) return Status::NotInitialized("Get resources are closing");
  if (!record_->failure.ok()) return record_->failure;
  if (record_->state != detail::ResourceState::Ready)
    return Status::InvalidState("resident Get resources are not ready");
  if (record_->leased) return Status::Busy("Get resources are already executing");
  record_->leased = true;
  out->record_ = record_;
  return Status::OK();
}
ExecutionResourceManager::~ExecutionResourceManager() {
  if (!Shutdown().ok()) {
    auto retain = [](auto& entries) {
      for (auto& entry : entries) {
        std::lock_guard<std::mutex> lock(entry.record->mutex);
        if (entry.record->state != detail::ResourceState::Closed)
          (void)new decltype(entry.record)(entry.record);
      }
    };
    retain(puts_);
    retain(gets_);
  }
}
template<class C>
std::shared_ptr<detail::ExecutionResourceRecord> ExecutionResourceManager::Find(
    const std::vector<Entry<C>>& entries, C c) const {
  for (const auto& entry : entries) if (entry.config == c) return entry.record;
  return {};
}
template<class C>
Status ExecutionResourceManager::Register(std::vector<Entry<C>>& entries, C c, Factory f) {
  std::unique_lock<std::shared_mutex> lock(lifecycle_);
  if (ready_ || region_.base) return Status::InvalidState("register resources before initialization");
  if (!f) return Status::InvalidArgument("null provider factory");
  if (Find(entries, c)) return Status::AlreadyExists("execution resource already registered");
  entries.push_back({c, std::make_shared<detail::ExecutionResourceRecord>(std::move(f))});
  return Status::OK();
}
Status ExecutionResourceManager::Register(PutExecutionConfig c, Factory f) { return Register(puts_, c, std::move(f)); }
Status ExecutionResourceManager::Register(GetExecutionConfig c, Factory f) { return Register(gets_, c, std::move(f)); }
bool ExecutionResourceManager::Contains(PutExecutionConfig c) const {
  std::shared_lock<std::shared_mutex> lock(lifecycle_); return bool(Find(puts_, c));
}
bool ExecutionResourceManager::Contains(GetExecutionConfig c) const {
  std::shared_lock<std::shared_mutex> lock(lifecycle_); return bool(Find(gets_, c));
}
Status ExecutionResourceManager::Initialize(const RemoteRegionDesc& region) {
  std::unique_lock<std::shared_mutex> lock(lifecycle_);
  if (ready_ || region_.base) return Status::InvalidState("resource manager requires shutdown");
  if (!region.base || !region.length || region.length > UINT64_MAX - region.base)
    return Status::InvalidArgument("invalid cache region");
  region_ = region;
  ready_ = true;
  return Status::OK();
}
template<class C>
Status ExecutionResourceManager::Acquire(const std::vector<Entry<C>>& entries, C c,
                                         std::uint32_t capacity, ResourceLease* out) {
  if (!out || *out) return Status::InvalidArgument("resource lease must be empty");
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  if (!ready_) return Status::NotInitialized("resource manager is not accepting work");
  auto record = Find(entries, c);
  if (!record) return Status::NotImplemented("execution provider is unavailable");
  std::unique_lock<std::mutex> lock(record->mutex);
  if (record->leased || record->resident_refs) return Status::Busy("execution owns the resource lease or resident reference");
  auto s = record->Create();
  if (!s.ok()) return s;
  s = record->operation->ValidateRequestCapacity(capacity);
  if (!s.ok()) return s;
  s = record->Initialize(region_);
  if (!s.ok()) return s;
  record->leased = true;
  out->record_ = std::move(record);
  return Status::OK();
}
template<class C, class R, class Q>
Status ExecutionResourceManager::Describe(const std::vector<Entry<C>>& entries, C c,
                                          const R& request, Q* out) {
  if (!out) return Status::InvalidArgument("null resource requirements output");
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  if (!ready_) return Status::NotInitialized("resource manager is not accepting work");
  auto record = Find(entries, c);
  if (!record) return Status::NotImplemented("execution provider is unavailable");
  std::lock_guard<std::mutex> lock(record->mutex);
  if (record->leased || record->resident_refs) return Status::Busy("execution owns the resource lease or resident reference");
  auto s = record->Create();
  std::uint32_t capacity{};
  if (s.ok()) s = record->operation->DescribeRequirements(request, &capacity);
  if (s.ok()) *out = {c, capacity};
  return s;
}
Status ExecutionResourceManager::Describe(PutExecutionConfig c, const BatchPutRequest& r,
                                          PutResourceRequirements* out) {
  return Describe(puts_, c, r, out);
}
Status ExecutionResourceManager::Describe(GetExecutionConfig c, const BatchGetRequest& r,
                                          GetResourceRequirements* out) {
  return Describe(gets_, c, r, out);
}
Status ExecutionResourceManager::AcquirePut(const PutResourceRequirements& r, PutResourceLease* out) {
  return Acquire(puts_, r.config, r.capacity, out);
}
Status ExecutionResourceManager::AcquireGet(const GetResourceRequirements& r, GetResourceLease* out) {
  return Acquire(gets_, r.config, r.capacity, out);
}
Status ExecutionResourceManager::PinGet(const GetResourceRequirements& r, GetResourceReference* out) {
  if (!out || *out) return Status::InvalidArgument("Get resource reference must be empty");
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  if (!ready_) return Status::NotInitialized("resource manager is not accepting work");
  auto record = Find(gets_, r.config);
  if (!record) return Status::NotImplemented("execution provider is unavailable");
  std::lock_guard<std::mutex> lock(record->mutex);
  if (record->leased || record->resident_refs) return Status::Busy("Get resources already have an owner");
  auto s = record->Create();
  if (s.ok()) s = record->operation->ValidateRequestCapacity(r.capacity);
  if (s.ok()) s = record->Initialize(region_);
  if (!s.ok()) return s;
  ++record->resident_refs;
  out->record_ = std::move(record);
  return Status::OK();
}
template<class C>
Status ExecutionResourceManager::Recover(const std::vector<Entry<C>>& entries, C c) {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  if (!ready_) return Status::NotInitialized("resource manager is not accepting work");
  auto record = Find(entries, c);
  if (!record) return Status::NotImplemented("execution provider is unavailable");
  std::lock_guard<std::mutex> lock(record->mutex);
  if (record->leased || record->resident_refs) return Status::Busy("execution owns the resource lease or resident reference");
  auto s = record->Close();
  if (!s.ok()) return s;
  record->state = detail::ResourceState::Empty;
  return record->Initialize(region_);
}
Status ExecutionResourceManager::Recover(PutExecutionConfig c) { return Recover(puts_, c); }
Status ExecutionResourceManager::Recover(GetExecutionConfig c) { return Recover(gets_, c); }
Status ExecutionResourceManager::Close(ResourceLease& lease) {
  if (!lease) return Status::OK();
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(lease.record_->mutex);
  return lease.record_->Close();
}
Status ExecutionResourceManager::Close(PutResourceLease& lease) { return Close(static_cast<ResourceLease&>(lease)); }
Status ExecutionResourceManager::Close(GetResourceLease& lease) { return Close(static_cast<ResourceLease&>(lease)); }
Status ExecutionResourceManager::Close(GetResourceReference& reference) {
  if (!reference) return Status::OK();
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(reference.record_->mutex);
  return reference.record_->Close();
}
Status ExecutionResourceManager::Shutdown() {
  std::unique_lock<std::shared_mutex> life(lifecycle_);
  ready_ = false;
  Status first;
  auto close = [&](auto& entries) {
    for (auto& entry : entries) {
      std::lock_guard<std::mutex> lock(entry.record->mutex);
      entry.record->accepting_work = false;
      auto s = (entry.record->leased || entry.record->resident_refs)
          ? Status::Busy("execution still owns resource lease or resident state") : entry.record->Close();
      if (!s.ok() && first.ok()) first = s;
    }
  };
  close(puts_);
  close(gets_);
  if (first.ok()) region_ = {};
  return first;
}
}  // namespace kvaf
