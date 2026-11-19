#include "cache_region_service.h"
#include <chrono>
#include <random>
#include <atomic>
#include <exception>
namespace kvaf::execution {
namespace {
Response Error(Status s) {
  Response r;
  r.status = std::move(s);
  return r;
}
}  // namespace
CacheRegionService::CacheRegionService(std::uint64_t capabilities, Allocate a, Create c)
    : capabilities_(capabilities | kRequired), allocate_(std::move(a)), create_(std::move(c)) {
  std::random_device random;
  epoch_ = (std::uint64_t(random()) << 32) ^ random() ^
           std::chrono::steady_clock::now().time_since_epoch().count();
  if (!epoch_) epoch_ = 1;
}
CacheRegionService::~CacheRegionService() {
  if (!Shutdown().ok())
    for (auto& e : records_) {
      (void)e.second->session.release();
      (void)e.second->allocation.release();
    }
}
std::shared_ptr<CacheRegionService::Record> CacheRegionService::Lookup(std::uint64_t id) {
  std::lock_guard<std::mutex> lock(index_);
  auto it = records_.find(id);
  return it == records_.end() ? nullptr : it->second;
}
Response CacheRegionService::CreateRecord(const Request& q) {
  if (!q.client || !q.request)
    return Error(Status::InvalidArgument("creation requires client and request IDs"));
  if (q.epoch != epoch_)
    return Error(Status::InvalidState("CPU service epoch changed; creation refused"));
  std::shared_ptr<Record> r;
  std::unique_lock<std::mutex> lock;
  {
    std::lock_guard<std::mutex> index(index_);
    auto it = requests_.find({q.client, q.request});
    if (it != requests_.end())
      r = it->second;
    else {
      r = std::make_shared<Record>();
      r->request = q;
      r->response.epoch = epoch_;
      r->response.state = State::Creating;
      if (!next_id_) return Error(Status::InvalidState("resource ID space exhausted"));
      const auto id = next_id_++;
      r->response.region = q.op == Op::CreateRegion ? id : q.region;
      r->response.session = q.op == Op::OpenSession ? id : 0;
      // Acquire before publishing: all other requests wait on this same record.
      lock = std::unique_lock<std::mutex>(r->mutex);
      records_.emplace(id, r);
      requests_.emplace(std::make_pair(q.client, q.request), r);
    }
  }
  if (!lock.owns_lock()) {
    lock = std::unique_lock<std::mutex>(r->mutex);
    if (Encode(r->request) != Encode(q))
      return Error(Status::InvalidArgument("create request ID reused with different parameters"));
    return r->response;
  }
  Status status;
  try {
    if (q.op == Op::CreateRegion) {
      if (q.mode > TransportMode::UrmaAndUbmem || !q.logical_bytes || !q.slot_size ||
          q.logical_bytes % q.slot_size || q.auxiliary_bytes > UINT64_MAX - q.logical_bytes ||
          (q.mode != TransportMode::Urma && !(capabilities_ & kFabric)) ||
          (q.mode != TransportMode::Ubmem && !(capabilities_ & kUrma)))
        status = Status::InvalidArgument("invalid or unsupported region allocation");
      else
        status = allocate_(q, &r->allocation);
      if (status.ok() && !r->allocation)
        status = Status::Internal("allocator returned no ownership");
      if (status.ok() &&
          (!r->allocation->descriptor.base ||
           r->allocation->descriptor.length != q.logical_bytes + q.auxiliary_bytes ||
           r->allocation->descriptor.length > UINT64_MAX - r->allocation->descriptor.base))
        status = Status::Internal("invalid allocation descriptor");
      if (status.ok()) {
        r->response.binding = r->allocation->descriptor;
        r->response.logical_bytes = q.logical_bytes;
        r->response.slot_size = q.slot_size;
      }
    } else {
      auto parent = Lookup(q.region);
      if (!parent)
        status = Status::NotFound("cache region not found");
      else {
        std::lock_guard<std::mutex> parent_lock(parent->mutex);
        if (parent->response.epoch != q.epoch || parent->request.client != q.client ||
            parent->response.state != State::Ready || !parent->allocation)
          status = Status::InvalidState("region epoch, owner or state mismatch");
        else {
          status =
              q.put ? ValidateExecution(parent->request.mode,
                                        PutExecutionConfig{q.transport, q.executor})
                    : ValidateExecution(parent->request.mode,
                                        GetExecutionConfig{q.transport, q.executor, q.algorithm});
          if (status.ok() &&
              (!q.workers || q.workers > (q.executor == Executor::Aicpu ? 8U : 48U) || !q.jettys ||
               q.jettys > 64 || !q.request_capacity || q.request_capacity > 65536 ||
               q.cpu_threads > 64 || q.auxiliary_bytes > parent->request.auxiliary_bytes))
            status = Status::InvalidArgument("invalid execution capacity");
          if (status.ok()) {
            r->parent = parent;
            ++parent->children;
          }
        }
      }
      if (status.ok()) {
        r->session = create_(q);
        status = r->session ? r->session->Open(q, *r->parent->allocation, &r->response.binding)
                            : Status::NotImplemented("session provider unavailable");
        if (status.ok() &&
            (!r->response.binding.base ||
             r->response.binding.length != r->parent->allocation->descriptor.length ||
             r->response.binding.length > UINT64_MAX - r->response.binding.base))
          status = Status::Internal("invalid session binding");
        r->response.logical_bytes = r->parent->request.logical_bytes;
        r->response.slot_size = r->parent->request.slot_size;
      }
    }
  } catch (const std::exception& e) {
    status = Status::Internal(e.what());
  }
  r->response.status = status;
  r->response.state = status.ok()
                          ? State::Ready
                          : ((r->allocation || r->session) ? State::Recovering : State::Failed);
  // Keep the failed owner attached until a confirmed close; never repeat create.
  return r->response;
}
Status CacheRegionService::CloseRecord(const std::shared_ptr<Record>& r) {
  if (r->response.state == State::Closed) return Status::OK();
  if (r->children) return Status::Busy("region still has execution sessions");
  auto status = r->session      ? r->session->Close()
                : r->allocation ? r->allocation->Close()
                                : Status::OK();
  if (!status.ok()) {
    r->response.state = State::Recovering;
    r->response.status = status;
    return status;
  }
  r->session.reset();
  r->allocation.reset();
  if (r->parent) {
    std::lock_guard<std::mutex> parent(r->parent->mutex);
    --r->parent->children;
    r->parent.reset();
  }
  r->response.state = State::Closed;
  r->response.status = Status::OK();
  return Status::OK();
}
Response CacheRegionService::Handle(const Request& q) {
  std::shared_lock<std::shared_mutex> lifetime(lifetime_);
  if (closing_) return Error(Status::InvalidState("service is closing"));
  if (q.op == Op::Capabilities) {
    Response r;
    r.status = Status::OK();
    r.state = State::Ready;
    r.capabilities = capabilities_;
    r.epoch = epoch_;
    return r;
  }
  if (q.op == Op::CreateRegion || q.op == Op::OpenSession) return CreateRecord(q);
  std::shared_ptr<Record> r;
  if (q.op == Op::Query && q.request) {
    std::lock_guard<std::mutex> index(index_);
    auto it = requests_.find({q.client, q.request});
    if (it != requests_.end()) r = it->second;
  } else
    r = Lookup(q.session ? q.session : q.region);
  if (!r) return Error(Status::NotFound("execution record not found"));
  std::lock_guard<std::mutex> lock(r->mutex);
  if (r->request.client != q.client || (q.epoch && q.epoch != r->response.epoch) ||
      (q.region && q.region != r->response.region))
    return Error(Status::InvalidArgument("resource identity mismatch"));
  if (q.op == Op::Query) return r->response;
  if (q.op == Op::CloseSession || q.op == Op::ReleaseRegion) {
    if ((q.op == Op::CloseSession) != (r->request.op == Op::OpenSession))
      return Error(Status::InvalidArgument("wrong resource direction"));
    auto s = CloseRecord(r);
    auto response = r->response;
    response.status = s;
    return response;
  }
  if (r->response.state != State::Ready)
    return Error(Status::InvalidState("resource is not ready"));
  if (q.op == Op::Fence)
    return Error(Status::NotImplemented(
        "visibility requires completed transport operations, not a control-plane CPU fence"));
  if (q.op != Op::SessionCall || !r->session)
    return Error(Status::InvalidArgument("invalid execution control command"));
  if (!q.request) return Error(Status::InvalidArgument("session calls require an idempotency ID"));
  auto cached = r->calls.find(q.request);
  if (cached != r->calls.end())
    return cached->second.first == q.payload
               ? cached->second.second
               : Error(Status::InvalidArgument("session request ID reused with different payload"));
  // Bound retained control history. No eviction: old IDs must never execute again.
  if (r->calls.size() >= 4096)
    return Error(Status::InvalidState("session control capacity exhausted; close session"));
  // Reserve the journal entry before invoking a side-effecting command. If a
  // later allocation/handler throws, the same ID remains unresolved, never rerun.
  auto initial = r->response;
  initial.status = Status::CompletionUnknown("control operation did not complete");
  auto inserted = r->calls.emplace(q.request, std::make_pair(q.payload, std::move(initial)));
  auto& response = inserted.first->second.second;
  try {
    response.status = r->session->Call(q.payload, &response.payload);
  } catch (const std::exception& e) {
    response.status = Status::Internal(e.what());
  }
  return response;
}
Status CacheRegionService::Shutdown() {
  std::unique_lock<std::shared_mutex> lifetime(lifetime_);
  closing_ = true;
  Status first;
  for (int sessions = 1; sessions >= 0; --sessions)
    for (auto& e : records_) {
      auto& r = e.second;
      if ((r->request.op == Op::OpenSession) != bool(sessions)) continue;
      std::lock_guard<std::mutex> lock(r->mutex);
      auto s = CloseRecord(r);
      if (!s.ok() && first.ok()) first = s;
    }
  return first;
}
}  // namespace kvaf::execution
